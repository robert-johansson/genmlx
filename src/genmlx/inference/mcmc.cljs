(ns genmlx.inference.mcmc
  "MCMC inference algorithms: MH, MALA, HMC, NUTS, Custom Proposal MH,
   Enumerative Gibbs, Involutive MCMC.
   MH uses the GFI regenerate operation.
   Gradient-based methods operate on compiled model score functions."
  (:require [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist.core :as dc]
            [genmlx.inference.util :as u]
            [genmlx.learning :as learn]))

;; ---------------------------------------------------------------------------
;; Device management — CPU for scalar inference, GPU for vectorized
;; ---------------------------------------------------------------------------

(defn- resolve-device
  "Resolve a :device option to an MLX device.
   :cpu → mx/cpu, :gpu → mx/gpu, nil → default for caller."
  [device]
  (case device
    :cpu mx/cpu
    :gpu mx/gpu
    nil))

(defn- with-device
  "Run f with the given MLX device as default, restoring the original after.
   If device is nil, runs f with no device change."
  [device f]
  (if-let [d (resolve-device device)]
    (let [prev (mx/default-device)]
      (mx/set-default-device! d)
      (try (f)
        (finally (mx/set-default-device! prev))))
    (f)))

;; ---------------------------------------------------------------------------
;; Generic sample collector (shared burn-in / thin / callback loop)
;; ---------------------------------------------------------------------------

(defn- collect-samples
  "Generic MCMC sample collection loop with burn-in, thinning, and callback.
   - `step-fn`:    (fn [state key] -> {:state new-state :accepted? bool})
   - `extract-fn`: (fn [state] -> sample)
   - `init-state`: initial MCMC state
   - Returns vector of samples with {:acceptance-rate ...} metadata."
  [{:keys [samples burn thin callback key]} step-fn extract-fn init-state]
  (let [total-iters (+ burn (* samples thin))]
    (loop [i 0, state init-state, acc (transient []), n 0, n-accepted 0, rk key]
      (if (>= n samples)
        (with-meta (persistent! acc)
          {:acceptance-rate (/ n-accepted total-iters)})
        (let [[step-key next-key] (rng/split-or-nils rk)
              {:keys [state accepted?]} (step-fn state step-key)
              past-burn? (>= i burn)
              keep? (and past-burn? (zero? (mod (- i burn) thin)))]
          (when (and callback keep?)
            (callback {:iter n :value (extract-fn state) :accepted? accepted?}))
          (recur (inc i) state
                 (if keep? (conj! acc (extract-fn state)) acc)
                 (if keep? (inc n) n)
                 (if accepted? (inc n-accepted) n-accepted)
                 next-key))))))

;; ---------------------------------------------------------------------------
;; Metropolis-Hastings (via regenerate)
;; ---------------------------------------------------------------------------

(defn mh-step
  "One MH step. regenerate proposes + computes acceptance ratio.
   Optional `key` for functional PRNG."
  ([current-trace selection]
   (mh-step current-trace selection nil))
  ([current-trace selection key]
   (let [gf (:gen-fn current-trace)
         result (p/regenerate gf current-trace selection)
         w (mx/realize (:weight result))]
     (if (u/accept-mh? w key)
       (:trace result)
       current-trace))))

(defn mh
  "Metropolis-Hastings inference. Returns vector of traces.

   opts: {:samples N :burn B :thin T :selection sel :callback fn :key prng-key}
   model: generative function
   args: model arguments
   observations: choice map of observed values"
  [{:keys [samples burn thin selection callback key]
    :or {burn 0 thin 1 selection sel/all}}
   model args observations]
  (let [{:keys [trace]} (p/generate model args observations)]
    (collect-samples
      {:samples samples :burn burn :thin thin :callback callback :key key}
      (fn [state step-key]
        (let [new-trace (mh-step state selection step-key)]
          {:state new-trace :accepted? (not (identical? new-trace state))}))
      identity
      trace)))

;; ---------------------------------------------------------------------------
;; Custom Proposal MH
;; ---------------------------------------------------------------------------

(defn mh-custom-step
  "One MH step with a custom proposal generative function.
   proposal-gf: generative function that takes [current-trace-choices]
                and proposes new choices for some addresses.
   backward-gf: (optional) generative function for the backward proposal.
                 If nil, assumes symmetric proposal (backward = forward).
   model: the target model generative function.
   current-trace: the current model trace.
   key: PRNG key."
  ([current-trace model proposal-gf key]
   (mh-custom-step current-trace model proposal-gf nil key))
  ([current-trace model proposal-gf backward-gf key]
   (let [[k1 k2 k3] (rng/split-n (rng/ensure-key key) 3)
         ;; 1. Run propose on the proposal GF → forward choices + forward score
         proposal-args [(:choices current-trace)]
         forward-result (p/propose proposal-gf proposal-args)
         forward-choices (:choices forward-result)
         forward-score (:weight forward-result)
         ;; 2. Apply proposed choices to model via update → new trace
         update-result (p/update model current-trace forward-choices)
         new-trace (:trace update-result)
         ;; 3. Compute backward score
         backward-score (if backward-gf
                          ;; Run assess on backward proposal
                          (let [{:keys [weight]} (p/assess backward-gf
                                                           [(:choices new-trace)]
                                                           (:choices current-trace))]
                            weight)
                          ;; Symmetric proposal: backward score = forward score
                          forward-score)
         ;; 4. Accept/reject using update weight
         ;; log-alpha = update-weight + backward_score - forward_score
         update-weight (:weight update-result)
         _ (mx/eval! update-weight forward-score backward-score)
         log-alpha (mx/realize (mx/add update-weight
                                 (mx/subtract backward-score forward-score)))]
     (if (u/accept-mh? log-alpha k3)
       new-trace
       current-trace))))

(defn mh-custom
  "MH inference with custom proposal. Returns vector of traces.

   opts: {:samples N :burn B :thin T :proposal-gf gf :backward-gf gf
          :callback fn :key prng-key}
   model: generative function
   args: model arguments
   observations: choice map of observed values"
  [{:keys [samples burn thin proposal-gf backward-gf callback key]
    :or {burn 0 thin 1}}
   model args observations]
  (let [{:keys [trace]} (p/generate model args observations)]
    (collect-samples
      {:samples samples :burn burn :thin thin :callback callback :key key}
      (fn [state step-key]
        (let [new-trace (mh-custom-step state model proposal-gf backward-gf step-key)]
          {:state new-trace :accepted? (not (identical? new-trace state))}))
      identity
      trace)))

;; ---------------------------------------------------------------------------
;; Compiled MH (random-walk proposal in parameter space)
;; ---------------------------------------------------------------------------

(defn- compiled-mh-step
  "One compiled MH step with random-walk Gaussian proposal.
   Operates on flat parameter arrays — no GFI overhead per step."
  [params score-fn proposal-std param-shape key]
  (let [[propose-key accept-key] (rng/split-or-nils key)
        noise    (if propose-key
                   (rng/normal propose-key param-shape)
                   (mx/random-normal param-shape))
        proposal (mx/add params (mx/multiply proposal-std noise))
        score-current  (score-fn params)
        score-proposal (score-fn proposal)
        _ (mx/eval! score-current score-proposal)
        log-alpha (- (mx/item score-proposal) (mx/item score-current))
        accept? (u/accept-mh? log-alpha accept-key)]
    {:state (if accept? proposal params) :accepted? accept?}))

(defn- make-compiled-chain
  "Build a compiled K-step MH chain as one Metal dispatch.
   Returns compiled fn: (params [D], noise [K,D], uniforms [K]) → params [D].
   Noise and uniforms are generated OUTSIDE — compile-fn freezes random ops."
  [k-steps score-fn proposal-std n-params]
  (let [chain-fn
        (fn [params noise-2d uniforms-1d]
          (loop [p params, i 0]
            (if (>= i k-steps) p
              (let [row (mx/reshape
                          (mx/take-idx noise-2d (mx/array [i] mx/int32) 0)
                          [n-params])
                    proposal (mx/add p (mx/multiply proposal-std row))
                    s-cur (score-fn p)
                    s-prop (score-fn proposal)
                    log-alpha (mx/subtract s-prop s-cur)
                    log-u (mx/log (mx/index uniforms-1d i))
                    accept? (mx/greater log-alpha log-u)]
                (recur (mx/where accept? proposal p) (inc i))))))
        compiled (mx/compile-fn chain-fn)]
    ;; Trace call to cache the Metal program
    (mx/eval! (compiled (mx/array (vec (repeat n-params 0.0)))
                        (mx/random-normal [k-steps n-params])
                        (mx/random-uniform [k-steps])))
    compiled))

(defn- run-loop-compiled-mh
  "Run compiled MH with loop compilation for burn-in and optional thinning.
   Uses compiled chains for burn-in (block-size steps per dispatch).
   For collection: compiled chain if thin > 1, eager step if thin = 1."
  [{:keys [samples burn thin callback]} init-params n-params
   score-fn proposal-std burn-chain burn-block-size thin-chain]
  (let [param-shape [n-params]
        ;; Phase 1: Burn-in via compiled chain blocks
        params (if (and burn-chain (> burn 0))
                 (let [n-blocks (js/Math.ceil (/ burn burn-block-size))]
                   (loop [p init-params, b 0]
                     (if (>= b n-blocks) p
                       (let [noise (mx/random-normal [burn-block-size n-params])
                             uniforms (mx/random-uniform [burn-block-size])
                             p' (burn-chain p noise uniforms)]
                         (mx/eval! p')
                         (recur p' (inc b))))))
                 init-params)
        ;; Phase 2: Collect samples (mx/tidy prevents Metal resource leak)
        result (loop [p params, acc (transient []), i 0]
                 (if (>= i samples)
                   (persistent! acc)
                   (let [p' (mx/tidy
                              (fn []
                                (if thin-chain
                                  ;; thin > 1: compiled chain of thin steps
                                  (let [noise (mx/random-normal [thin n-params])
                                        uniforms (mx/random-uniform [thin])
                                        r (thin-chain p noise uniforms)]
                                    (mx/eval! r) r)
                                  ;; thin = 1: eager step (faster than 1-step compiled)
                                  (let [{:keys [state]} (compiled-mh-step
                                                          p score-fn proposal-std
                                                          param-shape nil)]
                                    state))))]
                     (when callback
                       (callback {:iter i :value (mx/->clj p')}))
                     (recur p' (conj! acc (mx/->clj p')) (inc i)))))]
    result))

(defn compiled-mh
  "Compiled MH inference with random-walk Gaussian proposal.
   Extracts latent parameters into a flat array, compiles the score function,
   and iterates in parameter space — bypassing GFI regenerate overhead.

   When compile? is true (default), uses loop compilation: entire K-step chains
   are compiled into single Metal dispatches for ~5x speedup. Burn-in runs in
   blocks of :block-size steps. Collection runs thin steps per sample.

   opts: {:samples N :burn B :thin T :addresses [addr...]
          :proposal-std σ :compile? bool :callback fn :key prng-key
          :device :cpu|:gpu :block-size K}
   model: generative function
   args: model arguments
   observations: choice map of observed values

   Returns vector of parameter samples (JS arrays via mx/->clj).
   Default device: :cpu (faster for scalar parameters)."
  [{:keys [samples burn thin addresses proposal-std compile? callback key device
           block-size]
    :or {burn 0 thin 1 proposal-std 0.1 compile? true device :cpu
         block-size 50}}
   model args observations]
  (with-device device
    #(let [score-fn  (u/make-score-fn model args observations addresses)
           score-fn  (if compile? (mx/compile-fn score-fn) score-fn)
           {:keys [trace]} (p/generate model args observations)
           init-params (u/extract-params trace addresses)
           n-params (count addresses)
           std (mx/scalar proposal-std)]
       (if compile?
         ;; Loop-compiled path: compiled chains for burn-in + optional thin
         (let [burn-block (min (max burn 1) block-size)
               burn-chain (when (> burn 0)
                            (make-compiled-chain burn-block score-fn std n-params))
               thin-chain (when (> thin 1)
                            (make-compiled-chain thin score-fn std n-params))]
           (run-loop-compiled-mh
             {:samples samples :burn burn :thin thin :callback callback}
             init-params n-params score-fn std
             burn-chain burn-block thin-chain))
         ;; Fallback: per-step eager path
         (collect-samples
           {:samples samples :burn burn :thin thin :callback callback :key key}
           (fn [params step-key]
             (compiled-mh-step params score-fn std (mx/shape init-params) step-key))
           mx/->clj
           init-params)))))

;; ---------------------------------------------------------------------------
;; Vectorized Compiled MH (N parallel chains)
;; ---------------------------------------------------------------------------

(defn- vectorized-mh-step
  "One vectorized MH step for N parallel chains.
   params: [N, D], score-fn returns [N], accept/reject via mx/where."
  [params score-fn proposal-std param-shape n-chains key]
  (let [[propose-key accept-key] (rng/split-or-nils key)
        noise    (if propose-key
                   (rng/normal propose-key param-shape)
                   (mx/random-normal param-shape))
        proposal (mx/add params (mx/multiply proposal-std noise))
        score-current  (score-fn params)
        score-proposal (score-fn proposal)
        _ (mx/eval! score-current score-proposal)
        log-alphas (mx/subtract score-proposal score-current)
        u (if accept-key
            (rng/uniform accept-key [n-chains])
            (mx/random-uniform [n-chains]))
        accept-mask (mx/less (mx/log u) log-alphas)
        _ (mx/eval! accept-mask)
        n-accepted (mx/item (mx/sum accept-mask))
        new-params (mx/where (mx/expand-dims accept-mask 1) proposal params)
        _ (mx/eval! new-params)]
    {:state new-params :n-accepted (int n-accepted)}))

(defn vectorized-compiled-mh
  "Vectorized compiled MH: N independent chains running in parallel.
   MLX broadcasting executes the model ONCE per step for all N chains.

   opts: {:samples N :burn B :thin T :addresses [addr...]
          :proposal-std σ :compile? bool :n-chains C :callback fn :key prng-key
          :device :cpu|:gpu}
   model: generative function
   args: model arguments
   observations: choice map of observed values

   Returns vector of [N-chains, D] JS arrays (one per sample).
   Metadata: {:acceptance-rate mean-rate}.
   Default device: :gpu (vectorized operations benefit from GPU parallelism)."
  [{:keys [samples burn thin addresses proposal-std compile? n-chains callback key device]
    :or {burn 0 thin 1 proposal-std 0.1 compile? true n-chains 10 device :gpu}}
   model args observations]
  (with-device device
    #(let [score-fn (u/make-vectorized-score-fn model args observations addresses)
        score-fn (if compile? (mx/compile-fn score-fn) score-fn)
        ;; Initialize N chains from independent generate calls
        init-params (mx/stack
                      (mapv (fn [_]
                              (let [{:keys [trace]} (p/generate model args observations)]
                                (u/extract-params trace addresses)))
                            (range n-chains)))
        param-shape (mx/shape init-params)
        std (mx/scalar proposal-std)
        total-iters (+ burn (* samples thin))
        d (count addresses)]
    (loop [i 0, state init-params, acc (transient []), n 0, total-accepted 0, rk key]
      (if (>= n samples)
        (with-meta (persistent! acc)
          {:acceptance-rate (/ total-accepted (* total-iters n-chains))})
        (let [[step-key next-key] (rng/split-or-nils rk)
              {:keys [state n-accepted]} (vectorized-mh-step
                                           state score-fn std param-shape n-chains step-key)
              past-burn? (>= i burn)
              keep? (and past-burn? (zero? (mod (- i burn) thin)))]
          (when (and callback keep?)
            (callback {:iter n :value (mx/->clj state) :n-accepted n-accepted}))
          (recur (inc i) state
                 (if keep? (conj! acc (mx/->clj state)) acc)
                 (if keep? (inc n) n)
                 (+ total-accepted n-accepted)
                 next-key)))))))

;; ---------------------------------------------------------------------------
;; Enumerative Gibbs Sampling
;; ---------------------------------------------------------------------------

(defn gibbs-step-with-support
  "One Gibbs step with explicit support enumeration.
   support-values: vector of possible values for the address.
   Returns a new trace."
  [current-trace addr support-values key]
  (let [gf (:gen-fn current-trace)
        ;; For each candidate value, compute model score
        log-scores (mapv (fn [val]
                           (let [constraint (cm/choicemap addr val)
                                 {:keys [trace]} (p/update gf current-trace constraint)]
                             (:score trace)))
                         support-values)
        ;; Normalize via log-softmax
        log-scores-arr (mx/array (mapv mx/realize log-scores))
        log-probs (mx/subtract log-scores-arr (mx/logsumexp log-scores-arr))
        ;; Sample from categorical
        chosen-idx (mx/realize (rng/categorical (rng/ensure-key key) log-probs))
        chosen-val (nth support-values (int chosen-idx))
        ;; Update trace with chosen value
        {:keys [trace]} (p/update gf current-trace (cm/choicemap addr chosen-val))]
    trace))

(defn gibbs
  "Gibbs sampling over discrete addresses with known support.

   opts: {:samples N :burn B :thin T :callback fn :key prng-key}
   model: generative function
   args: model arguments
   observations: choice map of observed values
   schedule: vector of {:addr keyword :support [values...]} maps
             specifying which addresses to sweep and their support."
  [{:keys [samples burn thin callback key]
    :or {burn 0 thin 1}}
   model args observations schedule]
  (let [{:keys [trace]} (p/generate model args observations)]
    (collect-samples
      {:samples samples :burn burn :thin thin :callback callback :key key}
      (fn [state step-key]
        (let [keys (rng/split-n (rng/ensure-key step-key) (count schedule))
              new-trace (reduce (fn [t [spec ki]]
                                  (gibbs-step-with-support
                                    t (:addr spec) (:support spec) ki))
                                state
                                (map vector schedule keys))]
          {:state new-trace :accepted? true}))
      identity
      trace)))

;; ---------------------------------------------------------------------------
;; Involutive MCMC
;; ---------------------------------------------------------------------------

(defn involutive-mh-step
  "One involutive MH step.
   proposal-gf: generative function for auxiliary randomness.
                Takes [current-trace-choices] and produces auxiliary choices.
   involution: pure function (fn [trace-cm aux-cm] -> [new-trace-cm new-aux-cm])
               Must be its own inverse: involution(involution(t, a)) = (t, a).
   model: the target model generative function.
   current-trace: the current model trace.
   key: PRNG key."
  [current-trace model proposal-gf involution key]
  (let [[k1 k2] (rng/split (rng/ensure-key key))
        ;; 1. Propose auxiliary choices
        fwd-result (p/propose proposal-gf [(:choices current-trace)])
        aux-choices (:choices fwd-result)
        fwd-score (:weight fwd-result)
        ;; 2. Apply involution (may return optional log|det J| as third element)
        inv-result (involution (:choices current-trace) aux-choices)
        new-trace-cm (nth inv-result 0)
        new-aux-cm (nth inv-result 1)
        log-abs-det-J (if (>= (count inv-result) 3)
                        (let [j (nth inv-result 2)]
                          (if (mx/array? j) j (mx/scalar j)))
                        (mx/scalar 0.0))
        ;; 3. Update model with new choices
        update-result (p/update model current-trace new-trace-cm)
        new-trace (:trace update-result)
        ;; 4. Score backward auxiliary choices under proposal
        bwd-result (p/assess proposal-gf [(:choices new-trace)] new-aux-cm)
        bwd-score (:weight bwd-result)
        ;; 5. Accept/reject: log-alpha = update-weight + bwd_score - fwd_score + log|det J|
        update-weight (:weight update-result)
        _ (mx/eval! update-weight fwd-score bwd-score log-abs-det-J)
        log-alpha (mx/realize (mx/add (mx/add update-weight
                                        (mx/subtract bwd-score fwd-score))
                                log-abs-det-J))]
    (if (u/accept-mh? log-alpha k2)
      new-trace
      current-trace)))

(defn involutive-mh
  "Involutive MCMC inference. Returns vector of traces.

   opts: {:samples N :burn B :thin T :proposal-gf gf :involution fn
          :callback fn :key prng-key}
   model: generative function
   args: model arguments
   observations: choice map of observed values"
  [{:keys [samples burn thin proposal-gf involution callback key]
    :or {burn 0 thin 1}}
   model args observations]
  (let [{:keys [trace]} (p/generate model args observations)]
    (collect-samples
      {:samples samples :burn burn :thin thin :callback callback :key key}
      (fn [state step-key]
        (let [new-trace (involutive-mh-step state model proposal-gf involution step-key)]
          {:state new-trace :accepted? (not (identical? new-trace state))}))
      identity
      trace)))

;; ---------------------------------------------------------------------------
;; MALA (Metropolis-Adjusted Langevin Algorithm)
;; ---------------------------------------------------------------------------

(defn- log-proposal-density
  "Log-density of a Gaussian proposal centered at `mean`, evaluated at `from`."
  [from mean two-eps-sq]
  (mx/negative (mx/divide (mx/sum (mx/square (mx/subtract from mean))) two-eps-sq)))

(defn- mala-step
  "One MALA step. Returns {:state q-next :accepted? bool}.
   val-grad-compiled computes score and gradient in a single compiled pass."
  [q val-grad-compiled eps half-eps2 two-eps-sq q-shape key]
  (let [[noise-key accept-key] (rng/split-or-nils key)
        ;; MALA proposal: q' = q + eps^2/2 * grad + eps * noise
        [_ g] (val-grad-compiled q)
        noise (if noise-key
                (rng/normal noise-key q-shape)
                (mx/random-normal q-shape))
        q' (mx/add q (mx/multiply half-eps2 g) (mx/multiply eps noise))
        _ (mx/eval! q' g)
        ;; Compute acceptance ratio with asymmetric proposal correction
        [score-q' g'] (val-grad-compiled q')
        _ (mx/eval! score-q' g')
        ;; Forward/backward proposal log-densities
        fwd-mean (mx/add q (mx/multiply half-eps2 g))
        bwd-mean (mx/add q' (mx/multiply half-eps2 g'))
        log-fwd (log-proposal-density q' fwd-mean two-eps-sq)
        log-bwd (log-proposal-density q bwd-mean two-eps-sq)
        ;; Score at q — reuse val-grad
        [score-q _] (val-grad-compiled q)
        _ (mx/eval! log-fwd log-bwd score-q)
        log-accept (+ (- (mx/item score-q') (mx/item score-q))
                     (- (mx/item log-bwd) (mx/item log-fwd)))
        accept? (u/accept-mh? log-accept accept-key)
        q-next (if accept? q' q)]
    {:state q-next :accepted? accept?}))

(defn- make-compiled-mala-chain
  "Build a compiled K-step MALA chain as one Metal dispatch.
   Returns compiled fn: (q [D], score scalar, grad [D], noise [K,D], uniforms [K])
     → #js [q', score', grad'].
   Score and gradient are threaded through iterations — only 1 val-grad call per
   step (at the proposal), compared to 3 in the eager mala-step."
  [k-steps val-grad-fn eps half-eps2 two-eps-sq n-params]
  (let [chain-fn
        (fn [q score-q grad-q noise-2d uniforms-1d]
          (loop [q q, sq score-q, gq grad-q, i 0]
            (if (>= i k-steps)
              #js [q sq gq]
              (let [;; Extract noise row i
                    noise-i (mx/reshape
                              (mx/take-idx noise-2d (mx/array [i] mx/int32) 0)
                              [n-params])
                    ;; Propose: q' = q + half-eps2 * grad + eps * noise
                    q' (mx/add q (mx/multiply half-eps2 gq)
                                 (mx/multiply eps noise-i))
                    ;; Score and gradient at proposal — THE ONLY val-grad call
                    [sq' gq'] (val-grad-fn q')
                    ;; Forward/backward means for asymmetric correction
                    fwd-mean (mx/add q (mx/multiply half-eps2 gq))
                    bwd-mean (mx/add q' (mx/multiply half-eps2 gq'))
                    ;; Log proposal densities
                    log-fwd (log-proposal-density q' fwd-mean two-eps-sq)
                    log-bwd (log-proposal-density q bwd-mean two-eps-sq)
                    ;; Acceptance ratio
                    log-alpha (mx/add (mx/subtract sq' sq)
                                      (mx/subtract log-bwd log-fwd))
                    log-u (mx/log (mx/index uniforms-1d i))
                    accept? (mx/greater log-alpha log-u)
                    ;; Branchless select (scalar accept? broadcasts to [D])
                    new-q  (mx/where accept? q' q)
                    new-sq (mx/where accept? sq' sq)
                    new-gq (mx/where accept? gq' gq)]
                (recur new-q new-sq new-gq (inc i))))))
        compiled (mx/compile-fn chain-fn)]
    ;; Warm-up trace call to cache the Metal program
    (let [init-q (mx/zeros [n-params])
          [init-s init-g] (val-grad-fn init-q)]
      (mx/eval! init-s init-g)
      (mx/eval! (compiled init-q init-s init-g
                          (mx/random-normal [k-steps n-params])
                          (mx/random-uniform [k-steps]))))
    compiled))

(defn- run-loop-compiled-mala
  "Run compiled MALA with loop compilation for burn-in and collection.
   Threads [q, score, grad] across chain blocks, eliminating redundant
   val-grad calls. Uses compiled chains for both burn-in and thinning."
  [{:keys [samples burn thin callback]} init-q n-params
   val-grad-compiled burn-chain burn-block-size thin-chain thin-steps]
  (let [;; Compute initial score and gradient
        [init-score init-grad] (val-grad-compiled init-q)
        _ (mx/eval! init-score init-grad)
        ;; Phase 1: Burn-in via compiled chain blocks
        [params score grad]
        (if (and burn-chain (> burn 0))
          (let [n-blocks (js/Math.ceil (/ burn burn-block-size))]
            (loop [q init-q, sq init-score, gq init-grad, b 0]
              (if (>= b n-blocks) [q sq gq]
                (let [noise (mx/random-normal [burn-block-size n-params])
                      uniforms (mx/random-uniform [burn-block-size])
                      r (burn-chain q sq gq noise uniforms)
                      q' (aget r 0) sq' (aget r 1) gq' (aget r 2)]
                  (mx/eval! q' sq' gq')
                  (recur q' sq' gq' (inc b))))))
          [init-q init-score init-grad])
        ;; Phase 2: Collect samples (mx/tidy prevents Metal resource leak)
        result (loop [q params, sq score, gq grad, acc (transient []), i 0]
                 (if (>= i samples)
                   (persistent! acc)
                   (let [r (mx/tidy
                             (fn []
                               (let [noise (mx/random-normal [thin-steps n-params])
                                     uniforms (mx/random-uniform [thin-steps])
                                     r (thin-chain q sq gq noise uniforms)]
                                 (mx/eval! (aget r 0) (aget r 1) (aget r 2))
                                 r)))
                         q' (aget r 0) sq' (aget r 1) gq' (aget r 2)]
                     (when callback
                       (callback {:iter i :value (mx/->clj q')}))
                     (recur q' sq' gq' (conj! acc (mx/->clj q')) (inc i)))))]
    result))

(defn mala
  "MALA inference using gradient information for proposals.

   When compile? is true (default), uses loop compilation: entire K-step chains
   are compiled into single Metal dispatches. Score and gradient are cached
   across iterations, reducing val-grad calls from 3 to 1 per step.

   opts: {:samples N :step-size eps :burn B :thin T :addresses [addr...]
          :compile? bool :callback fn :key prng-key :device :cpu|:gpu
          :block-size K}
   model: generative function
   args: model arguments
   observations: choice map of observed values
   Default device: :cpu (faster for scalar parameters)."
  [{:keys [samples step-size burn thin addresses compile? callback key device
           block-size]
    :or {step-size 0.01 burn 0 thin 1 compile? true device :cpu
         block-size 50}}
   model args observations]
  (with-device device
    #(let [score-fn          (u/make-score-fn model args observations addresses)
           val-grad-fn       (mx/value-and-grad score-fn)
           val-grad-compiled (mx/compile-fn val-grad-fn)
           eps              (mx/scalar step-size)
           half-eps2        (mx/scalar (* 0.5 step-size step-size))
           two-eps-sq       (mx/scalar (* 2.0 step-size step-size))
           {:keys [trace]} (p/generate model args observations)
           init-q           (u/extract-params trace addresses)
           n-params         (count addresses)
           q-shape          (mx/shape init-q)]
       (if compile?
         ;; Loop-compiled path
         (let [burn-block (min (max burn 1) block-size)
               burn-chain (when (> burn 0)
                            (make-compiled-mala-chain
                              burn-block val-grad-fn eps half-eps2 two-eps-sq n-params))
               thin-steps (max thin 1)
               thin-chain (make-compiled-mala-chain
                            thin-steps val-grad-fn eps half-eps2 two-eps-sq n-params)]
           (run-loop-compiled-mala
             {:samples samples :burn burn :thin thin :callback callback}
             init-q n-params val-grad-compiled
             burn-chain burn-block thin-chain thin-steps))
         ;; Fallback: per-step eager path
         (collect-samples
           {:samples samples :burn burn :thin thin :callback callback :key key}
           (fn [q step-key]
             (mala-step q val-grad-compiled eps half-eps2 two-eps-sq q-shape step-key))
           mx/->clj
           init-q)))))

;; ---------------------------------------------------------------------------
;; Vectorized MALA (N parallel chains)
;; ---------------------------------------------------------------------------

(defn- vectorized-log-proposal-density
  "Vectorized log-density of Gaussian proposals. All arrays [N,D].
   Returns [N]-shaped log-densities (sum over D dimension)."
  [from mean two-eps-sq]
  (mx/negative (mx/divide (mx/sum (mx/square (mx/subtract from mean)) [1]) two-eps-sq)))

(defn- vectorized-mala-step
  "One vectorized MALA step for N parallel chains.
   q: [N,D], grad-fn: [N,D]->[N,D], score-fn: [N,D]->[N].
   Accept/reject via mx/where per chain."
  [q score-fn grad-fn eps half-eps2 two-eps-sq n-chains key]
  (let [[noise-key accept-key] (rng/split-or-nils key)
        param-shape (mx/shape q)
        ;; Gradient at current position
        g (grad-fn q)
        ;; MALA proposal: q' = q + eps^2/2 * grad + eps * noise
        noise (if noise-key
                (rng/normal noise-key param-shape)
                (mx/random-normal param-shape))
        q' (mx/add q (mx/multiply half-eps2 g) (mx/multiply eps noise))
        _ (mx/eval! q' g)
        ;; Gradient at proposal
        g' (grad-fn q')
        _ (mx/eval! g')
        ;; Forward/backward proposal log-densities [N]
        fwd-mean (mx/add q (mx/multiply half-eps2 g))
        bwd-mean (mx/add q' (mx/multiply half-eps2 g'))
        log-fwd (vectorized-log-proposal-density q' fwd-mean two-eps-sq)
        log-bwd (vectorized-log-proposal-density q bwd-mean two-eps-sq)
        ;; Scores [N]
        score-q  (score-fn q)
        score-q' (score-fn q')
        _ (mx/eval! log-fwd log-bwd score-q score-q')
        ;; Per-chain acceptance ratio [N]
        log-alphas (mx/add (mx/subtract score-q' score-q)
                           (mx/subtract log-bwd log-fwd))
        u (if accept-key
            (rng/uniform accept-key [n-chains])
            (mx/random-uniform [n-chains]))
        accept-mask (mx/less (mx/log u) log-alphas)
        _ (mx/eval! accept-mask)
        n-accepted (mx/item (mx/sum accept-mask))
        ;; Per-chain select: expand mask [N] -> [N,1] for broadcasting against [N,D]
        new-q (mx/where (mx/expand-dims accept-mask 1) q' q)
        _ (mx/eval! new-q)]
    {:state new-q :n-accepted (int n-accepted)}))

(defn vectorized-mala
  "Vectorized MALA: N independent chains running in parallel.
   MLX broadcasting computes gradients for all N chains simultaneously
   via the sum trick.

   opts: {:samples N :burn B :thin T :step-size eps :addresses [addr...]
          :compile? bool :n-chains C :callback fn :key prng-key
          :device :cpu|:gpu}
   model: generative function
   args: model arguments
   observations: choice map of observed values

   Returns vector of [N-chains, D] JS arrays (one per sample).
   Metadata: {:acceptance-rate mean-rate}.
   Default device: :gpu (vectorized operations benefit from GPU parallelism)."
  [{:keys [samples burn thin step-size addresses compile? n-chains callback key device]
    :or {burn 0 thin 1 step-size 0.01 compile? true n-chains 10 device :gpu}}
   model args observations]
  (with-device device
    #(let [{:keys [score-fn grad-fn]}
           (u/make-compiled-vectorized-score-and-grad model args observations addresses)
           init-params (u/init-vectorized-params model args observations addresses n-chains)
           eps       (mx/scalar step-size)
           half-eps2 (mx/scalar (* 0.5 step-size step-size))
           two-eps-sq (mx/scalar (* 2.0 step-size step-size))
           total-iters (+ burn (* samples thin))]
       (loop [i 0, state init-params, acc (transient []), n 0, total-accepted 0, rk key]
         (if (>= n samples)
           (with-meta (persistent! acc)
             {:acceptance-rate (/ total-accepted (* total-iters n-chains))})
           (let [[step-key next-key] (rng/split-or-nils rk)
                 {:keys [state n-accepted]}
                   (vectorized-mala-step state score-fn grad-fn eps half-eps2 two-eps-sq
                                         n-chains step-key)
                 past-burn? (>= i burn)
                 keep? (and past-burn? (zero? (mod (- i burn) thin)))]
             (when (and callback keep?)
               (callback {:iter n :value (mx/->clj state) :n-accepted n-accepted}))
             (recur (inc i) state
                    (if keep? (conj! acc (mx/->clj state)) acc)
                    (if keep? (inc n) n)
                    (+ total-accepted n-accepted)
                    next-key)))))))

;; ---------------------------------------------------------------------------
;; Shared Hamiltonian helper
;; ---------------------------------------------------------------------------

;; ---------------------------------------------------------------------------
;; Mass matrix helpers for HMC / NUTS
;; ---------------------------------------------------------------------------

(defn- sample-momentum
  "Sample momentum p ~ N(0, M).
   metric nil → identity (standard normal).
   metric vector → diagonal M (scale by sqrt of diagonal).
   metric matrix → dense M (scale by Cholesky factor)."
  [metric q-shape key]
  (let [z (if key (rng/normal key q-shape) (mx/random-normal q-shape))]
    (cond
      (nil? metric)    z
      (= 1 (count (mx/shape metric)))  ;; diagonal: p = sqrt(diag) * z
        (mx/multiply (mx/sqrt metric) z)
      :else  ;; dense: p = L * z where M = L L^T
        (let [L (mx/cholesky metric)]
          (mx/squeeze (mx/matmul L (mx/reshape z [-1 1])))))))

(defn- kinetic-energy
  "Kinetic energy: 0.5 * p^T M^{-1} p.
   metric nil → 0.5 * sum(p^2).
   metric vector → 0.5 * sum(p^2 / diag).
   metric matrix → 0.5 * p^T M^{-1} p."
  [p metric half]
  (cond
    (nil? metric)
      (mx/multiply half (mx/sum (mx/square p)))
    (= 1 (count (mx/shape metric)))  ;; diagonal
      (mx/multiply half (mx/sum (mx/divide (mx/square p) metric)))
    :else  ;; dense
      (let [Minv-p (mx/solve metric p)]
        (mx/multiply half (mx/sum (mx/multiply p Minv-p))))))

(defn- inv-mass-multiply
  "Compute M^{-1} * p for the leapfrog position update.
   metric nil → p (identity).
   metric vector → p / diag.
   metric matrix → M^{-1} p."
  [p metric]
  (cond
    (nil? metric)    p
    (= 1 (count (mx/shape metric)))  (mx/divide p metric)
    :else            (mx/solve metric p)))

(defn- hamiltonian
  "Compute H = neg-U(q) + kinetic-energy(p, M). Returns JS number."
  ([neg-U-fn q p half] (hamiltonian neg-U-fn q p half nil))
  ([neg-U-fn q p half metric]
   (let [neg-U (neg-U-fn q)
         K (kinetic-energy p metric half)]
     (mx/eval! neg-U K)
     (+ (mx/item neg-U) (mx/item K)))))

;; ---------------------------------------------------------------------------
;; HMC (Hamiltonian Monte Carlo)
;; ---------------------------------------------------------------------------

(defn- leapfrog-step
  "Single leapfrog step with eval per step to bound graph size.
   Used by NUTS which needs per-step control.
   Optional metric for mass matrix (nil = identity).
   Returns [q p] Clojure vector."
  ([grad-U q p eps half-eps] (leapfrog-step grad-U q p eps half-eps nil))
  ([grad-U q p eps half-eps metric]
   (let [r (mx/tidy
             (fn []
               (let [g (grad-U q)
                     p (mx/subtract p (mx/multiply half-eps g))
                     q (mx/add q (mx/multiply eps (inv-mass-multiply p metric)))
                     g (grad-U q)
                     p (mx/subtract p (mx/multiply half-eps g))]
                 (mx/eval! q p)
                 #js [q p])))]
     [(aget r 0) (aget r 1)])))

(defn- leapfrog-trajectory
  "Run L leapfrog steps (unfused, used by NUTS)."
  ([grad-U q p eps half-eps L] (leapfrog-trajectory grad-U q p eps half-eps L nil))
  ([grad-U q p eps half-eps L metric]
   (loop [i 0, q q, p p]
     (if (>= i L)
       [q p]
       (let [[q' p'] (leapfrog-step grad-U q p eps half-eps metric)]
         (recur (inc i) q' p'))))))

(defn- leapfrog-trajectory-fused
  "Fused leapfrog: L+1 gradient evals instead of 2L.
   Adjacent half-kicks between steps are merged into full kicks.
   Builds one lazy graph — no per-step eval or tidy.
   Optional metric for mass matrix (nil = identity)."
  ([grad-U q p eps half-eps L] (leapfrog-trajectory-fused grad-U q p eps half-eps L nil))
  ([grad-U q p eps half-eps L metric]
   ;; Initial half-kick
   (let [g (grad-U q)
         p (mx/subtract p (mx/multiply half-eps g))
         ;; First drift: q += eps * M^{-1} p
         q (mx/add q (mx/multiply eps (inv-mass-multiply p metric)))]
     ;; L-1 interior steps: full kick (two halves fused) + drift
     (loop [i 1, q q, p p]
       (if (>= i L)
         ;; Final half-kick only (no more drift)
         (let [g (grad-U q)
               p (mx/subtract p (mx/multiply half-eps g))]
           [q p])
         (let [g (grad-U q)
               p (mx/subtract p (mx/multiply eps g))
               q (mx/add q (mx/multiply eps (inv-mass-multiply p metric)))]
           (recur (inc i) q p)))))))

(defn- hmc-step
  "One HMC step. Returns {:state q-next :accepted? bool}.
   metric: nil (identity), vector (diagonal), or matrix (dense)."
  [q neg-U-compiled grad-neg-U eps half-eps half q-shape leapfrog-steps metric key]
  (let [[momentum-key accept-key] (rng/split-or-nils key)
        ;; Sample momentum from N(0, M)
        p0 (doto (sample-momentum metric q-shape momentum-key) mx/eval!)
        ;; Current Hamiltonian
        current-H (hamiltonian neg-U-compiled q p0 half metric)
        ;; Fused leapfrog — L+1 gradient evals, one lazy graph
        [q' p'] (let [r (mx/tidy
                          (fn []
                            (let [[q' p'] (leapfrog-trajectory-fused
                                            grad-neg-U q p0 eps half-eps
                                            leapfrog-steps metric)]
                              (mx/eval! q' p')
                              #js [q' p'])))]
                  [(aget r 0) (aget r 1)])
        ;; Proposed Hamiltonian
        proposed-H (hamiltonian neg-U-compiled q' p' half metric)
        ;; Accept/reject
        log-accept (- current-H proposed-H)
        accept? (u/accept-mh? log-accept accept-key)
        q-next (if accept? q' q)]
    {:state q-next :accepted? accept?}))

(defn- make-compiled-hmc-chain
  "Build a compiled K-step HMC chain as one Metal dispatch.
   Each step contains L leapfrog sub-steps (fused: L+1 gradient evals per step).
   Identity mass matrix only.
   Returns compiled fn: (q [D], momentum [K,D], uniforms [K]) → q [D].
   Momentum and uniforms are generated OUTSIDE — compile-fn freezes random ops."
  [k-steps neg-U-fn grad-neg-U eps half-eps half n-params leapfrog-steps]
  (let [chain-fn
        (fn [q momentum-2d uniforms-1d]
          (loop [q q, k 0]
            (if (>= k k-steps) q
              (let [;; Extract momentum row k
                    p0 (mx/reshape
                         (mx/take-idx momentum-2d (mx/array [k] mx/int32) 0)
                         [n-params])
                    ;; Current Hamiltonian: H = neg-U(q) + 0.5*sum(p²)
                    current-H (mx/add (neg-U-fn q)
                                      (mx/multiply half (mx/sum (mx/square p0))))
                    ;; Inline fused leapfrog (L+1 grad evals instead of 2L)
                    ;; Initial half-kick: p -= half-eps * grad(neg-U)
                    g0 (grad-neg-U q)
                    p1 (mx/subtract p0 (mx/multiply half-eps g0))
                    ;; First drift: q += eps * p (identity mass)
                    q1 (mx/add q (mx/multiply eps p1))
                    ;; L-1 interior steps: full kick + drift
                    [q-lf p-lf]
                    (loop [j 1, qj q1, pj p1]
                      (if (>= j leapfrog-steps) [qj pj]
                        (let [gj (grad-neg-U qj)
                              pj (mx/subtract pj (mx/multiply eps gj))
                              qj (mx/add qj (mx/multiply eps pj))]
                          (recur (inc j) qj pj))))
                    ;; Final half-kick
                    g-final (grad-neg-U q-lf)
                    p-final (mx/subtract p-lf (mx/multiply half-eps g-final))
                    ;; Proposed Hamiltonian
                    proposed-H (mx/add (neg-U-fn q-lf)
                                       (mx/multiply half (mx/sum (mx/square p-final))))
                    ;; Accept/reject
                    log-alpha (mx/subtract current-H proposed-H)
                    log-u (mx/log (mx/index uniforms-1d k))
                    accept? (mx/greater log-alpha log-u)]
                (recur (mx/where accept? q-lf q) (inc k))))))
        compiled (mx/compile-fn chain-fn)]
    ;; Warm-up trace call
    (mx/eval! (compiled (mx/zeros [n-params])
                        (mx/random-normal [k-steps n-params])
                        (mx/random-uniform [k-steps])))
    compiled))

(defn- run-loop-compiled-hmc
  "Run compiled HMC with loop compilation for burn-in and optional thinning.
   Uses compiled chains for burn-in (block-size steps per dispatch).
   For collection: compiled chain if thin > 1, eager step if thin = 1."
  [{:keys [samples burn thin callback]} init-q n-params
   neg-U-compiled grad-neg-U eps half-eps half q-shape
   leapfrog-steps metric burn-chain burn-block-size thin-chain]
  (let [;; Phase 1: Burn-in via compiled chain blocks
        params (if (and burn-chain (> burn 0))
                 (let [n-blocks (js/Math.ceil (/ burn burn-block-size))]
                   (loop [q init-q, b 0]
                     (if (>= b n-blocks) q
                       (let [momentum (mx/random-normal [burn-block-size n-params])
                             uniforms (mx/random-uniform [burn-block-size])
                             q' (burn-chain q momentum uniforms)]
                         (mx/eval! q')
                         (recur q' (inc b))))))
                 init-q)
        ;; Phase 2: Collect samples (mx/tidy prevents Metal resource leak)
        result (loop [q params, acc (transient []), i 0]
                 (if (>= i samples)
                   (persistent! acc)
                   (let [q' (mx/tidy
                              (fn []
                                (if thin-chain
                                  ;; thin > 1: compiled chain of thin steps
                                  (let [momentum (mx/random-normal [thin n-params])
                                        uniforms (mx/random-uniform [thin])
                                        r (thin-chain q momentum uniforms)]
                                    (mx/eval! r) r)
                                  ;; thin = 1: eager step
                                  (let [{:keys [state]}
                                        (hmc-step q neg-U-compiled grad-neg-U
                                                  eps half-eps half q-shape
                                                  leapfrog-steps metric nil)]
                                    state))))]
                     (when callback
                       (callback {:iter i :value (mx/->clj q')}))
                     (recur q' (conj! acc (mx/->clj q')) (inc i)))))]
    result))

(defn hmc
  "Hamiltonian Monte Carlo sampling.

   When compile? is true (default) and metric is nil, uses loop compilation:
   entire K-step chains (each with L leapfrog sub-steps) are compiled into
   single Metal dispatches.

   opts: {:samples N :step-size eps :leapfrog-steps L :burn B
          :thin T :addresses [addr...] :compile? bool :callback fn :key prng-key
          :metric M :block-size K :device :cpu|:gpu}

   metric: mass matrix (optional, default identity).
     nil           — identity mass matrix (standard HMC)
     MLX vector    — diagonal mass matrix (vector of diagonal entries)
     MLX matrix    — dense mass matrix (positive definite)

   model: generative function
   args: model arguments
   observations: choice map of observed values

   Returns vector of MLX arrays (parameter samples).
   Default device: :cpu (faster for scalar parameters)."
  [{:keys [samples step-size leapfrog-steps burn thin addresses compile? callback
           key metric device block-size]
    :or {step-size 0.01 leapfrog-steps 20 burn 100 thin 1 compile? true
         device :cpu block-size 20}}
   model args observations]
  (with-device device
    #(let [score-fn (u/make-score-fn model args observations addresses)
           neg-U    (fn [q] (mx/negative (score-fn q)))
           grad-neg-U-raw (mx/grad neg-U)
           grad-neg-U (if compile? (mx/compile-fn grad-neg-U-raw) grad-neg-U-raw)
           neg-U-compiled (if compile? (mx/compile-fn neg-U) neg-U)
           eps      (mx/scalar step-size)
           half-eps (mx/scalar (* 0.5 step-size))
           half     (mx/scalar 0.5)
           {:keys [trace]} (p/generate model args observations)
           init-q (u/extract-params trace addresses)
           n-params (count addresses)
           q-shape (mx/shape init-q)]
       (if (and compile? (nil? metric))
         ;; Loop-compiled path (identity mass matrix only)
         (let [burn-block (min (max burn 1) block-size)
               burn-chain (when (> burn 0)
                            (make-compiled-hmc-chain
                              burn-block neg-U grad-neg-U-raw
                              eps half-eps half n-params leapfrog-steps))
               thin-chain (when (> thin 1)
                            (make-compiled-hmc-chain
                              thin neg-U grad-neg-U-raw
                              eps half-eps half n-params leapfrog-steps))]
           (run-loop-compiled-hmc
             {:samples samples :burn burn :thin thin :callback callback}
             init-q n-params neg-U-compiled grad-neg-U eps half-eps half q-shape
             leapfrog-steps metric burn-chain burn-block thin-chain))
         ;; Fallback: per-step eager path (or non-identity metric)
         (collect-samples
           {:samples samples :burn burn :thin thin :callback callback :key key}
           (fn [q step-key]
             (hmc-step q neg-U-compiled grad-neg-U eps half-eps half q-shape
                       leapfrog-steps metric step-key))
           mx/->clj
           init-q)))))

;; ---------------------------------------------------------------------------
;; Vectorized HMC (N parallel chains)
;; ---------------------------------------------------------------------------

(defn- vectorized-kinetic-energy
  "Kinetic energy for [N,D] momentum. Returns [N]-shaped energies.
   Identity mass matrix only (metric=nil)."
  [p half]
  (mx/multiply half (mx/sum (mx/square p) [1])))

(defn- vectorized-leapfrog-fused
  "Fused leapfrog for [N,D]-shaped position and momentum.
   grad-fn: [N,D] -> [N,D] (per-chain gradients via sum trick).
   All operations broadcast naturally over N chains."
  [grad-fn q p eps half-eps L]
  ;; Initial half-kick
  (let [g (grad-fn q)
        p (mx/subtract p (mx/multiply half-eps g))
        q (mx/add q (mx/multiply eps p))]
    ;; L-1 interior steps: full kick + drift
    (loop [i 1, q q, p p]
      (if (>= i L)
        ;; Final half-kick only
        (let [g (grad-fn q)
              p (mx/subtract p (mx/multiply half-eps g))]
          [q p])
        (let [g (grad-fn q)
              p (mx/subtract p (mx/multiply eps g))
              q (mx/add q (mx/multiply eps p))]
          (recur (inc i) q p))))))

(defn- vectorized-hmc-step
  "One vectorized HMC step for N parallel chains.
   q: [N,D]. neg-U-fn: [N,D]->[N]. grad-fn: [N,D]->[N,D].
   Accept/reject via mx/where per chain."
  [q neg-U-fn grad-fn eps half-eps half n-chains leapfrog-steps key]
  (let [[momentum-key accept-key] (rng/split-or-nils key)
        param-shape (mx/shape q)
        ;; Sample [N,D] momentum
        p0 (if momentum-key
             (rng/normal momentum-key param-shape)
             (mx/random-normal param-shape))
        _ (mx/eval! p0)
        ;; Current Hamiltonian [N] = neg-U(q) + K(p)
        neg-U-q (neg-U-fn q)
        K-current (vectorized-kinetic-energy p0 half)
        current-H (mx/add neg-U-q K-current)
        _ (mx/eval! current-H)
        ;; Fused leapfrog — grad-fn uses sum trick for [N,D] gradients
        ;; Note: grad-fn computes d(score)/d(params), neg-U = -score,
        ;; so gradient of neg-U = -grad-fn. Leapfrog subtracts grad of neg-U,
        ;; which is subtracting (-grad-fn) = adding grad-fn. We negate here.
        neg-grad-fn (fn [q] (mx/negative (grad-fn q)))
        [q' p'] (vectorized-leapfrog-fused neg-grad-fn q p0 eps half-eps leapfrog-steps)
        _ (mx/eval! q' p')
        ;; Proposed Hamiltonian [N]
        neg-U-q' (neg-U-fn q')
        K-proposed (vectorized-kinetic-energy p' half)
        proposed-H (mx/add neg-U-q' K-proposed)
        _ (mx/eval! proposed-H)
        ;; Per-chain accept/reject [N]
        log-alphas (mx/subtract current-H proposed-H)
        u (if accept-key
            (rng/uniform accept-key [n-chains])
            (mx/random-uniform [n-chains]))
        accept-mask (mx/less (mx/log u) log-alphas)
        _ (mx/eval! accept-mask)
        n-accepted (mx/item (mx/sum accept-mask))
        new-q (mx/where (mx/expand-dims accept-mask 1) q' q)
        _ (mx/eval! new-q)]
    {:state new-q :n-accepted (int n-accepted)}))

(defn vectorized-hmc
  "Vectorized HMC: N independent chains running in parallel.
   MLX broadcasting computes leapfrog trajectories for all N chains
   simultaneously via the sum trick for gradients.

   opts: {:samples N :burn B :thin T :step-size eps :leapfrog-steps L
          :addresses [addr...] :n-chains C :callback fn :key prng-key
          :device :cpu|:gpu}

   Note: Identity mass matrix only (no metric support).

   Returns vector of [N-chains, D] JS arrays (one per sample).
   Metadata: {:acceptance-rate mean-rate}.
   Default device: :gpu."
  [{:keys [samples burn thin step-size leapfrog-steps addresses n-chains callback key device]
    :or {burn 100 thin 1 step-size 0.01 leapfrog-steps 20 n-chains 10 device :gpu}}
   model args observations]
  (with-device device
    #(let [;; Build vectorized score and gradient functions
           vec-score-fn (u/make-vectorized-score-fn model args observations addresses)
           compiled-score-fn (mx/compile-fn vec-score-fn)
           neg-U-fn (fn [q] (mx/negative (compiled-score-fn q)))
           grad-fn (mx/compile-fn (u/make-vectorized-grad-score model args observations addresses))
           init-params (u/init-vectorized-params model args observations addresses n-chains)
           eps       (mx/scalar step-size)
           half-eps  (mx/scalar (* 0.5 step-size))
           half      (mx/scalar 0.5)
           total-iters (+ burn (* samples thin))]
       (loop [i 0, state init-params, acc (transient []), n 0, total-accepted 0, rk key]
         (if (>= n samples)
           (with-meta (persistent! acc)
             {:acceptance-rate (/ total-accepted (* total-iters n-chains))})
           (let [[step-key next-key] (rng/split-or-nils rk)
                 {:keys [state n-accepted]}
                   (vectorized-hmc-step state neg-U-fn grad-fn eps half-eps half
                                         n-chains leapfrog-steps step-key)
                 past-burn? (>= i burn)
                 keep? (and past-burn? (zero? (mod (- i burn) thin)))]
             (when (and callback keep?)
               (callback {:iter n :value (mx/->clj state) :n-accepted n-accepted}))
             (recur (inc i) state
                    (if keep? (conj! acc (mx/->clj state)) acc)
                    (if keep? (inc n) n)
                    (+ total-accepted n-accepted)
                    next-key)))))))

;; ---------------------------------------------------------------------------
;; NUTS (No-U-Turn Sampler)
;; ---------------------------------------------------------------------------

(defn- compute-u-turn?
  "Check NUTS U-turn criterion."
  [q-minus q-plus p-minus p-plus]
  (let [diff (mx/subtract q-plus q-minus)
        check-fwd (mx/sum (mx/multiply diff p-plus))
        check-bwd (mx/sum (mx/multiply diff p-minus))]
    (mx/eval! check-fwd check-bwd)
    (and (>= (mx/item check-fwd) 0)
         (>= (mx/item check-bwd) 0))))

(defn- nuts-base-case
  "NUTS base case: single leapfrog step."
  [{:keys [neg-ld grad eps half-eps half metric]} q p v log-u current-H]
  (let [actual-eps (if (pos? v) eps (mx/negative eps))
        actual-half (if (pos? v) half-eps (mx/negative half-eps))
        [q' p'] (leapfrog-step grad q p actual-eps actual-half metric)
        proposed-H (hamiltonian neg-ld q' p' half metric)
        n' (if (<= log-u (- proposed-H)) 1 0)
        s' (< (- proposed-H current-H) 1000)
        alpha (min 1.0 (js/Math.exp (- current-H proposed-H)))]
    {:q-minus q' :p-minus p' :q-plus q' :p-plus p'
     :q' q' :n' n' :s' s' :alpha alpha :n-alpha 1}))

(defn- build-tree
  "Recursively build NUTS tree.
   `ctx` holds constant config: {:neg-ld :grad :eps :half-eps :half}.
   Optional `key` threads functional PRNG through recursion."
  [ctx q p log-u v j current-H key]
  (if (zero? j)
    (nuts-base-case ctx q p v log-u current-H)
    (let [[k1 k2 k3] (rng/split-n-or-nils key 3)
          tree1 (build-tree ctx q p log-u v (dec j) current-H k1)]
      (if (not (:s' tree1))
        tree1
        (let [[q2 p2] (if (pos? v)
                        [(:q-plus tree1) (:p-plus tree1)]
                        [(:q-minus tree1) (:p-minus tree1)])
              tree2 (build-tree ctx q2 p2 log-u v (dec j) current-H k2)
              total-n (+ (:n' tree1) (:n' tree2))
              accept-subtree? (and (pos? total-n)
                                   (if k3
                                     (< (mx/realize (rng/uniform k3 []))
                                        (/ (:n' tree2) total-n))
                                     (< (js/Math.random)
                                        (/ (:n' tree2) total-n))))
              q' (if accept-subtree? (:q' tree2) (:q' tree1))
              q-minus (if (pos? v) (:q-minus tree1) (:q-minus tree2))
              p-minus (if (pos? v) (:p-minus tree1) (:p-minus tree2))
              q-plus  (if (pos? v) (:q-plus tree2)  (:q-plus tree1))
              p-plus  (if (pos? v) (:p-plus tree2)  (:p-plus tree1))
              s' (and (:s' tree2)
                      (compute-u-turn? q-minus q-plus p-minus p-plus))]
          {:q-minus q-minus :p-minus p-minus
           :q-plus q-plus :p-plus p-plus
           :q' q' :n' total-n :s' s'
           :alpha (+ (:alpha tree1) (:alpha tree2))
           :n-alpha (+ (:n-alpha tree1) (:n-alpha tree2))})))))

(defn nuts
  "No-U-Turn Sampler (NUTS).

   opts: {:samples N :step-size eps :max-depth J :burn B :thin T
          :addresses [addr...] :compile? bool :callback fn :key prng-key
          :metric M}

   metric: mass matrix (optional, default identity). Same as HMC:
     nil           — identity mass matrix
     MLX vector    — diagonal mass matrix
     MLX matrix    — dense mass matrix (positive definite)

   Returns vector of MLX arrays (parameter samples).
   Default device: :cpu (faster for scalar parameters)."
  [{:keys [samples step-size max-depth burn thin addresses compile? callback key metric device]
    :or {step-size 0.01 max-depth 10 burn 0 thin 1 compile? true device :cpu}}
   model args observations]
  (with-device device
    #(let [score-fn (u/make-score-fn model args observations addresses)
           neg-log-density (fn [q] (mx/negative (score-fn q)))
           grad-neg-ld (let [g (mx/grad neg-log-density)]
                         (if compile? (mx/compile-fn g) g))
           neg-ld-compiled (if compile? (mx/compile-fn neg-log-density) neg-log-density)
           eps (mx/scalar step-size)
           half-eps (mx/scalar (* 0.5 step-size))
           half (mx/scalar 0.5)
           ctx {:neg-ld neg-ld-compiled :grad grad-neg-ld :eps eps :half-eps half-eps
                :half half :metric metric}
           {:keys [trace]} (p/generate model args observations)
           init-q (u/extract-params trace addresses)
           q-shape (mx/shape init-q)]
       (collect-samples
         {:samples samples :burn burn :thin thin :callback callback :key key}
         (fn [q step-key]
        (let [[momentum-key slice-key dir-key tree-key]
              (rng/split-n-or-nils step-key 4)
              ;; Sample momentum from N(0, M) and compute current Hamiltonian
              p0 (doto (sample-momentum metric q-shape momentum-key) mx/eval!)
              current-H (hamiltonian neg-ld-compiled q p0 half metric)
              log-u (+ (if slice-key
                         (js/Math.log (mx/realize (rng/uniform slice-key [])))
                         (js/Math.log (js/Math.random)))
                       (- current-H))
              ;; Build tree
              new-q (loop [j 0
                           q-minus q, p-minus p0
                           q-plus q, p-plus p0
                           q' q, depth-n 1, continue? true
                           dk dir-key, tk tree-key]
                      (if (or (not continue?) (>= j max-depth))
                        q'
                        (let [[dk1 dk-next] (rng/split-or-nils dk)
                              [tk1 tk-next] (rng/split-or-nils tk)
                              v (if dk1
                                  (if (< (mx/realize (rng/uniform dk1 [])) 0.5) -1 1)
                                  (if (< (js/Math.random) 0.5) -1 1))
                              [qs ps] (if (pos? v)
                                        [q-plus p-plus]
                                        [q-minus p-minus])
                              tree (build-tree ctx qs ps log-u v j current-H tk1)
                              accept? (and (:s' tree)
                                           (if tk1
                                             (let [[ak _] (rng/split tk1)]
                                               (< (mx/realize (rng/uniform ak []))
                                                  (/ (:n' tree) (max 1 depth-n))))
                                             (< (js/Math.random)
                                                (/ (:n' tree) (max 1 depth-n)))))
                              q'' (if accept? (:q' tree) q')
                              qm' (if (neg? v) (:q-minus tree) q-minus)
                              pm' (if (neg? v) (:p-minus tree) p-minus)
                              qp' (if (pos? v) (:q-plus tree) q-plus)
                              pp' (if (pos? v) (:p-plus tree) p-plus)
                              cont? (and (:s' tree)
                                         (compute-u-turn? qm' qp' pm' pp'))]
                          (recur (inc j) qm' pm' qp' pp' q''
                                 (+ depth-n (:n' tree)) cont?
                                 dk-next tk-next))))]
          {:state new-q :accepted? (not (identical? new-q q))}))
         mx/->clj
         init-q))))

;; ---------------------------------------------------------------------------
;; Elliptical Slice Sampling (Murray, Adams, MacKay 2010)
;; ---------------------------------------------------------------------------

(defn- log-prior-gaussian
  "Log-density of N(0, σ²*I) at params."
  [params prior-std d]
  (let [z (mx/divide params (mx/scalar prior-std))]
    (+ (* -0.5 (mx/realize (mx/sum (mx/square z))))
       (* (- d) (+ (js/Math.log prior-std) (* 0.5 (js/Math.log (* 2 js/Math.PI))))))))

(defn elliptical-slice-step
  "One elliptical slice sampling step for models with Gaussian priors.
   current-trace: current model trace
   selection: addresses with Gaussian prior to resample (vector of keywords)
   prior-std: scalar standard deviation of the Gaussian prior (N(0, prior-std^2*I))
   key: PRNG key
   Returns the new trace (always accepts — no MH rejection)."
  [current-trace selection prior-std key]
  (let [gf (:gen-fn current-trace)
        addrs (if (vector? selection) selection [selection])
        f (u/extract-params current-trace addrs)
        d (count addrs)
        [k1 k2 k3] (rng/split-n (rng/ensure-key key) 3)
        ;; 1. nu ~ N(0, σ²I)
        nu (mx/multiply (mx/scalar prior-std) (rng/normal k1 [d]))
        _ (mx/eval! nu)
        ;; 2. Current log-likelihood = total score - prior log-prob
        current-score (mx/realize (:score current-trace))
        current-ll (- current-score (log-prior-gaussian f prior-std d))
        ;; 3. Threshold
        log-y (+ current-ll (js/Math.log (mx/realize (rng/uniform k2 []))))
        ;; 4. Initial angle bracket
        init-theta (* 2.0 js/Math.PI (mx/realize (rng/uniform k3 [])))
        theta-min (- init-theta (* 2.0 js/Math.PI))
        theta-max init-theta]
    ;; 5. Shrinking bracket loop
    (loop [theta init-theta
           tmin theta-min
           tmax theta-max
           iter 0
           rk key]
      (let [;; Propose: f' = f*cos(θ) + nu*sin(θ)
            f' (mx/add (mx/multiply f (mx/scalar (js/Math.cos theta)))
                       (mx/multiply nu (mx/scalar (js/Math.sin theta))))
            _ (mx/eval! f')
            ;; Update trace at selected addresses
            constraints (reduce (fn [cm [i addr]]
                                  (cm/set-choice cm [addr] (mx/index f' i)))
                                cm/EMPTY (map-indexed vector addrs))
            {:keys [trace]} (p/update gf current-trace constraints)
            new-score (mx/realize (:score trace))
            new-ll (- new-score (log-prior-gaussian f' prior-std d))]
        (if (> new-ll log-y)
          trace
          (if (>= iter 100)
            current-trace
            (let [[tmin' tmax'] (if (< theta 0) [theta tmax] [tmin theta])
                  [rk1 rk2] (rng/split (rng/ensure-key rk))
                  new-theta (+ tmin' (* (- tmax' tmin') (mx/realize (rng/uniform rk1 []))))]
              (recur new-theta tmin' tmax' (inc iter) rk2))))))))

(defn elliptical-slice
  "Elliptical slice sampling for models with Gaussian priors.
   opts: {:samples N :burn B :thin T :selection [addr...] :prior-std σ :key k}
   model: generative function
   args: model arguments
   observations: choice map of observed values
   Returns vector of traces."
  [{:keys [samples burn thin selection prior-std key]
    :or {burn 0 thin 1 prior-std 1.0}}
   model args observations]
  (let [{:keys [trace]} (p/generate model args observations)]
    (collect-samples
      {:samples samples :burn burn :thin thin :key key}
      (fn [state step-key]
        (let [new-trace (elliptical-slice-step state selection prior-std step-key)]
          {:state new-trace :accepted? true}))
      identity
      trace)))

;; ---------------------------------------------------------------------------
;; MAP (Maximum A Posteriori) Optimization
;; ---------------------------------------------------------------------------

(defn map-optimize
  "MAP (Maximum A Posteriori) optimization via gradient ascent on log p(latents | obs).

   opts:
     :iterations  - gradient steps (default 1000)
     :optimizer   - :sgd or :adam (default :adam)
     :lr          - learning rate (default 0.01)
     :addresses   - vector of latent addresses to optimize
     :callback    - (fn [{:iter :score :params}])
     :device      - :cpu|:gpu (default :cpu)

   Returns {:trace Trace :score number :params [numbers] :score-history [numbers]}"
  [{:keys [iterations optimizer lr addresses callback device]
    :or {iterations 1000 optimizer :adam lr 0.01 device :cpu}}
   model args observations]
  (with-device device
    #(let [score-fn   (u/make-score-fn model args observations addresses)
           val-grad   (mx/compile-fn (mx/value-and-grad score-fn))
           {:keys [trace]} (p/generate model args observations)
           init-params (u/extract-params trace addresses)
           opt-state   (when (= optimizer :adam) (learn/adam-init init-params))]
       (loop [i 0
              params init-params
              opt-st opt-state
              history (transient [])]
         (if (>= i iterations)
           (let [final-cm (reduce (fn [cm [j addr]]
                                    (cm/set-choice cm [addr] (mx/index params j)))
                                  observations
                                  (map-indexed vector addresses))
                 {:keys [trace]} (p/generate model args final-cm)
                 score (mx/realize (:score trace))]
             {:trace trace
              :score score
              :params (mx/->clj params)
              :score-history (persistent! history)})
           (let [[score grad] (val-grad params)
                 neg-grad (mx/negative grad)
                 _ (mx/eval! score neg-grad)
                 score-val (mx/item score)
                 [new-params new-opt-st]
                 (case optimizer
                   :sgd [(learn/sgd-step params neg-grad lr) nil]
                   :adam (learn/adam-step params neg-grad opt-st {:lr lr}))]
             (when callback
               (callback {:iter i :score score-val :params (mx/->clj params)}))
             (recur (inc i) new-params new-opt-st
                    (conj! history score-val))))))))

;; ---------------------------------------------------------------------------
;; Vectorized MAP (N random restarts in parallel)
;; ---------------------------------------------------------------------------

(defn vectorized-map-optimize
  "Vectorized MAP optimization: N random restarts optimized simultaneously.
   All N parameter vectors are updated in parallel via the sum trick for gradients.
   After optimization, returns the best restart (highest score).

   opts:
     :iterations  - gradient steps (default 1000)
     :optimizer   - :sgd or :adam (default :adam)
     :lr          - learning rate (default 0.01)
     :addresses   - vector of latent addresses to optimize
     :n-restarts  - number of parallel random restarts (default 10)
     :callback    - (fn [{:iter :best-score :scores}])
     :device      - :cpu|:gpu (default :gpu)

   Returns {:params [best-params] :score best-score
            :all-params [N-restarts, D] :all-scores [N]
            :score-history [best-score-per-iter]}"
  [{:keys [iterations optimizer lr addresses n-restarts callback device]
    :or {iterations 1000 optimizer :adam lr 0.01 n-restarts 10 device :gpu}}
   model args observations]
  (with-device device
    #(let [{:keys [score-fn grad-fn]}
           (u/make-compiled-vectorized-score-and-grad model args observations addresses)
           init-params (u/init-vectorized-params model args observations addresses n-restarts)
           ;; Adam state: m and v are [N,D] — element-wise ops broadcast naturally
           opt-state (when (= optimizer :adam)
                       {:m (mx/zeros (mx/shape init-params))
                        :v (mx/zeros (mx/shape init-params))
                        :t 0})
           ;; Hoist constants outside loop to reduce per-iteration allocations
           lr-s (mx/scalar lr)
           b1-s (mx/scalar 0.9)  b1c-s (mx/scalar 0.1)
           b2-s (mx/scalar 0.999) b2c-s (mx/scalar 0.001)
           eps-s (mx/scalar 1e-8)]
       (loop [i 0
              params init-params
              opt-st opt-state
              history (transient [])]
         (if (>= i iterations)
           ;; Find best restart
           (let [final-scores (score-fn params)
                 _ (mx/eval! final-scores params)
                 best-idx (mx/item (mx/argmax final-scores))
                 best-params (mx/take-idx params (mx/array best-idx mx/int32))
                 best-score (mx/item (mx/index final-scores best-idx))
                 ;; Reconstruct trace from best params
                 final-cm (reduce (fn [cm [j addr]]
                                    (cm/set-choice cm [addr] (mx/index best-params j)))
                                  observations
                                  (map-indexed vector addresses))
                 {:keys [trace]} (p/generate model args final-cm)]
             {:trace trace
              :score best-score
              :params (mx/->clj best-params)
              :all-params (mx/->clj params)
              :all-scores (mx/->clj final-scores)
              :score-history (persistent! history)})
           ;; Gradient step for all N restarts simultaneously
           ;; Use mx/tidy to free intermediate arrays each iteration
           (let [r (mx/tidy
                     (fn []
                       (let [grad (grad-fn params)
                             neg-grad (mx/negative grad)
                             scores (score-fn params)
                             _ (mx/eval! scores neg-grad)
                             best-score (mx/item (mx/amax scores))
                             [new-params new-opt-st]
                             (case optimizer
                               :sgd (let [new-p (mx/subtract params (mx/multiply lr-s neg-grad))]
                                      (mx/eval! new-p)
                                      [new-p nil])
                               :adam (let [t (inc (:t opt-st))
                                           m (mx/add (mx/multiply b1-s (:m opt-st))
                                                     (mx/multiply b1c-s neg-grad))
                                           v (mx/add (mx/multiply b2-s (:v opt-st))
                                                     (mx/multiply b2c-s (mx/square neg-grad)))
                                           m-hat (mx/divide m (mx/scalar (- 1.0 (js/Math.pow 0.9 t))))
                                           v-hat (mx/divide v (mx/scalar (- 1.0 (js/Math.pow 0.999 t))))
                                           upd (mx/divide m-hat (mx/add (mx/sqrt v-hat) eps-s))
                                           new-p (mx/subtract params (mx/multiply lr-s upd))]
                                       (mx/eval! new-p m v)
                                       [new-p {:m m :v v :t t}]))]
                         #js [new-params new-opt-st best-score])))
                 new-params (aget r 0)
                 new-opt-st (aget r 1)
                 best-score (aget r 2)]
             (when callback
               (callback {:iter i :best-score best-score}))
             (recur (inc i) new-params new-opt-st
                    (conj! history best-score))))))))
