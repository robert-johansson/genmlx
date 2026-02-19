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
            [genmlx.inference.util :as u]))

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
         ;; 2. Apply proposed choices to model via update → new trace + update weight
         update-result (p/update model current-trace forward-choices)
         new-trace (:trace update-result)
         update-weight (:weight update-result)
         ;; 3. Compute backward score
         backward-score (if backward-gf
                          ;; Run assess on backward proposal
                          (let [{:keys [weight]} (p/assess backward-gf
                                                           [(:choices new-trace)]
                                                           (:choices current-trace))]
                            weight)
                          ;; Symmetric proposal: backward score = forward score
                          forward-score)
         ;; 4. Accept/reject: log-alpha = update-weight + backward-score - forward-score
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
        ;; 2. Apply involution
        [new-trace-cm new-aux-cm] (involution (:choices current-trace) aux-choices)
        ;; 3. Update model with new choices
        update-result (p/update model current-trace new-trace-cm)
        new-trace (:trace update-result)
        update-weight (:weight update-result)
        ;; 4. Score backward auxiliary choices under proposal
        bwd-result (p/assess proposal-gf [(:choices new-trace)] new-aux-cm)
        bwd-score (:weight bwd-result)
        ;; 5. Accept/reject
        log-alpha (mx/realize (mx/add update-weight
                                (mx/subtract bwd-score fwd-score)))]
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
  "One MALA step. Returns {:state q-next :accepted? bool}."
  [q score-fn-compiled grad-score eps half-eps2 two-eps-sq q-shape key]
  (let [[noise-key accept-key] (rng/split-or-nils key)
        ;; MALA proposal: q' = q + eps^2/2 * grad + eps * noise
        g (grad-score q)
        noise (if noise-key
                (rng/normal noise-key q-shape)
                (mx/random-normal q-shape))
        q' (mx/add q (mx/multiply half-eps2 g) (mx/multiply eps noise))
        _ (mx/eval! q' g)
        ;; Compute acceptance ratio with asymmetric proposal correction
        g' (doto (grad-score q') mx/eval!)
        ;; Forward/backward proposal log-densities
        fwd-mean (mx/add q (mx/multiply half-eps2 g))
        bwd-mean (mx/add q' (mx/multiply half-eps2 g'))
        log-fwd (log-proposal-density q' fwd-mean two-eps-sq)
        log-bwd (log-proposal-density q bwd-mean two-eps-sq)
        ;; Score difference — compiled forward pass
        score-q  (score-fn-compiled q)
        score-q' (score-fn-compiled q')
        _ (mx/eval! log-fwd log-bwd score-q score-q')
        log-accept (+ (- (mx/item score-q') (mx/item score-q))
                     (- (mx/item log-bwd) (mx/item log-fwd)))
        accept? (u/accept-mh? log-accept accept-key)
        q-next (if accept? q' q)]
    {:state q-next :accepted? accept?}))

(defn mala
  "MALA inference using gradient information for proposals.

   opts: {:samples N :step-size eps :burn B :thin T :addresses [addr...]
          :callback fn :key prng-key}
   model: generative function
   args: model arguments
   observations: choice map of observed values"
  [{:keys [samples step-size burn thin addresses callback key]
    :or {step-size 0.01 burn 0 thin 1}}
   model args observations]
  (let [score-fn         (u/make-score-fn model args observations addresses)
        score-fn-compiled (mx/compile-fn score-fn)
        grad-score       (mx/compile-fn (mx/grad score-fn))
        eps              (mx/scalar step-size)
        half-eps2        (mx/scalar (* 0.5 step-size step-size))
        two-eps-sq       (mx/scalar (* 2.0 step-size step-size))
        ;; Initialize
        {:keys [trace]} (p/generate model args observations)
        init-q           (u/extract-params trace addresses)
        q-shape          (mx/shape init-q)]
    (collect-samples
      {:samples samples :burn burn :thin thin :callback callback :key key}
      (fn [q step-key]
        (mala-step q score-fn-compiled grad-score eps half-eps2 two-eps-sq q-shape step-key))
      mx/->clj
      init-q)))

;; ---------------------------------------------------------------------------
;; Shared Hamiltonian helper
;; ---------------------------------------------------------------------------

(defn- hamiltonian
  "Compute H = neg-U(q) + 0.5 * sum(p^2). Evals both terms, returns JS number."
  [neg-U-fn q p half]
  (let [neg-U (neg-U-fn q)
        K (mx/multiply half (mx/sum (mx/square p)))]
    (mx/eval! neg-U K)
    (+ (mx/item neg-U) (mx/item K))))

;; ---------------------------------------------------------------------------
;; HMC (Hamiltonian Monte Carlo)
;; ---------------------------------------------------------------------------

(defn- leapfrog-step
  "Single leapfrog step with eval per step to bound graph size.
   Used by NUTS which needs per-step control.
   Returns [q p] Clojure vector."
  [grad-U q p eps half-eps]
  (let [r (mx/tidy
            (fn []
              (let [g (grad-U q)
                    p (mx/subtract p (mx/multiply half-eps g))
                    q (mx/add q (mx/multiply eps p))
                    g (grad-U q)
                    p (mx/subtract p (mx/multiply half-eps g))]
                (mx/eval! q p)
                #js [q p])))]
    [(aget r 0) (aget r 1)]))

(defn- leapfrog-trajectory
  "Run L leapfrog steps (unfused, used by NUTS)."
  [grad-U q p eps half-eps L]
  (loop [i 0, q q, p p]
    (if (>= i L)
      [q p]
      (let [[q' p'] (leapfrog-step grad-U q p eps half-eps)]
        (recur (inc i) q' p')))))

(defn- leapfrog-trajectory-fused
  "Fused leapfrog: L+1 gradient evals instead of 2L.
   Adjacent half-kicks between steps are merged into full kicks.
   Builds one lazy graph — no per-step eval or tidy."
  [grad-U q p eps half-eps L]
  ;; Initial half-kick
  (let [g (grad-U q)
        p (mx/subtract p (mx/multiply half-eps g))
        ;; First drift
        q (mx/add q (mx/multiply eps p))]
    ;; L-1 interior steps: full kick (two halves fused) + drift
    (loop [i 1, q q, p p]
      (if (>= i L)
        ;; Final half-kick only (no more drift)
        (let [g (grad-U q)
              p (mx/subtract p (mx/multiply half-eps g))]
          [q p])
        (let [g (grad-U q)
              p (mx/subtract p (mx/multiply eps g))
              q (mx/add q (mx/multiply eps p))]
          (recur (inc i) q p))))))

(defn- hmc-step
  "One HMC step. Returns {:state q-next :accepted? bool}."
  [q neg-U-compiled grad-neg-U eps half-eps half q-shape leapfrog-steps key]
  (let [[momentum-key accept-key] (rng/split-or-nils key)
        ;; Sample momentum
        p0 (doto (if momentum-key
                   (rng/normal momentum-key q-shape)
                   (mx/random-normal q-shape))
              mx/eval!)
        ;; Current Hamiltonian
        current-H (hamiltonian neg-U-compiled q p0 half)
        ;; Fused leapfrog — L+1 gradient evals, one lazy graph
        [q' p'] (let [r (mx/tidy
                          (fn []
                            (let [[q' p'] (leapfrog-trajectory-fused
                                            grad-neg-U q p0 eps half-eps leapfrog-steps)]
                              (mx/eval! q' p')
                              #js [q' p'])))]
                  [(aget r 0) (aget r 1)])
        ;; Proposed Hamiltonian
        proposed-H (hamiltonian neg-U-compiled q' p' half)
        ;; Accept/reject
        log-accept (- current-H proposed-H)
        accept? (u/accept-mh? log-accept accept-key)
        q-next (if accept? q' q)]
    {:state q-next :accepted? accept?}))

(defn hmc
  "Hamiltonian Monte Carlo sampling.

   opts: {:samples N :step-size eps :leapfrog-steps L :burn B
          :thin T :addresses [addr...] :compile? bool :callback fn :key prng-key}
   model: generative function
   args: model arguments
   observations: choice map of observed values

   Returns vector of MLX arrays (parameter samples)."
  [{:keys [samples step-size leapfrog-steps burn thin addresses compile? callback key]
    :or {step-size 0.01 leapfrog-steps 20 burn 100 thin 1 compile? true}}
   model args observations]
  (let [score-fn (u/make-score-fn model args observations addresses)
        neg-U    (fn [q] (mx/negative (score-fn q)))
        grad-neg-U (let [g (mx/grad neg-U)]
                     (if compile? (mx/compile-fn g) g))
        ;; Compile neg-U itself for fast Hamiltonian evaluation
        neg-U-compiled (if compile? (mx/compile-fn neg-U) neg-U)
        eps      (mx/scalar step-size)
        half-eps (mx/scalar (* 0.5 step-size))
        half     (mx/scalar 0.5)
        ;; Initialize
        {:keys [trace]} (p/generate model args observations)
        init-q (u/extract-params trace addresses)
        q-shape (mx/shape init-q)]
    (collect-samples
      {:samples samples :burn burn :thin thin :callback callback :key key}
      (fn [q step-key]
        (hmc-step q neg-U-compiled grad-neg-U eps half-eps half q-shape leapfrog-steps step-key))
      mx/->clj
      init-q)))

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
  [{:keys [neg-ld grad eps half-eps half]} q p v log-u current-H]
  (let [actual-eps (if (pos? v) eps (mx/negative eps))
        actual-half (if (pos? v) half-eps (mx/negative half-eps))
        [q' p'] (leapfrog-step grad q p actual-eps actual-half)
        proposed-H (hamiltonian neg-ld q' p' half)
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
          :addresses [addr...] :compile? bool :callback fn :key prng-key}

   Returns vector of MLX arrays (parameter samples)."
  [{:keys [samples step-size max-depth burn thin addresses compile? callback key]
    :or {step-size 0.01 max-depth 10 burn 0 thin 1 compile? true}}
   model args observations]
  (let [score-fn (u/make-score-fn model args observations addresses)
        neg-log-density (fn [q] (mx/negative (score-fn q)))
        ;; Compile both gradient and forward pass
        grad-neg-ld (let [g (mx/grad neg-log-density)]
                      (if compile? (mx/compile-fn g) g))
        neg-ld-compiled (if compile? (mx/compile-fn neg-log-density) neg-log-density)
        eps (mx/scalar step-size)
        half-eps (mx/scalar (* 0.5 step-size))
        half (mx/scalar 0.5)
        ctx {:neg-ld neg-ld-compiled :grad grad-neg-ld :eps eps :half-eps half-eps :half half}
        {:keys [trace]} (p/generate model args observations)
        init-q (u/extract-params trace addresses)
        q-shape (mx/shape init-q)]
    (collect-samples
      {:samples samples :burn burn :thin thin :callback callback :key key}
      (fn [q step-key]
        (let [[momentum-key slice-key dir-key tree-key]
              (rng/split-n-or-nils step-key 4)
              ;; Sample momentum and compute current Hamiltonian
              p0 (doto (if momentum-key
                         (rng/normal momentum-key q-shape)
                         (mx/random-normal q-shape))
                    mx/eval!)
              current-H (hamiltonian neg-ld-compiled q p0 half)
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
      init-q)))
