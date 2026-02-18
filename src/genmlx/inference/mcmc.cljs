(ns genmlx.inference.mcmc
  "MCMC inference algorithms: MH, MALA, HMC, NUTS.
   MH uses the GFI regenerate operation.
   Gradient-based methods operate on compiled model score functions."
  (:require [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.mlx :as mx]))

;; ---------------------------------------------------------------------------
;; Metropolis-Hastings (via regenerate)
;; ---------------------------------------------------------------------------

(defn mh-step
  "One MH step. regenerate proposes + computes acceptance ratio."
  [current-trace selection]
  (let [gf (tr/get-gen-fn current-trace)
        result (p/regenerate gf current-trace selection)
        w (do (mx/eval! (:weight result)) (mx/item (:weight result)))
        accept? (or (>= w 0) (< (js/Math.log (js/Math.random)) w))]
    (if accept? (:trace result) current-trace)))

(defn mh
  "Metropolis-Hastings inference. Returns vector of traces.

   opts: {:samples N :burn B :thin T :selection sel}
   model: generative function
   args: model arguments
   observations: choice map of observed values"
  [{:keys [samples burn thin selection callback]
    :or {burn 0 thin 1 selection sel/all}}
   model args observations]
  (let [{:keys [trace]} (p/generate model args observations)]
    (loop [i 0, current trace, acc (transient []), n 0]
      (if (>= n samples)
        (persistent! acc)
        (let [current' (mh-step current selection)
              past-burn? (>= i burn)
              keep? (and past-burn? (zero? (mod (- i burn) thin)))]
          (when (and callback keep?)
            (callback {:iter n :trace current'}))
          (recur (inc i)
                 current'
                 (if keep? (conj! acc current') acc)
                 (if keep? (inc n) n)))))))

;; ---------------------------------------------------------------------------
;; MALA (Metropolis-Adjusted Langevin Algorithm)
;; ---------------------------------------------------------------------------

(defn mala
  "MALA inference using gradient information for proposals.

   opts: {:samples N :step-size eps :burn B :thin T :addresses [addr...]}
   model: generative function
   args: model arguments
   observations: choice map of observed values"
  [{:keys [samples step-size burn thin addresses callback]
    :or {step-size 0.01 burn 0 thin 1}}
   model args observations]
  (let [indexed-addrs (mapv vector (range) addresses)
        score-fn (fn [params]
                   (let [cm (reduce
                              (fn [cm [i addr]]
                                (cm/set-choice cm [addr] (mx/index params i)))
                              observations
                              indexed-addrs)]
                     (:weight (p/generate model args cm))))
        score-fn-compiled (mx/compile-fn score-fn)
        grad-score (mx/compile-fn (mx/grad score-fn))
        eps (mx/scalar step-size)
        half-eps2 (mx/scalar (* 0.5 step-size step-size))
        ;; Initialize
        {:keys [trace]} (p/generate model args observations)
        init-q (mx/array (mapv #(let [v (cm/get-choice (tr/get-choices trace) [%])]
                                  (mx/eval! v) (mx/item v))
                               addresses))
        q-shape (mx/shape init-q)]
    (loop [i 0, q init-q, acc (transient []), n 0]
      (if (>= n samples)
        (persistent! acc)
        (let [;; MALA proposal: q' = q + eps^2/2 * grad + eps * noise
              g (grad-score q)
              noise (mx/random-normal q-shape)
              q' (mx/add q (mx/add (mx/multiply half-eps2 g)
                                    (mx/multiply eps noise)))
              _ (mx/eval! q' g)
              ;; Compute acceptance ratio with asymmetric proposal correction
              g' (grad-score q')
              _ (mx/eval! g')
              ;; Forward proposal log-density
              fwd-mean (mx/add q (mx/multiply half-eps2 g))
              fwd-diff (mx/subtract q' fwd-mean)
              log-fwd (mx/negative (mx/divide (mx/sum (mx/square fwd-diff))
                                               (mx/scalar (* 2.0 step-size step-size))))
              ;; Backward proposal log-density
              bwd-mean (mx/add q' (mx/multiply half-eps2 g'))
              bwd-diff (mx/subtract q bwd-mean)
              log-bwd (mx/negative (mx/divide (mx/sum (mx/square bwd-diff))
                                               (mx/scalar (* 2.0 step-size step-size))))
              ;; Score difference — compiled forward pass
              score-q  (score-fn-compiled q)
              score-q' (score-fn-compiled q')
              _ (mx/eval! log-fwd log-bwd score-q score-q')
              log-accept (+ (- (mx/item score-q') (mx/item score-q))
                           (- (mx/item log-bwd) (mx/item log-fwd)))
              accept? (or (>= log-accept 0) (< (js/Math.log (js/Math.random)) log-accept))
              q-next (if accept? q' q)
              past-burn? (>= i burn)
              keep? (and past-burn? (zero? (mod (- i burn) thin)))]
          (when (and callback keep?)
            (callback {:iter n :value (mx/->clj q-next) :accepted? accept?}))
          (recur (inc i) q-next
                 (if keep? (conj! acc q-next) acc)
                 (if keep? (inc n) n)))))))

;; ---------------------------------------------------------------------------
;; HMC (Hamiltonian Monte Carlo)
;; ---------------------------------------------------------------------------

(defn- leapfrog-step
  "Single leapfrog step with eval per step to bound graph size.
   Used by NUTS which needs per-step control."
  [grad-U q p eps half-eps]
  (mx/tidy
    (fn []
      (let [g (grad-U q)
            p (mx/subtract p (mx/multiply half-eps g))
            q (mx/add q (mx/multiply eps p))
            g (grad-U q)
            p (mx/subtract p (mx/multiply half-eps g))]
        (mx/eval! q p)
        #js [q p]))))

(defn- leapfrog-trajectory
  "Run L leapfrog steps (unfused, used by NUTS)."
  [grad-U q p eps half-eps L]
  (loop [i 0, q q, p p]
    (if (>= i L)
      [q p]
      (let [result (leapfrog-step grad-U q p eps half-eps)]
        (recur (inc i) (aget result 0) (aget result 1))))))

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

(defn hmc
  "Hamiltonian Monte Carlo sampling.

   opts: {:samples N :step-size eps :leapfrog-steps L :burn B
          :thin T :addresses [addr...] :compile? bool :callback fn}
   model: generative function
   args: model arguments
   observations: choice map of observed values

   Returns vector of MLX arrays (parameter samples)."
  [{:keys [samples step-size leapfrog-steps burn thin addresses compile? callback]
    :or {step-size 0.01 leapfrog-steps 20 burn 100 thin 1 compile? true}}
   model args observations]
  (let [;; Pre-compute indexed addresses once (avoid allocation per gradient call)
        indexed-addrs (mapv vector (range) addresses)
        score-fn (fn [params]
                   (let [cm (reduce
                              (fn [cm [i addr]]
                                (cm/set-choice cm [addr] (mx/index params i)))
                              observations
                              indexed-addrs)]
                     (:weight (p/generate model args cm))))
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
        init-q (mx/array (mapv #(let [v (cm/get-choice (tr/get-choices trace) [%])]
                                  (mx/eval! v) (mx/item v))
                               addresses))
        q-shape (mx/shape init-q)]
    (loop [i 0, q init-q, acc (transient []), n 0, n-accepted 0]
      (if (>= n samples)
        (with-meta (persistent! acc)
          {:acceptance-rate (/ n-accepted (+ burn (* samples thin)))})
        (let [;; Sample momentum
              p0 (mx/random-normal q-shape)
              ;; Current Hamiltonian — compiled forward, no backward pass
              current-neg-U (neg-U-compiled q)
              current-K (mx/multiply half (mx/sum (mx/multiply p0 p0)))
              _ (mx/eval! p0 current-neg-U current-K)
              current-H (+ (mx/item current-neg-U) (mx/item current-K))
              ;; Fused leapfrog — L+1 gradient evals, one lazy graph
              result (mx/tidy
                       (fn []
                         (let [[q' p'] (leapfrog-trajectory-fused
                                         grad-neg-U q p0 eps half-eps leapfrog-steps)]
                           (mx/eval! q' p')
                           #js [q' p'])))
              q' (aget result 0)
              p' (aget result 1)
              ;; Proposed Hamiltonian — compiled forward, no backward pass
              proposed-neg-U (neg-U-compiled q')
              proposed-K (mx/multiply half (mx/sum (mx/multiply p' p')))
              _ (mx/eval! proposed-neg-U proposed-K)
              proposed-H (+ (mx/item proposed-neg-U) (mx/item proposed-K))
              ;; Accept/reject
              log-accept (- current-H proposed-H)
              accept? (or (>= log-accept 0) (< (js/Math.log (js/Math.random)) log-accept))
              q-next (if accept? q' q)
              past-burn? (>= i burn)
              keep? (and past-burn? (zero? (mod (- i burn) thin)))]
          (when (and callback keep?)
            (callback {:iter n :value (mx/->clj q-next) :accepted? accept?}))
          (recur (inc i) q-next
                 (if keep? (conj! acc q-next) acc)
                 (if keep? (inc n) n)
                 (if accept? (inc n-accepted) n-accepted)))))))

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
  [neg-log-density grad-neg-ld q p v eps half-eps half log-u current-H]
  (let [actual-eps (if (pos? v) eps (mx/negative eps))
        actual-half (if (pos? v) half-eps (mx/negative half-eps))
        result (leapfrog-step grad-neg-ld q p actual-eps actual-half)
        q' (aget result 0)
        p' (aget result 1)
        proposed-neg-U (neg-log-density q')
        proposed-K (mx/multiply half (mx/sum (mx/multiply p' p')))
        _ (mx/eval! proposed-neg-U proposed-K)
        proposed-H (+ (mx/item proposed-neg-U) (mx/item proposed-K))
        n' (if (<= log-u (- proposed-H)) 1 0)
        s' (< (- proposed-H current-H) 1000)
        alpha (min 1.0 (js/Math.exp (- current-H proposed-H)))]
    {:q-minus q' :p-minus p' :q-plus q' :p-plus p'
     :q' q' :n' n' :s' s' :alpha alpha :n-alpha 1}))

(defn- build-tree
  "Recursively build NUTS tree."
  [neg-log-density grad-neg-ld q p log-u v eps half-eps half j current-H]
  (if (zero? j)
    (nuts-base-case neg-log-density grad-neg-ld q p v eps half-eps half log-u current-H)
    (let [tree1 (build-tree neg-log-density grad-neg-ld
                             q p log-u v eps half-eps half (dec j) current-H)]
      (if (not (:s' tree1))
        tree1
        (let [[q2 p2] (if (pos? v)
                        [(:q-plus tree1) (:p-plus tree1)]
                        [(:q-minus tree1) (:p-minus tree1)])
              tree2 (build-tree neg-log-density grad-neg-ld
                                q2 p2 log-u v eps half-eps half (dec j) current-H)
              total-n (+ (:n' tree1) (:n' tree2))
              q' (if (and (pos? total-n)
                          (< (js/Math.random) (/ (:n' tree2) total-n)))
                   (:q' tree2)
                   (:q' tree1))
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
          :addresses [addr...] :compile? bool :callback fn}

   Returns vector of MLX arrays (parameter samples)."
  [{:keys [samples step-size max-depth burn thin addresses compile? callback]
    :or {step-size 0.01 max-depth 10 burn 0 thin 1 compile? true}}
   model args observations]
  (let [indexed-addrs (mapv vector (range) addresses)
        score-fn (fn [params]
                   (let [cm (reduce
                              (fn [cm [i addr]]
                                (cm/set-choice cm [addr] (mx/index params i)))
                              observations
                              indexed-addrs)]
                     (:weight (p/generate model args cm))))
        neg-log-density (fn [q] (mx/negative (score-fn q)))
        ;; Compile both gradient and forward pass
        grad-neg-ld (let [g (mx/grad neg-log-density)]
                      (if compile? (mx/compile-fn g) g))
        neg-ld-compiled (if compile? (mx/compile-fn neg-log-density) neg-log-density)
        eps (mx/scalar step-size)
        half-eps (mx/scalar (* 0.5 step-size))
        half (mx/scalar 0.5)
        {:keys [trace]} (p/generate model args observations)
        init-q (mx/array (mapv #(let [v (cm/get-choice (tr/get-choices trace) [%])]
                                  (mx/eval! v) (mx/item v))
                               addresses))
        q-shape (mx/shape init-q)]
    (loop [i 0, q init-q, acc (transient []), n 0]
      (if (>= n samples)
        (persistent! acc)
        (let [;; Sample momentum and compute current Hamiltonian
              p0 (mx/random-normal q-shape)
              current-neg-U (neg-ld-compiled q)
              current-K (mx/multiply half (mx/sum (mx/multiply p0 p0)))
              _ (mx/eval! p0 current-neg-U current-K)
              current-H (+ (mx/item current-neg-U) (mx/item current-K))
              log-u (+ (js/Math.log (js/Math.random)) (- current-H))
              ;; Build tree
              new-q (loop [j 0
                           q-minus q, p-minus p0
                           q-plus q, p-plus p0
                           q' q, depth-n 1, continue? true]
                      (if (or (not continue?) (>= j max-depth))
                        q'
                        (let [v (if (< (js/Math.random) 0.5) -1 1)
                              [qs ps] (if (pos? v)
                                        [q-plus p-plus]
                                        [q-minus p-minus])
                              tree (build-tree neg-ld-compiled grad-neg-ld
                                               qs ps log-u v eps half-eps half j current-H)
                              q'' (if (and (:s' tree)
                                           (< (js/Math.random)
                                              (/ (:n' tree) (max 1 depth-n))))
                                    (:q' tree) q')
                              qm' (if (neg? v) (:q-minus tree) q-minus)
                              pm' (if (neg? v) (:p-minus tree) p-minus)
                              qp' (if (pos? v) (:q-plus tree) q-plus)
                              pp' (if (pos? v) (:p-plus tree) p-plus)
                              cont? (and (:s' tree)
                                         (compute-u-turn? qm' qp' pm' pp'))]
                          (recur (inc j) qm' pm' qp' pp' q''
                                 (+ depth-n (:n' tree)) cont?))))
              past-burn? (>= i burn)
              keep? (and past-burn? (zero? (mod (- i burn) thin)))]
          (when (and callback keep?)
            (callback {:iter n :value (mx/->clj new-q)}))
          (recur (inc i) new-q
                 (if keep? (conj! acc new-q) acc)
                 (if keep? (inc n) n)))))))
