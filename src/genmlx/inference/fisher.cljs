(ns genmlx.inference.fisher
  "Fisher information matrix for parametric models.
   Enables model comparison (Laplace/BIC), natural gradient,
   confidence intervals, and posterior approximation.

   Uses the observed Fisher: F(θ) = -∇²_θ log p(y; θ), computed via
   central finite differences on the IS gradient from differentiable.cljs.
   Fixed random key ensures deterministic Hessian (same particles for
   all finite difference evaluations)."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dynamic :as dyn]
            [genmlx.vectorized :as vec]
            [genmlx.inference.differentiable :as diff]))

;; ---------------------------------------------------------------------------
;; Observed Fisher via Hessian of log-ML
;; ---------------------------------------------------------------------------

(defn- basis-vector
  "Create a one-hot basis vector e_i of dimension D."
  [i D]
  (let [data (vec (concat (repeat i 0.0) [1.0] (repeat (- D i 1) 0.0)))]
    (mx/array data)))

(defn- build-param-store
  "Build a param-store map from a flat params array and param names."
  [params-array param-names]
  {:params (into {}
             (map-indexed (fn [i nm] [nm (mx/index params-array i)])
                          param-names))})

(defn- eval-log-ml
  "Evaluate log p(y; θ) at a parameter point. Returns JS number.
   Directly runs vgenerate (no autograd), avoiding graph caching issues."
  [model args observations param-names params-array n-particles key]
  (let [_ (mx/materialize! params-array)  ;; Force lazy params to concrete values
        store (build-param-store params-array param-names)
        gf (vary-meta model assoc :genmlx.dynamic/param-store store)
        vtrace (dyn/vgenerate gf args observations n-particles key)
        log-ml (vec/vtrace-log-ml-estimate vtrace)]
    (mx/materialize! log-ml)
    (mx/item log-ml)))

(defn observed-fisher
  "Observed Fisher information matrix: F(θ) = -∇²_θ log p(y; θ).

   Computed via central finite differences on the scalar loss function.
   For D parameters, requires D²+1 loss evaluations (D diagonal + D*(D-1)/2
   off-diagonal, each needing 4 evaluations, sharing some).
   Uses a fixed random key so the IS estimate is deterministic
   (same particles for all evaluations).

   opts:
     :n-particles  - IS particles (default 2000)
     :key          - PRNG key (fixed for deterministic Hessian)
     :epsilon      - finite difference step size (default 0.01, larger for float32)

   Returns {:fisher [D,D] MLX array, :log-ml MLX scalar}."
  [{:keys [n-particles key epsilon] :or {n-particles 2000 epsilon 0.01}}
   model args observations param-names params-array]
  (let [key (rng/ensure-key key)
        model (dyn/auto-key model)  ;; Wrap once, reuse for all evaluations
        D (first (mx/shape params-array))
        eps2 (* epsilon epsilon)
        ;; Evaluate log p(y; θ). Fisher = -Hessian of log-ML, so we negate.
        f0 (eval-log-ml model args observations param-names
                            params-array n-particles key)
        ;; Pre-compute f(θ ± ε·e_i) for all i
        f-plus (mapv (fn [i]
                       (eval-log-ml model args observations param-names
                                        (mx/add params-array
                                                (mx/multiply (mx/scalar epsilon)
                                                             (basis-vector i D)))
                                        n-particles key))
                     (range D))
        f-minus (mapv (fn [i]
                        (eval-log-ml model args observations param-names
                                         (mx/subtract params-array
                                                      (mx/multiply (mx/scalar epsilon)
                                                                   (basis-vector i D)))
                                         n-particles key))
                      (range D))
        ;; Pre-compute f(θ+εe_i+εe_j) for off-diagonal pairs (i<j)
        ;; Store as flat map {[i j] -> value}
        f-cross (into {}
                  (for [i (range D) j (range (inc i) D)]
                    [[i j] (eval-log-ml
                              model args observations param-names
                              (mx/add params-array
                                      (mx/multiply (mx/scalar epsilon)
                                                   (mx/add (basis-vector i D)
                                                           (basis-vector j D))))
                              n-particles key)]))
        ;; Build Hessian as JS numbers
        hessian-data
        (vec (for [i (range D)]
               (vec (for [j (range D)]
                      (if (= i j)
                        ;; Central difference: -(f+ - 2f0 + f-) / ε²
                        ;; Negated because Fisher = -Hessian(log-ML)
                        (- (/ (+ (nth f-plus i) (- (* 2.0 f0)) (nth f-minus i))
                              eps2))
                        ;; Forward difference: -(f++ - fi+ - fj+ + f0) / ε²
                        (let [key-pair (if (< i j) [i j] [j i])]
                          (- (/ (+ (f-cross key-pair)
                                   (- (nth f-plus i))
                                   (- (nth f-plus j))
                                   f0)
                                eps2))))))))
        ;; Convert to MLX matrix
        fisher-flat (mx/array (vec (apply concat hessian-data)))
        fisher (mx/reshape fisher-flat [D D])]
    (mx/materialize! fisher)
    {:fisher fisher
     :log-ml (mx/scalar f0)}))

;; ---------------------------------------------------------------------------
;; Laplace approximation
;; ---------------------------------------------------------------------------

(defn- log-det-via-cholesky
  "Compute log|A| via Cholesky decomposition. Requires A positive definite.
   log|A| = 2 * sum(log(diag(L))) where A = L*L^T."
  [a]
  (let [L (mx/cholesky a)
        diag-L (mx/diag L)]
    (mx/multiply (mx/scalar 2.0) (mx/sum (mx/log diag-L)))))

(defn laplace-log-evidence
  "Laplace approximation to log marginal evidence:
   log p(y) ≈ log p(y; θ*) + D/2 · log(2π) - 1/2 · log|F(θ*)|

   where θ* is the MAP/MLE estimate and F is the observed Fisher.

   fisher-result: output of observed-fisher (must contain :fisher and :log-ml)
   D: number of parameters

   Returns {:log-evidence scalar, :log-ml scalar, :log-det-fisher scalar}."
  [{:keys [fisher log-ml]} D]
  (let [log-det (log-det-via-cholesky fisher)
        _ (mx/materialize! log-det)
        log-ml-val (mx/item log-ml)
        log-det-val (mx/item log-det)
        log-evidence (+ log-ml-val
                        (* 0.5 D (js/Math.log (* 2 js/Math.PI)))
                        (* -0.5 log-det-val))]
    {:log-evidence log-evidence
     :log-ml log-ml-val
     :log-det-fisher log-det-val}))

;; ---------------------------------------------------------------------------
;; Natural gradient
;; ---------------------------------------------------------------------------

(defn natural-gradient-step
  "Natural gradient step: θ_{t+1} = θ_t - lr · F(θ_t)⁻¹ · ∇_θ(-log p(y; θ_t)).

   grad: [D] gradient from log-ml-gradient (∂(-log-ML)/∂θ, descent direction)
   fisher: [D,D] Fisher information matrix
   params: current [D] parameter array
   lr: learning rate (default 1.0 — natural gradient is already well-scaled)

   Solves F·d = grad instead of inverting F explicitly.
   Returns new params array."
  [fisher grad params {:keys [lr] :or {lr 1.0}}]
  (let [d (mx/solve fisher (mx/reshape grad [(first (mx/shape grad)) 1]))
        d (mx/reshape d [(first (mx/shape grad))])
        new-params (mx/subtract params (mx/multiply (mx/scalar lr) d))]
    (mx/materialize! new-params)
    new-params))

;; ---------------------------------------------------------------------------
;; Confidence intervals (Cramér-Rao bound)
;; ---------------------------------------------------------------------------

(defn parameter-std-errors
  "Standard errors from the Fisher information matrix.
   SE(θ_i) = sqrt(F⁻¹_ii) — the Cramér-Rao lower bound.

   Returns {:std-errors [D] array, :covariance [D,D] array}."
  [fisher]
  (let [cov (mx/inv fisher)
        _ (mx/materialize! cov)
        diag (mx/diag cov)
        se (mx/sqrt (mx/maximum diag (mx/scalar 1e-10)))]
    (mx/materialize! se)
    {:std-errors se :covariance cov}))
