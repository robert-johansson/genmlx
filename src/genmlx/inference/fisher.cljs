(ns genmlx.inference.fisher
  "Fisher information matrix for parametric models.
   Enables model comparison (Laplace/BIC), natural gradient,
   confidence intervals, and posterior approximation.

   Uses the observed Fisher: F(θ) = -∇²_θ log p(y; θ), computed via
   central finite differences on the autodiff gradient (grad-then-FD).
   This is O(D) gradient evaluations instead of O(D²) scalar evaluations.
   Fixed random key ensures deterministic Hessian (same particles for
   all finite difference evaluations)."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.inference.differentiable :as diff]
            [genmlx.compiled-gen :as cg]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- basis-vector
  "Create a one-hot basis vector e_i of dimension D."
  [i D]
  (let [data (vec (concat (repeat i 0.0) [1.0] (repeat (- D i 1) 0.0)))]
    (mx/array data)))

(defn- adaptive-epsilon
  "Compute per-parameter epsilon for finite differences.
   For first-order FD on float32: ε = cbrt(eps_mach) * max(1, |x_i|) ≈ 0.005 * scale."
  [params-array]
  (let [abs-params (mx/abs params-array)
        scale (mx/maximum abs-params (mx/scalar 1.0))]
    (mx/multiply (mx/scalar 0.005) scale)))

(defn- eval-grad
  "Compute ∇(-log-ML) at a parameter point via autodiff. Returns [D] MLX array.
   Uses the same key for deterministic IS estimation."
  [model args observations param-names params-array n-particles key]
  (let [{:keys [grad log-ml]} (diff/log-ml-gradient
                                 {:n-particles n-particles :key key}
                                 model args observations param-names params-array)]
    (mx/materialize! grad log-ml)
    {:grad grad :log-ml log-ml}))

;; ---------------------------------------------------------------------------
;; Observed Fisher via grad-then-finite-diff (O(D) grad evals)
;; ---------------------------------------------------------------------------

(defn observed-fisher
  "Observed Fisher information matrix: F(θ) = -∇²_θ log p(y; θ).

   Computed via central finite differences on the autodiff gradient:
     H[*,i] = (∇f(θ+εᵢeᵢ) - ∇f(θ-εᵢeᵢ)) / (2εᵢ)
   Then Fisher = -H, symmetrized as F = (F + Fᵀ) / 2.

   Cost: 2D gradient evaluations (vs D²+1 scalar evals in pure FD).
   Epsilon is adaptive per parameter: εᵢ = 0.005 * max(1, |θᵢ|).

   opts:
     :n-particles  - IS particles (default 2000)
     :key          - PRNG key (fixed for deterministic Hessian)
     :epsilon      - override adaptive epsilon with fixed value (optional)
     :damping      - diagonal damping λ (default: trace-adaptive 1e-4·tr(F)/D)
     :compiled?    - use mx/compile-fn for ~5x faster gradient evals (default false)

   Returns {:fisher [D,D] MLX array, :log-ml MLX scalar}."
  [{:keys [n-particles key epsilon damping compiled?] :or {n-particles 2000}}
   model args observations param-names params-array]
  (let [key (rng/ensure-key key)
        D (first (mx/shape params-array))
        ;; Adaptive per-parameter epsilon (or fixed override)
        eps-vec (if epsilon
                  (mx/multiply (mx/scalar epsilon) (mx/ones [D]))
                  (adaptive-epsilon params-array))
        _ (mx/materialize! eps-vec)
        ;; Optionally compile gradient for faster repeated evaluation
        compiled-vg (when compiled?
                      (cg/compile-log-ml-gradient
                        {:n-particles n-particles :key key}
                        model args observations param-names))
        ;; Gradient evaluation function (compiled or interpreted)
        grad-fn (if compiled-vg
                  (fn [p]
                    (let [[neg-lml g] (compiled-vg p)]
                      (mx/materialize! neg-lml g)
                      {:grad g :log-ml (mx/negative neg-lml)}))
                  (fn [p]
                    (eval-grad model args observations param-names
                               p n-particles key)))
        ;; Evaluate gradient at center point for log-ML value
        {:keys [log-ml]} (grad-fn params-array)
        ;; Central finite differences on gradient: 2D evaluations
        hessian-cols
        (mapv (fn [i]
                (let [eps-i (mx/item (mx/index eps-vec i))
                      perturbation (mx/multiply (mx/scalar eps-i) (basis-vector i D))
                      {:keys [grad]} (grad-fn (mx/add params-array perturbation))
                      grad-plus grad
                      {:keys [grad]} (grad-fn (mx/subtract params-array perturbation))
                      grad-minus grad
                      ;; H[*,i] = (∇f(θ+ε·eᵢ) - ∇f(θ-ε·eᵢ)) / (2ε)
                      ;; Note: grad is ∂(-log-ML)/∂θ, so H is Hessian of -log-ML
                      ;; Fisher = Hessian of -log-ML (no extra negation needed)
                      col (mx/divide (mx/subtract grad-plus grad-minus)
                                     (mx/scalar (* 2.0 eps-i)))]
                  (mx/materialize! col)
                  col))
              (range D))
        ;; Stack columns into [D,D] matrix, then symmetrize
        fisher-raw (mx/stack hessian-cols 1)
        fisher-sym (mx/multiply (mx/scalar 0.5)
                                (mx/add fisher-raw (mx/transpose fisher-raw)))
        ;; Trace-adaptive damping: λ = max(damping, 1e-4·tr(F)/D)
        _ (mx/materialize! fisher-sym)
        tr-F (mx/item (mx/sum (mx/diag fisher-sym)))
        lambda (if damping
                 damping
                 (max 1e-6 (* 1e-4 (/ (js/Math.abs tr-F) D))))
        fisher (mx/add fisher-sym (mx/multiply (mx/scalar lambda) (mx/eye D)))]
    (mx/materialize! fisher)
    {:fisher fisher
     :log-ml log-ml
     :damping lambda}))

;; ---------------------------------------------------------------------------
;; Laplace approximation
;; ---------------------------------------------------------------------------

(defn- try-cholesky
  "Attempt Cholesky, return L or nil on failure."
  [a]
  (try
    (let [L (mx/cholesky a)]
      (mx/materialize! L)
      L)
    (catch :default _ nil)))

(defn- robust-cholesky
  "Cholesky decomposition with increasing damping fallback.
   Returns {:L cholesky-factor, :added-damping scalar}."
  [a D]
  (reduce (fn [_ tau]
            (let [a-damped (if (zero? tau) a (mx/add a (mx/multiply (mx/scalar tau) (mx/eye D))))
                  L (try-cholesky a-damped)]
              (when L (reduced {:L L :added-damping tau}))))
          nil
          [0.0 1e-6 1e-4 1e-2 1e-1 1.0 10.0]))

(defn- log-det-via-cholesky
  "Compute log|A| via Cholesky decomposition.
   log|A| = 2 * sum(log(diag(L))) where A = L*L^T."
  [L]
  (let [diag-L (mx/diag L)]
    (mx/multiply (mx/scalar 2.0) (mx/sum (mx/log diag-L)))))

(defn laplace-log-evidence
  "Laplace approximation to log marginal evidence:
   log p(y) ≈ log p(y; θ*) + D/2 · log(2π) - 1/2 · log|F(θ*)|

   where θ* is the MAP/MLE estimate and F is the observed Fisher.

   fisher-result: output of observed-fisher (must contain :fisher and :log-ml)
   D: number of parameters

   Returns {:log-evidence scalar, :log-ml scalar, :log-det-fisher scalar}."
  [{:keys [fisher log-ml]} D]
  (let [{:keys [L added-damping]} (robust-cholesky fisher D)
        log-det (log-det-via-cholesky L)
        _ (mx/materialize! log-det)
        log-ml-val (mx/item log-ml)
        log-det-val (mx/item log-det)
        log-evidence (+ log-ml-val
                        (* 0.5 D (js/Math.log (* 2 js/Math.PI)))
                        (* -0.5 log-det-val))]
    (cond-> {:log-evidence log-evidence
             :log-ml log-ml-val
             :log-det-fisher log-det-val}
      (pos? added-damping)
      (assoc :warning (str "Fisher near-singular, added damping " added-damping)))))

;; ---------------------------------------------------------------------------
;; Natural gradient
;; ---------------------------------------------------------------------------

(defn- try-solve
  "Attempt solve, return solution or nil on failure."
  [F b]
  (try
    (let [d (mx/solve F b)]
      (mx/materialize! d)
      d)
    (catch :default _ nil)))

(defn- robust-solve
  "Solve F·d = b with increasing damping fallback.
   Falls back to steepest descent (returns b) if all attempts fail."
  [F b D]
  (or (reduce (fn [_ tau]
                (let [F-damped (if (zero? tau) F (mx/add F (mx/multiply (mx/scalar tau) (mx/eye D))))
                      d (try-solve F-damped b)]
                  (when d (reduced d))))
              nil
              [0.0 1e-4 1e-2 1e-1 1.0])
      b))

(defn natural-gradient-step
  "Natural gradient step: θ_{t+1} = θ_t - lr · F(θ_t)⁻¹ · ∇_θ(-log p(y; θ_t)).

   grad: [D] gradient from log-ml-gradient (∂(-log-ML)/∂θ, descent direction)
   fisher: [D,D] Fisher information matrix
   params: current [D] parameter array
   lr: learning rate (default 1.0 — natural gradient is already well-scaled)

   Solves F·d = grad with robust damping fallback.
   Returns new params array."
  [fisher grad params {:keys [lr] :or {lr 1.0}}]
  (let [D (first (mx/shape grad))
        d (robust-solve fisher (mx/reshape grad [D 1]) D)
        d (mx/reshape d [D])
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
  (let [D (first (mx/shape fisher))
        ;; Use robust solve to get covariance = F⁻¹
        ;; Solve F · C = I  =>  C = F⁻¹
        cov (robust-solve fisher (mx/eye D) D)
        _ (mx/materialize! cov)
        diag-cov (mx/diag cov)
        se (mx/sqrt (mx/maximum diag-cov (mx/scalar 1e-10)))]
    (mx/materialize! se)
    {:std-errors se :covariance cov}))
