(ns genmlx.inference.vi
  "Variational Inference: ADVI with mean-field Gaussian guide,
   plus programmable VI objectives (ELBO, IWELBO, PWake, QWake)
   and gradient estimators (reparameterization, REINFORCE)."
  (:require [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.inference.util :as u]))

;; ---------------------------------------------------------------------------
;; Adam optimizer
;; ---------------------------------------------------------------------------

(defn- adam-state [params]
  {:m (mx/zeros (mx/shape params))
   :v (mx/zeros (mx/shape params))
   :t 0})

(defn- adam-step [params grad-arr state lr beta1 beta2 epsilon]
  (let [{:keys [m v t]} state
        t' (inc t)
        m' (mx/add (mx/multiply (mx/scalar beta1) m)
                   (mx/multiply (mx/scalar (- 1.0 beta1)) grad-arr))
        v' (mx/add (mx/multiply (mx/scalar beta2) v)
                   (mx/multiply (mx/scalar (- 1.0 beta2)) (mx/square grad-arr)))
        m-hat (mx/divide m' (mx/scalar (- 1.0 (js/Math.pow beta1 t'))))
        v-hat (mx/divide v' (mx/scalar (- 1.0 (js/Math.pow beta2 t'))))
        update (mx/divide (mx/multiply (mx/scalar lr) m-hat)
                          (mx/add (mx/sqrt v-hat) (mx/scalar epsilon)))
        params' (mx/subtract params update)]
    (mx/eval! params' m' v')
    [params' {:m m' :v v' :t t'}]))

;; ---------------------------------------------------------------------------
;; ELBO estimation
;; ---------------------------------------------------------------------------

(defn- elbo-estimate
  "Estimate ELBO via Monte Carlo with reparameterized samples.
   ELBO = E_q[log p(x) - log q(x)]
   Optional `key` uses functional PRNG for sampling."
  [variational-params log-density n-samples d vmapped-log-density key]
  (let [mu (mx/slice variational-params 0 d)
        log-sigma (mx/slice variational-params d (* 2 d))
        sigma (mx/exp log-sigma)
        ;; Draw samples from q via reparameterization
        eps (if key
              (rng/normal key [n-samples d])
              (mx/random-normal [n-samples d]))
        samples (mx/add mu (mx/multiply sigma eps))
        ;; log q(z) for each sample
        log-2pi-scalar (mx/scalar (js/Math.log (* 2 js/Math.PI)))
        diff-norm (mx/divide (mx/subtract samples mu) sigma)
        log-q-per-dim (mx/multiply (mx/scalar -0.5)
                                    (mx/add log-2pi-scalar
                                            (mx/multiply (mx/scalar 2.0) log-sigma)
                                            (mx/square diff-norm)))
        log-q (mx/sum log-q-per-dim [1])
        ;; log p for each sample via vmap
        log-p-vals (vmapped-log-density samples)
        ;; ELBO = mean(log_p - log_q)
        elbo (mx/mean (mx/subtract log-p-vals log-q))]
    elbo))

;; ---------------------------------------------------------------------------
;; VI main
;; ---------------------------------------------------------------------------

(defn vi
  "Variational Inference via ADVI.
   Uses a mean-field Gaussian guide and optimizes ELBO via Adam.

   opts: {:iterations N :learning-rate lr :elbo-samples N
          :beta1 b1 :beta2 b2 :epsilon eps :callback fn :key prng-key}

   log-density: (fn [params]) -> MLX scalar
   init-params: MLX array of initial parameter values

   Returns {:mu MLX-array :sigma MLX-array :elbo-history [numbers]
            :sample-fn (fn [n] -> samples)}"
  [{:keys [iterations learning-rate elbo-samples beta1 beta2 epsilon callback key]
    :or {iterations 1000 learning-rate 0.01 elbo-samples 10
         beta1 0.9 beta2 0.999 epsilon 1e-8}}
   log-density init-params]
  (let [d (or (first (mx/shape init-params)) 1)
        init-mu (if (zero? (mx/ndim init-params))
                  (mx/reshape init-params [1])
                  init-params)
        init-log-sigma (mx/zeros [d])
        init-vp (mx/tidy
                  (fn []
                    (let [vp (mx/concatenate [init-mu init-log-sigma])]
                      (mx/eval! vp)
                      vp)))
        vmapped-log-density (mx/vmap log-density)
        neg-elbo-fn (fn [vp]
                      (mx/negative (elbo-estimate vp log-density elbo-samples d vmapped-log-density nil)))
        grad-neg-elbo (mx/grad neg-elbo-fn)]
    (loop [i 0, vp init-vp
           opt-state (adam-state init-vp)
           elbo-history (transient [])
           rk key]
      (if (>= i iterations)
        (let [final-mu (mx/slice vp 0 d)
              final-log-sigma (mx/slice vp d (* 2 d))
              final-sigma (mx/exp final-log-sigma)]
          (mx/eval! final-mu final-sigma)
          {:mu final-mu
           :sigma final-sigma
           :elbo-history (persistent! elbo-history)
           :sample-fn (fn [n]
                        (let [sample-key (if rk rk (rng/fresh-key))
                              eps (rng/normal sample-key [n d])
                              samples (mx/add final-mu (mx/multiply final-sigma eps))]
                          (mx/eval! samples)
                          (if (= d 1)
                            (mapv #(mx/item (mx/index samples %)) (range n))
                            (mx/->clj samples))))})
        (let [[iter-key next-key] (rng/split-or-nils rk)
              g (doto (mx/tidy (fn [] (grad-neg-elbo vp))) mx/eval!)
              [vp' opt-state'] (adam-step vp g opt-state
                                          learning-rate beta1 beta2 epsilon)
              elbo-val (when (zero? (mod i (max 1 (quot iterations 100))))
                         (let [e (mx/tidy (fn [] (elbo-estimate vp' log-density elbo-samples d vmapped-log-density iter-key)))]
                           (mx/eval! e)
                           (mx/item e)))]
          (when (and callback elbo-val)
            (callback {:iter i :elbo elbo-val :params (mx/->clj vp')}))
          (recur (inc i) vp' opt-state'
                 (if elbo-val
                   (conj! elbo-history elbo-val)
                   elbo-history)
                 next-key))))))

;; ---------------------------------------------------------------------------
;; VI from model (convenience)
;; ---------------------------------------------------------------------------

(defn vi-from-model
  "Run VI on a generative model with observations.

   opts: same as vi, plus :addresses [addr...]
   model: generative function
   args: model arguments
   observations: choice map of observed values

   Returns same as vi."
  [opts model args observations addresses]
  (let [score-fn (u/make-score-fn model args observations addresses)
        {:keys [trace]} (p/generate model args observations)
        init-q (u/extract-params trace addresses)]
    (vi opts score-fn init-q)))

;; ---------------------------------------------------------------------------
;; Programmable VI Objectives
;; ---------------------------------------------------------------------------

(defn elbo-objective
  "Standard ELBO objective: E_q[log p(x,z) - log q(z)].
   log-p-fn: (fn [z] -> MLX scalar) — model log-density
   log-q-fn: (fn [z] -> MLX scalar) — guide log-density
   Returns (fn [samples] -> MLX scalar) where samples is [K d]."
  [log-p-fn log-q-fn]
  (fn [samples]
    (let [vmapped-p (mx/vmap log-p-fn)
          vmapped-q (mx/vmap log-q-fn)
          log-p (vmapped-p samples)
          log-q (vmapped-q samples)]
      (mx/mean (mx/subtract log-p log-q)))))

(defn iwelbo-objective
  "Importance-weighted ELBO (IWELBO/IWAE): tighter bound using K samples.
   log E[1/K * sum_k w_k] where w_k = p(x,z_k)/q(z_k)
   Approaches log p(x) as K -> infinity.
   Returns (fn [samples] -> MLX scalar)."
  [log-p-fn log-q-fn]
  (fn [samples]
    (let [vmapped-p (mx/vmap log-p-fn)
          vmapped-q (mx/vmap log-q-fn)
          log-p (vmapped-p samples)
          log-q (vmapped-q samples)
          log-w (mx/subtract log-p log-q)
          k (first (mx/shape samples))]
      ;; log(1/K * sum exp(log_w)) = logsumexp(log_w) - log(K)
      (mx/subtract (mx/logsumexp log-w) (mx/scalar (js/Math.log k))))))

(defn pwake-objective
  "P-Wake objective: trains the model to match the guide's proposals.
   Equivalent to minimizing KL(q || p).
   Returns (fn [samples] -> MLX scalar)."
  [log-p-fn log-q-fn]
  (fn [samples]
    (let [vmapped-p (mx/vmap log-p-fn)]
      ;; Maximize E_q[log p(z)] — the log-q term is constant w.r.t. model params
      (mx/mean (vmapped-p samples)))))

(defn qwake-objective
  "Q-Wake objective: trains the guide to approximate the posterior.
   Uses self-normalized importance weights.
   Returns (fn [samples] -> MLX scalar)."
  [log-p-fn log-q-fn]
  (fn [samples]
    (let [vmapped-p (mx/vmap log-p-fn)
          vmapped-q (mx/vmap log-q-fn)
          log-p (vmapped-p samples)
          log-q (vmapped-q samples)
          ;; Self-normalized importance weights
          log-w (mx/subtract log-p log-q)
          log-w-norm (mx/subtract log-w (mx/logsumexp log-w))
          w-norm (mx/exp log-w-norm)]
      ;; Maximize E_p[log q(z)] ≈ sum_k w_k * log q(z_k)
      (mx/sum (mx/multiply (mx/stop-gradient w-norm) log-q)))))

(defn reinforce-estimator
  "REINFORCE (score function) gradient estimator.
   For non-reparameterizable distributions.
   objective-fn: (fn [samples] -> MLX scalar)
   log-q-fn: (fn [z] -> MLX scalar)
   Returns (fn [samples] -> MLX scalar) with REINFORCE gradient."
  [objective-fn log-q-fn]
  (fn [samples]
    (let [vmapped-q (mx/vmap log-q-fn)
          log-q (vmapped-q samples)
          obj-val (objective-fn samples)
          ;; REINFORCE: (f(z) - baseline) * grad log q(z)
          ;; We use mean as baseline for variance reduction
          baseline (mx/stop-gradient (mx/mean log-q))]
      ;; Return surrogate loss whose gradient equals REINFORCE estimator
      (mx/add obj-val
              (mx/mean (mx/multiply (mx/stop-gradient
                                      (mx/subtract log-q baseline))
                                    log-q))))))

(defn programmable-vi
  "Programmable variational inference with pluggable objectives and estimators.

   opts:
     :iterations       - number of optimization steps
     :learning-rate    - Adam learning rate
     :n-samples        - MC samples per gradient estimate
     :objective        - :elbo (default), :iwelbo, :pwake, :qwake, or custom fn
     :estimator        - :reparam (default) or :reinforce
     :callback         - fn called each step
     :key              - PRNG key

   log-p-fn: (fn [z] -> MLX scalar) — model log-density
   log-q-fn: (fn [z params] -> MLX scalar) — parameterized guide log-density
   sample-fn: (fn [params key n] -> [n d] MLX array) — guide sampler
   init-params: initial variational parameters (MLX array)

   Returns {:params :loss-history}"
  [{:keys [iterations learning-rate n-samples objective estimator callback key]
    :or {iterations 1000 learning-rate 0.01 n-samples 10
         objective :elbo estimator :reparam}}
   log-p-fn log-q-fn sample-fn init-params]
  (let [;; Build loss function (parameterized)
        loss-fn (fn [params iter-key]
                  (let [samples (sample-fn params iter-key n-samples)
                        ;; Rebuild objective with current params
                        log-q-curr (fn [z] (log-q-fn z params))
                        obj-fn (case objective
                                 :elbo (elbo-objective log-p-fn log-q-curr)
                                 :iwelbo (iwelbo-objective log-p-fn log-q-curr)
                                 :pwake (pwake-objective log-p-fn log-q-curr)
                                 :qwake (qwake-objective log-p-fn log-q-curr)
                                 (fn [s] (objective log-p-fn log-q-curr s)))
                        obj-val (if (= estimator :reinforce)
                                  ((reinforce-estimator obj-fn log-q-curr) samples)
                                  (obj-fn samples))]
                    ;; We minimize negative objective
                    (mx/negative obj-val)))
        grad-loss (fn [params iter-key]
                    (let [g (mx/grad (fn [p] (loss-fn p iter-key)))]
                      {:loss (loss-fn params iter-key) :grad (g params)}))]
    ;; Optimization loop
    (loop [i 0 params init-params
           opt-state (adam-state init-params)
           losses (transient [])
           rk key]
      (if (>= i iterations)
        {:params params :loss-history (persistent! losses)}
        (let [[iter-key next-key] (rng/split-or-nils rk)
              {:keys [loss grad]} (grad-loss params iter-key)
              _ (mx/eval! loss grad)
              loss-val (mx/item loss)
              [params' opt-state'] (adam-step params grad opt-state
                                              learning-rate 0.9 0.999 1e-8)]
          (when callback
            (callback {:iter i :loss loss-val}))
          (recur (inc i) params' opt-state'
                 (conj! losses loss-val) next-key))))))
