(ns genmlx.inference.vi
  "Variational Inference (ADVI with mean-field Gaussian guide)."
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
  (let [d (let [s (mx/shape init-params)] (if (empty? s) 1 (first s)))
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
        (let [[iter-key next-key] (if rk (rng/split rk) [nil nil])
              g (mx/tidy (fn [] (grad-neg-elbo vp)))
              _ (mx/eval! g)
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
