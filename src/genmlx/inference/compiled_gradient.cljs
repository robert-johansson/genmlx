(ns genmlx.inference.compiled-gradient
  "WP-4: Gradient through compiled inference chains and SMC sweeps.

   Enables computing gradients of inference outputs (final score, log-ML,
   posterior expectations) w.r.t. model parameters via mx/grad.

   Key insight: compiled chains and SMC sweeps are deterministic functions
   of pre-generated noise + model parameters. mx/grad differentiates through
   the entire computation graph.

   Three entry points:
   - mcmc-score-gradient:  d(final-score)/d(model-params) through MH chain
   - smc-log-ml-gradient:  d(log-ML)/d(model-params) through compiled SMC
   - make-differentiable-chain: build a differentiable MH chain function"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.inference.util :as u]
            [genmlx.inference.compiled-smc :as csmc]
            [genmlx.inference.differentiable-resample :as dr]
            [genmlx.compiled :as compiled]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]))

;; =========================================================================
;; Differentiable MH chain
;; =========================================================================

(defn make-differentiable-chain
  "Build a differentiable T-step MH chain as a pure function.

   score-fn: (fn [params-tensor] -> scalar) — must be differentiable
   proposal-std: scalar or [K] array — random walk step size
   T: number of MH steps
   K: number of parameters

   Returns (fn [init-params noise uniforms] -> final-params)
   where noise is [T,K] standard normal and uniforms is [T] uniform.

   The entire chain is differentiable through mx/grad because:
   - mx/where has straight-through gradient on the selected branch
   - All score computations flow through differentiable ops"
  [score-fn proposal-std T K]
  (fn [init-params noise-2d uniforms-1d]
    (loop [p init-params, t 0]
      (if (>= t T)
        p
        (let [;; Extract noise for this step
              row (mx/reshape (mx/take-idx noise-2d (mx/array [t] mx/int32) 0) [K])
              ;; Propose
              proposal (mx/add p (mx/multiply proposal-std row))
              ;; Score current and proposed
              s-cur (score-fn p)
              s-prop (score-fn proposal)
              ;; Accept/reject via mx/where (differentiable)
              log-alpha (mx/subtract s-prop s-cur)
              log-u (mx/log (mx/index uniforms-1d t))
              accept? (mx/greater log-alpha log-u)]
          (recur (mx/where accept? proposal p) (inc t)))))))

(defn mcmc-score-gradient
  "Compute gradient of final MCMC score w.r.t. model parameters.

   Uses a parameterized score function where model-params affect the
   distribution arguments (e.g., prior std, likelihood noise).

   make-score: (fn [model-params] -> (fn [latent-tensor] -> scalar))
     Factory that builds a score function parameterized by model-params.
   init-latent: [K] initial latent values
   model-params: [P] current model parameter values
   opts:
     :steps — number of MH steps (default 10)
     :proposal-std — step size (default 0.1)
     :key — PRNG key

   Returns {:value scalar :grad [P]-array}."
  [make-score init-latent model-params
   {:keys [steps proposal-std key]
    :or {steps 10 proposal-std 0.1}}]
  (let [K (first (mx/shape init-latent))
        rk (rng/ensure-key key)
        [nk uk] (rng/split rk)
        noise (rng/normal nk [steps K])
        uniforms (rng/uniform uk [steps])
        _ (mx/materialize! noise uniforms)
        std-arr (if (number? proposal-std)
                  (mx/scalar proposal-std)
                  proposal-std)
        ;; Objective: run chain with parameterized score, return final score
        objective
        (fn [params]
          (let [score-fn (make-score params)
                chain-fn (make-differentiable-chain score-fn std-arr steps K)
                final-latent (chain-fn init-latent noise uniforms)]
            (score-fn final-latent)))
        vag (mx/value-and-grad objective)
        [value grad] (vag model-params)]
    (mx/materialize! value grad)
    {:value value :grad grad}))

;; =========================================================================
;; Differentiable SMC
;; =========================================================================

(defn smc-log-ml-gradient
  "Compute gradient of log-ML w.r.t. model parameters through compiled SMC.

   Requires gumbel-softmax resampling (differentiable).

   make-extend: (fn [model-params] -> extend-fn)
     Factory that builds a parameterized extend step.
   model-params: [P] current model parameter values
   observations-seq: seq of observation ChoiceMaps
   opts:
     :particles — number of particles N (default 50)
     :tau — softmax temperature (default 1.0)
     :key — PRNG key

   Returns {:value scalar :grad [P]-array}.

   NOTE: This is the full gradient-through-SMC path. It computes
   d(log-ML)/d(model-params) where log-ML = sum_t log(mean(exp(log-w_t))).
   The gradient flows through:
   1. Noise transforms (proposal distributions)
   2. Log-prob computations (scoring)
   3. Gumbel-softmax resampling (particle selection)
   4. LogSumExp (marginal likelihood)"
  [kernel init-state model-params observations-seq
   {:keys [particles tau key]
    :or {particles 50 tau 1.0}}]
  (let [obs-vec (vec observations-seq)
        T (count obs-vec)
        schema (:schema kernel)
        source (:source kernel)
        static-sites (filterv :static? (:trace-sites schema))
        all-addrs (mapv :addr static-sites)
        K (count all-addrs)
        rk (rng/ensure-key key)
        [nk gk] (rng/split rk)
        ;; Pre-generate all noise (fixed across gradient evaluations)
        extend-noise (rng/normal nk [T particles K])
        gumbel-noise (dr/generate-gumbel-noise gk T particles)
        _ (mx/materialize! extend-noise gumbel-noise)
        tau-arr (mx/scalar tau)
        extend-fn (compiled/make-smc-extend-step schema source)
        ;; Objective: log-ML as function of model-params
        ;; For now, model-params modulate the init-state
        ;; (full parameterization requires make-parameterized-extend)
        objective
        (fn [params]
          ;; params affects init-state (simple case: params IS init-state)
          (let [init-n (mx/broadcast-to params [particles])
                result
                (loop [t 0
                       current-particles nil
                       current-state init-n
                       log-ml (mx/scalar 0.0)]
                  (if (>= t T)
                    log-ml
                    (let [obs-t (nth obs-vec t)
                          noise-t (mx/index extend-noise t)
                          kernel-args [(mx/ensure-array t) current-state]
                          {:keys [obs-log-prob values-map retval]}
                          (extend-fn noise-t kernel-args obs-t)
                          new-particles (mx/stack (mapv #(get values-map %) all-addrs) 1)
                          new-state (or retval (get values-map (first all-addrs)))
                          ;; Log-ML increment
                          ml-inc (mx/subtract (mx/logsumexp obs-log-prob)
                                              (mx/scalar (js/Math.log particles)))
                          ;; Gumbel-softmax resampling (differentiable)
                          gumbel-t (mx/index gumbel-noise t)
                          {:keys [particles]} (dr/gumbel-softmax
                                                new-particles obs-log-prob
                                                gumbel-t tau-arr)
                          ;; Soft-resample state
                          resampled-state (dr/gumbel-softmax-1d
                                            new-state obs-log-prob
                                            gumbel-t tau-arr)]
                      (recur (inc t) particles resampled-state
                             (mx/add log-ml ml-inc)))))]
            result))
        vag (mx/value-and-grad objective)
        [value grad] (vag model-params)]
    (mx/materialize! value grad)
    {:value value :grad grad}))

;; =========================================================================
;; Simple gradient helpers
;; =========================================================================

(defn score-gradient-through-chain
  "Simple interface: gradient of final score after running MH chain.

   model: generative function with schema
   args: model arguments
   observations: ChoiceMap of observed values
   addresses: vector of latent addresses
   opts: {:steps :proposal-std :key}

   Returns {:value scalar :grad [K]-array}."
  [model args observations addresses
   {:keys [steps proposal-std key]
    :or {steps 10 proposal-std 0.1}}]
  (let [{:keys [score-fn latent-index tensor-native?]}
        (u/make-tensor-score-fn model args observations addresses)
        K (count latent-index)
        ;; Get initial params from a generate call
        model-k (dyn/auto-key model)
        {:keys [trace]} (p/generate model-k args observations)
        init-params (u/extract-params-by-index trace latent-index)
        _ (mx/materialize! init-params)
        rk (rng/ensure-key key)
        [nk uk] (rng/split rk)
        noise (rng/normal nk [steps K])
        uniforms (rng/uniform uk [steps])
        _ (mx/materialize! noise uniforms)
        std-arr (mx/scalar proposal-std)
        ;; Build differentiable chain
        chain-fn (make-differentiable-chain score-fn std-arr steps K)
        ;; Objective: final score
        objective (fn [p0]
                    (let [final (chain-fn p0 noise uniforms)]
                      (score-fn final)))
        vag (mx/value-and-grad objective)
        [value grad] (vag init-params)]
    (mx/materialize! value grad)
    {:value value :grad grad :init-params init-params}))
