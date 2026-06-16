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
            [genmlx.inference.differentiable-resample :as dr]
            [genmlx.compiled-ops :as cops]))

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
              row (mx/index noise-2d t)
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
        extend-fn (or (cops/make-smc-extend-step schema source)
                      (throw (ex-info "Kernel cannot be compiled for gradient SMC"
                                      {:kernel kernel})))
        ;; Objective: log-ML as function of model-params
        ;; For now, model-params modulate the init-state
        ;; (full parameterization requires make-parameterized-extend)
        objective
        (fn [params]
          ;; params affects init-state (simple case: params IS init-state)
          (let [init-n (mx/broadcast-to params [particles])]
            (loop [t 0
                   current-state init-n
                   log-ml (mx/scalar 0.0)]
              (if (>= t T)
                log-ml
                (let [obs-t (nth obs-vec t)
                      noise-t (mx/index extend-noise t)
                      kernel-args [(mx/ensure-array t) current-state]
                      {:keys [obs-log-prob values-map retval]}
                      (extend-fn noise-t kernel-args obs-t)
                      new-state (or retval (get values-map (first all-addrs)))
                      ;; Log-ML increment
                      ml-inc (mx/subtract (mx/logsumexp obs-log-prob)
                                          (mx/scalar (js/Math.log particles)))
                      ;; Soft-resample state (differentiable)
                      gumbel-t (mx/index gumbel-noise t)
                      resampled-state (dr/gumbel-softmax-1d
                                        new-state obs-log-prob
                                        gumbel-t tau-arr)]
                  (recur (inc t) resampled-state
                         (mx/add log-ml ml-inc)))))))
        ;; genmlx-wys4: the objective broadcasts the init-state param across
        ;; particles (params IS init-state), which is only defined for a scalar /
        ;; length-1 param. A general [P]-vector param has no init-state semantics
        ;; on this path and would otherwise surface as a confusing broadcast_to
        ;; shape error. Fail fast with an explicit contract error; full multi-param
        ;; compiled SMC log-ML gradient requires make-parameterized-extend.
        _ (let [pshape (mx/shape (mx/ensure-array model-params))
                psize (reduce * 1 pshape)]
            (when (> psize 1)
              (throw (ex-info (str "compiled-smc-log-ml-gradient supports a scalar / "
                                   "length-1 init-state param only; got param shape "
                                   (pr-str pshape) " (size " psize "). Multi-param "
                                   "parameterization requires make-parameterized-extend "
                                   "(genmlx-wys4).")
                              {:param-shape pshape :param-size psize
                               :supported "scalar or length-1"}))))
        vag (mx/value-and-grad objective)
        [value grad] (vag model-params)]
    (mx/materialize! value grad)
    {:value value :grad grad}))

