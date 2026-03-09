(ns genmlx.inference.ekf
  "Extended Kalman filter middleware for nonlinear state-space models.

   Generalizes the Kalman middleware to nonlinear dynamics and observations
   by linearizing via automatic differentiation (mx/grad). Same handler
   middleware pattern: the cognitive architecture is a gen function, the EKF
   determines how latent states are handled.

   Two levels of API (matching Kalman):

   1. Pure building blocks — ekf-linearize, ekf-predict, ekf-update.
      Use directly for explicit control. ekf-update delegates to
      kalman-update after linearization — full code reuse.

   2. Handler middleware — make-ekf-dispatch + ekf-generate + ekf-fold.
      The cognitive architecture uses ekf-latent and ekf-obs trace sites.
      Same gen function works under standard handlers (sampling)
      or EKF handler (analytical marginalization).

   Limitation: 1D latent states only (scalar per element). The Jacobian
   is a scalar derivative computed via mx/grad. Multi-dimensional EKF
   would require matrix Jacobians — deferred until needed."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.choicemap :as cm]
            [genmlx.handler :as h]
            [genmlx.runtime :as rt]
            [genmlx.inference.kalman :as kal]
            [genmlx.inference.analytical :as ana]
            [genmlx.mlx.random :as rng])
  (:require-macros [genmlx.dist.macros :refer [defdist]]))

;; ---------------------------------------------------------------------------
;; Distributions that carry nonlinear structure
;; ---------------------------------------------------------------------------

(defdist ekf-latent
  "Nonlinear latent dynamics: z_t ~ N(f(z_{t-1}), process-noise).
   transition-fn: (fn [z] -> z') — differentiable, element-wise.
   prev-value:    previous latent state value.
   process-noise: std dev of process noise.

   Under standard handler: samples from N(f(prev), noise).
   Under EKF handler: linearizes f at belief mean, analytical predict."
  [transition-fn prev-value process-noise]
  (sample [key]
    (dc/dist-sample
      (dist/gaussian (transition-fn prev-value) process-noise)
      key))
  (log-prob [v]
    (dc/dist-log-prob
      (dist/gaussian (transition-fn prev-value) process-noise)
      v)))

(defdist ekf-obs
  "Observation with nonlinear structure: x ~ N(h(z), noise-std).
   obs-fn:       (fn [z] -> predicted-obs) — differentiable, element-wise.
   latent-value: current latent state (used under standard handler).
   noise-std:    observation noise std dev.
   mask:         1=observed, 0=missing.

   Under standard handler: masked Gaussian log-prob.
   Under EKF handler: linearizes h at belief mean, delegates to kalman-update."
  [obs-fn latent-value noise-std mask]
  (sample [key]
    (dc/dist-sample
      (dist/gaussian (obs-fn latent-value) noise-std)
      key))
  (log-prob [v]
    (mx/multiply mask
      (dc/dist-log-prob
        (dist/gaussian (obs-fn latent-value) noise-std)
        v))))

;; ---------------------------------------------------------------------------
;; Pure EKF operations (Level 1)
;; ---------------------------------------------------------------------------

(defn ekf-linearize
  "Linearize f at z0 via automatic differentiation.
   f:  differentiable function (fn [array] -> array), element-wise.
   z0: linearization point, [P]-shaped or scalar.

   Returns [f(z0), A] where A is the element-wise Jacobian df/dz.

   Uses the identity: for element-wise f, grad(sum ∘ f) gives
   per-element derivatives d(sum f_i)/dz_j = df_j/dz_j."
  [f z0]
  [(f z0) ((mx/grad (fn [z] (mx/sum (f z)))) z0)])

(defn ekf-predict
  "EKF predict step: linearize f at belief mean, propagate uncertainty.
   belief:        {:mean [P], :var [P]}
   transition-fn: nonlinear dynamics (fn [z] -> z'), differentiable
   process-noise: std dev of process noise

   Returns updated belief {:mean :var}.
   Mean is f(z0) (nonlinear). Variance uses Jacobian: A²·var + Q²."
  [belief transition-fn process-noise]
  (let [{:keys [mean var]} belief
        [f-mean A] (ekf-linearize transition-fn mean)]
    {:mean f-mean
     :var  (mx/add (mx/multiply A (mx/multiply A var))
                   (mx/multiply process-noise process-noise))}))

(defn ekf-update
  "EKF update step: linearize h at belief mean, then Kalman update.
   Delegates to kalman-update after computing linearized parameters.

   belief:    {:mean [P], :var [P]} — predicted belief
   obs:       observed value
   obs-fn:    nonlinear observation function (fn [z] -> obs-prediction)
   noise-std: observation noise std dev
   mask:      observation mask (1=observed, 0=missing)

   Returns {:belief updated, :ll per-element marginal LL}."
  [belief obs obs-fn noise-std mask]
  (let [z0 (:mean belief)
        [h-z0 H] (ekf-linearize obs-fn z0)
        ;; Linear approximation: h(z) ≈ H·z + (h(z0) - H·z0)
        base-mean (mx/subtract h-z0 (mx/multiply H z0))]
    (kal/kalman-update belief obs base-mean H noise-std mask)))

;; ---------------------------------------------------------------------------
;; Handler middleware (Level 2)
;; ---------------------------------------------------------------------------
;;
;; Same design as Kalman: intercepts ekf-latent and ekf-obs trace sites.
;; LL accumulates in :ekf-ll, not :score/:weight.
;; Initial belief = {mean: zeros, var: zeros}. Handler always predicts —
;; at t=0, predict({0,0}, f, q) gives {f(0), q²}.

(defn make-ekf-dispatch
  "Create EKF dispatch map for use with wrap-analytical.

   latent-addr: keyword address of the latent state site

   Returns dispatch map: {:ekf-latent handler, :ekf-obs handler}."
  [latent-addr]
  {:ekf-latent
   (fn [state addr dist]
     (if (= addr latent-addr)
       (let [{:keys [transition-fn process-noise]} (:params dist)
             belief (or (:ekf-belief state)
                        {:mean (mx/zeros [(:ekf-n state)])
                         :var  (mx/zeros [(:ekf-n state)])})
             new-belief (ekf-predict belief transition-fn process-noise)]
         [(:mean new-belief)
          (-> state
              (assoc :ekf-belief new-belief)
              (update :choices cm/set-value addr (:mean new-belief)))])
       nil))

   :ekf-obs
   (fn [state addr dist]
     (let [{:keys [obs-fn noise-std mask]} (:params dist)
           belief (:ekf-belief state)
           constraint (cm/get-submap (:constraints state) addr)
           obs (cm/get-value constraint)
           {:keys [belief ll]} (ekf-update belief obs obs-fn noise-std mask)
           n (:ekf-n state)]
       [obs (-> state
                (assoc :ekf-belief belief)
                (update :choices cm/set-value addr obs)
                (update :ekf-ll
                  #(mx/add (or % (mx/zeros [n])) ll)))]))})

(defn make-ekf-transition
  "Handler middleware: wraps generate-transition for EKF.

   latent-addr: keyword address of the latent state

   Returns a transition function composable via wrap-analytical."
  [latent-addr]
  (ana/wrap-analytical h/generate-transition (make-ekf-dispatch latent-addr)))

(defn ekf-generate
  "Run a gen function body under the EKF handler.

   gf:          DynamicGF with ekf-latent/ekf-obs trace sites
   args:        gen function arguments
   constraints: choicemap with observation constraints
   latent-addr: keyword address of the latent state
   n:           number of elements (e.g. patients)
   key:         PRNG key

   opts (map):
   - :param-store  parameter store for param sites
   - :init-belief  initial belief {:mean :var} (default: {zeros, zeros})

   Returns handler result with :ekf-belief and :ekf-ll."
  [gf args constraints latent-addr n key & [opts]]
  (let [{:keys [param-store init-belief]} opts
        transition (make-ekf-transition latent-addr)
        init-state (cond-> {:choices cm/EMPTY
                            :score (mx/scalar 0.0)
                            :weight (mx/scalar 0.0)
                            :key key
                            :constraints constraints
                            :ekf-n n
                            :ekf-belief (or init-belief
                                            {:mean (mx/zeros [n])
                                             :var  (mx/zeros [n])})}
                     param-store (assoc :param-store param-store))]
    (rt/run-handler transition init-state
      (fn [rt] (apply (:body-fn gf) rt args)))))

(defn ekf-fold
  "Fold a per-step gen function over T timesteps under the EKF handler.

   step-fn:     gen function with ekf-latent and ekf-obs trace sites
   latent-addr: keyword address of the latent state
   n:           number of elements (patients)
   T:           number of timesteps
   context-fn:  (fn [t] -> {:args [step-fn-args], :constraints choicemap})

   Uses {mean: zeros, var: zeros} initial belief. Handler always predicts —
   at t=0 this gives the N(f(0), q²) prior.

   Returns {:ll [P]-shaped total marginal LL, :belief final belief}."
  [step-fn latent-addr n T context-fn]
  (loop [t 0
         belief {:mean (mx/zeros [n]) :var (mx/zeros [n])}
         acc-ll (mx/zeros [n])]
    (if (>= t T)
      {:ll acc-ll :belief belief}
      (let [{:keys [args constraints]} (context-fn t)
            result (ekf-generate
                     step-fn args constraints latent-addr n
                     (rng/fresh-key t)
                     {:init-belief belief})
            step-ll (or (:ekf-ll result) (mx/zeros [n]))]
        (recur (inc t)
               (:ekf-belief result)
               (mx/add acc-ll step-ll))))))
