(ns genmlx.inference.kalman
  "Kalman filter middleware for linear-Gaussian state-space models.

   The Kalman filter is handler middleware in the Ring/GenMLX sense:
   the cognitive architecture IS the gen function with latent trace sites,
   the Kalman filter determines how those latent states are handled.

   Two levels of API:

   1. Pure building blocks — kalman-predict, kalman-update, kalman-sequential-update.
      Use these directly in gen function bodies for explicit control.

   2. Handler middleware — make-kalman-transition + kalman-generate + kalman-fold.
      The cognitive architecture is a gen function using kalman-latent and
      kalman-obs trace sites. kalman-fold runs it over T timesteps under
      the Kalman handler, analytically marginalizing latent states.
      Same gen function works under standard handlers (sampling instead
      of marginalizing).

   Both levels produce identical results."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.choicemap :as cm]
            [genmlx.handler :as h]
            [genmlx.runtime :as rt]
            [genmlx.inference.analytical :as ana]
            [genmlx.mlx.random :as rng])
  (:require-macros [genmlx.dist.macros :refer [defdist]]))

;; ---------------------------------------------------------------------------
;; Distributions that carry linear structure
;; ---------------------------------------------------------------------------

(defdist kalman-latent
  "Latent state transition: z_t ~ N(coeff * prev, noise).
   Under standard handler: samples from Gaussian.
   Under Kalman handler: provides structure for predict step."
  [transition-coeff prev-value process-noise]
  (sample [key]
    (dc/dist-sample
      (dist/gaussian (mx/multiply transition-coeff prev-value) process-noise)
      key))
  (log-prob [v]
    (dc/dist-log-prob
      (dist/gaussian (mx/multiply transition-coeff prev-value) process-noise)
      v)))

(defdist kalman-obs
  "Observation with linear-Gaussian structure:
   x ~ N(base-mean + loading * latent-value, noise-std).
   mask: 1=observed, 0=missing. Masked observations contribute 0 to LL.

   Under standard handler: masked Gaussian log-prob.
   Under Kalman handler: provides loading + noise + mask for update step."
  [base-mean loading latent-value noise-std mask]
  (sample [key]
    (dc/dist-sample
      (dist/gaussian (mx/add base-mean (mx/multiply loading latent-value)) noise-std)
      key))
  (log-prob [v]
    (mx/multiply mask
      (dc/dist-log-prob
        (dist/gaussian (mx/add base-mean (mx/multiply loading latent-value)) noise-std)
        v))))

;; ---------------------------------------------------------------------------
;; Pure Kalman operations (Level 1)
;; ---------------------------------------------------------------------------

(def ^:private LOG-2PI 1.8378770664093453)

(defn kalman-init
  "Initial belief state: N(0, 1) prior.
   n: number of elements (e.g. number of patients for [P]-shaped arrays)."
  [n]
  {:mean (mx/zeros [n]) :var (mx/ones [n])})

(defn kalman-predict
  "Kalman predict step: z_{t|t-1} from z_{t-1|t-1}.
   belief:           {:mean array, :var array}
   transition-coeff: scalar A
   process-noise:    scalar Q (std dev, not variance)
   Returns updated belief {:mean :var}."
  [belief transition-coeff process-noise]
  (let [{:keys [mean var]} belief]
    {:mean (mx/multiply transition-coeff mean)
     :var  (mx/add (mx/multiply transition-coeff
                     (mx/multiply transition-coeff var))
                   (mx/multiply process-noise process-noise))}))

(defn kalman-update
  "Kalman update step: incorporate one observation.
   belief:    {:mean array, :var array} — current latent belief
   obs:       observed value (array, same shape as belief)
   base-mean: AR/deterministic prediction (array)
   loading:   factor loading (scalar or array)
   noise-std: observation noise std dev (scalar or array)
   mask:      observation mask (1=observed, 0=missing)
   Returns {:belief updated-belief, :ll per-element marginal LL}."
  [belief obs base-mean loading noise-std mask]
  (let [{:keys [mean var]} belief
        r2       (mx/multiply noise-std noise-std)
        pred-obs (mx/add base-mean (mx/multiply loading mean))
        innov    (mx/subtract obs pred-obs)
        S        (mx/add (mx/multiply loading (mx/multiply loading var)) r2)
        K        (mx/divide (mx/multiply loading var) S)
        ;; Marginal LL: -0.5 * (log(2π) + log(S) + innov²/S)
        ll       (mx/multiply (mx/scalar -0.5)
                   (mx/add (mx/scalar LOG-2PI)
                     (mx/add (mx/log S)
                       (mx/divide (mx/multiply innov innov) S))))
        ;; Masked update: if mask=0, belief unchanged, ll=0
        new-mean (mx/add mean (mx/multiply mask (mx/multiply K innov)))
        new-var  (mx/subtract var
                   (mx/multiply mask
                     (mx/multiply K (mx/multiply loading var))))
        masked-ll (mx/multiply mask ll)]
    {:belief {:mean new-mean :var new-var}
     :ll     masked-ll}))

(defn kalman-sequential-update
  "Sequential Kalman updates for multiple observations.
   belief:       {:mean array, :var array}
   observations: [{:obs :base-mean :loading :noise-std :mask} ...]
   Returns {:belief updated, :ll per-element total LL}."
  [belief observations]
  (reduce
    (fn [{:keys [belief ll]} {:keys [obs base-mean loading noise-std mask]}]
      (let [result (kalman-update belief obs base-mean loading noise-std mask)]
        {:belief (:belief result) :ll (mx/add ll (:ll result))}))
    {:belief belief :ll (mx/zeros (mx/shape (:mean belief)))}
    observations))

(defn kalman-step
  "One complete Kalman step: predict + sequential observation updates.
   belief:       {:mean array, :var array}
   latent:       {:transition-coeff scalar, :process-noise scalar}
   observations: [{:obs :base-mean :loading :noise-std :mask} ...]
   Returns {:belief updated, :ll per-element total LL for this step}."
  [belief {:keys [transition-coeff process-noise]} observations]
  (let [pred (kalman-predict belief transition-coeff process-noise)]
    (kalman-sequential-update pred observations)))

;; ---------------------------------------------------------------------------
;; Handler middleware (Level 2)
;; ---------------------------------------------------------------------------
;;
;; The handler intercepts kalman-latent and kalman-obs trace sites.
;; Observation LL accumulates in :kalman-ll ([P]-shaped, per-element),
;; NOT in :score/:weight (which stay at 0). This keeps the LL structure
;; intact for the caller to aggregate (sum, mean, etc.).
;;
;; Key design: initial belief = {mean: zeros, var: zeros}. The handler
;; ALWAYS predicts on kalman-latent. At t=0, predict({0,0}, rho, q)
;; gives {0, q²} — the correct prior. No skip-predict flag needed.

(defn make-kalman-dispatch
  "Create Kalman dispatch map for use with wrap-analytical.

   latent-addr: keyword address of the latent state site

   Returns dispatch map: {:kalman-latent handler, :kalman-obs handler}."
  [latent-addr]
  {:kalman-latent
   (fn [state addr dist]
     (if (= addr latent-addr)
       (let [{:keys [transition-coeff process-noise]} (:params dist)
             belief (or (:kalman-belief state)
                        {:mean (mx/zeros [(:kalman-n state)])
                         :var  (mx/zeros [(:kalman-n state)])})
             new-belief (kalman-predict belief transition-coeff process-noise)]
         [(:mean new-belief)
          (-> state
              (assoc :kalman-belief new-belief)
              (update :choices cm/set-value addr (:mean new-belief)))])
       ;; Not our latent addr — return nil to fall through via wrap-analytical
       nil))

   :kalman-obs
   (fn [state addr dist]
     (let [{:keys [base-mean loading noise-std mask]} (:params dist)
           belief (:kalman-belief state)
           constraint (cm/get-submap (:constraints state) addr)
           obs (cm/get-value constraint)
           {:keys [belief ll]} (kalman-update belief obs base-mean loading noise-std mask)
           n (:kalman-n state)]
       [obs (-> state
                (assoc :kalman-belief belief)
                (update :choices cm/set-value addr obs)
                (update :kalman-ll
                  #(mx/add (or % (mx/zeros [n])) ll)))]))})

(defn make-kalman-transition
  "Handler middleware: wraps generate-transition for Kalman filtering.

   The cognitive architecture is a gen function that uses:
   - (trace :z (kalman-latent rho z-prev noise)) for latent dynamics
   - (trace :obs (kalman-obs base-mean loading z noise-std mask)) for observations

   Under this handler:
   - kalman-latent sites: Kalman predict, return belief mean
   - kalman-obs sites: Kalman update with constraint, accumulate LL in :kalman-ll
   - Other sites: delegate to generate-transition

   Handler state additions:
   - :kalman-belief {:mean :var}
   - :kalman-n      number of elements
   - :kalman-ll     [P]-shaped accumulated marginal LL"
  [latent-addr]
  (ana/wrap-analytical h/generate-transition (make-kalman-dispatch latent-addr)))

(defn kalman-generate
  "Run a gen function body under the Kalman handler.

   gf:          DynamicGF (gen function) with kalman-latent/kalman-obs sites
   args:        gen function arguments
   constraints: choicemap with observation constraints
   latent-addr: keyword address of the latent state
   n:           number of elements in belief (e.g. patients)
   key:         PRNG key

   opts (map):
   - :param-store  parameter store for param sites
   - :init-belief  initial belief {:mean :var} (default: {zeros, zeros})

   Returns handler result map with :retval, :choices, :score, :weight,
   plus :kalman-belief and :kalman-ll ([P]-shaped marginal LL)."
  [gf args constraints latent-addr n key & [opts]]
  (let [{:keys [param-store init-belief]} opts
        transition (make-kalman-transition latent-addr)
        init-state (cond-> {:choices cm/EMPTY
                            :score (mx/scalar 0.0)
                            :weight (mx/scalar 0.0)
                            :key key
                            :constraints constraints
                            :kalman-n n
                            :kalman-belief (or init-belief
                                              {:mean (mx/zeros [n])
                                               :var  (mx/zeros [n])})}
                     param-store (assoc :param-store param-store))]
    (rt/run-handler transition init-state
      (fn [rt] (apply (:body-fn gf) rt args)))))

(defn kalman-fold
  "Fold a per-step gen function over T timesteps under the Kalman handler.

   step-fn:     gen function with kalman-latent and kalman-obs trace sites
   latent-addr: keyword address of the latent state
   n:           number of elements (patients)
   T:           number of timesteps
   context-fn:  (fn [t] -> {:args [step-fn-args], :constraints choicemap})
                builds per-timestep args and observation constraints.

   Uses {mean: zeros, var: zeros} initial belief. The handler always
   predicts — at t=0 this gives the N(0, q²) prior.

   Returns [P]-shaped total marginal LL (not summed — caller aggregates)."
  [step-fn latent-addr n T context-fn]
  (loop [t 0
         belief {:mean (mx/zeros [n]) :var (mx/zeros [n])}
         acc-ll (mx/zeros [n])]
    (if (>= t T)
      acc-ll
      (let [{:keys [args constraints]} (context-fn t)
            result (kalman-generate
                     step-fn args constraints latent-addr n
                     (rng/fresh-key t)
                     {:init-belief belief})
            step-ll (or (:kalman-ll result) (mx/zeros [n]))]
        (recur (inc t)
               (:kalman-belief result)
               (mx/add acc-ll step-ll))))))
