(ns genmlx.inference.kalman
  "Kalman filter middleware for linear-Gaussian state-space models.

   The Kalman filter is handler middleware in the Ring/GenMLX sense:
   the cognitive architecture IS the gen function with latent trace sites,
   the Kalman filter determines how those latent states are handled.

   Two levels of API:

   1. Pure building blocks — kalman-predict, kalman-update, kalman-sequential-update.
      Use these directly in gen function bodies for explicit control.

   2. Handler middleware — make-kalman-transition + kalman-generate.
      Wraps generate-transition to intercept kalman-latent and kalman-obs
      trace sites, running Kalman filtering transparently. The cognitive
      architecture stays clean (latent states as trace sites), and the
      handler does the math.

   Both levels produce identical results. Level 1 is more explicit,
   Level 2 is more composable."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.choicemap :as cm]
            [genmlx.handler :as h]
            [genmlx.runtime :as rt]
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
   Under standard handler: samples from Gaussian.
   Under Kalman handler: provides loading + noise for update step."
  [base-mean loading latent-value noise-std]
  (sample [key]
    (dc/dist-sample
      (dist/gaussian (mx/add base-mean (mx/multiply loading latent-value)) noise-std)
      key))
  (log-prob [v]
    (dc/dist-log-prob
      (dist/gaussian (mx/add base-mean (mx/multiply loading latent-value)) noise-std)
      v)))

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

(defn make-kalman-transition
  "Handler middleware: wraps generate-transition for Kalman filtering.

   The cognitive architecture is a gen function that uses:
   - (trace :z (kalman-latent rho z-prev noise)) for latent dynamics
   - (trace :obs (kalman-obs base-mean loading z noise-std)) for observations

   Under this handler:
   - kalman-latent sites: Kalman predict, return belief mean
   - kalman-obs sites: Kalman update with constraint, accumulate marginal LL
   - Other sites: delegate to generate-transition

   Handler state additions:
   - :kalman-belief {:mean :var}
   - :kalman-n      number of elements
   - :kalman-masks  {addr -> mask-array} for missing data"
  [latent-addr]
  (fn [state addr dist]
    (cond
      ;; Latent site: Kalman predict
      (and (= addr latent-addr) (= (:type dist) :kalman-latent))
      (let [{:keys [transition-coeff process-noise]} (:params dist)
            belief (or (:kalman-belief state)
                       (kalman-init (:kalman-n state)))
            new-belief (kalman-predict belief transition-coeff process-noise)]
        [(:mean new-belief)
         (-> state
             (assoc :kalman-belief new-belief)
             (update :choices cm/set-value addr (:mean new-belief)))])

      ;; Observation site: Kalman update
      (= (:type dist) :kalman-obs)
      (let [{:keys [base-mean loading noise-std]} (:params dist)
            belief (:kalman-belief state)
            constraint (cm/get-submap (:constraints state) addr)
            obs (cm/get-value constraint)
            mask (get (:kalman-masks state) addr
                      (mx/ones (mx/shape (:mean belief))))
            {:keys [belief ll]} (kalman-update belief obs base-mean loading noise-std mask)
            total-ll (mx/sum ll)]
        [obs (-> state
                 (assoc :kalman-belief belief)
                 (update :choices cm/set-value addr obs)
                 (update :score #(mx/add % total-ll))
                 (update :weight #(mx/add % total-ll)))])

      ;; Standard site: delegate
      :else
      (h/generate-transition state addr dist))))

(defn kalman-generate
  "Run a gen function body under the Kalman handler.

   gf:          DynamicGF (gen function) with kalman-latent/kalman-obs sites
   args:        gen function arguments
   constraints: choicemap with observation constraints
   latent-addr: keyword address of the latent state
   n:           number of elements in belief (e.g. patients)
   key:         PRNG key

   opts (map):
   - :masks        {addr -> mask-array} for missing data
   - :param-store  parameter store for param sites
   - :init-belief  custom initial belief (default: N(0,1))

   Returns handler result map with :retval, :weight, :score, :choices,
   plus :kalman-belief (final belief state)."
  [gf args constraints latent-addr n key & [opts]]
  (let [{:keys [masks param-store init-belief]} opts
        transition (make-kalman-transition latent-addr)
        init-state (cond-> {:choices cm/EMPTY
                            :score (mx/scalar 0.0)
                            :weight (mx/scalar 0.0)
                            :key key
                            :constraints constraints
                            :kalman-n n
                            :kalman-belief (or init-belief (kalman-init n))
                            :kalman-masks (or masks {})}
                     param-store (assoc :param-store param-store))]
    (rt/run-handler transition init-state
      (fn [rt] (apply (:body-fn gf) rt args)))))
