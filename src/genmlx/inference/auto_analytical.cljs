(ns genmlx.inference.auto-analytical
  "Address-based analytical handlers for Level 3 automatic elimination.

   Dispatches on trace ADDRESS (not dist type), works with standard
   distributions (dist/gaussian, dist/beta-dist, etc.), and reuses
   the existing conjugate math from conjugate.cljs.

   The key difference from conjugate.cljs:
   - conjugate.cljs: dispatches on (:type dist) — requires nn-prior/nn-obs etc.
   - auto_analytical: dispatches on addr keyword — works with dist/gaussian etc.

   Score accounting:
   - Prior sites: NO score/weight contribution (prior is marginalized out)
   - Obs sites (constrained): marginal LL → both :score and :weight
   - Obs sites (unconstrained): fallthrough to base handler (nil return)

   State keys:
   - :auto-posteriors {prior-addr -> posterior-map}

   Posterior maps by family:
   - :normal-normal    {:mean MLX-scalar, :var MLX-scalar}
   - :beta-bernoulli   {:alpha MLX-scalar, :beta MLX-scalar}
   - :gamma-poisson    {:shape MLX-scalar, :rate MLX-scalar}
   - :gamma-exponential {:shape MLX-scalar, :rate MLX-scalar}"
  (:require [genmlx.mlx :as mx]
            [genmlx.choicemap :as cm]
            [genmlx.conjugacy :as conj]
            [genmlx.inference.conjugate :as conjugate]))

;; ---------------------------------------------------------------------------
;; Pure update functions — thin wrappers over conjugate.cljs
;; ---------------------------------------------------------------------------

(def ^:private LOG-2PI 1.8378770664093453)
(def ^:private MASK-ON (mx/scalar 1.0))

(defn nn-update-step
  "Normal-Normal conjugate update. Wraps conjugate/nn-update with mask=1.
   posterior: {:mean :var}, obs-value, obs-var (sigma^2)
   Returns: {:mean :var :ll}"
  [posterior obs-value obs-var]
  (let [r (conjugate/nn-update posterior obs-value obs-var MASK-ON)]
    {:mean (:mean (:posterior r)) :var (:var (:posterior r)) :ll (:ll r)}))

(defn bb-update-step
  "Beta-Bernoulli conjugate update. Wraps conjugate/bb-update with mask=1.
   posterior: {:alpha :beta}, obs-value (0 or 1)
   Returns: {:alpha :beta :ll}"
  [posterior obs-value]
  (let [r (conjugate/bb-update posterior obs-value MASK-ON)]
    {:alpha (:alpha (:posterior r)) :beta (:beta (:posterior r)) :ll (:ll r)}))

(defn gp-update-step
  "Gamma-Poisson conjugate update. Wraps conjugate/gp-update with mask=1.
   posterior: {:shape :rate}, obs-value (count)
   Returns: {:shape :rate :ll}"
  [posterior obs-value]
  (let [r (conjugate/gp-update posterior obs-value MASK-ON)]
    {:shape (:shape (:posterior r)) :rate (:rate (:posterior r)) :ll (:ll r)}))

(defn ge-update-step
  "Gamma-Exponential conjugate update.
   posterior: {:shape :rate}
   obs-value: positive real (as MLX scalar)
   Returns: {:shape :rate :ll}"
  [{:keys [shape rate]} obs-value]
  (let [;; Marginal: Lomax (Pareto Type II)
        ;; p(x | shape, rate) = shape * rate^shape / (rate + x)^(shape+1)
        ;; log p(x) = log(shape) + shape*log(rate) - (shape+1)*log(rate+x)
        ll (-> (mx/log shape)
               (mx/add (mx/multiply shape (mx/log rate)))
               (mx/subtract (mx/multiply (mx/add shape (mx/scalar 1.0))
                                         (mx/log (mx/add rate obs-value)))))
        new-shape (mx/add shape (mx/scalar 1.0))
        new-rate (mx/add rate obs-value)]
    {:shape new-shape :rate new-rate :ll ll}))

;; ---------------------------------------------------------------------------
;; Address-based dispatch
;; ---------------------------------------------------------------------------

(defn make-address-dispatch
  "Build a handler transition that dispatches on trace address.
   base: fallback transition for non-intercepted addresses
   addr-handlers: {addr (fn [state addr dist] -> [value state'] or nil)}

   When a handler returns nil, falls through to base transition."
  [base addr-handlers]
  (fn [state addr dist]
    (if-let [handler (get addr-handlers addr)]
      (or (handler state addr dist)
          (base state addr dist))
      (base state addr dist))))

;; ---------------------------------------------------------------------------
;; Generic handler builder (shared across all conjugate families)
;; ---------------------------------------------------------------------------

(defn- make-conjugate-handlers
  "Build address-based prior + obs handlers for a conjugate family.

   prior-addr:     keyword address of the prior trace site
   obs-addrs:      vector of observation site addresses
   init-posterior: (fn [dist-params] -> posterior-map) — extract initial posterior
   posterior-mean: (fn [posterior-map] -> MLX-scalar) — point estimate from posterior
   update-step:    (fn [posterior obs-value dist-params] -> {:posterior map :ll scalar})

   Returns {addr handler-fn} for prior + all obs addresses."
  [prior-addr obs-addrs init-posterior posterior-mean update-step]
  (let [prior-handler
        (fn [state addr dist]
          (let [posterior (init-posterior (:params dist))
                post-mean (posterior-mean posterior)]
            ;; Return prior mean; don't add to score (marginalized out)
            [post-mean (-> state
                         (assoc-in [:auto-posteriors prior-addr] posterior)
                         (update :choices cm/set-value addr post-mean))]))

        obs-handler
        (fn [state addr dist]
          (let [constraint (cm/get-submap (:constraints state) addr)]
            (when (cm/has-value? constraint)
              ;; Only handle constrained obs analytically
              (let [obs-value (cm/get-value constraint)
                    posterior (get-in state [:auto-posteriors prior-addr])
                    {:keys [posterior ll]} (update-step posterior obs-value (:params dist))
                    post-mean (posterior-mean posterior)]
                [obs-value (-> state
                             (assoc-in [:auto-posteriors prior-addr] posterior)
                             (update :choices cm/set-value addr obs-value)
                             (update :score #(mx/add % ll))
                             (update :weight #(mx/add % ll))
                             ;; Update prior's value to posterior mean
                             (update :choices cm/set-value prior-addr post-mean))]))))]
    (merge {prior-addr prior-handler}
           (into {} (map (fn [oa] [oa obs-handler]) obs-addrs)))))

;; ---------------------------------------------------------------------------
;; Per-family configs
;; ---------------------------------------------------------------------------

(defn make-auto-nn-handlers
  "Build address-based handlers for a Normal-Normal conjugate pair."
  [prior-addr obs-addrs]
  (make-conjugate-handlers prior-addr obs-addrs
    (fn [{:keys [mu sigma]}]
      {:mean mu :var (mx/multiply sigma sigma)})
    :mean
    (fn [posterior obs-value {:keys [sigma]}]
      (let [obs-var (mx/multiply sigma sigma)
            {:keys [mean var ll]} (nn-update-step posterior obs-value obs-var)]
        {:posterior {:mean mean :var var} :ll ll}))))

(defn make-auto-bb-handlers
  "Build address-based handlers for a Beta-Bernoulli conjugate pair."
  [prior-addr obs-addrs]
  (make-conjugate-handlers prior-addr obs-addrs
    (fn [{:keys [alpha beta-param]}]
      {:alpha alpha :beta beta-param})
    (fn [{:keys [alpha beta]}] (mx/divide alpha (mx/add alpha beta)))
    (fn [posterior obs-value _params]
      (let [{:keys [alpha beta ll]} (bb-update-step posterior obs-value)]
        {:posterior {:alpha alpha :beta beta} :ll ll}))))

(defn make-auto-gp-handlers
  "Build address-based handlers for a Gamma-Poisson conjugate pair."
  [prior-addr obs-addrs]
  (make-conjugate-handlers prior-addr obs-addrs
    (fn [{:keys [shape-param rate]}]
      {:shape shape-param :rate rate})
    (fn [{:keys [shape rate]}] (mx/divide shape rate))
    (fn [posterior obs-value _params]
      (let [{:keys [shape rate ll]} (gp-update-step posterior obs-value)]
        {:posterior {:shape shape :rate rate} :ll ll}))))

(defn make-auto-ge-handlers
  "Build address-based handlers for a Gamma-Exponential conjugate pair."
  [prior-addr obs-addrs]
  (make-conjugate-handlers prior-addr obs-addrs
    (fn [{:keys [shape-param rate]}]
      {:shape shape-param :rate rate})
    (fn [{:keys [shape rate]}] (mx/divide shape rate))
    (fn [posterior obs-value _params]
      (let [{:keys [shape rate ll]} (ge-update-step posterior obs-value)]
        {:posterior {:shape shape :rate rate} :ll ll}))))

;; ---------------------------------------------------------------------------
;; Auto-Kalman handlers (for detected linear-Gaussian chains)
;; ---------------------------------------------------------------------------

(defn- kalman-predict-belief
  "Predict belief for step i given belief from step i-1.
   transition: the transition descriptor from steps[i-1]
   noise-var: process noise variance for step i"
  [prev-belief transition noise-var]
  (let [coeff (if (= :direct (:type transition))
                (mx/scalar 1.0)
                (let [c (:coefficient transition)]
                  (if (number? c) (mx/scalar c) c)))]
    {:mean (mx/multiply coeff (:mean prev-belief))
     :var (mx/add (mx/multiply coeff (mx/multiply coeff (:var prev-belief)))
                  noise-var)}))

(defn- cascade-predictions
  "After updating latent at step-idx, re-predict all downstream latent beliefs.
   This ensures observations later in the chain use beliefs that incorporate
   earlier observations, even when the gen body executes all latents before
   all observations.

   steps: chain steps vector
   step-idx: index of the step just updated
   state: current handler state with :auto-kalman-beliefs
   noise-vars: vector of noise variances per step (pre-computed)"
  [steps step-idx state noise-vars]
  (let [n (count steps)]
    (reduce
      (fn [st j]
        ;; Only re-predict if this step has already been initialized
        ;; (noise-var recorded). Steps not yet executed will naturally
        ;; use the updated predecessor belief when they run.
        (if-let [nv (get noise-vars j)]
          (let [prev-belief (get-in st [:auto-kalman-beliefs (:latent (nth steps (dec j)))])
                transition (:transition (nth steps (dec j)))
                new-belief (kalman-predict-belief prev-belief transition nv)]
            (-> st
              (assoc-in [:auto-kalman-beliefs (:latent (nth steps j))] new-belief)
              (update :choices cm/set-value (:latent (nth steps j)) (:mean new-belief))))
          (reduced st)))
      state
      (range (inc step-idx) n))))

(defn make-auto-kalman-handlers
  "Build address-based handlers for a Kalman chain.
   Reuses the same math as kalman.cljs but dispatches on address.

   chain: output of detect-kalman-chains, a chain descriptor with :steps
   Each step has :latent, :observations, :transition, :noise-std.

   Stores per-step beliefs in :auto-kalman-beliefs keyed by latent address.
   After each observation update, cascades re-prediction through downstream
   latents so the Kalman filter processes observations sequentially even when
   the gen body executes all latents before all observations.

   Returns {addr handler-fn} for all latent and observation addresses."
  [chain]
  (let [steps (:steps chain)
        n-steps (count steps)
        ;; Map latent addr -> step index for cascade lookup
        latent->idx (into {} (map-indexed (fn [i s] [(:latent s) i]) steps))
        ;; Map from obs addr to its parent latent addr and dep type
        obs-to-latent (into {}
                        (mapcat (fn [step]
                                  (map (fn [oa odt] [oa {:latent (:latent step) :dep-type odt}])
                                       (:observations step)
                                       (:obs-dep-types step)))
                                steps))
        ;; Build latent handlers — one per step
        latent-handlers
        (into {}
          (map-indexed
            (fn [i step]
              [(:latent step)
               (fn [state addr dist]
                 (let [params (:params dist)
                       prev-latent (when (> i 0) (:latent (nth steps (dec i))))
                       prev-belief (when prev-latent
                                     (get-in state [:auto-kalman-beliefs prev-latent]))
                       new-belief
                       (if (nil? prev-belief)
                         ;; Root: initialize from prior parameters
                         (let [prior-mean (:mu params)
                               prior-std (:sigma params)
                               prior-var (mx/multiply prior-std prior-std)]
                           {:mean prior-mean :var prior-var})
                         ;; Non-root: Kalman predict step
                         (let [transition (:transition (nth steps (dec i)))
                               noise-std (:sigma params)
                               noise-var (mx/multiply noise-std noise-std)]
                           (kalman-predict-belief prev-belief transition noise-var)))]
                   ;; Store noise-var for cascade re-prediction
                   [(:mean new-belief)
                    (-> state
                      (assoc-in [:auto-kalman-beliefs addr] new-belief)
                      (assoc-in [:auto-kalman-noise-vars i]
                                (mx/multiply (:sigma params) (:sigma params)))
                      (update :choices cm/set-value addr (:mean new-belief)))]))])
            steps))

        ;; Build observation handlers
        obs-handlers
        (into {}
          (map
            (fn [[obs-addr {:keys [latent dep-type]}]]
              [obs-addr
               (fn [state addr dist]
                 (let [constraint (cm/get-submap (:constraints state) addr)]
                   (when (cm/has-value? constraint)
                     (let [obs-value (cm/get-value constraint)
                           params (:params dist)
                           obs-std (:sigma params)
                           obs-var (mx/multiply obs-std obs-std)
                           loading (if (= :direct (:type dep-type))
                                     (mx/scalar 1.0)
                                     (let [c (:coefficient dep-type)]
                                       (if (number? c) (mx/scalar c) c)))
                           offset (if (= :direct (:type dep-type))
                                    (mx/scalar 0.0)
                                    (let [o (:offset dep-type)]
                                      (if (number? o) (mx/scalar o)
                                        (if (= 0 o) (mx/scalar 0.0) o))))
                           ;; Get belief for THIS latent step
                           belief (get-in state [:auto-kalman-beliefs latent])
                           ;; Kalman update
                           pred-obs (mx/add offset (mx/multiply loading (:mean belief)))
                           innov (mx/subtract obs-value pred-obs)
                           S (mx/add (mx/multiply loading
                                       (mx/multiply loading (:var belief)))
                                     obs-var)
                           K (mx/divide (mx/multiply loading (:var belief)) S)
                           ll (mx/multiply (mx/scalar -0.5)
                                (mx/add (mx/scalar LOG-2PI)
                                  (mx/add (mx/log S)
                                    (mx/divide (mx/multiply innov innov) S))))
                           new-mean (mx/add (:mean belief) (mx/multiply K innov))
                           new-var (mx/subtract (:var belief)
                                     (mx/multiply K (mx/multiply loading (:var belief))))
                           new-belief {:mean new-mean :var new-var}
                           step-idx (get latent->idx latent)
                           ;; Update this latent's belief
                           state' (-> state
                                    (assoc-in [:auto-kalman-beliefs latent] new-belief)
                                    (update :choices cm/set-value addr obs-value)
                                    (update :score #(mx/add % ll))
                                    (update :weight #(mx/add % ll))
                                    (update :choices cm/set-value latent new-mean))
                           ;; Cascade re-prediction to all downstream latents
                           noise-vars (:auto-kalman-noise-vars state')
                           state'' (if (and noise-vars (< (inc step-idx) n-steps))
                                     (cascade-predictions steps step-idx state' noise-vars)
                                     state')]
                       [obs-value state'']))))])
            obs-to-latent))]
    (merge latent-handlers obs-handlers)))

(defn build-auto-kalman-handlers
  "Build address-based handlers for all detected Kalman chains.
   Returns a merged map of {addr handler-fn}."
  [chains]
  (reduce
    (fn [handlers chain]
      (merge handlers (make-auto-kalman-handlers chain)))
    {}
    chains))

;; ---------------------------------------------------------------------------
;; Factory dispatch
;; ---------------------------------------------------------------------------

(def family->handler-factory
  "Map from conjugate family keyword to handler factory function."
  {:normal-normal      make-auto-nn-handlers
   :beta-bernoulli     make-auto-bb-handlers
   :gamma-poisson      make-auto-gp-handlers
   :gamma-exponential  make-auto-ge-handlers})

(defn build-auto-handlers
  "Build address-based handlers from detected conjugate pairs.
   Returns a merged map of {addr handler-fn} for all conjugate sites."
  [conjugate-pairs]
  (let [grouped (conj/group-by-prior conjugate-pairs)]
    (reduce
      (fn [handlers [prior-addr pairs]]
        (let [family (:family (first pairs))
              factory (get family->handler-factory family)
              obs-addrs (mapv :obs-addr pairs)]
          (if factory
            (merge handlers (factory prior-addr obs-addrs))
            handlers)))
      {}
      grouped)))

;; ---------------------------------------------------------------------------
;; Utility: check if any conjugate obs is constrained
;; ---------------------------------------------------------------------------

(defn some-conjugate-obs-constrained?
  "Check if any observation site in a conjugate pair is actually constrained.
   If no conjugate obs are constrained, there's no benefit to analytical handler."
  [conjugate-pairs constraints]
  (boolean
    (some (fn [{:keys [obs-addr]}]
            (cm/has-value? (cm/get-submap constraints obs-addr)))
          conjugate-pairs)))
