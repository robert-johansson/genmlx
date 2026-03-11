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
            [genmlx.affine :as affine]
            [genmlx.selection :as sel]
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
;; MVN-MVN conjugate update (multivariate Normal-Normal)
;; ---------------------------------------------------------------------------

(defn- log-det-cholesky
  "Compute log|det(A)| = 2 * sum(log(diag(L))) where L = cholesky(A)."
  [L]
  (mx/multiply (mx/scalar 2.0) (mx/sum (mx/log (mx/diag L)))))

(defn- mvn-well-conditioned?
  "Check if a covariance matrix is well-conditioned (min diag > threshold).
   Returns false if any diagonal element is below 1e-6."
  [cov-matrix]
  (let [diag-vals (mx/diag cov-matrix)
        min-diag (mx/item (mx/amin diag-vals))]
    (> min-diag 1e-6)))

(defn mvn-update-step
  "MVN-MVN conjugate update.
   Prior:     mu ~ N(m0, S0)
   Obs:       y | mu ~ N(mu, R)
   Posterior: mu | y ~ N(m1, S1) where
     S1 = (S0^-1 + R^-1)^-1
     m1 = S1 * (S0^-1 * m0 + R^-1 * y)
   Marginal:  y ~ N(m0, S0 + R)
     log p(y) = -0.5 * (d*log(2pi) + log|S0+R| + (y-m0)^T(S0+R)^-1(y-m0))

   Uses Cholesky-first approach per architect spec.
   posterior: {:mean-vec [d], :cov-matrix [d,d]}
   obs-value: [d] array
   obs-cov: [d,d] observation covariance matrix
   Returns: {:mean-vec :cov-matrix :ll} or nil if ill-conditioned."
  [{:keys [mean-vec cov-matrix]} obs-value obs-cov]
  (let [d (first (mx/shape mean-vec))
        ;; Marginal covariance M = S0 + R
        M (mx/add cov-matrix obs-cov)]
    ;; Condition number guard
    (when (mvn-well-conditioned? M)
      (let [L-M (mx/cholesky M)
            log-det-M (log-det-cholesky L-M)
            ;; Marginal LL
            diff (mx/subtract obs-value mean-vec)
            M-inv-diff (mx/flatten (mx/solve M (mx/reshape diff [d 1])))
            mahal (mx/sum (mx/multiply diff M-inv-diff))
            ll (mx/multiply (mx/scalar -0.5)
                            (mx/add (mx/scalar (* d LOG-2PI))
                                    (mx/add log-det-M mahal)))
            ;; Posterior: S1 = (S0^-1 + R^-1)^-1, m1 = S1*(S0^-1*m0 + R^-1*y)
            S0-inv (mx/inv cov-matrix)
            R-inv (mx/inv obs-cov)
            S1 (mx/inv (mx/add S0-inv R-inv))
            m1 (mx/flatten
                (mx/matmul S1
                           (mx/add (mx/matmul S0-inv (mx/reshape mean-vec [d 1]))
                                   (mx/matmul R-inv (mx/reshape obs-value [d 1])))))]
        {:mean-vec m1 :cov-matrix S1 :ll ll}))))

(defn make-auto-mvn-handlers
  "Build address-based handlers for a MVN-Normal conjugate pair."
  [prior-addr obs-addrs]
  (let [prior-handler
        (fn [state addr dist]
          (let [params (:params dist)
                mean-vec (:mean-vec params)
                cov-matrix (:cov-matrix params)
                posterior {:mean-vec mean-vec :cov-matrix cov-matrix}]
            ;; Return prior mean; don't add to score (marginalized out)
            [mean-vec (-> state
                          (assoc-in [:auto-posteriors prior-addr] posterior)
                          (update :choices cm/set-value addr mean-vec))]))

        obs-handler
        (fn [state addr dist]
          (let [constraint (cm/get-submap (:constraints state) addr)]
            (when (cm/has-value? constraint)
              (let [obs-value (cm/get-value constraint)
                    posterior (get-in state [:auto-posteriors prior-addr])
                    obs-cov (:cov-matrix (:params dist))
                    result (mvn-update-step posterior obs-value obs-cov)]
                ;; Fallthrough if ill-conditioned
                (when result
                  (let [{:keys [mean-vec cov-matrix ll]} result]
                    [obs-value (-> state
                                   (assoc-in [:auto-posteriors prior-addr]
                                             {:mean-vec mean-vec :cov-matrix cov-matrix})
                                   (update :choices cm/set-value addr obs-value)
                                   (update :score #(mx/add % ll))
                                   (update :weight #(mx/add % ll))
                                   (update :choices cm/set-value prior-addr mean-vec))]))))))]
    (merge {prior-addr prior-handler}
           (into {} (map (fn [oa] [oa obs-handler]) obs-addrs)))))

;; ---------------------------------------------------------------------------
;; Factory dispatch
;; ---------------------------------------------------------------------------

(def family->handler-factory
  "Map from conjugate family keyword to handler factory function."
  {:normal-normal make-auto-nn-handlers
   :beta-bernoulli make-auto-bb-handlers
   :gamma-poisson make-auto-gp-handlers
   :gamma-exponential make-auto-ge-handlers
   :mvn-normal make-auto-mvn-handlers})

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
;; Regenerate-specific handlers (WP-0, L3.5)
;;
;; Key differences from generate/assess handlers:
;; - Prior handler: uses old value from :old-choices, inits posterior, 0 score.
;;   If prior IS selected → returns nil (Case A fallthrough to base handler).
;; - Obs handler: uses value from :constraints (regen-constraints built from
;;   old-choices), computes marginal LL → adds to :score ONLY (not :weight).
;;   If prior was selected (no posterior) → returns nil (fallthrough).
;; ---------------------------------------------------------------------------

(defn- make-regenerate-conjugate-handlers
  "Build regenerate-specific address-based handlers for a conjugate family.

   Same structure as make-conjugate-handlers but with regenerate weight semantics:
   - Prior: 0 score (marginalized), init posterior from old value. Nil if selected.
   - Obs: marginal LL → :score ONLY (not :weight). Nil if prior was selected."
  [prior-addr obs-addrs init-posterior posterior-mean update-step]
  (let [prior-handler
        (fn [state addr dist]
          ;; Case A: prior is selected → fall through entirely
          (when-not (and (:selection state)
                         (sel/selected? (:selection state) addr))
            ;; Case B: prior NOT selected → use old value, init posterior, 0 score
            (let [old-val (cm/get-value (cm/get-submap (:old-choices state) addr))
                  posterior (init-posterior (:params dist))
                  post-mean (posterior-mean posterior)]
              [old-val (-> state
                           (assoc-in [:auto-posteriors prior-addr] posterior)
                           (update :choices cm/set-value addr old-val))])))

        obs-handler
        (fn [state addr dist]
          ;; If prior was selected (no posterior initialized) → fall through
          (when-let [posterior (get-in state [:auto-posteriors prior-addr])]
            ;; Check old-choices directly (no regen-constraints needed)
            ;; Skip if this obs is in the selection (being resampled)
            (when-not (and (:selection state)
                           (sel/selected? (:selection state) addr))
              (let [old-sub (cm/get-submap (:old-choices state) addr)]
                (when (cm/has-value? old-sub)
                  ;; Obs is constrained → compute marginal LL
                  ;; Add to :score ONLY, not :weight (weight tracks proposal ratio)
                  (let [obs-value (cm/get-value old-sub)
                        {:keys [posterior ll]} (update-step posterior obs-value (:params dist))
                        post-mean (posterior-mean posterior)]
                    [obs-value (-> state
                                   (assoc-in [:auto-posteriors prior-addr] posterior)
                                   (update :choices cm/set-value addr obs-value)
                                   (update :score #(mx/add % ll))
                                 ;; Update prior's value to posterior mean
                                   (update :choices cm/set-value prior-addr post-mean))]))))))]
    (merge {prior-addr prior-handler}
           (into {} (map (fn [oa] [oa obs-handler]) obs-addrs)))))

;; Per-family regenerate handler factories

(defn make-regenerate-nn-handlers
  "Build regenerate-specific handlers for Normal-Normal."
  [prior-addr obs-addrs]
  (make-regenerate-conjugate-handlers prior-addr obs-addrs
                                      (fn [{:keys [mu sigma]}]
                                        {:mean mu :var (mx/multiply sigma sigma)})
                                      :mean
                                      (fn [posterior obs-value {:keys [sigma]}]
                                        (let [obs-var (mx/multiply sigma sigma)
                                              {:keys [mean var ll]} (nn-update-step posterior obs-value obs-var)]
                                          {:posterior {:mean mean :var var} :ll ll}))))

(defn make-regenerate-bb-handlers
  "Build regenerate-specific handlers for Beta-Bernoulli."
  [prior-addr obs-addrs]
  (make-regenerate-conjugate-handlers prior-addr obs-addrs
                                      (fn [{:keys [alpha beta-param]}]
                                        {:alpha alpha :beta beta-param})
                                      (fn [{:keys [alpha beta]}] (mx/divide alpha (mx/add alpha beta)))
                                      (fn [posterior obs-value _params]
                                        (let [{:keys [alpha beta ll]} (bb-update-step posterior obs-value)]
                                          {:posterior {:alpha alpha :beta beta} :ll ll}))))

(defn make-regenerate-gp-handlers
  "Build regenerate-specific handlers for Gamma-Poisson."
  [prior-addr obs-addrs]
  (make-regenerate-conjugate-handlers prior-addr obs-addrs
                                      (fn [{:keys [shape-param rate]}]
                                        {:shape shape-param :rate rate})
                                      (fn [{:keys [shape rate]}] (mx/divide shape rate))
                                      (fn [posterior obs-value _params]
                                        (let [{:keys [shape rate ll]} (gp-update-step posterior obs-value)]
                                          {:posterior {:shape shape :rate rate} :ll ll}))))

(defn make-regenerate-ge-handlers
  "Build regenerate-specific handlers for Gamma-Exponential."
  [prior-addr obs-addrs]
  (make-regenerate-conjugate-handlers prior-addr obs-addrs
                                      (fn [{:keys [shape-param rate]}]
                                        {:shape shape-param :rate rate})
                                      (fn [{:keys [shape rate]}] (mx/divide shape rate))
                                      (fn [posterior obs-value _params]
                                        (let [{:keys [shape rate ll]} (ge-update-step posterior obs-value)]
                                          {:posterior {:shape shape :rate rate} :ll ll}))))

(defn make-regenerate-mvn-handlers
  "Build regenerate-specific handlers for MVN-Normal."
  [prior-addr obs-addrs]
  (let [prior-handler
        (fn [state addr dist]
          (when-not (and (:selection state)
                         (sel/selected? (:selection state) addr))
            (let [old-val (cm/get-value (cm/get-submap (:old-choices state) addr))
                  params (:params dist)
                  posterior {:mean-vec (:mean-vec params) :cov-matrix (:cov-matrix params)}]
              [old-val (-> state
                           (assoc-in [:auto-posteriors prior-addr] posterior)
                           (update :choices cm/set-value addr old-val))])))

        obs-handler
        (fn [state addr dist]
          (when-let [posterior (get-in state [:auto-posteriors prior-addr])]
            (when-not (and (:selection state)
                           (sel/selected? (:selection state) addr))
              (let [old-sub (cm/get-submap (:old-choices state) addr)]
                (when (cm/has-value? old-sub)
                  (let [obs-value (cm/get-value old-sub)
                        obs-cov (:cov-matrix (:params dist))
                        result (mvn-update-step posterior obs-value obs-cov)]
                    (when result
                      (let [{:keys [mean-vec cov-matrix ll]} result]
                        [obs-value (-> state
                                       (assoc-in [:auto-posteriors prior-addr]
                                                 {:mean-vec mean-vec :cov-matrix cov-matrix})
                                       (update :choices cm/set-value addr obs-value)
                                       (update :score #(mx/add % ll))
                                       (update :choices cm/set-value prior-addr mean-vec))]))))))))]
    (merge {prior-addr prior-handler}
           (into {} (map (fn [oa] [oa obs-handler]) obs-addrs)))))

(def regenerate-family->handler-factory
  "Map from conjugate family keyword to regenerate-specific handler factory."
  {:normal-normal make-regenerate-nn-handlers
   :beta-bernoulli make-regenerate-bb-handlers
   :gamma-poisson make-regenerate-gp-handlers
   :gamma-exponential make-regenerate-ge-handlers
   :mvn-normal make-regenerate-mvn-handlers})

(defn build-regenerate-handlers
  "Build regenerate-specific address-based handlers from detected conjugate pairs.
   Returns a merged map of {addr handler-fn}."
  [conjugate-pairs]
  (let [grouped (conj/group-by-prior conjugate-pairs)]
    (reduce
     (fn [handlers [prior-addr pairs]]
       (let [family (:family (first pairs))
             factory (get regenerate-family->handler-factory family)
             obs-addrs (mapv :obs-addr pairs)]
         (if factory
           (merge handlers (factory prior-addr obs-addrs))
           handlers)))
     {}
     grouped)))

;; Regenerate-specific Kalman handlers

(defn make-regenerate-kalman-handlers
  "Build regenerate-specific Kalman handlers.
   Same as make-auto-kalman-handlers but with regenerate weight semantics:
   - Latent: use old value, init belief, 0 score. Nil if selected.
   - Obs: marginal LL → :score ONLY (not :weight). Nil if latent was selected."
  [chain]
  (let [steps (:steps chain)
        n-steps (count steps)
        latent->idx (into {} (map-indexed (fn [i s] [(:latent s) i]) steps))
        obs-to-latent (into {}
                            (mapcat (fn [step]
                                      (map (fn [oa odt] [oa {:latent (:latent step) :dep-type odt}])
                                           (:observations step)
                                           (:obs-dep-types step)))
                                    steps))

        latent-handlers
        (into {}
              (map-indexed
               (fn [i step]
                 [(:latent step)
                  (fn [state addr dist]
                 ;; If selected → fall through (Case A)
                    (when-not (and (:selection state)
                                   (sel/selected? (:selection state) addr))
                   ;; Use old value, init belief, 0 score
                      (let [old-val (cm/get-value (cm/get-submap (:old-choices state) addr))
                            params (:params dist)
                            prev-latent (when (> i 0) (:latent (nth steps (dec i))))
                            prev-belief (when prev-latent
                                          (get-in state [:auto-kalman-beliefs prev-latent]))
                            new-belief
                            (if (nil? prev-belief)
                              {:mean (:mu params)
                               :var (mx/multiply (:sigma params) (:sigma params))}
                              (let [transition (:transition (nth steps (dec i)))
                                    noise-var (mx/multiply (:sigma params) (:sigma params))]
                                (kalman-predict-belief prev-belief transition noise-var)))]
                        [old-val (-> state
                                     (assoc-in [:auto-kalman-beliefs addr] new-belief)
                                     (assoc-in [:auto-kalman-noise-vars i]
                                               (mx/multiply (:sigma params) (:sigma params)))
                                     (update :choices cm/set-value addr old-val))])))])
               steps))

        obs-handlers
        (into {}
              (map
               (fn [[obs-addr {:keys [latent dep-type]}]]
                 [obs-addr
                  (fn [state addr dist]
                 ;; If latent was selected (no belief initialized) → fall through
                    (when-let [belief (get-in state [:auto-kalman-beliefs latent])]
                   ;; Read from old-choices directly, skip if obs is selected
                      (when-not (and (:selection state)
                                     (sel/selected? (:selection state) addr))
                        (let [old-sub (cm/get-submap (:old-choices state) addr)]
                          (when (cm/has-value? old-sub)
                            (let [obs-value (cm/get-value old-sub)
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
                             ;; Score only, NOT weight
                                  state' (-> state
                                             (assoc-in [:auto-kalman-beliefs latent] new-belief)
                                             (update :choices cm/set-value addr obs-value)
                                             (update :score #(mx/add % ll))
                                             (update :choices cm/set-value latent new-mean))
                                  noise-vars (:auto-kalman-noise-vars state')
                                  state'' (if (and noise-vars (< (inc step-idx) n-steps))
                                            (cascade-predictions steps step-idx state' noise-vars)
                                            state')]
                              [obs-value state'']))))))])
               obs-to-latent))]
    (merge latent-handlers obs-handlers)))

(defn build-regenerate-kalman-handlers
  "Build regenerate-specific Kalman handlers for all chains."
  [chains]
  (reduce
   (fn [handlers chain]
     (merge handlers (make-regenerate-kalman-handlers chain)))
   {}
   chains))

;; ---------------------------------------------------------------------------
;; Unified regenerate handler builder (from conjugate-pairs)
;; ---------------------------------------------------------------------------

(defn build-all-regenerate-handlers
  "Build all regenerate-specific handlers from conjugate pairs.
   Mirrors the rewrite engine's logic: detects Kalman chains, builds
   regenerate-specific handlers for both chain and non-chain pairs.
   Returns merged {addr handler-fn}."
  [conjugate-pairs]
  (let [;; Detect Kalman chains (same as rewrite engine)
        chains (affine/detect-kalman-chains conjugate-pairs)
        kalman-handlers (build-regenerate-kalman-handlers chains)
        ;; Addresses claimed by Kalman chains
        kalman-addrs (set (concat (mapcat :latent-addrs chains)
                                  (mapcat :obs-addrs chains)))
        ;; Remaining pairs (not in Kalman chains)
        remaining (remove (fn [p]
                            (or (contains? kalman-addrs (:prior-addr p))
                                (contains? kalman-addrs (:obs-addr p))))
                          conjugate-pairs)
        ;; Build regenerate handlers for remaining pairs
        non-chain-handlers (build-regenerate-handlers remaining)]
    (merge kalman-handlers non-chain-handlers)))

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
