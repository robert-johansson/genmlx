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
            [genmlx.mlx.constants :refer [LOG-2PI]]
            [genmlx.choicemap :as cm]
            [genmlx.conjugacy :as conj]
            [genmlx.affine :as affine]
            [genmlx.selection :as sel]
            [genmlx.inference.conjugate :as conjugate]))

;; ---------------------------------------------------------------------------
;; Pure update functions — thin wrappers over conjugate.cljs
;; ---------------------------------------------------------------------------

(def ^:private MASK-ON (mx/scalar 1.0))

(defn nn-update-step
  "Normal-Normal conjugate update. Wraps conjugate/nn-update with mask=1.
   posterior: {:mean :var}, obs-value, obs-var (sigma^2)
   Returns: {:mean :var :ll}"
  [posterior obs-value obs-var]
  (let [r (conjugate/nn-update posterior obs-value obs-var MASK-ON)]
    {:mean (:mean (:posterior r)) :var (:var (:posterior r)) :ll (:ll r)}))

(defn nn-iid-update-step
  "Normal-IID-Normal conjugate update. Processes [T] observations at once.
   posterior: {:mean :var}, obs-values: [T]-shaped array, obs-var (sigma^2)
   Returns: {:mean :var :ll}

   Math: T observations y_1..y_T drawn from N(mu, sigma^2) with a SHARED
   mu ~ N(prior-mean m0, prior-var tau^2).
     posterior-prec = 1/tau^2 + T/sigma^2
     posterior-mean = posterior-var * (m0/tau^2 + sum(y)/sigma^2)
     marginal LL    = log N(y ; m0*1, Sigma),  Sigma = sigma^2 I + tau^2 11^T

   The marginal is the JOINT multivariate normal: because mu is shared, the y_i
   are correlated (Cov(y_i,y_j) = tau^2 for i!=j), NOT independent. Computed in
   closed form via the matrix-determinant lemma / Sherman-Morrison (fixes
   genmlx-ke9i: the previous code summed independent per-point marginals
   sum_i log N(y_i; m0, tau^2+sigma^2), ignoring the shared-mu covariance — off
   by ~3.66 nats for T>1, correct only at T=1)."
  [posterior obs-values obs-var]
  (let [{:keys [mean var]} posterior
        obs-values (mx/ensure-array obs-values)
        t-val (mx/scalar (first (mx/shape obs-values)))
        sum-obs (mx/sum obs-values)
        ;; Posterior (unchanged)
        inv-prior (mx/divide (mx/scalar 1.0) var)
        inv-obs-total (mx/divide t-val obs-var)
        new-var (mx/divide (mx/scalar 1.0) (mx/add inv-prior inv-obs-total))
        new-mean (mx/multiply new-var
                              (mx/add (mx/multiply inv-prior mean)
                                      (mx/divide sum-obs obs-var)))
        ;; Marginal LL = log N(y; m0*1, sigma^2 I + tau^2 11^T)  (shared-mu joint).
        ;; det(Sigma)   = (sigma^2)^(T-1) * (sigma^2 + T*tau^2)
        ;; Sigma^{-1}   = (1/sigma^2)(I - (tau^2/(sigma^2+T*tau^2)) 11^T)
        ;; quad = d' Sigma^{-1} d = (sum d_i^2 - (tau^2/denom)(sum d_i)^2)/sigma^2
        s2 obs-var                                   ; sigma^2
        t2 var                                       ; tau^2
        innov (mx/subtract obs-values mean)          ; d_i = y_i - m0   [T]
        sum-d (mx/sum innov)
        sum-d2 (mx/sum (mx/multiply innov innov))
        denom (mx/add s2 (mx/multiply t-val t2))     ; sigma^2 + T*tau^2
        logdet (mx/add (mx/multiply (mx/subtract t-val (mx/scalar 1.0))
                                    (mx/log (mx/multiply MASK-ON s2)))
                       (mx/log denom))
        quad (mx/divide (mx/subtract sum-d2
                                     (mx/multiply (mx/divide t2 denom)
                                                  (mx/multiply sum-d sum-d)))
                        s2)
        ll (mx/multiply (mx/scalar -0.5)
                        (mx/add (mx/multiply t-val (mx/scalar LOG-2PI))
                                (mx/add logdet quad)))]
    {:mean new-mean :var new-var :ll ll}))

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

(defn- ge-update-step
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

(defn- make-conjugate-handlers-core
  "Build address-based prior + obs handlers for a conjugate family, parameterized
   by `mode` (:generate or :regenerate) — mirroring make-kalman-handlers-core.

   prior-addr:     keyword address of the prior trace site
   obs-addrs:      vector of observation site addresses
   init-posterior: (fn [dist-params] -> posterior-map) — extract initial posterior
   posterior-mean: (fn [posterior-map] -> MLX-scalar) — point estimate from posterior
   update-step:    (fn [posterior obs-value dist-params] -> {:posterior map :ll scalar})

   :generate — prior returns posterior mean (always runs); obs reads
     :constraints, ll → :score + :weight.
   :regenerate — prior returns the old value and falls through (nil) when the
     site is selected; obs reads :old-choices, requires an initialized posterior,
     skips selected obs, ll → :score only.
   :update (genmlx-6hcu) — prior marginalizes when UNCONSTRAINED (a constrained
     prior re-opens the pair → fall through to the base transition, mirroring
     regenerate Case A); obs reads :constraints with fallback to :old-choices
     (new-over-old) so the full marginal LL is refolded over the merged obs,
     ll → :score + :weight; a CHANGED obs charges its old value into :discard.

   Returns {addr handler-fn} for prior + all obs addresses."
  [prior-addr obs-addrs init-posterior posterior-mean update-step mode]
  (let [regenerate? (= mode :regenerate)
        update? (= mode :update)
        prior-handler
        (fn [state addr dist]
          (let [proceed?
                (cond
                  regenerate?
                  (not (and (:selection state)
                            (sel/selected? (:selection state) addr)))
                  ;; Update (genmlx-6hcu): marginalize unless the prior latent is
                  ;; itself constrained — pinning it re-opens the pair (scored
                  ;; jointly), so fall through to the base transition. (The
                  ;; dispatcher also declines the whole analytical-update for such
                  ;; constraints; this is the per-pair backstop.)
                  update?
                  (not (cm/has-value? (cm/get-submap (:constraints state) prior-addr)))
                  ;; Generate/assess (genmlx-b470): the pair group marginalizes
                  ;; only when the prior itself is UNCONSTRAINED and EVERY obs
                  ;; is constrained. A constrained prior must keep its value
                  ;; (the base transition scores it); a partially-constrained
                  ;; obs set would otherwise produce a score that is neither
                  ;; joint nor marginal. Declining here leaves no posterior in
                  ;; state, so the obs handlers below fall through too.
                  :else
                  (let [cs (:constraints state)]
                    (and (not (cm/has-value? (cm/get-submap cs prior-addr)))
                         (every? #(cm/has-value? (cm/get-submap cs %)) obs-addrs))))]
            (when proceed?
              (let [posterior (init-posterior (:params dist))
                    value (if regenerate?
                            (cm/get-value (cm/get-submap (:old-choices state) addr))
                            (posterior-mean posterior))]
                ;; Return point estimate / old value; don't add to score (marginalized out)
                [value (-> state
                           (assoc-in [:auto-posteriors prior-addr] posterior)
                           (update :choices cm/set-value addr value))]))))

        obs-handler
        (fn [state addr dist]
          ;; Requires an initialized posterior — absent whenever the prior
          ;; handler fell through (selected under regenerate; constrained prior
          ;; or partially-constrained obs under generate) → base transition.
          (when-let [current-posterior (get-in state [:auto-posteriors prior-addr])]
            ;; Regenerate: skip obs that are themselves being resampled.
            (when-not (and regenerate? (:selection state)
                           (sel/selected? (:selection state) addr))
              (let [old-sub (cm/get-submap (:old-choices state) addr)
                    new-sub (cm/get-submap (:constraints state) addr)
                    ;; new-over-old: update reads the CHANGED obs from :constraints
                    ;; and UNCHANGED obs from :old-choices, so the full marginal LL
                    ;; is refolded; regenerate reads old, generate reads constraints.
                    constraint (cond
                                 regenerate? old-sub
                                 update?     (if (cm/has-value? new-sub) new-sub old-sub)
                                 :else       new-sub)]
                (when (cm/has-value? constraint)
                  (let [obs-value (cm/get-value constraint)
                        {:keys [posterior ll]} (update-step current-posterior obs-value (:params dist))
                        post-mean (posterior-mean posterior)
                        ;; update: a changed obs charges its old value to :discard
                        changed? (and update? (cm/has-value? new-sub) (cm/has-value? old-sub))]
                    [obs-value (cond-> state
                                 true (assoc-in [:auto-posteriors prior-addr] posterior)
                                 true (update :choices cm/set-value addr obs-value)
                                 true (update :score mx/add ll)
                                 (not regenerate?) (update :weight mx/add ll)
                                 changed? (update :discard cm/set-value addr (cm/get-value old-sub))
                                 ;; Update prior's value to posterior mean
                                 true (update :choices cm/set-value prior-addr post-mean))]))))))]
    (assoc (zipmap obs-addrs (repeat obs-handler)) prior-addr prior-handler)))

;; ---------------------------------------------------------------------------
;; Per-family configs
;; ---------------------------------------------------------------------------

;; Each family contributes the three closures that differ between conjugate
;; pairs: how to seed the posterior from the prior dist params, the posterior
;; point estimate, and the per-observation update. The handler control flow
;; (generate vs regenerate, score/weight accounting) is shared in
;; make-conjugate-handlers-core.
(def ^:private conjugate-family-specs
  {:normal-normal
   {:init-posterior (fn [{:keys [mu sigma]}] {:mean mu :var (mx/multiply sigma sigma)})
    :posterior-mean :mean
    :update-step (fn [posterior obs-value {:keys [sigma]}]
                   (let [obs-var (mx/multiply sigma sigma)
                         {:keys [mean var ll]} (nn-update-step posterior obs-value obs-var)]
                     {:posterior {:mean mean :var var} :ll ll}))}
   :normal-iid-normal
   {:init-posterior (fn [{:keys [mu sigma]}] {:mean mu :var (mx/multiply sigma sigma)})
    :posterior-mean :mean
    :update-step (fn [posterior obs-value {:keys [sigma]}]
                   (let [obs-var (mx/multiply sigma sigma)]
                     ;; nn-iid-update-step hard-codes the HOMOSCEDASTIC
                     ;; normal-iid-normal closed form (det Sigma =
                     ;; (s2)^(T-1)*(s2+T*tau2), etc.), treating obs-var as a
                     ;; SCALAR s2. A per-element [T] sigma (heteroscedastic
                     ;; iid-gaussian) is a different MVN marginal
                     ;; N(y; m0*1, diag(sigma_i^2)+tau2 11^T); the homoscedastic
                     ;; form silently mis-scores it as a [T]-shaped ll. Bail to
                     ;; the handler joint path, which scores the per-element sigma
                     ;; correctly via dist-log-prob :iid-gaussian (genmlx-symr).
                     ;; This is the correctness guarantee: it catches a [T] sigma
                     ;; however it is expressed, even when detect-conjugate-pairs'
                     ;; static gate could not prove the shape.
                     (when (seq (mx/shape obs-var))
                       (throw (ex-info "heteroscedastic [T]-sigma iid-gaussian: homoscedastic conjugate form invalid; bailing analytical elimination to the handler joint path"
                                       {:genmlx.analytical/bail true})))
                     (let [{:keys [mean var ll]} (nn-iid-update-step posterior obs-value obs-var)]
                       {:posterior {:mean mean :var var} :ll ll})))}
   :beta-bernoulli
   {:init-posterior (fn [{:keys [alpha beta-param]}] {:alpha alpha :beta beta-param})
    :posterior-mean (fn [{:keys [alpha beta]}] (mx/divide alpha (mx/add alpha beta)))
    :update-step (fn [posterior obs-value _params]
                   (let [{:keys [alpha beta ll]} (bb-update-step posterior obs-value)]
                     {:posterior {:alpha alpha :beta beta} :ll ll}))}
   :gamma-poisson
   {:init-posterior (fn [{:keys [shape-param rate]}] {:shape shape-param :rate rate})
    :posterior-mean (fn [{:keys [shape rate]}] (mx/divide shape rate))
    :update-step (fn [posterior obs-value _params]
                   (let [{:keys [shape rate ll]} (gp-update-step posterior obs-value)]
                     {:posterior {:shape shape :rate rate} :ll ll}))}
   :gamma-exponential
   {:init-posterior (fn [{:keys [shape-param rate]}] {:shape shape-param :rate rate})
    :posterior-mean (fn [{:keys [shape rate]}] (mx/divide shape rate))
    :update-step (fn [posterior obs-value _params]
                   (let [{:keys [shape rate ll]} (ge-update-step posterior obs-value)]
                     {:posterior {:shape shape :rate rate} :ll ll}))}
   ;; Dirichlet–Categorical (genmlx-cf0d). Prior theta ~ Dirichlet(alpha), obs
   ;; x ~ Categorical(theta) — written (dist/categorical (mx/log theta)) so the
   ;; logit-parameterized categorical scores log p(x=k)=log theta_k (conjugacy
   ;; detection accepts ONLY this log-link form). The latent value is the VECTOR
   ;; posterior mean alpha/sum(alpha). Per-obs predictive log-evidence is
   ;; log alpha_k - log(sum alpha); the posterior folds alpha <- alpha + e_k, so
   ;; the generic core's sequential fold of N obs yields the exact ordered
   ;; Dirichlet-multinomial sequence marginal.
   :dirichlet-categorical
   ;; Force float32: a synthesized model may pass an INT vector literal
   ;; (dist/dirichlet [1 2 3]); log/divide downstream need a float dtype.
   {:init-posterior (fn [{:keys [alpha]}] {:alpha (mx/ensure-array alpha mx/float32)})
    :posterior-mean (fn [{:keys [alpha]}] (mx/divide alpha (mx/sum alpha)))
    :update-step
    (fn [{:keys [alpha]} obs-value _params]
      (let [d (first (mx/shape alpha))
            dt (mx/dtype alpha)
            ;; one-hot indicator for the observed category, built at graph level
            ;; so it works for a JS-int OR an MLX-scalar obs-value:
            ;; (arange D == k) cast to alpha's dtype.
            onehot (mx/astype (mx/equal (mx/astype (mx/arange d) dt) obs-value) dt)
            sum-alpha (mx/sum alpha)
            alpha-k (mx/sum (mx/multiply alpha onehot))   ;; gather alpha_k
            ll (mx/subtract (mx/log alpha-k) (mx/log sum-alpha))
            alpha' (mx/add alpha onehot)]                 ;; alpha <- alpha + e_k
        {:posterior {:alpha alpha'} :ll ll}))}})

(defn- make-family-handlers
  "Build conjugate handlers for `family` from conjugate-family-specs, in `mode`."
  [family mode prior-addr obs-addrs]
  (let [{:keys [init-posterior posterior-mean update-step]} (conjugate-family-specs family)]
    (make-conjugate-handlers-core prior-addr obs-addrs
                                  init-posterior posterior-mean update-step mode)))

;; ---------------------------------------------------------------------------
;; Auto-Kalman handlers (for detected linear-Gaussian chains)
;; ---------------------------------------------------------------------------

(defn- resolve-affine-form
  "Resolve an affine coefficient/offset FORM from schema extraction to an MLX
   scalar for Kalman math: a number or an MLX array. Returns nil for symbolic
   forms (a model-arg symbol, an unevaluated op list) — the caller must bail
   or decline; substituting a guess silently mis-scores (genmlx-rmy7)."
  [x]
  (cond (number? x)   (mx/scalar x)
        (mx/array? x) x
        :else         nil))

(defn- kalman-predict-belief
  "Predict belief for step i given belief from step i-1:
     mean' = c * mean + b        (the transition offset b — a drift chain
                                  z' ~ N(z + b, q) — shifts the mean; it was
                                  silently dropped before, genmlx-rmy7)
     var'  = c^2 * var + q       (a constant offset does not move the variance)
   transition: the transition descriptor from steps[i-1]
   noise-var: process noise variance for step i.
   Throws the analytical bail (caught by the dispatcher, which redoes the op
   on the handler joint path) when the coefficient or offset is a symbolic
   form this runtime cannot evaluate — the genmlx-0e0j discipline; falling
   back per-site would mix marginal and plug-in scoring."
  [prev-belief transition noise-var]
  (let [direct? (= :direct (:type transition))
        coeff  (if direct? (mx/scalar 1.0) (resolve-affine-form (:coefficient transition)))
        offset (if direct? (mx/scalar 0.0) (resolve-affine-form (:offset transition)))]
    (when (or (nil? coeff) (nil? offset))
      (throw (ex-info "Kalman transition coefficient/offset unresolvable (symbolic); bailing analytical elimination to the handler joint path"
                      {:genmlx.analytical/bail true :transition transition})))
    {:mean (mx/add (mx/multiply coeff (:mean prev-belief)) offset)
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

(defn- resolve-loading-offset
  "Compute loading coefficient and offset for a Kalman observation dep type.
   Returns [loading offset] for :direct and :affine with numeric/MLX-array
   coefficient AND offset. Returns nil for :nonlinear or when EITHER form is
   symbolic (a model-arg symbol, an unevaluated op list) — the caller bails to
   the handler joint path. A symbolic offset used to be silently substituted
   with 0.0, mis-scoring every obs update on such a chain (genmlx-rmy7)."
  [dep-type]
  (case (:type dep-type)
    :direct [(mx/scalar 1.0) (mx/scalar 0.0)]
    :affine (let [c (resolve-affine-form (:coefficient dep-type))
                  o (resolve-affine-form (:offset dep-type))]
              (when (and c o) [c o]))
    ;; :nonlinear or unknown — can't handle analytically
    nil))

(defn- kalman-init-belief
  "Initialize or predict Kalman belief for step i."
  [i steps state params]
  (let [prev-step (when (pos? i) (nth steps (dec i)))
        noise-var (mx/multiply (:sigma params) (:sigma params))]
    (if-some [prev-belief (some->> (:latent prev-step)
                                   (get (:auto-kalman-beliefs state)))]
      (kalman-predict-belief prev-belief (:transition prev-step) noise-var)
      {:mean (:mu params) :var noise-var})))

(defn- kalman-obs-update
  "Pure Kalman observation update math.
   Returns {:ll :new-belief} given current belief, obs value, obs variance, dep type.
   Returns nil if dep-type is not resolvable (symbolic forms, nonlinear)."
  [belief obs-value obs-var dep-type]
  (when-let [[loading offset] (resolve-loading-offset dep-type)]
    (let [pred-obs (mx/add offset (mx/multiply loading (:mean belief)))
          innov (mx/subtract obs-value pred-obs)
          S (mx/add (mx/multiply loading (mx/multiply loading (:var belief))) obs-var)
          K (mx/divide (mx/multiply loading (:var belief)) S)
          ll (mx/multiply (mx/scalar -0.5)
                          (mx/add (mx/scalar LOG-2PI)
                                  (mx/add (mx/log S)
                                          (mx/divide (mx/multiply innov innov) S))))
          new-mean (mx/add (:mean belief) (mx/multiply K innov))
          new-var (mx/subtract (:var belief) (mx/multiply K (mx/multiply loading (:var belief))))]
      {:ll ll :new-belief {:mean new-mean :var new-var}})))

(defn- make-kalman-handlers-core
  "Build Kalman handlers parameterized by mode.
   :generate — latent returns predicted mean, obs reads :constraints, ll → :score + :weight
   :regenerate — latent checks selection/uses old-val, obs reads :old-choices, ll → :score only"
  [chain mode]
  (let [steps (:steps chain)
        n-steps (count steps)
        latent->idx (into {} (map-indexed (fn [i s] [(:latent s) i]) steps))
        obs-to-latent (into {}
                            (mapcat (fn [step]
                                      (map (fn [oa odt] [oa {:latent (:latent step) :dep-type odt}])
                                           (:observations step)
                                           (:obs-dep-types step)))
                                    steps))
        regenerate? (= mode :regenerate)
        chain-latents (mapv :latent steps)
        chain-obs (vec (mapcat :observations steps))
        chain-key (first chain-latents)
        ;; Generate/assess gate (genmlx-b470): the chain marginalizes only when
        ;; no chain latent is constrained and every chain obs is constrained.
        ;; A constrained latent must keep its value (base transition scores it);
        ;; a partially-constrained obs set would leave the remaining latents at
        ;; predicted means with no score — neither joint nor marginal. The
        ;; result is cached in state under [:auto-kalman-ok chain-key] once a
        ;; latent handler commits (the gate can only be consulted again on the
        ;; decline path, where handlers consistently fall through).
        chain-ok?
        (fn [state]
          (if regenerate?
            true
            (if-some [cached (get-in state [:auto-kalman-ok chain-key])]
              cached
              (let [cs (:constraints state)]
                (and (not-any? #(cm/has-value? (cm/get-submap cs %)) chain-latents)
                     (every? #(cm/has-value? (cm/get-submap cs %)) chain-obs))))))

        latent-handlers
        (into {}
              (map-indexed
               (fn [i step]
                 [(:latent step)
                  (fn [state addr dist]
                    (when-not (and regenerate? (:selection state)
                                   (sel/selected? (:selection state) addr))
                      (when (chain-ok? state)
                        (let [params (:params dist)
                              new-belief (kalman-init-belief i steps state params)
                              value (if regenerate?
                                      (cm/get-value (cm/get-submap (:old-choices state) addr))
                                      (:mean new-belief))
                              noise-var (mx/multiply (:sigma params) (:sigma params))]
                          [value (-> state
                                     (assoc-in [:auto-kalman-ok chain-key] true)
                                     (assoc-in [:auto-kalman-beliefs addr] new-belief)
                                     (assoc-in [:auto-kalman-noise-vars i] noise-var)
                                     (update :choices cm/set-value addr value))]))))])
               steps))

        obs-handlers
        (into {}
              (map
               (fn [[obs-addr {:keys [latent dep-type]}]]
                 [obs-addr
                  (fn [state addr dist]
                    (when-let [belief (get-in state [:auto-kalman-beliefs latent])]
                      (when-not (and regenerate? (:selection state)
                                     (sel/selected? (:selection state) addr))
                        (let [obs-value
                              (if regenerate?
                                (let [s (cm/get-submap (:old-choices state) addr)]
                                  (when (cm/has-value? s) (cm/get-value s)))
                                (let [s (cm/get-submap (:constraints state) addr)]
                                  (when (cm/has-value? s) (cm/get-value s))))]
                          (when obs-value
                            (let [obs-var (mx/multiply (:sigma (:params dist)) (:sigma (:params dist)))
                                  kres (kalman-obs-update belief obs-value obs-var dep-type)]
                              (when (nil? kres)
                                ;; genmlx-0e0j: the latent was already marginalized
                                ;; (belief present) but this Kalman obs update is
                                ;; unresolvable/ill-conditioned. Don't fall through
                                ;; to base (which scores the obs at the point
                                ;; estimate while the latent contributed 0). Bail to
                                ;; the handler joint path.
                                (throw (ex-info "Kalman obs update unresolvable; bailing analytical elimination to the handler joint path"
                                                {:genmlx.analytical/bail true :addr addr :latent latent})))
                              (let [{:keys [ll new-belief]} kres
                                    step-idx (get latent->idx latent)
                                      state' (cond-> (-> state
                                                         (assoc-in [:auto-kalman-beliefs latent] new-belief)
                                                         (update :choices cm/set-value addr obs-value)
                                                         (update :score mx/add ll)
                                                         (update :choices cm/set-value latent (:mean new-belief)))
                                               (not regenerate?)
                                               (update :weight mx/add ll))
                                      noise-vars (:auto-kalman-noise-vars state')
                                      state'' (if (and noise-vars (< (inc step-idx) n-steps))
                                                (cascade-predictions steps step-idx state' noise-vars)
                                                state')]
                                  [obs-value state''])))))))])
               obs-to-latent))]
    (merge latent-handlers obs-handlers)))

(defn make-auto-kalman-handlers
  "Build address-based handlers for a Kalman chain (generate mode)."
  [chain]
  (make-kalman-handlers-core chain :generate))

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
   Returns false if any diagonal element is below 1e-6, is non-finite, or the
   check itself fails (nil/malformed matrix).

   NOTE: the mx/item here forces a GPU eval inside a handler transition — a
   deliberate exception to the eval-at-boundaries rule. It is safe because the
   analytical path is scalar-only (never batched) and the dispatcher excludes
   mx/in-grad?; declining (false) on any anomaly falls through to the base
   transition, which is always correct."
  [cov-matrix]
  (try
    (let [diag-vals (mx/diag cov-matrix)
          min-diag (mx/item (mx/amin diag-vals))]
      (and (js/isFinite min-diag) (> min-diag 1e-6)))
    (catch :default _ false)))

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
            ;; Posterior via Kalman gain form (avoids 3x mx/inv):
            ;; K = S0 * M^{-1}, m1 = m0 + K*(y-m0), S1 = S0 - K*S0
            ;; Use mx/solve instead of mx/inv for numerical stability
            M-inv-S0 (mx/solve M cov-matrix) ;; M^{-1} * S0, shape [d,d]
            S0-M-inv-diff (mx/flatten ;; S0 * M^{-1} * (y-m0)
                           (mx/matmul cov-matrix (mx/reshape M-inv-diff [d 1])))
            m1 (mx/add mean-vec S0-M-inv-diff)
            S1 (mx/subtract cov-matrix (mx/matmul cov-matrix M-inv-S0))]
        {:mean-vec m1 :cov-matrix S1 :ll ll}))))

(defn- make-mvn-handlers-core
  "Build address-based handlers for a MVN-Normal conjugate pair, parameterized
   by `mode` (:generate or :regenerate) — mirroring make-conjugate-handlers-core.
   The MVN update keeps its ill-conditioned fallthrough (nil from mvn-update-step)."
  [prior-addr obs-addrs mode]
  (let [regenerate? (= mode :regenerate)
        prior-handler
        (fn [state addr dist]
          (let [proceed?
                (if regenerate?
                  (not (and (:selection state)
                            (sel/selected? (:selection state) addr)))
                  ;; Generate/assess (genmlx-b470): marginalize only when the
                  ;; prior is unconstrained and every obs is constrained —
                  ;; mirrors make-conjugate-handlers-core.
                  (let [cs (:constraints state)]
                    (and (not (cm/has-value? (cm/get-submap cs prior-addr)))
                         (every? #(cm/has-value? (cm/get-submap cs %)) obs-addrs))))]
            (when proceed?
              (let [{{:keys [mean-vec cov-matrix]} :params} dist
                    posterior {:mean-vec mean-vec :cov-matrix cov-matrix}
                    value (if regenerate?
                            (cm/get-value (cm/get-submap (:old-choices state) addr))
                            mean-vec)]
                ;; Return prior mean / old value; don't add to score (marginalized out)
                [value (-> state
                           (assoc-in [:auto-posteriors prior-addr] posterior)
                           (update :choices cm/set-value addr value))]))))

        obs-handler
        (fn [state addr dist]
          ;; Requires an initialized posterior — absent whenever the prior
          ;; handler fell through → base transition.
          (when-let [cur-post (get-in state [:auto-posteriors prior-addr])]
            (when-not (and regenerate? (:selection state)
                           (sel/selected? (:selection state) addr))
              (let [constraint (if regenerate?
                                 (cm/get-submap (:old-choices state) addr)
                                 (cm/get-submap (:constraints state) addr))]
                (when (cm/has-value? constraint)
                  (let [obs-value (cm/get-value constraint)
                        {{obs-cov :cov-matrix} :params} dist
                        result (mvn-update-step cur-post obs-value obs-cov)]
                    (if result
                      (let [{:keys [mean-vec cov-matrix ll]} result]
                        [obs-value (cond-> state
                                     true (assoc-in [:auto-posteriors prior-addr]
                                                    {:mean-vec mean-vec :cov-matrix cov-matrix})
                                     true (update :choices cm/set-value addr obs-value)
                                     true (update :score mx/add ll)
                                     (not regenerate?) (update :weight mx/add ll)
                                     true (update :choices cm/set-value prior-addr mean-vec))])
                      ;; genmlx-0e0j: the prior was already marginalized (cur-post
                      ;; present) but this obs update is ill-conditioned. Falling
                      ;; through to the base transition would score the obs at the
                      ;; prior-mean point estimate while the prior contributed 0 —
                      ;; a hybrid weight that is neither the true joint marginal
                      ;; nor a valid importance weight. Bail the whole op to the
                      ;; handler joint path (caught in dynamic/run-dispatched*).
                      (throw (ex-info "MVN obs update ill-conditioned; bailing analytical elimination to the handler joint path"
                                      {:genmlx.analytical/bail true :addr addr :prior prior-addr})))))))))]
    (assoc (zipmap obs-addrs (repeat obs-handler)) prior-addr prior-handler)))

;; ---------------------------------------------------------------------------
;; Factory dispatch
;; ---------------------------------------------------------------------------

(defn- make-auto-handlers-for
  "Generate-mode handler factory for `family`: a (fn [prior-addr obs-addrs])."
  [family]
  (if (= family :mvn-normal)
    (fn [prior-addr obs-addrs] (make-mvn-handlers-core prior-addr obs-addrs :generate))
    (fn [prior-addr obs-addrs] (make-family-handlers family :generate prior-addr obs-addrs))))

(def family->handler-factory
  "Map from conjugate family keyword to handler factory function."
  {:normal-normal (make-auto-handlers-for :normal-normal)
   :normal-iid-normal (make-auto-handlers-for :normal-iid-normal)
   :beta-bernoulli (make-auto-handlers-for :beta-bernoulli)
   :gamma-poisson (make-auto-handlers-for :gamma-poisson)
   :gamma-exponential (make-auto-handlers-for :gamma-exponential)
   :dirichlet-categorical (make-auto-handlers-for :dirichlet-categorical)
   :mvn-normal (make-auto-handlers-for :mvn-normal)})

(defn build-auto-handlers
  "Build address-based handlers from detected conjugate pairs.
   Multi-parent obs pairs are dropped first (handler-map merge is last-wins,
   so two priors claiming one obs would silently mis-marginalize, genmlx-b470).
   Returns a merged map of {addr handler-fn} for all conjugate sites."
  [conjugate-pairs]
  (let [grouped (conj/group-by-prior
                 (conj/drop-mixed-family-priors
                  (conj/drop-multi-parent-pairs conjugate-pairs)))]
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

;; Regenerate handlers reuse the same per-family specs and mode-parameterized
;; cores as generate (see make-conjugate-handlers-core / make-mvn-handlers-core);
;; only the :regenerate mode flag differs.

(defn- make-regenerate-handlers-for
  "Regenerate-mode handler factory for `family`: a (fn [prior-addr obs-addrs])."
  [family]
  (if (= family :mvn-normal)
    (fn [prior-addr obs-addrs] (make-mvn-handlers-core prior-addr obs-addrs :regenerate))
    (fn [prior-addr obs-addrs] (make-family-handlers family :regenerate prior-addr obs-addrs))))

(def regenerate-family->handler-factory
  "Map from conjugate family keyword to regenerate-specific handler factory."
  {:normal-normal (make-regenerate-handlers-for :normal-normal)
   :normal-iid-normal (make-regenerate-handlers-for :normal-iid-normal)
   :beta-bernoulli (make-regenerate-handlers-for :beta-bernoulli)
   :gamma-poisson (make-regenerate-handlers-for :gamma-poisson)
   :gamma-exponential (make-regenerate-handlers-for :gamma-exponential)
   :dirichlet-categorical (make-regenerate-handlers-for :dirichlet-categorical)
   :mvn-normal (make-regenerate-handlers-for :mvn-normal)})

(defn build-regenerate-handlers
  "Build regenerate-specific address-based handlers from detected conjugate pairs.
   Multi-parent obs pairs are dropped first (see build-auto-handlers).
   Returns a merged map of {addr handler-fn}."
  [conjugate-pairs]
  (let [grouped (conj/group-by-prior
                 (conj/drop-mixed-family-priors
                  (conj/drop-multi-parent-pairs conjugate-pairs)))]
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

(defn- make-regenerate-kalman-handlers
  "Build regenerate-specific Kalman handlers (regenerate mode)."
  [chain]
  (make-kalman-handlers-core chain :regenerate))

(defn- build-regenerate-kalman-handlers
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
   Mirrors the rewrite engine's logic: builds regenerate-specific handlers
   for both chain and non-chain pairs.
   Optional :chains kwarg avoids re-detecting Kalman chains (reuse from plan).
   Returns merged {addr handler-fn}."
  [conjugate-pairs & {:keys [chains]}]
  (let [chains (or chains (affine/detect-kalman-chains conjugate-pairs))
        kalman-handlers (build-regenerate-kalman-handlers chains)
        kalman-addrs (set (concat (mapcat :latent-addrs chains)
                                  (mapcat :obs-addrs chains)))
        remaining (remove (fn [p]
                            (or (contains? kalman-addrs (:prior-addr p))
                                (contains? kalman-addrs (:obs-addr p))))
                          conjugate-pairs)
        non-chain-handlers (build-regenerate-handlers remaining)]
    (merge kalman-handlers non-chain-handlers)))

;; ---------------------------------------------------------------------------
;; Update-specific handlers (genmlx-6hcu) — SCALAR conjugate families only.
;; MVN and Kalman analytical UPDATE are not implemented; models containing them
;; decline the analytical-update path entirely (at schema construction) and use
;; the joint handler path, so no factory is registered for those families here.
;; ---------------------------------------------------------------------------

(defn- make-update-handlers-for
  "Update-mode handler factory for `family`: a (fn [prior-addr obs-addrs])."
  [family]
  (fn [prior-addr obs-addrs] (make-family-handlers family :update prior-addr obs-addrs)))

(def update-family->handler-factory
  "Map from conjugate family keyword to update-specific handler factory.
   No :mvn-normal — MVN analytical update is unimplemented (decline to joint)."
  {:normal-normal     (make-update-handlers-for :normal-normal)
   :normal-iid-normal (make-update-handlers-for :normal-iid-normal)
   :beta-bernoulli    (make-update-handlers-for :beta-bernoulli)
   :gamma-poisson     (make-update-handlers-for :gamma-poisson)
   :gamma-exponential (make-update-handlers-for :gamma-exponential)})

(defn build-update-handlers
  "Build update-specific address-based handlers from conjugate pairs (scalar
   families only). Multi-parent obs pairs are dropped first (see
   build-auto-handlers). Pairs whose family has no update factory (e.g. MVN)
   are skipped — callers must decline the analytical-update path for models that
   contain them. Returns a merged map of {addr handler-fn}."
  [conjugate-pairs]
  (let [grouped (conj/group-by-prior
                 (conj/drop-mixed-family-priors
                  (conj/drop-multi-parent-pairs conjugate-pairs)))]
    (reduce
     (fn [handlers [prior-addr pairs]]
       (let [family (:family (first pairs))
             factory (get update-family->handler-factory family)
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
  "Check if any observation site in a conjugate pair is actually constrained
   AND its prior is NOT constrained. If both prior and obs are constrained,
   there's nothing to marginalize — the analytical handler would compute
   the marginal likelihood instead of the correct joint log-prob."
  [conjugate-pairs constraints]
  (boolean
   (some (fn [{:keys [prior-addr obs-addr]}]
           (and (cm/has-value? (cm/get-submap constraints obs-addr))
                (not (cm/has-value? (cm/get-submap constraints prior-addr)))))
         conjugate-pairs)))
