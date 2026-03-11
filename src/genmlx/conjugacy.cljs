(ns genmlx.conjugacy
  "Conjugacy detection for Level 3 automatic analytical elimination.

   Analyzes schema trace sites to detect conjugate prior-likelihood pairs.
   Works with standard distributions (dist/gaussian, dist/beta-dist, etc.)
   — no special distribution types required.

   Detection is purely static: walks schema trace sites, checks dist-type
   pairs against the conjugacy table, and classifies dependencies.

   The output — :conjugate-pairs on the schema — drives WP-1's address-based
   analytical handlers and WP-2's auto-wiring in DynamicGF."
  (:require [genmlx.affine :as affine]))

;; ---------------------------------------------------------------------------
;; Conjugacy table
;; ---------------------------------------------------------------------------

(def conjugacy-table
  "Known conjugate prior-likelihood families.
   Key: [prior-dist-type obs-dist-type]
   Value: {:family keyword, param access keys for runtime, natural-param-idx
           for static analysis of which dist-arg position is the natural parameter}.

   nil values = explicitly NOT conjugate (prevents false positives)."
  {[:gaussian :gaussian]
   {:family :normal-normal
    :prior-mean-key :mu        ;; key in prior's (:params dist) for mean
    :prior-std-key :sigma      ;; key in prior's (:params dist) for std
    :obs-mean-key :mu          ;; key in obs's (:params dist) for mean
    :obs-noise-key :sigma      ;; key in obs's (:params dist) for noise std
    :natural-param-idx 0}      ;; which dist-arg index is the natural parameter

   [:beta-dist :bernoulli]
   {:family :beta-bernoulli
    :prior-alpha-key :alpha
    :prior-beta-key :beta-param
    :obs-prob-key :p
    :natural-param-idx 0}

   [:gamma-dist :poisson]
   {:family :gamma-poisson
    :prior-shape-key :shape-param
    :prior-rate-key :rate
    :obs-rate-key :rate
    :natural-param-idx 0}

   [:gamma-dist :exponential]
   {:family :gamma-exponential
    :prior-shape-key :shape-param
    :prior-rate-key :rate
    :obs-rate-key :rate
    :natural-param-idx 0}

   [:dirichlet :categorical]
   {:family :dirichlet-categorical
    :prior-alpha-key :alpha
    :obs-logits-key :logits
    :natural-param-idx 0}

   ;; Explicitly NOT conjugate
   [:gaussian :bernoulli] nil
   [:beta-dist :gaussian] nil
   [:gaussian :poisson] nil})

;; ---------------------------------------------------------------------------
;; Dependency classification
;; ---------------------------------------------------------------------------

(defn- symbol-resolves-to-addr?
  "Check if a source form symbol could resolve to a given trace address.
   In the schema, when a let binding captures a trace result:
     (let [mu (trace :mu ...)] ...)
   the env maps symbol 'mu to deps #{:mu}. The dist-args for downstream
   traces contain the symbol 'mu. We check if the symbol name matches
   the address name (keyword)."
  [sym addr]
  (and (symbol? sym)
       (= (name sym) (name addr))))

(defn classify-dependency
  "Classify how an obs site depends on a prior site's value.
   Returns {:type :direct|:affine|:nonlinear} with metadata.

   :direct — the prior's value appears directly as the natural parameter
             argument (e.g., mu is position 0 in (dist/gaussian mu sigma))
   :affine — the natural parameter is an affine function of the prior
             (e.g., (mx/multiply 0.9 mu)), with :coefficient and :offset
   :nonlinear — dependency exists but through a non-trivial expression"
  [prior-addr obs-site family-info]
  (let [natural-idx (:natural-param-idx family-info)
        dist-args (:dist-args obs-site)
        natural-arg (when (and dist-args (< natural-idx (count dist-args)))
                      (nth dist-args natural-idx))]
    (cond
      ;; Direct: the natural parameter arg IS the prior symbol
      (symbol-resolves-to-addr? natural-arg prior-addr)
      {:type :direct}

      ;; Try affine analysis on the natural parameter expression
      :else
      (affine/classify-affine-dependency prior-addr obs-site natural-idx))))

;; ---------------------------------------------------------------------------
;; Conjugate pair detection
;; ---------------------------------------------------------------------------

(defn detect-conjugate-pairs
  "Scan schema trace sites for conjugate pairs.

   A pair (prior, obs) is conjugate if:
   1. Both sites are static (keyword addresses)
   2. obs depends on prior (prior-addr is in obs's :deps)
   3. [prior-dist-type, obs-dist-type] is in conjugacy-table with non-nil value
   4. The dependency is through the natural parameter position (:direct)

   Returns vector of {:prior-addr :obs-addr :family :prior-site :obs-site
                       :dependency-type :family-info}"
  [schema]
  (let [sites (:trace-sites schema)
        static-sites (filter :static? sites)
        site-map (into {} (map (juxt :addr identity)) static-sites)]
    (vec
      (for [obs-site static-sites
            prior-addr (:deps obs-site)
            :let [prior-site (get site-map prior-addr)]
            :when prior-site
            :let [table-key [(:dist-type prior-site) (:dist-type obs-site)]
                  family-info (get conjugacy-table table-key ::not-found)]
            ;; Must be in table AND non-nil (nil = explicitly not conjugate)
            :when (and (not= family-info ::not-found) (some? family-info))
            :let [dep-type (classify-dependency prior-addr obs-site family-info)]
            ;; :direct and :affine dependencies are conjugate
            :when (#{:direct :affine} (:type dep-type))]
        {:prior-addr prior-addr
         :obs-addr (:addr obs-site)
         :family (:family family-info)
         :prior-site prior-site
         :obs-site obs-site
         :dependency-type dep-type
         :family-info family-info}))))

;; ---------------------------------------------------------------------------
;; Grouping
;; ---------------------------------------------------------------------------

(defn group-by-prior
  "Group conjugate pairs by their prior address.
   Multiple observations of the same prior is a common pattern.
   Returns {prior-addr [pair1 pair2 ...]}."
  [pairs]
  (group-by :prior-addr pairs))

;; ---------------------------------------------------------------------------
;; Schema augmentation
;; ---------------------------------------------------------------------------

(defn augment-schema-with-conjugacy
  "Add conjugacy metadata to an extracted schema.
   Attaches :conjugate-pairs and :has-conjugate? to the schema map."
  [schema]
  (let [pairs (detect-conjugate-pairs schema)]
    (assoc schema
           :conjugate-pairs pairs
           :has-conjugate? (boolean (seq pairs)))))
