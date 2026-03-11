(ns genmlx.rewrite
  "Algebraic graph rewriting engine for Level 3 analytical elimination.

   Transforms a model's probabilistic graph by replacing conjugate pairs
   with deterministic posterior computations and marginal likelihood terms.
   The output is the minimal stochastic computation — only sites that
   cannot be eliminated analytically remain.

   Layered simplification:
   1. Kalman chains → collapse linear-Gaussian sequences
   2. Conjugacy elimination → merge conjugate pairs into closed form
   3. Rao-Blackwellization → sample from posterior (not prior) for shared priors

   Level 3 — WP-5: Algebraic Graph Rewriting."
  (:require [genmlx.dep-graph :as dep-graph]
            [genmlx.affine :as affine]
            [genmlx.conjugacy :as conj]
            [genmlx.inference.auto-analytical :as auto]
            [genmlx.handler :as h]))

;; ---------------------------------------------------------------------------
;; Rewrite rule protocol
;; ---------------------------------------------------------------------------

(defprotocol IRewriteRule
  "A rewrite rule that can eliminate or simplify stochastic nodes."
  (-applicable? [this graph schema]
    "Can this rule be applied to the current graph?")
  (-apply [this graph schema constraints]
    "Apply the rule. Returns {:graph' :handlers :eliminated :description}"))

;; ---------------------------------------------------------------------------
;; ConjugacyRule — eliminates conjugate prior via marginalization
;; ---------------------------------------------------------------------------

(defrecord ConjugacyRule [family prior-addr obs-addrs]
  IRewriteRule
  (-applicable? [this graph schema]
    (and (contains? (:nodes graph) prior-addr)
         (every? #(contains? (:nodes graph) %) obs-addrs)))
  (-apply [this graph schema constraints]
    (let [factory (get auto/family->handler-factory family)
          handlers (when factory (factory prior-addr obs-addrs))
          graph' (-> graph
                   (update :nodes disj prior-addr)
                   (update :edges (fn [edges]
                                    (into #{} (remove (fn [[a _]] (= a prior-addr))) edges))))]
      {:graph' graph'
       :handlers (or handlers {})
       :eliminated #{prior-addr}
       :description (str "Marginalized " (name prior-addr)
                         " via " (name family))})))

;; ---------------------------------------------------------------------------
;; KalmanRule — collapses linear-Gaussian chains
;; ---------------------------------------------------------------------------

(defrecord KalmanRule [chain]
  IRewriteRule
  (-applicable? [this graph schema]
    (let [latents (:latent-addrs chain)]
      (every? #(contains? (:nodes graph) %) latents)))
  (-apply [this graph schema constraints]
    (let [handlers (auto/make-auto-kalman-handlers chain)
          latents (set (:latent-addrs chain))
          graph' (-> graph
                   (update :nodes #(reduce disj % latents))
                   (update :edges (fn [edges]
                                    (into #{} (remove (fn [[a _]] (contains? latents a))) edges))))]
      {:graph' graph'
       :handlers handlers
       :eliminated latents
       :description (str "Kalman chain: "
                         (pr-str (:latent-addrs chain)))})))

;; ---------------------------------------------------------------------------
;; RaoBlackwellRule — sample from posterior instead of prior
;; ---------------------------------------------------------------------------

(defrecord RaoBlackwellRule [prior-addr conjugate-obs-addrs non-conjugate-children family]
  IRewriteRule
  (-applicable? [this graph schema]
    (and (contains? (:nodes graph) prior-addr)
         (seq conjugate-obs-addrs)
         (seq non-conjugate-children)))
  (-apply [this graph schema constraints]
    ;; Don't eliminate the prior — replace its handler so conjugate obs
    ;; contribute marginal LL and update the prior's value to posterior mean.
    ;; NOTE: Currently returns posterior MEAN (deterministic), not a sample
    ;; from the posterior distribution. Weights are correct via marginal LL,
    ;; but true posterior sampling (full Rao-Blackwell variance reduction)
    ;; requires deferred execution (Level 4 enhancement).
    (let [factory (get auto/family->handler-factory family)
          ;; Build handlers for the conjugate obs subset
          handlers (when factory (factory prior-addr conjugate-obs-addrs))]
      {:graph' graph  ;; Graph unchanged (prior still sampled)
       :handlers (or handlers {})
       :eliminated #{}  ;; Nothing eliminated, but variance reduced
       :description (str "Rao-Blackwellized " (name prior-addr))})))

;; ---------------------------------------------------------------------------
;; Rule generation from schema
;; ---------------------------------------------------------------------------

(defn- find-non-conjugate-children
  "Find children of prior-addr that are NOT in the conjugate obs set."
  [schema prior-addr conjugate-obs-addrs]
  (let [conj-set (set conjugate-obs-addrs)
        all-sites (:trace-sites schema)]
    (vec (keep (fn [site]
                 (when (and (contains? (:deps site) prior-addr)
                            (not (contains? conj-set (:addr site))))
                   (:addr site)))
               all-sites))))

(defn generate-rewrite-rules
  "Generate rewrite rules from schema conjugacy metadata.
   Each detected conjugate pair becomes a ConjugacyRule.
   Detected Kalman chains become KalmanRules.
   Shared priors with non-conjugate children become RaoBlackwellRules.

   Priority: Kalman > Conjugacy > RaoBlackwell
   (more structure eliminated first)"
  [schema conjugate-pairs]
  (let [;; Detect Kalman chains
        chains (affine/detect-kalman-chains conjugate-pairs)

        kalman-rules (mapv ->KalmanRule chains)

        ;; Addresses already claimed by Kalman chains
        kalman-latents (set (mapcat :latent-addrs chains))
        kalman-obs (set (mapcat :obs-addrs chains))
        kalman-addrs (into kalman-latents kalman-obs)

        ;; Group remaining conjugate pairs by prior (excluding Kalman-claimed)
        remaining-pairs (remove (fn [p]
                                  (or (contains? kalman-addrs (:prior-addr p))
                                      (contains? kalman-addrs (:obs-addr p))))
                                conjugate-pairs)
        grouped (group-by :prior-addr remaining-pairs)

        ;; For each grouped prior, check if it has non-conjugate children
        conjugacy-rules
        (vec
          (keep
            (fn [[prior-addr pairs]]
              (let [family (:family (first pairs))
                    obs-addrs (mapv :obs-addr pairs)
                    non-conj (find-non-conjugate-children schema prior-addr obs-addrs)]
                (if (seq non-conj)
                  ;; Shared prior — use RaoBlackwell instead
                  nil
                  ;; Pure conjugate — can eliminate
                  (->ConjugacyRule family prior-addr obs-addrs))))
            grouped))

        ;; RaoBlackwell rules for shared priors
        rb-rules
        (vec
          (keep
            (fn [[prior-addr pairs]]
              (let [family (:family (first pairs))
                    obs-addrs (mapv :obs-addr pairs)
                    non-conj (find-non-conjugate-children schema prior-addr obs-addrs)]
                (when (seq non-conj)
                  (->RaoBlackwellRule prior-addr obs-addrs non-conj family))))
            grouped))]

    ;; Priority ordering: Kalman first, then conjugacy, then RB
    (vec (concat kalman-rules conjugacy-rules rb-rules))))

;; ---------------------------------------------------------------------------
;; Rewrite engine
;; ---------------------------------------------------------------------------

(defn apply-rewrites
  "Progressively apply rewrite rules to simplify the model graph.
   Each rule eliminates stochastic nodes and adds handler contributions.

   Returns {:residual-graph   — graph of sites that must be sampled
            :handlers         — merged address-based handlers for eliminated sites
            :eliminated       — set of addresses that were analytically eliminated
            :rewrite-log      — sequence of applied rewrites (for debugging)}"
  [graph schema constraints rules]
  (reduce
    (fn [{:keys [residual-graph handlers eliminated rewrite-log] :as acc} rule]
      (if (-applicable? rule residual-graph schema)
        (let [{:keys [graph' handlers eliminated description]}
              (-apply rule residual-graph schema constraints)]
          (-> acc
            (assoc :residual-graph graph')
            (update :handlers merge handlers)
            (update :eliminated into eliminated)
            (update :rewrite-log conj description)))
        acc))
    {:residual-graph graph
     :handlers {}
     :eliminated #{}
     :rewrite-log []}
    rules))

;; ---------------------------------------------------------------------------
;; Integration: build analytical plan
;; ---------------------------------------------------------------------------

(defn build-analytical-plan
  "Analyze model and build the optimal analytical execution plan.
   schema must already have :conjugate-pairs (from augment-schema-with-conjugacy).

   Returns {:rewrite-result :auto-transition :stats}"
  [schema]
  (let [pairs (or (:conjugate-pairs schema) [])
        graph (dep-graph/build-dep-graph schema)
        rules (generate-rewrite-rules schema pairs)
        result (apply-rewrites graph schema nil rules)
        auto-transition (when (seq (:handlers result))
                          (auto/make-address-dispatch
                            h/generate-transition
                            (:handlers result)))]
    {:rewrite-result result
     :auto-transition auto-transition
     :stats {:total-sites (count (:nodes graph))
             :eliminated (count (:eliminated result))
             :residual (count (:nodes (:residual-graph result)))
             :rewrites-applied (count (:rewrite-log result))}}))
