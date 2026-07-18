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
  (:require [clojure.set :as set]
            [genmlx.dep-graph :as dep-graph]
            [genmlx.affine :as affine]
            [genmlx.conjugacy :as conj-detect]
            [genmlx.inference.auto-analytical :as auto]
            [genmlx.linear-gaussian :as lg]
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

(defn- prune-eliminated
  "Remove the `eliminated` address set from the graph's nodes and drop any
   edge originating at an eliminated node."
  [graph eliminated]
  (-> graph
      (update :nodes set/difference eliminated)
      (update :edges (fn [edges]
                       (into #{} (remove (fn [[a _]] (contains? eliminated a))) edges)))))

;; ---------------------------------------------------------------------------
;; ConjugacyRule — eliminates conjugate prior via marginalization
;; ---------------------------------------------------------------------------

(defrecord ConjugacyRule [family prior-addr obs-addrs]
  IRewriteRule
  (-applicable? [this graph schema]
    ;; Defensive guard: a family present in the conjugacy table but absent from
    ;; the handler factory map has NO runtime elimination — applying the rule
    ;; anyway would prune a still-stochastic latent from the graph and let
    ;; method-selection claim an exact marginal (genmlx-b470). Every table family
    ;; is currently wired (dirichlet-categorical since genmlx-cf0d); this keeps
    ;; any future detection-only entry from silently mis-eliminating.
    (and (some? (get auto/family->handler-factory family))
         (contains? (:nodes graph) prior-addr)
         (every? #(contains? (:nodes graph) %) obs-addrs)))
  (-apply [this graph schema constraints]
    (let [factory (get auto/family->handler-factory family)
          handlers (factory prior-addr obs-addrs)
          graph' (prune-eliminated graph #{prior-addr})]
      {:graph' graph'
       :handlers handlers
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
          graph' (prune-eliminated graph latents)]
      {:graph' graph'
       :handlers handlers
       :eliminated latents
       :description (str "Kalman chain: "
                         (pr-str (:latent-addrs chain)))})))

;; ---------------------------------------------------------------------------
;; RegressionRule — eliminates a coupled/affine linear-Gaussian block jointly
;; ---------------------------------------------------------------------------

(defrecord RegressionRule [block]
  IRewriteRule
  (-applicable? [this graph schema]
    (every? #(contains? (:nodes graph) %) (:latent-addrs block)))
  (-apply [this graph schema constraints]
    (let [handlers (lg/make-lg-handlers block)
          latents (set (:latent-addrs block))
          graph' (prune-eliminated graph latents)]
      {:graph' graph'
       :handlers handlers
       :eliminated latents
       :description (str "Linear-Gaussian block: " (pr-str (:latent-addrs block)))})))

;; ---------------------------------------------------------------------------
;; RaoBlackwellRule — sample from posterior instead of prior
;; ---------------------------------------------------------------------------

(defrecord RaoBlackwellRule [prior-addr conjugate-obs-addrs non-conjugate-children family]
  IRewriteRule
  (-applicable? [this graph schema]
    (and (some? (get auto/family->handler-factory family))
         (contains? (:nodes graph) prior-addr)
         (seq conjugate-obs-addrs)
         (seq non-conjugate-children)))
  (-apply [this graph schema constraints]
    ;; Don't eliminate the prior — replace its handler so conjugate obs
    ;; contribute marginal LL and update the prior's value to posterior mean.
    ;; NOTE: Currently returns posterior MEAN (deterministic), not a sample
    ;; from the posterior distribution. Weights are exact via marginal LL only
    ;; when every CONSTRAINED child is conjugate; with constrained
    ;; NON-conjugate children the generate weight is a plug-in approximation —
    ;; those children are scored at the deterministic posterior mean with no
    ;; correction term (genmlx-vluz). Particle/MCMC drivers strip the
    ;; analytical path (genmlx-540f), so this reaches direct p/generate
    ;; consumers only. True posterior sampling (full Rao-Blackwell variance
    ;; reduction) requires deferred execution (Level 4 enhancement).
    (let [factory (get auto/family->handler-factory family)
          ;; Build handlers for the conjugate obs subset
          handlers (factory prior-addr conjugate-obs-addrs)]
      {:graph' graph  ;; Graph unchanged (prior still sampled)
       :handlers handlers
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
    (into [] (keep (fn [site]
                     (when (and (contains? (:deps site) prior-addr)
                                (not (contains? conj-set (:addr site))))
                       (:addr site))))
          all-sites)))

(defn generate-rewrite-rules
  "Generate rewrite rules from schema conjugacy metadata.
   Each detected conjugate pair becomes a ConjugacyRule.
   Detected Kalman chains become KalmanRules.
   Coupled/affine linear-Gaussian blocks become RegressionRules.
   Shared priors with non-conjugate children become RaoBlackwellRules.

   Priority: Kalman > Regression > Conjugacy > RaoBlackwell
   (more structure eliminated first).

   lg-blocks: linear-Gaussian block descriptors (jointly eliminated).
   declined-addrs: concern-component addresses that cannot be exactly eliminated
   — excluded so no per-prior ConjugacyRule claims a wrong exact.

   Backward-compatible arities: [schema pairs] and [schema pairs chains] omit
   linear-Gaussian blocks (no RegressionRules)."
  ([schema conjugate-pairs] (generate-rewrite-rules schema conjugate-pairs nil nil nil))
  ([schema conjugate-pairs chains] (generate-rewrite-rules schema conjugate-pairs chains nil nil))
  ([schema conjugate-pairs chains lg-blocks declined-addrs]
  (let [kalman-rules (mapv ->KalmanRule chains)
        regression-rules (mapv ->RegressionRule lg-blocks)

        ;; Addresses already claimed by Kalman chains / regression blocks, plus
        ;; declined concern-components (kept off the scalar conjugacy path).
        kalman-latents (set (mapcat :latent-addrs chains))
        kalman-obs (set (mapcat :obs-addrs chains))
        claimed (set/union (into kalman-latents kalman-obs)
                           (into #{} (mapcat :all-addrs lg-blocks))
                           (or declined-addrs #{}))

        ;; Group remaining conjugate pairs by prior (excluding claimed addrs).
        ;; Multi-parent obs (one obs claimed by several priors) have no correct
        ;; scalar elimination — drop those pairs entirely (genmlx-b470).
        ;; Decline priors conjugate to >1 obs family (no correct single-family
        ;; scalar elimination — genmlx-1thx) in addition to multi-parent obs.
        remaining-pairs (conj-detect/drop-mixed-family-priors
                         (conj-detect/drop-multi-parent-pairs
                          (vec (remove (fn [p]
                                         (or (contains? claimed (:prior-addr p))
                                             (contains? claimed (:obs-addr p))))
                                       conjugate-pairs))))
        grouped (group-by :prior-addr remaining-pairs)

        ;; One pass over grouped priors: compute non-conjugate children once and
        ;; classify each prior as a pure-conjugate elimination (no non-conjugate
        ;; children) or a Rao-Blackwell (shared prior with non-conjugate children).
        classified
        (mapv
          (fn [[prior-addr pairs]]
            (let [family (:family (first pairs))
                  obs-addrs (mapv :obs-addr pairs)
                  non-conj (find-non-conjugate-children schema prior-addr obs-addrs)]
              (if (seq non-conj)
                {:rb (->RaoBlackwellRule prior-addr obs-addrs non-conj family)}
                {:conjugacy (->ConjugacyRule family prior-addr obs-addrs)})))
          grouped)]

    ;; Priority ordering: Kalman, then Regression, then conjugacy, then RB
    (vec (concat kalman-rules
                 regression-rules
                 (keep :conjugacy classified)
                 (keep :rb classified))))))

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
   source (optional) is the gen source form — required to detect linear-Gaussian
   regression blocks (their design matrix is recovered from the obs mean forms).

   Returns {:rewrite-result :auto-transition :kalman-chains :lg-blocks
            :declined-addrs :stats}"
  ([schema] (build-analytical-plan schema nil))
  ([schema source]
   (let [pairs (or (:conjugate-pairs schema) [])
         graph (dep-graph/build-dep-graph schema)
         chains (affine/detect-kalman-chains pairs)
         kalman-addrs (into (set (mapcat :latent-addrs chains))
                            (mapcat :obs-addrs chains))
         lg-result (if source
                     (lg/detect-lg-blocks schema source :exclude-addrs kalman-addrs)
                     {:blocks [] :declined-addrs #{}})
         lg-blocks (:blocks lg-result)
         declined (:declined-addrs lg-result)
         rules (generate-rewrite-rules schema pairs chains lg-blocks declined)
         result (apply-rewrites graph schema nil rules)
         auto-transition (when (seq (:handlers result))
                           (auto/make-address-dispatch
                             h/generate-transition
                             (:handlers result)))]
     {:rewrite-result result
      :auto-transition auto-transition
      :kalman-chains chains
      :lg-blocks lg-blocks
      :declined-addrs declined
      :stats {:total-sites (count (:nodes graph))
              :eliminated (count (:eliminated result))
              :residual (count (:nodes (:residual-graph result)))
              :rewrites-applied (count (:rewrite-log result))}})))
