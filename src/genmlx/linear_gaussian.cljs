(ns genmlx.linear-gaussian
  "L3 joint linear-Gaussian elimination (genmlx-lwhw).

   Eliminates a *block* of independent Gaussian-prior latents that jointly
   determine Gaussian observations through affine means — i.e. Bayesian linear
   regression. The per-prior scalar conjugate path (auto_analytical) is only
   correct when each observation depends on a SINGLE latent through the natural
   parameter directly; it silently drops the affine coefficient and offset, so a
   model like `y_j ~ N(slope*x_j + intercept, σ)` collapsed to the shared-mean
   marginal `y_j ~ N(intercept, σ)` (off by ~16 nats on the linreg benchmark).

   This module detects such blocks and eliminates them JOINTLY:

     latents  β ~ N(m0, S0)            (independent priors, S0 diagonal)
     obs      y_j ~ N(h_j·β + c_j, r_j)
     marginal y ~ N(X m0 + c, R + X S0 Xᵀ)
     posterior β|y ~ N(m1, S1), S1 = (S0⁻¹ + Xᵀ R⁻¹ X)⁻¹

   The design row h_j and offset c_j are recovered at generate time from the
   observation's source mean form, evaluated against the model args via the
   compiled expression evaluator (compile-expr) — no autodiff needed, since the
   mean is affine in the latents:

     c_j  = mean_j(all latents → 0, args)
     h_jk = mean_j(latent_k → 1, others → 0, args) − c_j

   Elimination is performed by a sequential vector-Kalman measurement update,
   which is algebraically identical to the batch marginal/posterior (chain rule
   of the joint Gaussian density) and fits the per-address handler structure.

   Scope: generate + assess (exact), regenerate (genmlx-m3tn: the block stays
   Rao-Blackwellised under MH moves over unrelated residual latents; selecting a
   block latent re-opens the whole block), and UPDATE (genmlx-6hcu: a value-only
   obs change re-folds the block marginal LL — weight = Δ marginal-LL, latents →
   new posterior mean, changed-obs old values to the discard; constraining a block
   latent re-opens it → the joint handler path).

   Partial conjugacy (genmlx-4q9d): when the obs noise depends on a RESIDUAL
   (non-block) latent — e.g. y_j ~ N(slope·x_j+intercept, sigma) with sigma ~ Gamma
   — the block is eliminated CONDITIONAL on the sampled residual (the marginal is
   exact per sample; the population gives E_residual[block marginal], a strictly
   Rao-Blackwellised IS/MH estimate).

   Blocks whose latents have non-conjugate children, whose noise references a
   BLOCK latent, with non-affine means, with residual noise latents that do not
   precede the obs, or latents not preceding their observations are DECLINED —
   their pairs are removed so no rule claims a (wrong) exact elimination."
  (:require [clojure.set :as set]
            [genmlx.mlx :as mx]
            [genmlx.mlx.constants :refer [LOG-2PI]]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.compiled :as compiled]))

;; ===========================================================================
;; Pure math — sequential vector-Kalman elimination
;; ===========================================================================

(defn lg-kalman-step
  "One vector-Kalman measurement update for a linear-Gaussian block.

   belief: {:mean [p] :cov [p p]}  (MLX arrays)
   obs:    {:y scalar :h [p] :c scalar :r variance-scalar}  (MLX)
   Returns {:mean [p] :cov [p p] :ll scalar}.

   S = hᵀP h + r;  K = P h / S;  innov = y − (hᵀm + c)
   m' = m + K·innov;  P' = P − K (P h)ᵀ;  ll = −½(log2π + log S + innov²/S)"
  [{:keys [mean cov]} {:keys [y h c r]}]
  (let [p (first (mx/shape mean))
        Ph    (mx/flatten (mx/matmul cov (mx/reshape h [p 1])))   ; P h, [p]
        S     (mx/add (mx/sum (mx/multiply h Ph)) r)               ; scalar
        K     (mx/divide Ph S)                                     ; [p]
        pred  (mx/add (mx/sum (mx/multiply h mean)) c)             ; scalar
        innov (mx/subtract y pred)                                 ; scalar
        new-mean (mx/add mean (mx/multiply K innov))
        new-cov  (mx/subtract cov (mx/outer K Ph))                 ; rank-1 downdate
        ll (mx/multiply (mx/scalar -0.5)
                        (mx/add (mx/scalar LOG-2PI)
                                (mx/add (mx/log S)
                                        (mx/divide (mx/multiply innov innov) S))))]
    {:mean new-mean :cov new-cov :ll ll}))

(defn lg-eliminate
  "Sequentially eliminate a linear-Gaussian block.
   m0: [p] prior mean, S0: [p p] prior covariance (diagonal),
   obs-specs: seq of {:y :h :c :r} (r is a variance).
   Returns {:marginal-ll scalar :post-mean [p] :post-cov [p p]}."
  [m0 S0 obs-specs]
  (let [final (reduce (fn [{:keys [mean cov ll]} ob]
                        (let [step (lg-kalman-step {:mean mean :cov cov} ob)]
                          (assoc step :ll (mx/add ll (:ll step)))))
                      {:mean m0 :cov S0 :ll (mx/scalar 0.0)}
                      obs-specs)]
    {:marginal-ll (:ll final) :post-mean (:mean final) :post-cov (:cov final)}))

;; ===========================================================================
;; Detection
;; ===========================================================================

(defn- form-symbols
  "Set of symbols appearing anywhere in a source form."
  [form]
  (cond
    (symbol? form) #{form}
    (seq? form)    (into #{} (mapcat form-symbols) form)
    (vector? form) (into #{} (mapcat form-symbols) form)
    (map? form)    (into #{} (mapcat form-symbols) (concat (keys form) (vals form)))
    :else #{}))

(defn- references-any?
  "Does the form reference any symbol whose name matches an address in addrs?"
  [form addrs]
  (let [names (into #{} (map name) addrs)]
    (boolean (some #(contains? names (name %)) (form-symbols form)))))

(defn- form-trace-addrs
  "Trace-site addresses (drawn from all-addrs) referenced anywhere in a source
   form, matched by symbol name. Used to separate an observation's MEAN deps
   (must be block latents) from its NOISE deps (may be residual latents)."
  [form all-addrs]
  (let [by-name (into {} (map (fn [a] [(name a) a])) all-addrs)]
    (into #{} (keep #(get by-name (name %))) (form-symbols form))))

(defn- connected-components
  "Undirected connected components over conjugate pairs (prior-addr ↔ obs-addr).
   Returns a seq of {:nodes :latents :obs :pairs}."
  [pairs]
  (let [priors (into #{} (map :prior-addr) pairs)
        nodes  (into priors (map :obs-addr) pairs)
        adj    (reduce (fn [m {:keys [prior-addr obs-addr]}]
                         (-> m
                             (update prior-addr (fnil conj #{}) obs-addr)
                             (update obs-addr (fnil conj #{}) prior-addr)))
                       {} pairs)]
    (loop [unseen nodes comps []]
      (if (empty? unseen)
        comps
        (let [start (first unseen)
              comp  (loop [stack [start] seen #{}]
                      (if (empty? stack)
                        seen
                        (let [x (peek stack) stack' (pop stack)]
                          (if (contains? seen x)
                            (recur stack' seen)
                            (recur (into stack' (get adj x)) (conj seen x))))))]
          (recur (set/difference unseen comp)
                 (conj comps {:nodes comp
                              :latents (set/intersection comp priors)
                              :obs (set/difference comp priors)
                              :pairs (filterv #(contains? comp (:prior-addr %)) pairs)})))))))

(defn- concern?
  "Is this component a regression-block concern (beyond the correct single-latent
   :direct scalar path)? True when ≥2 latents, or any dependency is affine."
  [comp]
  (or (>= (count (:latents comp)) 2)
      (some #(= :affine (:type (:dependency-type %))) (:pairs comp))))

(defn- mean-form [site] (first (:dist-args site)))
(defn- sigma-form [site] (second (:dist-args site)))

(defn- expand-expr-symbols
  "Recursively replace symbols bound to :expr forms in the binding env with
   their defining forms, so INDIRECT trace dependencies become visible to the
   dependency analysis (genmlx-b470): a let-bound `noise-std` referencing a
   traced `noise-prec` made noise-latents come out empty, and the obs handler
   then evaluated the sigma form with the residual latent missing from its
   env — a hard NAPI error at generate time."
  [form binding-env seen]
  (cond
    (symbol? form)
    (let [info (get binding-env (name form))]
      (if (and (= :expr (:kind info)) (not (contains? seen (name form))))
        (expand-expr-symbols (:form info) binding-env (conj seen (name form)))
        form))
    (seq? form) (doall (map #(expand-expr-symbols % binding-env seen) form))
    (vector? form) (mapv #(expand-expr-symbols % binding-env seen) form)
    :else form))

(defn- latent-interaction?
  "Does the form multiply/divide two (sub)expressions that BOTH reference block
   latents? Static complement to the runtime joint-affinity probe
   (genmlx-b470): a bilinear mean like (mx/multiply slope intercept) is affine
   PER-PAIR (each latent looks constant while the other is analyzed) but not
   JOINTLY affine, so 0/1 probing would fabricate h=0."
  [form latents]
  (cond
    (seq? form)
    (or (and (symbol? (first form))
             (contains? #{"multiply" "divide" "matmul" "outer"}
                        (name (first form)))
             (>= (count (filter #(references-any? % latents) (rest form))) 2))
        (boolean (some #(latent-interaction? % latents) (rest form))))
    (vector? form) (boolean (some #(latent-interaction? % latents) form))
    :else false))

(defn- eligible-block
  "Validate a concern component and, if eligible, build a block descriptor.
   Returns {:eligible? true :block desc} or {:eligible? false}.

   Eligible when: priors are independent (no latent in another latent's prior),
   obs MEANS depend only on block latents, no block latent has a non-conjugate
   child, obs noise references no block latent, block latents precede all obs in
   dep-order, and every obs mean/sigma form compiles.

   Partial conjugacy (genmlx-4q9d): obs NOISE may reference RESIDUAL (non-block)
   latents — the block is then eliminated CONDITIONAL on those residual values
   (read at runtime from :choices). Such residual noise latents must precede the
   block obs in dep-order, and are recorded as :noise-latents so the obs handlers
   inject their sampled values when evaluating the noise form."
  [comp schema binding-env site-map dep-idx all-trace-addrs]
  (let [latents (:latents comp)
        obs     (:obs comp)
        latent-sites (map site-map latents)
        obs-sites    (map site-map obs)
        ;; latents ordered by dep-order; obs likewise
        order-latents (vec (sort-by #(get dep-idx % 0) latents))
        order-obs     (vec (sort-by #(get dep-idx % 0) obs))
        ;; (a) independent priors: a latent's prior must not reference another latent
        indep-priors? (every? (fn [s] (not (references-any? (vec (:dist-args s)) latents)))
                               latent-sites)
        ;; Dependency analysis runs on EXPANDED forms (let-bound :expr symbols
        ;; replaced by their definitions) so indirect trace references are
        ;; visible (genmlx-b470). Compilation below still uses the original
        ;; forms — compile-expr resolves :expr bindings itself.
        x-mean (fn [s] (expand-expr-symbols (mean-form s) binding-env #{}))
        x-sigma (fn [s] (expand-expr-symbols (sigma-form s) binding-env #{}))
        ;; (b) obs MEAN references only block latents (noise handled separately below)
        mean-deps-ok? (every? (fn [s]
                                (set/subset? (form-trace-addrs (x-mean s) all-trace-addrs)
                                             latents))
                              obs-sites)
        ;; (b') obs MEAN has no multiplicative interaction between block
        ;; latents — bilinear means are not jointly affine (genmlx-b470)
        no-interaction? (every? (fn [s] (not (latent-interaction? (x-mean s) latents)))
                                obs-sites)
        ;; (c) no non-conjugate children of any block latent
        children-conjugate?
        (every? (fn [la]
                  (every? (fn [s] (or (not (contains? (:deps s) la))
                                      (contains? obs (:addr s))))
                          (:trace-sites schema)))
                latents)
        ;; (d) noise references no BLOCK latent (it may reference residual latents)
        sigma-block-free? (every? (fn [s] (not (references-any? (x-sigma s) latents))) obs-sites)
        ;; (d') residual latents the noise depends on (genmlx-4q9d): exclude block
        ;; latents and block obs — what remains are the residual noise latents.
        noise-latents (reduce (fn [acc s]
                                (set/union acc (set/difference
                                                (form-trace-addrs (x-sigma s) all-trace-addrs)
                                                latents obs)))
                              #{} obs-sites)
        ;; (d'') residual noise latents must precede all block obs in dep-order, so
        ;; their sampled value is available in :choices when the obs handler fires.
        noise-precede? (or (empty? noise-latents)
                           (let [min-obs (apply min (map #(get dep-idx % 0) order-obs))]
                             (every? #(< (get dep-idx % 0) min-obs) noise-latents)))
        ;; (e) block latents precede obs in dep-order
        order-ok? (or (empty? order-obs)
                      (< (apply max (map #(get dep-idx % 0) latents))
                         (apply min (map #(get dep-idx % 0) obs))))
        ;; (f) compile mean + sigma forms
        obs-fns (mapv (fn [a]
                        (let [s (site-map a)]
                          {:addr a
                           :mean-fn (compiled/compile-expr (mean-form s) binding-env #{})
                           :sigma-fn (compiled/compile-expr (sigma-form s) binding-env #{})}))
                      order-obs)
        forms-ok? (every? #(and (:mean-fn %) (:sigma-fn %)) obs-fns)]
    (if (and indep-priors? mean-deps-ok? no-interaction? children-conjugate?
             sigma-block-free? noise-precede? order-ok? forms-ok?)
      {:eligible? true
       :block {:id order-latents            ; stable id (blocks are disjoint)
               :latents order-latents
               :latent-index (into {} (map-indexed (fn [i a] [a i])) order-latents)
               :p (count order-latents)
               :obs obs-fns
               :noise-latents noise-latents
               :latent-addrs order-latents
               :obs-addrs order-obs
               :all-addrs (set/union latents obs)}}
      {:eligible? false})))

(defn detect-lg-blocks
  "Detect linear-Gaussian regression blocks in a schema.

   schema: extracted schema with :conjugate-pairs, :trace-sites, :dep-order.
   source: gen source form (params + body) for the binding environment.
   exclude-addrs: addresses already claimed (e.g. by Kalman chains) — skipped.

   Returns {:blocks [block-descriptor ...] :declined-addrs #{addr ...}}.
   Declined addresses are concern-components that cannot be exactly eliminated;
   callers should drop their conjugate pairs so no rule claims a wrong exact."
  [schema source & {:keys [exclude-addrs] :or {exclude-addrs #{}}}]
  (let [pairs (->> (:conjugate-pairs schema)
                   (filter #(= :normal-normal (:family %)))
                   (remove #(or (contains? exclude-addrs (:prior-addr %))
                                (contains? exclude-addrs (:obs-addr %)))))
        comps (filter concern? (connected-components pairs))]
    (if (empty? comps)
      {:blocks [] :declined-addrs #{}}
      (let [binding-env (compiled/build-binding-env source)
            site-map (into {} (map (juxt :addr identity)) (:trace-sites schema))
            dep-order (vec (:dep-order schema))
            dep-idx (into {} (map-indexed (fn [i a] [a i])) dep-order)
            all-trace-addrs (set (map :addr (:trace-sites schema)))]
        (reduce
          (fn [acc comp]
            (let [{:keys [eligible? block]}
                  (eligible-block comp schema binding-env site-map dep-idx all-trace-addrs)]
              (if eligible?
                (update acc :blocks conj block)
                (update acc :declined-addrs set/union (:nodes comp)))))
          {:blocks [] :declined-addrs #{}}
          comps)))))

;; ===========================================================================
;; Generate-mode handlers (used for generate AND assess)
;; ===========================================================================

(defn- coerce-scalar [x] (if (mx/array? x) x (mx/scalar x)))

(defn- obs-design
  "Recover the design row h [p] and offset c (scalar) for an observation, by
   probing its compiled mean form against the model args."
  [mean-fn latents args]
  (let [zero-v (zipmap latents (repeat (mx/scalar 0.0)))
        c (coerce-scalar (mean-fn zero-v args))
        h (mx/stack (mapv (fn [a]
                            (mx/subtract (coerce-scalar
                                          (mean-fn (assoc zero-v a (mx/scalar 1.0)) args))
                                         c))
                          latents))]
    {:h h :c c}))

(defn- probe-jointly-affine?
  "Runtime backstop for the static affine classification (genmlx-b470).

   The 0/1 probing in obs-design recovers h and c correctly ONLY when the mean
   form is JOINTLY affine in the block latents. Bilinear means like
   (mx/multiply slope intercept) classify affine PER-PAIR (each latent looks
   constant while the other is analyzed) yet probe to h=0, c=0 — a silently
   wrong exact LL. Verify the recovered design reproduces the mean form at two
   joint probe points:  mean(s,...,s) ?= c + s·Σh  for s ∈ {1, 2}
   (s=2 additionally catches pure-power forms like β² that pass s=1).
   Tolerance is relative, float32-scaled. Any evaluation failure declines.

   The mx/item here forces a GPU eval inside a handler transition — acceptable
   for the same reason as mvn-well-conditioned?: the analytical path is
   scalar-only and dispatcher-gated against mx/in-grad?.

   The design recovery itself (obs-design) runs INSIDE the try: a mean form
   whose evaluation fails (e.g. a hidden non-latent trace reference leaving
   nil in the env) declines the block instead of crashing generate."
  [mean-fn latents args]
  (try
    (let [{:keys [h c]} (obs-design mean-fn latents args)
          sum-h (mx/sum h)
          check (fn [s]
                  (let [probe-env (zipmap latents (repeat (mx/scalar s)))
                        lhs (mx/item (coerce-scalar (mean-fn probe-env args)))
                        rhs (mx/item (mx/add c (mx/multiply (mx/scalar s) sum-h)))]
                    (and (js/isFinite lhs) (js/isFinite rhs)
                         (<= (js/Math.abs (- lhs rhs))
                             (* 1e-4 (max 1.0 (js/Math.abs lhs) (js/Math.abs rhs)))))))]
      (and (check 1.0) (check 2.0)))
    (catch :default _ false)))

(defn make-lg-handlers
  "Build address handlers for a linear-Gaussian block, parameterized by `mode`.

   Latent handlers collect each prior (mean, var) and, once all are seen,
   materialise the joint belief. Observation handlers perform a vector-Kalman
   measurement update against the model args (read from :model-args in state),
   writing posterior means back to the latent choices. Handlers return nil (fall
   through) when the block cannot proceed (missing args/belief, unconstrained obs).

   :generate (default) — obs read :constraints; the marginal LL accumulates into
     BOTH :score and :weight; latents write the prior mean (placeholder,
     overwritten with the posterior mean by the obs handlers). Used for generate
     AND assess.
   :regenerate — mirrors the scalar-conjugate regenerate contract (genmlx-m3tn).
     The block is a JOINTLY coupled unit, so if ANY block latent is selected the
     WHOLE block re-opens: every block handler returns nil, falling through to the
     base regenerate transition (selected latents resample from prior, others keep
     their old value, obs scored plainly). Otherwise (no block latent selected)
     latents seed the belief and write their OLD value; obs read :old-choices, do
     the Kalman update, write posterior means, and accumulate the marginal LL into
     :score ONLY (not :weight) — so an MH move over an unrelated residual sees the
     block's Rao-Blackwellised contribution to the target without double-counting.

   Partial conjugacy (genmlx-4q9d): when the block has :noise-latents, the obs
   handlers read those residual latents' sampled values from :choices and inject
   them into the noise (sigma) form — eliminating the block CONDITIONAL on the
   current residual. The marginal LL is then exact per residual sample, and the
   overall estimate is E_residual[block marginal] over the particle population."
  ([block] (make-lg-handlers block :generate))
  ([block mode]
   (let [{:keys [id latents p obs noise-latents obs-addrs]} block
         noise-latents (or noise-latents #{})
         obs-addrs (or obs-addrs (mapv :addr obs))
         regenerate? (= mode :regenerate)
         update? (= mode :update)
         block-reopened?
         (fn [state]
           (cond
             ;; Regenerate: selecting ANY block address — latent OR obs — re-opens
             ;; the whole block (genmlx-b470: a selected obs must be resampled by
             ;; the base transition, not Kalman-conditioned on its old value).
             regenerate?
             (and (:selection state)
                  (boolean (some #(sel/selected? (:selection state) %)
                                 (concat latents obs-addrs))))
             ;; Update (genmlx-6hcu): pinning a block LATENT in the constraints
             ;; re-opens the block (it can no longer be marginalised — score it
             ;; jointly). Constraining an OBS is the normal update (re-eliminate).
             ;; The dispatcher also declines the whole analytical-update for latent
             ;; constraints; this is the per-block backstop.
             update?
             (boolean (some #(cm/has-value? (cm/get-submap (:constraints state) %)) latents))
             :else false))
         block-ok?
         ;; Runtime eligibility gate (genmlx-b470), consulted by EVERY block
         ;; handler before committing and cached under [:lg-ok id] once a
         ;; latent commits (on the decline path nothing is committed, so the
         ;; recomputation cost is only paid by declined blocks):
         ;; - generate/assess: no block latent constrained, every block obs
         ;;   constrained (otherwise latents would be left at placeholder
         ;;   means with no score — neither joint nor marginal);
         ;; - both modes: every obs mean form passes the joint-affinity probe
         ;;   (bilinear means classify affine per-pair but probe to h=0).
         ;; When false, all block handlers fall through to the base transition.
         (fn [state]
           (if-some [cached (get-in state [:lg-ok id])]
             cached
             (let [args (:model-args state)
                   cs (:constraints state)
                   constraints-ok?
                   (cond
                     regenerate? true
                     ;; Update: ok as long as no block latent is pinned (obs are
                     ;; read new-over-old, so they need not all be in :constraints).
                     update? (not-any? #(cm/has-value? (cm/get-submap cs %)) latents)
                     :else (and (not-any? #(cm/has-value? (cm/get-submap cs %)) latents)
                                (every? #(cm/has-value? (cm/get-submap cs %)) obs-addrs)))]
               (boolean
                (and args constraints-ok?
                     (every? (fn [{:keys [mean-fn]}]
                               (probe-jointly-affine? mean-fn latents args))
                             obs))))))
         latent-handlers
         (into {}
               (map (fn [la]
                      [la
                       (fn [state _addr dist]
                         (when-not (block-reopened? state)
                           (when (block-ok? state)
                             (let [{:keys [mu sigma]} (:params dist)
                                   var (mx/multiply sigma sigma)
                                   value (if (or regenerate? update?)
                                           (cm/get-value (cm/get-submap (:old-choices state) la))
                                           mu)
                                   st (-> state
                                          (assoc-in [:lg-ok id] true)
                                          (assoc-in [:lg-init id la] {:mean (coerce-scalar mu)
                                                                      :var (coerce-scalar var)})
                                          (update :choices cm/set-value la value))
                                   inits (get-in st [:lg-init id])
                                   st (if (= (count inits) p)
                                        (let [m0 (mx/stack (mapv #(:mean (get inits %)) latents))
                                              s0 (mx/diag (mx/stack (mapv #(:var (get inits %)) latents)))]
                                          (assoc-in st [:lg-belief id] {:mean m0 :cov s0}))
                                        st)]
                               [value st]))))]))
               latents)
         obs-handlers
         (into {}
               (map (fn [{:keys [addr mean-fn sigma-fn]}]
                      [addr
                       (fn [state _addr _dist]
                         (when (and (not (block-reopened? state))
                                    (block-ok? state))
                           (let [belief (get-in state [:lg-belief id])
                                 args (:model-args state)
                                 old-sub (cm/get-submap (:old-choices state) addr)
                                 new-sub (cm/get-submap (:constraints state) addr)
                                 ;; new-over-old: update reads a CHANGED obs from
                                 ;; :constraints, an UNCHANGED obs from :old-choices,
                                 ;; so the full marginal LL is refolded.
                                 constraint (cond
                                              regenerate? old-sub
                                              update?     (if (cm/has-value? new-sub) new-sub old-sub)
                                              :else       new-sub)
                                 ;; Residual noise latents (genmlx-4q9d): their sampled
                                 ;; values, read live from :choices (set before the obs by
                                 ;; dep-order). Empty for v1 constant/block-free noise.
                                 noise-ready? (every? #(cm/has-value? (cm/get-submap (:choices state) %))
                                                      noise-latents)]
                             (when (and belief args noise-ready? (cm/has-value? constraint))
                               (let [y (cm/get-value constraint)
                                     {:keys [h c]} (obs-design mean-fn latents args)
                                     noise-vals (into {} (map (fn [nl]
                                                                [nl (cm/get-value
                                                                     (cm/get-submap (:choices state) nl))]))
                                                      noise-latents)
                                     sigma-env (merge (zipmap latents (repeat (mx/scalar 0.0)))
                                                      noise-vals)
                                     s (coerce-scalar (sigma-fn sigma-env args))
                                     r (mx/multiply s s)
                                     {:keys [mean cov ll]} (lg-kalman-step belief {:y y :h h :c c :r r})
                                     st (reduce (fn [st i]
                                                  (update st :choices cm/set-value
                                                          (nth latents i) (mx/index mean i)))
                                                state (range p))]
                                 [y (cond-> (-> st
                                                (assoc-in [:lg-belief id] {:mean mean :cov cov})
                                                (update :choices cm/set-value addr y)
                                                (update :score mx/add ll))
                                      (not regenerate?) (update :weight mx/add ll)
                                      ;; update: a changed obs charges its old value to :discard
                                      (and update? (cm/has-value? new-sub) (cm/has-value? old-sub))
                                      (update :discard cm/set-value addr (cm/get-value old-sub)))])))))]))
               obs)]
     (merge latent-handlers obs-handlers))))
