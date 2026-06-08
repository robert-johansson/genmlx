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

   v1 scope: generate + assess (exact). regenerate/update DECLINE (the block is
   excluded from regenerate handlers, so its latents fall through to sampling).
   Blocks whose latents have non-conjugate children, latent-dependent noise,
   non-affine means, or latents not preceding their observations are DECLINED —
   their pairs are removed so no rule claims a (wrong) exact elimination."
  (:require [clojure.set :as set]
            [genmlx.mlx :as mx]
            [genmlx.mlx.constants :refer [LOG-2PI]]
            [genmlx.choicemap :as cm]
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
                        (let [r (lg-kalman-step {:mean mean :cov cov} ob)]
                          {:mean (:mean r) :cov (:cov r)
                           :ll (mx/add ll (:ll r))}))
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

(defn- eligible-block
  "Validate a concern component and, if eligible, build a block descriptor.
   Returns {:eligible? true :block desc} or {:eligible? false}.

   Eligible when: priors are independent (no latent in another latent's prior),
   obs depend only on block latents, no latent has a non-conjugate child, obs
   noise is latent-independent, latents precede all obs in dep-order, and every
   obs mean/sigma form compiles."
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
        ;; (b) obs depend only on block latents (mean only references block latents)
        obs-deps-ok? (every? (fn [s]
                               (set/subset? (set/intersection (:deps s) all-trace-addrs)
                                            latents))
                             obs-sites)
        ;; (c) no non-conjugate children of any block latent
        children-conjugate?
        (every? (fn [la]
                  (every? (fn [s] (or (not (contains? (:deps s) la))
                                      (contains? obs (:addr s))))
                          (:trace-sites schema)))
                latents)
        ;; (d) noise latent-independent: sigma form references no block latent
        noise-ok? (every? (fn [s] (not (references-any? (sigma-form s) latents))) obs-sites)
        ;; (e) latents precede obs in dep-order
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
    (if (and indep-priors? obs-deps-ok? children-conjugate? noise-ok? order-ok? forms-ok?)
      {:eligible? true
       :block {:id order-latents            ; stable id (blocks are disjoint)
               :latents order-latents
               :latent-index (into {} (map-indexed (fn [i a] [a i])) order-latents)
               :p (count order-latents)
               :obs obs-fns
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

(defn make-lg-handlers
  "Build generate-mode address handlers for a linear-Gaussian block.

   Latent handlers collect each prior (mean, var) and, once all are seen,
   materialise the joint belief. Observation handlers perform a vector-Kalman
   measurement update against the model args (read from :model-args in state),
   accumulating the marginal LL into :score and :weight and writing posterior
   means back to the latent choices. Handlers return nil (fall through) when the
   block cannot proceed (missing args/belief, unconstrained obs)."
  [block]
  (let [{:keys [id latents p obs]} block
        latent-handlers
        (into {}
              (map (fn [la]
                     [la
                      (fn [state _addr dist]
                        (let [{:keys [mu sigma]} (:params dist)
                              var (mx/multiply sigma sigma)
                              st (-> state
                                     (assoc-in [:lg-init id la] {:mean (coerce-scalar mu)
                                                                 :var (coerce-scalar var)})
                                     (update :choices cm/set-value la mu))
                              inits (get-in st [:lg-init id])
                              st (if (= (count inits) p)
                                   (let [m0 (mx/stack (mapv #(:mean (get inits %)) latents))
                                         s0 (mx/diag (mx/stack (mapv #(:var (get inits %)) latents)))]
                                     (assoc-in st [:lg-belief id] {:mean m0 :cov s0}))
                                   st)]
                          [mu st]))]))
              latents)
        obs-handlers
        (into {}
              (map (fn [{:keys [addr mean-fn sigma-fn]}]
                     [addr
                      (fn [state _addr _dist]
                        (let [belief (get-in state [:lg-belief id])
                              args (:model-args state)
                              constraint (cm/get-submap (:constraints state) addr)]
                          (when (and belief args (cm/has-value? constraint))
                            (let [y (cm/get-value constraint)
                                  {:keys [h c]} (obs-design mean-fn latents args)
                                  s (coerce-scalar
                                     (sigma-fn (zipmap latents (repeat (mx/scalar 0.0))) args))
                                  r (mx/multiply s s)
                                  {:keys [mean cov ll]} (lg-kalman-step belief {:y y :h h :c c :r r})
                                  st (reduce (fn [st i]
                                               (update st :choices cm/set-value
                                                       (nth latents i) (mx/index mean i)))
                                             state (range p))]
                              [y (-> st
                                     (assoc-in [:lg-belief id] {:mean mean :cov cov})
                                     (update :choices cm/set-value addr y)
                                     (update :score mx/add ll)
                                     (update :weight mx/add ll))]))))]))
              obs)]
    (merge latent-handlers obs-handlers)))
