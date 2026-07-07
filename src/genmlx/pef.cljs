(ns genmlx.pef
  "Path-Equivalence Fuzzing (PEF) — genmlx-0bgi.

   Institutionalizes the 2026-07-06 audit (genmlx-ansg): every critical found
   there was a DIVERGENCE between two implementations of the same math,
   discovered by hand-comparing paths on a hand-built model. PEF generates
   random models from a togglable feature grammar, runs every registered path
   pair with the same PRNG key, and asserts equivalence — making 'the handler
   is ground truth; compilation is optimization' an enforced property instead
   of a convention.

   Layout:
     1. Feature profiles (togglable grammar; failures shrink to feature sets)
     2. Genome generators (test.check, deterministic per seed)
     3. Genome -> quoted source form -> DynamicGF (SCI eval + make-gen-fn,
        the msa-score wrap-model mechanism — schema extraction runs for real)
     4. Constraint / selection generators
     5. Comparison policies (:eps / :statistical; see P1 note on :bit-exact)
     6. Path registry P1-P6 (data, extensible). P1/P2 mirror the gfi
        compiled-equivalence / fast-vs-general law bodies but use PEF's
        non-finite-TOLERANT comparator: on key-dependent tail draws both
        paths can agree on -Inf/NaN, which is path agreement (model health
        under extreme params is the genmlx-7oen scope, excluded v1) —
        the laws' internal finiteness asserts are correct for healthy
        models but wrong for a fuzzer
     7. Invariants I2-I5 as registry entries (I1 soundness = P1-P5; I3
        crash-freedom is enforced by the runner on every op it executes)
     8. Runner + repro artifacts (paste-runnable one-liners)

   Comparison-policy note (documented deviation from the spec text): the
   spec asks P1 for BIT-EQUAL outputs under the same key. The compiled path
   samples via pre-generated standardized noise + transforms while the
   handler samples each site directly, so their PRNG consumption differs by
   construction and outputs agree only in DISTRIBUTION for sampling ops;
   score/weight parity on IDENTICAL choices is exact math and is compared at
   :eps 1e-4 (float32 reduction-order jitter — same tolerance as the gfi
   compiled-equivalence laws). I4's round-trip, where the spec's bit-exact
   intent is realizable (no resampling), IS checked bit-exact on choices."
  (:require [clojure.test.check.generators :as gen]
            [clojure.string :as str]
            [sci.core :as sci]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.trace :as tr]
            [genmlx.gfi :as gfi]
            [genmlx.inference.importance :as imp]
            [genmlx.inference.smc :as smc]))

;; ===========================================================================
;; 1. Feature profiles
;; ===========================================================================

(def full-profile
  "The complete v1 feature grammar. Each key independently togglable so a
   shrunk failure names a minimal feature set."
  {:name        :full
   :max-sites   8
   :dists       #{:gaussian :exponential :uniform :bernoulli :categorical
                  :laplace :log-normal}
   :deps?       true      ; dist-args referencing earlier sites (direct)
   :affine?     true      ; (mx/add (mx/multiply v c) o), coeff may be an arg
   :nonlinear?  true      ; (mx/exp v) / (mx/multiply v v)
   :rebinding?  true      ; same let name rebound (the 94qc shape)
   :derived?    true      ; derived locals under other names
   :nested?     true      ; trace call inside another site's dist-args (dv66)
   :splices?    true      ; sub-gfs; retval->site and site->arg (njzu, both ways)
   :branches?   true      ; if/if-not; M4-eligible and structure-changing arms
   :max-args    2})

(def smoke-profile
  "fast-core smoke: full grammar minus splices (sub-model eval dominates
   runtime) at fewer sites — wide feature coverage within the 45s budget."
  (assoc full-profile :name :smoke :max-sites 5 :splices? false))

(def independent-sites-profile
  "I2 liveness profile: literal-param gaussian/exponential/uniform sites, no
   deps/branches/splices — every model here is static and MUST compile (L1-M2)."
  {:name :independent-sites
   :max-sites 5
   :dists #{:gaussian :exponential :uniform}
   :deps? false :affine? false :nonlinear? false :rebinding? false
   :derived? false :nested? false :splices? false :branches? false
   :max-args 0})

(def sequential-obs-profile
  "P6 profile: chains of gaussian sites (deps allowed) — every site
   observable, suitable for per-site sequential SMC vs one-shot IS."
  {:name :sequential-obs
   :max-sites 4
   :dists #{:gaussian}
   :deps? true :affine? true :nonlinear? false :rebinding? false
   :derived? false :nested? false :splices? false :branches? false
   :max-args 0})

(def profiles
  {:full full-profile :smoke smoke-profile
   :independent-sites independent-sites-profile
   :sequential-obs sequential-obs-profile})

(def liveness-threshold
  "I2: minimum fraction of independent-sites-profile models on which the
   L1 compiled path must fire. Every model in that profile is static with
   noise-transform-covered dists, so anything below this means an
   eligibility gate silently started routing to the handler. Set below 1.0
   only to absorb future benign gate tightenings; today the observed
   fraction is 1.0."
  0.9)

;; ===========================================================================
;; 2. Genome generators
;; ===========================================================================
;; A genome is pure data: {:n-args n :entries [entry...] :sub-gfs [genome...]}
;; entry :type one of :trace :derived :splice :branch (see emitters below).
;; All randomness flows through test.check generators, so (a) one seed
;; determines everything and (b) quick-check shrinking operates on genomes.

(defn- gen-lit-params
  "Literal parameter generator per dist family. Ranges avoid degenerate
   supports (tiny sigmas, p near 0/1) — out-of-support fuzzing is excluded
   in v1 (blocked on genmlx-7oen)."
  [d]
  (case d
    :gaussian    (gen/tuple (gen/double* {:min -3 :max 3 :NaN? false :infinite? false})
                            (gen/double* {:min 0.3 :max 3 :NaN? false :infinite? false}))
    :log-normal  (gen/tuple (gen/double* {:min -1 :max 1 :NaN? false :infinite? false})
                            (gen/double* {:min 0.3 :max 1.5 :NaN? false :infinite? false}))
    :laplace     (gen/tuple (gen/double* {:min -3 :max 3 :NaN? false :infinite? false})
                            (gen/double* {:min 0.3 :max 3 :NaN? false :infinite? false}))
    :exponential (gen/fmap vector (gen/double* {:min 0.3 :max 3 :NaN? false :infinite? false}))
    :uniform     (gen/fmap (fn [[a w]] [a (+ a w)])
                           (gen/tuple (gen/double* {:min -3 :max 2 :NaN? false :infinite? false})
                                      (gen/double* {:min 0.5 :max 3 :NaN? false :infinite? false})))
    :bernoulli   (gen/fmap vector (gen/double* {:min 0.15 :max 0.85 :NaN? false :infinite? false}))
    :categorical (gen/fmap (fn [ls] [(vec ls)])
                           (gen/vector (gen/double* {:min -1 :max 1 :NaN? false :infinite? false}) 3))))

(def ^:private loc-dists
  "Families whose FIRST parameter is a location that may depend on earlier
   sites/args. Scale/rate/probability params stay literal in v1 (a negative
   sigma is out-of-support garbage — excluded until genmlx-7oen)."
  #{:gaussian :laplace :log-normal})

(defn- gen-dep-param
  "Location-param generator that references earlier bindings.
   prior-syms: symbols bound before this site. n-args: model arg count."
  [profile prior-syms n-args]
  (let [opts (cond-> []
               (:deps? profile)
               (conj (gen/fmap (fn [s] {:kind :dep :sym s}) (gen/elements prior-syms)))
               (:affine? profile)
               (conj (gen/fmap (fn [[s c o arg? ai]]
                                 (if (and arg? (pos? n-args))
                                   {:kind :affine :sym s :arg-idx ai}
                                   {:kind :affine :sym s :coeff c :offset o}))
                               (gen/tuple (gen/elements prior-syms)
                                          (gen/double* {:min -1.5 :max 1.5 :NaN? false :infinite? false})
                                          (gen/double* {:min -2 :max 2 :NaN? false :infinite? false})
                                          gen/boolean
                                          (gen/choose 0 (max 0 (dec n-args))))))
               (:nonlinear? profile)
               (conj (gen/fmap (fn [[s op]] {:kind :nonlin :sym s :op op})
                               (gen/tuple (gen/elements prior-syms)
                                          (gen/elements [:square :exp])))))]
    (when (seq opts) (gen/one-of opts))))

(defn- gen-trace-entry
  "One trace site. May draw a dep/nested location param when the profile and
   context allow."
  [profile prior-syms n-args addr-idx nested-ok?]
  (gen/bind
   (gen/elements (vec (:dists profile)))
   (fn [d]
     (gen/bind (gen-lit-params d)
       (fn [lits]
         (let [lit-params (mapv (fn [v] {:kind :lit :v v}) lits)
               dep-gen (when (and (contains? loc-dists d) (seq prior-syms))
                         (gen-dep-param profile prior-syms n-args))
               arg-loc? (and (contains? loc-dists d) (pos? n-args))
               nested-gen (when (and nested-ok? (:nested? profile) (contains? loc-dists d))
                            (gen/bind (gen-lit-params :gaussian)
                              (fn [nl]
                                (gen/return
                                 {:kind :nested
                                  :site {:addr (keyword (str "n" addr-idx))
                                         :dist :gaussian
                                         :params (mapv (fn [v] {:kind :lit :v v}) nl)}}))))
               loc-opts (cond-> [(gen/return (first lit-params))]
                          dep-gen    (conj dep-gen)
                          arg-loc?   (conj (gen/fmap (fn [i] {:kind :arg :idx i})
                                                     (gen/choose 0 (dec n-args))))
                          nested-gen (conj nested-gen))]
           (gen/fmap
            (fn [loc]
              {:type :trace
               :addr (keyword (str "x" addr-idx))
               :dist d
               :params (assoc lit-params 0 loc)})
            (gen/one-of loc-opts))))))))

(defn- gen-branch-entry
  [profile prior-syms n-args addr-idx]
  (gen/bind
   (gen/tuple (gen-lit-params :gaussian) (gen-lit-params :gaussian)
              gen/boolean gen/boolean
              (if (pos? n-args) (gen/choose 0 (dec n-args)) (gen/return 0)))
   (fn [[pt pf same? _ ai]]
     (let [cond-ref (if (pos? n-args)
                      {:kind :arg :idx ai}
                      (when (seq prior-syms) {:kind :sym :ref (last prior-syms)}))]
       (if (nil? cond-ref)
         ;; no condition source available — degrade to a plain trace site
         (gen-trace-entry profile prior-syms n-args addr-idx false)
         (gen/return
          {:type :branch
           :addr (keyword (str "b" addr-idx))
           :addr-f (if same? (keyword (str "b" addr-idx)) (keyword (str "b" addr-idx "f")))
           :same? same?
           :cond-ref cond-ref
           :params-t (mapv (fn [v] {:kind :lit :v v}) pt)
           :params-f (mapv (fn [v] {:kind :lit :v v}) pf)}))))))

(defn- gen-sub-genome
  "Sub-model for a splice: 1-3 plain gaussian/exponential sites, optionally
   taking one arg it uses as a location (the site->splice-arg direction)."
  [take-arg?]
  (gen/bind (gen/choose 1 3)
    (fn [n-sites]
      (gen/fmap
       (fn [param-vs]
         {:n-args (if take-arg? 1 0)
          :sub-gfs []
          :entries (vec
                    (map-indexed
                     (fn [i [mu sg]]
                       {:type :trace
                        :addr (keyword (str "y" i))
                        :sym (symbol (str "w" i))
                        :dist :gaussian
                        :params [(if (and take-arg? (zero? i))
                                   {:kind :arg :idx 0}
                                   {:kind :lit :v mu})
                                 {:kind :lit :v sg}]})
                     param-vs))
          :ret-sym (symbol (str "w" (dec n-sites)))})
       (gen/vector (gen-lit-params :gaussian) n-sites)))))

(defn- gen-entries
  "Sequential entry generation: each entry sees the symbols bound before it.
   Returns a generator of [entries sub-genomes]."
  [profile n-args]
  (gen/bind (gen/choose 1 (:max-sites profile))
    (fn [n-sites]
      (letfn [(step [i prior-syms entries subs]
                (if (= i n-sites)
                  (gen/return [entries subs])
                  (gen/bind
                   ;; entry-kind choice, gated by profile
                   (gen/frequency
                    (cond-> [[6 (gen/return :trace)]]
                      (and (:branches? profile)
                           (or (pos? n-args) (seq prior-syms)))
                      (conj [1 (gen/return :branch)])
                      (:splices? profile)
                      (conj [1 (gen/return :splice)])
                      (and (:derived? profile) (seq prior-syms))
                      (conj [1 (gen/return :derived)])))
                   (fn [kind]
                     (gen/bind
                      (case kind
                        :trace (gen-trace-entry profile prior-syms n-args i true)
                        :branch (gen-branch-entry profile prior-syms n-args i)
                        :derived (gen/fmap
                                  (fn [[s c o]] {:type :derived :of s :coeff c :offset o})
                                  (gen/tuple (gen/elements prior-syms)
                                             (gen/double* {:min -1.5 :max 1.5 :NaN? false :infinite? false})
                                             (gen/double* {:min -2 :max 2 :NaN? false :infinite? false})))
                        :splice (gen/bind (gen/tuple gen/boolean (gen/return nil))
                                  (fn [[feed-arg? _]]
                                    (let [feed? (and feed-arg? (seq prior-syms))]
                                      (gen/fmap
                                       (fn [sub] {:type :splice
                                                  :addr (keyword (str "sp" i))
                                                  :sub-idx (count subs)
                                                  :arg-sym (when feed? (last prior-syms))
                                                  ::sub sub})
                                       (gen-sub-genome feed?))))))
                      (fn [entry]
                        (gen/bind
                         ;; binding symbol: fresh, or rebind an earlier name
                         (if (and (:rebinding? profile) (seq prior-syms))
                           (gen/frequency [[4 (gen/return (symbol (str "v" i)))]
                                           [1 (gen/elements prior-syms)]])
                           (gen/return (symbol (str "v" i))))
                         (fn [sym]
                           (let [sub (::sub entry)
                                 entry (-> entry (dissoc ::sub) (assoc :sym sym))]
                             (step (inc i)
                                   (vec (distinct (conj prior-syms sym)))
                                   (conj entries entry)
                                   (if sub (conj subs sub) subs)))))))))))]
        (step 0 [] [] [])))))

(defn gen-genome
  "test.check generator of model genomes under `profile`."
  [profile]
  (gen/bind (gen/choose 0 (:max-args profile 0))
    (fn [n-args]
      (gen/fmap
       (fn [[entries subs]]
         {:n-args n-args
          :entries entries
          :sub-gfs subs
          :ret-sym (:sym (last entries))})
       (gen-entries profile n-args)))))

;; ===========================================================================
;; 3. Genome -> source form -> model
;; ===========================================================================

(defn- arg-sym [i] (symbol (str "a" i)))
;; Sub-gf references are namespace-qualified (pefsub/s0) and injected via SCI
;; :namespaces — plain :bindings symbols fail to resolve under nbb's
;; file-load analysis context, namespaced ones resolve everywhere (the same
;; mechanism dist/mx ride on).
(defn- sub-sym [i] (symbol "pefsub" (str "s" i)))

(def ^:private dist-sym
  {:gaussian 'dist/gaussian :exponential 'dist/exponential
   :uniform 'dist/uniform :bernoulli 'dist/bernoulli
   :categorical 'dist/categorical :laplace 'dist/laplace
   :log-normal 'dist/log-normal})

(declare site-call)

(defn- param-form [prm]
  (case (:kind prm)
    :lit    (let [v (:v prm)]
              (if (vector? v) (list 'mx/array v) v))
    :dep    (:sym prm)
    :affine (if (:arg-idx prm)
              (list 'mx/multiply (:sym prm) (arg-sym (:arg-idx prm)))
              (list 'mx/add (list 'mx/multiply (:sym prm) (:coeff prm)) (:offset prm)))
    :nonlin (case (:op prm)
              :square (list 'mx/multiply (:sym prm) (:sym prm))
              :exp    (list 'mx/exp (:sym prm)))
    :arg    (arg-sym (:idx prm))
    :nested (site-call (:site prm))))

(defn- dist-call [d params]
  (list* (dist-sym d) (map param-form params)))

(defn- site-call [{:keys [addr dist params]}]
  (list 'trace addr (dist-call dist params)))

(defn- cond-form [{:keys [kind idx ref]}]
  (if (= kind :arg)
    (list 'pos? (arg-sym idx))
    (list 'pos? (list 'mx/item ref))))

(defn- entry-expr [e]
  (case (:type e)
    :trace   (site-call e)
    :derived (list 'mx/add (list 'mx/multiply (:of e) (:coeff e)) (:offset e))
    :splice  (if (:arg-sym e)
               (list 'splice (:addr e) (sub-sym (:sub-idx e)) (:arg-sym e))
               (list 'splice (:addr e) (sub-sym (:sub-idx e))))
    :branch  (list 'if (cond-form (:cond-ref e))
                   (site-call {:addr (:addr e) :dist :gaussian :params (:params-t e)})
                   (site-call {:addr (:addr-f e) :dist :gaussian :params (:params-f e)}))))

(defn genome->source
  "Quoted gen source form ([args] (let [bindings...] ret)) for a genome —
   exactly what make-gen-fn walks for schema extraction."
  [genome]
  (let [args (mapv arg-sym (range (:n-args genome)))
        bindings (vec (mapcat (fn [e] [(:sym e) (entry-expr e)]) (:entries genome)))]
    (list args (list 'let bindings (:ret-sym genome)))))

(def ^:private base-sci-namespaces
  {'dist {'gaussian dist/gaussian 'exponential dist/exponential
          'uniform dist/uniform 'bernoulli dist/bernoulli
          'categorical dist/categorical 'laplace dist/laplace
          'log-normal dist/log-normal}
   'mx   {'add mx/add 'multiply mx/multiply 'exp mx/exp
          'item mx/item 'array mx/array}})

(defn- eval-body-fn
  "SCI-eval (fn [trace splice a0..] body) with sub-gf bindings injected
   under the pefsub namespace (see sub-sym)."
  [genome body-form sub-models]
  (let [fn-form (list 'fn
                      (into '[trace splice] (map arg-sym (range (:n-args genome))))
                      body-form)
        subs (into {} (map-indexed (fn [i m] [(symbol (str "s" i)) m]) sub-models))]
    (sci/eval-string (pr-str fn-form)
                     {:namespaces (assoc base-sci-namespaces 'pefsub subs)})))

(defn genome->model
  "Materialize a genome as a DynamicGF: SCI-evals the body (so execution is a
   real closure) and hands make-gen-fn the QUOTED source (so schema
   extraction, L1 compilation, and L3 conjugacy analysis all run for real)."
  [genome]
  (let [sub-models (mapv genome->model (:sub-gfs genome))
        source (genome->source genome)
        body-form (second source)
        f (eval-body-fn genome body-form sub-models)]
    (dyn/auto-key
     (dyn/make-gen-fn
      (fn [rt & as] (apply f (.-trace rt) (.-splice rt) as))
      source))))

(declare run-one-pair pair-by-name)

(defn source->model
  "Materialize a hand-written gen source form ([args] body) as a DynamicGF —
   the corpus entry point (pef_corpus.cljs freezes minimal audit models as
   source forms). `subs` maps splice symbols (sub0, ...) to already-built
   GFI values."
  ([source] (source->model source nil))
  ([source subs]
   (let [arg-syms (first source)
         body-form (second source)
         fn-form (list 'fn (into '[trace splice] arg-syms) body-form)
         f (sci/eval-string (pr-str fn-form)
                            {:namespaces (assoc base-sci-namespaces
                                                'pefsub (or subs {}))})]
     (dyn/auto-key
      (dyn/make-gen-fn
       (fn [rt & as] (apply f (.-trace rt) (.-splice rt) as))
       source)))))

(defn check-model
  "Run a pair-name subset against ONE model bundle {:model :args :meta} —
   the corpus runner. Returns {:pass? :failures}."
  [bundle {:keys [pair-names key]}]
  (let [key (or key (rng/fresh-key 424242))
        bundle (update bundle :meta #(or % {}))
        failures
        (into []
              (keep (fn [nm]
                      (let [spec (get pair-by-name nm)]
                        (when (and spec ((:applicable? spec) bundle))
                          (let [r (run-one-pair bundle spec key)]
                            (when-not (:pass? r)
                              {:pair nm :details (:details r)}))))))
              pair-names)]
    {:pass? (empty? failures) :failures failures}))

(defn- gen-args
  "Deterministic model-arg values for a genome (from the model's own seed)."
  [genome seed]
  (mapv (fn [i]
          ;; small nonzero magnitudes, sign varies — branch conditions on args
          ;; exercise both arms across models
          (let [h (hash [seed i])]
            (* (if (even? h) 1.0 -1.0)
               (+ 0.5 (mod (js/Math.abs h) 3)))))
        (range (:n-args genome))))

(defn model-for
  "THE deterministic entry point: (seed, idx, profile) -> materialized model
   bundle. Everything downstream (keys, constraints, selections) derives from
   the same coordinates, so any failure is replayable from them alone."
  [{:keys [seed idx profile]}]
  (let [genome (gen/generate (gen-genome profile) 30 (+ (* seed 100003) idx))
        model (genome->model genome)]
    {:genome genome
     :source (genome->source genome)
     :model model
     :args (gen-args genome (+ seed idx))
     :meta {:has-site-branches? (boolean (some #(and (= :branch (:type %))
                                                     (= :sym (get-in % [:cond-ref :kind])))
                                               (:entries genome)))
            :has-branches? (boolean (some #(= :branch (:type %)) (:entries genome)))
            :has-splices? (boolean (some #(= :splice (:type %)) (:entries genome)))}}))

;; ===========================================================================
;; 4. Constraints and selections (deterministic from coordinates)
;; ===========================================================================

(defn- leaf-paths
  "All leaf address paths of a choicemap, depth-first."
  ([cmap] (leaf-paths cmap []))
  ([cmap prefix]
   (cond
     (nil? cmap) []
     (cm/has-value? cmap) [prefix]
     :else (mapcat (fn [[k sub]] (leaf-paths sub (conj prefix k)))
                   (cm/-submaps cmap)))))

(defn- get-path [cmap path] (cm/get-choice cmap path))

(defn- subset-choicemap
  "Choicemap holding the simulated values at a deterministic subset of leaf
   paths (selector k: keep path when (mod (hash [salt path]) 3) < k)."
  [choices salt k]
  (reduce (fn [acc path]
            (if (< (mod (js/Math.abs (hash [salt path])) 3) k)
              (cm/set-choice acc path (get-path choices path))
              acc))
          cm/EMPTY
          (leaf-paths choices)))

(defn constraint-sets
  "Deterministic constraint variants for a simulated trace: none, a partial
   subset, and all sites."
  [trace salt]
  (let [choices (:choices trace)]
    {:none cm/EMPTY
     :partial (subset-choicemap choices salt 2)
     :all (reduce (fn [acc path] (cm/set-choice acc path (get-path choices path)))
                  cm/EMPTY (leaf-paths choices))}))

(defn selection-sets
  "Deterministic selection variants over a trace's leaf paths: all, none, a
   subset, its complement (and thereby hierarchical-into-splice selections,
   since subset paths include splice-qualified leaves)."
  [trace salt]
  (let [paths (leaf-paths (:choices trace))
        subset (filterv #(even? (js/Math.abs (hash [salt %]))) paths)]
    {:all sel/all
     :none sel/none
     :subset (sel/from-paths subset)
     :complement (sel/complement-sel (sel/from-paths subset))}))

;; ===========================================================================
;; 5. Comparison helpers
;; ===========================================================================

(defn- ev [x]
  (cond
    (nil? x) nil
    (mx/array? x) (do (mx/eval! x) (mx/item x))
    :else x))

(defn- close?
  "Numeric agreement for PATH comparison. Non-finite values agree when they
   agree exactly: -Inf==-Inf (both paths score the same event impossible) and
   NaN==NaN (both paths produce the same poisoned value — path EQUIVALENCE
   holds even though the model itself is unhealthy; tail-model health is the
   genmlx-7oen support-guard scope, excluded from PEF v1)."
  [a b tol]
  (cond
    (and (nil? a) (nil? b)) true
    (or (nil? a) (nil? b)) false
    :else (let [a (ev a) b (ev b)]
            (cond
              (and (js/isFinite a) (js/isFinite b))
              (<= (js/Math.abs (- a b))
                  (+ tol (* tol (js/Math.max (js/Math.abs a) (js/Math.abs b)))))
              (and (js/isNaN a) (js/isNaN b)) true
              :else (= a b)))))

(def ^:private eps 1e-4)

(defn- choices=
  "Leaf-wise numeric equality of two choicemaps at tolerance tol
   (tol 0.0 = exact value equality after realize)."
  [ca cb tol]
  (let [pa (set (leaf-paths ca)) pb (set (leaf-paths cb))]
    (and (= pa pb)
         (every? (fn [path]
                   (let [va (mx/realize (get-path ca path))
                         vb (mx/realize (get-path cb path))]
                     (if (zero? tol) (= va vb) (close? va vb tol))))
                 pa))))

;; ===========================================================================
;; 6. Path registry
;; ===========================================================================
;; Pair spec: {:name kw :applicable? (fn [bundle] bool)
;;             :run (fn [bundle key] {:pass? bool :details ...})
;;             :compare policy}

(defn- alt-vs-stripped
  "Generic 'alternate path vs stripped handler path' comparison — shared by
   P1 (full/branch-rewrite compile) and P3 (prefix compile), since both
   reduce to: the model with its alternate paths vs strip-compiled(model),
   compared per GFI op on identical choices. Covers all 6 ops:
   generate/assess/update/project directly; simulate via assess-on-choices
   (sampling paths differ by construction); regenerate via the fast/general
   law in P2 plus compiled-regenerate score parity here."
  [{:keys [model args]} key]
  (let [model (dyn/with-key (dyn/auto-key model) key)
        handler (dyn/auto-key (gfi/strip-compiled model))
        t (p/simulate model args)
        constraints (:choices t)
        ;; generate under full constraints: identical choices by construction
        ga (p/generate model args constraints)
        gb (p/generate handler args constraints)
        ;; assess
        aa (p/assess model args constraints)
        ab (p/assess handler args constraints)
        ;; update: swap in a second simulated trace's choices
        t2 (p/simulate model args)
        ua (p/update model (:trace ga) (:choices t2))
        ub (p/update handler (:trace gb) (:choices t2))
        ;; project over a deterministic subset selection
        sels (selection-sets t 7)
        pa (p/project model (:trace ga) (:subset sels))
        pb (p/project handler (:trace gb) (:subset sels))
        checks {:simulate-score (close? (:score t)
                                        (:weight (p/assess handler args constraints)) eps)
                :generate-score (close? (:score (:trace ga)) (:score (:trace gb)) eps)
                :generate-weight (close? (:weight ga) (:weight gb) eps)
                :assess-weight (close? (:weight aa) (:weight ab) eps)
                :update-weight (close? (:weight ua) (:weight ub) eps)
                :update-score (close? (:score (:trace ua)) (:score (:trace ub)) eps)
                :project (close? pa pb eps)}]
    {:pass? (every? val checks)
     :details (into {} (remove val checks))}))

(defn- p4-batched-vs-scalar
  "Batched vs scalar equivalence. Per-particle key alignment is impossible by
   construction (batched sampling draws one [N] tensor from one key, not N
   per-particle streams — documented deviation), so the EXACT comparison runs
   where sampling is out of the picture: vgenerate under FULL constraints
   stacked from N scalar traces must reproduce each scalar generate's weight
   and score per particle. vsimulate/vgenerate additionally get DIRECT shape
   checks ([n]-shaped scores/weights). The vectorized gfi laws are NOT used
   here: they assert finiteness internally, which key-dependent tail models
   (exp() blowups) fail in BOTH paths equally — model health, not path
   divergence (the genmlx-7oen scope)."
  [{:keys [model args] :as bundle} key]
  (let [model (dyn/auto-key model)
        [k0 k1] (rng/split (rng/ensure-key key))
        results
        (for [n [1 7]]
          (let [ks (rng/split-n k0 n)
                scalar (mapv (fn [ki] (p/simulate (dyn/with-key model ki) args)) ks)
                paths (leaf-paths (:choices (first scalar)))
                aligned? (every? #(= (set paths) (set (leaf-paths (:choices %)))) scalar)]
            (if-not aligned?
              ;; structure-changing branches make particles non-stackable —
              ;; graceful decline
              {:pass? true :skipped :non-uniform-structure}
              (let [constraints (reduce (fn [acc path]
                                          (cm/set-choice acc path
                                            (mx/stack (mapv #(get-path (:choices %) path) scalar))))
                                        cm/EMPTY paths)
                    vt (dyn/vgenerate model args constraints n k1)
                    vw (mx/->clj (:weight vt))
                    vs (mx/->clj (:score vt))
                    vw (if (number? vw) [vw] vw)
                    vs (if (number? vs) [vs] vs)
                    per (mapv (fn [t] (p/generate model args (:choices t))) scalar)
                    ok? (every? identity
                                (map (fn [i r]
                                       (and (close? (nth vw i) (:weight r) eps)
                                            (close? (nth vs i) (:score (:trace r)) eps)))
                                     (range n) per))]
                {:pass? ok? :details {:n n :vw vw :scalar-w (mapv #(ev (:weight %)) per)}}))))
        ;; direct batched-shape checks: vsimulate must produce an [n]-shaped
        ;; score with [n]-shaped leaves (no finiteness assert — see docstring).
        ;; vregenerate is exercised only through its documented decline
        ;; surface elsewhere (branch/splice batched regen is unsupported by
        ;; contract for non-fast-eligible selections; the spliced decline —
        ;; formerly a null deref, genmlx-89jo — is pinned in
        ;; regen_gate_test.cljs since the 0901fc6 gate fix).
        shape-run
        (try
          (let [n 7
                vt (dyn/vsimulate model args n k1)
                score-shape (mx/shape (:score vt))
                leaf-ok? (every? (fn [path]
                                   (= [n] (mx/shape (get-path (:choices vt) path))))
                                 (take 3 (leaf-paths (:choices vt))))]
            {:pass? (and (= [n] score-shape) leaf-ok?)
             :details {:score-shape score-shape}})
          (catch :default e
            ;; batched execution declines loudly on host-control-flow models
            ;; (site-conditioned branches use mx/item) — those are excluded by
            ;; :applicable?, so any throw here is a real failure
            {:pass? false :details {:vsimulate-error (.-message e)}}))
        bad (remove :pass? results)]
    {:pass? (and (empty? bad) (:pass? shape-run))
     :details {:constrained (vec bad) :shapes (:details shape-run)}}))

(defn- p5-analytical
  "P5a: on models WITHOUT an analytical plan the analytical dispatcher's
   decline must be invisible — identical results to strip-analytical under
   the same key. P5b: on models WHERE the plan fired, generate weight under
   full-observation constraints must equal the exact marginal — cross-checked
   against a high-N importance-sampling band (statistical policy; closed-form
   oracles live in the corpus, which pins the gaussian-gaussian family)."
  [{:keys [model args]} key]
  (let [model (dyn/auto-key model)
        plan (get-in model [:schema :analytical-plan])
        stripped (dyn/strip-analytical-path model)
        t (p/simulate (dyn/with-key stripped key) args)
        obs (subset-choicemap (:choices t) 13 2)]
    (if (nil? plan)
      (let [ga (p/generate (dyn/with-key model key) args obs)
            gb (p/generate (dyn/with-key stripped key) args obs)]
        {:pass? (and (close? (:weight ga) (:weight gb) eps)
                     (choices= (:choices (:trace ga)) (:choices (:trace gb)) eps))
         :details {:mode :declined}})
      ;; plan present: IS statistical band on the marginal
      (let [w (ev (:weight (p/generate (dyn/with-key model key) args obs)))
            n-seeds 5
            estimates (mapv (fn [i]
                              (ev (:log-ml-estimate
                                   (imp/importance-sampling
                                    {:samples 300 :key (rng/fresh-key (+ 1000 i))}
                                    model args obs))))
                            (range n-seeds))
            m (/ (reduce + estimates) n-seeds)
            sd (js/Math.sqrt (/ (reduce + (map #(let [d (- % m)] (* d d)) estimates))
                                (max 1 (dec n-seeds))))
            band (+ (* 6 sd) 0.15)]
        (if (empty? (leaf-paths obs))
          {:pass? (close? w 0.0 eps) :details {:mode :fired-empty-obs}}
          {:pass? (<= (js/Math.abs (- w m)) band)
           :details {:mode :fired :weight w :is-mean m :band band}})))))

(defn- p6-smc-vs-is
  "The uxjm class as a permanent property: sequential-obs SMC log-ML and
   one-shot IS log-ML estimate the same marginal. Band derived from the
   across-seed spread of both estimators — never a hand-tuned constant."
  [{:keys [model args]} key]
  (let [model (dyn/auto-key model)
        t (p/simulate (dyn/with-key model key) args)
        paths (leaf-paths (:choices t))
        ;; observe the LAST half of sites, one per SMC step
        obs-paths (vec (drop (quot (count paths) 2) paths))]
    (if (empty? obs-paths)
      {:pass? true :skipped :no-obs}
      (let [obs-vals (mapv #(get-path (:choices t) %) obs-paths)
            obs-seq (mapv (fn [path v] (cm/set-choice cm/EMPTY path v)) obs-paths obs-vals)
            full-obs (reduce (fn [acc [path v]] (cm/set-choice acc path v))
                             cm/EMPTY (map vector obs-paths obs-vals))
            n-seeds 4
            smc-est (mapv (fn [i]
                            (ev (:log-ml-estimate
                                 (smc/smc {:particles 120 :key (rng/fresh-key (+ 2000 i))}
                                          model args obs-seq))))
                          (range n-seeds))
            is-est (mapv (fn [i]
                           (ev (:log-ml-estimate
                                (imp/importance-sampling
                                 {:samples 400 :key (rng/fresh-key (+ 3000 i))}
                                 model args full-obs))))
                         (range n-seeds))
            mean (fn [xs] (/ (reduce + xs) (count xs)))
            sd (fn [xs] (let [m (mean xs)]
                          (js/Math.sqrt (/ (reduce + (map #(let [d (- % m)] (* d d)) xs))
                                           (max 1 (dec (count xs)))))))
            gap (js/Math.abs (- (mean smc-est) (mean is-est)))
            band (+ (* 6 (js/Math.sqrt (+ (js/Math.pow (sd smc-est) 2)
                                          (js/Math.pow (sd is-est) 2))))
                    0.2)]
        {:pass? (<= gap band)
         :details {:smc (mean smc-est) :is (mean is-est) :gap gap :band band}}))))

(defn- i4-discard-roundtrip
  "I4: update with the discard as constraints restores the original trace
   BIT-EXACTLY (choices=, tol 0 — pure replay, no resampling) and the two
   weights antisymmetrize."
  [{:keys [model args]} key]
  (let [model (dyn/with-key (dyn/auto-key model) key)
        t1 (p/simulate model args)
        t2 (p/simulate model args)
        new-constraints (subset-choicemap (:choices t2) 17 2)]
    (if (empty? (leaf-paths new-constraints))
      {:pass? true :skipped :empty-constraints}
      (let [{:keys [trace weight discard]} (p/update model t1 new-constraints)
            back (p/update model trace discard)]
        ;; a structure-changing update (branch flip) removes sites: the
        ;; round-trip contract only covers structure-preserving updates
        (if (not= (set (leaf-paths (:choices t1)))
                  (set (leaf-paths (:choices (:trace back)))))
          {:pass? true :skipped :structure-change}
          (let [w1 (ev weight) w2 (ev (:weight back))
                anti? (if (and (js/isFinite w1) (js/isFinite w2))
                        (close? (+ w1 w2) 0.0 eps)
                        ;; key-dependent tail draws: both non-finite in the
                        ;; same way = path agreement; antisymmetry is
                        ;; unevaluable (model health, genmlx-7oen scope)
                        (or (and (js/isNaN w1) (js/isNaN w2)) (= w1 (- w2))))]
            {:pass? (and (choices= (:choices t1) (:choices (:trace back)) 0.0) anti?)
             :details {:w1 w1 :w2 w2}}))))))

(defn- i5-score-type
  "I5: every produced trace carries a legal score-type."
  [{:keys [model args]} key]
  (let [model (dyn/with-key (dyn/auto-key model) key)
        legal #{:joint :marginal :beam-marginal}
        t (p/simulate model args)
        g (:trace (p/generate model args (subset-choicemap (:choices t) 5 2)))
        r (:trace (p/regenerate model t sel/all))]
    {:pass? (every? #(contains? legal (tr/score-type %)) [t g r])
     :details {:types (mapv tr/score-type [t g r])}}))

(defn- p2-fast-vs-general
  "Fast per-site regenerate vs the forced general retained-only path, same
   key — the :regenerate-fast-general-equivalence law body with the tolerant
   comparator (weights that agree on -Inf/NaN are path-agreement)."
  [{:keys [model args]} key]
  (let [model (dyn/auto-key model)
        t (p/simulate (dyn/with-key model key) args)
        paths (leaf-paths (:choices t))]
    (if (empty? paths)
      {:pass? true :skipped :no-sites}
      (let [sl (sel/from-paths [(first paths)])
            k (rng/fresh-key 7)
            fast (p/regenerate (dyn/with-key model k) t sl)
            general (binding [dyn/*force-general-regen* true]
                      (p/regenerate (dyn/with-key model k) t sl))]
        {:pass? (close? (:weight fast) (:weight general) 0.01)
         :details {:fast (ev (:weight fast)) :general (ev (:weight general))}}))))

(def pairs
  "The path registry (data, extensible). I1 soundness = P1-P5 collectively;
   I3 crash-freedom is enforced by the runner over every :run it executes."
  [{:name :p1-compiled-vs-handler
    :applicable? (fn [{:keys [model]}]
                   (boolean (some #(get (:schema model) %)
                                  [:compiled-simulate :compiled-generate
                                   :compiled-update :compiled-regenerate])))
    :run alt-vs-stripped
    :compare {:type :eps :tol eps}}

   {:name :p2-regen-fast-vs-general
    ;; direct implementation of the :regenerate-fast-general-equivalence law
    ;; body with PEF's non-finite-tolerant comparator (the gfi law's approx=
    ;; cannot evaluate key-dependent -Inf/NaN tail draws where both paths
    ;; agree — path equivalence still holds there)
    :applicable? (constantly true)
    :run p2-fast-vs-general
    :compare {:type :eps :tol 0.01}}

   {:name :p3-prefix-vs-handler
    :applicable? (fn [{:keys [model]}]
                   (boolean (some #(get (:schema model) %)
                                  [:compiled-prefix :compiled-prefix-generate
                                   :compiled-prefix-update :compiled-prefix-regenerate])))
    :run alt-vs-stripped
    :compare {:type :eps :tol eps}}

   {:name :p4-batched-vs-scalar
    :applicable? (fn [{:keys [meta]}]
                   ;; site-conditioned branches call mx/item in the body,
                   ;; which breaks batched execution by contract
                   (not (:has-site-branches? meta)))
    :run p4-batched-vs-scalar
    :compare {:type :eps :tol eps}}

   {:name :p5-analytical
    :applicable? (constantly true)
    :run p5-analytical
    :compare {:type :statistical :seeds 5}}

   {:name :p6-smc-vs-is
    :applicable? (fn [{:keys [meta]}] (not (:has-site-branches? meta)))
    :run p6-smc-vs-is
    :compare {:type :statistical :seeds 4}}

   {:name :i4-discard-roundtrip
    :applicable? (constantly true)
    :run i4-discard-roundtrip
    :compare {:type :bit-exact}}

   {:name :i5-score-type
    :applicable? (constantly true)
    :run i5-score-type
    :compare {:type :exact}}])

(def ^:private pair-by-name (into {} (map (juxt :name identity)) pairs))

;; ===========================================================================
;; 7. I2 liveness
;; ===========================================================================

(defn compiled-liveness
  "Fraction of independent-sites-profile models on which the L1 compiled
   path fires (anti-regression coverage: a silently-narrowed eligibility
   gate fails this, not just perf)."
  [{:keys [seed n-models]}]
  (let [fired (count
               (filter (fn [i]
                         (let [{:keys [model]} (model-for {:seed seed :idx i
                                                           :profile independent-sites-profile})]
                           (some? (get-in model [:schema :compiled-simulate]))))
                       (range n-models)))]
    {:fired fired :total n-models :fraction (/ fired n-models)}))

;; ===========================================================================
;; 8. Runner + repro artifacts
;; ===========================================================================

(defn format-repro
  "The non-negotiable artifact contract: a failure prints as a
   paste-runnable one-liner that regenerates the exact model and re-runs the
   failing pair."
  [{:keys [seed idx profile-name pair op details source]}]
  (str "(genmlx.pef/reproduce {:seed " seed " :idx " (or idx 0)
       " :profile-name " (or profile-name :full)
       " :pair " pair (when op (str " :op " op)) "})"
       (when details (str " ;; " (pr-str details)))
       (when source (str "\n  ;; source: " (pr-str source)))))

(defn- run-one-pair
  "Run one pair over one model bundle; I3 (crash-freedom) turns any
   un-graceful throw into a failure artifact."
  [bundle pair-spec key]
  (try
    ((:run pair-spec) bundle key)
    (catch :default e
      {:pass? false
       :details {:i3-crash (.-message e)
                 :genmlx-error (:genmlx/error (ex-data e))}})))

(defn run-pef
  "Run the fuzz sweep. opts:
     :seed      — master seed (everything derives from it)
     :n-models  — models to generate
     :profile   — feature profile map (or :profile-name keyword)
     :pairs     — pair-name subset (default: all)
   Returns {:pass? :n-models :n-checks :failures [artifact...]}."
  [{:keys [seed n-models profile profile-name] :as opts
    :or {n-models 100}}]
  (let [profile (or profile (get profiles profile-name) full-profile)
        pair-specs (if-let [names (:pairs opts)]
                     (mapv pair-by-name names)
                     pairs)
        failures (volatile! [])
        checks (volatile! 0)]
    (doseq [i (range n-models)]
      (let [bundle (try (model-for {:seed seed :idx i :profile profile})
                        (catch :default e
                          {:materialize-error (.-message e)}))]
        (if (:materialize-error bundle)
          (vswap! failures conj {:seed seed :idx i :profile-name (:name profile)
                                 :pair :materialize
                                 :details {:error (:materialize-error bundle)}})
          (let [key (rng/fresh-key (+ (* seed 7919) i))]
            (doseq [spec pair-specs
                    :when (and spec ((:applicable? spec) bundle))]
              (vswap! checks inc)
              (let [r (run-one-pair bundle spec key)]
                (when-not (:pass? r)
                  (vswap! failures conj
                          {:seed seed :idx i :profile-name (:name profile)
                           :pair (:name spec) :details (:details r)
                           :source (:source bundle)}))))))))
    {:pass? (empty? @failures)
     :n-models n-models
     :n-checks @checks
     :failures @failures}))

(defn reproduce
  "Replay a failure artifact: regenerate the exact model and re-run the
   named pair. Returns the pair result (plus the source for inspection)."
  [{:keys [seed idx profile-name pair]}]
  (let [profile (get profiles (or profile-name :full) full-profile)
        bundle (model-for {:seed seed :idx idx :profile profile})
        spec (get pair-by-name pair)
        key (rng/fresh-key (+ (* seed 7919) idx))]
    (assoc (if spec
             (run-one-pair bundle spec key)
             {:pass? false :details {:unknown-pair pair}})
           :source (:source bundle))))
