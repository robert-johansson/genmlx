(ns genmlx.compiled-gen
  "Compile gen functions for fast repeated inference.

   Two levels of compilation:

   Level 0: mx/compile-fn wrapper — fuses MLX operations into single Metal
   kernel while ClojureScript handler runs normally. ~5x speedup.

   Level 1: Schema-based direct execution — eliminates handler overhead
   (volatile!, persistent map threading, multimethod dispatch, per-site
   ChoiceMap construction). Model body still runs, but trace/param
   operations use pre-resolved direct function calls and mutable JS
   accumulators instead of the full handler machinery. ~2-3x on top of
   Level 0 for models with many trace sites.

   Both levels require static trace structure (same addresses every execution)."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.dist.core :as dc]
            [genmlx.vectorized :as vec]
            [genmlx.protocols :as p]))

;; ===========================================================================
;; Level 1: Schema-based direct execution
;; ===========================================================================

;; ---------------------------------------------------------------------------
;; Schema discovery
;; ---------------------------------------------------------------------------

(defn discover-schema
  "Run model body once to discover trace structure.
   Returns vector of {:addr keyword, :dist-type keyword, :constrained? bool}."
  [model args constraints n key]
  (let [sites (volatile! [])
        key (rng/ensure-key key)
        key-state #js [key]
        body-fn (:body-fn model)
        param-store (:genmlx.dynamic/param-store (meta model))

        trace-fn
        (fn [addr dist]
          (let [constrained? (cm/has-value? (cm/get-submap constraints addr))]
            (vswap! sites conj {:addr addr
                                :dist-type (:type dist)
                                :constrained? constrained?})
            ;; Return actual value so model body continues correctly
            (if constrained?
              (cm/get-value (cm/get-submap constraints addr))
              (let [k (aget key-state 0)
                    [k1 k2] (rng/split k)]
                (aset key-state 0 k1)
                (dc/dist-sample-n dist k2 n)))))

        param-fn
        (fn [nm default-value]
          (let [d (if (mx/array? default-value) default-value (mx/scalar default-value))]
            (if param-store
              (or (get-in param-store [:params nm]) d)
              d)))

        rt #js {:trace trace-fn :splice nil :param param-fn}]
    (rng/seed! key)
    (apply body-fn rt args)
    @sites))

;; ---------------------------------------------------------------------------
;; Method pre-resolution
;; ---------------------------------------------------------------------------

(defn resolve-methods
  "Pre-resolve multimethod dispatch for each unique dist type in schema.
   Returns {dist-type {:sample-n fn, :log-prob fn}} with direct function refs."
  [schema]
  (let [types (distinct (map :dist-type schema))
        sample-methods (cljs.core/methods dc/dist-sample-n*)
        lp-methods (cljs.core/methods dc/dist-log-prob)
        default-sample (get sample-methods :default)]
    (into {}
      (map (fn [t]
             [t {:sample-n (or (get sample-methods t) default-sample)
                 :log-prob (get lp-methods t)}])
           types))))

;; ---------------------------------------------------------------------------
;; Fast batched generate (Level 1 core)
;; ---------------------------------------------------------------------------

(defn fast-vgenerate
  "Schema-based batched generate with minimal overhead.
   Eliminates: volatile!, persistent map state threading, multimethod dispatch,
   per-site ChoiceMap construction. Model body still runs normally.
   Returns VectorizedTrace."
  [body-fn schema resolved args constraints n key param-store]
  (let [;; Pre-split all PRNG keys for latent sites
        n-latent (count (remove :constrained? schema))
        [body-key split-key] (rng/split key)
        latent-keys (when (pos? n-latent) (rng/split-n split-key n-latent))

        ;; Mutable accumulators — JS arrays for zero-overhead mutation
        key-idx #js [0]
        score #js [(mx/scalar 0.0)]
        weight #js [(mx/scalar 0.0)]
        ;; Collect [addr value] pairs, build ChoiceMap at end
        choice-pairs (volatile! (transient []))

        trace-fn
        (fn [addr dist]
          (let [dt (:type dist)
                {:keys [sample-n log-prob]} (get resolved dt)
                constraint (cm/get-submap constraints addr)]
            (if (cm/has-value? constraint)
              ;; Constrained site: use observation, accumulate log-prob
              (let [value (cm/get-value constraint)
                    lp (log-prob dist value)]
                (aset score 0 (mx/add (aget score 0) lp))
                (aset weight 0 (mx/add (aget weight 0) lp))
                (vswap! choice-pairs conj! [addr value])
                value)
              ;; Latent site: sample with pre-split key
              (let [ki (aget key-idx 0)
                    k (nth latent-keys ki)
                    _ (aset key-idx 0 (inc ki))
                    value (sample-n dist k n)
                    lp (log-prob dist value)]
                (aset score 0 (mx/add (aget score 0) lp))
                (vswap! choice-pairs conj! [addr value])
                value))))

        param-fn
        (fn [nm default-value]
          (let [d (if (mx/array? default-value) default-value (mx/scalar default-value))]
            (if param-store
              (or (get-in param-store [:params nm]) d)
              d)))

        rt #js {:trace trace-fn :splice nil :param param-fn}]
    ;; Seed global PRNG (required by MLX random functions)
    (rng/seed! body-key)
    (let [retval (apply body-fn rt args)
          ;; Build ChoiceMap from collected pairs (single pass)
          pairs (persistent! @choice-pairs)
          choices (reduce (fn [cm [addr value]]
                            (cm/set-value cm addr value))
                          cm/EMPTY
                          pairs)]
      (vec/->VectorizedTrace nil args choices
                              (aget score 0) (aget weight 0)
                              n retval))))

(defn fast-vgenerate-score-only
  "Like fast-vgenerate but skips ChoiceMap construction entirely.
   Returns {:score MLX-scalar, :weight MLX-scalar, :retval any}.
   Optimal for gradient computation where only score/weight matter."
  [body-fn schema resolved args constraints n key param-store]
  (let [n-latent (count (remove :constrained? schema))
        [body-key split-key] (rng/split key)
        latent-keys (when (pos? n-latent) (rng/split-n split-key n-latent))
        key-idx #js [0]
        score #js [(mx/scalar 0.0)]
        weight #js [(mx/scalar 0.0)]

        trace-fn
        (fn [addr dist]
          (let [dt (:type dist)
                {:keys [sample-n log-prob]} (get resolved dt)
                constraint (cm/get-submap constraints addr)]
            (if (cm/has-value? constraint)
              (let [value (cm/get-value constraint)
                    lp (log-prob dist value)]
                (aset score 0 (mx/add (aget score 0) lp))
                (aset weight 0 (mx/add (aget weight 0) lp))
                value)
              (let [ki (aget key-idx 0)
                    k (nth latent-keys ki)
                    _ (aset key-idx 0 (inc ki))
                    value (sample-n dist k n)
                    lp (log-prob dist value)]
                (aset score 0 (mx/add (aget score 0) lp))
                value))))

        param-fn
        (fn [nm default-value]
          (let [d (if (mx/array? default-value) default-value (mx/scalar default-value))]
            (if param-store
              (or (get-in param-store [:params nm]) d)
              d)))

        rt #js {:trace trace-fn :splice nil :param param-fn}]
    (rng/seed! body-key)
    (let [retval (apply body-fn rt args)]
      {:score (aget score 0) :weight (aget weight 0) :retval retval})))

;; ---------------------------------------------------------------------------
;; Schema validation
;; ---------------------------------------------------------------------------

(defn- validate-schema-static
  "Run model twice with different keys, verify same trace addresses.
   Returns schema if static, throws if dynamic."
  [model args constraints n]
  (let [s1 (discover-schema model args constraints n (rng/fresh-key 1))
        s2 (discover-schema model args constraints n (rng/fresh-key 2))
        addrs1 (mapv :addr s1)
        addrs2 (mapv :addr s2)]
    (when (not= addrs1 addrs2)
      (throw (ex-info
               (str "compile-gen: model has dynamic trace structure. "
                    "First run: " addrs1 ", second run: " addrs2 ". "
                    "Cannot compile — use the dynamic path instead.")
               {:addrs-1 addrs1 :addrs-2 addrs2})))
    s1))

;; ===========================================================================
;; CompiledGF — GFI-compatible compiled gen fn
;; ===========================================================================

(defrecord CompiledGF [body-fn schema resolved source]
  p/IGenerativeFunction
  (simulate [this args]
    (let [key (let [k (:genmlx.dynamic/key (meta this))]
                (cond
                  (= k :genmlx.dynamic/auto-key) (rng/fresh-key)
                  k k
                  :else (rng/fresh-key)))
          n 1
          vtrace (fast-vgenerate body-fn schema resolved args cm/EMPTY
                                  n key (:genmlx.dynamic/param-store (meta this)))
          ;; Extract scalar values from [1]-shaped arrays
          choices (:choices vtrace)]
      (tr/make-trace {:gen-fn this :args args
                      :choices choices
                      :retval (:retval vtrace)
                      :score (:score vtrace)})))

  p/IGenerate
  (generate [this args constraints]
    (let [key (let [k (:genmlx.dynamic/key (meta this))]
                (cond
                  (= k :genmlx.dynamic/auto-key) (rng/fresh-key)
                  k k
                  :else (rng/fresh-key)))
          n 1
          vtrace (fast-vgenerate body-fn schema resolved args constraints
                                  n key (:genmlx.dynamic/param-store (meta this)))]
      {:trace (tr/make-trace {:gen-fn this :args args
                              :choices (:choices vtrace)
                              :retval (:retval vtrace)
                              :score (:score vtrace)})
       :weight (:weight vtrace)})))

;; ---------------------------------------------------------------------------
;; Public API: compile-gen
;; ---------------------------------------------------------------------------

(defn compile-gen
  "Compile a gen fn for fast repeated execution.
   Discovers trace structure, pre-resolves multimethod dispatch, validates
   structural staticness. Returns a CompiledGF implementing simulate/generate.

   The compiled version eliminates per-trace-site overhead:
   - No volatile!/vreset! per site
   - No persistent map state threading
   - No multimethod dispatch (methods pre-resolved to direct fn refs)
   - ChoiceMap built once at the end, not incrementally

   Requires: model has static trace structure (same addresses every execution).
   Does not support splice (sub-generative-function calls).

   opts:
     :args         - model args for schema discovery (default [])
     :constraints  - observation choicemap for schema discovery (default EMPTY)
     :n-particles  - batch size for schema discovery (default 10)"
  ([model] (compile-gen model {}))
  ([model {:keys [args constraints n-particles]
           :or {args [] constraints cm/EMPTY n-particles 10}}]
   (let [model (dyn/auto-key model)
         schema (validate-schema-static model args constraints n-particles)
         resolved (resolve-methods schema)]
     (->CompiledGF (:body-fn model) schema resolved (:source model)))))

;; ===========================================================================
;; Level 0: mx/compile-fn wrappers (upgraded to use fast path internally)
;; ===========================================================================

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- build-param-store
  [params-array param-names]
  {:params (into {}
             (map-indexed (fn [i nm] [nm (mx/index params-array i)])
                          param-names))})

(defn- make-fast-loss-fn
  "Build loss function using Level 1 fast path: params-array → neg-log-ML scalar.
   Schema discovered once at construction time. No handler overhead per call."
  [model args observations param-names n-particles key]
  (let [;; Discover schema and pre-resolve methods once
        model-keyed (dyn/auto-key model)
        schema (discover-schema model-keyed args observations n-particles key)
        resolved (resolve-methods schema)
        body-fn (:body-fn model-keyed)]
    (fn [p]
      (let [store (build-param-store p param-names)
            result (fast-vgenerate-score-only
                     body-fn schema resolved args observations n-particles key
                     store)
            w (:weight result)]
        ;; neg-log-ML = -(logsumexp(weights) - log(N))
        (mx/subtract (mx/scalar (js/Math.log n-particles))
                     (mx/logsumexp w))))))

(defn- make-loss-fn
  "Build loss function using standard vgenerate (Level 0 fallback).
   key, model, args, observations, n-particles are captured in closure."
  [model args observations param-names n-particles key]
  (fn [p]
    (let [store (build-param-store p param-names)
          gf (vary-meta model assoc :genmlx.dynamic/param-store store)
          vtrace (dyn/vgenerate gf args observations n-particles key)]
      (mx/negative (vec/vtrace-log-ml-estimate vtrace)))))

;; ---------------------------------------------------------------------------
;; Compiled log-ML
;; ---------------------------------------------------------------------------

(defn compile-log-ml
  "Compile the log-ML estimator for a parametric model.
   Returns a function (fn [params-array] -> neg-log-ml) that runs as a
   single Metal kernel dispatch.

   key is frozen into the compiled function (same particles every call).
   Use for Fisher information where deterministic evaluation is required.

   opts:
     :n-particles  - IS particles (default 1000)
     :key          - PRNG key (frozen into compilation)
     :fast?        - use Level 1 fast path (default true)"
  [{:keys [n-particles key fast?] :or {n-particles 1000 fast? true}}
   model args observations param-names]
  (let [model (dyn/auto-key model)
        key (rng/ensure-key key)
        loss-fn (if fast?
                  (make-fast-loss-fn model args observations param-names n-particles key)
                  (make-loss-fn model args observations param-names n-particles key))
        compiled (mx/compile-fn loss-fn)]
    ;; Warm up: trigger compilation with dummy params
    (let [D (count param-names)
          dummy (mx/zeros [D])]
      (compiled dummy)
      (mx/materialize! (compiled dummy)))
    compiled))

;; ---------------------------------------------------------------------------
;; Compiled gradient
;; ---------------------------------------------------------------------------

(defn compile-log-ml-gradient
  "Compile the log-ML gradient estimator.
   Returns a function (fn [params-array] -> [neg-log-ml, grad])
   where both forward and backward passes are a single Metal dispatch.

   key is frozen (deterministic IS, same particles every call).
   Ideal for Fisher information and fixed-key optimization.

   opts:
     :n-particles  - IS particles (default 1000)
     :key          - PRNG key (frozen)
     :fast?        - use Level 1 fast path (default true)"
  [{:keys [n-particles key fast?] :or {n-particles 1000 fast? true}}
   model args observations param-names]
  (let [model (dyn/auto-key model)
        key (rng/ensure-key key)
        loss-fn (if fast?
                  (make-fast-loss-fn model args observations param-names n-particles key)
                  (make-loss-fn model args observations param-names n-particles key))
        vg (mx/value-and-grad loss-fn)
        compiled-vg (mx/compile-fn vg)]
    ;; Warm up
    (let [D (count param-names)
          dummy (mx/zeros [D])]
      (compiled-vg dummy)
      (mx/materialize! (first (compiled-vg dummy))))
    compiled-vg))

;; ---------------------------------------------------------------------------
;; Compiled gradient with varying key (for optimization loops)
;; ---------------------------------------------------------------------------

(defn compile-log-ml-gradient-keyed
  "Like compile-log-ml-gradient but key is an INPUT, not frozen.
   Returns (fn [params-array key-array] -> [neg-log-ml, grad]).
   Gradient is w.r.t. params only (argnums=[0]).

   For optimization loops where each step uses a fresh key.

   opts:
     :n-particles  - IS particles (default 1000)
     :fast?        - use Level 1 fast path (default true)"
  [{:keys [n-particles fast?] :or {n-particles 1000 fast? true}}
   model args observations param-names]
  (let [model-keyed (dyn/auto-key model)
        ;; Discover schema once with a dummy key
        schema (when fast?
                 (discover-schema model-keyed args observations n-particles
                                  (rng/fresh-key 0)))
        resolved (when fast? (resolve-methods schema))
        body-fn (:body-fn model-keyed)
        loss-fn (if fast?
                  (fn [p key]
                    (let [store (build-param-store p param-names)
                          result (fast-vgenerate-score-only
                                   body-fn schema resolved args observations
                                   n-particles key store)
                          w (:weight result)]
                      (mx/subtract (mx/scalar (js/Math.log n-particles))
                                   (mx/logsumexp w))))
                  (fn [p key]
                    (let [store (build-param-store p param-names)
                          gf (vary-meta model-keyed assoc
                               :genmlx.dynamic/param-store store
                               :genmlx.dynamic/key key)
                          vtrace (dyn/vgenerate gf args observations n-particles key)]
                      (mx/negative (vec/vtrace-log-ml-estimate vtrace)))))
        vg (mx/value-and-grad loss-fn [0])
        compiled-vg (mx/compile-fn vg)]
    ;; Warm up
    (let [D (count param-names)
          dummy-p (mx/zeros [D])
          dummy-k (rng/fresh-key 0)]
      (compiled-vg dummy-p dummy-k)
      (mx/materialize! (first (compiled-vg dummy-p dummy-k))))
    compiled-vg))
