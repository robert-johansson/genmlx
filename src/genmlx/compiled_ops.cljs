(ns genmlx.compiled-ops
  "Compiled GFI operations: generate, update, assess, project, regenerate.
   Fused loop compilation and tensor-native score functions.

   Split from genmlx.compiled — these are the GFI operation builders and
   fused inference operations that consume the shared infrastructure
   (noise transforms, binding env, expression compiler, site specs)
   defined in genmlx.compiled."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.handler :as h]
            [genmlx.compiled :as compiled]))

;; ===========================================================================
;; WP-1: Compiled Generate for Static Models
;; ===========================================================================
;;
;; Architecture: same as compiled simulate, but with per-site constraint
;; checking and weight accumulation. No mx/compile-fn (constraint checks
;; are data-dependent branches). Raw noise transforms only.

(defn- build-generate-site-step
  "Build the generate step for one trace site.
   Returns (fn [state args-vec constraints] -> state) where state has
   {:values :score :weight :key}.
   Constrained: use constraint value, add log-prob to score AND weight.
   Unconstrained: sample via noise transform, add log-prob to score only."
  [site-spec]
  (let [{:keys [addr compiled-args dist-type]} site-spec
        nt (get compiled/noise-transforms-full dist-type)]
    (when nt
      (let [log-prob-fn (:log-prob nt)]
        (cond
          (:noise-fn nt)
          ;; Standard distribution with noise transform
          (let [noise-fn (:noise-fn nt)
                transform-fn (:transform nt)]
            (fn [{:keys [values score weight key] :as state} args-vec constraints]
              (let [constraint (cm/get-submap constraints addr)]
                (if (cm/has-value? constraint)
                  ;; Constrained: use value, score + weight, no key split
                  (let [value (cm/get-value constraint)
                        eval-args (mapv #(% values args-vec) compiled-args)
                        lp (apply log-prob-fn value eval-args)]
                    {:values (assoc values addr value)
                     :score (mx/add score lp)
                     :weight (mx/add weight lp)
                     :key key})
                  ;; Unconstrained: sample via noise transform
                  (let [eval-args (mapv #(% values args-vec) compiled-args)
                        [k1 k2] (rng/split key)
                        noise (noise-fn k2)
                        value (apply transform-fn noise eval-args)
                        lp (apply log-prob-fn value eval-args)]
                    {:values (assoc values addr value)
                     :score (mx/add score lp)
                     :weight weight
                     :key k1})))))

          (:args-noise-fn nt)
          ;; Dynamic-shape distribution (e.g., iid-gaussian): noise shape
          ;; depends on dist-args, so use args-noise-fn
          (let [args-noise-fn (:args-noise-fn nt)
                transform-fn (:transform nt)]
            (fn [{:keys [values score weight key] :as state} args-vec constraints]
              (let [constraint (cm/get-submap constraints addr)]
                (if (cm/has-value? constraint)
                  ;; Constrained: use value, score + weight
                  (let [value (cm/get-value constraint)
                        eval-args (mapv #(% values args-vec) compiled-args)
                        lp (apply log-prob-fn value eval-args)]
                    {:values (assoc values addr value)
                     :score (mx/add score lp)
                     :weight (mx/add weight lp)
                     :key key})
                  ;; Unconstrained: sample via args-noise-fn
                  (let [eval-args (mapv #(% values args-vec) compiled-args)
                        [k1 k2] (rng/split key)
                        noise (args-noise-fn eval-args k2)
                        value (apply transform-fn noise eval-args)
                        lp (apply log-prob-fn value eval-args)]
                    {:values (assoc values addr value)
                     :score (mx/add score lp)
                     :weight weight
                     :key k1})))))

          :else
          ;; Delta distribution: no noise transform
          (fn [{:keys [values score weight key] :as state} args-vec constraints]
            (let [constraint (cm/get-submap constraints addr)]
              (if (cm/has-value? constraint)
                ;; Constrained delta: log-prob is 0 if value matches, -inf otherwise
                (let [value (cm/get-value constraint)
                      eval-args (mapv #(% values args-vec) compiled-args)
                      lp (apply log-prob-fn value eval-args)]
                  {:values (assoc values addr value)
                   :score (mx/add score lp)
                   :weight (mx/add weight lp)
                   :key key})
                ;; Unconstrained delta: value = first arg, lp = 0
                ;; Split key for PRNG equivalence with handler
                (let [eval-args (mapv #(% values args-vec) compiled-args)
                      [k1 _k2] (rng/split key)
                      value (first eval-args)]
                  {:values (assoc values addr value)
                   :score score
                   :weight weight
                   :key k1})))))))))

(defn make-compiled-generate
  "Build a compiled generate function from a gen schema and source.

   Returns (fn [key args-vec constraints] -> {:values :score :weight :retval})
   or nil if the model can't be compiled.

   No mx/compile-fn: constraint checks are data-dependent branches."
  [schema source]
  (when-let [{:keys [site-specs retval-fn]} (compiled/prepare-static-sites schema source)]
    (let [step-fns (mapv build-generate-site-step site-specs)]
      (when (and (every? some? step-fns) retval-fn)
        (fn compiled-generate [key args-vec constraints]
          (let [mlx-args (compiled/ensure-mlx-args args-vec)
                result
                (reduce
                 (fn [state step-fn]
                   (step-fn state mlx-args constraints))
                 {:values {} :score (mx/scalar 0.0) :weight (mx/scalar 0.0) :key key}
                 step-fns)]
            {:values (:values result)
             :score (:score result)
             :weight (:weight result)
             :retval (when retval-fn
                       (retval-fn (:values result) mlx-args))}))))))

(defn make-branch-rewritten-generate
  "Build a compiled generate for models with rewritable branches (L1-M4).
   Returns (fn [key args-vec constraints] -> {:values :score :weight :retval}) or nil."
  [schema source]
  (when-let [{:keys [site-specs retval-fn]} (compiled/prepare-branch-sites schema source)]
    (let [step-fns (mapv build-generate-site-step site-specs)]
      (when (every? some? step-fns)
        (fn compiled-branch-generate [key args-vec constraints]
          (let [mlx-args (compiled/ensure-mlx-args args-vec)
                result
                (reduce
                 (fn [state step-fn]
                   (step-fn state mlx-args constraints))
                 {:values {} :score (mx/scalar 0.0) :weight (mx/scalar 0.0) :key key}
                 step-fns)]
            {:values (:values result)
             :score (:score result)
             :weight (:weight result)
             :retval (when retval-fn
                       (retval-fn (:values result) mlx-args))}))))))

(defn make-compiled-prefix-generate
  "Build a compiled prefix generate function from a gen schema and source.
   Returns {:fn (fn [key args-vec constraints] -> {:values :score :weight})
            :addrs [keyword...]}
   or nil if partial compilation isn't applicable.
   Same gates as make-compiled-prefix. Uses build-generate-site-step for
   constraint-aware weight accumulation. No mx/compile-fn."
  [schema source]
  (when-let [{:keys [compiled-sites addrs]} (compiled/prepare-prefix-sites schema source)]
    (let [step-fns (mapv build-generate-site-step compiled-sites)]
      (when (every? some? step-fns)
        {:fn (fn compiled-prefix-generate [key args-vec constraints]
               (let [mlx-args (compiled/ensure-numeric-mlx-args args-vec)
                     result
                     (reduce
                      (fn [state step-fn]
                        (step-fn state mlx-args constraints))
                      {:values {} :score (mx/scalar 0.0) :weight (mx/scalar 0.0) :key key}
                      step-fns)]
                 {:values (:values result)
                  :score (:score result)
                  :weight (:weight result)}))
         :addrs addrs}))))

(defn get-compiled-generate
  "Returns the compiled-generate function for a gen-fn, or nil."
  [gf]
  (:compiled-generate (:schema gf)))

;; ===========================================================================
;; WP-3: Compiled Update for Static Models
;; ===========================================================================
;;
;; Architecture: same as compiled generate, but NO sampling. Values come from
;; constraints (case 1) or old-choices (case 2). Log-prob computed with CURRENT
;; distribution params (which may differ from old trace if upstream changed).
;; Weight = new-score - old-score, computed in DynamicGF.update (not here).

(defn- build-update-site-step
  "Build the update step for one trace site.
   Returns (fn [state args-vec constraints old-choices] -> state) where state has
   {:values :score :discard :key}, or nil if dist-type has no noise transform."
  [site-spec]
  (let [{:keys [addr compiled-args dist-type]} site-spec
        nt (get compiled/noise-transforms-full dist-type)]
    (when nt
      (let [log-prob-fn (:log-prob nt)]
        (fn [{:keys [values score discard key]} args-vec constraints old-choices]
          (let [constraint (cm/get-submap constraints addr)
                eval-args (mapv #(% values args-vec) compiled-args)]
            (if (cm/has-value? constraint)
              ;; Case 1: Constrained — use new value, discard old
              (let [value (cm/get-value constraint)
                    lp (apply log-prob-fn value eval-args)
                    old-val (cm/get-value (cm/get-submap old-choices addr))]
                {:values (assoc values addr value)
                 :score (mx/add score lp)
                 :discard (assoc discard addr old-val)
                 :key key})
              ;; Case 2: Unconstrained — keep old value, re-score with current params
              (let [value (cm/get-value (cm/get-submap old-choices addr))
                    lp (apply log-prob-fn value eval-args)]
                {:values (assoc values addr value)
                 :score (mx/add score lp)
                 :discard discard
                 :key key}))))))))

(defn make-compiled-update
  "Build a compiled update function from a gen schema and source.
   Returns (fn [key args-vec constraints old-choices]
             -> {:values :score :discard :retval}) or nil."
  [schema source]
  (when-let [{:keys [site-specs retval-fn]} (compiled/prepare-static-sites schema source)]
    (let [step-fns (mapv build-update-site-step site-specs)]
      (when (and (every? some? step-fns) retval-fn)
        (fn compiled-update [key args-vec constraints old-choices]
          (let [mlx-args (compiled/ensure-mlx-args args-vec)
                result
                (reduce
                 (fn [state step-fn]
                   (step-fn state mlx-args constraints old-choices))
                 {:values {} :score (mx/scalar 0.0) :discard {} :key key}
                 step-fns)]
            {:values (:values result)
             :score (:score result)
             :discard (:discard result)
             :retval (when retval-fn
                       (retval-fn (:values result) mlx-args))}))))))

(defn get-compiled-update
  "Returns the compiled-update function for a gen-fn, or nil."
  [gf]
  (:compiled-update (:schema gf)))

(defn make-branch-rewritten-update
  "Build a compiled update for models with rewritable branches (L1-M4).
   Returns (fn [key args-vec constraints old-choices]
             -> {:values :score :discard :retval}) or nil."
  [schema source]
  (when-let [{:keys [site-specs retval-fn]} (compiled/prepare-branch-sites schema source)]
    (let [step-fns (mapv build-update-site-step site-specs)]
      (when (every? some? step-fns)
        (fn compiled-branch-update [key args-vec constraints old-choices]
          (let [mlx-args (compiled/ensure-mlx-args args-vec)
                result
                (reduce
                 (fn [state step-fn]
                   (step-fn state mlx-args constraints old-choices))
                 {:values {} :score (mx/scalar 0.0) :discard {} :key key}
                 step-fns)]
            {:values (:values result)
             :score (:score result)
             :discard (:discard result)
             :retval (when retval-fn
                       (retval-fn (:values result) mlx-args))}))))))

(defn make-compiled-prefix-update
  "Build a compiled prefix update function.
   Returns {:fn compiled-fn :addrs [keyword...]} or nil."
  [schema source]
  (when-let [{:keys [compiled-sites addrs]} (compiled/prepare-prefix-sites schema source)]
    (let [step-fns (mapv build-update-site-step compiled-sites)]
      (when (every? some? step-fns)
        {:fn (fn compiled-prefix-update [key args-vec constraints old-choices]
               (let [mlx-args (compiled/ensure-numeric-mlx-args args-vec)
                     result
                     (reduce
                      (fn [state step-fn]
                        (step-fn state mlx-args constraints old-choices))
                      {:values {} :score (mx/scalar 0.0) :discard {} :key key}
                      step-fns)]
                 {:values (:values result)
                  :score (:score result)
                  :discard (:discard result)}))
         :addrs addrs}))))

(defn make-replay-update-transition
  "Build a replay transition for partial update compilation.
   At prefix sites: return pre-computed value, no key split, no score/discard
   modification (already counted in compiled prefix).
   At other sites: delegate to h/update-transition."
  [compiled-values]
  (fn [state addr dist]
    (if (contains? compiled-values addr)
      ;; Replay: set pre-computed value. No key split (update never splits keys
      ;; for constrained/unconstrained-with-old-choice cases).
      (let [value (get compiled-values addr)]
        [value (update state :choices cm/set-value addr value)])
      ;; Dynamic site: standard update
      (h/update-transition state addr dist))))

;; ===========================================================================
;; WP-5: Compiled Assess
;; ===========================================================================
;;
;; Assess: all choices provided, compute total log-prob. No sampling, no key.
;; Simplest compiled operation — only log-prob functions needed.

(defn- build-assess-site-step
  "Build the assess step for one trace site.
   Returns (fn [state args-vec choices] -> state) where state has
   {:values :score}. Extracts value from choices, computes log-prob."
  [site-spec]
  (let [{:keys [addr compiled-args dist-type]} site-spec
        nt (get compiled/noise-transforms-full dist-type)]
    (when nt
      (let [log-prob-fn (:log-prob nt)]
        (fn [{:keys [values score]} args-vec choices]
          (let [value (cm/get-value (cm/get-submap choices addr))
                eval-args (mapv #(% values args-vec) compiled-args)
                lp (apply log-prob-fn value eval-args)]
            {:values (assoc values addr value)
             :score (mx/add score lp)}))))))

(defn make-compiled-assess
  "Build a compiled assess function. Returns (fn [args-vec choices] -> {:score :retval}) or nil."
  [schema source]
  (when-let [{:keys [site-specs retval-fn]} (compiled/prepare-static-sites schema source)]
    (let [step-fns (mapv build-assess-site-step site-specs)]
      (when (and (every? some? step-fns) retval-fn)
        (fn compiled-assess [args-vec choices]
          (let [mlx-args (compiled/ensure-mlx-args args-vec)
                result
                (reduce
                 (fn [state step-fn]
                   (step-fn state mlx-args choices))
                 {:values {} :score (mx/scalar 0.0)}
                 step-fns)]
            {:score (:score result)
             :retval (retval-fn (:values result) mlx-args)}))))))

(defn make-branch-rewritten-assess
  "Build a compiled assess for branch-rewritten models (L1-M4).
   Returns (fn [args-vec choices] -> {:score :retval}) or nil."
  [schema source]
  (when-let [{:keys [site-specs retval-fn]} (compiled/prepare-branch-sites schema source)]
    (let [step-fns (mapv build-assess-site-step site-specs)]
      (when (every? some? step-fns)
        (fn compiled-branch-assess [args-vec choices]
          (let [mlx-args (compiled/ensure-mlx-args args-vec)
                result
                (reduce
                 (fn [state step-fn]
                   (step-fn state mlx-args choices))
                 {:values {} :score (mx/scalar 0.0)}
                 step-fns)]
            {:score (:score result)
             :retval (when retval-fn
                       (retval-fn (:values result) mlx-args))}))))))

(defn make-compiled-prefix-assess
  "Build a compiled prefix assess function.
   Returns {:fn compiled-fn :addrs [keyword...]} or nil."
  [schema source]
  (when-let [{:keys [compiled-sites addrs]} (compiled/prepare-prefix-sites schema source)]
    (let [step-fns (mapv build-assess-site-step compiled-sites)]
      (when (every? some? step-fns)
        {:fn (fn compiled-prefix-assess [args-vec choices]
               (let [mlx-args (compiled/ensure-numeric-mlx-args args-vec)
                     result
                     (reduce
                      (fn [state step-fn]
                        (step-fn state mlx-args choices))
                      {:values {} :score (mx/scalar 0.0)}
                      step-fns)]
                 {:values (:values result)
                  :score (:score result)}))
         :addrs addrs}))))

(defn make-replay-assess-transition
  "Build a replay transition for partial assess compilation.
   At prefix sites: return pre-computed value, no key split, no score
   modification (already counted in compiled prefix).
   At other sites: delegate to h/assess-transition."
  [compiled-values]
  (fn [state addr dist]
    (if (contains? compiled-values addr)
      (let [value (get compiled-values addr)]
        [value (update state :choices cm/set-value addr value)])
      (h/assess-transition state addr dist))))

(defn get-compiled-assess
  "Returns the compiled-assess function for a gen-fn, or nil."
  [gf]
  (:compiled-assess (:schema gf)))

;; ===========================================================================
;; WP-5: Compiled Project
;; ===========================================================================
;;
;; Project: compute log-prob of selected addresses in a trace. No sampling.

(defn- build-project-site-step
  "Build the project step for one trace site.
   Returns (fn [state args-vec old-choices selection] -> state) where state has
   {:values :score :weight}. Replays value from old-choices, accumulates
   log-prob in score and (if selected) in weight."
  [site-spec]
  (let [{:keys [addr compiled-args dist-type]} site-spec
        nt (get compiled/noise-transforms-full dist-type)]
    (when nt
      (let [log-prob-fn (:log-prob nt)]
        (fn [{:keys [values score weight]} args-vec old-choices selection]
          (let [value (cm/get-value (cm/get-submap old-choices addr))
                eval-args (mapv #(% values args-vec) compiled-args)
                lp (apply log-prob-fn value eval-args)]
            {:values (assoc values addr value)
             :score (mx/add score lp)
             :weight (if (sel/selected? selection addr)
                       (mx/add weight lp)
                       weight)}))))))

(defn make-compiled-project
  "Build a compiled project function from a gen schema and source.
   Returns (fn [args-vec old-choices selection] -> scalar) or nil.
   No key parameter — project never samples."
  [schema source]
  (when (and (:static? schema)
             (seq (:trace-sites schema))
             (empty? (:splice-sites schema))
             (empty? (:param-sites schema)))
    (let [binding-env (compiled/build-binding-env source)
          static-sites (filterv :static? (:trace-sites schema))
          site-specs
          (mapv (fn [ts]
                  (let [cargs (mapv #(compiled/compile-expr % binding-env #{})
                                    (:dist-args ts))]
                    (when (every? some? cargs)
                      {:addr (:addr ts)
                       :compiled-args cargs
                       :dist-type (:dist-type ts)})))
                static-sites)
          step-fns (when (every? some? site-specs)
                     (mapv build-project-site-step site-specs))]
      (when (and step-fns (every? some? step-fns))
        (fn compiled-project [args-vec old-choices selection]
          (let [mlx-args (compiled/ensure-mlx-args args-vec)
                result
                (reduce
                 (fn [state step-fn]
                   (step-fn state mlx-args old-choices selection))
                 {:values {} :score (mx/scalar 0.0) :weight (mx/scalar 0.0)}
                 step-fns)]
            (:weight result)))))))

(defn make-branch-rewritten-project
  "Build a compiled project for models with rewritable branches (L1-M4).
   Returns (fn [args-vec old-choices selection] -> scalar) or nil."
  [schema source]
  (when (and (:has-branches? schema)
             (empty? (:splice-sites schema))
             (empty? (:param-sites schema))
             (not (:has-loops? schema))
             (not (:dynamic-addresses? schema)))
    (when-let [raw-sites (compiled/extract-rewritable-sites source)]
      (when (seq raw-sites)
        (when-let [{:keys [site-specs retval-fn addrs]}
                   (compiled/compile-branch-rewritten-site-specs schema source raw-sites)]
          (let [step-fns (mapv build-project-site-step site-specs)]
            (when (every? some? step-fns)
              (fn compiled-branch-project [args-vec old-choices selection]
                (let [mlx-args (compiled/ensure-mlx-args args-vec)
                      result
                      (reduce
                       (fn [state step-fn]
                         (step-fn state mlx-args old-choices selection))
                       {:values {} :score (mx/scalar 0.0) :weight (mx/scalar 0.0)}
                       step-fns)]
                  (:weight result))))))))))

(defn make-compiled-prefix-project
  "Build a compiled prefix project function.
   Returns {:fn (fn [args-vec old-choices selection] -> {:values :weight})
            :addrs [keyword...]} or nil."
  [schema source]
  (when (and (not (:static? schema))
             (seq (:trace-sites schema))
             (empty? (:splice-sites schema))
             (empty? (:param-sites schema)))
    (let [raw-prefix (compiled/extract-prefix-sites source)]
      (when (seq raw-prefix)
        (let [binding-env (compiled/build-binding-env source)
              compiled-sites
              (reduce
               (fn [acc site]
                 (let [cargs (mapv #(compiled/compile-expr % binding-env #{})
                                   (:dist-args site))
                       nt (get compiled/noise-transforms-full (:dist-type site))]
                   (if (and nt (every? some? cargs))
                     (conj acc (assoc site :compiled-args cargs))
                     (reduced acc))))
               []
               raw-prefix)]
          (when (seq compiled-sites)
            (let [step-fns (mapv build-project-site-step compiled-sites)
                  addrs (mapv :addr compiled-sites)]
              (when (every? some? step-fns)
                {:fn (fn compiled-prefix-project [args-vec old-choices selection]
                       (let [mlx-args (compiled/ensure-numeric-mlx-args args-vec)
                             result
                             (reduce
                              (fn [state step-fn]
                                (step-fn state mlx-args old-choices selection))
                              {:values {} :score (mx/scalar 0.0) :weight (mx/scalar 0.0)}
                              step-fns)]
                         {:values (:values result)
                          :weight (:weight result)}))
                 :addrs addrs}))))))))

(defn make-replay-project-transition
  "Build a replay transition for partial project compilation.
   At prefix sites: return pre-computed value, no key split, no score/weight
   modification (already counted in compiled prefix).
   At other sites: delegate to h/project-transition."
  [compiled-values]
  (fn [state addr dist]
    (if (contains? compiled-values addr)
      (let [value (get compiled-values addr)]
        [value (update state :choices cm/set-value addr value)])
      (h/project-transition state addr dist))))

(defn get-compiled-project
  "Returns the compiled-project function for a gen-fn, or nil."
  [gf]
  (:compiled-project (:schema gf)))

;; ===========================================================================
;; WP-6: Compiled Regenerate
;; ===========================================================================
;;
;; Regenerate: resample selected sites, keep unselected. Compute proposal ratio
;; (weight) = sum over selected sites of (new-lp - old-lp). DynamicGF computes
;; final weight = new_score - old_score - proposal_ratio.
;; No mx/compile-fn: selection check is data-dependent.

(defn- build-regenerate-site-step
  "Build the regenerate step for one trace site.
   Returns (fn [state args-vec old-choices selection] -> state) where state has
   {:values :score :weight :key}.
   Selected: resample via noise transform, weight += new-lp - old-lp.
   Unselected: keep old value, score old-lp, weight unchanged, NO key split."
  [site-spec]
  (let [{:keys [addr compiled-args dist-type]} site-spec
        nt (get compiled/noise-transforms-full dist-type)]
    (when nt
      (let [log-prob-fn (:log-prob nt)]
        (if (:noise-fn nt)
          ;; Standard distribution with noise transform
          (let [noise-fn (:noise-fn nt)
                transform-fn (:transform nt)]
            (fn [{:keys [values score weight key]} args-vec old-choices selection]
              (let [eval-args (mapv #(% values args-vec) compiled-args)]
                (if (sel/selected? selection addr)
                  ;; Selected: resample via noise transform
                  (let [[k1 k2] (rng/split key)
                        noise (noise-fn k2)
                        new-val (apply transform-fn noise eval-args)
                        new-lp (apply log-prob-fn new-val eval-args)
                        old-val (cm/get-value (cm/get-submap old-choices addr))
                        old-lp (apply log-prob-fn old-val eval-args)]
                    {:values (assoc values addr new-val)
                     :score (mx/add score new-lp)
                     :weight (mx/add weight (mx/subtract new-lp old-lp))
                     :key k1})
                  ;; Not selected: keep old value, no key split
                  (let [val (cm/get-value (cm/get-submap old-choices addr))
                        lp (apply log-prob-fn val eval-args)]
                    {:values (assoc values addr val)
                     :score (mx/add score lp)
                     :weight weight
                     :key key})))))
          ;; Delta distribution: no noise, value = first arg
          (fn [{:keys [values score weight key]} args-vec old-choices selection]
            (let [eval-args (mapv #(% values args-vec) compiled-args)]
              (if (sel/selected? selection addr)
                ;; Selected delta: "resample" = same value (deterministic), lp = 0
                ;; Split key for PRNG equivalence with handler
                (let [[k1 _k2] (rng/split key)
                      new-val (first eval-args)]
                  ;; new-lp = 0, old-lp = 0 for delta → weight += 0
                  {:values (assoc values addr new-val)
                   :score score
                   :weight weight
                   :key k1})
                ;; Not selected: keep old value, lp = 0
                (let [val (cm/get-value (cm/get-submap old-choices addr))]
                  {:values (assoc values addr val)
                   :score score
                   :weight weight
                   :key key})))))))))

(defn make-compiled-regenerate
  "Build a compiled regenerate function from a gen schema and source.
   Returns (fn [key args-vec old-choices selection]
             -> {:values :score :weight :retval})
   or nil if the model can't be compiled.
   :weight = proposal ratio (NOT final weight — DynamicGF computes that)."
  [schema source]
  (when (and (:static? schema)
             (seq (:trace-sites schema))
             (empty? (:splice-sites schema))
             (empty? (:param-sites schema)))
    (let [binding-env (compiled/build-binding-env source)
          static-sites (filterv :static? (:trace-sites schema))
          site-specs
          (mapv (fn [ts]
                  (let [cargs (mapv #(compiled/compile-expr % binding-env #{})
                                    (:dist-args ts))]
                    (when (every? some? cargs)
                      {:addr (:addr ts)
                       :compiled-args cargs
                       :dist-type (:dist-type ts)})))
                static-sites)
          step-fns (when (every? some? site-specs)
                     (mapv build-regenerate-site-step site-specs))
          return-expr (compiled/extract-return-expr (:return-form schema))
          retval-fn (compiled/compile-expr return-expr binding-env #{})]
      (when (and step-fns (every? some? step-fns) retval-fn)
        (fn compiled-regenerate [key args-vec old-choices selection]
          (let [mlx-args (compiled/ensure-mlx-args args-vec)
                result
                (reduce
                 (fn [state step-fn]
                   (step-fn state mlx-args old-choices selection))
                 {:values {} :score (mx/scalar 0.0) :weight (mx/scalar 0.0) :key key}
                 step-fns)]
            {:values (:values result)
             :score (:score result)
             :weight (:weight result)
             :retval (when retval-fn
                       (retval-fn (:values result) mlx-args))}))))))

(defn make-branch-rewritten-regenerate
  "Build a compiled regenerate for models with rewritable branches (L1-M4).
   Returns (fn [key args-vec old-choices selection]
             -> {:values :score :weight :retval}) or nil."
  [schema source]
  (when (and (:has-branches? schema)
             (empty? (:splice-sites schema))
             (empty? (:param-sites schema))
             (not (:has-loops? schema))
             (not (:dynamic-addresses? schema)))
    (when-let [raw-sites (compiled/extract-rewritable-sites source)]
      (when (seq raw-sites)
        (when-let [{:keys [site-specs retval-fn addrs]}
                   (compiled/compile-branch-rewritten-site-specs schema source raw-sites)]
          (let [step-fns (mapv build-regenerate-site-step site-specs)]
            (when (every? some? step-fns)
              (fn compiled-branch-regenerate [key args-vec old-choices selection]
                (let [mlx-args (compiled/ensure-mlx-args args-vec)
                      result
                      (reduce
                       (fn [state step-fn]
                         (step-fn state mlx-args old-choices selection))
                       {:values {} :score (mx/scalar 0.0) :weight (mx/scalar 0.0) :key key}
                       step-fns)]
                  {:values (:values result)
                   :score (:score result)
                   :weight (:weight result)
                   :retval (when retval-fn
                             (retval-fn (:values result) mlx-args))})))))))))

(defn make-compiled-prefix-regenerate
  "Build a compiled prefix regenerate function.
   Returns {:fn (fn [key args-vec old-choices selection]
                  -> {:values :score :weight})
            :addrs [keyword...]}
   or nil if partial compilation isn't applicable."
  [schema source]
  (when (and (not (:static? schema))
             (seq (:trace-sites schema))
             (empty? (:splice-sites schema))
             (empty? (:param-sites schema)))
    (let [raw-prefix (compiled/extract-prefix-sites source)]
      (when (seq raw-prefix)
        (let [binding-env (compiled/build-binding-env source)
              compiled-sites
              (reduce
               (fn [acc site]
                 (let [cargs (mapv #(compiled/compile-expr % binding-env #{})
                                   (:dist-args site))
                       nt (get compiled/noise-transforms-full (:dist-type site))]
                   (if (and nt (every? some? cargs))
                     (conj acc (assoc site :compiled-args cargs))
                     (reduced acc))))
               []
               raw-prefix)]
          (when (seq compiled-sites)
            (let [step-fns (mapv build-regenerate-site-step compiled-sites)
                  addrs (mapv :addr compiled-sites)]
              (when (every? some? step-fns)
                {:fn (fn compiled-prefix-regenerate [key args-vec old-choices selection]
                       (let [mlx-args (compiled/ensure-numeric-mlx-args args-vec)
                             result
                             (reduce
                              (fn [state step-fn]
                                (step-fn state mlx-args old-choices selection))
                              {:values {} :score (mx/scalar 0.0) :weight (mx/scalar 0.0) :key key}
                              step-fns)]
                         {:values (:values result)
                          :score (:score result)
                          :weight (:weight result)}))
                 :addrs addrs}))))))))

(defn make-replay-regenerate-transition
  "Build a replay transition for partial regenerate compilation.
   At prefix sites: replay pre-computed value, split key for selected sites
   (matching handler's regenerate-transition), no split for unselected.
   Score/weight NOT modified (already counted in prefix result).
   At other sites: delegate to h/regenerate-transition."
  [compiled-values]
  (fn [state addr dist]
    (if (contains? compiled-values addr)
      (let [value (get compiled-values addr)
            selected? (sel/selected? (:selection state) addr)]
        [value (cond-> (update state :choices cm/set-value addr value)
                 selected? (#(let [[k1 _] (rng/split (:key %))]
                               (assoc % :key k1))))])
      (h/regenerate-transition state addr dist))))

(defn get-compiled-regenerate
  "Returns the compiled-regenerate function for a gen-fn, or nil."
  [gf]
  (:compiled-regenerate (:schema gf)))

;; ===========================================================================
;; Level 1-M5: Combinator-Aware Compilation Utility
;; ===========================================================================

(defn get-compiled-simulate
  "Returns the compiled-simulate function for a gen-fn, or nil if not compilable.
   Checks :compiled-simulate on (:schema gf)."
  [gf]
  (:compiled-simulate (:schema gf)))

;; ===========================================================================
;; WP-9B: Fused Loop Compilation
;; ===========================================================================
;;
;; Fuses unfold/scan loops into single mx/compile-fn invocations.
;; Pre-generates noise [T, K] on host, passes to compiled function.
;; The compiled function unrolls T steps with noise-indexed site steps.

(def ^:private noise-type-map
  "Maps distribution types to their noise source type (:normal or :uniform)."
  {:gaussian :normal, :normal :normal, :log-normal :normal,
   :uniform :uniform, :bernoulli :uniform, :flip :uniform,
   :exponential :uniform, :laplace :uniform, :cauchy :uniform})

(defn- build-fused-site-step
  "Build a site step that reads noise from noise-row[noise-index]
   instead of generating from a PRNG key.
   Returns (fn [{:keys [values score]} args-vec noise-row] -> {:values :score})
   or nil if the dist-type has no noise transform.
   For delta sites, noise-index is ignored (no noise consumed)."
  [site-spec noise-index]
  (let [{:keys [addr compiled-args dist-type]} site-spec
        nt (get compiled/noise-transforms-full dist-type)]
    (when nt
      (if (:noise-fn nt)
        ;; Standard distribution: extract noise from row at noise-index
        (let [transform-fn (:transform nt)
              log-prob-fn (:log-prob nt)]
          (fn [{:keys [values score]} args-vec noise-row]
            (let [eval-args (mapv #(% values args-vec) compiled-args)
                  noise (mx/index noise-row noise-index)
                  value (apply transform-fn noise eval-args)
                  lp (apply log-prob-fn value eval-args)]
              {:values (assoc values addr value)
               :score (mx/add score lp)})))
        ;; Delta: no noise needed
        (fn [{:keys [values score]} args-vec _noise-row]
          (let [eval-args (mapv #(% values args-vec) compiled-args)
                value (first eval-args)]
            {:values (assoc values addr value)
             :score score}))))))

(defn- assign-noise-indices
  "Assign sequential noise indices to site-specs that have noise-fns.
   Returns vector of indices (nil for delta/unsupported sites)."
  [site-specs]
  (second
   (reduce (fn [[idx acc] s]
             (if (and s (:noise-fn (get compiled/noise-transforms-full (:dist-type s))))
               [(inc idx) (conj acc idx)]
               [idx (conj acc nil)]))
           [0 []] site-specs)))

(defn- extract-noise-site-types
  "Filter site-specs to those with noise-fns."
  [site-specs]
  (filterv (fn [s] (and s (:noise-fn (get compiled/noise-transforms-full (:dist-type s)))))
           site-specs))

(defn generate-noise-matrix
  "Generate [T, K] noise matrix where each column has the correct distribution.
   noise-site-types: vector of {:dist-type ...} for noise-consuming sites.
   Returns [T, K] MLX array."
  [key T noise-site-types]
  (if (empty? noise-site-types)
    (mx/zeros [T 1])
    (let [cols (loop [sites noise-site-types, k key, cols []]
                 (if (empty? sites)
                   cols
                   (let [[k1 k2] (rng/split k)
                         site (first sites)
                         col (case (get noise-type-map (:dist-type site))
                               :normal (rng/normal k1 [T])
                               :uniform (rng/uniform k1 [T]))]
                     (recur (rest sites) k2 (conj cols col)))))]
      (if (= 1 (count cols))
        (mx/reshape (first cols) [T 1])
        (mx/stack cols 1)))))

(defn make-fused-unfold-simulate
  "Build a fused unfold simulate: T steps as single mx/compile-fn invocation.
   Auto-generates step function from kernel schema.
   Returns {:compiled-fn :noise-dim :addr-order :noise-site-types :extra-args :state-keys}
   or nil if kernel can't be fused.

   The compiled-fn signature:
     (fn [init-state noise-2d] -> [outputs-tensor [T,K+N], step-scores [T], total-score])
   where outputs columns 0..K-1 are site values in addr-order, columns K..K+N-1 are
   state values (N=1 for scalar state, N=len(state-keys) for map state)."
  [schema source T extra-args]
  (when (and (:static? schema)
             (seq (:trace-sites schema))
             (empty? (:splice-sites schema))
             (empty? (:param-sites schema)))
    (let [binding-env (compiled/build-binding-env source)
          static-sites (filterv :static? (:trace-sites schema))
          site-specs (compiled/build-fused-site-specs static-sites binding-env)
          noise-indices (assign-noise-indices site-specs)
          noise-site-types (extract-noise-site-types site-specs)
          noise-dim (count noise-site-types)
          fused-steps (mapv (fn [spec ni] (build-fused-site-step spec ni))
                            site-specs noise-indices)
          ;; Compile return expression — detect map-valued state
          return-expr (compiled/extract-return-expr (:return-form schema))
          map-state? (map? return-expr)
          state-keys (when map-state? (vec (sort (keys return-expr))))
          n-state (if map-state? (count state-keys) 1)
          retval-fn (compiled/compile-expr return-expr binding-env #{})
          addr-order (mapv :addr static-sites)
          n-sites (count static-sites)]
      (when (and (every? some? site-specs)
                 (every? some? fused-steps)
                 retval-fn
                 (pos? noise-dim))
        (let [extra-arrs (mapv mx/ensure-array extra-args)
              ;; Build the fused loop function
              unfold-fn
              (fn [init-state noise-2d]
                (loop [t 0
                       state init-state
                       total-score (mx/scalar 0.0)
                       outputs []
                       scores []]
                  (if (>= t T)
                    [(mx/stack outputs) (mx/stack scores) total-score]
                    (let [t-arr (mx/scalar (float t))
                          ;; Unpack flat state to map for map-state kernels
                          state-for-args (if state-keys
                                          (into {} (map-indexed
                                                    (fn [i k] [k (mx/index state i)])
                                                    state-keys))
                                          state)
                          args-vec (into [t-arr state-for-args] extra-arrs)
                          noise-row (mx/index noise-2d t)
                          result (reduce
                                  (fn [st step-f] (step-f st args-vec noise-row))
                                  {:values {} :score (mx/scalar 0.0)}
                                  fused-steps)
                          new-state (retval-fn (:values result) args-vec)
                          step-score (:score result)
                          ;; Pack state into flat values for the output row
                          site-vals (mapv #(get (:values result) %) addr-order)
                          new-state-flat (if state-keys
                                          (mx/stack (mapv #(get new-state %) state-keys))
                                          new-state)
                          row (if state-keys
                                (mx/stack (into site-vals
                                                (mapv #(get new-state %) state-keys)))
                                (mx/stack (conj site-vals new-state)))]
                      (recur (inc t)
                             new-state-flat
                             (mx/add total-score step-score)
                             (conj outputs row)
                             (conj scores step-score))))))
              compiled (mx/compile-fn unfold-fn)]
          ;; Warm up with dummy data
          (let [dummy-state (if state-keys
                              (mx/zeros [n-state])
                              (mx/scalar 0.0))
                dummy-noise (mx/zeros [T (max 1 noise-dim)])]
            (let [[outputs scores sc] (compiled dummy-state dummy-noise)]
              (mx/materialize! outputs scores sc)))
          {:compiled-fn compiled
           :noise-dim noise-dim
           :addr-order addr-order
           :noise-site-types noise-site-types
           :extra-args extra-args
           :state-keys state-keys})))))

(defn make-fused-scan-simulate
  "Build a fused scan simulate: T steps as single mx/compile-fn invocation.
   Scan kernel takes [carry input] and returns [new-carry output].
   Returns {:compiled-fn :noise-dim :addr-order :noise-site-types}
   or nil if kernel can't be fused.

   The compiled-fn signature:
     (fn [init-carry inputs-tensor noise-2d] -> [outputs-tensor [T,K+2], step-scores [T], total-score])
   where outputs columns: 0..K-1 site values, K carry, K+1 output."
  [schema source T]
  (when (and (:static? schema)
             (seq (:trace-sites schema))
             (empty? (:splice-sites schema))
             (empty? (:param-sites schema)))
    (let [binding-env (compiled/build-binding-env source)
          static-sites (filterv :static? (:trace-sites schema))
          site-specs (compiled/build-fused-site-specs static-sites binding-env)
          noise-indices (assign-noise-indices site-specs)
          noise-site-types (extract-noise-site-types site-specs)
          noise-dim (count noise-site-types)
          fused-steps (mapv (fn [spec ni] (build-fused-site-step spec ni))
                            site-specs noise-indices)
          ;; For scan, return form should be a vector [carry-expr output-expr]
          return-expr (compiled/extract-return-expr (:return-form schema))
          carry-fn (when (vector? return-expr)
                     (compiled/compile-expr (first return-expr) binding-env #{}))
          output-fn (when (vector? return-expr)
                      (compiled/compile-expr (second return-expr) binding-env #{}))
          addr-order (mapv :addr static-sites)
          n-sites (count static-sites)]
      (when (and (every? some? site-specs)
                 (every? some? fused-steps)
                 carry-fn output-fn
                 (pos? noise-dim))
        (let [scan-fn
              (fn [init-carry inputs-tensor noise-2d]
                (loop [t 0
                       carry init-carry
                       total-score (mx/scalar 0.0)
                       outputs []
                       scores []]
                  (if (>= t T)
                    [(mx/stack outputs) (mx/stack scores) total-score]
                    (let [input-t (mx/index inputs-tensor t)
                          args-vec [carry input-t]
                          noise-row (mx/index noise-2d t)
                          result (reduce
                                  (fn [st step-f] (step-f st args-vec noise-row))
                                  {:values {} :score (mx/scalar 0.0)}
                                  fused-steps)
                          new-carry (carry-fn (:values result) args-vec)
                          output-val (output-fn (:values result) args-vec)
                          step-score (:score result)
                          site-vals (mapv #(get (:values result) %) addr-order)
                          row (mx/stack (into (conj site-vals new-carry) [output-val]))]
                      (recur (inc t)
                             new-carry
                             (mx/add total-score step-score)
                             (conj outputs row)
                             (conj scores step-score))))))
              compiled (mx/compile-fn scan-fn)]
          ;; Warm up
          (let [dummy-carry (mx/scalar 0.0)
                dummy-inputs (mx/zeros [T])
                dummy-noise (mx/zeros [T (max 1 noise-dim)])]
            (let [[outputs scores sc] (compiled dummy-carry dummy-inputs dummy-noise)]
              (mx/materialize! outputs scores sc)))
          {:compiled-fn compiled
           :noise-dim noise-dim
           :addr-order addr-order
           :noise-site-types noise-site-types})))))

(defn fusable-kernel?
  "Check if a kernel can be fused into a single Metal dispatch.
   Returns true if the kernel has a static schema with noise transforms
   for all non-delta trace sites."
  [gf]
  (let [schema (:schema gf)]
    (and schema
         (:static? schema)
         (seq (:trace-sites schema))
         (empty? (:splice-sites schema))
         (empty? (:param-sites schema))
         (let [nts (mapv #(get compiled/noise-transforms-full (:dist-type %))
                         (filterv :static? (:trace-sites schema)))]
           (and (every? some? nts)
                (some :noise-fn nts))))))

(defn make-fused-map-simulate
  "Build a fused map simulate that processes all N elements in one call.
   No mx/compile-fn needed — MLX broadcasting handles [N]-shaped arrays.
   Stacks element args into [N]-shaped arrays, pre-generates [N] noise per site,
   runs site steps once with broadcasting.

   Returns (fn [key stacked-args] -> {:values {addr -> [N]-arr} :scores [N]-arr :retval [N]-arr})
   or nil if kernel can't be fused.

   stacked-args: vector of [N]-shaped arrays (one per kernel param)."
  [schema source]
  (when (and (:static? schema)
             (seq (:trace-sites schema))
             (empty? (:splice-sites schema))
             (empty? (:param-sites schema)))
    (let [binding-env (compiled/build-binding-env source)
          static-sites (filterv :static? (:trace-sites schema))
          site-specs (compiled/build-fused-site-specs static-sites binding-env)
          noise-indices (assign-noise-indices site-specs)
          noise-site-types (extract-noise-site-types site-specs)
          noise-dim (count noise-site-types)
          fused-steps (mapv (fn [spec ni] (build-fused-site-step spec ni))
                            site-specs noise-indices)
          return-expr (compiled/extract-return-expr (:return-form schema))
          retval-fn (compiled/compile-expr return-expr binding-env #{})
          addr-order (mapv :addr static-sites)]
      (when (and (every? some? site-specs)
                 (every? some? fused-steps)
                 retval-fn
                 (pos? noise-dim))
        {:fused-fn
         (fn [key stacked-args N]
           ;; stacked-args: vector of [N]-shaped arrays (one per kernel param)
           ;; Pre-generate [N, K] noise, transpose to [K, N] so mx/index returns [N]
           (let [noise-2d (generate-noise-matrix key N noise-site-types)
                 ;; Transpose: [N, K] → [K, N]. mx/index on [K, N] at idx i → [N]
                 noise-cols (mx/transpose noise-2d)
                 result (reduce
                         (fn [st step-f] (step-f st stacked-args noise-cols))
                         {:values {} :score (mx/zeros [N])}
                         fused-steps)
                 retval (retval-fn (:values result) stacked-args)]
             {:values (:values result)
              :scores (:score result)
              :retval retval}))
         :noise-site-types noise-site-types
         :addr-order addr-order}))))

;; ===========================================================================
;; Level 2: Tensor-Native Score Function
;; ===========================================================================
;;
;; Bypasses GFI protocol entirely. Takes a [K] tensor of latent values,
;; uses L1 noise-transform :log-prob closures to compute total log-prob.
;; Observations are baked in as constants.
;;
;; This is the key building block for Level 2 compiled inference:
;; - Compiled MCMC inner loops use tensor-score instead of p/generate
;; - Compiled SMC extend steps use tensor-score for weight computation

(defn make-tensor-score
  "Build a tensor-native score function: [K]-tensor → scalar log-prob.
   Bypasses GFI protocol — uses L1 noise-transform log-prob closures directly.
   Observations are baked in as constants. Only latent values come from the tensor.

   Returns (fn [latent-tensor] -> MLX scalar) or nil if model can't be compiled.

   latent-tensor: [K] MLX array where K = number of latent sites.
   The addr-index for the tensor is returned as metadata via make-tensor-score-with-index.

   schema: the :schema from a DynamicGF
   source: the :source from a DynamicGF
   args: argument vector (will be converted to MLX arrays)
   observations: ChoiceMap of observed values"
  [schema source args observations]
  (when (and schema (:static? schema)
             (seq (:trace-sites schema))
             (empty? (:splice-sites schema))
             (empty? (:param-sites schema)))
    (let [binding-env (compiled/build-binding-env source)
          static-sites (filterv :static? (:trace-sites schema))
          site-specs (compiled/build-fused-site-specs static-sites binding-env)]
      (when (every? some? site-specs)
        (let [mlx-args (compiled/ensure-mlx-args (vec args))
              ;; Separate observed vs latent using source-order static-sites
              ;; (matches L1 compiled paths in compiled/prepare-static-sites)
              all-addrs (mapv :addr static-sites)
              obs-addrs (set (map first (cm/addresses observations)))
              latent-addrs (vec (remove obs-addrs all-addrs))
              latent-index (into {} (map-indexed (fn [i a] [a i]) latent-addrs))
              ;; Pre-extract observed values
              obs-values (into {} (keep (fn [addr]
                                          (when (obs-addrs addr)
                                            (let [sub (cm/get-submap observations addr)]
                                              (when (cm/has-value? sub)
                                                [addr (cm/get-value sub)]))))
                                  all-addrs))
              ;; Build per-site log-prob step functions
              ;; Each returns (fn [values-map] -> log-prob-scalar) or nil
              site-lp-fns
              (mapv
                (fn [site-spec]
                  (let [{:keys [addr compiled-args dist-type]} site-spec
                        nt (get compiled/noise-transforms-full dist-type)]
                    (when nt
                      (let [log-prob-fn (:log-prob nt)]
                        (fn [values-map]
                          (let [eval-args (mapv #(% values-map mlx-args) compiled-args)]
                            (apply log-prob-fn (get values-map addr) eval-args)))))))
                site-specs)]
          (when (every? some? site-lp-fns)
            ;; Build the tensor-score closure
            (let [dep-order (:dep-order schema)]
              (fn tensor-score [latent-tensor]
                ;; Build values-map: latent from tensor, observed baked in
                (let [values-map
                      (reduce
                        (fn [vm addr]
                          (assoc vm addr
                                 (if-let [idx (get latent-index addr)]
                                   (mx/index latent-tensor idx)
                                   (get obs-values addr))))
                        {}
                        dep-order)]
                  ;; Sum all site log-probs
                  (reduce
                    (fn [score lp-fn]
                      (mx/add score (lp-fn values-map)))
                    (mx/scalar 0.0)
                    site-lp-fns))))))))))

(defn make-tensor-score-with-index
  "Like make-tensor-score but also returns the latent addr-index.
   Returns {:score-fn (fn [K-tensor] -> scalar) :latent-index {addr -> int}} or nil."
  [schema source args observations]
  (when (and schema (:static? schema)
             (seq (:trace-sites schema))
             (empty? (:splice-sites schema))
             (empty? (:param-sites schema)))
    (let [static-sites (filterv :static? (:trace-sites schema))
          obs-addrs (set (map first (cm/addresses observations)))
          latent-addrs (vec (remove obs-addrs (mapv :addr static-sites)))
          latent-index (into {} (map-indexed (fn [i a] [a i]) latent-addrs))
          score-fn (make-tensor-score schema source args observations)]
      (when score-fn
        {:score-fn score-fn
         :latent-index latent-index}))))

;; =========================================================================
;; Compiled SMC extend step (L2 WP-2)
;; =========================================================================

(defn make-smc-extend-step
  "Build a compiled SMC extend step for a kernel's schema.
   Returns (fn [noise-slice kernel-args observations]
              -> {:values-map {addr -> [N]-array} :log-prob [N]-array})
   or nil if kernel can't be compiled.

   noise-slice: [N,K_latent] standard normal noise
   kernel-args: vector of kernel arguments (will be converted to MLX)
   observations: ChoiceMap of observed values for this step

   The returned values-map maps each address to an [N]-shaped MLX array.
   log-prob is [N]-shaped total log-probability per particle."
  [schema source]
  (when (and schema (:static? schema)
             (seq (:trace-sites schema))
             (empty? (:splice-sites schema))
             (empty? (:param-sites schema)))
    (let [binding-env (compiled/build-binding-env source)
          static-sites (filterv :static? (:trace-sites schema))
          site-specs (compiled/build-fused-site-specs static-sites binding-env)]
      (when (every? some? site-specs)
        (let [dep-order (:dep-order schema)
              all-addrs (mapv :addr static-sites)
              addr-index (into {} (map-indexed (fn [i a] [a i]) all-addrs))
              K (count all-addrs)
              ;; Compile the return expression for state propagation
              retval-fn (when-let [rf (:return-form schema)]
                          (compiled/compile-expr rf binding-env #{}))]
          (fn smc-extend [noise-slice kernel-args observations]
            (let [N (first (mx/shape noise-slice))
                  mlx-args (compiled/ensure-mlx-args (vec kernel-args))
                  ;; Figure out latent vs observed
                  obs-addrs (set (map first (cm/addresses observations)))
                  latent-addrs (vec (remove obs-addrs all-addrs))
                  latent-index (into {} (map-indexed (fn [i a] [a i]) latent-addrs))
                  ;; Pre-extract noise columns: transpose [N,K] → [K,N], index rows
                  noise-transposed (mx/transpose noise-slice)
                  noise-cols (mapv (fn [k] (mx/index noise-transposed k))
                                   (range (count latent-addrs)))
                  ;; Pre-extract observed values, broadcast to [N]
                  obs-values (into {} (keep (fn [addr]
                                              (when (obs-addrs addr)
                                                (let [sub (cm/get-submap observations addr)]
                                                  (when (cm/has-value? sub)
                                                    (let [v (cm/get-value sub)]
                                                      [addr (mx/broadcast-to v [N])])))))
                                            all-addrs))
                  ;; Build values-map in dependency order
                  ;; Latent sites: propose via noise transform
                  ;; Observed sites: use baked-in value (already [N])
                  values-map
                  (reduce
                    (fn [vm addr]
                      (if-let [idx (get latent-index addr)]
                        ;; Latent: noise transform
                        (let [site-idx (get addr-index addr)
                              site-spec (nth site-specs site-idx)
                              nt (get compiled/noise-transforms-full (:dist-type site-spec))
                              noise-col (nth noise-cols idx)]
                          (if (:noise-fn nt)
                            (let [eval-args (mapv #(% vm mlx-args) (:compiled-args site-spec))
                                  proposed (apply (:transform nt) noise-col eval-args)]
                              (assoc vm addr proposed))
                            ;; Delta: value = first dist arg
                            (let [eval-args (mapv #(% vm mlx-args) (:compiled-args site-spec))]
                              (assoc vm addr (first eval-args)))))
                        ;; Observed: bake in constant
                        (assoc vm addr (get obs-values addr))))
                    {}
                    dep-order)
                  ;; Single-pass log-prob accumulation split by latent vs observed
                  ;; For bootstrap PF: weight = obs log-prob only
                  {:keys [latent-log-prob obs-log-prob]}
                  (reduce
                    (fn [{:keys [latent-log-prob obs-log-prob]} ss]
                      (let [{:keys [addr compiled-args dist-type]} ss
                            nt (get compiled/noise-transforms-full dist-type)
                            v (get values-map addr)
                            eval-args (mapv #(% values-map mlx-args) compiled-args)
                            lp (apply (:log-prob nt) v eval-args)]
                        (if (obs-addrs addr)
                          {:latent-log-prob latent-log-prob
                           :obs-log-prob (mx/add obs-log-prob lp)}
                          {:latent-log-prob (mx/add latent-log-prob lp)
                           :obs-log-prob obs-log-prob})))
                    {:latent-log-prob (mx/zeros [N])
                     :obs-log-prob (mx/zeros [N])}
                    site-specs)
                  total-log-prob (mx/add latent-log-prob obs-log-prob)]
              {:values-map values-map
               :log-prob total-log-prob
               :latent-log-prob latent-log-prob
               :obs-log-prob obs-log-prob
               :addr-index addr-index
               :all-addrs all-addrs
               :retval (when retval-fn (retval-fn values-map mlx-args))})))))))

