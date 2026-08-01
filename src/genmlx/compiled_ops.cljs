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
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
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
      (let [log-prob-fn (:log-prob nt)
            ;; Shared constrained/sample logic for both noise sources,
            ;; mirroring build-regenerate-site-step's make-noise-step factoring
            ;; (genmlx-b210): the branches differ only in how noise is drawn.
            make-noise-step
            (fn [draw-noise]
              (let [transform-fn (:transform nt)]
                (fn [{:keys [values score weight key]} args-vec constraints]
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
                            noise (draw-noise eval-args k2)
                            value (apply transform-fn noise eval-args)
                            lp (apply log-prob-fn value eval-args)]
                        {:values (assoc values addr value)
                         :score (mx/add score lp)
                         :weight weight
                         :key k1}))))))]
        (cond
          (:noise-fn nt)
          ;; Standard distribution with noise transform
          (let [noise-fn (:noise-fn nt)]
            (make-noise-step (fn [_eval-args k] (noise-fn k))))

          (:args-noise-fn nt)
          ;; Dynamic-shape distribution (e.g., iid-gaussian): noise shape
          ;; depends on dist-args, so use args-noise-fn
          (let [args-noise-fn (:args-noise-fn nt)]
            (make-noise-step (fn [eval-args k] (args-noise-fn eval-args k))))

          ;; Delta ONLY when the dist-type really is delta (genmlx-b210)
          (= dist-type :delta)
          (fn [{:keys [values score weight key]} args-vec constraints]
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
                {:keys [values score weight]}
                (reduce
                 (fn [state step-fn]
                   (step-fn state mlx-args constraints))
                 {:values {} :score (mx/scalar 0.0) :weight (mx/scalar 0.0) :key key}
                 step-fns)]
            {:values values
             :score score
             :weight weight
             ;; retval-fn proven truthy by the outer guard (every? some? + retval-fn)
             ;; RAW args, matching M2 simulate + the handler: an arg-derived
             ;; retval keeps its caller-facing type across ALL ops (genmlx-8mih)
             :retval (retval-fn values args-vec)}))))))

(defn make-branch-rewritten-generate
  "Build a compiled generate for models with rewritable branches (L1-M4).
   Returns (fn [key args-vec constraints] -> {:values :score :weight :retval}) or nil."
  [schema source]
  (when-let [{:keys [site-specs retval-fn seed-conds addrs]}
             (compiled/prepare-branch-sites schema source)]
    (let [step-fns (mapv build-generate-site-step site-specs)]
      (when (every? some? step-fns)
        (fn compiled-branch-generate [key args-vec constraints]
          (let [mlx-args (compiled/ensure-mlx-args args-vec)
                {:keys [values score weight]}
                (reduce
                 (fn [state step-fn]
                   (step-fn state mlx-args constraints))
                 ;; Branch conditions resolve against the RAW args (genmlx-b210)
                 {:values (seed-conds args-vec)
                  :score (mx/scalar 0.0) :weight (mx/scalar 0.0) :key key}
                 step-fns)]
            ;; Strip the reserved branch-cond bookkeeping keys so the trace
            ;; choicemap holds only real addresses (genmlx-gc4w); the simulate
            ;; path already does this via unpack-result.
            {:values (select-keys values addrs)
             :score score
             :weight weight
             ;; retval-fn proven truthy by prepare-branch-sites
             ;; RAW args, matching M2 simulate + the handler: an arg-derived
             ;; retval keeps its caller-facing type across ALL ops (genmlx-8mih)
             :retval (retval-fn values args-vec)}))))))

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
                 (select-keys result [:values :score :weight])))
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
                {:keys [values score discard]}
                (reduce
                 (fn [state step-fn]
                   (step-fn state mlx-args constraints old-choices))
                 {:values {} :score (mx/scalar 0.0) :discard {} :key key}
                 step-fns)]
            {:values values
             :score score
             :discard discard
             ;; retval-fn proven truthy by the outer guard (every? some? + retval-fn)
             ;; RAW args, matching M2 simulate + the handler: an arg-derived
             ;; retval keeps its caller-facing type across ALL ops (genmlx-8mih)
             :retval (retval-fn values args-vec)}))))))

(defn get-compiled-update
  "Returns the compiled-update function for a gen-fn, or nil."
  [gf]
  (:compiled-update (:schema gf)))

(defn make-branch-rewritten-update
  "Build a compiled update for models with rewritable branches (L1-M4).
   Returns (fn [key args-vec constraints old-choices]
             -> {:values :score :discard :retval}) or nil."
  [schema source]
  (when-let [{:keys [site-specs retval-fn seed-conds addrs]}
             (compiled/prepare-branch-sites schema source)]
    (let [step-fns (mapv build-update-site-step site-specs)]
      (when (every? some? step-fns)
        (fn compiled-branch-update [key args-vec constraints old-choices]
          (let [mlx-args (compiled/ensure-mlx-args args-vec)
                {:keys [values score discard]}
                (reduce
                 (fn [state step-fn]
                   (step-fn state mlx-args constraints old-choices))
                 {:values (seed-conds args-vec)
                  :score (mx/scalar 0.0) :discard {} :key key}
                 step-fns)]
            ;; Strip reserved branch-cond keys from the trace choicemap (genmlx-gc4w)
            {:values (select-keys values addrs)
             :score score
             :discard discard
             ;; RAW args, matching M2 simulate + the handler: an arg-derived
             ;; retval keeps its caller-facing type across ALL ops (genmlx-8mih)
             :retval (retval-fn values args-vec)}))))))

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
                 (select-keys result [:values :score :discard])))
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
                {:keys [values score]}
                (reduce
                 (fn [state step-fn]
                   (step-fn state mlx-args choices))
                 {:values {} :score (mx/scalar 0.0)}
                 step-fns)]
            {:score score
             ;; RAW args, matching M2 simulate + the handler: an arg-derived
             ;; retval keeps its caller-facing type across ALL ops (genmlx-8mih)
             :retval (retval-fn values args-vec)}))))))

(defn make-branch-rewritten-assess
  "Build a compiled assess for branch-rewritten models (L1-M4).
   Returns (fn [args-vec choices] -> {:score :retval}) or nil."
  [schema source]
  (when-let [{:keys [site-specs retval-fn seed-conds]}
             (compiled/prepare-branch-sites schema source)]
    (let [step-fns (mapv build-assess-site-step site-specs)]
      (when (every? some? step-fns)
        (fn compiled-branch-assess [args-vec choices]
          (let [mlx-args (compiled/ensure-mlx-args args-vec)
                {:keys [values score]}
                (reduce
                 (fn [state step-fn]
                   (step-fn state mlx-args choices))
                 {:values (seed-conds args-vec) :score (mx/scalar 0.0)}
                 step-fns)]
            {:score score
             ;; RAW args, matching M2 simulate + the handler: an arg-derived
             ;; retval keeps its caller-facing type across ALL ops (genmlx-8mih)
             :retval (retval-fn values args-vec)}))))))

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
                 (select-keys result [:values :score])))
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
  (when-let [{:keys [site-specs]} (compiled/prepare-static-sites schema source)]
    (let [step-fns (mapv build-project-site-step site-specs)]
      (when (every? some? step-fns)
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
  (when-let [{:keys [site-specs seed-conds]}
             (compiled/prepare-branch-sites schema source)]
    (let [step-fns (mapv build-project-site-step site-specs)]
      (when (every? some? step-fns)
        (fn compiled-branch-project [args-vec old-choices selection]
          (let [mlx-args (compiled/ensure-mlx-args args-vec)
                result
                (reduce
                 (fn [state step-fn]
                   (step-fn state mlx-args old-choices selection))
                 {:values (seed-conds args-vec)
                  :score (mx/scalar 0.0) :weight (mx/scalar 0.0)}
                 step-fns)]
            (:weight result)))))))

(defn make-compiled-prefix-project
  "Build a compiled prefix project function.
   Returns {:fn (fn [args-vec old-choices selection] -> {:values :weight})
            :addrs [keyword...]} or nil."
  [schema source]
  (when-let [{:keys [compiled-sites addrs]} (compiled/prepare-prefix-sites schema source)]
    (let [step-fns (mapv build-project-site-step compiled-sites)]
      (when (every? some? step-fns)
        {:fn (fn compiled-prefix-project [args-vec old-choices selection]
               (let [mlx-args (compiled/ensure-numeric-mlx-args args-vec)
                     result
                     (reduce
                      (fn [state step-fn]
                        (step-fn state mlx-args old-choices selection))
                      {:values {} :score (mx/scalar 0.0) :weight (mx/scalar 0.0)}
                      step-fns)]
                 (select-keys result [:values :weight])))
         :addrs addrs}))))

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
      (let [log-prob-fn (:log-prob nt)
            ;; Shared resample logic for both noise sources (genmlx-b210: the
            ;; old code treated :args-noise-fn sites as Delta — a selected
            ;; iid-gaussian was never resampled and contributed weight 0).
            make-noise-step
            (fn [draw-noise]
              (let [transform-fn (:transform nt)]
                (fn [{:keys [values score weight key]} args-vec old-choices selection]
                  (let [eval-args (mapv #(% values args-vec) compiled-args)]
                    (if (sel/selected? selection addr)
                      ;; Selected: resample via noise transform
                      (let [[k1 k2] (rng/split key)
                            noise (draw-noise eval-args k2)
                            new-val (apply transform-fn noise eval-args)
                            new-lp (apply log-prob-fn new-val eval-args)
                            old-val (cm/get-value (cm/get-submap old-choices addr))
                            old-lp (apply log-prob-fn old-val eval-args)]
                        {:values (assoc values addr new-val)
                         :score (mx/add score new-lp)
                         :weight (mx/add weight (mx/subtract new-lp old-lp))
                         :key k1})
                      ;; Not selected: keep old value, no key split
                      (let [v (cm/get-value (cm/get-submap old-choices addr))
                            lp (apply log-prob-fn v eval-args)]
                        {:values (assoc values addr v)
                         :score (mx/add score lp)
                         :weight weight
                         :key key}))))))]
        (cond
          (:noise-fn nt)
          (let [noise-fn (:noise-fn nt)]
            (make-noise-step (fn [_eval-args k] (noise-fn k))))

          (:args-noise-fn nt)
          (let [args-noise-fn (:args-noise-fn nt)]
            (make-noise-step (fn [eval-args k] (args-noise-fn eval-args k))))

          ;; Delta ONLY when the dist-type really is delta (genmlx-b210)
          (= dist-type :delta)
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
                (let [v (cm/get-value (cm/get-submap old-choices addr))]
                  {:values (assoc values addr v)
                   :score score
                   :weight weight
                   :key key}))))

          :else nil)))))

(defn- prepare-regen-parts
  "Shared prelude for the full and cone-restricted compiled regenerates:
   per-site regenerate step-fns + compiled retval-fn, or nil when the model
   is not M2-compilable."
  [schema source]
  (when-let [{:keys [site-specs retval-fn]}
             (compiled/prepare-static-sites schema source)]
    (let [step-fns (mapv build-regenerate-site-step site-specs)]
      (when (and (every? some? step-fns) retval-fn)
        {:site-specs site-specs :step-fns step-fns :retval-fn retval-fn}))))

(defn make-compiled-regenerate
  "Build a compiled regenerate function from a gen schema and source.
   Returns (fn [key args-vec old-choices selection]
             -> {:values :score :weight :retval})
   or nil if the model can't be compiled.
   :weight = proposal ratio (NOT final weight — DynamicGF computes that)."
  [schema source]
  (when-let [{:keys [step-fns retval-fn]} (prepare-regen-parts schema source)]
    (fn compiled-regenerate [key args-vec old-choices selection]
      (let [mlx-args (compiled/ensure-mlx-args args-vec)
            {:keys [values score weight]}
            (reduce
             (fn [state step-fn]
               (step-fn state mlx-args old-choices selection))
             {:values {} :score (mx/scalar 0.0) :weight (mx/scalar 0.0) :key key}
             step-fns)]
        {:values values
         :score score
         :weight weight
         ;; retval-fn proven truthy by prepare-regen-parts
         ;; RAW args, matching M2 simulate + the handler: an arg-derived
         ;; retval keeps its caller-facing type across ALL ops (genmlx-8mih)
         :retval (retval-fn values args-vec)}))))

(defn make-cone-regenerate
  "Cone-restricted compiled regenerate (genmlx-ltx2). For a SINGLE-site
   selection on a flat static model, only the selected site s and its DIRECT
   children (sites whose dist params read s's value through deterministic
   code — schema :direct-children) change log-prob; every other retained
   site's contribution cancels exactly. Reuses the same per-site step-fns as
   the full compiled regenerate over just that cone, twice:
     OLD pass (selection = none): partial-old = Σ lp_old(cone)
     NEW pass (real selection):   partial-new = Σ lp_new(cone),
                                  :weight = proposal ratio at s
   Returns the same result shape as make-compiled-regenerate with the
   incremental score old-score − partial-old + partial-new, so the caller's
   weight algebra W = (score' − old-score) − ratio = Σ_children (lp_new −
   lp_old) is unchanged. Key discipline matches the handler exactly —
   retained sites never split, so for a single-site selection the resampled
   value is bit-identical to the full paths under the same key. Graph work is
   O(|cone|); the values-map seed is O(T) cheap host assocs.
   Returns (fn [key args-vec old-choices selection old-score]
             -> {:values :score :weight :retval}, nil to decline per-call)
   or nil when the model can't take the cone path at all."
  [schema source]
  (when-let [{:keys [site-specs step-fns retval-fn]}
             (prepare-regen-parts schema source)]
    (let [addrs (mapv :addr site-specs)
          addr->step (zipmap addrs step-fns)
          direct-children (:direct-children schema)
          dep-order (:dep-order schema)]
      (when (and (seq addrs) (map? direct-children) (seq dep-order))
        (fn cone-regenerate [key args-vec old-choices selection old-score]
          (let [selected (filterv #(sel/selected? selection %) addrs)]
            ;; MVP gate: exactly one selected schema address; anything else
            ;; declines to the full compiled path (multi-site cones deferred).
            (when (= 1 (count selected))
              (let [s (first selected)
                    cone (conj (get direct-children s #{}) s)
                    cone-order (filterv cone (vec dep-order))
                    cone-steps (mapv addr->step cone-order)
                    mlx-args (compiled/ensure-mlx-args args-vec)
                    ;; Seed EVERY site's value from the old trace: cone steps
                    ;; read parent values from this map, and retval-fn needs
                    ;; the full map. Host-map assocs only — no graph nodes.
                    values-old (reduce (fn [m a]
                                         (assoc m a (cm/get-value
                                                     (cm/get-submap old-choices a))))
                                       {} addrs)
                    run (fn [sel']
                          (reduce (fn [state step-fn]
                                    (step-fn state mlx-args old-choices sel'))
                                  {:values values-old
                                   :score (mx/scalar 0.0)
                                   :weight (mx/scalar 0.0)
                                   :key key}
                                  cone-steps))
                    old-res (run sel/none)
                    new-res (run selection)
                    score' (mx/add (mx/subtract old-score (:score old-res))
                                   (:score new-res))]
                {:values (:values new-res)
                 :score score'
                 :weight (:weight new-res)
                 :retval (retval-fn (:values new-res) args-vec)}))))))))

(def ^:private vcone-dist-constructors
  "dist-type -> runtime Distribution constructor for the BATCHED cone path
   (genmlx-js93). [N]-lane resampling must go through dc/dist-sample-n on a
   real Distribution — bit-identical to the batched handler transition — not
   the scalar noise transforms. A dist-type absent here (e.g. :iid-gaussian,
   no defdist constructor) declines the whole model to the batched handler."
  {:gaussian dist/gaussian :normal dist/gaussian
   :uniform dist/uniform
   :bernoulli dist/bernoulli :flip dist/bernoulli
   :exponential dist/exponential
   :log-normal dist/log-normal
   :laplace dist/laplace
   :cauchy dist/cauchy
   :delta dist/delta})

(defn make-vcone-regenerate
  "Batched cone-restricted regenerate (genmlx-js93) — the [N]-lane
   generalization of make-cone-regenerate. For a SINGLE-site selection on a
   flat static model the cone is address-determined (per-model, not
   per-particle), so the static? gate IS the lane-uniformity gate: all N
   chains recompute the same {s} ∪ direct-children(s) sites as one broadcast.

   Sampling and log-probs go through the real Distribution (constructed from
   the compiled arg closures' [N]-shaped values) via dc/dist-sample-n /
   dc/dist-log-prob with the handler's exact key discipline (retained sites
   never split; one split at s), so results are BIT-IDENTICAL to the batched
   handler path under the same key.

   Returns (fn [key args-vec old-choices selection old-score n]
             -> {:choices :score :weight :retval}, nil to decline per-call)
   or nil when the model can't take the path at all. :weight is the proposal
   ratio; the caller computes W = (score' − old-score) − ratio, all [N]."
  [schema source]
  (when-let [{:keys [site-specs retval-fn]} (prepare-regen-parts schema source)]
    (let [addrs (mapv :addr site-specs)
          specs-by-addr (into {} (map (juxt :addr identity)) site-specs)
          direct-children (:direct-children schema)
          dep-order (:dep-order schema)]
      (when (and (seq addrs) (map? direct-children) (seq dep-order)
                 (every? #(contains? vcone-dist-constructors (:dist-type %))
                         site-specs))
        (fn vcone-regenerate [key args-vec old-choices selection old-score n]
          (let [selected (filterv #(sel/selected? selection %) addrs)]
            (when (= 1 (count selected))
              (let [s (first selected)
                    cone (conj (get direct-children s #{}) s)
                    cone-order (filterv cone (vec dep-order))
                    mlx-args (compiled/ensure-mlx-args args-vec)
                    values-old (reduce (fn [m a]
                                         (assoc m a (cm/get-value
                                                     (cm/get-submap old-choices a))))
                                       {} addrs)
                    mk-dist (fn [values addr]
                              (let [{:keys [compiled-args dist-type]} (specs-by-addr addr)
                                    eval-args (mapv #(% values mlx-args) compiled-args)]
                                (apply (vcone-dist-constructors dist-type) eval-args)))
                    ;; OLD pass: Σ lp_old over the cone under old values
                    partial-old (reduce (fn [acc a]
                                          (mx/add acc (dc/dist-log-prob
                                                       (mk-dist values-old a)
                                                       (get values-old a))))
                                        (mx/scalar 0.0) cone-order)
                    ;; resample s exactly like the batched handler transition:
                    ;; [k1 k2] split, dist-sample-n with k2 (k1 unused — no
                    ;; later selected site exists in a single-site selection)
                    [_k1 k2] (rng/split key)
                    dist-s (mk-dist values-old s) ; s's params read only parents
                    v' (dc/dist-sample-n dist-s k2 n)
                    new-lp-s (dc/dist-log-prob dist-s v')
                    old-lp-s (dc/dist-log-prob dist-s (get values-old s))
                    values-new (assoc values-old s v')
                    ;; NEW pass: Σ lp_new over the cone under s ↦ v'
                    partial-new (reduce (fn [acc a]
                                          (mx/add acc
                                                  (if (= a s)
                                                    new-lp-s
                                                    (dc/dist-log-prob
                                                     (mk-dist values-new a)
                                                     (get values-new a)))))
                                        (mx/scalar 0.0) cone-order)
                    score' (mx/add (mx/subtract old-score partial-old)
                                   partial-new)]
                {:choices (cm/set-value old-choices s v')
                 :score score'
                 :weight (mx/subtract new-lp-s old-lp-s)
                 :retval (retval-fn values-new args-vec)}))))))))

(defn make-fused-vmh
  "Fused vectorized single-site MH sweep (genmlx-hwhp) — the O(1)-per-move
   host-work driver behind mcmc/vmh. The per-move vmh-step path pays light
   O(T) host work every move (values-map seed, merge walk over all leaves,
   eligibility scan, per-move VectorizedTrace construction), which caps the
   speedup over the batched handler at the heavy/light interpretation ratio
   (~20x, genmlx-da04). This runner threads the values map and [N] score
   ACROSS moves: per move it touches only {s} ∪ direct-children(s) — one
   leaf assoc, O(|cone|) graph nodes, one GPU sync — and builds the
   choicemap/retval ONCE at sweep end.

   PRNG discipline replicates the per-move driver EXACTLY (per move:
   [k1 k2]=split(k); [regen-key accept-key]=split(k1); [_ ksample]=
   split(regen-key); uniform(accept-key,[n]); k=k2), and the merge nodes are
   built with the same where(mask, proposed, current) shapes — so the final
   choices and score are BIT-IDENTICAL to a vmh-step-per-move sweep under
   the same key.

   Returns (fn [args-vec old-choices score0 n key0 sweep-addrs]
             -> {:choices :score :retval :key}, nil to decline when any swept
   address is outside the plan) or nil when the model can't take the path."
  [schema source]
  (when-let [{:keys [site-specs retval-fn]} (prepare-regen-parts schema source)]
    (let [addrs (mapv :addr site-specs)
          specs-by-addr (into {} (map (juxt :addr identity)) site-specs)
          direct-children (:direct-children schema)
          dep-order (:dep-order schema)]
      (when (and (seq addrs) (map? direct-children) (seq dep-order)
                 (every? #(contains? vcone-dist-constructors (:dist-type %))
                         site-specs))
        (let [;; per-address cone order, precomputed once (was an O(T) filterv
              ;; per move)
              plan (into {}
                         (map (fn [s]
                                [s (filterv (conj (get direct-children s #{}) s)
                                            (vec dep-order))]))
                         addrs)
              ;; mask->leaf-rank replica (vectorized.cljs) so merge nodes are
              ;; shaped identically to merge-choicemap-by-mask's
              lift-mask (fn [mask v]
                          (let [r (count (mx/shape v))]
                            (if (> r 1)
                              (mx/reshape mask (into [(first (mx/shape mask))]
                                                     (repeat (dec r) 1)))
                              mask)))]
          (fn fused-vmh [args-vec old-choices score0 n key0 sweep-addrs]
            (when (every? #(contains? plan %) sweep-addrs)
              (let [mlx-args (compiled/ensure-mlx-args args-vec)
                    mk-dist (fn [values addr]
                              (let [{:keys [compiled-args dist-type]} (specs-by-addr addr)
                                    eval-args (mapv #(% values mlx-args) compiled-args)]
                                (apply (vcone-dist-constructors dist-type) eval-args)))
                    values0 (reduce (fn [m a]
                                      (assoc m a (cm/get-value
                                                  (cm/get-submap old-choices a))))
                                    {} addrs)
                    [values score kfin]
                    (loop [as (seq sweep-addrs)
                           values values0
                           score score0
                           k (rng/ensure-key key0)]
                      (if (nil? as)
                        [values score k]
                        (let [s (first as)
                              [k1 k2] (rng/split k)
                              [regen-key accept-key] (rng/split k1)
                              [_k ksample] (rng/split regen-key)
                              cone-order (plan s)
                              partial-old (reduce (fn [acc a]
                                                    (mx/add acc (dc/dist-log-prob
                                                                 (mk-dist values a)
                                                                 (get values a))))
                                                  (mx/scalar 0.0) cone-order)
                              dist-s (mk-dist values s)
                              v' (dc/dist-sample-n dist-s ksample n)
                              new-lp-s (dc/dist-log-prob dist-s v')
                              old-lp-s (dc/dist-log-prob dist-s (get values s))
                              values-prop (assoc values s v')
                              partial-new (reduce (fn [acc a]
                                                    (mx/add acc
                                                            (if (= a s)
                                                              new-lp-s
                                                              (dc/dist-log-prob
                                                               (mk-dist values-prop a)
                                                               (get values-prop a)))))
                                                  (mx/scalar 0.0) cone-order)
                              score-prop (mx/add (mx/subtract score partial-old)
                                                 partial-new)
                              ratio (mx/subtract new-lp-s old-lp-s)
                              w (mx/subtract (mx/subtract score-prop score) ratio)
                              u (rng/uniform accept-key [n])
                              mask (mx/less (mx/log u) w)
                              cur (get values s)
                              merged-v (mx/where (lift-mask mask cur) v' cur)
                              merged-score (mx/where mask score-prop score)]
                          ;; ONE GPU sync per move; merged leaves stay lazy
                          ;; depth-1 and evaluate inside a later move's sync
                          (mx/materialize! merged-score)
                          (recur (next as)
                                 (assoc values s merged-v)
                                 merged-score
                                 k2))))
                    touched (set sweep-addrs)
                    final-choices (reduce (fn [acc a] (cm/set-value acc a (get values a)))
                                          old-choices touched)]
                ;; sweep-end sync: settle every still-lazy merged leaf
                (apply mx/materialize! score (map values touched))
                {:choices final-choices
                 :score score
                 :retval (retval-fn values args-vec)
                 :key kfin}))))))))

(defn make-branch-rewritten-regenerate
  "Build a compiled regenerate for models with rewritable branches (L1-M4).
   Returns (fn [key args-vec old-choices selection]
             -> {:values :score :weight :retval}) or nil."
  [schema source]
  (when-let [{:keys [site-specs retval-fn seed-conds addrs]}
             (compiled/prepare-branch-sites schema source)]
    (let [step-fns (mapv build-regenerate-site-step site-specs)]
      (when (every? some? step-fns)
        (fn compiled-branch-regenerate [key args-vec old-choices selection]
          (let [mlx-args (compiled/ensure-mlx-args args-vec)
                {:keys [values score weight]}
                (reduce
                 (fn [state step-fn]
                   (step-fn state mlx-args old-choices selection))
                 {:values (seed-conds args-vec)
                  :score (mx/scalar 0.0) :weight (mx/scalar 0.0) :key key}
                 step-fns)]
            ;; Strip reserved branch-cond keys from the trace choicemap (genmlx-gc4w)
            {:values (select-keys values addrs)
             :score score
             :weight weight
             ;; RAW args, matching M2 simulate + the handler: an arg-derived
             ;; retval keeps its caller-facing type across ALL ops (genmlx-8mih)
             :retval (retval-fn values args-vec)}))))))

(defn make-compiled-prefix-regenerate
  "Build a compiled prefix regenerate function.
   Returns {:fn (fn [key args-vec old-choices selection]
                  -> {:values :score :weight})
            :addrs [keyword...]}
   or nil if partial compilation isn't applicable."
  [schema source]
  (when-let [{:keys [compiled-sites addrs]} (compiled/prepare-prefix-sites schema source)]
    (let [step-fns (mapv build-regenerate-site-step compiled-sites)]
      (when (every? some? step-fns)
        {:fn (fn compiled-prefix-regenerate [key args-vec old-choices selection]
               (let [mlx-args (compiled/ensure-numeric-mlx-args args-vec)
                     result
                     (reduce
                      (fn [state step-fn]
                        (step-fn state mlx-args old-choices selection))
                      {:values {} :score (mx/scalar 0.0) :weight (mx/scalar 0.0) :key key}
                      step-fns)]
                 (select-keys result [:values :score :weight])))
         :addrs addrs}))))

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
            selected? (sel/selected? (:selection state) addr)
            state' (update state :choices cm/set-value addr value)]
        ;; Selected sites split the key (matching handler's regenerate-transition);
        ;; unselected sites replay the value with the key untouched.
        [value (if selected?
                 (let [[k1 _] (rng/split (:key state'))]
                   (assoc state' :key k1))
                 state')])
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
   or nil if the dist-type cannot be fused.
   For delta sites, noise-index is ignored (no noise consumed).

   Only :noise-fn distributions and true :delta sites are fusable. A transform
   entry with :args-noise-fn (iid-gaussian — noise shape depends on dist args)
   must return nil: the previous fallback treated it as Delta, so fused
   Map/Unfold/Scan kernels gave value=mu with ZERO score contribution,
   silently (genmlx-b210)."
  [site-spec noise-index]
  (let [{:keys [addr compiled-args dist-type]} site-spec
        nt (get compiled/noise-transforms-full dist-type)]
    (when nt
      (cond
        (:noise-fn nt)
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

        (= dist-type :delta)
        ;; Delta: no noise needed
        (fn [{:keys [values score]} args-vec _noise-row]
          (let [eval-args (mapv #(% values args-vec) compiled-args)
                value (first eval-args)]
            {:values (assoc values addr value)
             :score score}))

        :else nil))))

(defn- noise-fn-site?
  "True when a site-spec exists and its dist-type has a :noise-fn transform."
  [s]
  (boolean (and s (:noise-fn (get compiled/noise-transforms-full (:dist-type s))))))

(defn- assign-noise-indices
  "Assign sequential noise indices to site-specs that have noise-fns.
   Returns vector of indices (nil for delta/unsupported sites)."
  [site-specs]
  (second
   (reduce (fn [[idx acc] s]
             (if (noise-fn-site? s)
               [(inc idx) (conj acc idx)]
               [idx (conj acc nil)]))
           [0 []] site-specs)))

(defn- extract-noise-site-types
  "Filter site-specs to those with noise-fns."
  [site-specs]
  (filterv noise-fn-site? site-specs))

(defn generate-noise-matrix
  "Generate [T, K] noise matrix where each column has the correct distribution.
   noise-site-types: vector of {:dist-type ...} for noise-consuming sites.
   Returns [T, K] MLX array."
  [key T noise-site-types]
  (if (empty? noise-site-types)
    (mx/zeros [T 1])
    (let [[_k cols]
          (reduce (fn [[k cols] site]
                    (let [[k1 k2] (rng/split k)
                          col (case (get noise-type-map (:dist-type site))
                                :normal (rng/normal k1 [T])
                                :uniform (rng/uniform k1 [T]))]
                      [k2 (conj cols col)]))
                  [key []]
                  noise-site-types)]
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
  (when-let [{:keys [site-specs retval-fn addrs]}
             (compiled/prepare-static-sites schema source)]
    (let [noise-indices (assign-noise-indices site-specs)
          noise-site-types (extract-noise-site-types site-specs)
          noise-dim (count noise-site-types)
          fused-steps (mapv (fn [spec ni] (build-fused-site-step spec ni))
                            site-specs noise-indices)
          ;; Map-valued state detection needs the RAW return expression
          return-expr (compiled/extract-return-expr (:return-form schema))
          map-state? (map? return-expr)
          state-keys (when map-state? (vec (sort (keys return-expr))))
          addr-order addrs]
      (when (and (every? some? fused-steps)
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
  (when-let [{:keys [site-specs binding-env addrs]}
             (compiled/prepare-static-sites schema source)]
    (let [noise-indices (assign-noise-indices site-specs)
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
          addr-order addrs]
      (when (and (every? some? fused-steps)
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
          {:compiled-fn compiled
           :noise-dim noise-dim
           :addr-order addr-order
           :noise-site-types noise-site-types})))))

(defn fusable-kernel?
  "Check if a kernel can be fused into a single lazy-graph evaluation.
   Returns true if the kernel has a static schema where every trace site is
   either a :noise-fn distribution or a true delta, with at least one
   noise-driven site. Sites with :args-noise-fn (iid-gaussian) are NOT
   fusable — they must decline here, mirroring build-fused-site-step
   (genmlx-b210)."
  [gf]
  (let [schema (:schema gf)]
    (and schema
         (:static? schema)
         (seq (:trace-sites schema))
         (empty? (:splice-sites schema))
         (empty? (:param-sites schema))
         (let [sites (filterv :static? (:trace-sites schema))
               nts (mapv #(get compiled/noise-transforms-full (:dist-type %)) sites)]
           (and (every? some? nts)
                (every? (fn [[site nt]]
                          (or (:noise-fn nt) (= :delta (:dist-type site))))
                        (map vector sites nts))
                (some :noise-fn nts))))))

(defn make-fused-map-simulate
  "Build a fused map simulate that processes all N elements in one call.
   No mx/compile-fn needed — MLX broadcasting handles [N]-shaped arrays.
   Stacks element args into [N]-shaped arrays, pre-generates [N] noise per site,
   runs site steps once with broadcasting.

   Returns {:fused-fn (fn [key stacked-args N] -> {:values {addr -> [N]-arr}
                                                   :scores [N]-arr :retval [N]-arr})
            :noise-site-types :addr-order}
   or nil if kernel can't be fused.

   stacked-args: vector of [N]-shaped arrays (one per kernel param)."
  [schema source]
  (when-let [{:keys [site-specs retval-fn addrs]}
             (compiled/prepare-static-sites schema source)]
    (let [noise-indices (assign-noise-indices site-specs)
          noise-site-types (extract-noise-site-types site-specs)
          noise-dim (count noise-site-types)
          fused-steps (mapv (fn [spec ni] (build-fused-site-step spec ni))
                            site-specs noise-indices)
          addr-order addrs]
      (when (and (every? some? fused-steps)
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

;; ---------------------------------------------------------------------------
;; Vectorized family scoring (genmlx-yopl)
;; ---------------------------------------------------------------------------
;; A homogeneous OBSERVED site family — same dist-type, dist-arg source forms
;; identical up to numeric literals — is scored as ONE stacked [G] log-prob
;; (per-literal-position [G] constant columns, one arg graph, elementwise lp,
;; mx/sum) instead of G scalar per-site subgraphs. The value of this is not
;; the forward pass (compile_fuse already fuses that into one kernel): it is
;; the VJP, which mirrors the forward structure — per-site scalar scoring
;; put ~2 backward kernels + gather/scatter/squeeze plumbing PER SITE in
;; every fused MCMC step (the genmlx-kzoy census); the stacked form
;; vectorizes the backward into a handful of [G]-shaped kernels.

(def ^:private family-elementwise-dists
  "Dist types whose :log-prob closure is elementwise MLX math that
   broadcasts over stacked [G] value/arg tensors — eligible for vectorized
   family scoring. Excludes :delta (no lp term) and vector-valued iid types
   (their lp reduces internally)."
  #{:gaussian :normal :uniform :bernoulli :flip :exponential :log-normal
    :laplace :cauchy})

(def ^:private family-value-op-names
  "Source-form op names EVERY argument of which is a broadcastable VALUE, so
   a per-site numeric literal appearing there can safely be replaced by a
   materialized [G] family column.

   The list is a WHITELIST on purpose (genmlx-spid). abstract-dist-args used
   to replace every numeric literal regardless of position, which turns the
   index of `(mx/index xs 0)` / `(mx/index xs 1)` … into a float32 [G]
   column — and mx/index is the one gather-family op without
   ensure-int-indices, so the family's arg graph throws
   `[gather] Got indices with invalid dtype` the first time it is evaluated,
   killing compiled-mh / fused-MALA / fused-HMC for the model instead of
   declining to the handler. The same hazard exists for every structural
   (shape / axis / count / dtype) argument position: mx/reshape,
   mx/broadcast-to, the axis of a reduction, mx/slice bounds, mx/split-arr
   sections, mx/mat-get — all of them require host numbers or ints and would
   be handed a float32 array.

   Enumerating every unsafe (op, position) pair is open-ended and one miss
   is a crash, so the safe direction is inverted: a literal is abstracted
   only when the whole path from the dist-arg root down to it passes through
   ops proven elementwise-in-every-argument. Anything else keeps the literal
   in the signature, so sites that differ there simply do not group into a
   family and fall through to per-site compiled scoring — the handler-equal
   path. The cost is a missed vectorization; the alternative was a throw."
  #{"add" "+" "subtract" "-" "multiply" "*" "divide" "/" "negate"
    "scalar" "abs" "exp" "expm1" "log" "log1p" "log2" "log10" "sqrt" "rsqrt"
    "square" "power" "pow" "sin" "cos" "tan" "sinh" "cosh" "tanh"
    "sigmoid" "erf" "sign" "floor" "ceil" "round" "reciprocal"
    "maximum" "minimum" "clip" "logaddexp" "relu" "softplus"})

(defn- family-value-op?
  "Is `h` a head symbol whose every argument position is a plain value?
   The namespace guard mirrors compiled/resolve-fn: only `mx`/`genmlx.mlx`
   and unqualified core ops compile at all, so a same-named symbol from some
   other namespace can never reach the arg graph."
  [h]
  (and (symbol? h)
       (contains? #{nil "mx" "genmlx.mlx"} (namespace h))
       (contains? family-value-op-names (name h))))

(defn- abstract-dist-args
  "Abstract a site's dist-arg source forms for family matching: a numeric
   literal in a VALUE position becomes a positional placeholder symbol
   ᐩfam<i>; the literal values are collected in walk order. A literal
   anywhere else keeps its value (see family-value-op-names) — sites then
   differ in signature and simply do not merge. Declines (nil) forms that
   embed a (trace ...) call (an inline site definition must never be
   family-merged) or map literals (walk-order stability).
   Returns {:sig [forms'] :lits [numbers]} or nil."
  [dist-args]
  (let [counter (volatile! 0)
        lits (volatile! [])
        ok? (volatile! true)
        walk (fn walk [form value-pos?]
               (cond
                 (number? form)
                 (if value-pos?
                   (let [i @counter]
                     (vswap! counter inc)
                     (vswap! lits conj form)
                     (symbol (str "ᐩfam" i)))
                   form)

                 (and (seq? form) (seq form))
                 (if (and (symbol? (first form)) (= "trace" (name (first form))))
                   (do (vreset! ok? false) form)
                   (let [child-pos? (and value-pos? (family-value-op? (first form)))]
                     (cons (first form)
                           (doall (map #(walk % child-pos?) (rest form))))))

                 ;; A vector is a shape / index list far more often than a
                 ;; value here, and compile-expr turns it into a runtime
                 ;; mapv — abstracting inside one would put an array where a
                 ;; host number is required.
                 (vector? form) (mapv #(walk % false) form)
                 (map? form) (do (vreset! ok? false) form)
                 :else form))
        sig (mapv #(walk % true) dist-args)]
    (when @ok?
      {:sig sig :lits @lits})))

(defn- detect-observed-families
  "Group observed static sites into homogeneous families. sites: source-order
   static trace-sites (schema maps with :addr :dist-type :dist-args) whose
   values are present in the observations. Only groups of >= 2 vectorize.
   Returns a seq of {:dist-type kw :sig forms :members [addr...]
                     :lits-per-site [[num...]...]}."
  [sites]
  (->> sites
       (keep (fn [site]
               (when (contains? family-elementwise-dists (:dist-type site))
                 (when-let [{:keys [sig lits]} (abstract-dist-args (:dist-args site))]
                   {:dist-type (:dist-type site)
                    :sig sig
                    :addr (:addr site)
                    :lits lits}))))
       (group-by (juxt :dist-type :sig))
       vals
       (filter #(>= (count %) 2))
       (map (fn [group]
              {:dist-type (:dist-type (first group))
               :sig (:sig (first group))
               :members (mapv :addr group)
               :lits-per-site (mapv :lits group)}))))

(defn- compile-family-sig
  "Compile a family's abstracted sig against the binding env. Literal
   columns whose G values are all equal stay plain numbers (no [G] input
   for constants); varying columns become materialized [G] constants
   appended to the args vector, with the placeholder symbols bound as
   synthetic :param entries — compile-expr is reused unchanged.
   Returns {:cargs [fn-or-nil ...] :args+lits [...] :env env'}."
  [sig lits-per-site binding-env mlx-args]
  (let [n-args (count mlx-args)
        n-lits (count (first lits-per-site))
        lit-cols (mapv (fn [p]
                         (let [col (mapv #(nth % p) lits-per-site)]
                           (if (apply = col)
                             (first col)
                             (let [a (mx/array col)]
                               (mx/materialize! a)
                               a))))
                       (range n-lits))
        env' (reduce (fn [e p]
                       (assoc e (str "ᐩfam" p)
                              {:kind :param :index (+ n-args p)}))
                     binding-env
                     (range n-lits))]
    {:cargs (mapv #(compiled/compile-expr % env' #{}) sig)
     :args+lits (into mlx-args lit-cols)
     :env env'}))

;; ---------------------------------------------------------------------------
;; Structural latent-affinity proof (genmlx-yy8u)
;; ---------------------------------------------------------------------------

(def ^:private affine-additive-heads #{"add" "+" "subtract" "-"})
(def ^:private affine-product-heads  #{"multiply" "*"})
(def ^:private affine-quotient-heads #{"divide" "/"})
;; Single-value passthroughs: degree of the first argument, every other
;; argument required latent-free (mx/scalar's optional dtype).
(def ^:private affine-passthrough-heads #{"negate" "scalar"})

(defn- affine-head-name
  "Head name of a call form when it is one compile-expr can actually
   resolve (see compiled/resolve-fn: only `mx`/`genmlx.mlx`-qualified and
   unqualified ops compile), else nil."
  [h]
  (when (and (symbol? h) (contains? #{nil "mx" "genmlx.mlx"} (namespace h)))
    (name h)))

(defn latent-affinity
  "STRUCTURAL degree of a dist-arg source form in the LATENT trace
   addresses, walking the same binding-env compiled/compile-expr resolves
   against:

     :const     — provably latent-free
     :affine    — provably jointly degree <= 1 in the latents
     :nonlinear — anything not affirmatively proven affine

   This is the soundness gate genmlx-1fbs specified and the original
   implementation replaced with a numeric probe (genmlx-yy8u). A probe that
   samples the residual at q = 0, at each basis vector and at ONE interior
   point can only ever be evidence: the residual of any
   g(q) = alpha*q + beta*(q^2 - q) is pinned to exactly zero at every one of
   those points, so `mean = 100*slope + 0.005*slope^2` was ACCEPTED as
   affine and the compiled score diverged 945 nats from the handler at
   slope = 10 with a 3x-wrong gradient at the mode.

   The rule that makes it a proof rather than evidence: JOINT affinity needs
   every product/quotient to have at most one latent-bearing factor.
   `(mx/multiply slope slope)` and `(mx/multiply a b)` with both latent-
   bearing are quadratic; `x / latent` is not affine either. Affinity in
   each latent SEPARATELY would not do — `(mx/multiply a b)` is affine in a
   for fixed b.

   Conservative in the safe direction: an unrecognized op with latent-free
   arguments is :const (the compiled ops are pure functions of their
   arguments), and anything else is :nonlinear, which only ever causes the
   affine emission to decline to the exact stacked/per-site path."
  [form binding-env latent-addrs visited]
  (letfn [(deg [f v] (latent-affinity f binding-env latent-addrs v))
          (worst [ds] (cond (some #{:nonlinear} ds) :nonlinear
                            (some #{:affine} ds)    :affine
                            :else                   :const))
          (all-const? [ds] (every? #(= :const %) ds))]
    (cond
      (or (number? form) (boolean? form) (keyword? form) (string? form)
          (nil? form))
      :const

      (symbol? form)
      (let [info (get binding-env (name form))]
        (case (:kind info)
          :param    :const
          :trace    (if (contains? latent-addrs (:addr info)) :affine :const)
          :expr     (if (contains? visited (name form))
                      :nonlinear
                      (deg (:form info) (conj visited (name form))))
          :poisoned :nonlinear
          ;; Absent from the binding env: compile-expr resolves it as a
          ;; closed-over var, which cannot read the values-map.
          :const))

      ;; (:key m) map access
      (and (seq? form) (seq form) (keyword? (first form)))
      (if (= :const (deg (second form) visited)) :const :nonlinear)

      (and (seq? form) (seq form) (symbol? (first form)))
      (let [head (first form)
            args (vec (rest form))]
        (if (and (= "trace" (name head)) (keyword? (first args)))
          (if (contains? latent-addrs (first args)) :affine :const)
          (let [h (affine-head-name head)
                ds (mapv #(deg % visited) args)]
            (cond
              (nil? h) :nonlinear

              (contains? affine-additive-heads h) (worst ds)

              (contains? affine-product-heads h)
              (cond
                (some #{:nonlinear} ds) :nonlinear
                (> (count (filter #{:affine} ds)) 1) :nonlinear
                :else (worst ds))

              (contains? affine-quotient-heads h)
              (cond
                (empty? ds) :nonlinear
                (some #{:nonlinear} ds) :nonlinear
                ;; Dividing BY anything latent-bearing is not affine; the
                ;; 1-arg reciprocal form is only affine when constant.
                (some #{:affine} (rest ds)) :nonlinear
                (< (count ds) 2) (if (= :const (first ds)) :const :nonlinear)
                :else (first ds))

              (contains? affine-passthrough-heads h)
              (cond
                (empty? ds) :const
                (not (all-const? (rest ds))) :nonlinear
                :else (first ds))

              :else (if (all-const? ds) :const :nonlinear)))))

      (vector? form)
      (if (all-const? (mapv #(deg % visited) form)) :const :nonlinear)

      (map? form)
      (if (all-const? (mapv #(deg % visited)
                            (concat (keys form) (vals form))))
        :const :nonlinear)

      :else :nonlinear)))

;; ---------------------------------------------------------------------------
;; Family build-time safety (genmlx-spid, genmlx-1ol8)
;; ---------------------------------------------------------------------------

(defn- stacked-obs
  "Observation values for `members`, stacked into a [G] constant.

   The coercion is load-bearing: mx/stack's NAPI signature is array-ONLY
   while every other observation consumer takes Either<&MxArray, f64>, so
   `(cm/choicemap :a 1.0 :b 2.0)` — legal everywhere else, and blessed by
   the choicemap docstring — used to throw `Failed to recover MxArray type
   from napi value` out of the score build, naming nothing the user wrote
   (genmlx-1ol8). Single-site models never hit it, which is why the in-tree
   raw-number choicemaps (all on non-static doseq models that decline before
   family detection) missed it."
  [obs-values members]
  (let [s (mx/stack (mapv (fn [a]
                            (let [v (get obs-values a)]
                              (if (number? v) (mx/scalar v) v)))
                          members))]
    (mx/materialize! s)
    s))

(defn- probe-values-map
  "A values-map that binds every latent to a scalar 0 on top of the baked
   observations — enough to force a family's arg graph once at build time."
  [obs-values latent-order]
  (reduce (fn [m a] (assoc m a (mx/scalar 0.0)))
          (into {} (map (fn [[a v]] [a (if (number? v) (mx/scalar v) v)]))
                obs-values)
          latent-order))

(defn- lp-fn-usable?
  "Force a freshly built family/emission closure ONCE at a probe point.

   Compilation is an OPTIMIZATION: CLAUDE.md rule 5 promises the handler as
   ground truth and a fallback, never a crash. But a family's arg graph is
   built lazily — the literal columns are only assembled into MLX ops when
   the closure runs — so a dtype/shape mismatch introduced by the family
   merge surfaces on FIRST EVALUATION, inside the MCMC step, far from any
   build-time try/catch (genmlx-spid). Evaluating once here converts that
   whole class into a decline: the emission is dropped, its members fall
   back to per-site compiled scoring, and a notice is printed so the decline
   is not silent. One extra sync per emission, at build time only."
  [lp-fn probe-input what]
  (try
    (let [v (lp-fn probe-input)]
      (mx/materialize! v)
      true)
    (catch :default e
      (println (str "  [genmlx] compiled " what
                    " scoring declined to the per-site path: "
                    (.-message e)))
      false)))

(defn- build-family-lp-fns
  "Build one stacked log-prob closure per family (see compile-family-sig
   for the literal-column mechanics). reduce-fn folds the elementwise
   [.., G]-shaped lp to the score shape (full mx/sum for the scalar score;
   last-axis keepdims sum for the batched [N,1] score).
   Returns [lp-fns family-addrs]; a family whose sig fails to compile, whose
   build throws, or whose arg graph fails its build-time smoke evaluation
   falls back to per-site scoring (dropped from family-addrs)."
  [families binding-env mlx-args obs-values latent-order reduce-fn]
  (let [probe-vm (delay (probe-values-map obs-values latent-order))
        built
        (keep (fn [{:keys [dist-type sig members lits-per-site]}]
                (try
                  (let [log-prob-fn (:log-prob (get compiled/noise-transforms-full dist-type))
                        {:keys [cargs args+lits]}
                        (compile-family-sig sig lits-per-site binding-env mlx-args)
                        stacked (stacked-obs obs-values members)]
                    (when (every? some? cargs)
                      (let [lp-fn (fn [values-map]
                                    (let [eval-args (mapv #(% values-map args+lits) cargs)]
                                      (reduce-fn (apply log-prob-fn stacked eval-args))))]
                        (when (lp-fn-usable? lp-fn @probe-vm "family")
                          {:lp-fn lp-fn :members members}))))
                  (catch :default e
                    (println (str "  [genmlx] compiled family scoring declined to"
                                  " the per-site path: " (.-message e)))
                    nil)))
              families)]
    [(mapv :lp-fn built)
     (into #{} (mapcat :members) built)]))

(def ^:private affine-family-disabled?
  "The rung-2 EMISSION PAIR — matmul-form affine families AND stacked
   latent priors.

   ON by default; GENMLX_AFFINE_FAMILY=0 is the kill switch.

   HISTORY, because the default moved twice. It shipped opt-in in a358ab4
   and was flipped default-ON in 822c4cd. The 2026-08-01 regression audit
   then showed the affinity DETECTOR was unsound and it was forced back to
   opt-in for the day: `build-affine-family-lp-fns` judged global affinity
   from the value at q=0, at each latent basis vector, and at ONE interior
   point with |q| <= 0.75, accepting when err <= 1e-4 * max(1, |mean|). The
   residual is structurally pinned to zero at q=0 and at every basis point,
   so the whole judgement rested on that one point; and the tolerance was
   scaled by the magnitude of the MEAN, never by sigma. Reproduced on
   sm_120 at the then-default knob: mean = 100*slope + 0.005*slope^2 was
   ACCEPTED, and the compiled score diverged from the handler by 945 nats
   at slope=10 with a 3x-wrong gradient at the mode; in a sigma-blind
   variant (|mu|~1e4, sigma=0.01) by 90,007 nats at the mode. Every
   MALA/HMC/NUTS proposal is computed from that gradient, so the chain
   targeted a different distribution while every finiteness / determinism /
   acceptance-rate assertion still passed (genmlx-yy8u).

   The default is restored because the detector is now the sound structural
   sig-walk genmlx-1fbs specified — `latent-affinity` proves joint affinity
   from the source form (every product/quotient carries at most one
   latent-bearing factor) and the numeric probe survives only as a
   cross-check, now at three points including |q| ~ 3 and ~10 and with a
   sigma-scaled tolerance. Both repros above DECLINE
   (:affine-families 0) and reproduce the handler to float32 rounding;
   family_score_test pins the declines, and pins score+gradient equivalence
   at |q| = 10 on a model where the emission does engage.

   Original measurement, sm_120 at S=100, 2026-07-29 (taken before the
   detector was sound — models that engage now are a subset of the models
   that engaged then, so these are upper bounds):

     affine emission alone   HMC 16875 -> 19677 tape  (NET NEGATIVE)
     both + lazy values-map  HMC 104.84 -> 62.72 real launches/step

   Alone, the affine matmul removes nothing because the per-latent
   index/one-hot extraction survives for the PRIOR sites; the prior sites
   are what keep the values-map alive, and the values-map is what forces
   the extraction. Score all K priors as one stacked lp over the latent
   tensor and the last consumer disappears, so the extraction is never
   emitted: Gather 20 -> 2 per step (the irreducible noise-slot floor) and
   Scatter Sum 18 -> 0. It also stranded the leapfrog integrator updates
   no longer, since their backward no longer ends in a scatter — 18
   standalone kernels per step fused into their neighbours with no engine
   change (the program's rung 3, verified free).

   The lazy values-map itself is NOT gated: with this knob off the consumed
   set equals dep-order, making it a measured no-op."
  (delay (= "0" (aget (.-env js/process) "GENMLX_AFFINE_FAMILY"))))

;; WHY MATMUL AND NOT A FUSABLE REDUCE (measured 2026-07-29, sm_120,
;; genmlx-1fbs). Matmul is a fusion barrier, so emitting mean_g = Σ_k
;; B[g,k]·q_k as a broadcast-multiply plus a sum-reduce looks strictly
;; better — reductions demonstrably DO fuse here (the census shows
;; …MultiplyAddNegativeSum kernels). It measures WORSE and was reverted:
;; HMC went 62.72 -> 71.71 real launches/step. The census says why. The
;; forward did fuse, to one CompiledBroadcastMultiplySum, but the vjp
;; split into CompiledBroadcastMultiply + a standalone Sum — 900 + 900
;; kernels at S=100 — because the backward reduces over g, the LEADING
;; axis, and only the trailing-axis reduction fused. So the reduce form
;; costs 3 kernels per grad eval against the matmul's 2, and transposing
;; the intermediate only moves the unfused reduction from the backward to
;; the forward. Rung-4 candidate 2 ("the fwd matmul dissolves into the
;; score kernel") therefore cannot pay off through this route either: the
;; forward matmul is already the cheap half.

(defn- mags
  "|x| for each value, as host numbers, with a SINGLE device sync.
   Values may be plain numbers (an all-equal literal column stays unboxed)
   or MLX arrays; the array magnitudes are stacked and read back once.

   One sync per comparison instead measurably dominates build time on a
   cold context: the detectors below need ~11 magnitudes, which cost
   ~1.3 s of the 1.44 s MALA S=10 warmup when read one at a time
   (measured 2026-07-29, sm_120 — genmlx-1fbs). Warmup is a reported
   parity metric with an enforced ceiling, so this is not micro-tuning."
  [xs]
  (let [xs (vec xs)
        arr-idx (vec (keep-indexed (fn [i x] (when-not (number? x) i)) xs))
        host (mapv #(when (number? %) (js/Math.abs %)) xs)]
    (if (empty? arr-idx)
      host
      (let [stacked (mx/stack (mapv #(mx/amax (mx/abs (nth xs %))) arr-idx))]
        (mx/materialize! stacked)
        (let [hv (vec (js->clj (mx/->clj stacked)))]
          (reduce (fn [acc [j i]] (assoc acc i (nth hv j)))
                  host
                  (map-indexed vector arr-idx)))))))


(defn- eval-args-at-latents
  "Evaluate a site's compiled dist-args with the latents bound to `q`
   (positionally over latent-order) and observed values baked in. Returns
   the evaluated args, or nil if any arg fails or comes back nil."
  [compiled-args obs-values mlx-args latent-order q]
  (try
    (let [vm (reduce (fn [m [i a]] (assoc m a (mx/scalar (nth q i))))
                     obs-values
                     (map-indexed vector latent-order))
          vals (mapv #(% vm mlx-args) compiled-args)]
      (when (every? some? vals) vals))
    (catch :default _ nil)))

(defn- latent-free-args
  "Evaluated dist-args, given they are already PROVEN latent-free by
   `latent-affinity` (:const on every arg — the caller's gate).

   Bit-stability across two differing latent assignments is kept as a
   numeric cross-check, not as the proof: two sample points can agree by
   coincidence for a genuinely latent-dependent graph (any g with
   g(probe-a) = g(probe-b)), which is the same unsoundness genmlx-yy8u found
   in the affine detector. The structural proof is the gate; this only ever
   causes an additional decline, and it also catches a non-deterministic
   graph the source walk cannot see."
  [compiled-args obs-values mlx-args latent-order]
  (let [k (count latent-order)
        probe-a (mapv #(+ 0.5 (* 0.25 (js/Math.sin (* 12.9898 (inc %)))))
                      (range k))
        probe-b (mapv #(- -1.25 (* 0.5 (js/Math.cos (* 7.3311 (inc %)))))
                      (range k))
        va (eval-args-at-latents compiled-args obs-values mlx-args
                                 latent-order probe-a)
        vb (eval-args-at-latents compiled-args obs-values mlx-args
                                 latent-order probe-b)]
    (when (and va vb (= (count va) (count vb)))
      ;; All drifts in one sync (see `mags`).
      (let [pairs (map vector va vb)
            both-numbers? (mapv (fn [[a b]] (and (number? a) (number? b))) pairs)
            drifts (mags (map (fn [[a b] nums?]
                                (if nums? (- a b) (mx/subtract a b)))
                              pairs both-numbers?))]
        (when (every? zero? drifts) va)))))

(defn- stack-arg-column
  "Collapse one dist-arg position across a group of sites into a single
   constant: a plain number when every site agrees (no [K] input for a
   shared literal), else a materialized [K] column in group order. Returns
   nil if any value is non-scalar (a stacked column would change meaning)."
  [vals]
  (cond
    (and (every? number? vals) (apply = vals)) (first vals)
    (some #(and (not (number? %)) (pos? (mx/ndim %))) vals) nil
    :else (let [a (mx/stack (mapv #(if (number? %) (mx/scalar %) %) vals))]
            (mx/materialize! a)
            a)))

(defn- build-stacked-prior-lp-fns
  "Stacked latent-prior scoring (genmlx-1fbs): score ALL K latent priors as
   ONE elementwise log-prob over the latent tensor itself —

     lp = Σ_k log-prob(q_k; muCol_k, sigmaCol_k)

   — instead of K per-site subgraphs each opening with a gather of its own
   latent. The columns are factory-time constants (same lifecycle as the
   family lit-cols), so the forward is one elementwise kernel chain over
   [K] (scalar) / [N,K] (batched) and the backward is its elementwise
   cotangent — no index forward, no scatter-add backward.

   Requires FULL COVER: every latent site eligible, one shared dist-type,
   equal arity, and every dist-arg PROVEN latent-free — structurally, by
   `latent-affinity` over the source form (:const), with the bit-stability
   probe kept only as a cross-check (genmlx-yy8u sibling: two probe points
   agreeing is evidence, not proof). A partial cover would have to slice the
   tensor, and a slice's vjp is a scatter — exactly the tax this removes —
   so partial groups decline to per-site scoring instead. Hierarchical
   priors (latent-dependent args) therefore keep the per-site path, and the
   lazy values-map keeps them correct.

   latent-order fixes the column order: the scalar tensor layout, or the
   batched caller's `addresses`. Returns [lp-fns handled-addrs]; each
   lp-fn takes the latent tensor DIRECTLY, not a values-map."
  [latent-specs latent-order obs-values mlx-args binding-env probe-tensor
   reduce-fn]
  (if (or @affine-family-disabled? (empty? latent-order))
    [[] #{}]
    (try
      (let [specs (mapv latent-specs latent-order)
            latent-set (set latent-order)
            dts (into #{} (map :dist-type) specs)
            arities (into #{} (map #(count (:compiled-args %))) specs)
            dt (first dts)
            ;; STRUCTURAL proof first — every prior dist-arg latent-free.
            latent-free-source?
            (and (every? some? specs)
                 (every? (fn [s]
                           (let [da (:dist-args s)]
                             ;; nil dist-args = no source form to prove
                             ;; against; decline rather than assume.
                             (and (seq da)
                                  (every? #(= :const (latent-affinity
                                                      % binding-env latent-set #{}))
                                          da))))
                         specs))
            ;; Numeric cross-check AND the column values in one pass.
            arg-vals (when latent-free-source?
                       (mapv #(latent-free-args (:compiled-args %) obs-values
                                                mlx-args latent-order)
                             specs))]
        (if-not (and (= 1 (count dts))
                     (= 1 (count arities))
                     (contains? family-elementwise-dists dt)
                     (:log-prob (get compiled/noise-transforms-full dt))
                     arg-vals
                     (every? some? arg-vals))
          [[] #{}]
          (let [log-prob-fn (:log-prob (get compiled/noise-transforms-full dt))
                cols (mapv (fn [j] (stack-arg-column (mapv #(nth % j) arg-vals)))
                           (range (first arities)))]
            (if-not (every? some? cols)
              [[] #{}]
              (let [lp-fn (fn stacked-prior-lp [latent-tensor]
                            (reduce-fn (apply log-prob-fn latent-tensor cols)))]
                (if (lp-fn-usable? lp-fn @probe-tensor "stacked-prior")
                  [[lp-fn] (set latent-order)]
                  [[] #{}]))))))
      (catch :default e
        (println (str "  [genmlx] compiled stacked-prior scoring declined to"
                      " the per-site path: " (.-message e)))
        [[] #{}]))))

(defn- build-affine-family-lp-fns
  "Matmul-form emission (genmlx-1fbs) for gaussian observed families whose
   MEAN sig-arg is AFFINE in the latents and whose SIGMA is latent-free:

     mean_g = Σ_k B[g,k]·latent_k + base_g
     scalar:  mean = (matmul B q) + base          (B [G,K], q [K])
     batched: mean = (matmul params Bᵀ) + base    (params [N,K])

   One matmul forward and one matmul backward per grad eval — the vjp
   Bᵀ·dmu also ABSORBS the per-family cotangent reduction — replacing the
   per-latent index/one-hot extraction and the scatter-add gradient
   assembly (the HMC-1.0x gather/scatter tax, ~4 kernels/eval).

   Detection is a STRUCTURAL PROOF plus a numeric cross-check. The mean sig
   must be proven :affine and the sigma sig :const by `latent-affinity`,
   which walks the source form and requires every product/quotient to carry
   at most ONE latent-bearing factor — the analysis genmlx-1fbs specified.
   The original implementation shipped the probe ALONE, and a numeric probe
   cannot decide global affinity: it evaluated the mean at q = 0, at each
   basis vector and at ONE interior point with |q| <= 0.75, accepting at
   1e-4 * max(1, |mean|). The residual of a quadratic is structurally
   pinned to zero at q = 0 and at every basis point, so the verdict rested
   on a single point inside the unit ball while MCMC explores far outside
   it, and the tolerance was scaled by the MEAN with sigma appearing
   nowhere. Measured on sm_120: `mean = 100*slope + 0.005*slope^2` was
   ACCEPTED, the compiled score diverged 945 nats from the handler at
   slope = 10 and the gradient at the mode was 3x wrong; with
   |mu| ~ 1e4 / sigma = 0.01 the divergence was 90,007 nats AT THE MODE
   (genmlx-yy8u).

   The numeric check is retained as a cross-check on the proof — it also
   catches a non-deterministic graph a source walk cannot see — but it now
   probes THREE points, two of them well outside the unit ball (|q| ~ 3 and
   ~10), and its tolerance is max(1e-4 * min|sigma|, 1e-5 * scale): the
   sigma term is the one that matters for the score, the scale term is only
   the float32 evaluation-noise floor. Sigma must additionally be bit-stable
   across every eval. A failed proof or probe falls back to the stacked
   emission — never an error.

   B/base/sigma/stacked-obs are factory-time materialized constants, the
   same lifecycle as the family lit-cols.

   latent-order fixes B's column order: the scalar tensor layout, or the
   batched caller's `addresses`. Returns [lp-fns handled-addrs]; each
   lp-fn takes the latent tensor DIRECTLY ([K] scalar / [N,K] batched),
   not a values-map."
  [families binding-env mlx-args obs-values latent-order probe-tensor
   reduce-fn batched?]
  (if (or @affine-family-disabled? (empty? latent-order))
    [[] #{}]
    (let [k-lat (count latent-order)
          latent-set (set latent-order)
          built
          (keep
           (fn [{:keys [dist-type sig members lits-per-site]}]
             (try
               (when (contains? #{:gaussian :normal} dist-type)
                 (let [{:keys [cargs args+lits env]}
                       (compile-family-sig sig lits-per-site binding-env mlx-args)
                       [mean-c sigma-c] cargs
                       g (count members)]
                   (when (and (= 2 (count cargs)) mean-c sigma-c
                              ;; THE soundness gate. Everything below is
                              ;; construction and cross-checking.
                              (= :affine (latent-affinity (first sig) env
                                                          latent-set #{}))
                              (= :const (latent-affinity (second sig) env
                                                         latent-set #{})))
                     (let [eval-at (fn [q]
                                     (let [vm (reduce (fn [m [i a]]
                                                        (assoc m a (mx/scalar (nth q i))))
                                                      obs-values
                                                      (map-indexed vector latent-order))]
                                       [(mean-c vm args+lits) (sigma-c vm args+lits)]))
                           zeros (vec (repeat k-lat 0.0))
                           [base sigma0] (eval-at zeros)
                           basis (mapv #(eval-at (assoc zeros % 1.0)) (range k-lat))
                           cols (mapv (fn [[m _]] (mx/subtract m base)) basis)
                           ;; Three probe points, NOT one, and two of them
                           ;; far outside the unit ball where MCMC actually
                           ;; goes (genmlx-yy8u).
                           probes [(mapv #(+ 0.5 (* 0.25 (js/Math.sin (* 12.9898 (inc %)))))
                                         (range k-lat))
                                   (mapv #(* 3.0 (+ 1.0 (* 0.5 (js/Math.sin (* 3.1 (inc %))))))
                                         (range k-lat))
                                   (mapv #(* -10.0 (+ 1.0 (* 0.3 (js/Math.cos (* 2.7 (inc %))))))
                                         (range k-lat))]
                           evals (mapv eval-at probes)
                           predicteds (mapv (fn [q]
                                              (reduce (fn [acc k]
                                                        (mx/add acc (mx/multiply
                                                                     (nth cols k)
                                                                     (mx/scalar (nth q k)))))
                                                      base (range k-lat)))
                                            probes)
                           ;; Every probe magnitude in ONE sync (see `mags`):
                           ;; sigma drifts, then the per-probe affinity
                           ;; residuals, then their scales, then min|sigma|,
                           ;; then the coefficient columns.
                           sigma-diffs (mapv #(mx/subtract % sigma0)
                                             (into (mapv second basis)
                                                   (mapv second evals)))
                           n-sig (count sigma-diffs)
                           n-pr (count probes)
                           sigma-min-el (if (number? sigma0)
                                          sigma0
                                          (mx/amin (mx/abs sigma0)))
                           ms (mags (concat sigma-diffs
                                            (mapv (fn [[m _] p] (mx/subtract m p))
                                                  evals predicteds)
                                            predicteds
                                            [sigma-min-el]
                                            cols))
                           sigma-stable? (every? zero? (take n-sig ms))
                           errs (subvec (vec ms) n-sig (+ n-sig n-pr))
                           scale (apply max 1.0 (subvec (vec ms) (+ n-sig n-pr)
                                                        (+ n-sig n-pr n-pr)))
                           sigma-min (nth ms (+ n-sig n-pr n-pr))
                           ;; Two floors ANDed, not maxed (genmlx-yy8u review).
                           ;; `(max sigma-term scale-term)` lets the scale term
                           ;; WIN whenever the mean is large — with |mu| ~ 1e6
                           ;; and sigma = 0.01 it admitted a residual of 10,
                           ;; i.e. 1000 sigma, which is exactly the regime that
                           ;; measured 90,007 nats. The scale term exists only
                           ;; as a float32 evaluation-noise floor, so keep it,
                           ;; but cap every accepted residual in SIGMA units so
                           ;; no admitted error can ever be worth more than
                           ;; ~1e-4 nats regardless of |mean|.
                           tol      (max (* 1e-4 sigma-min) (* 1e-5 scale))
                           sigma-cap (* 1e-2 sigma-min)
                           within?  (fn [e] (and (<= e tol) (<= e sigma-cap)))
                           any-coeff? (boolean (some pos? (drop (+ n-sig n-pr n-pr 1) ms)))]
                       (when (and sigma-stable? (every? within? errs) any-coeff?)
                         (let [log-prob-fn (:log-prob (get compiled/noise-transforms-full dist-type))
                               bcast (fn [x]
                                       (cond
                                         (number? x) (mx/broadcast-to (mx/scalar x) [g])
                                         (zero? (mx/ndim x)) (mx/broadcast-to x [g])
                                         :else x))
                               b-mat (let [b (mx/stack (mapv bcast cols) 1)
                                           b (if batched? (mx/transpose b) b)]
                                       (mx/materialize! b)
                                       b)
                               base-v (let [b (bcast base)] (mx/materialize! b) b)
                               sigma-v (if (number? sigma0)
                                         sigma0
                                         (do (mx/materialize! sigma0) sigma0))
                               stacked (stacked-obs obs-values members)
                               lp-fn (if batched?
                                       (fn affine-batched-lp [params]
                                         (reduce-fn
                                          (log-prob-fn stacked
                                                       (mx/add (mx/matmul params b-mat) base-v)
                                                       sigma-v)))
                                       (fn affine-lp [latent-tensor]
                                         (reduce-fn
                                          (log-prob-fn stacked
                                                       (mx/add (mx/matmul b-mat latent-tensor)
                                                               base-v)
                                                       sigma-v))))]
                           (when (lp-fn-usable? lp-fn @probe-tensor "affine-family")
                             {:lp-fn lp-fn :members members})))))))
               (catch :default e
                 (println (str "  [genmlx] compiled affine-family scoring declined"
                               " to the stacked path: " (.-message e)))
                 nil)))
           families)]
      [(mapv :lp-fn built)
       (into #{} (mapcat :members) built)])))

(defn- build-scoring-plan
  "Assemble the scoring plan shared by the scalar and batched tensor scores.
   The two differ only in three things — the latent ordering (tensor layout
   vs the caller's `addresses`), how an elementwise lp is reduced to the
   score shape, and whether the affine basis is transposed — so they are
   parameters here rather than two copies of the strategy stack.

   Strategy order, first match wins per site: affine matmul family >
   stacked observed family > stacked latent priors > per-site. Every
   detector may DECLINE — on a structural proof it could not complete, on a
   numeric cross-check, or because the emission it built THREW when forced
   once at a probe point (genmlx-spid) — in which case the site falls
   through to the next strategy; a site with no supported log-prob declines
   the whole build (nil). No detector may propagate a throw: compilation is
   an optimization and the handler is ground truth (CLAUDE.md rule 5).

   Returns {:site-lp-fns  fns taking a values-map
            :tensor-lp-fns fns taking the latent tensor/params DIRECTLY
            :vm-addrs     addresses the values-map must carry, or nil
            :latent-index {addr -> column}
            :emission     engagement report}
   or nil."
  [schema site-specs binding-env mlx-args obs-values latent-order
   reduce-fn batched?]
  (let [latent-index (into {} (map-indexed (fn [i a] [a i]) latent-order))
        static-sites (filterv :static? (:trace-sites schema))
        families (detect-observed-families
                  (filterv #(contains? obs-values (:addr %)) static-sites))
        ;; A zero latent point of the shape the tensor lp-fns consume,
        ;; used to force each freshly built emission once (lp-fn-usable?).
        probe-tensor (delay (let [k (count latent-order)]
                              (if batched? (mx/zeros [1 k]) (mx/zeros [k]))))
        [affine-lp-fns affine-addrs]
        (build-affine-family-lp-fns families binding-env mlx-args obs-values
                                    latent-order probe-tensor reduce-fn batched?)
        [family-lp-fns family-addrs]
        (build-family-lp-fns (remove #(contains? affine-addrs
                                                 (first (:members %)))
                                     families)
                             binding-env mlx-args obs-values latent-order
                             reduce-fn)
        ;; The latent site-specs carry compiled closures only; the STRUCTURAL
        ;; latent-free proof needs the source dist-args, so merge them back
        ;; from the schema by address.
        dist-args-by-addr (into {} (map (juxt :addr :dist-args)) static-sites)
        [prior-lp-fns prior-addrs]
        (build-stacked-prior-lp-fns
         (into {} (comp (filter #(contains? latent-index (:addr %)))
                        (map (fn [ss]
                               [(:addr ss)
                                (assoc ss :dist-args
                                       (get dist-args-by-addr (:addr ss)))])))
               site-specs)
         latent-order obs-values mlx-args binding-env probe-tensor reduce-fn)
        per-site
        (mapv
         (fn [site-spec]
           (let [{:keys [addr compiled-args dist-type]} site-spec
                 nt (get compiled/noise-transforms-full dist-type)]
             (if (or (contains? family-addrs addr)
                     (contains? affine-addrs addr)
                     (contains? prior-addrs addr))
               ::family
               (when nt
                 (let [log-prob-fn (:log-prob nt)]
                   (fn [values-map]
                     (let [eval-args (mapv #(% values-map mlx-args) compiled-args)]
                       (apply log-prob-fn (get values-map addr) eval-args))))))))
         site-specs)]
    (when (every? some? per-site)
      (let [site-lp-fns (into (filterv fn? per-site) family-lp-fns)
            ;; LAZY VALUES-MAP (genmlx-1fbs): the map exists only to feed
            ;; values-map-based lp-fns. With no survivors — every site
            ;; scored by an affine family or the stacked prior — nothing
            ;; reads it, so the per-latent extraction (mx/index scalar,
            ;; one-hot matmul batched) is never emitted at all. That is the
            ;; step that makes the two rung-2 emissions pay. The rule is
            ;; binary on purpose: an empty consumer list provably reads
            ;; nothing, whereas a per-address consumption analysis would
            ;; have to out-guess the compiled-arg closures.
            vm-addrs (when (seq site-lp-fns) (:dep-order schema))]
        {:site-lp-fns   site-lp-fns
         :tensor-lp-fns (into (vec affine-lp-fns) prior-lp-fns)
         :vm-addrs      vm-addrs
         :latent-index  latent-index
         ;; Engagement report. It exists so a silent decline is VISIBLE:
         ;; every emission above is probe-detected and can fall back
         ;; correctly and quietly, and then a kernel census reads as "the
         ;; optimization did not pay" when the truth is "it never ran".
         :emission {:affine-families   (count affine-lp-fns)
                    :stacked-priors?   (boolean (seq prior-lp-fns))
                    :per-site-lps      (count (filterv fn? per-site))
                    :extracted-latents (count (filterv latent-index vm-addrs))}}))))

(defn build-tensor-score
  "make-tensor-score plus an :emission report (see build-scoring-plan —
   tests pin engagement through it).

   Returns {:score-fn f :emission {...}} or nil."
  [schema source args observations]
  (when-let [{:keys [site-specs addrs binding-env]}
             (compiled/prepare-static-sites schema source)]
    (let [mlx-args (compiled/ensure-mlx-args (vec args))
          ;; Separate observed vs latent using source-order static-sites
          ;; (matches compiled/prepare-static-sites addr order)
          all-addrs addrs
          obs-addrs (set (map first (cm/addresses observations)))
          latent-addrs (vec (remove obs-addrs all-addrs))
          ;; Pre-extract observed values
          obs-values (into {} (keep (fn [addr]
                                      (when (obs-addrs addr)
                                        (let [sub (cm/get-submap observations addr)]
                                          (when (cm/has-value? sub)
                                            [addr (cm/get-value sub)]))))
                              all-addrs))
          plan (build-scoring-plan schema site-specs binding-env mlx-args
                                   obs-values latent-addrs mx/sum false)]
      (when plan
        (let [{:keys [site-lp-fns tensor-lp-fns vm-addrs emission
                      latent-index]} plan]
          {:emission emission
           :score-fn
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
                     vm-addrs)]
               ;; Sum all site log-probs; affine families and stacked priors
               ;; read the latent tensor directly (no per-latent index).
               (reduce
                 (fn [score lp-fn]
                   (mx/add score (lp-fn latent-tensor)))
                 (reduce
                   (fn [score lp-fn]
                     (mx/add score (lp-fn values-map)))
                   (mx/scalar 0.0)
                   site-lp-fns)
                 tensor-lp-fns)))})))))

(defn make-tensor-score
  "Build a tensor-native score function: [K]-tensor → scalar log-prob.
   Bypasses GFI protocol — uses L1 noise-transform log-prob closures directly.
   Observations are baked in as constants. Only latent values come from the tensor.

   Homogeneous observed site families are scored as one stacked [G]
   log-prob (genmlx-yopl — see the family helpers above); all other sites
   keep per-site lp subgraphs. The total is the same joint score up to
   float32 summation order.

   Returns (fn [latent-tensor] -> MLX scalar) or nil if model can't be compiled.

   latent-tensor: [K] MLX array where K = number of latent sites.
   The addr-index for the tensor is returned as metadata via make-tensor-score-with-index.

   schema: the :schema from a DynamicGF
   source: the :source from a DynamicGF
   args: argument vector (will be converted to MLX arrays)
   observations: ChoiceMap of observed values"
  [schema source args observations]
  (:score-fn (build-tensor-score schema source args observations)))

(defn make-tensor-score-with-index
  "Like make-tensor-score but also returns the latent addr-index and the
   :emission report (see build-tensor-score).
   Returns {:score-fn (fn [K-tensor] -> scalar) :latent-index {addr -> int}
            :emission {...}} or nil."
  [schema source args observations]
  (when (and schema (:static? schema)
             (seq (:trace-sites schema))
             (empty? (:splice-sites schema))
             (empty? (:param-sites schema)))
    (let [static-sites (filterv :static? (:trace-sites schema))
          obs-addrs (set (map first (cm/addresses observations)))
          latent-addrs (vec (remove obs-addrs (mapv :addr static-sites)))
          latent-index (into {} (map-indexed (fn [i a] [a i]) latent-addrs))
          built (build-tensor-score schema source args observations)]
      (when built
        {:score-fn (:score-fn built)
         :emission (:emission built)
         :latent-index latent-index}))))

(defn make-batched-tensor-score-with-index
  "Batched (shape-based) tensor-native score: (fn [[N,K] params] -> [N])
   joint log-prob, one graph for all N chains — the tensor-native analog of
   u/make-batched-score-fn, with the same family vectorization as
   make-tensor-score (family lps broadcast [N,1] × [G] → [N,G], last-axis
   keepdims sum). Column k of params carries `(nth addresses k)` — the
   CALLER's ordering, matching the [N,D] state the fused vectorized chains
   build from `addresses`. Latent columns are extracted with one-hot
   matmuls ([N,K]×[K,1] → [N,1]) because MLX's transpose+index does not
   differentiate (see u/make-differentiable-vectorized-score-fn).
   Returns {:score-fn fn :latent-index {addr -> int}} or nil when the model
   doesn't tensor-compile or `addresses` doesn't cover exactly the
   non-observed static sites."
  [schema source args observations addresses]
  (when (and schema (:static? schema)
             (seq (:trace-sites schema))
             (empty? (:splice-sites schema))
             (empty? (:param-sites schema)))
    (when-let [{:keys [site-specs addrs binding-env]}
               (compiled/prepare-static-sites schema source)]
      (let [mlx-args (compiled/ensure-mlx-args (vec args))
            obs-addrs (set (map first (cm/addresses observations)))
            latent-addrs (vec (remove obs-addrs addrs))
            addresses (vec addresses)]
        ;; The caller's addresses must be exactly the latent set — a subset
        ;; would leave sites unvalued (the scalar path samples the full
        ;; joint instead; here we decline to the GFI batched score).
        (when (= (set addresses) (set latent-addrs))
          (let [k (count addresses)
                latent-index (into {} (map-indexed (fn [i a] [a i]) addresses))
                one-hots (mapv (fn [i]
                                 (let [v (vec (repeat k 0.0))
                                       a (mx/array (assoc v i 1.0) [k 1])]
                                   (mx/materialize! a)
                                   a))
                               (range k))
                obs-values (into {} (keep (fn [addr]
                                            (when (obs-addrs addr)
                                              (let [sub (cm/get-submap observations addr)]
                                                (when (cm/has-value? sub)
                                                  [addr (cm/get-value sub)]))))
                                          addrs))
                ;; Same strategy stack as the scalar score; the batched
                ;; differences are the caller's `addresses` ordering, the
                ;; last-axis keepdims reduce ([N,1]), and the transposed
                ;; affine basis.
                plan (build-scoring-plan schema site-specs binding-env
                                         mlx-args obs-values addresses
                                         #(mx/sum % -1 true) true)]
            (when plan
              (let [{:keys [site-lp-fns tensor-lp-fns vm-addrs emission]} plan]
                {:emission emission
                 :score-fn
                 (fn batched-tensor-score [params]
                   (let [values-map
                         (reduce
                           (fn [vm addr]
                             (assoc vm addr
                                    (if-let [idx (get latent-index addr)]
                                      (mx/matmul params (nth one-hots idx))
                                      (get obs-values addr))))
                           {}
                           vm-addrs)
                         ;; Terms are [N,1] (latent-arg lps, family sums) or
                         ;; scalar-broadcastable; total [N,1] → squeeze → [N].
                         ;; Affine families and stacked priors read the
                         ;; [N,K] params directly.
                         total (reduce
                                 (fn [score lp-fn]
                                   (mx/add score (lp-fn params)))
                                 (reduce
                                   (fn [score lp-fn]
                                     (mx/add score (lp-fn values-map)))
                                   (mx/scalar 0.0)
                                   site-lp-fns)
                                 tensor-lp-fns)]
                     (mx/squeeze total [1])))
                 :latent-index latent-index}))))))))

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
  (when-let [{:keys [site-specs binding-env addrs]}
             (compiled/prepare-static-sites schema source)]
    ;; Every site must be a NORMAL-noise distribution or a true
    ;; delta. The supplied noise slice is standard normal, so a
    ;; latent uniform/bernoulli/exponential/laplace/cauchy site
    ;; (whose :noise-fn is inverse-CDF from UNIFORM noise) would be
    ;; fed the wrong noise and corrupt the particle + its bootstrap
    ;; weight (genmlx-j22a). A latent :args-noise-fn site
    ;; (iid-gaussian) would silently degrade to value=first-arg
    ;; (genmlx-b210). Both are declined at build time, falling back
    ;; to the handler SMC path.
    (when (every? (fn [ss]
                    (or (= :delta (:dist-type ss))
                        (contains? compiled/normal-noise-dist-types
                                   (:dist-type ss))))
                  site-specs)
        (let [dep-order (:dep-order schema)
              all-addrs addrs
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
                            ;; Delta: value = first dist arg (only true deltas
                            ;; reach here — the build-time gate above declines
                            ;; every other no-noise-fn transform)
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
               :retval (when retval-fn (retval-fn values-map mlx-args))}))))))

