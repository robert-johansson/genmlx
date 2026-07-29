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

(defn- abstract-dist-args
  "Abstract a site's dist-arg source forms for family matching: every
   numeric literal becomes a positional placeholder symbol ᐩfam<i>; the
   literal values are collected in walk order. Declines (nil) forms that
   embed a (trace ...) call (an inline site definition must never be
   family-merged) or map literals (walk-order stability).
   Returns {:sig [forms'] :lits [numbers]} or nil."
  [dist-args]
  (let [counter (volatile! 0)
        lits (volatile! [])
        ok? (volatile! true)
        walk (fn walk [form]
               (cond
                 (number? form)
                 (let [i @counter]
                   (vswap! counter inc)
                   (vswap! lits conj form)
                   (symbol (str "ᐩfam" i)))

                 (and (seq? form) (seq form))
                 (if (and (symbol? (first form)) (= "trace" (name (first form))))
                   (do (vreset! ok? false) form)
                   (doall (map walk form)))

                 (vector? form) (mapv walk form)
                 (map? form) (do (vreset! ok? false) form)
                 :else form))
        sig (mapv walk dist-args)]
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
   Returns {:cargs [fn-or-nil ...] :args+lits [...]}."
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
     :args+lits (into mlx-args lit-cols)}))

(defn- build-family-lp-fns
  "Build one stacked log-prob closure per family (see compile-family-sig
   for the literal-column mechanics). reduce-fn folds the elementwise
   [.., G]-shaped lp to the score shape (full mx/sum for the scalar score;
   last-axis keepdims sum for the batched [N,1] score).
   Returns [lp-fns family-addrs]; a family whose sig fails to compile falls
   back to per-site scoring (dropped from family-addrs)."
  [families binding-env mlx-args obs-values reduce-fn]
  (let [built
        (keep (fn [{:keys [dist-type sig members lits-per-site]}]
                (let [log-prob-fn (:log-prob (get compiled/noise-transforms-full dist-type))
                      {:keys [cargs args+lits]}
                      (compile-family-sig sig lits-per-site binding-env mlx-args)
                      stacked (let [s (mx/stack (mapv obs-values members))]
                                (mx/materialize! s)
                                s)]
                  (when (every? some? cargs)
                    {:lp-fn (fn [values-map]
                              (let [eval-args (mapv #(% values-map args+lits) cargs)]
                                (reduce-fn (apply log-prob-fn stacked eval-args))))
                     :members members})))
              families)]
    [(mapv :lp-fn built)
     (into #{} (mapcat :members) built)]))

(def ^:private affine-family-disabled?
  "Matmul-form family emission is OPT-IN via GENMLX_AFFINE_FAMILY=1
   (genmlx-1fbs). Measured 2026-07-29 (sm_120): the emission alone is
   net-NEGATIVE on kernel count — it adds the matmuls but the per-latent
   index/one-hot extraction survives for the PRIOR sites, so the
   gather/scatter tax it targets does not drop (MALA tape 4287 -> 4592,
   HMC 16875 -> 19677 at S=100). Flip the default only together with
   stacked latent-prior scoring (the rung-2 companion recorded on the
   bean), re-measured by census."
  (delay (not= "1" (aget (.-env js/process) "GENMLX_AFFINE_FAMILY"))))

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

   Detection is by PROBE, not source walk: the compiled mean arg is
   evaluated at zero, at each latent basis vector (column extraction), and
   at a fixed pseudo-random point; affinity must hold at 1e-4 rel
   (float32) and sigma must be bit-stable across every eval (latent-free
   graphs are deterministic, so any drift is real latent dependence). A
   failed probe falls back to the stacked emission — never an error.
   B/base/sigma/stacked-obs are factory-time materialized constants, the
   same lifecycle as the family lit-cols.

   latent-order fixes B's column order: the scalar tensor layout, or the
   batched caller's `addresses`. Returns [lp-fns handled-addrs]; each
   lp-fn takes the latent tensor DIRECTLY ([K] scalar / [N,K] batched),
   not a values-map."
  [families binding-env mlx-args obs-values latent-order reduce-fn batched?]
  (if (or @affine-family-disabled? (empty? latent-order))
    [[] #{}]
    (let [k-lat (count latent-order)
          ;; |x| as a JS number for probe checks; eval results may be plain
          ;; numbers (all-equal literal columns stay unboxed).
          mag (fn [x] (if (number? x)
                        (js/Math.abs x)
                        (mx/item (mx/amax (mx/abs x)))))
          built
          (keep
           (fn [{:keys [dist-type sig members lits-per-site]}]
             (when (contains? #{:gaussian :normal} dist-type)
               (let [{:keys [cargs args+lits]}
                     (compile-family-sig sig lits-per-site binding-env mlx-args)
                     [mean-c sigma-c] cargs
                     g (count members)]
                 (when (and (= 2 (count cargs)) mean-c sigma-c)
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
                         probe (mapv #(+ 0.5 (* 0.25 (js/Math.sin (* 12.9898 (inc %)))))
                                     (range k-lat))
                         [probe-mean probe-sigma] (eval-at probe)
                         predicted (reduce (fn [acc k]
                                             (mx/add acc (mx/multiply (nth cols k)
                                                                      (mx/scalar (nth probe k)))))
                                           base (range k-lat))
                         sigma-stable? (every? #(zero? (mag (mx/subtract % sigma0)))
                                               (conj (mapv second basis) probe-sigma))
                         err (mag (mx/subtract probe-mean predicted))
                         scale (max 1.0 (mag predicted))
                         any-coeff? (boolean (some #(pos? (mag %)) cols))]
                     (when (and sigma-stable? (<= err (* 1e-4 scale)) any-coeff?)
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
                             stacked (let [s (mx/stack (mapv obs-values members))]
                                       (mx/materialize! s)
                                       s)]
                         {:lp-fn (if batched?
                                   (fn affine-batched-lp [params]
                                     (reduce-fn
                                      (log-prob-fn stacked
                                                   (mx/add (mx/matmul params b-mat) base-v)
                                                   sigma-v)))
                                   (fn affine-lp [latent-tensor]
                                     (reduce-fn
                                      (log-prob-fn stacked
                                                   (mx/add (mx/matmul b-mat latent-tensor) base-v)
                                                   sigma-v))))
                          :members members})))))))
           families)]
      [(mapv :lp-fn built)
       (into #{} (mapcat :members) built)])))

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
  (when-let [{:keys [site-specs addrs binding-env]}
             (compiled/prepare-static-sites schema source)]
    (let [mlx-args (compiled/ensure-mlx-args (vec args))
          ;; Separate observed vs latent using source-order static-sites
          ;; (matches compiled/prepare-static-sites addr order)
          all-addrs addrs
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
          ;; Vectorized family scoring (genmlx-yopl): observed homogeneous
          ;; families take a stacked lp; everything else stays per-site.
          ;; Affine gaussian families further take the matmul form
          ;; (genmlx-1fbs), consuming the latent tensor directly.
          static-sites (filterv :static? (:trace-sites schema))
          families (detect-observed-families
                    (filterv #(contains? obs-values (:addr %)) static-sites))
          [affine-lp-fns affine-addrs]
          (build-affine-family-lp-fns families binding-env mlx-args obs-values
                                      latent-addrs mx/sum false)
          [family-lp-fns family-addrs]
          (build-family-lp-fns (remove #(contains? affine-addrs
                                                   (first (:members %)))
                                       families)
                               binding-env mlx-args obs-values mx/sum)
          ;; Build per-site log-prob step functions
          ;; Each returns (fn [values-map] -> log-prob-scalar), ::family for
          ;; family-scored sites, or nil (unsupported → whole build declines)
          per-site
          (mapv
            (fn [site-spec]
              (let [{:keys [addr compiled-args dist-type]} site-spec
                    nt (get compiled/noise-transforms-full dist-type)]
                (if (or (contains? family-addrs addr)
                        (contains? affine-addrs addr))
                  ::family
                  (when nt
                    (let [log-prob-fn (:log-prob nt)]
                      (fn [values-map]
                        (let [eval-args (mapv #(% values-map mlx-args) compiled-args)]
                          (apply log-prob-fn (get values-map addr) eval-args))))))))
            site-specs)]
      (when (every? some? per-site)
        ;; Build the tensor-score closure
        (let [site-lp-fns (into (filterv fn? per-site) family-lp-fns)
              dep-order (:dep-order schema)]
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
              ;; Sum all site log-probs; affine families read the latent
              ;; tensor directly (no per-latent index).
              (reduce
                (fn [score lp-fn]
                  (mx/add score (lp-fn latent-tensor)))
                (reduce
                  (fn [score lp-fn]
                    (mx/add score (lp-fn values-map)))
                  (mx/scalar 0.0)
                  site-lp-fns)
                affine-lp-fns))))))))

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
                static-sites (filterv :static? (:trace-sites schema))
                families (detect-observed-families
                          (filterv #(contains? obs-values (:addr %)) static-sites))
                [affine-lp-fns affine-addrs]
                (build-affine-family-lp-fns families binding-env mlx-args
                                            obs-values addresses
                                            #(mx/sum % -1 true) true)
                [family-lp-fns family-addrs]
                (build-family-lp-fns (remove #(contains? affine-addrs
                                                         (first (:members %)))
                                             families)
                                     binding-env mlx-args obs-values
                                     #(mx/sum % -1 true))
                per-site
                (mapv
                  (fn [site-spec]
                    (let [{:keys [addr compiled-args dist-type]} site-spec
                          nt (get compiled/noise-transforms-full dist-type)]
                      (if (or (contains? family-addrs addr)
                              (contains? affine-addrs addr))
                        ::family
                        (when nt
                          (let [log-prob-fn (:log-prob nt)]
                            (fn [values-map]
                              (let [eval-args (mapv #(% values-map mlx-args) compiled-args)]
                                (apply log-prob-fn (get values-map addr) eval-args))))))))
                  site-specs)]
            (when (every? some? per-site)
              (let [site-lp-fns (into (filterv fn? per-site) family-lp-fns)
                    dep-order (:dep-order schema)]
                {:score-fn
                 (fn batched-tensor-score [params]
                   (let [values-map
                         (reduce
                           (fn [vm addr]
                             (assoc vm addr
                                    (if-let [idx (get latent-index addr)]
                                      (mx/matmul params (nth one-hots idx))
                                      (get obs-values addr))))
                           {}
                           dep-order)
                         ;; Terms are [N,1] (latent-arg lps, family sums) or
                         ;; scalar-broadcastable; total [N,1] → squeeze → [N].
                         ;; Affine families read the [N,K] params directly.
                         total (reduce
                                 (fn [score lp-fn]
                                   (mx/add score (lp-fn params)))
                                 (reduce
                                   (fn [score lp-fn]
                                     (mx/add score (lp-fn values-map)))
                                   (mx/scalar 0.0)
                                   site-lp-fns)
                                 affine-lp-fns)]
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

