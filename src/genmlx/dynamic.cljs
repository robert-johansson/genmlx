(ns genmlx.dynamic
  "DynamicGF — implements the full GFI via a dispatcher stack that
   selects execution strategy per operation: custom handlers, L3
   analytical, L1 compiled, or L0 handler fallback. See dispatch.cljs
   for the protocol and ARCHITECTURE.md Part V for the design."
  (:require [genmlx.protocols :as p]
            [genmlx.handler :as h]
            [genmlx.runtime :as rt]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.selection :as sel]
            [genmlx.vectorized :as vec]
            [genmlx.edit :as edit]
            [genmlx.diff :as diff]
            [genmlx.schema :as schema]
            [genmlx.compiled :as compiled]
            [genmlx.compiled-ops :as cops]
            [genmlx.conjugacy :as conj]
            [genmlx.rewrite :as rewrite]
            [genmlx.linear-gaussian :as lg]
            [genmlx.inference.auto-analytical :as auto]
            [genmlx.dispatch :as dispatch]
            [clojure.set :as set]))

;; Cached zero constant for init states (MLX scalars are immutable)
(def ^:private SCORE-ZERO (mx/scalar 0.0))

;; Forward declarations
(declare execute-sub)
(declare execute-sub-project)
(declare execute-sub-assess)

;; Sentinel value for auto-key: generates a fresh key per GFI call
(def ^:private auto-key-sentinel ::auto-key)

(defn- ensure-key
  "Extract or generate a PRNG key from the gen-fn's metadata.
   Auto-key sentinel -> fresh key. Existing key -> use it. Missing -> error."
  [this]
  (let [k (::key (meta this))]
    (cond
      (= k auto-key-sentinel) (rng/fresh-key)
      k k
      :else (throw (ex-info "No PRNG key on gen-fn. Use (dyn/with-key gf key) or (dyn/auto-key gf)."
                            {:gen-fn (.-source this)})))))

(defn- find-unused-constraints
  "Return set of top-level constraint keys not consumed by the trace, or nil."
  [constraints result-choices]
  (when (and constraints
             (not= constraints cm/EMPTY)
             (instance? cm/Node constraints))
    (let [constraint-keys (set (keys (:m constraints)))
          trace-keys (when (instance? cm/Node result-choices)
                       (set (keys (:m result-choices))))
          unused (set/difference constraint-keys (or trace-keys #{}))]
      (when (seq unused) unused))))

;; ---------------------------------------------------------------------------
;; Trace construction helpers
;; ---------------------------------------------------------------------------

(defn- attach-splice-scores
  "Attach splice scores (and nested splice scores) as metadata on a trace, if
   present. vary-meta, not with-meta: the trace already carries its
   score-type tag (genmlx-lbae) which must survive."
  [trace result]
  (let [ss (:splice-scores result)
        nss (:nested-splice-scores result)]
    (if (or ss nss)
      (vary-meta trace (fn [m] (cond-> (or m {})
                                 ss (assoc ::splice-scores ss)
                                 nss (assoc ::nested-splice-scores nss))))
      trace)))

(defn- make-result-trace
  "Build a Trace from a handler/compiled result map. Tags the trace with the
   state's accumulated score-type — :joint unless a spliced sub-result
   carried a non-joint score (merge-sub-result lubs it into the state)."
  [gf args result]
  (tr/with-score-type
    (tr/make-trace {:gen-fn gf :args args
                    :choices (:choices result)
                    :retval (:retval result)
                    :score (:score result)})
    (or (:score-type result) :joint)))

(defn- attach-unused
  "Assoc :unused-constraints onto a result map when the trace left some
   top-level constraint keys unconsumed."
  [result-map constraints result-choices]
  (if-let [unused (find-unused-constraints constraints result-choices)]
    (assoc result-map :unused-constraints unused)
    result-map))

(defn- make-generate-result
  "Build the generate return map: {:trace :weight}, with unused-constraint
   detection and splice-score attachment."
  [trace weight constraints result-choices result]
  (-> {:trace (attach-splice-scores trace result)
       :weight weight}
      (attach-unused constraints result-choices)))

(defn- make-update-result
  "Build the update return map: {:trace :weight :discard}, with unused-constraint
   detection and splice-score attachment."
  [trace weight discard constraints result-choices result]
  (-> {:trace (attach-splice-scores trace result)
       :weight weight
       :discard discard}
      (attach-unused constraints result-choices)))

(def ^:dynamic *force-general-regen*
  "When true, regenerate always takes the retained-only GENERAL path (the
   project-pass algebra), never the fast path. Used by the fast≡general law
   test to pin equivalence on fast-eligible models. Default false (the fast
   path is auto-selected when provably equivalent)."
  false)

(defn- make-regen-result
  "Build the regenerate return map from handler result, computing the MH weight."
  [gf trace result old-score]
  (let [new-score (:score result)
        proposal-ratio (:weight result)
        weight (mx/subtract (mx/subtract new-score old-score) proposal-ratio)
        new-trace (tr/with-score-type
                    (tr/make-trace {:gen-fn gf :args (:args trace)
                                    :choices (:choices result)
                                    :retval (:retval result)
                                    :score new-score})
                    (or (:score-type result) :joint))]
    {:trace (attach-splice-scores new-trace result)
     :weight weight}))

(defn- selected-path?
  "Does selection `sel` select the full leaf path `path` (a vector of
   addresses)? Descends sub-selections for all but the last address, so it
   handles hierarchical selections over spliced sub-models."
  [sel path]
  (cond
    (nil? sel) false
    (empty? path) false
    (= 1 (count path)) (sel/selected? sel (first path))
    :else (selected-path? (sel/get-subselection sel (first path)) (rest path))))

(defn- regen-retained-selection
  "Selection of the RETAINED leaf addresses of a regenerate move: leaf paths
   present in BOTH the new and old choices, minus the selected ones. Returns
   nil when nothing is retained (weight is then 0)."
  [new-choices old-choices selection]
  (let [old-set (set (cm/addresses old-choices))
        retained (into [] (comp (filter old-set)
                                (remove #(selected-path? selection %)))
                       (cm/addresses new-choices))]
    (when (seq retained)
      (sel/from-paths retained))))

(defn- make-regen-result-general
  "Build the regenerate return for the retained-only GENERAL path
   (genmlx-hmch, genmlx-yep2).

   W = project(new-trace, retained) - project(old-trace, retained)
     = Σ_retained [lp(v; new ctx) - lp(v; old ctx)],
   where retained = leaf addresses present in BOTH executions and unselected.
   Selected, fresh (structure change), and removed sites are excluded and
   contribute 0. Each project pass restores the corresponding execution context
   (args + values of that trace) and recurses through splices, so dependent
   retained sites — whose parameters moved because a selected upstream site was
   resampled — are scored correctly under both contexts (the yep2 fix), and
   spliced sub-models compose with no weight bookkeeping in the parent."
  [gf trace result selection]
  (let [new-trace (-> (tr/make-trace {:gen-fn gf :args (:args trace)
                                      :choices (:choices result)
                                      :retval (:retval result)
                                      :score (:score result)})
                      (tr/with-score-type (or (:score-type result) :joint))
                      (attach-splice-scores result))
        retained-sel (regen-retained-selection (:choices result) (:choices trace) selection)
        weight (if retained-sel
                 (mx/subtract (p/project gf new-trace retained-sel)
                              (p/project gf trace retained-sel))
                 SCORE-ZERO)]
    {:trace new-trace :weight weight}))

(defn- regen-fast-eligible?
  "The fast regenerate path (per-site convention, no project pass) is provably
   equivalent to the general retained-only path iff (a) no structure change can
   occur and (b) the selected objects — trace sites AND splices — are mutually
   independent (no selected object feeds another selected object's distribution
   parameters or arguments). Otherwise the general path is required for
   correctness.

   (a) holds when the model has no branches (the address set is fixed).
   (b) holds when, for every selected static site, none of its dependency set
       (:deps trace-address edges + :splice-deps splice-retval edges,
       genmlx-njzu/dv66) contains another selected object, and for every
       selection-touched splice, no selected site feeds its args and no other
       touched splice feeds it.

   A selected object feeding a RETAINED (unselected) one is fine on the fast
   path: the retained site's (lp-new − lp-old) lands in the weight via the
   score difference, matching the general path's retained-only term — only
   jointly-selected interacting pairs diverge (verified both ways in
   regen_gate_test).

   Returns false conservatively whenever the schema lacks the static info
   needed to prove eligibility (dynamic addresses, loops, missing trace-sites)."
  [gf selection]
  (let [schema (:schema gf)
        sites (or (:trace-sites schema) [])
        splices (or (:splice-sites schema) [])]
    (cond
      *force-general-regen* false
      (:has-branches? schema) false
      (:dynamic-addresses? schema) false
      (:has-loops? schema) false
      ;; Opaque-escape bodies have hidden trace sites the walker never recorded,
      ;; so fast-eligibility cannot be proven (a hidden structure change or
      ;; interdependent hidden site would mis-weight or throw). Force the general
      ;; retained-only path (genmlx-9yuw).
      (:opaque-gen-escape? schema) false
      :else
      ;; Spliced sub-gfs recurse through p/regenerate and gate themselves —
      ;; execute-sub composes their weights exactly. What they can NOT see is
      ;; coupling through the PARENT: a splice retval feeding a selected parent
      ;; site, or a selected parent site feeding a splice's args. When both
      ;; ends of such an edge are selected, the fast per-site weight is wrong
      ;; (genmlx-njzu), so those edges join the mutual-independence check.
      (let [selected (into #{} (comp (map :addr)
                                     (filter #(sel/selected? selection %)))
                           sites)
            ;; A splice is "touched" when its address is selected or the
            ;; selection descends into it (hierarchical / complement forms).
            touched? (fn [addr]
                       (or (sel/selected? selection addr)
                           (not (identical? sel/none
                                            (sel/get-subselection selection addr)))))
            touched-splices (into #{} (comp (map :addr) (filter touched?))
                                  splices)]
        (and
         (every? (fn [s]
                   (or (not (contains? selected (:addr s)))
                       (and (empty? (set/intersection selected (set (:deps s))))
                            (empty? (set/intersection touched-splices
                                                      (set (:splice-deps s)))))))
                 sites)
         (every? (fn [sp]
                   (or (not (contains? touched-splices (:addr sp)))
                       (and (empty? (set/intersection selected (set (:deps sp))))
                            (empty? (set/intersection (disj touched-splices (:addr sp))
                                                      (set (:splice-deps sp)))))))
                 splices))))))

;; ---------------------------------------------------------------------------
;; Execution helpers — each takes [gf args key opts] and returns a GFI result.
;; Grouped by execution strategy (handler, compiled, prefix, analytical).
;; ---------------------------------------------------------------------------

;; -- Common: param-store and body-fn extraction --

(defn- param-store [gf] (::param-store (meta gf)))
(defn- run-body [gf rt args] (apply (:body-fn gf) rt args))

;; -- Transition-parameterized handler execution --
;;
;; Each function takes [transition gf args key opts] and runs the model
;; body through run-handler with the given transition. The handler-table
;; partially applies standard transitions; with-handler uses custom ones.

(defn- run-simulate [transition gf args key _opts]
  (let [result (rt/run-handler transition
                 {:choices cm/EMPTY :score SCORE-ZERO :key key
                  :executor execute-sub :param-store (param-store gf)}
                 (fn [rt] (run-body gf rt args)))]
    (attach-splice-scores (make-result-trace gf args result) result)))

(defn- run-generate [transition gf args key {:keys [constraints]}]
  (let [result (rt/run-handler transition
                 {:choices cm/EMPTY :score SCORE-ZERO :weight SCORE-ZERO
                  :key key :constraints constraints
                  :executor execute-sub :param-store (param-store gf)}
                 (fn [rt] (run-body gf rt args)))
        trace (make-result-trace gf args result)]
    (make-generate-result trace (:weight result) constraints (:choices result) result)))

(defn- run-update
  "args is the execution-time argument vector: (:args trace) for plain
   update, the thesis x' for update-with-args. Retained/constrained sites
   are re-scored under it; the result trace carries it."
  [transition gf args key {:keys [trace constraints]}]
  (let [result (rt/run-handler transition
                 {:choices cm/EMPTY :score SCORE-ZERO :weight SCORE-ZERO
                  :key key :constraints constraints
                  :old-choices (:choices trace)
                  :old-splice-scores (::splice-scores (meta trace))
                  :old-nested-splice-scores (::nested-splice-scores (meta trace))
                  :discard cm/EMPTY
                  :executor execute-sub :param-store (param-store gf)}
                 (fn [rt] (run-body gf rt args)))
        new-trace (make-result-trace gf args result)]
    ;; Thesis update weight: non-fresh score (handler :weight) minus the
    ;; recorded old score. Freshly sampled new addresses are drawn from the
    ;; internal proposal and cancel — score-delta would wrongly include them.
    (make-update-result new-trace
      (mx/subtract (:weight result) (:score trace))
      (:discard result) constraints (:choices result) result)))

(defn- run-regen [transition gf _args key {:keys [trace selection]}]
  (let [old-score (:score trace)
        result (rt/run-handler transition
                 {:choices cm/EMPTY :score SCORE-ZERO :weight SCORE-ZERO
                  :key key :selection selection
                  :old-choices (:choices trace)
                  :old-splice-scores (::splice-scores (meta trace))
                  :old-nested-splice-scores (::nested-splice-scores (meta trace))
                  :executor execute-sub :param-store (param-store gf)}
                 (fn [rt] (run-body gf rt (:args trace))))]
    (make-regen-result gf trace result old-score)))

(defn- run-regen-general*
  "Retained-only regenerate via the handler (genmlx-hmch, genmlx-yep2), with the
   GENERAL transition as a parameter so custom (e.g. grammar-masked) regenerate
   moves can route through the same retained-only path. Builds the new trace with
   `general-transition` (selected sites resample, unselected-absent sites sample
   fresh — a structure change instead of the old throw, unselected-present sites
   are retained); the weight is then two project passes over the retained
   selection (new-context minus old-context), exact for dependent joint moves and
   recursing through splices."
  [general-transition gf _args key {:keys [trace selection]}]
  (let [result (rt/run-handler general-transition
                 {:choices cm/EMPTY :score SCORE-ZERO :weight SCORE-ZERO
                  :key key :selection selection
                  :old-choices (:choices trace)
                  :old-splice-scores (::splice-scores (meta trace))
                  :old-nested-splice-scores (::nested-splice-scores (meta trace))
                  :executor execute-sub :param-store (param-store gf)}
                 (fn [rt] (run-body gf rt (:args trace))))]
    (make-regen-result-general gf trace result selection)))

(defn- run-regen-general
  "Retained-only regenerate via the standard general transition."
  [gf args key opts]
  (run-regen-general* h/regenerate-transition-general gf args key opts))

(defn- run-regen-gated
  "Regenerate fast/general gating, parameterized by the fast and general
   transitions: take the fast per-site convention when it is provably equivalent
   (regen-fast-eligible?), else the general retained-only path. The fast path
   keeps the MCMC hot loop (single-site mh-cycle) free of the extra project pass.

   The handler path passes the standard transitions; a custom (with-handler /
   grammar constrain) path passes its own wrapped fast + general transitions so a
   structure-changing constrained move still gets the correct retained-only
   weight instead of the fast (new-score − old-score) − ratio (genmlx-fayo C8)."
  [fast-transition general-transition gf args key {:keys [selection] :as opts}]
  (if (regen-fast-eligible? gf selection)
    (run-regen fast-transition gf args key opts)
    (run-regen-general* general-transition gf args key opts)))

(defn- run-regen-handler
  "Handler-path regenerate dispatcher (standard transitions)."
  [gf args key opts]
  (run-regen-gated h/regenerate-transition h/regenerate-transition-general
                   gf args key opts))

(defn- run-assess [transition gf args key {:keys [constraints]}]
  (let [result (rt/run-handler transition
                 {:choices cm/EMPTY :score SCORE-ZERO :weight SCORE-ZERO
                  :key key :constraints constraints
                  :executor execute-sub-assess :param-store (param-store gf)}
                 (fn [rt] (run-body gf rt args)))]
    {:retval (:retval result) :weight (:score result)}))

(defn- run-project [transition gf _args key {:keys [trace selection]}]
  (let [result (rt/run-handler transition
                 {:choices cm/EMPTY :score SCORE-ZERO :weight SCORE-ZERO
                  :key key :selection selection
                  :old-choices (:choices trace)
                  :constraints cm/EMPTY
                  :executor execute-sub-project :param-store (param-store gf)}
                 (fn [rt] (run-body gf rt (:args trace))))]
    (:weight result)))

(defn- run-propose [transition gf args key _opts]
  (let [result (rt/run-handler transition
                 {:choices cm/EMPTY :score SCORE-ZERO :key key
                  :executor execute-sub :param-store (param-store gf)}
                 (fn [rt] (run-body gf rt args)))]
    {:choices (:choices result)
     :weight (:score result)
     :retval (:retval result)}))

;; -- L1-M2: Compiled path --

(defn- make-compiled-trace
  "Build a :joint-tagged Trace from a compiled-path result. Compiled paths
   are score-equivalent to the handler (joint by construction) and have no
   splices, so the tag is unconditional."
  [gf args result]
  (tr/with-score-type
    (tr/make-trace {:gen-fn gf :args args
                    :choices (cm/from-flat-map (:values result))
                    :retval (:retval result) :score (:score result)})
    :joint))

(defn- run-simulate-compiled [gf args key _opts]
  (let [cfn (:compiled-simulate (:schema gf))]
    (make-compiled-trace gf args (cfn key (vec args)))))

(defn- run-generate-compiled [gf args key {:keys [constraints]}]
  (let [cfn (:compiled-generate (:schema gf))
        result (cfn key (vec args) constraints)]
    {:trace (make-compiled-trace gf args result)
     :weight (:weight result)}))

(defn- run-update-compiled
  "Static models have a fixed address set (no fresh/removed sites), so the
   thesis weight under execution args (x' for update-with-args) reduces to
   score' - score."
  [gf args key {:keys [trace constraints]}]
  (let [cfn (:compiled-update (:schema gf))
        result (cfn key (vec args) constraints (:choices trace))]
    {:trace (make-compiled-trace gf args result)
     :weight (mx/subtract (:score result) (:score trace))
     :discard (cm/from-flat-map (:discard result))}))

(defn- run-regen-compiled [gf _args key {:keys [trace selection]}]
  (let [cfn (:compiled-regenerate (:schema gf))
        old-score (:score trace)
        result (cfn key (vec (:args trace)) (:choices trace) selection)
        weight (mx/subtract (mx/subtract (:score result) old-score) (:weight result))]
    {:trace (make-compiled-trace gf (:args trace) result)
     :weight weight}))

(defn- run-assess-compiled [gf args key {:keys [constraints]}]
  (let [cfn (:compiled-assess (:schema gf))
        r (cfn (vec args) constraints)]
    {:retval (:retval r) :weight (:score r)}))

(defn- run-project-compiled [gf _args key {:keys [trace selection]}]
  (let [cfn (:compiled-project (:schema gf))]
    (cfn (vec (:args trace)) (:choices trace) selection)))

;; -- L1-M3: Prefix compiled path --

(defn- run-simulate-prefix [gf args key _opts]
  (let [pfx (:compiled-prefix (:schema gf))
        result (pfx key (vec args))
        replay (compiled/make-replay-simulate-transition (:values result))
        handler-result (rt/run-handler replay
                         {:choices cm/EMPTY :score (:score result) :key key
                          :executor execute-sub :param-store (param-store gf)}
                         (fn [rt] (run-body gf rt args)))]
    (attach-splice-scores (make-result-trace gf args handler-result) handler-result)))

(defn- run-generate-prefix [gf args key {:keys [constraints]}]
  (let [pfx (:compiled-prefix-generate (:schema gf))
        result (pfx key (vec args) constraints)
        replay (compiled/make-replay-generate-transition (:values result))
        handler-result (rt/run-handler replay
                         {:choices cm/EMPTY :score (:score result)
                          :weight (:weight result) :key key :constraints constraints
                          :executor execute-sub :param-store (param-store gf)}
                         (fn [rt] (run-body gf rt args)))
        trace (make-result-trace gf args handler-result)]
    (make-generate-result trace (:weight handler-result) constraints
                          (:choices handler-result) handler-result)))

(defn- run-update-prefix [gf args key {:keys [trace constraints]}]
  (let [pfx (:compiled-prefix-update (:schema gf))
        result (pfx key (vec args) constraints (:choices trace))
        replay (cops/make-replay-update-transition (:values result))
        ;; Prefix sites never sample fresh (values come from constraints or
        ;; old choices — the static prefix's address set is arg-independent),
        ;; so the prefix score seeds the non-fresh :weight accumulator as
        ;; well as :score.
        handler-result (rt/run-handler replay
                         {:choices cm/EMPTY :score (:score result)
                          :weight (:score result) :key key :constraints constraints
                          :old-choices (:choices trace) :discard (cm/from-flat-map (:discard result))
                          :old-nested-splice-scores (::nested-splice-scores (meta trace))
                          :executor execute-sub :param-store (param-store gf)}
                         (fn [rt] (run-body gf rt args)))
        new-trace (make-result-trace gf args handler-result)]
    (make-update-result new-trace
      (mx/subtract (:weight handler-result) (:score trace))
      (:discard handler-result) constraints (:choices handler-result) handler-result)))

(defn- run-regen-prefix [gf _args key {:keys [trace selection]}]
  (let [pfx (:compiled-prefix-regenerate (:schema gf))
        old-score (:score trace)
        result (pfx key (vec (:args trace)) (:choices trace) selection)
        replay (cops/make-replay-regenerate-transition (:values result))
        handler-result (rt/run-handler replay
                         {:choices cm/EMPTY :score (:score result)
                          :weight (:weight result) :key key :selection selection
                          :old-choices (:choices trace)
                          :old-nested-splice-scores (::nested-splice-scores (meta trace))
                          :executor execute-sub :param-store (param-store gf)}
                         (fn [rt] (run-body gf rt (:args trace))))]
    (make-regen-result gf trace handler-result old-score)))

(defn- run-assess-prefix [gf args key {:keys [constraints]}]
  (let [pfx (:compiled-prefix-assess (:schema gf))
        result (pfx (vec args) constraints)
        replay (cops/make-replay-assess-transition (:values result))
        handler-result (rt/run-handler replay
                         {:choices cm/EMPTY :score (:score result)
                          :weight (:score result) :key key :constraints constraints
                          :executor execute-sub-assess :param-store (param-store gf)}
                         (fn [rt] (run-body gf rt args)))]
    {:retval (:retval handler-result) :weight (:score handler-result)}))

(defn- run-project-prefix [gf args key {:keys [trace selection]}]
  (let [pfx (:compiled-prefix-project (:schema gf))
        result (pfx (vec (:args trace)) (:choices trace) selection)
        replay (cops/make-replay-project-transition (:values result))
        handler-result (rt/run-handler replay
                         {:choices cm/EMPTY :score SCORE-ZERO
                          :weight (:weight result) :key key :selection selection
                          :old-choices (:choices trace) :constraints cm/EMPTY
                          :executor execute-sub-project :param-store (param-store gf)}
                         (fn [rt] (run-body gf rt (:args trace))))]
    (:weight handler-result)))

;; -- L3: Analytical path --

(defn- analytical-fired?
  "Did any analytical handler actually marginalize during this run? When every
   handler fell through (constrained priors, partially-constrained obs,
   probe-declined blocks — genmlx-b470), the result is plain joint scoring and
   must NOT be labeled :marginal."
  [result]
  (boolean (or (seq (:auto-posteriors result))
               (seq (:auto-kalman-beliefs result))
               (seq (:lg-belief result)))))

(defn- run-generate-analytical [gf args key {:keys [constraints]}]
  (let [schema (:schema gf)
        transition (auto/make-address-dispatch
                     h/generate-transition (:auto-handlers schema))
        result (rt/run-handler transition
                 {:choices cm/EMPTY :score SCORE-ZERO :weight SCORE-ZERO
                  :key key :constraints constraints :model-args args
                  :auto-posteriors {} :auto-kalman-beliefs {} :auto-kalman-noise-vars {}
                  :executor execute-sub :param-store (param-store gf)}
                 (fn [rt] (run-body gf rt args)))
        trace (cond-> (make-result-trace gf args result)
                (analytical-fired? result)
                (tr/with-score-type :marginal))]
    (make-generate-result trace (:weight result) constraints (:choices result) result)))

(defn- run-assess-analytical [gf args key {:keys [constraints]}]
  (let [schema (:schema gf)
        transition (auto/make-address-dispatch
                     h/assess-transition (:auto-handlers schema))
        result (rt/run-handler transition
                 {:choices cm/EMPTY :score SCORE-ZERO :weight SCORE-ZERO
                  :key key :constraints constraints :model-args args
                  :auto-posteriors {} :auto-kalman-beliefs {} :auto-kalman-noise-vars {}
                  :executor execute-sub-assess :param-store (param-store gf)}
                 (fn [rt] (run-body gf rt args)))]
    {:retval (:retval result) :weight (:score result)}))

(defn- run-regen-analytical [gf _args key {:keys [trace selection]}]
  (let [schema (:schema gf)
        old-score (:score trace)
        result (rt/run-handler (:auto-regenerate-transition schema)
                 {:choices cm/EMPTY :score SCORE-ZERO :weight SCORE-ZERO
                  :key key :selection selection
                  :old-choices (:choices trace)
                  ;; Linear-Gaussian block regenerate handlers recover the design
                  ;; matrix by probing the obs mean forms against the model args.
                  :model-args (:args trace)
                  :auto-posteriors {} :auto-kalman-beliefs {} :auto-kalman-noise-vars {}
                  :old-splice-scores (::splice-scores (meta trace))
                  :old-nested-splice-scores (::nested-splice-scores (meta trace))
                  :executor execute-sub :param-store (param-store gf)}
                 (fn [rt] (run-body gf rt (:args trace))))
        regen-result (make-regen-result gf trace result old-score)]
    (if (analytical-fired? result)
      (clojure.core/update regen-result :trace tr/with-score-type :marginal)
      regen-result)))

(defn- run-update-analytical
  "Analytical :update for eliminated linear-Gaussian blocks + scalar conjugate
   pairs (genmlx-6hcu). Runs the :auto-update-transition: each eliminated
   structure re-folds its marginal LL under the merged (new-over-old) obs into
   BOTH :score and :weight, sets its latent choices to the NEW posterior mean,
   and charges changed-obs old values into :discard. The thesis update weight
   ((:weight result) − old score) then collapses to the Δ marginal-LL of the
   changed structure(s): untouched blocks/pairs and retained residual sites
   contribute identically to both sides and cancel. The result trace stays
   :marginal. Gated to value-only obs updates (no re-opening latent constraint,
   no Kalman/MVN) by the analytical-dispatcher; otherwise the joint handler path
   runs via joint-rescore-marginal."
  [gf _args key {:keys [trace constraints]}]
  (let [schema (:schema gf)
        result (rt/run-handler (:auto-update-transition schema)
                 {:choices cm/EMPTY :score SCORE-ZERO :weight SCORE-ZERO
                  :key key :constraints constraints
                  :old-choices (:choices trace)
                  ;; LG block update handlers recover the design matrix by probing
                  ;; the obs mean forms against the model args.
                  :model-args (:args trace)
                  :auto-posteriors {} :auto-kalman-beliefs {} :auto-kalman-noise-vars {}
                  :old-splice-scores (::splice-scores (meta trace))
                  :old-nested-splice-scores (::nested-splice-scores (meta trace))
                  :discard cm/EMPTY
                  :executor execute-sub :param-store (param-store gf)}
                 (fn [rt] (run-body gf rt (:args trace))))
        new-trace (make-result-trace gf (:args trace) result)
        update-result (make-update-result new-trace
                        (mx/subtract (:weight result) (:score trace))
                        (:discard result) constraints (:choices result) result)]
    (if (analytical-fired? result)
      (clojure.core/update update-result :trace tr/with-score-type :marginal)
      update-result)))

;; -- Utility --

(defn- add-deleted-to-discard
  "Add old trace addresses not present in new trace to the discard.
   When a model switches branches, addresses visited in the old trace
   but absent in the new trace must appear in the discard (Gen.jl semantics)."
  [discard old-choices new-choices]
  (if (and (instance? cm/Node old-choices)
           (instance? cm/Node new-choices))
    (let [old-keys (set (keys (:m old-choices)))
          new-keys (set (keys (:m new-choices)))
          deleted (set/difference old-keys new-keys)]
      (reduce (fn [d addr]
                (let [old-sub (get (:m old-choices) addr)]
                  (if (cm/has-value? old-sub)
                    (cm/set-value d addr (cm/get-value old-sub))
                    (cm/set-submap d addr old-sub))))
              discard deleted))
    discard))

;; ---------------------------------------------------------------------------
;; Dispatchers — lookup tables mapping op → run-fn
;; ---------------------------------------------------------------------------

;; Transition-parameterized table: maps op to the generic run-* function.
;; Used by handler-dispatcher (with standard transitions) and
;; custom-transition-dispatcher (with the user-supplied transition).
(def ^:private transition-run-fns
  {:simulate run-simulate, :generate run-generate
   :update run-update,     :regenerate run-regen
   :assess run-assess,     :project run-project
   :propose run-propose})

;; Standard handler transitions per op.
(def ^:private standard-transitions
  {:simulate h/simulate-transition, :generate h/generate-transition
   :update h/update-transition,     :regenerate h/regenerate-transition
   :assess h/assess-transition,     :project h/project-transition
   :propose h/simulate-transition})

;; Handler table: each entry is (partial run-* standard-transition).
;; :regenerate is overridden with the fast/general gating dispatcher
;; (run-regen-handler) — the retained-only general path is selected when the
;; fast per-site convention is not provably equivalent (genmlx-hmch/yep2).
;; transition-run-fns keeps the plain run-regen for custom with-handler use.
(def ^:private handler-table
  (assoc (into {} (map (fn [[op run-fn]]
                         [op (partial run-fn (get standard-transitions op))]))
                 transition-run-fns)
         :regenerate run-regen-handler))

(def ^:private compiled-table
  {:simulate run-simulate-compiled, :generate run-generate-compiled
   :update run-update-compiled,     :regenerate run-regen-compiled
   :assess run-assess-compiled,     :project run-project-compiled})

(def ^:private prefix-table
  {:simulate run-simulate-prefix, :generate run-generate-prefix
   :update run-update-prefix,     :regenerate run-regen-prefix
   :assess run-assess-prefix,     :project run-project-prefix})

(def ^:private analytical-table
  {:generate run-generate-analytical
   :assess   run-assess-analytical
   :regenerate run-regen-analytical
   :update   run-update-analytical})

(def ^:private compiled-keys
  {:simulate :compiled-simulate,   :generate :compiled-generate
   :update :compiled-update,       :regenerate :compiled-regenerate
   :assess :compiled-assess,       :project :compiled-project})

(def ^:private prefix-keys
  {:simulate :compiled-prefix,              :generate :compiled-prefix-generate
   :update :compiled-prefix-update,         :regenerate :compiled-prefix-regenerate
   :assess :compiled-prefix-assess,         :project :compiled-prefix-project})

(def ^:private custom-dispatcher
  "Handles both with-handler (transition substitution) and with-dispatch
   (full dispatch override). Checks ::custom-dispatch first, then
   ::custom-transition."
  (reify dispatch/IDispatcher
    (resolve-transition [_ op _schema opts]
      (let [gf-meta (meta (:gf opts))]
        (cond
          ;; Full dispatch override: (fn [op gf args key opts] -> result)
          (::dispatch/custom-dispatch gf-meta)
          (let [df (::dispatch/custom-dispatch gf-meta)]
            {:run (fn [gf args key opts] (df op gf args key opts))
             :score-type (or (:score-type (meta df)) :joint)
             :label :custom})

          ;; Transition substitution: a single (fn [state addr dist]) used for
          ;; every op, or a per-op map {op -> transition} falling back to the
          ;; standard transition for omitted ops (genmlx-xwxh).
          (::dispatch/custom-transition gf-meta)
          (let [t  (::dispatch/custom-transition gf-meta)
                tf (if (map? t)
                     (or (get t op) (get standard-transitions op))
                     t)]
            {:run (if (and (= op :regenerate) (map? t) (:regenerate-general t))
                    ;; Custom (e.g. grammar-masked) regenerate ALWAYS takes the
                    ;; retained-only general path. The fast per-site weight
                    ;; (new-score − old-score) − ratio is only valid when the
                    ;; selected and retained sites are independent — but masking
                    ;; can couple them in ways the schema cannot see (a grammar
                    ;; where a later token's valid set depends on an earlier one,
                    ;; or a structure-changing move that shifts EOS and the number
                    ;; of sites). The general path's two project passes are
                    ;; themselves grammar-masked, so they capture that coupling
                    ;; exactly, and reduce to the fast result when sites really are
                    ;; independent (genmlx-fayo C8).
                    (partial run-regen-general* (:regenerate-general t))
                    (partial (get transition-run-fns op) tf))
             :score-type (or (:score-type (meta (if (map? t) tf t))) :joint)
             :label :custom}))))))

(defn- regen-reopens-analytical?
  "True when the regenerate selection selects any address that carries an analytical
   regenerate handler — an eliminated conjugate prior, a linear-Gaussian block
   latent/obs, or a Kalman state/obs. Selecting such an address RE-OPENS the
   marginalisation: the analytical handler declines and the site is scored JOINTLY
   this pass (genmlx-b470's block-reopened? / Case-A fallthrough, and the scalar
   conjugate \"prior selected → nil\" path). The incoming trace is :marginal, so the
   fast (new-score − old-score) difference in make-regen-result would subtract a
   marginal old score from a now-joint new score — not a valid MH weight
   (genmlx-wl1y). When this holds the analytical regenerate declines; the joint
   handler path then runs through joint-rescore-marginal, which converts the old
   trace to a joint score so BOTH sides share one decomposition. When it does NOT
   hold (e.g. an MH move over an unrelated residual) the block re-marginalises
   identically to the old trace, so the fast analytical path stays exact and
   Rao-Blackwellised (genmlx-m3tn / genmlx-4q9d)."
  [schema selection]
  (boolean
    (when-let [handlers (:auto-regenerate-handlers schema)]
      (and selection
           (some (fn [addr] (sel/selected? selection addr)) (keys handlers))))))

(defn- update-reopens-analytical?
  "True when the update CONSTRAINTS pin an eliminated LATENT (a conjugate prior
   or linear-Gaussian block latent) or a block noise latent (genmlx-4q9d).
   Pinning a latent re-opens its marginalisation (it must be scored jointly under
   the pinned value); changing a block's noise latent alters every obs variance,
   breaking the closed-form Δ marginal-LL. In either case the analytical update
   declines so the joint handler path + joint-rescore-marginal runs (correct, no
   Rao-Blackwell). Constraining only block OBS is the normal value-update case and
   does NOT re-open (genmlx-6hcu)."
  [schema constraints]
  (let [elim  (get-in schema [:analytical-plan :rewrite-result :eliminated])
        noise (into #{} (mapcat :noise-latents (:linear-gaussian-blocks schema)))
        reopen (into (set elim) noise)]
    (boolean
      (and (seq reopen)
           (some (fn [addr] (cm/has-value? (cm/get-submap constraints addr))) reopen)))))

(def ^:private analytical-dispatcher
  (reify dispatch/IDispatcher
    (resolve-transition [_ op schema opts]
      (when-let [run-fn (get analytical-table op)]
        (case op
          (:generate :assess)
          ;; seq, not truthiness: an empty handler map (all rules declined,
          ;; e.g. family without a runtime factory) must not claim the
          ;; analytical path (genmlx-b470).
          (when (and (not (mx/in-grad?))
                     (seq (:auto-handlers schema))
                     (auto/some-conjugate-obs-constrained?
                       (:conjugate-pairs schema) (:constraints opts)))
            {:run run-fn :score-type :marginal :label :analytical})

          :regenerate
          ;; The marginal old-score is only a valid subtrahend when THIS pass also
          ;; scores every eliminated structure marginally. A selection that re-opens
          ;; a block (genmlx-wl1y) flips it to joint scoring this pass — decline so
          ;; the joint handler path + joint-rescore-marginal differences two joint
          ;; scores consistently. Stable (no-reopen) moves keep the fast path.
          (when (and (:auto-regenerate-transition schema)
                     (= :marginal (tr/score-type (:trace opts)))
                     (not (regen-reopens-analytical? schema (:selection opts)))
                     ;; The analytical fast per-site weight
                     ;; (new-score − old-score − proposal-ratio) is exact ONLY
                     ;; for fast-eligible selections; for interdependent residual
                     ;; selections it mis-weights, and the compiled dispatcher's
                     ;; fast-eligible guard sits AFTER this one in the stack — so
                     ;; gate here too, declining to the handler general
                     ;; retained-only path otherwise (genmlx-9yuw).
                     (:selection opts)
                     (regen-fast-eligible? (:gf opts) (:selection opts)))
            {:run run-fn :score-type :marginal :label :analytical})

          :update
          ;; First analytical :update path (genmlx-6hcu). Same decomposition-
          ;; consistency rule as regenerate: the marginal old-score is a valid
          ;; subtrahend only when THIS pass also scores every eliminated structure
          ;; marginally. :auto-update-transition is built ONLY for fully-supported
          ;; models (scalar conjugate + LG blocks, no Kalman/MVN); a constraint that
          ;; pins an eliminated latent re-opens it (decline). Otherwise → joint path.
          ;; PLAIN update only: update-with-args (op :update + :argdiffs in opts,
          ;; genmlx-s8e8) re-executes under NEW args x' — a changed design X breaks
          ;; the closed-form Δ marginal-LL — so it keeps its joint-conversion path.
          (when (and (:auto-update-transition schema)
                     (not (contains? opts :argdiffs))
                     (= :marginal (tr/score-type (:trace opts)))
                     (not (update-reopens-analytical? schema (:constraints opts))))
            {:run run-fn :score-type :marginal :label :analytical})

          nil)))))

(def ^:private compiled-dispatcher
  (reify dispatch/IDispatcher
    (resolve-transition [_ op schema opts]
      (cond
        ;; Non-fast-eligible regenerate (dependent joint moves / structure
        ;; change) must take the handler general retained-only path: the
        ;; compiled, prefix, and branch-rewrite per-site regen conventions are
        ;; exact ONLY for fast-eligible selections (genmlx-hmch / genmlx-yep2).
        ;; Returning nil falls through to the handler-dispatcher, whose
        ;; run-regen-handler then takes the general path. The analytical
        ;; dispatcher sits EARLIER in the stack, so conjugate-posterior
        ;; regenerate is never bypassed by this.
        (and (= op :regenerate)
             (:selection opts)               ; absent for introspective resolve (inspect)
             (not (regen-fast-eligible? (:gf opts) (:selection opts))))
        nil

        (get schema (get compiled-keys op))
        {:run (get compiled-table op) :score-type :joint :label :compiled}

        (get schema (get prefix-keys op))
        {:run (get prefix-table op) :score-type :joint :label :prefix}))))

(def ^:private handler-dispatcher
  (reify dispatch/IDispatcher
    (resolve-transition [_ op _ _]
      {:run (get handler-table op) :score-type :joint :label :handler})))

(def ^:private default-dispatcher-stack
  [custom-dispatcher
   analytical-dispatcher
   compiled-dispatcher
   handler-dispatcher])

(declare run-dispatched* strip-alternate-paths)

(defn- joint-rescore-marginal
  "Score-type boundary check for joint-scoring update/project/regenerate
   (ARCHITECTURE §3.3, genmlx-pkmx, genmlx-lbae). Consuming a non-joint
   trace would subtract a marginal old score from a joint new score —
   silently mixing decompositions.

   :marginal traces CONVERT: alternate paths must stay semantically
   invisible (a model written at L0 runs unchanged at L3), so re-generate
   the trace fully constrained from its own choices via the handler path,
   yielding an identical-choices trace with an exact joint score. Costs one
   handler generate per conversion.

   :collapsed (and any other non-joint) traces THROW: their choicemaps are
   empty (all latents integrated out by enumerate/exact), so there is
   nothing to re-generate from — no conversion exists."
  [gf op key opts]
  (let [t (:trace opts)
        st (tr/score-type t)]
    (case st
      :joint opts
      :marginal
      (let [stripped (strip-alternate-paths gf)
            res (run-dispatched* stripped :generate (:args t) key
                                 {:constraints (:choices t)})
            converted (:trace res)]
        ;; Convergence guard: when a sub-gf reproduces a non-joint score
        ;; even fully constrained (e.g. an enumerate splice), no joint
        ;; conversion exists — throw, don't pass a still-mixed trace on.
        (when (not= :joint (tr/score-type converted))
          (throw (ex-info
                   (str "Joint-scoring " op ": re-generating the trace's"
                        " choices did not yield a joint score (a sub-gf"
                        " reproduces a " (tr/score-type converted) " score)")
                   {:genmlx/error :score-type-mismatch
                    :op op :score-type (tr/score-type converted)
                    :expected :joint})))
        (assoc opts :trace converted))
      :placeholder
      ;; genmlx-b2mj: the trace's :score is a 0.0 placeholder (e.g. compiled-SMC
      ;; particle values), not a real joint density. Differencing against it
      ;; would silently produce a wrong weight — reject instead.
      (throw (ex-info
               (str "Joint-scoring " op " cannot consume a :placeholder-scored"
                    " trace — its :score is a 0.0 placeholder (e.g. compiled-SMC"
                    " particle values, genmlx-b2mj), so the weight would be"
                    " wrong. Re-score the choices via p/generate first, or use"
                    " the trace for choice extraction only.")
               {:genmlx/error :placeholder-score
                :op op :score-type st :expected :joint}))
      (throw (ex-info
               (str "Joint-scoring " op " cannot consume a " st
                    "-scored trace — its choices do not determine a joint"
                    " density (collapsed traces have no recorded choices)")
               {:genmlx/error :score-type-mismatch
                :op op :score-type st :expected :joint})))))

;; strip-analytical-path is defined further down; run-dispatched*'s
;; analytical-bail fallback (genmlx-0e0j) needs a forward reference.
(declare strip-analytical-path)

(defn run-dispatched*
  "Core dispatch: walk the dispatcher stack and execute the first match.
   Public so genmlx.dev can reference it for start!/stop!."
  [gf op args key opts]
  (let [spec (dispatch/resolve default-dispatcher-stack op (:schema gf)
               (assoc opts :gf gf))]
    (assert spec (str "No dispatcher resolved for op " op))
    (let [opts* (if (and (contains? #{:update :project :regenerate} op)
                         (= :joint (:score-type spec)))
                  (joint-rescore-marginal gf op key opts)
                  opts)]
      (try
        ((:run spec) gf args key opts*)
        (catch :default e
          (if (:genmlx.analytical/bail (ex-data e))
            ;; An analytical handler discovered mid-run that its elimination is
            ;; invalid for THIS execution (e.g. an ill-conditioned MVN/Kalman
            ;; obs update reached AFTER its latent was already marginalized).
            ;; Re-run the whole op on the analytical-stripped gf so prior+obs
            ;; take the coherent handler JOINT path, instead of a hybrid weight
            ;; (prior marginalized to 0, obs scored at the prior-mean point
            ;; estimate). genmlx-0e0j.
            (run-dispatched* (strip-analytical-path gf) op args key opts)
            (throw e)))))))

;; Dispatch function atom. Defaults to run-dispatched*.
;; genmlx.dev/start! swaps this with a validating wrapper.
(defonce dispatch-fn (atom run-dispatched*))

(defn resolve-dispatch
  "Resolve the dispatch spec for a GFI operation on gf.
   Returns {:run fn :score-type kw :label kw}. Used by inspect."
  [gf op]
  (dispatch/resolve default-dispatcher-stack op (:schema gf) {:gf gf}))

;; ---------------------------------------------------------------------------
;; DynamicGF record — GFI protocol implementations
;; ---------------------------------------------------------------------------

(defrecord DynamicGF [body-fn source schema]
  p/IGenerativeFunction
  (simulate [this args]
    (let [key (ensure-key this)

          result (@dispatch-fn this :simulate args key {})]
      (mx/gfi-cleanup!)
      result))

  p/IGenerate
  (generate [this args constraints]
    (let [key (ensure-key this)

          result (@dispatch-fn this :generate args key {:constraints constraints})]
      (mx/gfi-cleanup!)
      result))

  p/IUpdate
  (update [this trace constraints]
    (let [key (ensure-key this)

          result (@dispatch-fn this :update (:args trace) key
                   {:trace trace :constraints constraints})]
      (mx/gfi-cleanup!)
      (clojure.core/update result :discard
        add-deleted-to-discard (:choices trace) (:choices (:trace result)))))

  p/IRegenerate
  (regenerate [this trace selection]
    (let [key (ensure-key this)

          result (@dispatch-fn this :regenerate (:args trace) key
                   {:trace trace :selection selection})]
      (mx/gfi-cleanup!)
      result))

  p/IAssess
  (assess [this args choices]
    (let [key (ensure-key this)

          result (@dispatch-fn this :assess args key {:constraints choices})]
      (mx/gfi-cleanup!)
      result))

  p/IPropose
  (propose [this args]
    (let [key (ensure-key this)

          result (@dispatch-fn this :propose args key {})]
      (mx/gfi-cleanup!)
      result))

  p/IProject
  (project [this trace selection]
    (let [key (ensure-key this)

          result (@dispatch-fn this :project (:args trace) key
                   {:trace trace :selection selection})]
      (mx/gfi-cleanup!)
      result)))

(def analytical-path-schema-keys
  "Every schema key the dispatcher stack consults for the L3 analytical
   path. Stripping these forces stochastic prior sampling and joint scores
   while keeping the L1 compiled paths (which are score-equivalent to the
   handler). Sampling-based methods need this: analytical generate pins
   eliminated latents at their deterministic posterior MEAN, and
   :auto-regenerate-transition intercepts regenerate — correct for marginal
   likelihoods, corrupting for particle diversity (smc) and trace-MH chains
   (genmlx-540f)."
  [:auto-handlers :conjugate-pairs :has-conjugate? :analytical-plan
   :auto-regenerate-transition :auto-regenerate-handlers
   :auto-update-transition :auto-update-handlers])

(def alternate-path-schema-keys
  "Every schema key the dispatcher stack consults for a non-handler execution
   path: L1-M2 full-compile keys, L1-M3 prefix keys, and the L3 analytical
   keys. strip-alternate-paths must remove ALL of them — leaving any behind
   lets a 'handler ground truth' comparison silently exercise a compiled or
   analytical path (genmlx-pkmx)."
  (into [:compiled-simulate :compiled-generate :compiled-update :compiled-assess
         :compiled-project :compiled-regenerate
         :compiled-prefix :compiled-prefix-generate :compiled-prefix-update
         :compiled-prefix-regenerate :compiled-prefix-assess :compiled-prefix-project]
        analytical-path-schema-keys))

(defn strip-alternate-paths
  "Return a copy of gf with all alternate execution paths removed from its
   schema — full-compile, prefix, and analytical — forcing the handler
   (interpreter) path for all GFI ops. Preserves the gen-fn's metadata (the
   PRNG ::key from with-key/auto-key lives there — genmlx-3lgy).
   Returns gf unchanged if it has no schema."
  [gf]
  (if-let [schema (:schema gf)]
    (with-meta
      (->DynamicGF (:body-fn gf) (:source gf)
                   (apply dissoc schema alternate-path-schema-keys))
      (meta gf))
    gf))

(defn strip-analytical-path
  "Return a copy of gf with ONLY the L3 analytical path removed from its
   schema, keeping the L1 compiled paths. The canonical strip for every
   sampling-based method: particle methods (smc, csmc, smcp3, importance)
   and trace-MH (kern/mh-kernel, mcmc/mh) — analytical generate returns
   eliminated latents at their deterministic posterior mean and intercepts
   regenerate, so unstripped chains/particles sample a corrupted posterior
   (genmlx-540f; chi2 290.9 vs crit 21.67 on two-gaussians mh-cycle).
   assoc-based so any GFI record with a :schema (DynamicGF, combinators)
   keeps its type and metadata (the PRNG ::key). Unchanged if no schema."
  [gf]
  (if-let [schema (:schema gf)]
    (assoc gf :schema (apply dissoc schema analytical-path-schema-keys))
    gf))

(defn- propagate-meta
  "Propagate the PRNG key and param-store to a sub-gf via metadata, when present."
  [gf key param-store]
  (cond-> gf
    key (vary-meta assoc ::key key)
    param-store (vary-meta assoc ::param-store param-store)))

(defn- extract-splice-meta
  "Extract splice metadata from a trace result into a flat result map."
  [result trace]
  (let [m (meta trace)]
    (cond-> result
      (::splice-scores m)       (assoc :splice-scores (::splice-scores m))
      (::nested-splice-scores m) (assoc :nested-splice-scores (::nested-splice-scores m)))))

(defn- attach-old-splice-meta
  "Attach old splice metadata to a reconstructed trace for regenerate/update."
  [trace old-splice-scores old-nested-splice-scores]
  (if (or old-splice-scores old-nested-splice-scores)
    (with-meta trace (cond-> {}
                       old-splice-scores (assoc ::splice-scores old-splice-scores)
                       old-nested-splice-scores (assoc ::nested-splice-scores old-nested-splice-scores)))
    trace))

(defn- execute-sub
  "Execute a sub-generative-function during handler execution.
   Delegates to the sub-gf's own GFI methods.
   Propagates param-store and key to sub-gfs via metadata."
  [gf args {:keys [constraints old-choices selection key old-splice-score
                    old-sub-splice-scores old-sub-nested-splice-scores param-store]}]
  (let [gf (propagate-meta gf key param-store)
        [trace result] (cond
                         ;; Regenerate mode
                         selection
                         (let [old-sub-score (or old-splice-score SCORE-ZERO)
                               old-trace (-> (tr/make-trace {:gen-fn gf :args args
                                                             :choices (or old-choices cm/EMPTY)
                                                             :retval nil :score old-sub-score})
                                             (attach-old-splice-meta old-sub-splice-scores
                                                                     old-sub-nested-splice-scores))
                               {:keys [trace weight]} (p/regenerate gf old-trace selection)]
                           ;; The child returns its final MH weight
                           ;; w = ΔS_child - pr_child, but the parent regenerate
                           ;; state accumulates proposal ratios (make-regen-result
                           ;; computes W = ΔS_total - accumulator). Convert back:
                           ;; pr_child = ΔS_child - w. The constructed old score
                           ;; cancels, so a missing old-splice-score stays exact.
                           [trace {:choices (:choices trace) :retval (:retval trace)
                                   :score (:score trace)
                                   :weight (mx/subtract
                                            (mx/subtract (:score trace) old-sub-score)
                                            weight)}])

                         ;; Update mode: has old-choices (possibly with new constraints)
                         (and old-choices (not= old-choices cm/EMPTY))
                         (let [old-sub-score (or old-splice-score SCORE-ZERO)
                               old-trace (-> (tr/make-trace {:gen-fn gf :args args
                                                             :choices old-choices
                                                             :retval nil :score old-sub-score})
                                             (attach-old-splice-meta old-sub-splice-scores
                                                                     old-sub-nested-splice-scores))
                               {:keys [trace weight discard]} (p/update gf old-trace
                                                                        (or constraints cm/EMPTY))]
                           ;; The child returns its thesis update weight
                           ;; w = nonfresh_child - old_child, but the parent's
                           ;; update accumulator holds non-fresh scores (the
                           ;; final weight subtracts the parent's recorded old
                           ;; score, which already includes the child's old
                           ;; score). Convert back: nonfresh_child = w + old.
                           ;; The constructed old score cancels, so a missing
                           ;; old-splice-score stays exact.
                           [trace {:choices (:choices trace) :retval (:retval trace)
                                   :score (:score trace)
                                   :weight (mx/add weight old-sub-score)
                                   :discard discard}])

                         ;; Generate with constraints
                         (and constraints (not= constraints cm/EMPTY))
                         (let [{:keys [trace weight]} (p/generate gf args constraints)]
                           [trace {:choices (:choices trace) :retval (:retval trace)
                                   :score (:score trace) :weight weight}])

                         ;; Plain simulate
                         :else
                         (let [trace (p/simulate gf args)]
                           [trace {:choices (:choices trace) :retval (:retval trace)
                                   :score (:score trace)}]))]
    ;; Propagate the sub-trace's score-type: a marginal-scored sub-result
    ;; must not launder into a joint-looking parent score (genmlx-lbae,
    ;; ARCHITECTURE §3.3 merge-sub-result).
    (extract-splice-meta (assoc result :score-type (tr/score-type trace)) trace)))

(defn- execute-sub-project
  "Execute sub-GF in project mode: replay via generate, then project.
   Propagates param-store and key via metadata."
  [gf args {:keys [old-choices selection key param-store]}]
  (let [gf (propagate-meta gf key param-store)
        {:keys [trace]} (p/generate gf args (or old-choices cm/EMPTY))
        weight (p/project gf trace (or selection sel/none))]
    {:choices (:choices trace)
     :retval (:retval trace)
     :score (:score trace)
     :score-type (tr/score-type trace)
     :weight weight}))

(defn- execute-sub-assess
  "Execute a sub-GF in assess mode: all choices must be provided.
   Propagates param-store and key via metadata."
  [gf args {:keys [constraints key param-store]}]
  (let [gf (propagate-meta gf key param-store)
        {:keys [retval weight]} (p/assess gf args (or constraints cm/EMPTY))]
    {:choices (or constraints cm/EMPTY) :retval retval
     :score weight :weight weight}))

(defn- attach-compiled-ops
  "Try all compiled operations for a model type. Each [schema-key builder-fn]
   pair calls (builder-fn schema source); non-nil results are assoc'd onto schema."
  [schema source ops]
  (reduce (fn [s [k builder]]
            (if-let [result (builder schema source)]
              (assoc s k result)
              s))
          schema ops))

(def ^:private static-ops
  [[:compiled-simulate compiled/make-compiled-simulate]
   [:compiled-generate cops/make-compiled-generate]
   [:compiled-update cops/make-compiled-update]
   [:compiled-assess cops/make-compiled-assess]
   [:compiled-project cops/make-compiled-project]
   [:compiled-regenerate cops/make-compiled-regenerate]])

(def ^:private branch-ops
  [[:compiled-simulate compiled/make-branch-rewritten-simulate]
   [:compiled-generate cops/make-branch-rewritten-generate]
   [:compiled-update cops/make-branch-rewritten-update]
   [:compiled-assess cops/make-branch-rewritten-assess]
   [:compiled-project cops/make-branch-rewritten-project]
   [:compiled-regenerate cops/make-branch-rewritten-regenerate]])

(def ^:private prefix-ops
  [[:compiled-prefix compiled/make-compiled-prefix :compiled-prefix-addrs]
   [:compiled-prefix-generate cops/make-compiled-prefix-generate nil]
   [:compiled-prefix-update cops/make-compiled-prefix-update nil]
   [:compiled-prefix-assess cops/make-compiled-prefix-assess nil]
   [:compiled-prefix-project cops/make-compiled-prefix-project nil]
   [:compiled-prefix-regenerate cops/make-compiled-prefix-regenerate nil]])

(defn- attach-prefix-ops
  "Try all prefix-compiled operations. Each entry is [fn-key builder addrs-key].
   Builder returns {:fn compiled-fn :addrs addr-vec} or nil.
   The :fn is stored under fn-key; :addrs under addrs-key when non-nil."
  [schema source]
  (reduce (fn [s [fn-key builder addrs-key]]
            (if-let [result (builder schema source)]
              (cond-> (assoc s fn-key (:fn result))
                addrs-key (assoc addrs-key (:addrs result)))
              s))
          schema prefix-ops))

(defn make-gen-fn
  "Create a DynamicGF from a body function and its source form.
   Extracts schema from the source form for Level 1 compilation.
   For static models, attempts L1-M2 (full compiled simulate).
   For branch models, attempts L1-M4 (branch rewriting).
   For non-static models, attempts L1-M3 (compiled prefix)."
  [body-fn source]
  (let [schema (schema/extract-schema source)
        schema (cond
                 ;; Hidden trace sites (trace/splice handed to opaque code):
                 ;; not statically analyzable — handler path only, no compilation.
                 (and schema (:opaque-gen-escape? schema))
                 schema

                 (and schema (:static? schema))
                 (attach-compiled-ops schema source static-ops)

                 (and schema (:has-branches? schema))
                 (if-let [csim (compiled/make-branch-rewritten-simulate schema source)]
                   (attach-compiled-ops (assoc schema :compiled-simulate csim)
                                        source (rest branch-ops))
                   (attach-prefix-ops schema source))

                 schema
                 (attach-prefix-ops schema source)

                 :else schema)
        ;; L3: full rewrite engine (Kalman > Conjugacy > RaoBlackwell).
        ;; Skip for opaque-escape bodies: conjugacy detection over only the
        ;; visible sites could mis-fire while real obs sites are hidden.
        schema (if (and schema (not (:opaque-gen-escape? schema)))
                 (let [augmented (conj/augment-schema-with-conjugacy schema)]
                   (if (:has-conjugate? augmented)
                     (let [plan (rewrite/build-analytical-plan augmented source)
                           ;; Keep both declined concern-components AND linear-Gaussian
                           ;; block addrs off the SCALAR regenerate path: declined addrs
                           ;; fall through to base regenerate (sampled), and block addrs
                           ;; would be mis-handled by the per-prior scalar conjugate path
                           ;; (the off-by-affine-coefficient bug). Blocks instead get
                           ;; dedicated joint regenerate handlers merged in below
                           ;; (genmlx-m3tn: Rao-Blackwell under MH moves).
                           lg-blocks (:lg-blocks plan)
                           lg-excluded (into (set (:declined-addrs plan))
                                             (mapcat :all-addrs lg-blocks))
                           regen-pairs (if (seq lg-excluded)
                                         (remove (fn [p]
                                                   (or (contains? lg-excluded (:prior-addr p))
                                                       (contains? lg-excluded (:obs-addr p))))
                                                 (:conjugate-pairs augmented))
                                         (:conjugate-pairs augmented))
                           scalar-regen-handlers (auto/build-all-regenerate-handlers
                                                  regen-pairs
                                                  :chains (:kalman-chains plan))
                           block-regen-handlers (reduce
                                                 (fn [m blk]
                                                   (merge m (lg/make-lg-handlers blk :regenerate)))
                                                 {} lg-blocks)
                           regen-handlers (merge scalar-regen-handlers block-regen-handlers)
                           ;; Opt 1: precompute dispatch transition once at construction
                           regen-transition (when (seq regen-handlers)
                                              (auto/make-address-dispatch
                                               h/regenerate-transition regen-handlers))
                           ;; Analytical UPDATE handlers (genmlx-6hcu): scalar
                           ;; conjugate pairs + LG blocks. MVN, Kalman, and
                           ;; Dirichlet–Categorical (genmlx-cf0d) analytical update
                           ;; are unimplemented, so a model containing ANY of them
                           ;; gets NO :auto-update-transition and its update falls
                           ;; to the joint handler path (correct, just no
                           ;; Rao-Blackwell) — never a mixed marginal/joint score
                           ;; (cf genmlx-wl1y). Declining the WHOLE update (not just
                           ;; the unsupported pair) is what prevents the mix.
                           update-supported? (and (empty? (:kalman-chains plan))
                                                  (not-any? #(#{:mvn-normal :dirichlet-categorical}
                                                              (:family %))
                                                            regen-pairs))
                           scalar-update-handlers (if update-supported?
                                                    (auto/build-update-handlers regen-pairs)
                                                    {})
                           block-update-handlers (reduce
                                                  (fn [m blk]
                                                    (merge m (lg/make-lg-handlers blk :update)))
                                                  {} lg-blocks)
                           update-handlers (merge scalar-update-handlers block-update-handlers)
                           update-transition (when (and update-supported? (seq update-handlers))
                                               (auto/make-address-dispatch
                                                h/update-transition update-handlers))]
                       (-> augmented
                           (assoc :auto-handlers (get-in plan [:rewrite-result :handlers]))
                           (assoc :auto-regenerate-handlers regen-handlers)
                           (assoc :auto-regenerate-transition regen-transition)
                           (assoc :auto-update-handlers (when update-transition update-handlers))
                           (assoc :auto-update-transition update-transition)
                           (assoc :linear-gaussian-blocks (:lg-blocks plan))
                           (assoc :analytical-plan plan)))
                     augmented))
                 schema)]
    (->DynamicGF body-fn source schema)))

(defn with-key
  "Return a copy of gf with the given PRNG key for reproducible execution.
   The key is stored as metadata and read by DynamicGF GFI methods."
  [gf key]
  (vary-meta gf assoc ::key key))

(defn auto-key
  "Mark a gen-fn to auto-generate fresh PRNG keys for each GFI call.
   For REPL, tests, and interactive use. Each call to simulate/generate/etc.
   gets a fresh key, so repeated calls produce different results.
   Inference entry points manage keys automatically — use this only
   when calling GFI methods directly."
  [gf]
  (vary-meta gf assoc ::key auto-key-sentinel))

(defn call
  "Call a generative function as a regular function (simulate and return value).
   Auto-keys the gen-fn for convenience."
  [gf & args]
  (:retval (p/simulate (auto-key gf) (vec args))))

;; ---------------------------------------------------------------------------
;; Vectorized execution (batched: N particles in one model run)
;; These bypass the dispatcher stack by design — vectorized execution always
;; uses batched handler transitions with [N]-shaped arrays via MLX broadcasting.
;; ---------------------------------------------------------------------------

(defn- tag-vtrace
  "Tag a VectorizedTrace :joint. Batched execution always runs batched
   handler transitions — no analytical or collapsed producer exists for
   vectorized traces — so the tag is unconditional (genmlx-lbae)."
  [vt]
  (tr/with-score-type vt :joint))

(defn vsimulate
  "Run model body ONCE with batched handler, producing a VectorizedTrace
   with [n]-shaped arrays at each choice site.
   gf: DynamicGF, args: model args, n: number of particles, key: PRNG key."
  [gf args n key]
  (let [key (rng/ensure-key key)

        result (rt/run-handler h/batched-simulate-transition
                               ;; [N]-shaped init score so the VectorizedTrace
                               ;; :score is [N] even when every site is
                               ;; constrained/deterministic and no [N] sample
                               ;; establishes the batch axis (genmlx-fgb6/x93e).
                               {:choices cm/EMPTY :score (mx/zeros [n])
                                :key key :batch-size n :batched? true
                                :executor execute-sub
                                :param-store (param-store gf)}
                               (fn [rt] (run-body gf rt args)))]
    (tag-vtrace
      (vec/->VectorizedTrace gf args (:choices result) (:score result)
                             (mx/zeros [n]) n (:retval result)))))

(defn vgenerate
  "Run model body ONCE with batched generate handler, producing a
   VectorizedTrace. Constrained sites use scalar observations;
   unconstrained sites produce [n]-shaped samples.
   gf: DynamicGF, args: model args, constraints: ChoiceMap,
   n: number of particles, key: PRNG key."
  [gf args constraints n key]
  (let [key (rng/ensure-key key)

        result (rt/run-handler h/batched-generate-transition
                               ;; [N]-shaped init score/weight: a single-site
                               ;; fully-observed model has no [N] sample to
                               ;; establish the batch axis, so a scalar init
                               ;; would leave :score/:weight shape [] instead of
                               ;; [N] (genmlx-fgb6/x93e/5nch/v4mz).
                               {:choices cm/EMPTY :score (mx/zeros [n])
                                :weight (mx/zeros [n]) :key key
                                :constraints constraints :batch-size n :batched? true
                                :executor execute-sub
                                :param-store (param-store gf)}
                               (fn [rt] (run-body gf rt args)))]
    (tag-vtrace
      (vec/->VectorizedTrace gf args (:choices result) (:score result)
                             (:weight result) n (:retval result)))))

(defn- vupdate*
  "Shared batched-update core: run the model body ONCE with the batched
   update handler under exec-args ((:args vtrace) for vupdate, the thesis
   x' for vupdate-args) and stamp the result vtrace with them."
  [gf vtrace exec-args constraints key]
  (let [key (rng/ensure-key key)

        n (:n-particles vtrace)
        result (rt/run-handler h/batched-update-transition
                               ;; [N]-shaped init: keeps :score/:weight [N] when
                               ;; every site is constrained (genmlx-x93e).
                               {:choices cm/EMPTY :score (mx/zeros [n])
                                :weight (mx/zeros [n]) :key key
                                :constraints constraints
                                :old-choices (:choices vtrace)
                                :discard cm/EMPTY
                                :batch-size n :batched? true
                                :executor execute-sub
                                :param-store (param-store gf)}
                               (fn [rt] (run-body gf rt exec-args)))
        ;; Thesis update weight: non-fresh score minus old [N]-shaped score —
        ;; the same convention as the scalar run-update path.
        weight (mx/subtract (:weight result) (:score vtrace))
        ;; On a structure-shrinking move (e.g. a host-arg branch flip that visits
        ;; :b under new args, deleting :a), the batched-update-transition writes
        ;; :discard only for OVERWRITTEN choices, never the un-revisited old
        ;; addresses. Mirror the scalar p/update / p/update-with-args
        ;; post-process so deleted old addresses (with their [N]-shaped values)
        ;; appear in the discard, keeping forward/reverse round-trips recoverable
        ;; (genmlx-6v3h). The weight is unaffected — the deleted site is already
        ;; charged via the old [N]-shaped score.
        discard (add-deleted-to-discard (:discard result)
                                        (:choices vtrace) (:choices result))]
    {:vtrace (tag-vtrace
               (vec/->VectorizedTrace gf exec-args (:choices result)
                                      (:score result) weight
                                      n (:retval result)))
     :weight weight
     :discard discard}))

(defn vupdate
  "Batched update: run model body ONCE with batched update handler.
   vtrace: VectorizedTrace with [n]-shaped choices, constraints: new observations.
   Returns new VectorizedTrace with updated weights."
  [gf vtrace constraints key]
  (vupdate* gf vtrace (:args vtrace) constraints key))

(defn vupdate-args
  "Batched update-with-args (genmlx-s8e8): vupdate under NEW model
   arguments. Retained [n]-shaped choices are re-scored under new-args;
   the weight is the [n]-shaped thesis update weight. Batch size is
   fixed — changing n is out of scope."
  [gf vtrace new-args constraints key]
  (vupdate* gf vtrace new-args constraints key))

(defn vproject
  "Batched project (genmlx-8xia): replay the vtrace's [N]-shaped choices and
   return the [N]-shaped sum of log-probs at the selected addresses. The
   batched counterpart of p/project, used to compute the retained-only batched
   regenerate weight."
  [gf vtrace selection key]
  (let [n (:n-particles vtrace)
        result (rt/run-handler h/batched-project-transition
                 ;; [N]-shaped init so an empty/scalar selection still yields an
                 ;; [N] projected weight (genmlx-x93e).
                 {:choices cm/EMPTY :score (mx/zeros [n]) :weight (mx/zeros [n])
                  :key (rng/ensure-key key) :selection selection
                  :old-choices (:choices vtrace) :constraints cm/EMPTY
                  :batch-size n :batched? true
                  :executor execute-sub-project :param-store (param-store gf)}
                 (fn [rt] (run-body gf rt (:args vtrace))))]
    (:weight result)))

(defn- make-vregen-result-general
  "Batched retained-only regenerate (genmlx-8xia, batched counterpart of
   make-regen-result-general). Builds the new [N] trace, then
   W = vproject(new, retained) - vproject(old, retained), where retained =
   leaf addresses present in both minus the selection. For non-structure-change
   dependent-joint selections (the batched yep2 case); the [N] weight is exact
   per particle because the per-site residual cancels in the two project passes."
  [gf vtrace selection key]
  (let [n (:n-particles vtrace)
        [k1 k2] (rng/split (rng/ensure-key key))
        result (rt/run-handler h/batched-regenerate-transition-general
                 ;; [N]-shaped init score (genmlx-x93e).
                 {:choices cm/EMPTY :score (mx/zeros [n]) :key k1 :selection selection
                  :old-choices (:choices vtrace) :batch-size n :batched? true
                  :executor execute-sub :param-store (param-store gf)}
                 (fn [rt] (run-body gf rt (:args vtrace))))
        new-score (:score result)
        new-vtrace (vec/->VectorizedTrace gf (:args vtrace) (:choices result)
                                          new-score nil n (:retval result))
        retained-sel (regen-retained-selection (:choices result) (:choices vtrace) selection)
        weight (if retained-sel
                 (mx/subtract (vproject gf new-vtrace retained-sel k2)
                              (vproject gf vtrace retained-sel k2))
                 (mx/subtract new-score new-score))]  ; [N]-shaped zero (select-all)
    {:vtrace (tag-vtrace (vec/->VectorizedTrace gf (:args vtrace) (:choices result)
                                                new-score weight n (:retval result)))
     :weight weight}))

(defn batched-sub-regen
  "Executor for a spliced DynamicGF sub-GF during BATCHED regenerate
   (genmlx-20p7). Threaded into the batched handler state as :batched-sub-regen
   and invoked by the runtime splice handler for the regenerate sub-case.

   Returns nil when the sub-selection is FAST-ELIGIBLE for the sub-GF — the
   caller then uses the per-site batched transition, which is exact there
   (single-site / independent / prior-resample). Otherwise (a dependent-joint
   selection inside the sub) runs the sub's batched retained-only general path
   and returns a batched sub-result whose :weight is the parent-fast-formula
   proposal ratio (child_new_score - child_old_score - w_child), so the parent's
   (new_score - old_score) - proposal_ratio yields the child's exact retained-
   only w_child. Throws if the sub ITSELF splices (deeper batched recursion is
   unimplemented — scalar p/regenerate is exact there)."
  [sub-gf sub-args sub-selection sub-old-choices n key param-store]
  (when-not (regen-fast-eligible? sub-gf sub-selection)
    (when (seq (:splice-sites (:schema sub-gf)))
      (throw (ex-info (str "batched regenerate: a dependent-joint selection inside a "
                           "spliced sub-GF that itself splices is unsupported "
                           "(genmlx-20p7). Use scalar p/regenerate per particle.")
                      {:genmlx/error :batched-nested-splice-regenerate :addr sub-args})))
    (let [sub-gf (propagate-meta sub-gf nil param-store)
          [k1 k2] (rng/split key)
          old-vt (vec/->VectorizedTrace sub-gf sub-args sub-old-choices SCORE-ZERO nil n nil)
          result (rt/run-handler h/batched-regenerate-transition-general
                   {:choices cm/EMPTY :score SCORE-ZERO :key k1 :selection sub-selection
                    :old-choices sub-old-choices :batch-size n :batched? true
                    :executor execute-sub :param-store param-store}
                   (fn [rt] (run-body sub-gf rt sub-args)))
          child-new-score (:score result)
          new-vt (vec/->VectorizedTrace sub-gf sub-args (:choices result)
                                        child-new-score nil n (:retval result))
          retained-sel (regen-retained-selection (:choices result) sub-old-choices sub-selection)
          w-child (if retained-sel
                    (mx/subtract (vproject sub-gf new-vt retained-sel k2)
                                 (vproject sub-gf old-vt retained-sel k2))
                    (mx/subtract child-new-score child-new-score))
          ;; child's full OLD joint score (re-scored under old choices) — the
          ;; parent's recorded old score already includes it, so the conversion
          ;; below makes the parent fast formula reproduce w_child exactly.
          child-old-score (vproject sub-gf old-vt
                                    (sel/from-paths (cm/addresses sub-old-choices)) k2)
          proposal-ratio (mx/subtract (mx/subtract child-new-score child-old-score) w-child)]
      {:choices (:choices result) :score child-new-score
       :weight proposal-ratio :retval (:retval result) :score-type :joint})))

(defn vregenerate
  "Batched regenerate (genmlx-hmch / yep2 / 8xia). Returns {:vtrace :weight}.

   Routing (cond order matters):
   1. Fast-eligible at the PARENT (single-site / mutually-independent direct
      sites, no structure change) → the per-site batched convention.
   2. Otherwise, a structure-changing (has-branches?) or splice-bearing model
      whose PARENT selection is not fast-eligible → throw (host control flow
      cannot shape-batch coherently — math-verifier §7).
   3. Otherwise (dependent-joint over the parent's own sites, no structure
      change) → the batched retained-only general path (two batched project
      passes — exact, no residual).

   SPLICE RECURSION (genmlx-8xia / genmlx-20p7): a model that only SPLICES a
   sub-GF and selects no direct parent site is fast-eligible at the parent and
   takes path 1. The ONE-LEVEL dependent-joint-inside-splice case is now handled
   (genmlx-20p7): the batched sub-regen is threaded into the handler state
   (:batched-sub-regen) so the sub-GF re-gates and a dependent-joint sub-selection
   gets the exact batched retained-only weight — no yep2 residual. The remaining
   gap is DEEPER nesting: a spliced sub-GF that ITSELF splices is unsupported
   (the inner sub-GF is only a runtime value, not resolvable from the schema at
   gate time) and throws — use scalar p/regenerate per particle there, which is
   exact."
  [gf vtrace selection key]
  (cond
    (regen-fast-eligible? gf selection)
    (let [key (rng/ensure-key key)
          n (:n-particles vtrace)
          old-score (:score vtrace)
          result (rt/run-handler h/batched-regenerate-transition
                                 {:choices cm/EMPTY :score SCORE-ZERO
                                  :weight SCORE-ZERO :key key
                                  :selection selection
                                  :old-choices (:choices vtrace)
                                  :batch-size n :batched? true
                                  :executor execute-sub
                                  :batched-sub-regen batched-sub-regen
                                  :param-store (param-store gf)}
                                 (fn [rt] (run-body gf rt (:args vtrace))))
          new-score (:score result)
          proposal-ratio (:weight result)
          weight (mx/subtract (mx/subtract new-score old-score) proposal-ratio)]
      {:vtrace (tag-vtrace
                 (vec/->VectorizedTrace gf (:args vtrace) (:choices result)
                                        new-score weight n (:retval result)))
       :weight weight})

    (or (:has-branches? (:schema gf)) (seq (:splice-sites (:schema gf))))
    (throw (ex-info
             (str "vregenerate: structure-changing or spliced batched regenerate "
                  "is not supported — host control flow cannot shape-batch "
                  "coherently across particles, and batched splice recursion is "
                  "unimplemented. Use scalar p/regenerate per particle.")
             {:genmlx/error :batched-regenerate-unsupported}))

    :else
    (make-vregen-result-general gf vtrace selection key)))

(defn loop-obs
  "Create flat constraints from prefix + values sequence.
   (loop-obs \"y\" [1.0 2.0 3.0]) => choicemap with :y0 1.0 :y1 2.0 :y2 3.0"
  [prefix values]
  (reduce-kv
   (fn [cm i v] (cm/set-value cm (keyword (str prefix i)) v))
   cm/EMPTY
   (vec values)))

(defn merge-obs
  "Merge multiple choicemaps into one."
  [& cms]
  (reduce cm/merge-cm cm/EMPTY cms))

;; ---------------------------------------------------------------------------
;; Protocol extensions on DynamicGF
;;   IEdit             — edit requests dispatch through edit/edit-dispatch
;;   IUpdateWithDiffs  — short-circuit when args and constraints are unchanged
;;   IHasArgumentGrads — DynamicGF does not declare argument differentiability
;; ---------------------------------------------------------------------------

(extend-type DynamicGF
  edit/IEdit
  (edit [gf trace edit-request]
    (edit/edit-dispatch gf trace edit-request))

  p/IUpdateWithArgs
  (update-with-args [gf trace new-args argdiffs constraints]
    (if (and (diff/no-change? argdiffs) (= constraints cm/EMPTY))
      ;; Caller asserts no arg changes (Gen.jl trust model) and there are
      ;; no constraints: the trace is unchanged.
      {:trace trace :weight SCORE-ZERO :discard cm/EMPTY}
      ;; Re-execute under x' — same dispatch as update (compiled/prefix/
      ;; handler all thread the positional args), same lbae score-type
      ;; boundary guard in run-dispatched*, same deleted-address discard
      ;; post-pass.
      (let [key (ensure-key gf)
            result (@dispatch-fn gf :update new-args key
                     {:trace trace :constraints constraints :argdiffs argdiffs})]
        (mx/gfi-cleanup!)
        (clojure.core/update result :discard
          add-deleted-to-discard (:choices trace) (:choices (:trace result))))))

  p/IUpdateWithDiffs
  (update-with-diffs [gf trace constraints argdiffs]
    ;; update with change hints = update-with-args at unchanged args
    (p/update-with-args gf trace (:args trace) argdiffs constraints))

  p/IHasArgumentGrads
  (has-argument-grads [_] nil))

(defn param
  "Read a trainable parameter outside a gen body.
   Returns the default value as an MLX array (no param store available
   outside gen body execution). Inside gen bodies, use the param local
   binding from the gen macro instead."
  [name default-value]
  (if (mx/array? default-value) default-value (mx/scalar default-value)))
