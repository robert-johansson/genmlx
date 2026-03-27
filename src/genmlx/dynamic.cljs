(ns genmlx.dynamic
  "DynamicDSLFunction — implements the full GFI by delegating to
   handler-based execution, with optional compiled fast path for
   static models (Level 1)."
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
            [genmlx.conjugacy :as conjugacy]
            [genmlx.rewrite :as rewrite]
            [genmlx.inference.auto-analytical :as auto-analytical]
            [clojure.set]))

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
          unused (clojure.set/difference constraint-keys (or trace-keys #{}))]
      (when (seq unused) unused))))

;; ---------------------------------------------------------------------------
;; Trace construction helpers
;; ---------------------------------------------------------------------------

(defn- attach-splice-scores
  "Attach splice scores as metadata on a trace, if present."
  [trace result]
  (if-let [ss (:splice-scores result)]
    (with-meta trace {::splice-scores ss})
    trace))

(defn- make-result-trace
  "Build a Trace from a handler/compiled result map."
  [gf args result]
  (tr/make-trace {:gen-fn gf :args args
                  :choices (:choices result)
                  :retval (:retval result)
                  :score (:score result)}))

(defn- make-generate-result
  "Build the generate return map: {:trace :weight}, with unused-constraint
   detection and splice-score attachment."
  [trace weight constraints result-choices result]
  (let [result-map {:trace (attach-splice-scores trace result)
                    :weight weight}]
    (if-let [unused (find-unused-constraints constraints result-choices)]
      (assoc result-map :unused-constraints unused)
      result-map)))

(defn- make-update-result
  "Build the update return map: {:trace :weight :discard}, with unused-constraint
   detection and splice-score attachment."
  [trace weight discard constraints result-choices result]
  (let [result-map {:trace (attach-splice-scores trace result)
                    :weight weight
                    :discard discard}]
    (if-let [unused (find-unused-constraints constraints result-choices)]
      (assoc result-map :unused-constraints unused)
      result-map)))

(defn- make-regen-result
  "Build the regenerate return map from handler result, computing the MH weight."
  [gf trace result old-score]
  (let [new-score (:score result)
        proposal-ratio (:weight result)
        weight (mx/subtract (mx/subtract new-score old-score) proposal-ratio)
        new-trace (tr/make-trace {:gen-fn gf :args (:args trace)
                                  :choices (:choices result)
                                  :retval (:retval result)
                                  :score new-score})]
    {:trace (attach-splice-scores new-trace result)
     :weight weight}))

;; ---------------------------------------------------------------------------
;; Simulate path helpers
;; ---------------------------------------------------------------------------

(defn- run-simulate-compiled
  "L1-M2: full compiled simulate path."
  [compiled-sim gf args key]
  (let [result (compiled-sim key (vec args))
        choices (cm/from-flat-map (:values result))]
    (tr/make-trace {:gen-fn gf :args args
                    :choices choices
                    :retval (:retval result)
                    :score (:score result)})))

(defn- run-simulate-prefix
  "L1-M3: compiled prefix + replay handler."
  [compiled-pfx gf args key body-fn this]
  (let [result (compiled-pfx key (vec args))
        replay (compiled/make-replay-simulate-transition (:values result))
        handler-result (rt/run-handler replay
                                       {:choices cm/EMPTY :score (:score result) :key key
                                        :executor execute-sub
                                        :param-store (::param-store (meta this))}
                                       (fn [rt] (apply body-fn rt args)))
        trace (make-result-trace gf args handler-result)]
    (attach-splice-scores trace handler-result)))

(defn- run-simulate-handler
  "L0: handler fallback simulate."
  [gf args key body-fn this]
  (let [result (rt/run-handler h/simulate-transition
                               {:choices cm/EMPTY :score SCORE-ZERO :key key
                                :executor execute-sub
                                :param-store (::param-store (meta this))}
                               (fn [rt] (apply body-fn rt args)))
        trace (make-result-trace gf args result)]
    (attach-splice-scores trace result)))

;; ---------------------------------------------------------------------------
;; Generate path helpers
;; ---------------------------------------------------------------------------

(defn- run-generate-analytical
  "L3: auto-analytical generate with conjugate elimination."
  [schema gf args key constraints body-fn this]
  (let [transition (auto-analytical/make-address-dispatch
                    h/generate-transition (:auto-handlers schema))
        result (rt/run-handler transition
                               {:choices cm/EMPTY :score SCORE-ZERO
                                :weight SCORE-ZERO
                                :key key :constraints constraints
                                :auto-posteriors {}
                                :auto-kalman-beliefs {}
                                :auto-kalman-noise-vars {}
                                :executor execute-sub
                                :param-store (::param-store (meta this))}
                               (fn [rt] (apply body-fn rt args)))
        trace (make-result-trace gf args result)
        ;; Tag trace with marginal score type so regenerate can match
        trace (vary-meta trace assoc ::score-type :marginal)]
    (make-generate-result trace (:weight result) constraints (:choices result) result)))

(defn- run-generate-compiled
  "L1: fully compiled generate (static or branch-rewritten)."
  [compiled-gen gf args key constraints]
  (let [result (compiled-gen key (vec args) constraints)
        choices (cm/from-flat-map (:values result))
        trace (tr/make-trace {:gen-fn gf :args args
                              :choices choices
                              :retval (:retval result)
                              :score (:score result)})]
    {:trace trace :weight (:weight result)}))

(defn- run-generate-prefix
  "L1-M3: compiled prefix generate + replay handler."
  [compiled-pfx-gen gf args key constraints body-fn this]
  (let [result (compiled-pfx-gen key (vec args) constraints)
        replay (compiled/make-replay-generate-transition (:values result))
        handler-result (rt/run-handler replay
                                       {:choices cm/EMPTY :score (:score result)
                                        :weight (:weight result)
                                        :key key :constraints constraints
                                        :executor execute-sub
                                        :param-store (::param-store (meta this))}
                                       (fn [rt] (apply body-fn rt args)))
        trace (make-result-trace gf args handler-result)]
    (make-generate-result trace (:weight handler-result) constraints
                          (:choices handler-result) handler-result)))

(defn- run-generate-handler
  "L0: handler fallback generate."
  [gf args key constraints body-fn this]
  (let [result (rt/run-handler h/generate-transition
                               {:choices cm/EMPTY :score SCORE-ZERO
                                :weight SCORE-ZERO
                                :key key :constraints constraints
                                :executor execute-sub
                                :param-store (::param-store (meta this))}
                               (fn [rt] (apply body-fn rt args)))
        trace (make-result-trace gf args result)]
    (make-generate-result trace (:weight result) constraints (:choices result) result)))

;; ---------------------------------------------------------------------------
;; Update path helpers
;; ---------------------------------------------------------------------------

(defn- run-update-compiled
  "L1: fully compiled update (static or branch-rewritten)."
  [compiled-upd gf trace key constraints]
  (let [result (compiled-upd key (vec (:args trace)) constraints (:choices trace))
        choices (cm/from-flat-map (:values result))
        discard (cm/from-flat-map (:discard result))
        new-trace (tr/make-trace {:gen-fn gf :args (:args trace)
                                  :choices choices
                                  :retval (:retval result)
                                  :score (:score result)})]
    {:trace new-trace
     :weight (mx/subtract (:score result) (:score trace))
     :discard discard}))

(defn- run-update-prefix
  "L1-M3: compiled prefix update + replay handler."
  [compiled-pfx-upd gf trace key constraints body-fn this]
  (let [result (compiled-pfx-upd key (vec (:args trace)) constraints (:choices trace))
        prefix-discard (cm/from-flat-map (:discard result))
        replay (cops/make-replay-update-transition (:values result))
        handler-result (rt/run-handler replay
                                       {:choices cm/EMPTY :score (:score result)
                                        :weight SCORE-ZERO
                                        :key key :constraints constraints
                                        :old-choices (:choices trace)
                                        :discard prefix-discard
                                        :executor execute-sub
                                        :param-store (::param-store (meta this))}
                                       (fn [rt] (apply body-fn rt (:args trace))))
        new-trace (make-result-trace gf (:args trace) handler-result)]
    (make-update-result new-trace
                        (mx/subtract (:score handler-result) (:score trace))
                        (:discard handler-result)
                        constraints (:choices handler-result) handler-result)))

(defn- add-deleted-to-discard
  "Add old trace addresses not present in new trace to the discard.
   When a model switches branches, addresses visited in the old trace
   but absent in the new trace must appear in the discard (Gen.jl semantics)."
  [discard old-choices new-choices]
  (if (and (instance? cm/Node old-choices)
           (instance? cm/Node new-choices))
    (let [old-keys (set (keys (:m old-choices)))
          new-keys (set (keys (:m new-choices)))
          deleted (clojure.set/difference old-keys new-keys)]
      (reduce (fn [d addr]
                (let [old-sub (get (:m old-choices) addr)]
                  (if (cm/has-value? old-sub)
                    (cm/set-value d addr (cm/get-value old-sub))
                    (cm/set-submap d addr old-sub))))
              discard
              deleted))
    discard))

(defn- run-update-handler
  "L0: handler fallback update."
  [gf trace key constraints body-fn this]
  (let [result (rt/run-handler h/update-transition
                               {:choices cm/EMPTY :score SCORE-ZERO
                                :weight SCORE-ZERO
                                :key key :constraints constraints
                                :old-choices (:choices trace)
                                :old-splice-scores (::splice-scores (meta trace))
                                :discard cm/EMPTY
                                :executor execute-sub
                                :param-store (::param-store (meta this))}
                               (fn [rt] (apply body-fn rt (:args trace))))
        new-trace (make-result-trace gf (:args trace) result)]
    (make-update-result new-trace
                        (mx/subtract (:score result) (:score trace))
                        (:discard result)
                        constraints (:choices result) result)))

;; ---------------------------------------------------------------------------
;; Regenerate path helpers
;; ---------------------------------------------------------------------------

(defn- run-regen-analytical
  "L3.5: auto-analytical regenerate with precomputed transition."
  [schema gf trace key selection old-score body-fn this]
  (let [result (rt/run-handler (:auto-regenerate-transition schema)
                               {:choices cm/EMPTY :score SCORE-ZERO
                                :weight SCORE-ZERO
                                :key key :selection selection
                                :old-choices (:choices trace)
                                :auto-posteriors {}
                                :auto-kalman-beliefs {}
                                :auto-kalman-noise-vars {}
                                :old-splice-scores (::splice-scores (meta trace))
                                :executor execute-sub
                                :param-store (::param-store (meta this))}
                               (fn [rt] (apply body-fn rt (:args trace))))
        regen-result (make-regen-result gf trace result old-score)]
    ;; Preserve marginal score type on the new trace
    (update regen-result :trace vary-meta assoc ::score-type :marginal)))

(defn- run-regen-compiled
  "L1: fully compiled regenerate (M2/M4)."
  [cregen gf trace key selection old-score]
  (let [result (cregen key (vec (:args trace)) (:choices trace) selection)
        new-score (:score result)
        proposal-ratio (:weight result)
        weight (mx/subtract (mx/subtract new-score old-score) proposal-ratio)
        choices (cm/from-flat-map (:values result))
        new-trace (tr/make-trace {:gen-fn gf :args (:args trace)
                                  :choices choices
                                  :retval (:retval result)
                                  :score new-score})]
    {:trace new-trace :weight weight}))

(defn- run-regen-prefix
  "L1-M3: prefix regenerate + replay handler."
  [prefix-regen gf trace key selection old-score body-fn this]
  (let [prefix-result (prefix-regen key (vec (:args trace))
                                    (:choices trace) selection)
        replay (cops/make-replay-regenerate-transition (:values prefix-result))
        handler-result (rt/run-handler replay
                                       {:choices cm/EMPTY :score (:score prefix-result)
                                        :weight (:weight prefix-result)
                                        :key key :selection selection
                                        :old-choices (:choices trace)
                                        :executor execute-sub
                                        :param-store (::param-store (meta this))}
                                       (fn [rt] (apply body-fn rt (:args trace))))]
    (make-regen-result gf trace handler-result old-score)))

(defn- run-regen-handler
  "L0: handler fallback regenerate."
  [gf trace key selection old-score body-fn this]
  (let [result (rt/run-handler h/regenerate-transition
                               {:choices cm/EMPTY :score SCORE-ZERO
                                :weight SCORE-ZERO
                                :key key :selection selection
                                :old-choices (:choices trace)
                                :old-splice-scores (::splice-scores (meta trace))
                                :executor execute-sub
                                :param-store (::param-store (meta this))}
                               (fn [rt] (apply body-fn rt (:args trace))))]
    (make-regen-result gf trace result old-score)))

;; ---------------------------------------------------------------------------
;; Assess path helpers
;; ---------------------------------------------------------------------------

(defn- run-assess-analytical
  "L3.5: auto-analytical assess with conjugate elimination."
  [schema gf args key choices body-fn this]
  (let [transition (auto-analytical/make-address-dispatch
                    h/assess-transition (:auto-handlers schema))
        result (rt/run-handler transition
                               {:choices cm/EMPTY :score SCORE-ZERO
                                :weight SCORE-ZERO
                                :key key :constraints choices
                                :auto-posteriors {}
                                :auto-kalman-beliefs {}
                                :auto-kalman-noise-vars {}
                                :executor execute-sub-assess
                                :param-store (::param-store (meta this))}
                               (fn [rt] (apply body-fn rt args)))]
    {:retval (:retval result)
     :weight (:score result)}))

(defn- run-assess-prefix
  "L1-M3: prefix assess + replay handler."
  [prefix-assess gf args key choices body-fn this]
  (let [prefix-result (prefix-assess (vec args) choices)
        replay-transition (cops/make-replay-assess-transition (:values prefix-result))
        result (rt/run-handler replay-transition
                               {:choices cm/EMPTY :score (:score prefix-result)
                                :weight (:score prefix-result)
                                :key key :constraints choices
                                :executor execute-sub-assess
                                :param-store (::param-store (meta this))}
                               (fn [rt] (apply body-fn rt args)))]
    {:retval (:retval result)
     :weight (:score result)}))

(defn- run-assess-handler
  "L0: handler fallback assess."
  [gf args key choices body-fn this]
  (let [result (rt/run-handler h/assess-transition
                               {:choices cm/EMPTY :score SCORE-ZERO
                                :weight SCORE-ZERO
                                :key key :constraints choices
                                :executor execute-sub-assess
                                :param-store (::param-store (meta this))}
                               (fn [rt] (apply body-fn rt args)))]
    {:retval (:retval result)
     :weight (:score result)}))

;; ---------------------------------------------------------------------------
;; Project path helpers
;; ---------------------------------------------------------------------------

(defn- run-project-prefix
  "L1-M3: prefix project + replay handler."
  [prefix-proj gf trace key selection body-fn this]
  (let [prefix-result (prefix-proj (vec (:args trace)) (:choices trace) selection)
        replay-transition (cops/make-replay-project-transition (:values prefix-result))
        result (rt/run-handler replay-transition
                               {:choices cm/EMPTY :score SCORE-ZERO
                                :weight (:weight prefix-result)
                                :key key :selection selection
                                :old-choices (:choices trace)
                                :constraints cm/EMPTY
                                :executor execute-sub-project
                                :param-store (::param-store (meta this))}
                               (fn [rt] (apply body-fn rt (:args trace))))]
    (:weight result)))

(defn- run-project-handler
  "L0: handler fallback project."
  [gf trace key selection body-fn this]
  (let [result (rt/run-handler h/project-transition
                               {:choices cm/EMPTY :score SCORE-ZERO
                                :weight SCORE-ZERO
                                :key key :selection selection
                                :old-choices (:choices trace)
                                :constraints cm/EMPTY
                                :executor execute-sub-project
                                :param-store (::param-store (meta this))}
                               (fn [rt] (apply body-fn rt (:args trace))))]
    (:weight result)))

;; ---------------------------------------------------------------------------
;; Analytical applicability predicate
;; ---------------------------------------------------------------------------

(defn- analytical-applicable?
  "Check if L3 auto-analytical handlers apply for the given schema + constraints.
   Skips analytical path inside mx/grad or mx/value-and-grad because the
   analytical handler uses volatile! which breaks gradient flow."
  [schema constraints]
  (and (not (mx/in-grad?))
       (:auto-handlers schema)
       (auto-analytical/some-conjugate-obs-constrained?
        (:conjugate-pairs schema) constraints)))

;; ---------------------------------------------------------------------------
;; DynamicGF record — GFI protocol implementations
;; ---------------------------------------------------------------------------

(defrecord DynamicGF [body-fn source schema]
  p/IGenerativeFunction
  (simulate [this args]
    (let [key (ensure-key this)
          _ (rng/seed! key)
          result
          (cond
            ;; L1-M2: full compiled simulate
            (:compiled-simulate schema)
            (run-simulate-compiled (:compiled-simulate schema) this args key)

            ;; L1-M3: compiled prefix + replay handler
            (:compiled-prefix schema)
            (run-simulate-prefix (:compiled-prefix schema) this args key body-fn this)

            ;; L0: handler fallback
            :else
            (run-simulate-handler this args key body-fn this))]
      (mx/gfi-cleanup!)
      result))

  p/IGenerate
  (generate [this args constraints]
    (let [key (ensure-key this)
          _ (rng/seed! key)
          result
          (cond
            ;; L3: auto-analytical generate
            (analytical-applicable? schema constraints)
            (run-generate-analytical schema this args key constraints body-fn this)

            ;; L1: fully compiled generate (static or branch-rewritten)
            (:compiled-generate schema)
            (run-generate-compiled (:compiled-generate schema) this args key constraints)

            ;; L1-M3: compiled prefix generate + replay handler
            (:compiled-prefix-generate schema)
            (run-generate-prefix (:compiled-prefix-generate schema) this args key constraints body-fn this)

            ;; L0: handler fallback
            :else
            (run-generate-handler this args key constraints body-fn this))]
      (mx/gfi-cleanup!)
      result))

  p/IUpdate
  (update [this trace constraints]
    (let [key (ensure-key this)
          _ (rng/seed! key)
          result
          (cond
            ;; L1: fully compiled update (static or branch-rewritten)
            (:compiled-update schema)
            (run-update-compiled (:compiled-update schema) this trace key constraints)

            ;; L1-M3: compiled prefix update + replay handler
            (:compiled-prefix-update schema)
            (run-update-prefix (:compiled-prefix-update schema) this trace key constraints body-fn this)

            ;; L0: handler fallback
            :else
            (run-update-handler this trace key constraints body-fn this))]
      ;; Post-process: add addresses deleted by branch switches to discard.
      ;; Old addresses not present in new trace must appear in discard (Gen.jl semantics).
      (mx/gfi-cleanup!)
      (clojure.core/update result :discard
                           add-deleted-to-discard (:choices trace) (:choices (:trace result)))))

  p/IRegenerate
  (regenerate [this trace selection]
    (let [key (ensure-key this)
          _ (rng/seed! key)
          old-score (:score trace)
          result
          (cond
            ;; L3.5: auto-analytical regenerate — only when trace has marginal scoring.
            ;; Case A (prior selected): handler returns nil, falls through to base
            ;; Case B (prior NOT selected): replays old value with marginal LL
            ;; Traces from simulate have joint LL (no ::score-type), so they fall
            ;; through to the standard handler, preserving weight = 0 for empty selection.
            (and (:auto-regenerate-transition schema)
                 (= :marginal (::score-type (meta trace))))
            (run-regen-analytical schema this trace key selection old-score body-fn this)

            ;; L1: fully compiled regenerate (M2/M4)
            (:compiled-regenerate schema)
            (run-regen-compiled (:compiled-regenerate schema) this trace key selection old-score)

            ;; L1-M3: prefix regenerate + replay handler
            (:compiled-prefix-regenerate schema)
            (run-regen-prefix (:compiled-prefix-regenerate schema) this trace key selection old-score body-fn this)

            ;; L0: handler fallback
            :else
            (run-regen-handler this trace key selection old-score body-fn this))]
      (mx/gfi-cleanup!)
      result))

  p/IAssess
  (assess [this args choices]
    (let [key (ensure-key this)
          _ (rng/seed! key)
          result
          (cond
            ;; L3.5: auto-analytical assess — only when prior is free (e.g., partial assess).
            ;; In standard assess (all choices provided), prior is constrained,
            ;; so analytical-applicable? returns false → handler fallback → joint LL.
            ;; This preserves the GFI identity: simulate.score == assess(choices).weight
            (analytical-applicable? schema choices)
            (run-assess-analytical schema this args key choices body-fn this)

            ;; L1: fully compiled assess (M2/M4)
            (:compiled-assess schema)
            (let [r ((:compiled-assess schema) (vec args) choices)]
              {:retval (:retval r)
               :weight (:score r)})

            ;; L1-M3: prefix assess + replay
            (:compiled-prefix-assess schema)
            (run-assess-prefix (:compiled-prefix-assess schema) this args key choices body-fn this)

            ;; L0: handler fallback
            :else
            (run-assess-handler this args key choices body-fn this))]
      (mx/gfi-cleanup!)
      result))

  p/IPropose
  (propose [this args]
    (let [key (ensure-key this)
          _ (rng/seed! key)
          result (rt/run-handler h/simulate-transition
                                 {:choices cm/EMPTY :score SCORE-ZERO :key key
                                  :executor execute-sub
                                  :param-store (::param-store (meta this))}
                                 (fn [rt] (apply body-fn rt args)))]
      (mx/gfi-cleanup!)
      {:choices (:choices result)
       :weight (:score result)
       :retval (:retval result)}))

  p/IProject
  (project [this trace selection]
    (let [key (ensure-key this)
          _ (rng/seed! key)
          result
          (cond
            ;; L1: fully compiled project (M2/M4)
            (:compiled-project schema)
            ((:compiled-project schema) (vec (:args trace)) (:choices trace) selection)

            ;; L1-M3: prefix project + replay
            (:compiled-prefix-project schema)
            (run-project-prefix (:compiled-prefix-project schema) this trace key selection body-fn this)

            ;; L0: handler fallback
            :else
            (run-project-handler this trace key selection body-fn this))]
      (mx/gfi-cleanup!)
      result)))

(defn- execute-sub
  "Execute a sub-generative-function during handler execution.
   Delegates to the sub-gf's own GFI methods.
   Propagates param-store and key to sub-gfs via metadata."
  [gf args {:keys [constraints old-choices selection key old-splice-score param-store]}]
  (let [gf (cond-> gf
             key (vary-meta assoc ::key key)
             param-store (vary-meta assoc ::param-store param-store))]
    (cond
      ;; Regenerate mode
      selection
      (let [{:keys [trace weight]}
            (p/regenerate gf
                          (tr/make-trace {:gen-fn gf :args args
                                          :choices (or old-choices cm/EMPTY)
                                          :retval nil :score (or old-splice-score SCORE-ZERO)})
                          selection)]
        {:choices (:choices trace) :retval (:retval trace)
         :score (:score trace) :weight weight})

      ;; Update mode: has old-choices (possibly with new constraints)
      (and old-choices (not= old-choices cm/EMPTY))
      (let [old-trace (tr/make-trace {:gen-fn gf :args args
                                      :choices old-choices
                                      :retval nil :score (or old-splice-score SCORE-ZERO)})
            {:keys [trace weight discard]} (p/update gf old-trace
                                                     (or constraints cm/EMPTY))]
        {:choices (:choices trace) :retval (:retval trace)
         :score (:score trace) :weight weight :discard discard})

      ;; Generate with constraints
      (and constraints (not= constraints cm/EMPTY))
      (let [{:keys [trace weight]} (p/generate gf args constraints)]
        {:choices (:choices trace) :retval (:retval trace)
         :score (:score trace) :weight weight})

      ;; Plain simulate
      :else
      (let [trace (p/simulate gf args)]
        {:choices (:choices trace) :retval (:retval trace)
         :score (:score trace)}))))

(defn- execute-sub-project
  "Execute sub-GF in project mode: replay via generate, then project.
   Propagates param-store and key via metadata."
  [gf args {:keys [old-choices selection key param-store]}]
  (let [gf (cond-> gf
             key (vary-meta assoc ::key key)
             param-store (vary-meta assoc ::param-store param-store))
        {:keys [trace]} (p/generate gf args (or old-choices cm/EMPTY))
        weight (p/project gf trace (or selection sel/none))]
    {:choices (:choices trace)
     :retval (:retval trace)
     :score (:score trace)
     :weight weight}))

(defn- execute-sub-assess
  "Execute a sub-GF in assess mode: all choices must be provided.
   Propagates param-store and key via metadata."
  [gf args {:keys [constraints key param-store]}]
  (let [gf (cond-> gf
             key (vary-meta assoc ::key key)
             param-store (vary-meta assoc ::param-store param-store))
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
        ;; L3: full rewrite engine (Kalman > Conjugacy > RaoBlackwell)
        schema (if schema
                 (let [augmented (conjugacy/augment-schema-with-conjugacy schema)]
                   (if (:has-conjugate? augmented)
                     (let [plan (rewrite/build-analytical-plan augmented)
                           regen-handlers (auto-analytical/build-all-regenerate-handlers
                                           (:conjugate-pairs augmented)
                                           :chains (:kalman-chains plan))
                           ;; Opt 1: precompute dispatch transition once at construction
                           regen-transition (when (seq regen-handlers)
                                              (auto-analytical/make-address-dispatch
                                               h/regenerate-transition regen-handlers))]
                       (-> augmented
                           (assoc :auto-handlers (get-in plan [:rewrite-result :handlers]))
                           (assoc :auto-regenerate-handlers regen-handlers)
                           (assoc :auto-regenerate-transition regen-transition)
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
;; Direct-mode param access (outside gen bodies)
;; ---------------------------------------------------------------------------

;; ---------------------------------------------------------------------------
;; Vectorized execution (batched: N particles in one model run)
;; ---------------------------------------------------------------------------

(defn vsimulate
  "Run model body ONCE with batched handler, producing a VectorizedTrace
   with [n]-shaped arrays at each choice site.
   gf: DynamicGF, args: model args, n: number of particles, key: PRNG key."
  [gf args n key]
  (let [key (rng/ensure-key key)
        _ (rng/seed! key)
        result (rt/run-handler h/batched-simulate-transition
                 {:choices cm/EMPTY :score SCORE-ZERO
                  :key key :batch-size n :batched? true
                  :executor execute-sub
                  :param-store (::param-store (meta gf))}
                 (fn [rt] (apply (:body-fn gf) rt args)))]
    (vec/->VectorizedTrace gf args (:choices result) (:score result)
                           (mx/zeros [n]) n (:retval result))))

(defn vgenerate
  "Run model body ONCE with batched generate handler, producing a
   VectorizedTrace. Constrained sites use scalar observations;
   unconstrained sites produce [n]-shaped samples.
   gf: DynamicGF, args: model args, constraints: ChoiceMap,
   n: number of particles, key: PRNG key."
  [gf args constraints n key]
  (let [key (rng/ensure-key key)
        _ (rng/seed! key)
        result (rt/run-handler h/batched-generate-transition
                 {:choices cm/EMPTY :score SCORE-ZERO
                  :weight SCORE-ZERO :key key
                  :constraints constraints :batch-size n :batched? true
                  :executor execute-sub
                  :param-store (::param-store (meta gf))}
                 (fn [rt] (apply (:body-fn gf) rt args)))]
    (vec/->VectorizedTrace gf args (:choices result) (:score result)
                           (:weight result) n (:retval result))))

(defn vupdate
  "Batched update: run model body ONCE with batched update handler.
   vtrace: VectorizedTrace with [n]-shaped choices, constraints: new observations.
   Returns new VectorizedTrace with updated weights."
  [gf vtrace constraints key]
  (let [key (rng/ensure-key key)
        _ (rng/seed! key)
        n (:n-particles vtrace)
        result (rt/run-handler h/batched-update-transition
                               {:choices cm/EMPTY :score SCORE-ZERO
                                :weight SCORE-ZERO :key key
                                :constraints constraints
                                :old-choices (:choices vtrace)
                                :discard cm/EMPTY
                                :batch-size n :batched? true
                                :executor execute-sub
                                :param-store (::param-store (meta gf))}
                               (fn [rt] (apply (:body-fn gf) rt (:args vtrace))))]
    {:vtrace (vec/->VectorizedTrace gf (:args vtrace) (:choices result)
                                    (:score result) (:weight result)
                                    n (:retval result))
     :weight (:weight result)
     :discard (:discard result)}))

(defn vregenerate
  "Batched regenerate: run model body ONCE with batched regenerate handler.
   vtrace: VectorizedTrace with [n]-shaped choices, selection: addresses to resample.
   Returns new VectorizedTrace with resampled selected addresses."
  [gf vtrace selection key]
  (let [key (rng/ensure-key key)
        _ (rng/seed! key)
        n (:n-particles vtrace)
        old-score (:score vtrace)
        result (rt/run-handler h/batched-regenerate-transition
                               {:choices cm/EMPTY :score SCORE-ZERO
                                :weight SCORE-ZERO :key key
                                :selection selection
                                :old-choices (:choices vtrace)
                                :batch-size n :batched? true
                                :executor execute-sub
                                :param-store (::param-store (meta gf))}
                               (fn [rt] (apply (:body-fn gf) rt (:args vtrace))))
        new-score (:score result)
        proposal-ratio (:weight result)
        weight (mx/subtract (mx/subtract new-score old-score) proposal-ratio)]
    {:vtrace (vec/->VectorizedTrace gf (:args vtrace) (:choices result)
                                    new-score weight n (:retval result))
     :weight weight}))

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
;; IEdit implementation on DynamicGF
;; ---------------------------------------------------------------------------

(extend-type DynamicGF
  edit/IEdit
  (edit [gf trace edit-request]
    (edit/edit-dispatch gf trace edit-request)))

;; ---------------------------------------------------------------------------
;; IUpdateWithDiffs implementation on DynamicGF
;; ---------------------------------------------------------------------------

(extend-type DynamicGF
  p/IUpdateWithDiffs
  (update-with-diffs [gf trace constraints argdiffs]
    (if (and (diff/no-change? argdiffs) (= constraints cm/EMPTY))
      ;; No arg changes and no constraints: trace is unchanged
      {:trace trace :weight SCORE-ZERO :discard cm/EMPTY}
      ;; Otherwise delegate to regular update (body must be re-executed)
      (p/update gf trace constraints))))

(defn param
  "Read a trainable parameter outside a gen body.
   Returns the default value as an MLX array (no param store available
   outside gen body execution). Inside gen bodies, use the param local
   binding from the gen macro instead."
  [name default-value]
  (if (mx/array? default-value) default-value (mx/scalar default-value)))

;; ---------------------------------------------------------------------------
;; IHasArgumentGrads — DynamicGF does not declare argument differentiability
;; ---------------------------------------------------------------------------

(extend-type DynamicGF
  p/IHasArgumentGrads
  (has-argument-grads [_] nil))
