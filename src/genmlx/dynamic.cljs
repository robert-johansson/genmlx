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
            [genmlx.conjugacy :as conjugacy]
            [genmlx.rewrite :as rewrite]
            [genmlx.inference.auto-analytical :as auto-analytical]
            [genmlx.schemas :as schemas]
            [genmlx.dispatch :as dispatch]
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
;; Execution helpers — each takes [gf args key opts] and returns a GFI result.
;; Grouped by execution strategy (handler, compiled, prefix, analytical).
;; ---------------------------------------------------------------------------

;; -- Common: param-store and body-fn extraction --

(defn- ps  [gf] (::param-store (meta gf)))
(defn- bfn [gf] (:body-fn gf))
(defn- run-body [gf rt args] (apply (bfn gf) rt args))

;; -- L0: Handler path --

(defn- run-simulate-handler [gf args key opts]
  (let [result (rt/run-handler h/simulate-transition
                 {:choices cm/EMPTY :score SCORE-ZERO :key key
                  :executor execute-sub :param-store (ps gf)}
                 (fn [rt] (run-body gf rt args)))]
    (attach-splice-scores (make-result-trace gf args result) result)))

(defn- run-generate-handler [gf args key {:keys [constraints]}]
  (let [result (rt/run-handler h/generate-transition
                 {:choices cm/EMPTY :score SCORE-ZERO :weight SCORE-ZERO
                  :key key :constraints constraints
                  :executor execute-sub :param-store (ps gf)}
                 (fn [rt] (run-body gf rt args)))
        trace (make-result-trace gf args result)]
    (make-generate-result trace (:weight result) constraints (:choices result) result)))

(defn- run-update-handler [gf args key {:keys [trace constraints]}]
  (let [result (rt/run-handler h/update-transition
                 {:choices cm/EMPTY :score SCORE-ZERO :weight SCORE-ZERO
                  :key key :constraints constraints
                  :old-choices (:choices trace)
                  :old-splice-scores (::splice-scores (meta trace))
                  :discard cm/EMPTY
                  :executor execute-sub :param-store (ps gf)}
                 (fn [rt] (run-body gf rt (:args trace))))
        new-trace (make-result-trace gf (:args trace) result)]
    (make-update-result new-trace
      (mx/subtract (:score result) (:score trace))
      (:discard result) constraints (:choices result) result)))

(defn- run-regen-handler [gf args key {:keys [trace selection]}]
  (let [old-score (:score trace)
        result (rt/run-handler h/regenerate-transition
                 {:choices cm/EMPTY :score SCORE-ZERO :weight SCORE-ZERO
                  :key key :selection selection
                  :old-choices (:choices trace)
                  :old-splice-scores (::splice-scores (meta trace))
                  :executor execute-sub :param-store (ps gf)}
                 (fn [rt] (run-body gf rt (:args trace))))]
    (make-regen-result gf trace result old-score)))

(defn- run-assess-handler [gf args key {:keys [constraints]}]
  (let [result (rt/run-handler h/assess-transition
                 {:choices cm/EMPTY :score SCORE-ZERO :weight SCORE-ZERO
                  :key key :constraints constraints
                  :executor execute-sub-assess :param-store (ps gf)}
                 (fn [rt] (run-body gf rt args)))]
    {:retval (:retval result) :weight (:score result)}))

(defn- run-project-handler [gf args key {:keys [trace selection]}]
  (let [result (rt/run-handler h/project-transition
                 {:choices cm/EMPTY :score SCORE-ZERO :weight SCORE-ZERO
                  :key key :selection selection
                  :old-choices (:choices trace)
                  :constraints cm/EMPTY
                  :executor execute-sub-project :param-store (ps gf)}
                 (fn [rt] (run-body gf rt (:args trace))))]
    (:weight result)))

(defn- run-propose-handler [gf args key opts]
  (let [result (rt/run-handler h/simulate-transition
                 {:choices cm/EMPTY :score SCORE-ZERO :key key
                  :executor execute-sub :param-store (ps gf)}
                 (fn [rt] (run-body gf rt args)))]
    {:choices (:choices result)
     :weight (:score result)
     :retval (:retval result)}))

;; -- L1-M2: Compiled path --

(defn- run-simulate-compiled [gf args key opts]
  (let [cfn (:compiled-simulate (:schema gf))
        result (cfn key (vec args))]
    (tr/make-trace {:gen-fn gf :args args
                    :choices (cm/from-flat-map (:values result))
                    :retval (:retval result) :score (:score result)})))

(defn- run-generate-compiled [gf args key {:keys [constraints]}]
  (let [cfn (:compiled-generate (:schema gf))
        result (cfn key (vec args) constraints)]
    {:trace (tr/make-trace {:gen-fn gf :args args
                            :choices (cm/from-flat-map (:values result))
                            :retval (:retval result) :score (:score result)})
     :weight (:weight result)}))

(defn- run-update-compiled [gf args key {:keys [trace constraints]}]
  (let [cfn (:compiled-update (:schema gf))
        result (cfn key (vec (:args trace)) constraints (:choices trace))]
    {:trace (tr/make-trace {:gen-fn gf :args (:args trace)
                            :choices (cm/from-flat-map (:values result))
                            :retval (:retval result) :score (:score result)})
     :weight (mx/subtract (:score result) (:score trace))
     :discard (cm/from-flat-map (:discard result))}))

(defn- run-regen-compiled [gf args key {:keys [trace selection]}]
  (let [cfn (:compiled-regenerate (:schema gf))
        old-score (:score trace)
        result (cfn key (vec (:args trace)) (:choices trace) selection)
        weight (mx/subtract (mx/subtract (:score result) old-score) (:weight result))]
    {:trace (tr/make-trace {:gen-fn gf :args (:args trace)
                            :choices (cm/from-flat-map (:values result))
                            :retval (:retval result) :score (:score result)})
     :weight weight}))

(defn- run-assess-compiled [gf args key {:keys [constraints]}]
  (let [cfn (:compiled-assess (:schema gf))
        r (cfn (vec args) constraints)]
    {:retval (:retval r) :weight (:score r)}))

(defn- run-project-compiled [gf args key {:keys [trace selection]}]
  (let [cfn (:compiled-project (:schema gf))]
    (cfn (vec (:args trace)) (:choices trace) selection)))

;; -- L1-M3: Prefix compiled path --

(defn- run-simulate-prefix [gf args key opts]
  (let [pfx (:compiled-prefix (:schema gf))
        result (pfx key (vec args))
        replay (compiled/make-replay-simulate-transition (:values result))
        handler-result (rt/run-handler replay
                         {:choices cm/EMPTY :score (:score result) :key key
                          :executor execute-sub :param-store (ps gf)}
                         (fn [rt] (run-body gf rt args)))]
    (attach-splice-scores (make-result-trace gf args handler-result) handler-result)))

(defn- run-generate-prefix [gf args key {:keys [constraints]}]
  (let [pfx (:compiled-prefix-generate (:schema gf))
        result (pfx key (vec args) constraints)
        replay (compiled/make-replay-generate-transition (:values result))
        handler-result (rt/run-handler replay
                         {:choices cm/EMPTY :score (:score result)
                          :weight (:weight result) :key key :constraints constraints
                          :executor execute-sub :param-store (ps gf)}
                         (fn [rt] (run-body gf rt args)))
        trace (make-result-trace gf args handler-result)]
    (make-generate-result trace (:weight handler-result) constraints
                          (:choices handler-result) handler-result)))

(defn- run-update-prefix [gf args key {:keys [trace constraints]}]
  (let [pfx (:compiled-prefix-update (:schema gf))
        result (pfx key (vec (:args trace)) constraints (:choices trace))
        replay (cops/make-replay-update-transition (:values result))
        handler-result (rt/run-handler replay
                         {:choices cm/EMPTY :score (:score result)
                          :weight SCORE-ZERO :key key :constraints constraints
                          :old-choices (:choices trace) :discard (cm/from-flat-map (:discard result))
                          :executor execute-sub :param-store (ps gf)}
                         (fn [rt] (run-body gf rt (:args trace))))
        new-trace (make-result-trace gf (:args trace) handler-result)]
    (make-update-result new-trace
      (mx/subtract (:score handler-result) (:score trace))
      (:discard handler-result) constraints (:choices handler-result) handler-result)))

(defn- run-regen-prefix [gf args key {:keys [trace selection]}]
  (let [pfx (:compiled-prefix-regenerate (:schema gf))
        old-score (:score trace)
        result (pfx key (vec (:args trace)) (:choices trace) selection)
        replay (cops/make-replay-regenerate-transition (:values result))
        handler-result (rt/run-handler replay
                         {:choices cm/EMPTY :score (:score result)
                          :weight (:weight result) :key key :selection selection
                          :old-choices (:choices trace)
                          :executor execute-sub :param-store (ps gf)}
                         (fn [rt] (run-body gf rt (:args trace))))]
    (make-regen-result gf trace handler-result old-score)))

(defn- run-assess-prefix [gf args key {:keys [constraints]}]
  (let [pfx (:compiled-prefix-assess (:schema gf))
        result (pfx (vec args) constraints)
        replay (cops/make-replay-assess-transition (:values result))
        handler-result (rt/run-handler replay
                         {:choices cm/EMPTY :score (:score result)
                          :weight (:score result) :key key :constraints constraints
                          :executor execute-sub-assess :param-store (ps gf)}
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
                          :executor execute-sub-project :param-store (ps gf)}
                         (fn [rt] (run-body gf rt (:args trace))))]
    (:weight handler-result)))

;; -- L3: Analytical path --

(defn- run-generate-analytical [gf args key {:keys [constraints]}]
  (let [schema (:schema gf)
        transition (auto-analytical/make-address-dispatch
                     h/generate-transition (:auto-handlers schema))
        result (rt/run-handler transition
                 {:choices cm/EMPTY :score SCORE-ZERO :weight SCORE-ZERO
                  :key key :constraints constraints
                  :auto-posteriors {} :auto-kalman-beliefs {} :auto-kalman-noise-vars {}
                  :executor execute-sub :param-store (ps gf)}
                 (fn [rt] (run-body gf rt args)))
        trace (vary-meta (make-result-trace gf args result) assoc ::score-type :marginal)]
    (make-generate-result trace (:weight result) constraints (:choices result) result)))

(defn- run-assess-analytical [gf args key {:keys [constraints]}]
  (let [schema (:schema gf)
        transition (auto-analytical/make-address-dispatch
                     h/assess-transition (:auto-handlers schema))
        result (rt/run-handler transition
                 {:choices cm/EMPTY :score SCORE-ZERO :weight SCORE-ZERO
                  :key key :constraints constraints
                  :auto-posteriors {} :auto-kalman-beliefs {} :auto-kalman-noise-vars {}
                  :executor execute-sub-assess :param-store (ps gf)}
                 (fn [rt] (run-body gf rt args)))]
    {:retval (:retval result) :weight (:score result)}))

(defn- run-regen-analytical [gf args key {:keys [trace selection]}]
  (let [schema (:schema gf)
        old-score (:score trace)
        result (rt/run-handler (:auto-regenerate-transition schema)
                 {:choices cm/EMPTY :score SCORE-ZERO :weight SCORE-ZERO
                  :key key :selection selection
                  :old-choices (:choices trace)
                  :auto-posteriors {} :auto-kalman-beliefs {} :auto-kalman-noise-vars {}
                  :old-splice-scores (::splice-scores (meta trace))
                  :executor execute-sub :param-store (ps gf)}
                 (fn [rt] (run-body gf rt (:args trace))))
        regen-result (make-regen-result gf trace result old-score)]
    (clojure.core/update regen-result :trace vary-meta assoc ::score-type :marginal)))

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
          deleted (clojure.set/difference old-keys new-keys)]
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

(def ^:private handler-table
  {:simulate run-simulate-handler, :generate run-generate-handler
   :update run-update-handler,     :regenerate run-regen-handler
   :assess run-assess-handler,     :project run-project-handler
   :propose run-propose-handler})

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
   :regenerate run-regen-analytical})

(def ^:private compiled-keys
  {:simulate :compiled-simulate,   :generate :compiled-generate
   :update :compiled-update,       :regenerate :compiled-regenerate
   :assess :compiled-assess,       :project :compiled-project})

(def ^:private prefix-keys
  {:simulate :compiled-prefix,              :generate :compiled-prefix-generate
   :update :compiled-prefix-update,         :regenerate :compiled-prefix-regenerate
   :assess :compiled-prefix-assess,         :project :compiled-prefix-project})

(def ^:private custom-transition-dispatcher
  (reify dispatch/IDispatcher
    (resolve-transition [_ op schema opts]
      (when-let [t (::dispatch/custom-transition (meta (:gf opts)))]
        {:run (fn [gf args key opts] (t op gf args key opts))
         :score-type (or (:score-type (meta t)) :joint)}))))

(def ^:private analytical-dispatcher
  (reify dispatch/IDispatcher
    (resolve-transition [_ op schema opts]
      (when-let [run-fn (get analytical-table op)]
        (case op
          (:generate :assess)
          (when (and (not (mx/in-grad?))
                     (:auto-handlers schema)
                     (auto-analytical/some-conjugate-obs-constrained?
                       (:conjugate-pairs schema) (:constraints opts)))
            {:run run-fn :score-type :marginal})

          :regenerate
          (when (and (:auto-regenerate-transition schema)
                     (= :marginal (::score-type (meta (:trace opts)))))
            {:run run-fn :score-type :marginal})

          nil)))))

(def ^:private compiled-dispatcher
  (reify dispatch/IDispatcher
    (resolve-transition [_ op schema opts]
      (cond
        (get schema (get compiled-keys op))
        {:run (get compiled-table op) :score-type :joint}

        (get schema (get prefix-keys op))
        {:run (get prefix-table op) :score-type :joint}))))

(def ^:private handler-dispatcher
  (reify dispatch/IDispatcher
    (resolve-transition [_ op _ _]
      {:run (get handler-table op) :score-type :joint})))

(def ^:private default-dispatcher-stack
  [custom-transition-dispatcher
   analytical-dispatcher
   compiled-dispatcher
   handler-dispatcher])

(defn- run-dispatched [gf op args key opts]
  (let [spec (dispatch/resolve default-dispatcher-stack op (:schema gf)
               (assoc opts :gf gf))
        result ((:run spec) gf args key opts)]
    result))

;; ---------------------------------------------------------------------------
;; DynamicGF record — GFI protocol implementations
;; ---------------------------------------------------------------------------

(defrecord DynamicGF [body-fn source schema]
  p/IGenerativeFunction
  (simulate [this args]
    (let [key (ensure-key this)
          _ (rng/seed! key)
          result (run-dispatched this :simulate args key {})]
      (mx/gfi-cleanup!)
      (schemas/validated schemas/SimulateReturn result "DynamicGF/simulate")
      result))

  p/IGenerate
  (generate [this args constraints]
    (let [key (ensure-key this)
          _ (rng/seed! key)
          result (run-dispatched this :generate args key {:constraints constraints})]
      (mx/gfi-cleanup!)
      (schemas/validated schemas/GenerateReturn result "DynamicGF/generate")
      result))

  p/IUpdate
  (update [this trace constraints]
    (let [key (ensure-key this)
          _ (rng/seed! key)
          result (run-dispatched this :update (:args trace) key
                   {:trace trace :constraints constraints})]
      ;; Post-process: add addresses deleted by branch switches to discard.
      (mx/gfi-cleanup!)
      (let [final (clojure.core/update result :discard
                    add-deleted-to-discard (:choices trace) (:choices (:trace result)))]
        (schemas/validated schemas/UpdateReturn final "DynamicGF/update")
        final)))

  p/IRegenerate
  (regenerate [this trace selection]
    (let [key (ensure-key this)
          _ (rng/seed! key)
          result (run-dispatched this :regenerate (:args trace) key
                   {:trace trace :selection selection})]
      (mx/gfi-cleanup!)
      (schemas/validated schemas/RegenerateReturn result "DynamicGF/regenerate")
      result))

  p/IAssess
  (assess [this args choices]
    (let [key (ensure-key this)
          _ (rng/seed! key)
          result (run-dispatched this :assess args key {:constraints choices})]
      (mx/gfi-cleanup!)
      (schemas/validated schemas/AssessReturn result "DynamicGF/assess")
      result))

  p/IPropose
  (propose [this args]
    (let [key (ensure-key this)
          _ (rng/seed! key)
          result (run-dispatched this :propose args key {})]
      (mx/gfi-cleanup!)
      (schemas/validated schemas/ProposeReturn result "DynamicGF/propose")
      result))

  p/IProject
  (project [this trace selection]
    (let [key (ensure-key this)
          _ (rng/seed! key)
          result (run-dispatched this :project (:args trace) key
                   {:trace trace :selection selection})]
      (mx/gfi-cleanup!)
      (schemas/validated schemas/ProjectReturn result "DynamicGF/project")
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
;; Vectorized execution (batched: N particles in one model run)
;; These bypass the dispatcher stack by design — vectorized execution always
;; uses batched handler transitions with [N]-shaped arrays via MLX broadcasting.
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
                                :param-store (ps gf)}
                               (fn [rt] (run-body gf rt args)))]
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
                                :param-store (ps gf)}
                               (fn [rt] (run-body gf rt args)))]
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
                                :param-store (ps gf)}
                               (fn [rt] (apply (bfn gf) rt (:args vtrace))))]
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
                                :param-store (ps gf)}
                               (fn [rt] (apply (bfn gf) rt (:args vtrace))))
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
