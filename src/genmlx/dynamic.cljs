(ns genmlx.dynamic
  "DynamicDSLFunction — implements the full GFI by delegating to
   handler-based execution."
  (:require [genmlx.protocols :as p]
            [genmlx.handler :as h]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.selection :as sel]
            [genmlx.vectorized :as vec]
            [genmlx.edit :as edit]
            [genmlx.diff :as diff]
            [clojure.set]))

;; Cached zero constant for init states (MLX scalars are immutable)
(def ^:private SCORE-ZERO (mx/scalar 0.0))

;; Forward declarations
(declare execute-sub)
(declare execute-sub-project)
(declare execute-sub-assess)

(defn- warn-unused-constraints
  "Warn if any top-level constraint keys were not consumed by the trace."
  [op-name constraints result-choices]
  (when (and constraints
             (not= constraints cm/EMPTY)
             (instance? cm/Node constraints))
    (let [constraint-keys (set (keys (:m constraints)))
          trace-keys (when (instance? cm/Node result-choices)
                       (set (keys (:m result-choices))))
          unused (clojure.set/difference constraint-keys (or trace-keys #{}))]
      (when (seq unused)
        (js/console.warn
          (str op-name ": constraint address(es) not found in trace: "
               (vec (sort unused)) ". "
               "Trace addresses: " (vec (sort (or trace-keys []))) ". "
               "Unused constraints are ignored."))))))

(defrecord DynamicGF [body-fn source]
  p/IGenerativeFunction
  (simulate [this args]
    (let [key (rng/next-key)
          result (h/run-handler h/simulate-handler
                   {:choices cm/EMPTY :score SCORE-ZERO :key key
                    :executor execute-sub}
                   #(apply body-fn args))
          trace (tr/make-trace
                  {:gen-fn this :args args
                   :choices (:choices result)
                   :retval  (:retval result)
                   :score   (:score result)})]
      (if-let [ss (:splice-scores result)]
        (with-meta trace {::splice-scores ss})
        trace)))

  p/IGenerate
  (generate [this args constraints]
    (let [key (rng/next-key)
          result (h/run-handler h/generate-handler
                   {:choices cm/EMPTY :score SCORE-ZERO
                    :weight SCORE-ZERO
                    :key key :constraints constraints
                    :executor execute-sub}
                   #(apply body-fn args))
          trace (tr/make-trace
                  {:gen-fn this :args args
                   :choices (:choices result)
                   :retval  (:retval result)
                   :score   (:score result)})]
      (warn-unused-constraints "generate" constraints (:choices result))
      {:trace (if-let [ss (:splice-scores result)]
                (with-meta trace {::splice-scores ss})
                trace)
       :weight (:weight result)}))

  p/IUpdate
  (update [this trace constraints]
    (let [key (rng/next-key)
          result (h/run-handler h/update-handler
                   {:choices cm/EMPTY :score SCORE-ZERO
                    :weight SCORE-ZERO
                    :key key :constraints constraints
                    :old-choices (:choices trace)
                    :old-splice-scores (::splice-scores (meta trace))
                    :discard cm/EMPTY
                    :executor execute-sub}
                   #(apply body-fn (:args trace)))
          new-trace (tr/make-trace
                      {:gen-fn this :args (:args trace)
                       :choices (:choices result)
                       :retval  (:retval result)
                       :score   (:score result)})]
      (warn-unused-constraints "update" constraints (:choices result))
      {:trace (if-let [ss (:splice-scores result)]
                (with-meta new-trace {::splice-scores ss})
                new-trace)
       :weight  (mx/subtract (:score result) (:score trace))
       :discard (:discard result)}))

  p/IRegenerate
  (regenerate [this trace selection]
    (let [key (rng/next-key)
          old-score (:score trace)
          result (h/run-handler h/regenerate-handler
                   {:choices cm/EMPTY :score SCORE-ZERO
                    :weight SCORE-ZERO  ;; tracks proposal ratio
                    :key key :selection selection
                    :old-choices (:choices trace)
                    :old-splice-scores (::splice-scores (meta trace))
                    :executor execute-sub}
                   #(apply body-fn (:args trace)))
          new-score (:score result)
          proposal-ratio (:weight result)
          ;; Gen.jl regenerate weight = new_score - old_score - proposal_ratio
          weight (mx/subtract (mx/subtract new-score old-score) proposal-ratio)
          new-trace (tr/make-trace
                      {:gen-fn this :args (:args trace)
                       :choices (:choices result)
                       :retval  (:retval result)
                       :score   new-score})]
      {:trace (if-let [ss (:splice-scores result)]
                (with-meta new-trace {::splice-scores ss})
                new-trace)
       :weight weight}))

  p/IAssess
  (assess [this args choices]
    (let [key (rng/next-key)
          result (h/run-handler h/assess-handler
                   {:choices cm/EMPTY :score SCORE-ZERO
                    :weight SCORE-ZERO
                    :key key :constraints choices
                    :executor execute-sub-assess}
                   #(apply body-fn args))]
      {:retval (:retval result)
       :weight (:score result)}))

  p/IPropose
  (propose [this args]
    (let [key (rng/next-key)
          result (h/run-handler h/simulate-handler
                   {:choices cm/EMPTY :score SCORE-ZERO :key key
                    :executor execute-sub}
                   #(apply body-fn args))]
      {:choices (:choices result)
       :weight  (:score result)
       :retval  (:retval result)}))

  p/IProject
  (project [this trace selection]
    (let [key (rng/next-key)
          result (h/run-handler h/project-handler
                   {:choices cm/EMPTY :score SCORE-ZERO
                    :weight SCORE-ZERO
                    :key key :selection selection
                    :old-choices (:choices trace)
                    :constraints cm/EMPTY
                    :executor execute-sub-project}
                   #(apply body-fn (:args trace)))]
      (:weight result)))

)

(defn- execute-sub
  "Execute a sub-generative-function during handler execution.
   Delegates to the sub-gf's own GFI methods."
  [gf args {:keys [constraints old-choices selection key old-splice-score]}]
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
       :score (:score trace)})))

(defn- execute-sub-project
  "Execute sub-GF in project mode: replay via generate, then project."
  [gf args {:keys [old-choices selection]}]
  (let [{:keys [trace]} (p/generate gf args (or old-choices cm/EMPTY))
        weight (p/project gf trace (or selection sel/none))]
    {:choices (:choices trace)
     :retval (:retval trace)
     :score (:score trace)
     :weight weight}))

(defn- execute-sub-assess
  "Execute a sub-GF in assess mode: all choices must be provided."
  [gf args {:keys [constraints]}]
  (let [{:keys [retval weight]} (p/assess gf args (or constraints cm/EMPTY))]
    {:choices (or constraints cm/EMPTY) :retval retval
     :score weight :weight weight}))

(defn make-gen-fn
  "Create a DynamicGF from a body function and its source form."
  [body-fn source]
  (->DynamicGF body-fn source))

(defn call
  "Call a generative function as a regular function (simulate and return value)."
  [gf & args]
  (:retval (p/simulate gf (vec args))))

(defn with-key
  "Execute f with a threaded PRNG key for reproducible inference.
   All DynamicGF GFI methods called within f will draw keys from the
   threaded key instead of calling rng/fresh-key."
  [key f]
  (binding [rng/*prng-key* (volatile! key)]
    (f)))

;; ---------------------------------------------------------------------------
;; User-facing effect operations called inside gen bodies
;;
;; These three functions — trace, splice, param — are the COMPLETE set of
;; effectful operations available within a `gen` body.  Everything else in a
;; gen body is pure ClojureScript.  Each dispatches to the active handler
;; (see handler.cljs) which performs the appropriate state transition.
;; ---------------------------------------------------------------------------

(defn trace
  "Sample from or constrain a distribution at the given address.
   Used inside gen bodies: (trace :addr (gaussian 0 10))"
  [addr dist]
  (h/trace-choice! addr dist))

(defn splice
  "Call a sub-generative-function at the given address namespace.
   Used inside gen bodies: (splice :sub-model my-model args...)"
  [addr gf & args]
  (h/trace-gf! addr gf (vec args)))

;; ---------------------------------------------------------------------------
;; Vectorized execution (batched: N particles in one model run)
;; ---------------------------------------------------------------------------

(defn vsimulate
  "Run model body ONCE with batched handler, producing a VectorizedTrace
   with [n]-shaped arrays at each choice site.
   gf: DynamicGF, args: model args, n: number of particles, key: PRNG key."
  [gf args n key]
  (let [key (rng/ensure-key key)
        result (h/run-handler h/batched-simulate-handler
                 {:choices cm/EMPTY :score SCORE-ZERO
                  :key key :batch-size n :batched? true
                  :executor execute-sub}
                 #(apply (:body-fn gf) args))]
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
        result (h/run-handler h/batched-generate-handler
                 {:choices cm/EMPTY :score SCORE-ZERO
                  :weight SCORE-ZERO :key key
                  :constraints constraints :batch-size n :batched? true
                  :executor execute-sub}
                 #(apply (:body-fn gf) args))]
    (vec/->VectorizedTrace gf args (:choices result) (:score result)
                           (:weight result) n (:retval result))))

(defn vupdate
  "Batched update: run model body ONCE with batched update handler.
   vtrace: VectorizedTrace with [n]-shaped choices, constraints: new observations.
   Returns new VectorizedTrace with updated weights."
  [gf vtrace constraints key]
  (let [key (rng/ensure-key key)
        n (:n-particles vtrace)
        result (h/run-handler h/batched-update-handler
                 {:choices cm/EMPTY :score SCORE-ZERO
                  :weight SCORE-ZERO :key key
                  :constraints constraints
                  :old-choices (:choices vtrace)
                  :discard cm/EMPTY
                  :batch-size n :batched? true
                  :executor execute-sub}
                 #(apply (:body-fn gf) (:args vtrace)))]
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
        n (:n-particles vtrace)
        old-score (:score vtrace)
        result (h/run-handler h/batched-regenerate-handler
                 {:choices cm/EMPTY :score SCORE-ZERO
                  :weight SCORE-ZERO :key key
                  :selection selection
                  :old-choices (:choices vtrace)
                  :batch-size n :batched? true
                  :executor execute-sub}
                 #(apply (:body-fn gf) (:args vtrace)))
        new-score (:score result)
        proposal-ratio (:weight result)
        weight (mx/subtract (mx/subtract new-score old-score) proposal-ratio)]
    {:vtrace (vec/->VectorizedTrace gf (:args vtrace) (:choices result)
                                     new-score weight n (:retval result))
     :weight weight}))

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

;; ---------------------------------------------------------------------------
;; Trainable parameter support
;; ---------------------------------------------------------------------------

(defn param
  "Declare/read a trainable parameter inside a gen body.
   name: parameter name (keyword)
   default-value: default parameter value (used when no param store is active)
   Returns the parameter value as an MLX array."
  [name default-value]
  (h/trace-param! name default-value))

;; ---------------------------------------------------------------------------
;; IHasArgumentGrads — DynamicGF does not declare argument differentiability
;; ---------------------------------------------------------------------------

(extend-type DynamicGF
  p/IHasArgumentGrads
  (has-argument-grads [_] nil))
