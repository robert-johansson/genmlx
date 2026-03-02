(ns genmlx.dynamic
  "DynamicDSLFunction — implements the full GFI by delegating to
   handler-based execution."
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
            [clojure.set]))

;; Cached zero constant for init states (MLX scalars are immutable)
(def ^:private SCORE-ZERO (mx/scalar 0.0))

;; Forward declarations
(declare execute-sub)
(declare execute-sub-project)
(declare execute-sub-assess)

;; Sentinel value for auto-key: generates a fresh key per GFI call
(def ^:private auto-key-sentinel ::auto-key)

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

(defrecord DynamicGF [body-fn source]
  p/IGenerativeFunction
  (simulate [this args]
    (let [key (let [k (::key (meta this))]
              (cond
                (= k auto-key-sentinel) (rng/fresh-key)
                k k
                :else (throw (ex-info "No PRNG key on gen-fn. Use (dyn/with-key gf key) or (dyn/auto-key gf)."
                                      {:gen-fn (.-source this)}))))
          _ (rng/seed! key)
          result (rt/run-handler h/simulate-transition
                   {:choices cm/EMPTY :score SCORE-ZERO :key key
                    :executor execute-sub
                    :param-store (::param-store (meta this))}
                   (fn [rt] (apply body-fn rt args)))
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
    (let [key (let [k (::key (meta this))]
              (cond
                (= k auto-key-sentinel) (rng/fresh-key)
                k k
                :else (throw (ex-info "No PRNG key on gen-fn. Use (dyn/with-key gf key) or (dyn/auto-key gf)."
                                      {:gen-fn (.-source this)}))))
          _ (rng/seed! key)
          result (rt/run-handler h/generate-transition
                   {:choices cm/EMPTY :score SCORE-ZERO
                    :weight SCORE-ZERO
                    :key key :constraints constraints
                    :executor execute-sub
                    :param-store (::param-store (meta this))}
                   (fn [rt] (apply body-fn rt args)))
          trace (tr/make-trace
                  {:gen-fn this :args args
                   :choices (:choices result)
                   :retval  (:retval result)
                   :score   (:score result)})]
      (let [result-map {:trace (if-let [ss (:splice-scores result)]
                                (with-meta trace {::splice-scores ss})
                                trace)
                        :weight (:weight result)}]
        (if-let [unused (find-unused-constraints constraints (:choices result))]
          (assoc result-map :unused-constraints unused)
          result-map))))

  p/IUpdate
  (update [this trace constraints]
    (let [key (let [k (::key (meta this))]
              (cond
                (= k auto-key-sentinel) (rng/fresh-key)
                k k
                :else (throw (ex-info "No PRNG key on gen-fn. Use (dyn/with-key gf key) or (dyn/auto-key gf)."
                                      {:gen-fn (.-source this)}))))
          _ (rng/seed! key)
          result (rt/run-handler h/update-transition
                   {:choices cm/EMPTY :score SCORE-ZERO
                    :weight SCORE-ZERO
                    :key key :constraints constraints
                    :old-choices (:choices trace)
                    :old-splice-scores (::splice-scores (meta trace))
                    :discard cm/EMPTY
                    :executor execute-sub
                    :param-store (::param-store (meta this))}
                   (fn [rt] (apply body-fn rt (:args trace))))
          new-trace (tr/make-trace
                      {:gen-fn this :args (:args trace)
                       :choices (:choices result)
                       :retval  (:retval result)
                       :score   (:score result)})]
      (let [result-map {:trace (if-let [ss (:splice-scores result)]
                                (with-meta new-trace {::splice-scores ss})
                                new-trace)
                        :weight  (mx/subtract (:score result) (:score trace))
                        :discard (:discard result)}]
        (if-let [unused (find-unused-constraints constraints (:choices result))]
          (assoc result-map :unused-constraints unused)
          result-map))))

  p/IRegenerate
  (regenerate [this trace selection]
    (let [key (let [k (::key (meta this))]
              (cond
                (= k auto-key-sentinel) (rng/fresh-key)
                k k
                :else (throw (ex-info "No PRNG key on gen-fn. Use (dyn/with-key gf key) or (dyn/auto-key gf)."
                                      {:gen-fn (.-source this)}))))
          _ (rng/seed! key)
          old-score (:score trace)
          result (rt/run-handler h/regenerate-transition
                   {:choices cm/EMPTY :score SCORE-ZERO
                    :weight SCORE-ZERO  ;; tracks proposal ratio
                    :key key :selection selection
                    :old-choices (:choices trace)
                    :old-splice-scores (::splice-scores (meta trace))
                    :executor execute-sub
                    :param-store (::param-store (meta this))}
                   (fn [rt] (apply body-fn rt (:args trace))))
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
    (let [key (let [k (::key (meta this))]
              (cond
                (= k auto-key-sentinel) (rng/fresh-key)
                k k
                :else (throw (ex-info "No PRNG key on gen-fn. Use (dyn/with-key gf key) or (dyn/auto-key gf)."
                                      {:gen-fn (.-source this)}))))
          _ (rng/seed! key)
          result (rt/run-handler h/assess-transition
                   {:choices cm/EMPTY :score SCORE-ZERO
                    :weight SCORE-ZERO
                    :key key :constraints choices
                    :executor execute-sub-assess
                    :param-store (::param-store (meta this))}
                   (fn [rt] (apply body-fn rt args)))]
      {:retval (:retval result)
       :weight (:score result)}))

  p/IPropose
  (propose [this args]
    (let [key (let [k (::key (meta this))]
              (cond
                (= k auto-key-sentinel) (rng/fresh-key)
                k k
                :else (throw (ex-info "No PRNG key on gen-fn. Use (dyn/with-key gf key) or (dyn/auto-key gf)."
                                      {:gen-fn (.-source this)}))))
          _ (rng/seed! key)
          result (rt/run-handler h/simulate-transition
                   {:choices cm/EMPTY :score SCORE-ZERO :key key
                    :executor execute-sub
                    :param-store (::param-store (meta this))}
                   (fn [rt] (apply body-fn rt args)))]
      {:choices (:choices result)
       :weight  (:score result)
       :retval  (:retval result)}))

  p/IProject
  (project [this trace selection]
    (let [key (let [k (::key (meta this))]
              (cond
                (= k auto-key-sentinel) (rng/fresh-key)
                k k
                :else (throw (ex-info "No PRNG key on gen-fn. Use (dyn/with-key gf key) or (dyn/auto-key gf)."
                                      {:gen-fn (.-source this)}))))
          _ (rng/seed! key)
          result (rt/run-handler h/project-transition
                   {:choices cm/EMPTY :score SCORE-ZERO
                    :weight SCORE-ZERO
                    :key key :selection selection
                    :old-choices (:choices trace)
                    :constraints cm/EMPTY
                    :executor execute-sub-project
                    :param-store (::param-store (meta this))}
                   (fn [rt] (apply body-fn rt (:args trace))))]
      (:weight result)))

)

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

(defn make-gen-fn
  "Create a DynamicGF from a body function and its source form."
  [body-fn source]
  (->DynamicGF body-fn source))

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
