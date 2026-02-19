(ns genmlx.dynamic
  "DynamicDSLFunction — implements the full GFI by delegating to
   handler-based execution."
  (:require [genmlx.protocols :as p]
            [genmlx.handler :as h]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.vectorized :as vec]))

;; Forward declaration
(declare execute-sub)

(defrecord DynamicGF [body-fn source]
  p/IGenerativeFunction
  (simulate [this args]
    (let [key (rng/fresh-key)
          result (h/run-handler h/simulate-handler
                   {:choices cm/EMPTY :score (mx/scalar 0.0) :key key
                    :executor execute-sub}
                   #(apply body-fn args))]
      (tr/make-trace
        {:gen-fn this :args args
         :choices (:choices result)
         :retval  (:retval result)
         :score   (:score result)})))

  p/IGenerate
  (generate [this args constraints]
    (let [key (rng/fresh-key)
          result (h/run-handler h/generate-handler
                   {:choices cm/EMPTY :score (mx/scalar 0.0)
                    :weight (mx/scalar 0.0)
                    :key key :constraints constraints
                    :executor execute-sub}
                   #(apply body-fn args))]
      {:trace (tr/make-trace
                {:gen-fn this :args args
                 :choices (:choices result)
                 :retval  (:retval result)
                 :score   (:score result)})
       :weight (:weight result)}))

  p/IUpdate
  (update [this trace constraints]
    (let [key (rng/fresh-key)
          result (h/run-handler h/update-handler
                   {:choices cm/EMPTY :score (mx/scalar 0.0)
                    :weight (mx/scalar 0.0)
                    :key key :constraints constraints
                    :old-choices (:choices trace)
                    :discard cm/EMPTY
                    :executor execute-sub}
                   #(apply body-fn (:args trace)))]
      {:trace (tr/make-trace
                {:gen-fn this :args (:args trace)
                 :choices (:choices result)
                 :retval  (:retval result)
                 :score   (:score result)})
       :weight  (:weight result)
       :discard (:discard result)}))

  p/IRegenerate
  (regenerate [this trace selection]
    (let [key (rng/fresh-key)
          old-score (:score trace)
          result (h/run-handler h/regenerate-handler
                   {:choices cm/EMPTY :score (mx/scalar 0.0)
                    :weight (mx/scalar 0.0)  ;; tracks proposal ratio
                    :key key :selection selection
                    :old-choices (:choices trace)
                    :executor execute-sub}
                   #(apply body-fn (:args trace)))
          new-score (:score result)
          proposal-ratio (:weight result)
          ;; Gen.jl regenerate weight = new_score - old_score - proposal_ratio
          weight (mx/subtract (mx/subtract new-score old-score) proposal-ratio)]
      {:trace (tr/make-trace
                {:gen-fn this :args (:args trace)
                 :choices (:choices result)
                 :retval  (:retval result)
                 :score   new-score})
       :weight weight}))

  p/IAssess
  (assess [this args choices]
    (let [key (rng/fresh-key)
          result (h/run-handler h/generate-handler
                   {:choices cm/EMPTY :score (mx/scalar 0.0)
                    :weight (mx/scalar 0.0)
                    :key key :constraints choices
                    :executor execute-sub}
                   #(apply body-fn args))]
      {:retval (:retval result)
       :weight (:weight result)}))

  p/IPropose
  (propose [this args]
    (let [key (rng/fresh-key)
          result (h/run-handler h/simulate-handler
                   {:choices cm/EMPTY :score (mx/scalar 0.0) :key key
                    :executor execute-sub}
                   #(apply body-fn args))]
      {:choices (:choices result)
       :weight  (:score result)
       :retval  (:retval result)}))

)

(defn- execute-sub
  "Execute a sub-generative-function during handler execution.
   Delegates to the sub-gf's own GFI methods."
  [gf args {:keys [constraints old-choices selection key]}]
  (cond
    ;; Regenerate mode
    selection
    (let [{:keys [trace weight]}
          (p/regenerate gf
            (tr/make-trace {:gen-fn gf :args args
                            :choices (or old-choices cm/EMPTY)
                            :retval nil :score (mx/scalar 0.0)})
            selection)]
      {:choices (:choices trace) :retval (:retval trace)
       :score (:score trace) :weight weight})

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

(defn make-gen-fn
  "Create a DynamicGF from a body function and its source form."
  [body-fn source]
  (->DynamicGF body-fn source))

(defn call
  "Call a generative function as a regular function (simulate and return value)."
  [gf & args]
  (:retval (p/simulate gf (vec args))))

;; ---------------------------------------------------------------------------
;; User-facing functions called inside gen bodies
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
                 {:choices cm/EMPTY :score (mx/scalar 0.0)
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
                 {:choices cm/EMPTY :score (mx/scalar 0.0)
                  :weight (mx/scalar 0.0) :key key
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
                 {:choices cm/EMPTY :score (mx/scalar 0.0)
                  :weight (mx/scalar 0.0) :key key
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
                 {:choices cm/EMPTY :score (mx/scalar 0.0)
                  :weight (mx/scalar 0.0) :key key
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
