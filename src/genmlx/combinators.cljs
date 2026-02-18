(ns genmlx.combinators
  "GFI combinators: Map, Unfold, Switch.
   These compose generative functions into higher-level models."
  (:require [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.selection :as sel]))

;; ---------------------------------------------------------------------------
;; Shared helpers for MapCombinator
;; ---------------------------------------------------------------------------

(defn- assemble-choices
  "Reduce indexed results into a choice map, extracting choices via `choices-fn`."
  [results choices-fn]
  (reduce (fn [cm [i r]]
            (cm/set-choice cm [i] (choices-fn r)))
          cm/EMPTY
          (map-indexed vector results)))

(defn- sum-field
  "Sum a field across results, starting from scalar 0.0."
  [results field-fn]
  (reduce (fn [acc r] (mx/add acc (field-fn r)))
          (mx/scalar 0.0)
          results))

;; ---------------------------------------------------------------------------
;; Map Combinator
;; ---------------------------------------------------------------------------
;; Applies a generative function independently to each element of input sequences.
;; Like Gen.jl's Map combinator.

(defrecord MapCombinator [kernel]
  p/IGenerativeFunction
  (simulate [this args]
    (let [n (count (first args))
          results (mapv (fn [i]
                          (p/simulate kernel (mapv #(nth % i) args)))
                        (range n))
          choices (assemble-choices results :choices)
          retvals (mapv :retval results)
          score (sum-field results :score)]
      (tr/make-trace {:gen-fn this :args args
                      :choices choices :retval retvals :score score})))

  p/IGenerate
  (generate [this args constraints]
    (let [n (count (first args))
          results (mapv (fn [i]
                          (p/generate kernel (mapv #(nth % i) args)
                                      (cm/get-submap constraints i)))
                        (range n))
          choices (assemble-choices results (comp :choices :trace))
          retvals (mapv (comp :retval :trace) results)
          score (sum-field results (comp :score :trace))
          weight (sum-field results :weight)]
      {:trace (tr/make-trace {:gen-fn this :args args
                              :choices choices :retval retvals :score score})
       :weight weight}))

  p/IUpdate
  (update [this trace constraints]
    (let [old-choices (:choices trace)
          args (:args trace)
          n (count (first args))
          results (mapv (fn [i]
                          (let [kernel-args (mapv #(nth % i) args)
                                old-trace (tr/make-trace
                                            {:gen-fn kernel :args kernel-args
                                             :choices (cm/get-submap old-choices i)
                                             :retval nil :score (mx/scalar 0.0)})]
                            (p/update kernel old-trace (cm/get-submap constraints i))))
                        (range n))
          choices (assemble-choices results (comp :choices :trace))
          retvals (mapv (comp :retval :trace) results)
          score (sum-field results (comp :score :trace))
          weight (sum-field results :weight)
          discard (assemble-choices
                    (filter :discard results)
                    :discard)]
      {:trace (tr/make-trace {:gen-fn this :args args
                              :choices choices :retval retvals :score score})
       :weight weight :discard discard}))

  p/IRegenerate
  (regenerate [this trace selection]
    (let [old-choices (:choices trace)
          args (:args trace)
          n (count (first args))
          results (mapv (fn [i]
                          (let [kernel-args (mapv #(nth % i) args)
                                old-trace (tr/make-trace
                                            {:gen-fn kernel :args kernel-args
                                             :choices (cm/get-submap old-choices i)
                                             :retval nil :score (mx/scalar 0.0)})]
                            (p/regenerate kernel old-trace
                                          (sel/get-subselection selection i))))
                        (range n))
          choices (assemble-choices results (comp :choices :trace))
          retvals (mapv (comp :retval :trace) results)
          score (sum-field results (comp :score :trace))
          weight (sum-field results :weight)]
      {:trace (tr/make-trace {:gen-fn this :args args
                              :choices choices :retval retvals :score score})
       :weight weight})))

(defn map-combinator
  "Create a Map combinator from a kernel generative function.
   The resulting GF applies the kernel independently to each element."
  [kernel]
  (->MapCombinator kernel))

;; ---------------------------------------------------------------------------
;; Unfold Combinator
;; ---------------------------------------------------------------------------
;; Sequential application — each step depends on the previous state.
;; Like Gen.jl's Unfold combinator for time-series models.

(defrecord UnfoldCombinator [kernel]
  p/IGenerativeFunction
  (simulate [this args]
    ;; args: [n init-state & extra-args]
    ;; kernel takes [t state & extra-args] and returns new-state
    (let [[n init-state & extra] args]
      (loop [t 0 state init-state
             choices cm/EMPTY score (mx/scalar 0.0)
             states []]
        (if (>= t n)
          (tr/make-trace {:gen-fn this :args args
                          :choices choices :retval states :score score})
          (let [trace (p/simulate kernel (into [t state] extra))
                new-state (:retval trace)]
            (recur (inc t)
                   new-state
                   (cm/set-choice choices [t] (:choices trace))
                   (mx/add score (:score trace))
                   (conj states new-state)))))))

  p/IGenerate
  (generate [this args constraints]
    (let [[n init-state & extra] args]
      (loop [t 0 state init-state
             choices cm/EMPTY score (mx/scalar 0.0) weight (mx/scalar 0.0)
             states []]
        (if (>= t n)
          {:trace (tr/make-trace {:gen-fn this :args args
                                  :choices choices :retval states :score score})
           :weight weight}
          (let [result (p/generate kernel (into [t state] extra)
                                   (cm/get-submap constraints t))
                trace (:trace result)
                new-state (:retval trace)]
            (recur (inc t)
                   new-state
                   (cm/set-choice choices [t] (:choices trace))
                   (mx/add score (:score trace))
                   (mx/add weight (:weight result))
                   (conj states new-state))))))))

(defn unfold-combinator
  "Create an Unfold combinator from a kernel generative function.
   The kernel takes [t state & extra-args] and returns new-state."
  [kernel]
  (->UnfoldCombinator kernel))

;; ---------------------------------------------------------------------------
;; Switch Combinator
;; ---------------------------------------------------------------------------
;; Selects between multiple generative functions based on an index.
;; Like Gen.jl's Switch combinator for mixture models.

(defrecord SwitchCombinator [branches]
  p/IGenerativeFunction
  (simulate [this args]
    ;; args: [index & branch-args]
    (let [[idx & branch-args] args
          trace (p/simulate (nth branches idx) (vec branch-args))]
      (tr/make-trace {:gen-fn this :args args
                      :choices (:choices trace)
                      :retval (:retval trace)
                      :score (:score trace)})))

  p/IGenerate
  (generate [this args constraints]
    (let [[idx & branch-args] args
          {:keys [trace weight]} (p/generate (nth branches idx) (vec branch-args) constraints)]
      {:trace (tr/make-trace {:gen-fn this :args args
                              :choices (:choices trace)
                              :retval (:retval trace)
                              :score (:score trace)})
       :weight weight})))

(defn switch-combinator
  "Create a Switch combinator from a vector of branch generative functions.
   The first argument selects which branch to execute."
  [& branches]
  (->SwitchCombinator (vec branches)))
