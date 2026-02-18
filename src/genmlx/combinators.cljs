(ns genmlx.combinators
  "GFI combinators: Map, Unfold, Switch.
   These compose generative functions into higher-level models."
  (:require [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.selection :as sel]))

;; ---------------------------------------------------------------------------
;; Map Combinator
;; ---------------------------------------------------------------------------
;; Applies a generative function independently to each element of input sequences.
;; Like Gen.jl's Map combinator.

(defrecord MapCombinator [kernel]
  p/IGenerativeFunction
  (simulate [this args]
    ;; args is a vector of sequences, one per kernel argument
    ;; e.g., for kernel(x, y): args = [[x1 x2 x3] [y1 y2 y3]]
    (let [n (count (first args))
          results (mapv (fn [i]
                          (let [kernel-args (mapv #(nth % i) args)]
                            (p/simulate kernel kernel-args)))
                        (range n))
          choices (reduce (fn [cm [i trace]]
                            (cm/set-choice cm [i] (tr/get-choices trace)))
                          cm/EMPTY
                          (map-indexed vector results))
          retvals (mapv tr/get-retval results)
          score (reduce (fn [acc trace]
                          (mx/add acc (tr/get-score trace)))
                        (mx/scalar 0.0)
                        results)]
      (tr/make-trace {:gen-fn this :args args
                      :choices choices :retval retvals :score score})))

  p/IGenerate
  (generate [this args constraints]
    (let [n (count (first args))
          results (mapv (fn [i]
                          (let [kernel-args (mapv #(nth % i) args)
                                sub-cm (cm/get-submap constraints i)]
                            (p/generate kernel kernel-args sub-cm)))
                        (range n))
          choices (reduce (fn [cm [i {:keys [trace]}]]
                            (cm/set-choice cm [i] (tr/get-choices trace)))
                          cm/EMPTY
                          (map-indexed vector results))
          retvals (mapv (comp tr/get-retval :trace) results)
          score (reduce (fn [acc {:keys [trace]}]
                          (mx/add acc (tr/get-score trace)))
                        (mx/scalar 0.0) results)
          weight (reduce (fn [acc {:keys [weight]}]
                           (mx/add acc weight))
                         (mx/scalar 0.0) results)]
      {:trace (tr/make-trace {:gen-fn this :args args
                              :choices choices :retval retvals :score score})
       :weight weight}))

  p/IUpdate
  (update [this trace constraints]
    (let [old-choices (tr/get-choices trace)
          args (tr/get-args trace)
          n (count (first args))
          results (mapv (fn [i]
                          (let [kernel-args (mapv #(nth % i) args)
                                sub-cm (cm/get-submap constraints i)
                                old-sub (cm/get-submap old-choices i)
                                old-trace (tr/make-trace
                                            {:gen-fn kernel :args kernel-args
                                             :choices old-sub :retval nil
                                             :score (mx/scalar 0.0)})]
                            (p/update kernel old-trace sub-cm)))
                        (range n))
          choices (reduce (fn [cm [i {:keys [trace]}]]
                            (cm/set-choice cm [i] (tr/get-choices trace)))
                          cm/EMPTY
                          (map-indexed vector results))
          retvals (mapv (comp tr/get-retval :trace) results)
          score (reduce (fn [acc {:keys [trace]}]
                          (mx/add acc (tr/get-score trace)))
                        (mx/scalar 0.0) results)
          weight (reduce (fn [acc {:keys [weight]}]
                           (mx/add acc weight))
                         (mx/scalar 0.0) results)
          discard (reduce (fn [cm [i {:keys [discard]}]]
                            (if discard
                              (cm/set-choice cm [i] discard)
                              cm))
                          cm/EMPTY
                          (map-indexed vector results))]
      {:trace (tr/make-trace {:gen-fn this :args args
                              :choices choices :retval retvals :score score})
       :weight weight :discard discard}))

  p/IRegenerate
  (regenerate [this trace selection]
    (let [old-choices (tr/get-choices trace)
          args (tr/get-args trace)
          n (count (first args))
          results (mapv (fn [i]
                          (let [kernel-args (mapv #(nth % i) args)
                                old-sub (cm/get-submap old-choices i)
                                sub-sel (sel/get-subselection selection i)
                                old-trace (tr/make-trace
                                            {:gen-fn kernel :args kernel-args
                                             :choices old-sub :retval nil
                                             :score (mx/scalar 0.0)})]
                            (p/regenerate kernel old-trace sub-sel)))
                        (range n))
          choices (reduce (fn [cm [i {:keys [trace]}]]
                            (cm/set-choice cm [i] (tr/get-choices trace)))
                          cm/EMPTY
                          (map-indexed vector results))
          retvals (mapv (comp tr/get-retval :trace) results)
          score (reduce (fn [acc {:keys [trace]}]
                          (mx/add acc (tr/get-score trace)))
                        (mx/scalar 0.0) results)
          weight (reduce (fn [acc {:keys [weight]}]
                           (mx/add acc weight))
                         (mx/scalar 0.0) results)]
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
          (let [kernel-args (into [t state] extra)
                trace (p/simulate kernel kernel-args)
                new-state (tr/get-retval trace)]
            (recur (inc t)
                   new-state
                   (cm/set-choice choices [t] (tr/get-choices trace))
                   (mx/add score (tr/get-score trace))
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
          (let [kernel-args (into [t state] extra)
                sub-cm (cm/get-submap constraints t)
                {:keys [trace weight] :as result}
                  (p/generate kernel kernel-args sub-cm)
                new-state (tr/get-retval trace)]
            (recur (inc t)
                   new-state
                   (cm/set-choice choices [t] (tr/get-choices trace))
                   (mx/add score (tr/get-score trace))
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
          branch (nth branches idx)]
      (let [trace (p/simulate branch (vec branch-args))]
        (tr/make-trace {:gen-fn this :args args
                        :choices (tr/get-choices trace)
                        :retval (tr/get-retval trace)
                        :score (tr/get-score trace)}))))

  p/IGenerate
  (generate [this args constraints]
    (let [[idx & branch-args] args
          branch (nth branches idx)
          {:keys [trace weight]} (p/generate branch (vec branch-args) constraints)]
      {:trace (tr/make-trace {:gen-fn this :args args
                              :choices (tr/get-choices trace)
                              :retval (tr/get-retval trace)
                              :score (tr/get-score trace)})
       :weight weight})))

(defn switch-combinator
  "Create a Switch combinator from a vector of branch generative functions.
   The first argument selects which branch to execute."
  [& branches]
  (->SwitchCombinator (vec branches)))
