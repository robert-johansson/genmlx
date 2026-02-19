(ns genmlx.combinators
  "GFI combinators: Map, Unfold, Switch, Scan, Mask, Mix, and more.
   These compose generative functions into higher-level models."
  (:require [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.selection :as sel]
            [genmlx.dist.core :as dc]
            [genmlx.edit :as edit]
            [genmlx.diff :as diff]))

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
          score (sum-field results :score)
          element-scores (mapv :score results)]
      (with-meta
        (tr/make-trace {:gen-fn this :args args
                        :choices choices :retval retvals :score score})
        {::element-scores element-scores})))

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
          weight (sum-field results :weight)
          element-scores (mapv (comp :score :trace) results)]
      {:trace (with-meta
                (tr/make-trace {:gen-fn this :args args
                                :choices choices :retval retvals :score score})
                {::element-scores element-scores})
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
          weight (mx/subtract score (:score trace))
          discard (assemble-choices
                    (filter :discard results)
                    :discard)
          element-scores (mapv (comp :score :trace) results)]
      {:trace (with-meta
                (tr/make-trace {:gen-fn this :args args
                                :choices choices :retval retvals :score score})
                {::element-scores element-scores})
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
          weight (mx/subtract (sum-field results :weight) (:score trace))
          element-scores (mapv (comp :score :trace) results)]
      {:trace (with-meta
                (tr/make-trace {:gen-fn this :args args
                                :choices choices :retval retvals :score score})
                {::element-scores element-scores})
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

(extend-type UnfoldCombinator
  p/IUpdate
  (update [this trace constraints]
    (let [kern (:kernel this)
          {:keys [args choices]} trace
          [n init-state & extra] args]
      (loop [t 0 state init-state
             new-choices cm/EMPTY score (mx/scalar 0.0)
             discard cm/EMPTY states []]
        (if (>= t n)
          {:trace (tr/make-trace {:gen-fn this :args args
                                  :choices new-choices :retval states :score score})
           :weight (mx/subtract score (:score trace)) :discard discard}
          (let [old-sub-choices (cm/get-submap choices t)
                kernel-args (into [t state] extra)
                old-trace (tr/make-trace
                            {:gen-fn kern :args kernel-args
                             :choices old-sub-choices
                             :retval nil :score (mx/scalar 0.0)})
                result (p/update kern old-trace (cm/get-submap constraints t))
                new-trace (:trace result)
                new-state (:retval new-trace)]
            (recur (inc t)
                   new-state
                   (cm/set-choice new-choices [t] (:choices new-trace))
                   (mx/add score (:score new-trace))
                   (if (:discard result)
                     (cm/set-choice discard [t] (:discard result))
                     discard)
                   (conj states new-state)))))))

  p/IRegenerate
  (regenerate [this trace selection]
    (let [kern (:kernel this)
          {:keys [args choices]} trace
          [n init-state & extra] args]
      (loop [t 0 state init-state
             new-choices cm/EMPTY score (mx/scalar 0.0) weight (mx/scalar 0.0)
             states []]
        (if (>= t n)
          {:trace (tr/make-trace {:gen-fn this :args args
                                  :choices new-choices :retval states :score score})
           :weight (mx/subtract weight (:score trace))}
          (let [old-sub-choices (cm/get-submap choices t)
                kernel-args (into [t state] extra)
                old-trace (tr/make-trace
                            {:gen-fn kern :args kernel-args
                             :choices old-sub-choices
                             :retval nil :score (mx/scalar 0.0)})
                result (p/regenerate kern old-trace
                                     (sel/get-subselection selection t))
                new-trace (:trace result)
                new-state (:retval new-trace)]
            (recur (inc t)
                   new-state
                   (cm/set-choice new-choices [t] (:choices new-trace))
                   (mx/add score (:score new-trace))
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

(extend-type SwitchCombinator
  p/IUpdate
  (update [this trace constraints]
    (let [orig-args (:args trace)
          [idx & branch-args] orig-args
          branch (nth (:branches this) idx)
          old-branch-trace (tr/make-trace
                             {:gen-fn branch :args (vec branch-args)
                              :choices (:choices trace)
                              :retval (:retval trace) :score (:score trace)})
          result (p/update branch old-branch-trace constraints)
          new-branch-trace (:trace result)]
      {:trace (tr/make-trace {:gen-fn this :args orig-args
                              :choices (:choices new-branch-trace)
                              :retval (:retval new-branch-trace)
                              :score (:score new-branch-trace)})
       :weight (:weight result) :discard (:discard result)}))

  p/IRegenerate
  (regenerate [this trace selection]
    (let [orig-args (:args trace)
          [idx & branch-args] orig-args
          branch (nth (:branches this) idx)
          old-branch-trace (tr/make-trace
                             {:gen-fn branch :args (vec branch-args)
                              :choices (:choices trace)
                              :retval (:retval trace) :score (:score trace)})
          result (p/regenerate branch old-branch-trace selection)
          new-branch-trace (:trace result)]
      {:trace (tr/make-trace {:gen-fn this :args orig-args
                              :choices (:choices new-branch-trace)
                              :retval (:retval new-branch-trace)
                              :score (:score new-branch-trace)})
       :weight (:weight result)})))

(defn switch-combinator
  "Create a Switch combinator from a vector of branch generative functions.
   The first argument selects which branch to execute."
  [& branches]
  (->SwitchCombinator (vec branches)))

;; ---------------------------------------------------------------------------
;; Mask Combinator
;; ---------------------------------------------------------------------------
;; Gates execution of a generative function on a boolean condition.
;; When masked (condition = false), the GF is not executed and contributes
;; zero score. Used by VectorizedSwitch to implement all-branch execution.

(defrecord MaskCombinator [inner]
  p/IGenerativeFunction
  (simulate [this args]
    ;; args: [active? & inner-args] where active? is boolean
    (let [[active? & inner-args] args]
      (if active?
        (let [trace (p/simulate inner (vec inner-args))]
          (tr/make-trace {:gen-fn this :args args
                          :choices (:choices trace)
                          :retval (:retval trace)
                          :score (:score trace)}))
        (tr/make-trace {:gen-fn this :args args
                        :choices cm/EMPTY
                        :retval nil
                        :score (mx/scalar 0.0)}))))

  p/IGenerate
  (generate [this args constraints]
    (let [[active? & inner-args] args]
      (if active?
        (let [{:keys [trace weight]} (p/generate inner (vec inner-args) constraints)]
          {:trace (tr/make-trace {:gen-fn this :args args
                                  :choices (:choices trace)
                                  :retval (:retval trace)
                                  :score (:score trace)})
           :weight weight})
        {:trace (tr/make-trace {:gen-fn this :args args
                                :choices cm/EMPTY
                                :retval nil
                                :score (mx/scalar 0.0)})
         :weight (mx/scalar 0.0)}))))

(defn mask-combinator
  "Create a Mask combinator that gates execution of an inner GF.
   First argument to the masked GF is a boolean active? flag."
  [inner]
  (->MaskCombinator inner))

;; ---------------------------------------------------------------------------
;; Vectorized Switch
;; ---------------------------------------------------------------------------
;; Executes ALL branches and combines results using mx/where based on
;; [N]-shaped index arrays. This enables vectorized models with discrete
;; latent structure (mixture models, clustering, etc.).

(defn- stack-branch-traces
  "Given N traces from the same GF, stack their values into [N]-shaped arrays."
  [traces]
  (let [first-choices (:choices (first traces))
        is-leaf (cm/has-value? first-choices)]
    {:choices (if is-leaf
               (cm/->Value (mx/stack (mapv #(cm/get-value (:choices %)) traces)))
               (let [addrs (cm/addresses first-choices)]
                 (reduce (fn [cm addr-path]
                           (cm/set-choice cm addr-path
                             (mx/stack (mapv #(cm/get-choice (:choices %) addr-path) traces))))
                         cm/EMPTY addrs)))
     :score (mx/stack (mapv :score traces))
     :retval (let [rvs (mapv :retval traces)]
               (if (mx/array? (first rvs)) (mx/stack rvs) rvs))}))

(defn vectorized-switch
  "Execute all branches with [N] independent samples each, then mask-select
   results based on [N]-shaped indices.
   branches: vector of generative functions
   index: [N]-shaped MLX int32 array of branch indices
   branch-args: arguments for each branch (shared across branches)
   Returns {:choices :score :retval} with [N]-shaped arrays at each site."
  [branches index branch-args]
  (let [n-val (first (mx/shape index))
        n-branches (count branches)
        ;; For each branch, produce N independent samples stacked into [N]-shaped arrays
        branch-data (mapv (fn [gf]
                            (let [traces (mapv (fn [_] (p/simulate gf branch-args))
                                              (range n-val))]
                              (stack-branch-traces traces)))
                          branches)
        ;; Combine branches using mx/where based on index
        first-choices (:choices (first branch-data))
        is-leaf (cm/has-value? first-choices)
        ;; Build combined choices
        ;; Note: reduce-kv over full vector (not rest) so indices match branch indices
        combined-choices
        (if is-leaf
          ;; Distribution branches: combine leaf values
          (let [vals (mapv #(cm/get-value (:choices %)) branch-data)
                combined (reduce-kv
                           (fn [acc i v]
                             (if (zero? i) acc
                               (mx/where (mx/equal index (mx/scalar i mx/int32)) v acc)))
                           (first vals) vals)]
            (cm/->Value combined))
          ;; GF branches: combine per-address
          (let [all-addrs (into #{} (mapcat #(cm/addresses (:choices %)) branch-data))]
            (reduce
              (fn [cm addr-path]
                (let [vals (mapv (fn [bd]
                                  (try (cm/get-choice (:choices bd) addr-path)
                                       (catch :default _ nil)))
                                branch-data)
                      combined (reduce-kv
                                 (fn [acc i v]
                                   (if (or (zero? i) (nil? v)) acc
                                     (mx/where (mx/equal index (mx/scalar i mx/int32)) v acc)))
                                 (or (first vals) (mx/zeros [n-val]))
                                 vals)]
                  (cm/set-choice cm addr-path combined)))
              cm/EMPTY all-addrs)))
        ;; Combine scores using where
        combined-score (reduce-kv
                         (fn [acc i bd]
                           (if (zero? i)
                             (:score bd)
                             (mx/where (mx/equal index (mx/scalar i mx/int32))
                                       (:score bd) acc)))
                         (mx/scalar 0.0)
                         (vec branch-data))
        ;; Combine retvals
        combined-retval (let [rvs (mapv :retval branch-data)]
                          (if (and (mx/array? (first rvs)) (> n-branches 1))
                            (reduce-kv
                              (fn [acc i rv]
                                (if (or (zero? i) (nil? rv)) acc
                                  (mx/where (mx/equal index (mx/scalar i mx/int32)) rv acc)))
                              (first rvs) rvs)
                            (first rvs)))]
    {:choices combined-choices
     :score combined-score
     :retval combined-retval}))

;; ---------------------------------------------------------------------------
;; Scan Combinator
;; ---------------------------------------------------------------------------
;; State-threading sequential combinator, equivalent to GenJAX's scan
;; (and jax.lax.scan). More general than Unfold: takes a carry-state
;; function (c, a) → (c, b) and applies it over a sequence, accumulating
;; both carry-state and outputs.

(defrecord ScanCombinator [kernel]
  p/IGenerativeFunction
  (simulate [this args]
    ;; args: [init-carry inputs] where inputs is a vector
    ;; kernel takes [carry input] and returns [new-carry output]
    (let [[init-carry inputs] args
          n (count inputs)]
      (loop [t 0 carry init-carry
             choices cm/EMPTY score (mx/scalar 0.0)
             outputs []]
        (if (>= t n)
          (tr/make-trace {:gen-fn this :args args
                          :choices choices
                          :retval {:carry carry :outputs outputs}
                          :score score})
          (let [trace (p/simulate kernel [carry (nth inputs t)])
                [new-carry output] (:retval trace)]
            (recur (inc t)
                   new-carry
                   (cm/set-choice choices [t] (:choices trace))
                   (mx/add score (:score trace))
                   (conj outputs output)))))))

  p/IGenerate
  (generate [this args constraints]
    (let [[init-carry inputs] args
          n (count inputs)]
      (loop [t 0 carry init-carry
             choices cm/EMPTY score (mx/scalar 0.0) weight (mx/scalar 0.0)
             outputs []]
        (if (>= t n)
          {:trace (tr/make-trace {:gen-fn this :args args
                                  :choices choices
                                  :retval {:carry carry :outputs outputs}
                                  :score score})
           :weight weight}
          (let [result (p/generate kernel [carry (nth inputs t)]
                                    (cm/get-submap constraints t))
                trace (:trace result)
                [new-carry output] (:retval trace)]
            (recur (inc t)
                   new-carry
                   (cm/set-choice choices [t] (:choices trace))
                   (mx/add score (:score trace))
                   (mx/add weight (:weight result))
                   (conj outputs output))))))))

(extend-type ScanCombinator
  p/IUpdate
  (update [this trace constraints]
    (let [kern (:kernel this)
          {:keys [args choices]} trace
          [init-carry inputs] args
          n (count inputs)]
      (loop [t 0 carry init-carry
             new-choices cm/EMPTY score (mx/scalar 0.0)
             discard cm/EMPTY outputs []]
        (if (>= t n)
          {:trace (tr/make-trace {:gen-fn this :args args
                                  :choices new-choices
                                  :retval {:carry carry :outputs outputs}
                                  :score score})
           :weight (mx/subtract score (:score trace)) :discard discard}
          (let [old-sub-choices (cm/get-submap choices t)
                old-trace (tr/make-trace
                            {:gen-fn kern :args [carry (nth inputs t)]
                             :choices old-sub-choices
                             :retval nil :score (mx/scalar 0.0)})
                result (p/update kern old-trace (cm/get-submap constraints t))
                new-trace (:trace result)
                [new-carry output] (:retval new-trace)]
            (recur (inc t)
                   new-carry
                   (cm/set-choice new-choices [t] (:choices new-trace))
                   (mx/add score (:score new-trace))
                   (if (:discard result)
                     (cm/set-choice discard [t] (:discard result))
                     discard)
                   (conj outputs output)))))))

  p/IRegenerate
  (regenerate [this trace selection]
    (let [kern (:kernel this)
          {:keys [args choices]} trace
          [init-carry inputs] args
          n (count inputs)]
      (loop [t 0 carry init-carry
             new-choices cm/EMPTY score (mx/scalar 0.0) weight (mx/scalar 0.0)
             outputs []]
        (if (>= t n)
          {:trace (tr/make-trace {:gen-fn this :args args
                                  :choices new-choices
                                  :retval {:carry carry :outputs outputs}
                                  :score score})
           :weight (mx/subtract weight (:score trace))}
          (let [old-sub-choices (cm/get-submap choices t)
                old-trace (tr/make-trace
                            {:gen-fn kern :args [carry (nth inputs t)]
                             :choices old-sub-choices
                             :retval nil :score (mx/scalar 0.0)})
                result (p/regenerate kern old-trace
                                     (sel/get-subselection selection t))
                new-trace (:trace result)
                [new-carry output] (:retval new-trace)]
            (recur (inc t)
                   new-carry
                   (cm/set-choice new-choices [t] (:choices new-trace))
                   (mx/add score (:score new-trace))
                   (mx/add weight (:weight result))
                   (conj outputs output))))))))

(defn scan-combinator
  "Create a Scan combinator from a kernel generative function.
   The kernel takes [carry input] and returns [new-carry output].
   The scan applies the kernel to each element of an input sequence,
   threading carry-state and accumulating outputs."
  [kernel]
  (->ScanCombinator kernel))

;; ---------------------------------------------------------------------------
;; Map / Contramap / Dimap Combinators
;; ---------------------------------------------------------------------------
;; Argument/return-value transformation wrappers for generative functions.

(defrecord ContramapGF [inner f]
  ;; Transform arguments before passing to inner GF
  p/IGenerativeFunction
  (simulate [this args]
    (let [transformed-args (f args)
          trace (p/simulate inner transformed-args)]
      (tr/make-trace {:gen-fn this :args args
                      :choices (:choices trace)
                      :retval (:retval trace)
                      :score (:score trace)})))

  p/IGenerate
  (generate [this args constraints]
    (let [transformed-args (f args)
          {:keys [trace weight]} (p/generate inner transformed-args constraints)]
      {:trace (tr/make-trace {:gen-fn this :args args
                              :choices (:choices trace)
                              :retval (:retval trace)
                              :score (:score trace)})
       :weight weight})))

(defrecord MapRetvalGF [inner g]
  ;; Transform return value from inner GF
  p/IGenerativeFunction
  (simulate [this args]
    (let [trace (p/simulate inner args)]
      (tr/make-trace {:gen-fn this :args args
                      :choices (:choices trace)
                      :retval (g (:retval trace))
                      :score (:score trace)})))

  p/IGenerate
  (generate [this args constraints]
    (let [{:keys [trace weight]} (p/generate inner args constraints)]
      {:trace (tr/make-trace {:gen-fn this :args args
                              :choices (:choices trace)
                              :retval (g (:retval trace))
                              :score (:score trace)})
       :weight weight})))

(defn contramap-gf
  "Transform arguments before passing to a generative function.
   f: (fn [args] -> transformed-args)"
  [gf f]
  (->ContramapGF gf f))

(defn map-retval
  "Transform the return value of a generative function.
   g: (fn [retval] -> transformed-retval)"
  [gf g]
  (->MapRetvalGF gf g))

(defn dimap
  "Transform both arguments and return value of a generative function.
   f: (fn [args] -> transformed-args)
   g: (fn [retval] -> transformed-retval)"
  [gf f g]
  (-> gf (contramap-gf f) (map-retval g)))

(extend-type ContramapGF
  p/IUpdate
  (update [this trace constraints]
    (let [transformed-args ((:f this) (:args trace))
          inner-trace (tr/make-trace {:gen-fn (:inner this) :args transformed-args
                                      :choices (:choices trace)
                                      :retval (:retval trace) :score (:score trace)})
          result (p/update (:inner this) inner-trace constraints)]
      {:trace (tr/make-trace {:gen-fn this :args (:args trace)
                              :choices (:choices (:trace result))
                              :retval (:retval (:trace result))
                              :score (:score (:trace result))})
       :weight (:weight result) :discard (:discard result)}))

  p/IRegenerate
  (regenerate [this trace selection]
    (let [transformed-args ((:f this) (:args trace))
          inner-trace (tr/make-trace {:gen-fn (:inner this) :args transformed-args
                                      :choices (:choices trace)
                                      :retval (:retval trace) :score (:score trace)})
          result (p/regenerate (:inner this) inner-trace selection)]
      {:trace (tr/make-trace {:gen-fn this :args (:args trace)
                              :choices (:choices (:trace result))
                              :retval (:retval (:trace result))
                              :score (:score (:trace result))})
       :weight (:weight result)})))

(extend-type MapRetvalGF
  p/IUpdate
  (update [this trace constraints]
    (let [inner-trace (tr/make-trace {:gen-fn (:inner this) :args (:args trace)
                                      :choices (:choices trace)
                                      :retval nil :score (:score trace)})
          result (p/update (:inner this) inner-trace constraints)]
      {:trace (tr/make-trace {:gen-fn this :args (:args trace)
                              :choices (:choices (:trace result))
                              :retval ((:g this) (:retval (:trace result)))
                              :score (:score (:trace result))})
       :weight (:weight result) :discard (:discard result)}))

  p/IRegenerate
  (regenerate [this trace selection]
    (let [inner-trace (tr/make-trace {:gen-fn (:inner this) :args (:args trace)
                                      :choices (:choices trace)
                                      :retval nil :score (:score trace)})
          result (p/regenerate (:inner this) inner-trace selection)]
      {:trace (tr/make-trace {:gen-fn this :args (:args trace)
                              :choices (:choices (:trace result))
                              :retval ((:g this) (:retval (:trace result)))
                              :score (:score (:trace result))})
       :weight (:weight result)})))

;; ---------------------------------------------------------------------------
;; Mix Combinator
;; ---------------------------------------------------------------------------
;; First-class mixture model support. Combines multiple component GFs
;; with mixing weights into a single generative function.

(defrecord MixCombinator [components log-weights-fn]
  p/IGenerativeFunction
  (simulate [this args]
    ;; Sample component index, then simulate that component
    (let [log-w (log-weights-fn args)
          idx-trace (p/simulate (dc/->Distribution
                                  :categorical {:logits log-w}) [])
          idx (mx/item (cm/get-value (:choices idx-trace)))
          component (nth components (int idx))
          comp-trace (p/simulate component args)]
      (tr/make-trace {:gen-fn this :args args
                      :choices (cm/set-choice (:choices comp-trace)
                                              [:component-idx]
                                              (mx/scalar (int idx) mx/int32))
                      :retval (:retval comp-trace)
                      :score (mx/add (:score comp-trace) (:score idx-trace))})))

  p/IGenerate
  (generate [this args constraints]
    (let [log-w (log-weights-fn args)
          ;; Check if component index is constrained
          idx-constraint (cm/get-submap constraints :component-idx)
          idx-result (if (cm/has-value? idx-constraint)
                       (let [idx-val (cm/get-value idx-constraint)
                             d (dc/->Distribution :categorical {:logits log-w})]
                         (dc/dist-generate d idx-constraint))
                       (let [d (dc/->Distribution :categorical {:logits log-w})]
                         {:trace (dc/dist-simulate d) :weight (mx/scalar 0.0)}))
          idx (mx/item (cm/get-value (:choices (:trace idx-result))))
          component (nth components (int idx))
          {:keys [trace weight]} (p/generate component args constraints)]
      {:trace (tr/make-trace {:gen-fn this :args args
                              :choices (cm/set-choice (:choices trace)
                                                      [:component-idx]
                                                      (mx/scalar (int idx) mx/int32))
                              :retval (:retval trace)
                              :score (mx/add (:score trace)
                                             (:score (:trace idx-result)))})
       :weight (mx/add weight (:weight idx-result))})))

(defn mix-combinator
  "Create a mixture model combinator.
   components: vector of component generative functions
   log-weights-fn: (fn [args] -> MLX array of log mixing weights)
                   or a fixed MLX array of log mixing weights."
  [components log-weights-fn]
  (let [lwf (if (fn? log-weights-fn)
              log-weights-fn
              (fn [_] log-weights-fn))]
    (->MixCombinator components lwf)))

(extend-type MixCombinator
  p/IUpdate
  (update [this trace constraints]
    (let [old-choices (:choices trace)
          old-idx (int (mx/item (cm/get-choice old-choices [:component-idx])))
          args (:args trace)
          log-w ((:log-weights-fn this) args)
          idx-dist (dc/->Distribution :categorical {:logits log-w})
          old-idx-score (dc/dist-log-prob idx-dist (mx/scalar old-idx mx/int32))
          ;; Check if component index is being updated
          idx-constraint (cm/get-submap constraints :component-idx)
          new-idx (if (cm/has-value? idx-constraint)
                    (int (mx/item (cm/get-value idx-constraint)))
                    old-idx)
          ;; Inner choices = everything except component-idx
          inner-old-choices (cm/->Node (dissoc (:m old-choices) :component-idx))
          inner-constraints (if (= constraints cm/EMPTY)
                              cm/EMPTY
                              (cm/->Node (dissoc (:m constraints) :component-idx)))]
      (if (= new-idx old-idx)
        ;; Same component: update inner only
        (let [component (nth (:components this) old-idx)
              inner-old-score (mx/subtract (:score trace) old-idx-score)
              inner-old-trace (tr/make-trace {:gen-fn component :args args
                                              :choices inner-old-choices
                                              :retval (:retval trace) :score inner-old-score})
              result (p/update component inner-old-trace inner-constraints)
              new-inner-trace (:trace result)
              new-score (mx/add (:score new-inner-trace) old-idx-score)]
          {:trace (tr/make-trace {:gen-fn this :args args
                                  :choices (cm/set-choice (:choices new-inner-trace)
                                                          [:component-idx]
                                                          (mx/scalar old-idx mx/int32))
                                  :retval (:retval new-inner-trace)
                                  :score new-score})
           :weight (mx/subtract new-score (:score trace))
           :discard (:discard result)})
        ;; Different component: generate new component from scratch
        (let [new-component (nth (:components this) new-idx)
              new-idx-score (dc/dist-log-prob idx-dist (mx/scalar new-idx mx/int32))
              gen-result (p/generate new-component args inner-constraints)
              new-inner-trace (:trace gen-result)
              new-score (mx/add (:score new-inner-trace) new-idx-score)]
          {:trace (tr/make-trace {:gen-fn this :args args
                                  :choices (cm/set-choice (:choices new-inner-trace)
                                                          [:component-idx]
                                                          (mx/scalar new-idx mx/int32))
                                  :retval (:retval new-inner-trace)
                                  :score new-score})
           :weight (mx/subtract new-score (:score trace))
           :discard old-choices}))))

  p/IRegenerate
  (regenerate [this trace selection]
    (let [old-choices (:choices trace)
          old-idx (int (mx/item (cm/get-choice old-choices [:component-idx])))
          args (:args trace)
          log-w ((:log-weights-fn this) args)
          idx-dist (dc/->Distribution :categorical {:logits log-w})
          old-idx-score (dc/dist-log-prob idx-dist (mx/scalar old-idx mx/int32))
          idx-selected? (sel/selected? selection :component-idx)]
      (if idx-selected?
        ;; Resample component index and simulate new component
        (let [new-idx-trace (dc/dist-simulate idx-dist)
              new-idx (int (mx/item (cm/get-value (:choices new-idx-trace))))
              new-idx-score (:score new-idx-trace)
              new-component (nth (:components this) new-idx)
              new-comp-trace (p/simulate new-component args)
              new-score (mx/add (:score new-comp-trace) new-idx-score)]
          {:trace (tr/make-trace {:gen-fn this :args args
                                  :choices (cm/set-choice (:choices new-comp-trace)
                                                          [:component-idx]
                                                          (mx/scalar new-idx mx/int32))
                                  :retval (:retval new-comp-trace)
                                  :score new-score})
           :weight (mx/subtract new-score (:score trace))})
        ;; Same component: regenerate within the component
        (let [component (nth (:components this) old-idx)
              inner-old-score (mx/subtract (:score trace) old-idx-score)
              inner-old-choices (cm/->Node (dissoc (:m old-choices) :component-idx))
              inner-old-trace (tr/make-trace {:gen-fn component :args args
                                              :choices inner-old-choices
                                              :retval (:retval trace) :score inner-old-score})
              result (p/regenerate component inner-old-trace selection)
              new-inner-trace (:trace result)
              new-score (mx/add (:score new-inner-trace) old-idx-score)]
          {:trace (tr/make-trace {:gen-fn this :args args
                                  :choices (cm/set-choice (:choices new-inner-trace)
                                                          [:component-idx]
                                                          (mx/scalar old-idx mx/int32))
                                  :retval (:retval new-inner-trace)
                                  :score new-score})
           :weight (mx/subtract new-score (:score trace))})))))

;; ---------------------------------------------------------------------------
;; IEdit implementations — delegate to edit-dispatch for all combinator types
;; ---------------------------------------------------------------------------

(extend-type MapCombinator
  edit/IEdit
  (edit [gf trace edit-request]
    (edit/edit-dispatch gf trace edit-request)))

(extend-type UnfoldCombinator
  edit/IEdit
  (edit [gf trace edit-request]
    (edit/edit-dispatch gf trace edit-request)))

(extend-type SwitchCombinator
  edit/IEdit
  (edit [gf trace edit-request]
    (edit/edit-dispatch gf trace edit-request)))

(extend-type ScanCombinator
  edit/IEdit
  (edit [gf trace edit-request]
    (edit/edit-dispatch gf trace edit-request)))

(extend-type MaskCombinator
  edit/IEdit
  (edit [gf trace edit-request]
    (edit/edit-dispatch gf trace edit-request)))

(extend-type MixCombinator
  edit/IEdit
  (edit [gf trace edit-request]
    (edit/edit-dispatch gf trace edit-request)))

(extend-type ContramapGF
  edit/IEdit
  (edit [gf trace edit-request]
    (edit/edit-dispatch gf trace edit-request)))

(extend-type MapRetvalGF
  edit/IEdit
  (edit [gf trace edit-request]
    (edit/edit-dispatch gf trace edit-request)))

;; ---------------------------------------------------------------------------
;; IUpdateWithDiffs implementations
;; ---------------------------------------------------------------------------

(extend-type MapCombinator
  p/IUpdateWithDiffs
  (update-with-diffs [this trace constraints argdiffs]
    (let [old-choices (:choices trace)
          args (:args trace)
          n (count (first args))
          old-element-scores (::element-scores (meta trace))
          has-constraints (not= constraints cm/EMPTY)]
      (cond
        ;; No changes to args and no new constraints: return trace unchanged
        (and (diff/no-change? argdiffs) (not has-constraints))
        {:trace trace :weight (mx/scalar 0.0) :discard cm/EMPTY}

        ;; vector-diff with stored element scores: optimize
        (and (or (diff/no-change? argdiffs)
                 (= (:diff-type argdiffs) :vector-diff))
             old-element-scores)
        (let [changed-set (if (diff/no-change? argdiffs)
                            #{}
                            (:changed argdiffs))
              kernel (:kernel this)
              ;; Determine which elements need updating: changed args OR new constraints
              update-set (into changed-set
                               (filter #(not= (cm/get-submap constraints %) cm/EMPTY))
                               (range n))
              results (mapv
                        (fn [i]
                          (if (contains? update-set i)
                            ;; Element changed: do full update
                            (let [kernel-args (mapv #(nth % i) args)
                                  old-trace (tr/make-trace
                                              {:gen-fn kernel :args kernel-args
                                               :choices (cm/get-submap old-choices i)
                                               :retval nil :score (mx/scalar 0.0)})]
                              (p/update kernel old-trace (cm/get-submap constraints i)))
                            ;; Element unchanged: reuse old choices and score
                            {:trace (tr/make-trace
                                      {:gen-fn kernel
                                       :args (mapv #(nth % i) args)
                                       :choices (cm/get-submap old-choices i)
                                       :retval (nth (:retval trace) i nil)
                                       :score (nth old-element-scores i)})
                             :weight (mx/scalar 0.0)
                             :discard cm/EMPTY}))
                        (range n))
              choices (assemble-choices results (comp :choices :trace))
              retvals (mapv (comp :retval :trace) results)
              score (sum-field results (comp :score :trace))
              weight (mx/subtract score (:score trace))
              discard (assemble-choices (filter #(and (:discard %) (not= (:discard %) cm/EMPTY)) results) :discard)
              element-scores (mapv (comp :score :trace) results)]
          {:trace (with-meta
                    (tr/make-trace {:gen-fn this :args args
                                    :choices choices :retval retvals :score score})
                    {::element-scores element-scores})
           :weight weight :discard discard})

        ;; Unknown change: fall back to full update
        :else (p/update this trace constraints)))))

(extend-type UnfoldCombinator
  p/IUpdateWithDiffs
  (update-with-diffs [this trace constraints argdiffs]
    (if (and (diff/no-change? argdiffs) (= constraints cm/EMPTY))
      {:trace trace :weight (mx/scalar 0.0) :discard cm/EMPTY}
      (p/update this trace constraints))))

(extend-type SwitchCombinator
  p/IUpdateWithDiffs
  (update-with-diffs [this trace constraints argdiffs]
    (if (and (diff/no-change? argdiffs) (= constraints cm/EMPTY))
      {:trace trace :weight (mx/scalar 0.0) :discard cm/EMPTY}
      (p/update this trace constraints))))

(extend-type ScanCombinator
  p/IUpdateWithDiffs
  (update-with-diffs [this trace constraints argdiffs]
    (if (and (diff/no-change? argdiffs) (= constraints cm/EMPTY))
      {:trace trace :weight (mx/scalar 0.0) :discard cm/EMPTY}
      (p/update this trace constraints))))
