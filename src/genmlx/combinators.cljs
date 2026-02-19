(ns genmlx.combinators
  "GFI combinators: Map, Unfold, Switch, Scan, Mask, Mix, and more.
   These compose generative functions into higher-level models."
  (:require [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.selection :as sel]
            [genmlx.dist.core :as dc]))

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

(extend-type UnfoldCombinator
  p/IUpdate
  (update [this trace constraints]
    (let [kern (:kernel this)
          {:keys [args choices]} trace
          [n init-state & extra] args]
      (loop [t 0 state init-state
             new-choices cm/EMPTY score (mx/scalar 0.0) weight (mx/scalar 0.0)
             discard cm/EMPTY states []]
        (if (>= t n)
          {:trace (tr/make-trace {:gen-fn this :args args
                                  :choices new-choices :retval states :score score})
           :weight weight :discard discard}
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
                   (mx/add weight (:weight result))
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
           :weight weight}
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

(defn vectorized-switch
  "Execute all branches and mask-select results based on [N]-shaped indices.
   branches: vector of generative functions
   index: [N]-shaped MLX int32 array of branch indices
   branch-args: arguments for each branch (shared across branches)
   Returns {:choices :score :retval} with [N]-shaped arrays at each site.

   Each branch is executed independently (all sites sampled),
   and results are combined using mx/where per branch index."
  [branches index branch-args]
  (let [n-branches (count branches)
        ;; Execute all branches via simulate
        branch-results (mapv (fn [gf] (p/simulate gf branch-args)) branches)
        ;; For each choice address, combine using mx/where
        ;; Get all addresses from all branches
        all-addrs (into #{} (mapcat #(cm/addresses (:choices %)) branch-results))
        ;; Build combined choices and score
        combined-choices (reduce
                           (fn [cm addr-path]
                             (let [;; For each branch, get the value at this address (or zeros)
                                   branch-vals (mapv (fn [trace]
                                                       (try
                                                         (cm/get-choice (:choices trace) addr-path)
                                                         (catch :default _ nil)))
                                                     branch-results)
                                   ;; Combine using where: start with branch 0, overlay others
                                   combined (reduce-kv
                                              (fn [acc i bval]
                                                (if (and bval (pos? i))
                                                  (let [mask (mx/equal index (mx/scalar i mx/int32))]
                                                    (mx/where mask bval acc))
                                                  acc))
                                              (or (first branch-vals)
                                                  (mx/scalar 0.0))
                                              (vec (rest branch-vals)))]
                               (cm/set-choice cm addr-path combined)))
                           cm/EMPTY
                           all-addrs)
        ;; Combine scores using where
        combined-score (reduce-kv
                         (fn [acc i trace]
                           (if (zero? i)
                             (:score trace)
                             (let [mask (mx/equal index (mx/scalar i mx/int32))]
                               (mx/where mask (:score trace) acc))))
                         (mx/scalar 0.0)
                         (vec branch-results))]
    {:choices combined-choices
     :score combined-score
     ;; Combine retvals using where (works for MLX arrays; for non-array retvals
     ;; falls back to branch-0 retval since mx/where requires arrays)
     :retval (let [retvals (mapv :retval branch-results)]
               (if (and (mx/array? (first retvals))
                        (> n-branches 1))
                 (reduce-kv
                   (fn [acc i rv]
                     (if (and rv (pos? i))
                       (let [mask (mx/equal index (mx/scalar i mx/int32))]
                         (mx/where mask rv acc))
                       acc))
                   (first retvals)
                   (vec (rest retvals)))
                 (first retvals)))}))

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
             new-choices cm/EMPTY score (mx/scalar 0.0) weight (mx/scalar 0.0)
             discard cm/EMPTY outputs []]
        (if (>= t n)
          {:trace (tr/make-trace {:gen-fn this :args args
                                  :choices new-choices
                                  :retval {:carry carry :outputs outputs}
                                  :score score})
           :weight weight :discard discard}
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
                   (mx/add weight (:weight result))
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
           :weight weight}
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
