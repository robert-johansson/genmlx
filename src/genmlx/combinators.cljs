(ns genmlx.combinators
  "GFI combinators: Map, Unfold, Switch, Scan, Mask, Mix, and more.
   These compose generative functions into higher-level models."
  (:require [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.selection :as sel]
            [genmlx.handler :as h]
            [genmlx.runtime :as rt]
            [genmlx.dist.core :as dc]
            [genmlx.edit :as edit]
            [genmlx.diff :as diff]
            [genmlx.compiled :as compiled]))

;; ---------------------------------------------------------------------------
;; Shared helpers
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

(defn- values->choices
  "Convert compiled result {:values {addr->val}} to a ChoiceMap."
  [values]
  (cm/from-flat-map values))

;; ---------------------------------------------------------------------------
;; Map Combinator
;; ---------------------------------------------------------------------------
;; Applies a generative function independently to each element of input sequences.
;; Like Gen.jl's Map combinator.

(defn- unpack-fused-map-outputs
  "Unpack [N]-shaped fused map values into per-element ChoiceMaps."
  [values scores N addr-order]
  (loop [i 0
         choices cm/EMPTY
         element-scores []]
    (if (>= i N)
      {:choices choices :element-scores element-scores}
      (let [elem-cm (reduce
                     (fn [scm addr]
                       (cm/set-value scm addr (mx/index (get values addr) i)))
                     cm/EMPTY
                     addr-order)
            elem-score (mx/index scores i)]
        (recur (inc i)
               (cm/set-choice choices [i] elem-cm)
               (conj element-scores elem-score))))))

(defn- map-simulate-fused
  "Fused Map simulate: all N elements in one call via broadcasting."
  [this args kernel fused-fn addr-order]
  (let [n (count (first args))
        key (rng/fresh-key)
        stacked-args (mapv (fn [arg-col]
                             (mx/stack (mapv mx/ensure-array arg-col)))
                           args)
        {:keys [values scores retval]}
        (fused-fn key stacked-args n)
        _ (mx/materialize! scores retval)
        total-score (mx/sum scores)
        _ (mx/materialize! total-score)
        {:keys [choices element-scores]}
        (unpack-fused-map-outputs values scores n addr-order)
        retvals (mapv #(mx/index retval %) (range n))]
    (with-meta
      (tr/make-trace {:gen-fn this :args args
                      :choices choices :retval retvals :score total-score})
      {::element-scores element-scores ::compiled-path true ::fused true})))

(defn- map-simulate-compiled
  "Compiled Map simulate: call compiled-simulate per element."
  [this args kernel csim]
  (let [n (count (first args))
        init-key (rng/fresh-key)
        results (loop [i 0 key init-key acc []]
                  (if (>= i n)
                    acc
                    (let [[k1 k2] (rng/split key)
                          elem-args (mapv #(nth % i) args)
                          result (csim k1 (vec elem-args))]
                      (recur (inc i) k2 (conj acc result)))))
        choices (reduce (fn [cm [i r]]
                          (reduce-kv (fn [cm2 addr val]
                                       (cm/set-choice cm2 [i addr] val))
                                     cm (:values r)))
                        cm/EMPTY (map-indexed vector results))
        retvals (mapv :retval results)
        score (reduce (fn [acc r] (mx/add acc (:score r)))
                      (mx/scalar 0.0) results)
        element-scores (mapv :score results)]
    (with-meta
      (tr/make-trace {:gen-fn this :args args
                      :choices choices :retval retvals :score score})
      {::element-scores element-scores ::compiled-path true})))

(defn- map-simulate-handler
  "Handler Map simulate: delegate to kernel per element."
  [this args kernel]
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

(defrecord MapCombinator [kernel]
  p/IGenerativeFunction
  (simulate [this args]
    (if-let [{:keys [fused-fn addr-order]} (compiled/make-fused-map-simulate
                                            (:schema kernel) (:source kernel))]
      (map-simulate-fused this args kernel fused-fn addr-order)
      (if-let [csim (compiled/get-compiled-simulate kernel)]
        (map-simulate-compiled this args kernel csim)
        (map-simulate-handler this args kernel))))

  p/IGenerate
  (generate [this args constraints]
    (if-let [cgen (compiled/get-compiled-generate kernel)]
      ;; Compiled path — call compiled-generate directly per element
      (let [n (count (first args))
            init-key (rng/fresh-key)]
        (loop [i 0 key init-key
               choices cm/EMPTY score (mx/scalar 0.0) weight (mx/scalar 0.0)
               retvals [] element-scores []]
          (if (>= i n)
            {:trace (with-meta
                      (tr/make-trace {:gen-fn this :args args
                                      :choices choices :retval retvals :score score})
                      {::element-scores element-scores ::compiled-path true})
             :weight weight}
            (let [[k1 k2] (rng/split key)
                  elem-args (mapv #(nth % i) args)
                  result (cgen k1 (vec elem-args) (cm/get-submap constraints i))
                  elem-choices (values->choices (:values result))]
              (recur (inc i) k2
                     (cm/set-choice choices [i] elem-choices)
                     (mx/add score (:score result))
                     (mx/add weight (:weight result))
                     (conj retvals (:retval result))
                     (conj element-scores (:score result)))))))
      ;; Fallback: handler path
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
         :weight weight})))

  p/IUpdate
  (update [this trace constraints]
    (if-let [cupd (compiled/get-compiled-update kernel)]
      ;; WP-8: compiled update path
      (let [old-choices (:choices trace)
            args (:args trace)
            n (count (first args))
            init-key (rng/fresh-key)]
        (loop [i 0 key init-key
               choices cm/EMPTY score (mx/scalar 0.0) discard cm/EMPTY
               retvals [] element-scores []]
          (if (>= i n)
            {:trace (with-meta
                      (tr/make-trace {:gen-fn this :args args
                                      :choices choices :retval retvals :score score})
                      {::element-scores element-scores ::compiled-path true})
             :weight (mx/subtract score (:score trace)) :discard discard}
            (let [[k1 k2] (rng/split key)
                  elem-args (mapv #(nth % i) args)
                  result (cupd k1 (vec elem-args)
                               (cm/get-submap constraints i)
                               (cm/get-submap old-choices i))
                  elem-choices (values->choices (:values result))
                  elem-discard (values->choices (:discard result))]
              (recur (inc i) k2
                     (cm/set-choice choices [i] elem-choices)
                     (mx/add score (:score result))
                     (if (seq (:discard result))
                       (cm/set-choice discard [i] elem-discard)
                       discard)
                     (conj retvals (:retval result))
                     (conj element-scores (:score result)))))))
      ;; Fallback: handler path
      (let [old-choices (:choices trace)
            args (:args trace)
            n (count (first args))
            old-element-scores (::element-scores (meta trace))
            results (mapv (fn [i]
                            (let [kernel-args (mapv #(nth % i) args)
                                  old-trace (tr/make-trace
                                             {:gen-fn kernel :args kernel-args
                                              :choices (cm/get-submap old-choices i)
                                              :retval nil :score (if old-element-scores (nth old-element-scores i) (mx/scalar 0.0))})]
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
         :weight weight :discard discard})))

  p/IRegenerate
  (regenerate [this trace selection]
    (if-let [cregen (compiled/get-compiled-regenerate kernel)]
      ;; WP-9A: compiled regenerate path
      (let [old-choices (:choices trace)
            args (:args trace)
            n (count (first args))
            init-key (rng/fresh-key)]
        (loop [i 0 key init-key
               choices cm/EMPTY score (mx/scalar 0.0) weight (mx/scalar 0.0)
               retvals [] element-scores []]
          (if (>= i n)
            {:trace (with-meta
                      (tr/make-trace {:gen-fn this :args args
                                      :choices choices :retval retvals :score score})
                      {::element-scores element-scores ::compiled-path true})
             :weight weight}
            (let [[k1 k2] (rng/split key)
                  elem-args (mapv #(nth % i) args)
                  old-sub-choices (cm/get-submap old-choices i)
                  result (cregen k1 (vec elem-args) old-sub-choices
                                 (sel/get-subselection selection i))
                  elem-choices (values->choices (:values result))
                  old-elem-score (or (some-> (::element-scores (meta trace))
                                             (nth i nil))
                                     (mx/scalar 0.0))
                  ;; compiled-regenerate returns proposal_ratio in :weight
                  ;; per-step weight = new_score - old_score - proposal_ratio
                  step-weight (mx/subtract (mx/subtract (:score result) old-elem-score)
                                           (:weight result))]
              (recur (inc i) k2
                     (cm/set-choice choices [i] elem-choices)
                     (mx/add score (:score result))
                     (mx/add weight step-weight)
                     (conj retvals (:retval result))
                     (conj element-scores (:score result)))))))
      ;; Fallback: handler path (weight bug fixed: no - (:score trace))
      (let [old-choices (:choices trace)
            args (:args trace)
            n (count (first args))
            old-element-scores (::element-scores (meta trace))
            results (mapv (fn [i]
                            (let [kernel-args (mapv #(nth % i) args)
                                  old-trace (tr/make-trace
                                             {:gen-fn kernel :args kernel-args
                                              :choices (cm/get-submap old-choices i)
                                              :retval nil :score (if old-element-scores (nth old-element-scores i) (mx/scalar 0.0))})]
                              (p/regenerate kernel old-trace
                                            (sel/get-subselection selection i))))
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
         :weight weight}))))

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

(defn- unpack-fused-outputs
  "Unpack fused output tensor [T, K+1] into per-step ChoiceMaps and retvals.
   addr-order: [keyword...] mapping column index 0..K-1 to addresses.
   Column K is the retval (new state).
   Returns {:choices cm :states [retval...] :step-scores [score...]}."
  [outputs-tensor scores-tensor T addr-order]
  (let [n-sites (count addr-order)]
    (loop [t 0
           choices cm/EMPTY
           states []
           step-scores []]
      (if (>= t T)
        {:choices choices :states states :step-scores step-scores}
        (let [row (mx/index outputs-tensor t)
              step-cm (reduce
                       (fn [scm [idx addr]]
                         (cm/set-value scm addr (mx/index row idx)))
                       cm/EMPTY
                       (map-indexed vector addr-order))
              retval (mx/index row n-sites)
              step-score (mx/index scores-tensor t)]
          (recur (inc t)
                 (cm/set-choice choices [t] step-cm)
                 (conj states retval)
                 (conj step-scores step-score)))))))

(defn- extras-match?
  "Check if cached extra args match current extra args."
  [cached-extras current-extras]
  (or (identical? cached-extras current-extras)
      (and (= (count cached-extras) (count current-extras))
           (every? true?
                   (map (fn [a b]
                          (let [av (mx/item (mx/ensure-array a))
                                bv (mx/item (mx/ensure-array b))]
                            (== av bv)))
                        cached-extras current-extras)))))

(defn- get-or-build-fused-unfold
  "Get cached or build new fused unfold simulate.
   Cache is stored as metadata on the combinator.
   Returns {:compiled-fn :noise-dim :addr-order :noise-site-types :extra-args} or nil."
  [cache kernel T extra]
  (when (and (pos? T) (compiled/fusable-kernel? kernel))
    (let [cached (get @cache T)]
      (if (and cached (extras-match? (:extra-args cached) extra))
        cached
        ;; Build new fused function
        (when-let [fused (compiled/make-fused-unfold-simulate
                          (:schema kernel) (:source kernel) T extra)]
          (swap! cache assoc T fused)
          fused)))))

(defn- unfold-simulate-fused
  "Fused Unfold simulate: 2 Metal dispatches for T steps."
  [this args kernel fused-cache n init-state extra]
  (let [{:keys [compiled-fn noise-dim addr-order noise-site-types]}
        (get-or-build-fused-unfold fused-cache kernel n extra)
        key (rng/fresh-key)
        noise (compiled/generate-noise-matrix key n noise-site-types)
        [outputs-tensor scores-tensor total-score]
        (compiled-fn (mx/ensure-array init-state) noise)
        _ (mx/materialize! outputs-tensor scores-tensor total-score)
        {:keys [choices states step-scores]}
        (unpack-fused-outputs outputs-tensor scores-tensor n addr-order)]
    (with-meta
      (tr/make-trace {:gen-fn this :args args
                      :choices choices :retval states :score total-score})
      {::step-scores step-scores ::compiled-path true ::fused true})))

(defn- unfold-simulate-compiled
  "Compiled Unfold simulate: call compiled-simulate per step."
  [this args kernel n init-state extra csim]
  (let [init-key (rng/fresh-key)]
    (loop [t 0 state init-state key init-key
           choices cm/EMPTY score (mx/scalar 0.0)
           states [] step-scores []]
      (if (>= t n)
        (with-meta
          (tr/make-trace {:gen-fn this :args args
                          :choices choices :retval states :score score})
          {::step-scores step-scores ::compiled-path true})
        (let [[k1 k2] (rng/split key)
              result (csim k1 (into [t state] extra))
              new-state (:retval result)
              step-choices (cm/from-flat-map (:values result))]
          (recur (inc t)
                 new-state k2
                 (cm/set-choice choices [t] step-choices)
                 (mx/add score (:score result))
                 (conj states new-state)
                 (conj step-scores (:score result))))))))

(defn- unfold-simulate-handler
  "Handler Unfold simulate: delegate to kernel per step."
  [this args kernel n init-state extra]
  (loop [t 0 state init-state
         choices cm/EMPTY score (mx/scalar 0.0)
         states [] step-scores []]
    (if (>= t n)
      (with-meta
        (tr/make-trace {:gen-fn this :args args
                        :choices choices :retval states :score score})
        {::step-scores step-scores})
      (let [trace (p/simulate kernel (into [t state] extra))
            new-state (:retval trace)]
        (recur (inc t)
               new-state
               (cm/set-choice choices [t] (:choices trace))
               (mx/add score (:score trace))
               (conj states new-state)
               (conj step-scores (:score trace)))))))

(defrecord UnfoldCombinator [kernel fused-cache]
  p/IGenerativeFunction
  (simulate [this args]
    (let [[n init-state & extra] args]
      (if-let [_fused (get-or-build-fused-unfold fused-cache kernel n extra)]
        (unfold-simulate-fused this args kernel fused-cache n init-state extra)
        (if-let [csim (compiled/get-compiled-simulate kernel)]
          (unfold-simulate-compiled this args kernel n init-state extra csim)
          (unfold-simulate-handler this args kernel n init-state extra)))))

  p/IGenerate
  (generate [this args constraints]
    (let [[n init-state & extra] args]
      (if-let [cgen (compiled/get-compiled-generate kernel)]
        ;; Compiled path
        (let [init-key (rng/fresh-key)]
          (loop [t 0 state init-state key init-key
                 choices cm/EMPTY score (mx/scalar 0.0) weight (mx/scalar 0.0)
                 states [] step-scores []]
            (if (>= t n)
              {:trace (with-meta
                        (tr/make-trace {:gen-fn this :args args
                                        :choices choices :retval states :score score})
                        {::step-scores step-scores ::compiled-path true})
               :weight weight}
              (let [[k1 k2] (rng/split key)
                    result (cgen k1 (into [t state] extra)
                                 (cm/get-submap constraints t))
                    new-state (:retval result)
                    step-choices (values->choices (:values result))]
                (recur (inc t) new-state k2
                       (cm/set-choice choices [t] step-choices)
                       (mx/add score (:score result))
                       (mx/add weight (:weight result))
                       (conj states new-state)
                       (conj step-scores (:score result)))))))
        ;; Fallback: handler path
        (loop [t 0 state init-state
               choices cm/EMPTY score (mx/scalar 0.0) weight (mx/scalar 0.0)
               states [] step-scores []]
          (if (>= t n)
            {:trace (with-meta
                      (tr/make-trace {:gen-fn this :args args
                                      :choices choices :retval states :score score})
                      {::step-scores step-scores})
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
                     (conj states new-state)
                     (conj step-scores (:score (:trace result)))))))))))

(extend-type UnfoldCombinator
  p/IUpdate
  (update [this trace constraints]
    (let [kern (:kernel this)
          cupd (compiled/get-compiled-update kern)
          {:keys [args choices]} trace
          [n init-state & extra] args
          old-step-scores (::step-scores (meta trace))
          ;; Find first step with non-empty constraints (prefix-skip boundary)
          first-changed (if old-step-scores
                          (loop [t 0]
                            (cond
                              (>= t n) n
                              (not= (cm/get-submap constraints t) cm/EMPTY) t
                              :else (recur (inc t))))
                          0)]
      ;; If no steps have constraints and we have metadata, return trace unchanged
      (if (and old-step-scores (= first-changed n))
        {:trace trace :weight (mx/scalar 0.0) :discard cm/EMPTY}
        ;; Build prefix from old trace (steps 0..first-changed-1)
        (let [prefix-choices (if (pos? first-changed)
                               (reduce (fn [cm t]
                                         (cm/set-choice cm [t] (cm/get-submap choices t)))
                                       cm/EMPTY (range first-changed))
                               cm/EMPTY)
              prefix-score (if (pos? first-changed)
                             (reduce (fn [acc t] (mx/add acc (nth old-step-scores t)))
                                     (mx/scalar 0.0) (range first-changed))
                             (mx/scalar 0.0))
              prefix-states (if (pos? first-changed)
                              (subvec (:retval trace) 0 first-changed)
                              [])
              prefix-step-scores (if (pos? first-changed)
                                   (subvec (vec old-step-scores) 0 first-changed)
                                   [])
              start-state (if (pos? first-changed)
                            (nth (:retval trace) (dec first-changed))
                            init-state)
              init-key (when cupd (rng/fresh-key))]
          ;; Execute steps first-changed..n-1
          (loop [t first-changed state start-state key init-key
                 new-choices prefix-choices score prefix-score
                 discard cm/EMPTY
                 states prefix-states step-scores prefix-step-scores]
            (if (>= t n)
              {:trace (with-meta
                        (tr/make-trace {:gen-fn this :args args
                                        :choices new-choices :retval states :score score})
                        (cond-> {::step-scores step-scores}
                          cupd (assoc ::compiled-path true)))
               :weight (mx/subtract score (:score trace)) :discard discard}
              (if cupd
                ;; WP-8: compiled update path
                (let [[k1 k2] (rng/split key)
                      kernel-args (into [t state] extra)
                      old-sub-choices (cm/get-submap choices t)
                      result (cupd k1 (vec kernel-args)
                                   (cm/get-submap constraints t) old-sub-choices)
                      step-choices (values->choices (:values result))
                      step-discard (values->choices (:discard result))
                      new-state (:retval result)]
                  (recur (inc t) new-state k2
                         (cm/set-choice new-choices [t] step-choices)
                         (mx/add score (:score result))
                         (if (seq (:discard result))
                           (cm/set-choice discard [t] step-discard)
                           discard)
                         (conj states new-state)
                         (conj step-scores (:score result))))
                ;; Fallback: handler path
                (let [old-sub-choices (cm/get-submap choices t)
                      kernel-args (into [t state] extra)
                      old-trace (tr/make-trace
                                 {:gen-fn kern :args kernel-args
                                  :choices old-sub-choices
                                  :retval nil :score (if old-step-scores (nth old-step-scores t) (mx/scalar 0.0))})
                      result (p/update kern old-trace (cm/get-submap constraints t))
                      new-trace (:trace result)
                      new-state (:retval new-trace)]
                  (recur (inc t)
                         new-state nil
                         (cm/set-choice new-choices [t] (:choices new-trace))
                         (mx/add score (:score new-trace))
                         (if (:discard result)
                           (cm/set-choice discard [t] (:discard result))
                           discard)
                         (conj states new-state)
                         (conj step-scores (:score new-trace)))))))))))

  p/IRegenerate
  (regenerate [this trace selection]
    (let [kern (:kernel this)
          cregen (compiled/get-compiled-regenerate kern)
          {:keys [args choices]} trace
          [n init-state & extra] args
          old-step-scores (::step-scores (meta trace))
          init-key (when cregen (rng/fresh-key))]
      (loop [t 0 state init-state key init-key
             new-choices cm/EMPTY score (mx/scalar 0.0) weight (mx/scalar 0.0)
             states [] step-scores []]
        (if (>= t n)
          {:trace (with-meta
                    (tr/make-trace {:gen-fn this :args args
                                    :choices new-choices :retval states :score score})
                    (cond-> {::step-scores step-scores}
                      cregen (assoc ::compiled-path true)))
           :weight weight}
          (let [old-sub-choices (cm/get-submap choices t)
                kernel-args (into [t state] extra)
                old-score (if old-step-scores (nth old-step-scores t) (mx/scalar 0.0))]
            (if cregen
              ;; WP-9A: compiled regenerate path
              (let [[k1 k2] (rng/split key)
                    result (cregen k1 (vec kernel-args) old-sub-choices
                                   (sel/get-subselection selection t))
                    step-choices (values->choices (:values result))
                    new-state (:retval result)
                    step-weight (mx/subtract (mx/subtract (:score result) old-score)
                                             (:weight result))]
                (recur (inc t) new-state k2
                       (cm/set-choice new-choices [t] step-choices)
                       (mx/add score (:score result))
                       (mx/add weight step-weight)
                       (conj states new-state)
                       (conj step-scores (:score result))))
              ;; Fallback: handler path (weight bug fixed: no - (:score trace))
              (let [old-trace (tr/make-trace
                               {:gen-fn kern :args kernel-args
                                :choices old-sub-choices
                                :retval nil :score old-score})
                    result (p/regenerate kern old-trace
                                         (sel/get-subselection selection t))
                    new-trace (:trace result)
                    new-state (:retval new-trace)]
                (recur (inc t) new-state nil
                       (cm/set-choice new-choices [t] (:choices new-trace))
                       (mx/add score (:score new-trace))
                       (mx/add weight (:weight result))
                       (conj states new-state)
                       (conj step-scores (:score new-trace)))))))))))

(defn unfold-combinator
  "Create an Unfold combinator from a kernel generative function.
   The kernel takes [t state & extra-args] and returns new-state."
  [kernel]
  (->UnfoldCombinator kernel (atom {})))

;; ---------------------------------------------------------------------------
;; Batched Unfold — IBatchedSplice for vectorized inference
;; ---------------------------------------------------------------------------
;; When spliced inside a batched handler (vsimulate/vgenerate), loops T times
;; running the kernel body-fn ONCE per step with all N particles via the
;; batched handler. O(T) kernel executions instead of O(N*T).

(def ^:private ZERO (mx/scalar 0.0))

(extend-type UnfoldCombinator
  p/IBatchedSplice
  (batched-splice [this state addr args]
    (let [kern (:kernel this)]
      (if-not (:body-fn kern)
        ;; Kernel is not a DynamicGF — fall back to generic slow path
        (h/combinator-batched-fallback state addr this (vec args))
        ;; Fast path: loop T times with batched handler
        (let [[n-steps init-state & extra] args
              batch-size (:batch-size state)
              sub-constraints (cm/get-submap (:constraints state) addr)
              [k1 k2] (rng/split (:key state))
              batch-zero (mx/zeros [batch-size])]
          (loop [t 0
                 carry init-state
                 acc-choices cm/EMPTY
                 acc-score batch-zero
                 acc-weight batch-zero
                 key k2]
            (if (>= t n-steps)
              ;; Merge accumulated result into parent state
              (let [sub-result {:choices acc-choices
                                :score acc-score
                                :weight (when (contains? state :weight) acc-weight)}
                    state' (-> state
                               (assoc :key k1)
                               (h/merge-sub-result addr sub-result))]
                [state' carry])
              ;; Run one kernel step with batched handler
              (let [[sk nk] (rng/split key)
                    step-constraints (cm/get-submap sub-constraints t)
                    has-constraints? (and step-constraints
                                          (not= step-constraints cm/EMPTY))
                    [transition init-sub]
                    (if has-constraints?
                      [h/batched-generate-transition
                       {:choices cm/EMPTY :score ZERO :weight ZERO
                        :key sk :constraints step-constraints
                        :batch-size batch-size :batched? true}]
                      [h/batched-simulate-transition
                       {:choices cm/EMPTY :score ZERO
                        :key sk :batch-size batch-size :batched? true}])
                    init-sub (if-let [ps (:param-store state)]
                               (assoc init-sub :param-store ps)
                               init-sub)
                    step-result (rt/run-handler transition init-sub
                                                (fn [rt] (apply (:body-fn kern) rt
                                                                (into [t carry] extra))))
                    step-retval (:retval step-result)]
                (recur (inc t)
                       step-retval
                       (cm/set-submap acc-choices t (:choices step-result))
                       (mx/add acc-score (:score step-result))
                       (mx/add acc-weight (or (:weight step-result) ZERO))
                       nk)))))))))

(defn unfold-empty-trace
  "Create a valid T=0 Unfold trace (no steps executed).
   Used to initialize particles for incremental unfold-extend."
  [unfold-gf init-state & extra-args]
  (let [args (into [0 init-state] extra-args)]
    (with-meta
      (tr/make-trace {:gen-fn unfold-gf :args args
                      :choices cm/EMPTY :retval [] :score (mx/scalar 0.0)})
      {::step-scores []})))

(defn unfold-extend
  "Extend an Unfold trace by ONE step, returning {:trace :weight}.
   The step-constraints are applied to the kernel's generate call.
   Calls mx/materialize! on weight and score to break lazy graph accumulation."
  [trace step-constraints key]
  (let [unfold-gf (:gen-fn trace)
        kern (:kernel unfold-gf)
        old-args (:args trace)
        [old-n init-state & extra] old-args
        prev-state (if (seq (:retval trace))
                     (last (:retval trace))
                     init-state)
        ;; Attach PRNG key — must use :genmlx.dynamic/key (not ::key)
        keyed-kern (vary-meta kern assoc :genmlx.dynamic/key key)
        ;; Generate one kernel step: t=old-n (0-indexed)
        kernel-args (into [old-n prev-state] extra)
        result (p/generate keyed-kern kernel-args step-constraints)
        step-trace (:trace result)
        step-weight (:weight result)
        step-score (:score step-trace)
        new-state (:retval step-trace)
        ;; Materialize to break lazy graph — critical for Metal buffer management
        _ (mx/materialize! step-weight step-score)
        ;; Build extended trace
        old-choices (:choices trace)
        old-score (:score trace)
        old-retval (:retval trace)
        old-step-scores (::step-scores (meta trace))
        new-choices (cm/set-choice old-choices [old-n] (:choices step-trace))
        new-score (mx/add old-score step-score)
        new-retval (conj old-retval new-state)
        new-step-scores (conj (vec old-step-scores) step-score)
        new-args (into [(inc old-n) init-state] extra)]
    {:trace (with-meta
              (tr/make-trace {:gen-fn unfold-gf :args new-args
                              :choices new-choices :retval new-retval :score new-score})
              {::step-scores new-step-scores})
     :weight step-weight}))

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
      (if-let [csim (compiled/get-compiled-simulate branch)]
        ;; L1-M5: compiled path
        (let [key (rng/fresh-key)
              result (csim key (vec branch-args))
              choices (cm/from-flat-map (:values result))]
          (with-meta
            (tr/make-trace {:gen-fn this :args args
                            :choices choices
                            :retval (:retval result)
                            :score (:score result)})
            {::switch-idx idx ::compiled-path true}))
        ;; L0: handler path
        (let [trace (p/simulate branch (vec branch-args))]
          (with-meta
            (tr/make-trace {:gen-fn this :args args
                            :choices (:choices trace)
                            :retval (:retval trace)
                            :score (:score trace)})
            {::switch-idx idx})))))

  p/IGenerate
  (generate [this args constraints]
    (let [[idx & branch-args] args
          branch (nth branches idx)]
      (if-let [cgen (compiled/get-compiled-generate branch)]
        ;; Compiled path
        (let [key (rng/fresh-key)
              result (cgen key (vec branch-args) constraints)
              choices (values->choices (:values result))]
          {:trace (with-meta
                    (tr/make-trace {:gen-fn this :args args
                                    :choices choices
                                    :retval (:retval result)
                                    :score (:score result)})
                    {::switch-idx idx ::compiled-path true})
           :weight (:weight result)})
        ;; Fallback: handler path
        (let [{:keys [trace weight]} (p/generate branch (vec branch-args) constraints)]
          {:trace (with-meta
                    (tr/make-trace {:gen-fn this :args args
                                    :choices (:choices trace)
                                    :retval (:retval trace)
                                    :score (:score trace)})
                    {::switch-idx idx})
           :weight weight})))))

(extend-type SwitchCombinator
  p/IUpdate
  (update [this trace constraints]
    (let [orig-args (:args trace)
          [new-idx & branch-args] orig-args
          old-idx (or (::switch-idx (meta trace)) new-idx)]
      (if (= old-idx new-idx)
        ;; Same branch: update in place
        (let [branch (nth (:branches this) new-idx)
              cupd (compiled/get-compiled-update branch)]
          (if cupd
            ;; WP-8: compiled update path
            (let [key (rng/fresh-key)
                  result (cupd key (vec branch-args) constraints (:choices trace))
                  new-choices (values->choices (:values result))
                  new-discard (values->choices (:discard result))]
              {:trace (with-meta
                        (tr/make-trace {:gen-fn this :args orig-args
                                        :choices new-choices
                                        :retval (:retval result)
                                        :score (:score result)})
                        {::switch-idx new-idx ::compiled-path true})
               :weight (mx/subtract (:score result) (:score trace))
               :discard new-discard})
            ;; Fallback: handler path
            (let [old-branch-trace (tr/make-trace
                                    {:gen-fn branch :args (vec branch-args)
                                     :choices (:choices trace)
                                     :retval (:retval trace) :score (:score trace)})
                  result (p/update branch old-branch-trace constraints)
                  new-branch-trace (:trace result)]
              {:trace (with-meta
                        (tr/make-trace {:gen-fn this :args orig-args
                                        :choices (:choices new-branch-trace)
                                        :retval (:retval new-branch-trace)
                                        :score (:score new-branch-trace)})
                        {::switch-idx new-idx})
               :weight (:weight result) :discard (:discard result)})))
        ;; Different branch: generate new branch from scratch
        (let [new-branch (nth (:branches this) new-idx)
              gen-result (p/generate new-branch (vec branch-args) constraints)
              new-branch-trace (:trace gen-result)
              new-score (:score new-branch-trace)]
          {:trace (with-meta
                    (tr/make-trace {:gen-fn this :args orig-args
                                    :choices (:choices new-branch-trace)
                                    :retval (:retval new-branch-trace)
                                    :score new-score})
                    {::switch-idx new-idx})
           :weight (mx/subtract new-score (:score trace))
           :discard (:choices trace)}))))

  p/IRegenerate
  (regenerate [this trace selection]
    (let [orig-args (:args trace)
          [idx & branch-args] orig-args
          branch (nth (:branches this) idx)]
      (if-let [cregen (compiled/get-compiled-regenerate branch)]
        ;; WP-9A: compiled regenerate path
        (let [key (rng/fresh-key)
              result (cregen key (vec branch-args) (:choices trace) selection)
              new-choices (values->choices (:values result))
              ;; compiled-regenerate returns proposal_ratio in :weight
              ;; weight = new_score - old_score - proposal_ratio
              weight (mx/subtract (mx/subtract (:score result) (:score trace))
                                  (:weight result))]
          {:trace (with-meta
                    (tr/make-trace {:gen-fn this :args orig-args
                                    :choices new-choices
                                    :retval (:retval result)
                                    :score (:score result)})
                    {::switch-idx idx ::compiled-path true})
           :weight weight})
        ;; Fallback: handler path
        (let [old-branch-trace (tr/make-trace
                                {:gen-fn branch :args (vec branch-args)
                                 :choices (:choices trace)
                                 :retval (:retval trace) :score (:score trace)})
              result (p/regenerate branch old-branch-trace selection)
              new-branch-trace (:trace result)]
          {:trace (with-meta
                    (tr/make-trace {:gen-fn this :args orig-args
                                    :choices (:choices new-branch-trace)
                                    :retval (:retval new-branch-trace)
                                    :score (:score new-branch-trace)})
                    {::switch-idx idx})
           :weight (:weight result)})))))

(defn switch-combinator
  "Create a Switch combinator from a vector of branch generative functions.
   The first argument selects which branch to execute."
  [& branches]
  (->SwitchCombinator (vec branches)))

;; ---------------------------------------------------------------------------
;; Batched Switch — IBatchedSplice for vectorized inference
;; ---------------------------------------------------------------------------
;; Runs ALL branches once with the batched handler, then mask-selects
;; results per particle using mx/where on the [N]-shaped index.

(defn- where-select-choicemap
  "Combine two choicemaps using mx/where on a boolean mask.
   Where mask is true, use cm-true; otherwise use cm-false.
   Requires both choicemaps to have the same address structure."
  [mask cm-true cm-false]
  (let [addrs (cm/addresses cm-true)]
    (reduce
     (fn [acc addr-path]
       (let [v-t (cm/get-choice cm-true addr-path)
             v-f (cm/get-choice cm-false addr-path)]
         (cm/set-choice acc addr-path (mx/where mask v-t v-f))))
     cm/EMPTY addrs)))

(extend-type SwitchCombinator
  p/IBatchedSplice
  (batched-splice [this state addr args]
    (let [brs (:branches this)
          all-dynamic? (every? :body-fn brs)]
      (if-not all-dynamic?
        ;; Not all branches are DynamicGF — fall back to slow path
        (h/combinator-batched-fallback state addr this (vec args))
        ;; Fast path: run all branches with batched handler, mx/where combine
        (let [[index & branch-args] args
              batch-size (:batch-size state)
              sub-constraints (cm/get-submap (:constraints state) addr)
              [k1 k2] (rng/split (:key state))
              batch-zero (mx/zeros [batch-size])
              ;; Run each branch once with batched handler
              branch-results
              (loop [i 0 results [] key k2]
                (if (>= i (count brs))
                  results
                  (let [[bk nk] (rng/split key)
                        has-constraints? (and sub-constraints
                                              (not= sub-constraints cm/EMPTY))
                        [transition init-sub]
                        (if has-constraints?
                          [h/batched-generate-transition
                           {:choices cm/EMPTY :score batch-zero :weight batch-zero
                            :key bk :constraints sub-constraints
                            :batch-size batch-size :batched? true}]
                          [h/batched-simulate-transition
                           {:choices cm/EMPTY :score batch-zero
                            :key bk :batch-size batch-size :batched? true}])
                        init-sub (if-let [ps (:param-store state)]
                                   (assoc init-sub :param-store ps)
                                   init-sub)
                        result (rt/run-handler transition init-sub
                                               (fn [rt] (apply (:body-fn (nth brs i)) rt
                                                               (vec branch-args))))]
                    (recur (inc i) (conj results result) nk))))
              ;; Combine results using mx/where based on [N]-shaped index
              combined-score
              (reduce-kv
               (fn [acc i br]
                 (let [mask (mx/equal index (mx/scalar i mx/int32))]
                   (mx/where mask (:score br) acc)))
               batch-zero branch-results)
              combined-weight
              (when (contains? state :weight)
                (reduce-kv
                 (fn [acc i br]
                   (let [mask (mx/equal index (mx/scalar i mx/int32))]
                     (mx/where mask (or (:weight br) batch-zero) acc)))
                 batch-zero branch-results))
              combined-choices
              (reduce-kv
               (fn [acc i br]
                 (let [mask (mx/equal index (mx/scalar i mx/int32))]
                   (if (zero? i)
                     (:choices br)
                     (where-select-choicemap mask (:choices br) acc))))
               cm/EMPTY branch-results)
              combined-retval
              (let [rvs (mapv :retval branch-results)]
                (when (mx/array? (first rvs))
                  (reduce-kv
                   (fn [acc i rv]
                     (let [mask (mx/equal index (mx/scalar i mx/int32))]
                       (mx/where mask rv acc)))
                   (first rvs) rvs)))
              ;; Merge into parent state
              sub-result {:choices combined-choices
                          :score combined-score
                          :weight combined-weight}
              state' (-> state
                         (assoc :key k1)
                         (h/merge-sub-result addr sub-result))]
          [state' combined-retval])))))

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

(extend-type MaskCombinator
  p/IUpdate
  (update [this trace constraints]
    (let [[active? & inner-args] (:args trace)]
      (if active?
        (let [inner (:inner this)
              old-inner-trace (tr/make-trace
                               {:gen-fn inner :args (vec inner-args)
                                :choices (:choices trace)
                                :retval (:retval trace) :score (:score trace)})
              result (p/update inner old-inner-trace constraints)
              new-trace (:trace result)]
          {:trace (tr/make-trace {:gen-fn this :args (:args trace)
                                  :choices (:choices new-trace)
                                  :retval (:retval new-trace)
                                  :score (:score new-trace)})
           :weight (:weight result) :discard (:discard result)})
        {:trace trace :weight (mx/scalar 0.0) :discard cm/EMPTY})))

  p/IRegenerate
  (regenerate [this trace selection]
    (let [[active? & inner-args] (:args trace)]
      (if active?
        (let [inner (:inner this)
              old-inner-trace (tr/make-trace
                               {:gen-fn inner :args (vec inner-args)
                                :choices (:choices trace)
                                :retval (:retval trace) :score (:score trace)})
              result (p/regenerate inner old-inner-trace selection)
              new-trace (:trace result)]
          {:trace (tr/make-trace {:gen-fn this :args (:args trace)
                                  :choices (:choices new-trace)
                                  :retval (:retval new-trace)
                                  :score (:score new-trace)})
           :weight (:weight result)})
        {:trace trace :weight (mx/scalar 0.0)}))))

;; ---------------------------------------------------------------------------
;; Recurse Combinator
;; ---------------------------------------------------------------------------
;; Fixed-point combinator for recursive generative functions.
;; Enables models that call themselves (random trees, linked lists, grammars).
;; maker: (fn [self] -> GF) — receives the combinator itself for recursion.

(defrecord RecurseCombinator [maker]
  p/IGenerativeFunction
  (simulate [this args]
    (let [gf (maker this)
          trace (p/simulate gf args)]
      (tr/make-trace {:gen-fn this :args args
                      :choices (:choices trace)
                      :retval (:retval trace)
                      :score (:score trace)})))

  p/IGenerate
  (generate [this args constraints]
    (let [gf (maker this)
          {:keys [trace weight]} (p/generate gf args constraints)]
      {:trace (tr/make-trace {:gen-fn this :args args
                              :choices (:choices trace)
                              :retval (:retval trace)
                              :score (:score trace)})
       :weight weight})))

(extend-type RecurseCombinator
  p/IUpdate
  (update [this trace constraints]
    (let [gf ((:maker this) this)
          old-inner-trace (tr/make-trace
                           {:gen-fn gf :args (:args trace)
                            :choices (:choices trace)
                            :retval (:retval trace) :score (:score trace)})
          result (p/update gf old-inner-trace constraints)
          new-trace (:trace result)]
      {:trace (tr/make-trace {:gen-fn this :args (:args trace)
                              :choices (:choices new-trace)
                              :retval (:retval new-trace)
                              :score (:score new-trace)})
       :weight (:weight result) :discard (:discard result)}))

  p/IRegenerate
  (regenerate [this trace selection]
    (let [gf ((:maker this) this)
          old-inner-trace (tr/make-trace
                           {:gen-fn gf :args (:args trace)
                            :choices (:choices trace)
                            :retval (:retval trace) :score (:score trace)})
          result (p/regenerate gf old-inner-trace selection)
          new-trace (:trace result)]
      {:trace (tr/make-trace {:gen-fn this :args (:args trace)
                              :choices (:choices new-trace)
                              :retval (:retval new-trace)
                              :score (:score new-trace)})
       :weight (:weight result)})))

(extend-type RecurseCombinator
  p/IProject
  (project [this trace selection]
    (let [gf ((:maker this) this)
          inner-trace (tr/make-trace
                       {:gen-fn gf :args (:args trace)
                        :choices (:choices trace)
                        :retval (:retval trace) :score (:score trace)})]
      (p/project gf inner-trace selection))))

(extend-type RecurseCombinator
  p/IUpdateWithDiffs
  (update-with-diffs [this trace constraints argdiffs]
    (if (and (diff/no-change? argdiffs) (= constraints cm/EMPTY))
      {:trace trace :weight (mx/scalar 0.0) :discard cm/EMPTY}
      (p/update this trace constraints))))

(extend-type RecurseCombinator
  edit/IEdit
  (edit [gf trace edit-request]
    (edit/edit-dispatch gf trace edit-request)))

(defn recurse
  "Create a Recurse combinator from a maker function.
   maker: (fn [self] -> GF) where self is the RecurseCombinator."
  [maker]
  (->RecurseCombinator maker))

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

(defn- unpack-fused-scan-outputs
  "Unpack fused scan output tensor [T, K+2] into choices, carries, and outputs.
   Columns 0..K-1: site values, K: carry, K+1: output."
  [outputs-tensor scores-tensor T addr-order]
  (let [n-sites (count addr-order)
        carry-idx n-sites
        output-idx (inc n-sites)]
    (loop [t 0
           choices cm/EMPTY
           carries []
           outputs []
           step-scores []]
      (if (>= t T)
        {:choices choices :carries carries :outputs outputs :step-scores step-scores}
        (let [row (mx/index outputs-tensor t)
              step-cm (reduce
                       (fn [scm [idx addr]]
                         (cm/set-value scm addr (mx/index row idx)))
                       cm/EMPTY
                       (map-indexed vector addr-order))
              carry (mx/index row carry-idx)
              output (mx/index row output-idx)
              step-score (mx/index scores-tensor t)]
          (recur (inc t)
                 (cm/set-choice choices [t] step-cm)
                 (conj carries carry)
                 (conj outputs output)
                 (conj step-scores step-score)))))))

(defn- get-or-build-fused-scan
  "Get cached or build new fused scan simulate."
  [cache kernel T]
  (when (and (pos? T) (compiled/fusable-kernel? kernel))
    (let [cached (get @cache T)]
      (if cached
        cached
        (when-let [fused (compiled/make-fused-scan-simulate
                          (:schema kernel) (:source kernel) T)]
          (swap! cache assoc T fused)
          fused)))))

(defn- scan-simulate-fused
  "Fused Scan simulate: compiled fn over all T steps."
  [this args kernel fused-cache init-carry inputs n]
  (let [{:keys [compiled-fn noise-dim addr-order noise-site-types]}
        (get-or-build-fused-scan fused-cache kernel n)
        key (rng/fresh-key)
        noise (compiled/generate-noise-matrix key n noise-site-types)
        [outputs-tensor scores-tensor total-score]
        (compiled-fn (mx/ensure-array init-carry)
                     (mx/stack (mapv mx/ensure-array inputs))
                     noise)
        _ (mx/materialize! outputs-tensor scores-tensor total-score)
        {:keys [choices carries outputs step-scores]}
        (unpack-fused-scan-outputs outputs-tensor scores-tensor n addr-order)]
    (with-meta
      (tr/make-trace {:gen-fn this :args args
                      :choices choices
                      :retval {:carry (last carries) :outputs outputs}
                      :score total-score})
      {::step-scores step-scores ::step-carries carries
       ::compiled-path true ::fused true})))

(defn- scan-simulate-compiled
  "Compiled Scan simulate: call compiled-simulate per step."
  [this args kernel init-carry inputs n csim]
  (let [init-key (rng/fresh-key)]
    (loop [t 0 carry init-carry key init-key
           choices cm/EMPTY score (mx/scalar 0.0)
           outputs [] step-scores [] step-carries []]
      (if (>= t n)
        (with-meta
          (tr/make-trace {:gen-fn this :args args
                          :choices choices
                          :retval {:carry carry :outputs outputs}
                          :score score})
          {::step-scores step-scores ::step-carries step-carries
           ::compiled-path true})
        (let [[k1 k2] (rng/split key)
              result (csim k1 [carry (nth inputs t)])
              [new-carry output] (:retval result)
              step-choices (cm/from-flat-map (:values result))]
          (recur (inc t)
                 new-carry k2
                 (cm/set-choice choices [t] step-choices)
                 (mx/add score (:score result))
                 (conj outputs output)
                 (conj step-scores (:score result))
                 (conj step-carries new-carry)))))))

(defn- scan-simulate-handler
  "Handler Scan simulate: delegate to kernel per step."
  [this args kernel init-carry inputs n]
  (loop [t 0 carry init-carry
         choices cm/EMPTY score (mx/scalar 0.0)
         outputs [] step-scores [] step-carries []]
    (if (>= t n)
      (with-meta
        (tr/make-trace {:gen-fn this :args args
                        :choices choices
                        :retval {:carry carry :outputs outputs}
                        :score score})
        {::step-scores step-scores ::step-carries step-carries})
      (let [trace (p/simulate kernel [carry (nth inputs t)])
            [new-carry output] (:retval trace)]
        (recur (inc t)
               new-carry
               (cm/set-choice choices [t] (:choices trace))
               (mx/add score (:score trace))
               (conj outputs output)
               (conj step-scores (:score trace))
               (conj step-carries new-carry))))))

(defrecord ScanCombinator [kernel fused-cache]
  p/IGenerativeFunction
  (simulate [this args]
    (let [[init-carry inputs] args
          n (count inputs)]
      (if-let [_fused (get-or-build-fused-scan fused-cache kernel n)]
        (scan-simulate-fused this args kernel fused-cache init-carry inputs n)
        (if-let [csim (compiled/get-compiled-simulate kernel)]
          (scan-simulate-compiled this args kernel init-carry inputs n csim)
          (scan-simulate-handler this args kernel init-carry inputs n)))))

  p/IGenerate
  (generate [this args constraints]
    (let [[init-carry inputs] args
          n (count inputs)]
      (if-let [cgen (compiled/get-compiled-generate kernel)]
        ;; Compiled path
        (let [init-key (rng/fresh-key)]
          (loop [t 0 carry init-carry key init-key
                 choices cm/EMPTY score (mx/scalar 0.0) weight (mx/scalar 0.0)
                 outputs [] step-scores [] step-carries []]
            (if (>= t n)
              {:trace (with-meta
                        (tr/make-trace {:gen-fn this :args args
                                        :choices choices
                                        :retval {:carry carry :outputs outputs}
                                        :score score})
                        {::step-scores step-scores ::step-carries step-carries
                         ::compiled-path true})
               :weight weight}
              (let [[k1 k2] (rng/split key)
                    result (cgen k1 [carry (nth inputs t)]
                                 (cm/get-submap constraints t))
                    [new-carry output] (:retval result)
                    step-choices (values->choices (:values result))]
                (recur (inc t) new-carry k2
                       (cm/set-choice choices [t] step-choices)
                       (mx/add score (:score result))
                       (mx/add weight (:weight result))
                       (conj outputs output)
                       (conj step-scores (:score result))
                       (conj step-carries new-carry))))))
        ;; Fallback: handler path
        (loop [t 0 carry init-carry
               choices cm/EMPTY score (mx/scalar 0.0) weight (mx/scalar 0.0)
               outputs [] step-scores [] step-carries []]
          (if (>= t n)
            {:trace (with-meta
                      (tr/make-trace {:gen-fn this :args args
                                      :choices choices
                                      :retval {:carry carry :outputs outputs}
                                      :score score})
                      {::step-scores step-scores ::step-carries step-carries})
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
                     (conj outputs output)
                     (conj step-scores (:score (:trace result)))
                     (conj step-carries new-carry)))))))))

(extend-type ScanCombinator
  p/IUpdate
  (update [this trace constraints]
    (let [kern (:kernel this)
          cupd (compiled/get-compiled-update kern)
          {:keys [args choices]} trace
          [init-carry inputs] args
          n (count inputs)
          old-step-scores (::step-scores (meta trace))
          old-step-carries (::step-carries (meta trace))
          ;; Find first step with non-empty constraints (prefix-skip boundary)
          first-changed (if (and old-step-scores old-step-carries)
                          (loop [t 0]
                            (cond
                              (>= t n) n
                              (not= (cm/get-submap constraints t) cm/EMPTY) t
                              :else (recur (inc t))))
                          0)]
      ;; If no steps have constraints and we have metadata, return trace unchanged
      (if (and old-step-scores old-step-carries (= first-changed n))
        {:trace trace :weight (mx/scalar 0.0) :discard cm/EMPTY}
        ;; Build prefix from old trace (steps 0..first-changed-1)
        (let [prefix-choices (if (pos? first-changed)
                               (reduce (fn [cm t]
                                         (cm/set-choice cm [t] (cm/get-submap choices t)))
                                       cm/EMPTY (range first-changed))
                               cm/EMPTY)
              prefix-score (if (pos? first-changed)
                             (reduce (fn [acc t] (mx/add acc (nth old-step-scores t)))
                                     (mx/scalar 0.0) (range first-changed))
                             (mx/scalar 0.0))
              prefix-outputs (if (pos? first-changed)
                               (subvec (:outputs (:retval trace)) 0 first-changed)
                               [])
              prefix-step-scores (if (pos? first-changed)
                                   (subvec (vec old-step-scores) 0 first-changed)
                                   [])
              prefix-step-carries (if (pos? first-changed)
                                    (subvec (vec old-step-carries) 0 first-changed)
                                    [])
              start-carry (if (pos? first-changed)
                            (nth old-step-carries (dec first-changed))
                            init-carry)
              init-key (when cupd (rng/fresh-key))]
          ;; Execute steps first-changed..n-1
          (loop [t first-changed carry start-carry key init-key
                 new-choices prefix-choices score prefix-score
                 discard cm/EMPTY
                 outputs prefix-outputs
                 step-scores prefix-step-scores step-carries prefix-step-carries]
            (if (>= t n)
              {:trace (with-meta
                        (tr/make-trace {:gen-fn this :args args
                                        :choices new-choices
                                        :retval {:carry carry :outputs outputs}
                                        :score score})
                        (cond-> {::step-scores step-scores ::step-carries step-carries}
                          cupd (assoc ::compiled-path true)))
               :weight (mx/subtract score (:score trace)) :discard discard}
              (if cupd
                ;; WP-8: compiled update path
                (let [[k1 k2] (rng/split key)
                      old-sub-choices (cm/get-submap choices t)
                      result (cupd k1 [carry (nth inputs t)]
                                   (cm/get-submap constraints t) old-sub-choices)
                      step-choices (values->choices (:values result))
                      step-discard (values->choices (:discard result))
                      [new-carry output] (:retval result)]
                  (recur (inc t) new-carry k2
                         (cm/set-choice new-choices [t] step-choices)
                         (mx/add score (:score result))
                         (if (seq (:discard result))
                           (cm/set-choice discard [t] step-discard)
                           discard)
                         (conj outputs output)
                         (conj step-scores (:score result))
                         (conj step-carries new-carry)))
                ;; Fallback: handler path
                (let [old-sub-choices (cm/get-submap choices t)
                      old-trace (tr/make-trace
                                 {:gen-fn kern :args [carry (nth inputs t)]
                                  :choices old-sub-choices
                                  :retval nil :score (if old-step-scores (nth old-step-scores t) (mx/scalar 0.0))})
                      result (p/update kern old-trace (cm/get-submap constraints t))
                      new-trace (:trace result)
                      [new-carry output] (:retval new-trace)]
                  (recur (inc t)
                         new-carry nil
                         (cm/set-choice new-choices [t] (:choices new-trace))
                         (mx/add score (:score new-trace))
                         (if (:discard result)
                           (cm/set-choice discard [t] (:discard result))
                           discard)
                         (conj outputs output)
                         (conj step-scores (:score new-trace))
                         (conj step-carries new-carry))))))))))

  p/IRegenerate
  (regenerate [this trace selection]
    (let [kern (:kernel this)
          cregen (compiled/get-compiled-regenerate kern)
          {:keys [args choices]} trace
          [init-carry inputs] args
          n (count inputs)
          old-step-scores (::step-scores (meta trace))
          init-key (when cregen (rng/fresh-key))]
      (loop [t 0 carry init-carry key init-key
             new-choices cm/EMPTY score (mx/scalar 0.0) weight (mx/scalar 0.0)
             outputs [] step-scores [] step-carries []]
        (if (>= t n)
          {:trace (with-meta
                    (tr/make-trace {:gen-fn this :args args
                                    :choices new-choices
                                    :retval {:carry carry :outputs outputs}
                                    :score score})
                    (cond-> {::step-scores step-scores ::step-carries step-carries}
                      cregen (assoc ::compiled-path true)))
           :weight weight}
          (let [old-sub-choices (cm/get-submap choices t)
                kernel-args [carry (nth inputs t)]
                old-score (if old-step-scores (nth old-step-scores t) (mx/scalar 0.0))]
            (if cregen
              ;; WP-9A: compiled regenerate path
              (let [[k1 k2] (rng/split key)
                    result (cregen k1 (vec kernel-args) old-sub-choices
                                   (sel/get-subselection selection t))
                    step-choices (values->choices (:values result))
                    [new-carry output] (:retval result)
                    step-weight (mx/subtract (mx/subtract (:score result) old-score)
                                             (:weight result))]
                (recur (inc t) new-carry k2
                       (cm/set-choice new-choices [t] step-choices)
                       (mx/add score (:score result))
                       (mx/add weight step-weight)
                       (conj outputs output)
                       (conj step-scores (:score result))
                       (conj step-carries new-carry)))
              ;; Fallback: handler path (weight bug fixed: no - (:score trace))
              (let [old-trace (tr/make-trace
                               {:gen-fn kern :args kernel-args
                                :choices old-sub-choices
                                :retval nil :score old-score})
                    result (p/regenerate kern old-trace
                                         (sel/get-subselection selection t))
                    new-trace (:trace result)
                    [new-carry output] (:retval new-trace)]
                (recur (inc t) new-carry nil
                       (cm/set-choice new-choices [t] (:choices new-trace))
                       (mx/add score (:score new-trace))
                       (mx/add weight (:weight result))
                       (conj outputs output)
                       (conj step-scores (:score new-trace))
                       (conj step-carries new-carry))))))))))

(defn scan-combinator
  "Create a Scan combinator from a kernel generative function.
   The kernel takes [carry input] and returns [new-carry output].
   The scan applies the kernel to each element of an input sequence,
   threading carry-state and accumulating outputs."
  [kernel]
  (->ScanCombinator kernel (atom {})))

;; ---------------------------------------------------------------------------
;; Batched Scan — IBatchedSplice for vectorized inference
;; ---------------------------------------------------------------------------

(extend-type ScanCombinator
  p/IBatchedSplice
  (batched-splice [this state addr args]
    (let [kern (:kernel this)]
      (if-not (:body-fn kern)
        (h/combinator-batched-fallback state addr this (vec args))
        (let [[init-carry inputs] args
              n-steps (count inputs)
              batch-size (:batch-size state)
              sub-constraints (cm/get-submap (:constraints state) addr)
              [k1 k2] (rng/split (:key state))
              batch-zero (mx/zeros [batch-size])]
          (loop [t 0
                 carry init-carry
                 acc-choices cm/EMPTY
                 acc-score batch-zero
                 acc-weight batch-zero
                 key k2]
            (if (>= t n-steps)
              (let [sub-result {:choices acc-choices
                                :score acc-score
                                :weight (when (contains? state :weight) acc-weight)}
                    state' (-> state
                               (assoc :key k1)
                               (h/merge-sub-result addr sub-result))]
                [state' carry])
              (let [[sk nk] (rng/split key)
                    step-constraints (cm/get-submap sub-constraints t)
                    has-constraints? (and step-constraints
                                          (not= step-constraints cm/EMPTY))
                    [transition init-sub]
                    (if has-constraints?
                      [h/batched-generate-transition
                       {:choices cm/EMPTY :score batch-zero :weight batch-zero
                        :key sk :constraints step-constraints
                        :batch-size batch-size :batched? true}]
                      [h/batched-simulate-transition
                       {:choices cm/EMPTY :score batch-zero
                        :key sk :batch-size batch-size :batched? true}])
                    init-sub (if-let [ps (:param-store state)]
                               (assoc init-sub :param-store ps)
                               init-sub)
                    step-result (rt/run-handler transition init-sub
                                                (fn [rt] ((:body-fn kern) rt carry (nth inputs t))))
                    [new-carry _output] (:retval step-result)]
                (recur (inc t)
                       new-carry
                       (cm/set-submap acc-choices t (:choices step-result))
                       (mx/add acc-score (:score step-result))
                       (mx/add acc-weight (or (:weight step-result) batch-zero))
                       nk)))))))))

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
          component (nth components (int idx))]
      (if-let [csim (compiled/get-compiled-simulate component)]
        ;; L1-M5: compiled path
        (let [key (rng/fresh-key)
              result (csim key (vec args))
              choices (cm/from-flat-map (:values result))
              choices (cm/set-choice choices [:component-idx]
                                     (mx/scalar (int idx) mx/int32))]
          (with-meta
            (tr/make-trace {:gen-fn this :args args
                            :choices choices
                            :retval (:retval result)
                            :score (mx/add (:score result) (:score idx-trace))})
            {::compiled-path true}))
        ;; L0: handler path
        (let [comp-trace (p/simulate component args)]
          (tr/make-trace {:gen-fn this :args args
                          :choices (cm/set-choice (:choices comp-trace)
                                                  [:component-idx]
                                                  (mx/scalar (int idx) mx/int32))
                          :retval (:retval comp-trace)
                          :score (mx/add (:score comp-trace) (:score idx-trace))})))))

  p/IGenerate
  (generate [this args constraints]
    (let [log-w (log-weights-fn args)
          ;; Check if component index is constrained
          idx-constraint (cm/get-submap constraints :component-idx)
          idx-result (if (cm/has-value? idx-constraint)
                       (let [d (dc/->Distribution :categorical {:logits log-w})]
                         (dc/dist-generate d idx-constraint))
                       (let [d (dc/->Distribution :categorical {:logits log-w})]
                         {:trace (dc/dist-simulate d) :weight (mx/scalar 0.0)}))
          idx (mx/item (cm/get-value (:choices (:trace idx-result))))
          component (nth components (int idx))
          comp-constraints (if (instance? cm/Node constraints)
                             (cm/->Node (dissoc (:m constraints) :component-idx))
                             constraints)]
      (if-let [cgen (compiled/get-compiled-generate component)]
        ;; Compiled path — only the component generate is compiled
        (let [key (rng/fresh-key)
              result (cgen key (vec args) comp-constraints)
              comp-choices (values->choices (:values result))]
          {:trace (with-meta
                    (tr/make-trace {:gen-fn this :args args
                                    :choices (cm/set-choice comp-choices
                                                            [:component-idx]
                                                            (mx/scalar (int idx) mx/int32))
                                    :retval (:retval result)
                                    :score (mx/add (:score result)
                                                   (:score (:trace idx-result)))})
                    {::compiled-path true})
           :weight (mx/add (:weight result) (:weight idx-result))})
        ;; Fallback: handler path
        (let [{:keys [trace weight]} (p/generate component args comp-constraints)]
          {:trace (tr/make-trace {:gen-fn this :args args
                                  :choices (cm/set-choice (:choices trace)
                                                          [:component-idx]
                                                          (mx/scalar (int idx) mx/int32))
                                  :retval (:retval trace)
                                  :score (mx/add (:score trace)
                                                 (:score (:trace idx-result)))})
           :weight (mx/add weight (:weight idx-result))})))))

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

;; ---------------------------------------------------------------------------
;; Batched Mix — IBatchedSplice for vectorized inference
;; ---------------------------------------------------------------------------
;; Like Switch: runs ALL components once with batched handler, mask-selects
;; per particle. Additionally samples [N]-shaped component indices from the
;; categorical distribution over log-weights and accounts for index score.

(extend-type MixCombinator
  p/IBatchedSplice
  (batched-splice [this state addr args]
    (let [comps (:components this)
          all-dynamic? (every? :body-fn comps)]
      (if-not all-dynamic?
        (h/combinator-batched-fallback state addr this (vec args))
        ;; Fast path
        (let [batch-size (:batch-size state)
              log-w ((:log-weights-fn this) args)
              sub-constraints (cm/get-submap (:constraints state) addr)
              ;; Check if component-idx is constrained
              idx-constraint (when (and sub-constraints
                                        (not= sub-constraints cm/EMPTY))
                               (cm/get-submap sub-constraints :component-idx))
              ;; Sample or constrain [N]-shaped component indices
              cat-dist (dc/->Distribution :categorical {:logits log-w})
              [k1 k2] (rng/split (:key state))
              [idx-vals idx-score idx-weight]
              (if (and idx-constraint (cm/has-value? idx-constraint))
                ;; Constrained: fixed value, weight = log-prob
                (let [v (cm/get-value idx-constraint)
                      lp (dc/dist-log-prob cat-dist v)]
                  [v lp lp])
                ;; Unconstrained: sample [N] indices
                (let [sampled (dc/dist-sample-n cat-dist k2 batch-size)
                      lp (dc/dist-log-prob cat-dist sampled)]
                  [sampled lp (mx/zeros [batch-size])]))
              ;; Inner constraints = everything except :component-idx
              inner-constraints
              (if (and sub-constraints (not= sub-constraints cm/EMPTY)
                       (instance? cm/Node sub-constraints))
                (let [inner-m (dissoc (:m sub-constraints) :component-idx)]
                  (if (empty? inner-m) cm/EMPTY (cm/->Node inner-m)))
                (or sub-constraints cm/EMPTY))
              batch-zero (mx/zeros [batch-size])
              ;; Run each component once with batched handler
              comp-results
              (loop [i 0 results [] key k1]
                (if (>= i (count comps))
                  results
                  (let [[ck nk] (rng/split key)
                        has-inner? (and inner-constraints
                                        (not= inner-constraints cm/EMPTY))
                        [transition init-sub]
                        (if has-inner?
                          [h/batched-generate-transition
                           {:choices cm/EMPTY :score batch-zero :weight batch-zero
                            :key ck :constraints inner-constraints
                            :batch-size batch-size :batched? true}]
                          [h/batched-simulate-transition
                           {:choices cm/EMPTY :score batch-zero
                            :key ck :batch-size batch-size :batched? true}])
                        init-sub (if-let [ps (:param-store state)]
                                   (assoc init-sub :param-store ps)
                                   init-sub)
                        result (rt/run-handler transition init-sub
                                               (fn [rt] (apply (:body-fn (nth comps i)) rt
                                                               (vec args))))]
                    (recur (inc i) (conj results result) nk))))
              ;; Combine per-particle results using mx/where on idx-vals
              combined-score
              (reduce-kv
               (fn [acc i cr]
                 (let [mask (mx/equal idx-vals (mx/scalar i mx/int32))]
                   (mx/where mask (:score cr) acc)))
               batch-zero comp-results)
              combined-weight
              (when (contains? state :weight)
                (reduce-kv
                 (fn [acc i cr]
                   (let [mask (mx/equal idx-vals (mx/scalar i mx/int32))]
                     (mx/where mask (or (:weight cr) batch-zero) acc)))
                 batch-zero comp-results))
              combined-choices
              (reduce-kv
               (fn [acc i cr]
                 (let [mask (mx/equal idx-vals (mx/scalar i mx/int32))]
                   (if (zero? i)
                     (:choices cr)
                     (where-select-choicemap mask (:choices cr) acc))))
               cm/EMPTY comp-results)
              combined-retval
              (let [rvs (mapv :retval comp-results)]
                (when (mx/array? (first rvs))
                  (reduce-kv
                   (fn [acc i rv]
                     (let [mask (mx/equal idx-vals (mx/scalar i mx/int32))]
                       (mx/where mask rv acc)))
                   (first rvs) rvs)))
              ;; Add component-idx to choices + add idx-score to combined score
              final-choices (cm/set-value combined-choices :component-idx idx-vals)
              final-score (mx/add combined-score idx-score)
              final-weight (when combined-weight
                             (mx/add combined-weight idx-weight))
              ;; Merge into parent state
              sub-result {:choices final-choices
                          :score final-score
                          :weight final-weight}
              state' (-> state
                         (assoc :key k2)
                         (h/merge-sub-result addr sub-result))]
          [state' combined-retval])))))

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
              cupd (compiled/get-compiled-update component)]
          (if cupd
            ;; WP-8: compiled update path
            (let [key (rng/fresh-key)
                  result (cupd key (vec args) inner-constraints inner-old-choices)
                  inner-score (:score result)
                  new-score (mx/add inner-score old-idx-score)
                  new-choices (cm/set-choice (values->choices (:values result))
                                             [:component-idx]
                                             (mx/scalar old-idx mx/int32))
                  new-discard (values->choices (:discard result))]
              {:trace (with-meta
                        (tr/make-trace {:gen-fn this :args args
                                        :choices new-choices
                                        :retval (:retval result)
                                        :score new-score})
                        {::compiled-path true})
               :weight (mx/subtract new-score (:score trace))
               :discard new-discard})
            ;; Fallback: handler path
            (let [inner-old-score (mx/subtract (:score trace) old-idx-score)
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
               :discard (:discard result)})))
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
              inner-old-choices (cm/->Node (dissoc (:m old-choices) :component-idx))]
          (if-let [cregen (compiled/get-compiled-regenerate component)]
            ;; WP-9A: compiled regenerate path
            (let [key (rng/fresh-key)
                  result (cregen key (vec args) inner-old-choices selection)
                  inner-score (:score result)
                  new-score (mx/add inner-score old-idx-score)
                  new-choices (cm/set-choice (values->choices (:values result))
                                             [:component-idx]
                                             (mx/scalar old-idx mx/int32))
                  ;; weight = new_score - old_score - proposal_ratio
                  proposal-ratio (:weight result)
                  weight (mx/subtract (mx/subtract new-score (:score trace)) proposal-ratio)]
              {:trace (with-meta
                        (tr/make-trace {:gen-fn this :args args
                                        :choices new-choices
                                        :retval (:retval result)
                                        :score new-score})
                        {::compiled-path true})
               :weight weight})
            ;; Fallback: handler path
            (let [inner-old-trace (tr/make-trace {:gen-fn component :args args
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
               :weight (:weight result)})))))))

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
                                             :retval nil :score (if old-element-scores (nth old-element-scores i) (mx/scalar 0.0))})]
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
    (cond
      ;; No changes at all: fast path
      (and (diff/no-change? argdiffs) (= constraints cm/EMPTY))
      {:trace trace :weight (mx/scalar 0.0) :discard cm/EMPTY}

      ;; Args changed: full update (strip step-scores to prevent invalid prefix-skip)
      (not (diff/no-change? argdiffs))
      (p/update this (with-meta trace (dissoc (meta trace) ::step-scores)) constraints)

      ;; Only constraints changed: prefix-skip via p/update
      :else
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
    (cond
      ;; No changes at all: fast path
      (and (diff/no-change? argdiffs) (= constraints cm/EMPTY))
      {:trace trace :weight (mx/scalar 0.0) :discard cm/EMPTY}

      ;; Args changed: full update (strip metadata to prevent invalid prefix-skip)
      (not (diff/no-change? argdiffs))
      (p/update this (with-meta trace (dissoc (meta trace) ::step-scores ::step-carries)) constraints)

      ;; Only constraints changed: prefix-skip via p/update
      :else
      (p/update this trace constraints))))

(extend-type MaskCombinator
  p/IUpdateWithDiffs
  (update-with-diffs [this trace constraints argdiffs]
    (if (and (diff/no-change? argdiffs) (= constraints cm/EMPTY))
      {:trace trace :weight (mx/scalar 0.0) :discard cm/EMPTY}
      (p/update this trace constraints))))

(extend-type ContramapGF
  p/IUpdateWithDiffs
  (update-with-diffs [this trace constraints argdiffs]
    (if (and (diff/no-change? argdiffs) (= constraints cm/EMPTY))
      {:trace trace :weight (mx/scalar 0.0) :discard cm/EMPTY}
      (p/update this trace constraints))))

(extend-type MapRetvalGF
  p/IUpdateWithDiffs
  (update-with-diffs [this trace constraints argdiffs]
    (if (and (diff/no-change? argdiffs) (= constraints cm/EMPTY))
      {:trace trace :weight (mx/scalar 0.0) :discard cm/EMPTY}
      (p/update this trace constraints))))

(extend-type MixCombinator
  p/IUpdateWithDiffs
  (update-with-diffs [this trace constraints argdiffs]
    (if (and (diff/no-change? argdiffs) (= constraints cm/EMPTY))
      {:trace trace :weight (mx/scalar 0.0) :discard cm/EMPTY}
      (p/update this trace constraints))))

;; ---------------------------------------------------------------------------
;; IProject implementations
;; ---------------------------------------------------------------------------

(extend-type MapCombinator
  p/IProject
  (project [this trace selection]
    (let [old-choices (:choices trace)
          args (:args trace)
          n (count (first args))
          kernel (:kernel this)
          old-element-scores (::element-scores (meta trace))]
      (reduce (fn [acc i]
                (let [kernel-args (mapv #(nth % i) args)
                      sub-trace (tr/make-trace
                                 {:gen-fn kernel :args kernel-args
                                  :choices (cm/get-submap old-choices i)
                                  :retval (nth (:retval trace) i nil)
                                  :score (if old-element-scores (nth old-element-scores i) (mx/scalar 0.0))})
                      w (p/project kernel sub-trace
                                   (sel/get-subselection selection i))]
                  (mx/add acc w)))
              (mx/scalar 0.0)
              (range n)))))

(extend-type UnfoldCombinator
  p/IProject
  (project [this trace selection]
    (let [kern (:kernel this)
          {:keys [args choices]} trace
          [n init-state & extra] args]
      (loop [t 0 state init-state
             weight (mx/scalar 0.0)]
        (if (>= t n)
          weight
          (let [kernel-args (into [t state] extra)
                sub-choices (cm/get-submap choices t)
                ;; Replay via generate to recover retval (carry state)
                {:keys [trace]} (p/generate kern kernel-args sub-choices)
                new-state (:retval trace)
                w (p/project kern trace
                             (sel/get-subselection selection t))]
            (recur (inc t) new-state (mx/add weight w))))))))

(extend-type SwitchCombinator
  p/IProject
  (project [this trace selection]
    (let [[idx & branch-args] (:args trace)
          branch (nth (:branches this) idx)
          branch-trace (tr/make-trace
                        {:gen-fn branch :args (vec branch-args)
                         :choices (:choices trace)
                         :retval (:retval trace) :score (:score trace)})]
      (p/project branch branch-trace selection))))

(extend-type ScanCombinator
  p/IProject
  (project [this trace selection]
    (let [kern (:kernel this)
          {:keys [args choices]} trace
          [init-carry inputs] args
          n (count inputs)]
      (loop [t 0 carry init-carry
             weight (mx/scalar 0.0)]
        (if (>= t n)
          weight
          (let [sub-choices (cm/get-submap choices t)
                {:keys [trace]} (p/generate kern [carry (nth inputs t)] sub-choices)
                [new-carry _output] (:retval trace)
                w (p/project kern trace
                             (sel/get-subselection selection t))]
            (recur (inc t) new-carry (mx/add weight w))))))))

(extend-type MaskCombinator
  p/IProject
  (project [this trace selection]
    (let [[active? & inner-args] (:args trace)]
      (if active?
        (let [inner-trace (tr/make-trace
                           {:gen-fn (:inner this) :args (vec inner-args)
                            :choices (:choices trace)
                            :retval (:retval trace) :score (:score trace)})]
          (p/project (:inner this) inner-trace selection))
        (mx/scalar 0.0)))))

(extend-type MixCombinator
  p/IProject
  (project [this trace selection]
    (let [old-choices (:choices trace)
          old-idx (int (mx/item (cm/get-choice old-choices [:component-idx])))
          args (:args trace)
          log-w ((:log-weights-fn this) args)
          idx-dist (dc/->Distribution :categorical {:logits log-w})
          ;; Project the component-idx if selected
          idx-weight (if (sel/selected? selection :component-idx)
                       (dc/dist-log-prob idx-dist (mx/scalar old-idx mx/int32))
                       (mx/scalar 0.0))
          ;; Project the inner component
          component (nth (:components this) old-idx)
          old-idx-score (dc/dist-log-prob idx-dist (mx/scalar old-idx mx/int32))
          inner-old-choices (cm/->Node (dissoc (:m old-choices) :component-idx))
          inner-old-score (mx/subtract (:score trace) old-idx-score)
          inner-trace (tr/make-trace {:gen-fn component :args args
                                      :choices inner-old-choices
                                      :retval (:retval trace) :score inner-old-score})
          inner-weight (p/project component inner-trace selection)]
      (mx/add idx-weight inner-weight))))

(extend-type ContramapGF
  p/IProject
  (project [this trace selection]
    (let [transformed-args ((:f this) (:args trace))
          inner-trace (tr/make-trace {:gen-fn (:inner this) :args transformed-args
                                      :choices (:choices trace)
                                      :retval (:retval trace) :score (:score trace)})]
      (p/project (:inner this) inner-trace selection))))

(extend-type MapRetvalGF
  p/IProject
  (project [this trace selection]
    (let [inner-trace (tr/make-trace {:gen-fn (:inner this) :args (:args trace)
                                      :choices (:choices trace)
                                      :retval nil :score (:score trace)})]
      (p/project (:inner this) inner-trace selection))))

;; ---------------------------------------------------------------------------
;; IAssess and IPropose implementations
;; ---------------------------------------------------------------------------

(extend-type MapCombinator
  p/IAssess
  (assess [this args choices]
    (let [n (count (first args))
          results (mapv (fn [i]
                          (p/assess (:kernel this)
                                    (mapv #(nth % i) args)
                                    (cm/get-submap choices i)))
                        (range n))]
      {:retval (mapv :retval results)
       :weight (sum-field results :weight)}))

  p/IPropose
  (propose [this args]
    (let [n (count (first args))
          results (mapv (fn [i]
                          (p/propose (:kernel this)
                                     (mapv #(nth % i) args)))
                        (range n))]
      {:choices (assemble-choices results :choices)
       :weight (sum-field results :weight)
       :retval (mapv :retval results)})))

(extend-type UnfoldCombinator
  p/IAssess
  (assess [this args choices]
    (let [[n init-state & extra] args]
      (loop [t 0 state init-state weight (mx/scalar 0.0) states []]
        (if (>= t n)
          {:retval states :weight weight}
          (let [result (p/assess (:kernel this)
                                 (into [t state] extra)
                                 (cm/get-submap choices t))
                new-state (:retval result)]
            (recur (inc t) new-state
                   (mx/add weight (:weight result))
                   (conj states new-state)))))))

  p/IPropose
  (propose [this args]
    (let [[n init-state & extra] args]
      (loop [t 0 state init-state
             choices cm/EMPTY weight (mx/scalar 0.0) states []]
        (if (>= t n)
          {:choices choices :weight weight :retval states}
          (let [result (p/propose (:kernel this)
                                  (into [t state] extra))
                new-state (:retval result)]
            (recur (inc t) new-state
                   (cm/set-choice choices [t] (:choices result))
                   (mx/add weight (:weight result))
                   (conj states new-state))))))))

(extend-type SwitchCombinator
  p/IAssess
  (assess [this args choices]
    (let [[idx & branch-args] args]
      (p/assess (nth (:branches this) idx) (vec branch-args) choices)))

  p/IPropose
  (propose [this args]
    (let [[idx & branch-args] args]
      (p/propose (nth (:branches this) (int idx)) (vec branch-args)))))

(extend-type ScanCombinator
  p/IAssess
  (assess [this args choices]
    (let [[init-carry inputs] args
          n (count inputs)]
      (loop [t 0 carry init-carry weight (mx/scalar 0.0) outputs []]
        (if (>= t n)
          {:retval {:carry carry :outputs outputs} :weight weight}
          (let [result (p/assess (:kernel this)
                                 [carry (nth inputs t)]
                                 (cm/get-submap choices t))
                [new-carry output] (:retval result)]
            (recur (inc t) new-carry
                   (mx/add weight (:weight result))
                   (conj outputs output)))))))

  p/IPropose
  (propose [this args]
    (let [[init-carry inputs] args
          n (count inputs)]
      (loop [t 0 carry init-carry
             choices cm/EMPTY weight (mx/scalar 0.0) outputs []]
        (if (>= t n)
          {:choices choices :weight weight
           :retval {:carry carry :outputs outputs}}
          (let [result (p/propose (:kernel this)
                                  [carry (nth inputs t)])
                [new-carry output] (:retval result)]
            (recur (inc t) new-carry
                   (cm/set-choice choices [t] (:choices result))
                   (mx/add weight (:weight result))
                   (conj outputs output))))))))

(extend-type MaskCombinator
  p/IAssess
  (assess [this args choices]
    (let [[active? & inner-args] args]
      (if active?
        (p/assess (:inner this) (vec inner-args) choices)
        {:retval nil :weight (mx/scalar 0.0)})))

  p/IPropose
  (propose [this args]
    (let [[active? & inner-args] args]
      (if active?
        (p/propose (:inner this) (vec inner-args))
        {:choices cm/EMPTY :weight (mx/scalar 0.0) :retval nil}))))

(extend-type RecurseCombinator
  p/IAssess
  (assess [this args choices]
    (p/assess ((:maker this) this) args choices))

  p/IPropose
  (propose [this args]
    (p/propose ((:maker this) this) args)))

(extend-type ContramapGF
  p/IAssess
  (assess [this args choices]
    (p/assess (:inner this) ((:f this) args) choices))

  p/IPropose
  (propose [this args]
    (p/propose (:inner this) ((:f this) args))))

(extend-type MapRetvalGF
  p/IAssess
  (assess [this args choices]
    (let [result (p/assess (:inner this) args choices)]
      {:retval ((:g this) (:retval result))
       :weight (:weight result)}))

  p/IPropose
  (propose [this args]
    (let [result (p/propose (:inner this) args)]
      {:choices (:choices result)
       :weight (:weight result)
       :retval ((:g this) (:retval result))})))

(extend-type MixCombinator
  p/IAssess
  (assess [this args choices]
    (let [log-w ((:log-weights-fn this) args)
          idx-val (cm/get-choice choices [:component-idx])
          idx (int (mx/item idx-val))
          idx-dist (dc/->Distribution :categorical {:logits log-w})
          idx-weight (dc/dist-log-prob idx-dist (mx/scalar idx mx/int32))
          component (nth (:components this) idx)
          inner-choices (if (instance? cm/Node choices)
                          (cm/->Node (dissoc (:m choices) :component-idx))
                          choices)
          inner-result (p/assess component args inner-choices)]
      {:retval (:retval inner-result)
       :weight (mx/add idx-weight (:weight inner-result))}))

  p/IPropose
  (propose [this args]
    (let [log-w ((:log-weights-fn this) args)
          idx-dist (dc/->Distribution :categorical {:logits log-w})
          idx-propose (dc/dist-propose idx-dist)
          idx (int (mx/item (cm/get-value (:choices idx-propose))))
          component (nth (:components this) idx)
          comp-result (p/propose component args)]
      {:choices (cm/set-choice (:choices comp-result)
                               [:component-idx]
                               (mx/scalar idx mx/int32))
       :weight (mx/add (:weight idx-propose) (:weight comp-result))
       :retval (:retval comp-result)})))
