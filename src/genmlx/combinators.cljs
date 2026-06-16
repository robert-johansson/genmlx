(ns genmlx.combinators
  "GFI combinators: Map, Unfold, Switch, Scan, Mask, Mix, and more.
   These compose generative functions into higher-level models."
  (:require [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.mlx.constants :refer [ZERO]]
            [genmlx.selection :as sel]
            [genmlx.handler :as h]
            [genmlx.runtime :as rt]
            [genmlx.dist.core :as dc]
            [genmlx.edit :as edit]
            [genmlx.diff :as diff]
            [genmlx.compiled-ops :as cops]))

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

(defn- ensure-kernel-key
  "Give the kernel an auto-key when it carries no PRNG key, so handler-path
   sub-calls (update/regenerate fallbacks) can run instead of throwing
   'No PRNG key on gen-fn'. Uses the metadata literal to avoid requiring
   genmlx.dynamic (circular) — same convention as unfold-extend."
  [kern]
  (if (:genmlx.dynamic/key (meta kern))
    kern
    (vary-meta kern assoc :genmlx.dynamic/key :genmlx.dynamic/auto-key)))

;; ---------------------------------------------------------------------------
;; Score-type propagation (genmlx-lbae)
;;
;; Combinator records bypass the DynamicGF dispatcher stack, so the
;; score-type machinery (tag producers, convert-or-throw at joint-scoring
;; boundaries — ARCHITECTURE §3.3, genmlx-pkmx) is applied here:
;;   - result traces are tagged with the lub of their element/sub-trace
;;     score-types (tag-from-traces), or :joint on compiled/fused element
;;     paths which cannot fire the analytical path (tag-joint)
;;   - update/regenerate/project entries convert a :marginal self-trace
;;     before any recorded old score is reused (ensure-joint-self)
;; ---------------------------------------------------------------------------

(defn- tag-joint
  "Tag a combinator result trace :joint — compiled/fused element paths are
   joint by construction."
  [trace]
  (tr/with-score-type trace :joint))

(defn- tag-from-traces
  "Tag a combinator result trace with the lub of its element/sub-trace
   score-types: a marginal sub-score summed into a composite total makes
   the composite score marginal."
  [trace sub-traces]
  (tr/with-score-type
    trace
    (reduce tr/combine-score-types :joint (map tr/score-type sub-traces))))

(defn- ensure-joint-self
  "Score-type boundary for combinator update/regenerate/project. A
   :marginal self-trace re-generates fully constrained from its own
   choices: sub-gfs fall through to plain joint scoring when their latents
   are constrained (genmlx-b470), restoring joint element scores before any
   recorded old score is reused in a weight. Throws when no joint
   conversion exists (a :collapsed sub-gf reproduces a collapsed score).
   No-op for joint traces."
  [this op trace]
  (let [st (tr/score-type trace)]
    (if (= :joint st)
      trace
      (let [converted (:trace (p/generate (ensure-kernel-key this)
                                          (:args trace) (:choices trace)))]
        (if (= :joint (tr/score-type converted))
          converted
          (throw (ex-info
                   (str "Combinator " op " cannot consume a " st
                        "-scored trace — re-generating its choices does not"
                        " yield a joint score")
                   {:genmlx/error :score-type-mismatch
                    :op op :score-type st :expected :joint})))))))

(defn- assemble-indexed-discards
  "Collect non-empty per-element discards under their ORIGINAL element
   indices. Filtering before positional reassembly records element i's
   discard under a compacted (wrong) index, breaking backward-request
   reversibility (genmlx-v740)."
  [results]
  (reduce (fn [cm [i r]]
            (let [d (:discard r)]
              (if (and d (not= d cm/EMPTY))
                (cm/set-choice cm [i] d)
                cm)))
          cm/EMPTY
          (map-indexed vector results)))

(defn- sum-field
  "Sum a field across results, starting from scalar 0.0."
  [results field-fn]
  (reduce (fn [acc r] (mx/add acc (field-fn r)))
          ZERO
          results))

(defn- values->choices
  "Convert compiled result {:values {addr->val}} to a ChoiceMap."
  [values]
  (cm/from-flat-map values))

(defn- without-component-idx
  "Drop the Mix combinator's :component-idx entry from a choicemap, leaving the
   inner component's choices. Non-Node inputs (e.g. EMPTY) pass through unchanged."
  [cm-choices]
  (if (instance? cm/Node cm-choices)
    (cm/->Node (dissoc (:m cm-choices) :component-idx))
    cm-choices))

(defn- batched-sub
  "Build [transition init-sub] for one batched sub-execution.
   Selects the generate transition (carrying weight) when constraints are
   present, else the simulate transition, and merges the parent param-store.
   zero is the [N]-shaped (or scalar) accumulator seed for score/weight."
  [state constraints key batch-size zero]
  (let [has-constraints? (and constraints (not= constraints cm/EMPTY))
        init-sub (if has-constraints?
                   {:choices cm/EMPTY :score zero :weight zero
                    :key key :constraints constraints
                    :batch-size batch-size :batched? true}
                   {:choices cm/EMPTY :score zero
                    :key key :batch-size batch-size :batched? true})
        init-sub (if-let [ps (:param-store state)]
                   (assoc init-sub :param-store ps)
                   init-sub)]
    [(if has-constraints? h/batched-generate-transition h/batched-simulate-transition)
     init-sub]))

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
    (tag-joint
      (with-meta
        (tr/make-trace {:gen-fn this :args args
                        :choices choices :retval retvals :score total-score})
        {::element-scores element-scores ::compiled-path true ::fused true}))))

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
                      ZERO results)
        element-scores (mapv :score results)]
    (tag-joint
      (with-meta
        (tr/make-trace {:gen-fn this :args args
                        :choices choices :retval retvals :score score})
        {::element-scores element-scores ::compiled-path true}))))

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
    (tag-from-traces
      (with-meta
        (tr/make-trace {:gen-fn this :args args
                        :choices choices :retval retvals :score score})
        {::element-scores element-scores})
      results)))

(defrecord MapCombinator [kernel]
  p/IGenerativeFunction
  (simulate [this args]
    (if-let [{:keys [fused-fn addr-order]} (cops/make-fused-map-simulate
                                            (:schema kernel) (:source kernel))]
      (map-simulate-fused this args kernel fused-fn addr-order)
      (if-let [csim (cops/get-compiled-simulate kernel)]
        (map-simulate-compiled this args kernel csim)
        (map-simulate-handler this args kernel))))

  p/IGenerate
  (generate [this args constraints]
    (if-let [cgen (cops/get-compiled-generate kernel)]
      ;; Compiled path — call compiled-generate directly per element
      (let [n (count (first args))
            init-key (rng/fresh-key)]
        (loop [i 0 key init-key
               choices cm/EMPTY score ZERO weight ZERO
               retvals [] element-scores []]
          (if (>= i n)
            {:trace (tag-joint
                      (with-meta
                        (tr/make-trace {:gen-fn this :args args
                                        :choices choices :retval retvals :score score})
                        {::element-scores element-scores ::compiled-path true}))
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
        {:trace (tag-from-traces
                  (with-meta
                    (tr/make-trace {:gen-fn this :args args
                                    :choices choices :retval retvals :score score})
                    {::element-scores element-scores})
                  (map :trace results))
         :weight weight})))

  p/IUpdate
  (update [this trace constraints]
    (let [trace (ensure-joint-self this :update trace)]
      (if-let [cupd (cops/get-compiled-update kernel)]
      ;; WP-8: compiled update path
      (let [old-choices (:choices trace)
            args (:args trace)
            n (count (first args))
            init-key (rng/fresh-key)]
        (loop [i 0 key init-key
               choices cm/EMPTY score ZERO discard cm/EMPTY
               retvals [] element-scores []]
          (if (>= i n)
            {:trace (tag-joint
                      (with-meta
                        (tr/make-trace {:gen-fn this :args args
                                        :choices choices :retval retvals :score score})
                        {::element-scores element-scores ::compiled-path true}))
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
                                              :retval nil :score (if old-element-scores (nth old-element-scores i) ZERO)})]
                              (p/update kernel old-trace (cm/get-submap constraints i))))
                          (range n))
            choices (assemble-choices results (comp :choices :trace))
            retvals (mapv (comp :retval :trace) results)
            score (sum-field results (comp :score :trace))
            ;; Thesis update weights are additive across elements, but each
            ;; element weight was computed against the constructed old score
            ;; (recorded element score, or 0 without metadata). Re-base
            ;; against the true total old score so both cases stay exact:
            ;; W = Σ w_i + Σ constructed_old_i - old_total.
            constructed-old (if old-element-scores
                              (reduce mx/add ZERO old-element-scores)
                              ZERO)
            weight (mx/subtract (mx/add (sum-field results :weight) constructed-old)
                                (:score trace))
            discard (assemble-indexed-discards results)
            element-scores (mapv (comp :score :trace) results)]
        {:trace (tag-from-traces
                  (with-meta
                    (tr/make-trace {:gen-fn this :args args
                                    :choices choices :retval retvals :score score})
                    {::element-scores element-scores})
                  (map :trace results))
         :weight weight :discard discard}))))

  p/IRegenerate
  ;; Per-element old scores come from ::element-scores metadata. When that
  ;; metadata is lost (splice-boundary trace reconstruction, serialize
  ;; round-trip), the loop runs against ZERO olds and the summed weight is
  ;; too high by exactly the old total — which the trace always records as
  ;; (:score trace). Correct at the end instead of silently returning wrong
  ;; weights (genmlx-v740). When metadata is present the path is unchanged.
  (regenerate [this trace selection]
    (let [trace (ensure-joint-self this :regenerate trace)]
     (if-let [cregen (cops/get-compiled-regenerate kernel)]
      ;; WP-9A: compiled regenerate path
      (let [old-choices (:choices trace)
            args (:args trace)
            n (count (first args))
            old-element-scores (::element-scores (meta trace))
            init-key (rng/fresh-key)]
        (loop [i 0 key init-key
               choices cm/EMPTY score ZERO weight ZERO
               retvals [] element-scores []]
          (if (>= i n)
            {:trace (tag-joint
                      (with-meta
                        (tr/make-trace {:gen-fn this :args args
                                        :choices choices :retval retvals :score score})
                        {::element-scores element-scores ::compiled-path true}))
             :weight (if old-element-scores
                       weight
                       (mx/subtract weight (:score trace)))}
            (let [[k1 k2] (rng/split key)
                  elem-args (mapv #(nth % i) args)
                  old-sub-choices (cm/get-submap old-choices i)
                  result (cregen k1 (vec elem-args) old-sub-choices
                                 (sel/get-subselection selection i))
                  elem-choices (values->choices (:values result))
                  old-elem-score (or (some-> old-element-scores (nth i nil))
                                     ZERO)
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
                                              :retval nil :score (if old-element-scores (nth old-element-scores i) ZERO)})]
                              (p/regenerate kernel old-trace
                                            (sel/get-subselection selection i))))
                          (range n))
            choices (assemble-choices results (comp :choices :trace))
            retvals (mapv (comp :retval :trace) results)
            score (sum-field results (comp :score :trace))
            weight (sum-field results :weight)
            element-scores (mapv (comp :score :trace) results)]
        {:trace (tag-from-traces
                  (with-meta
                    (tr/make-trace {:gen-fn this :args args
                                    :choices choices :retval retvals :score score})
                    {::element-scores element-scores})
                  (map :trace results))
         :weight (if old-element-scores
                   weight
                   (mx/subtract weight (:score trace)))})))))

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
  "Unpack fused output tensor [T, K+N] into per-step ChoiceMaps and retvals.
   addr-order: [keyword...] mapping column index 0..K-1 to addresses.
   state-keys: nil for scalar state (column K = retval),
               or [keyword...] for map state (columns K..K+N-1 = state values).
   Returns {:choices cm :states [retval...] :step-scores [score...]}."
  [outputs-tensor scores-tensor T addr-order state-keys]
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
              retval (if state-keys
                       (into {} (map-indexed
                                 (fn [i k] [k (mx/index row (+ n-sites i))])
                                 state-keys))
                       (mx/index row n-sites))
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
           (every? (fn [[a b]]
                     (== (mx/item (mx/ensure-array a))
                         (mx/item (mx/ensure-array b))))
                   (map vector cached-extras current-extras)))))

(defn- get-or-build-fused-unfold
  "Get cached or build new fused unfold simulate.
   Cache is stored as metadata on the combinator.
   Returns {:compiled-fn :noise-dim :addr-order :noise-site-types :extra-args} or nil."
  [cache kernel T extra]
  (when (and (pos? T) (cops/fusable-kernel? kernel))
    (let [cached (get @cache T)]
      (if (and cached (extras-match? (:extra-args cached) extra))
        cached
        ;; Build new fused function
        (when-let [fused (cops/make-fused-unfold-simulate
                          (:schema kernel) (:source kernel) T extra)]
          (swap! cache assoc T fused)
          fused)))))

(defn- unfold-simulate-fused
  "Fused Unfold simulate: 2 Metal dispatches for T steps."
  [this args kernel fused-cache n init-state extra]
  (let [{:keys [compiled-fn addr-order noise-site-types state-keys]}  ; noise-dim unused (genmlx-21kt)
        (get-or-build-fused-unfold fused-cache kernel n extra)
        key (rng/fresh-key)
        noise (cops/generate-noise-matrix key n noise-site-types)
        ;; Pack map init-state to flat [N] array for compiled fn
        init-flat (if state-keys
                    (mx/stack (mapv #(mx/ensure-array (get init-state %)) state-keys))
                    (mx/ensure-array init-state))
        [outputs-tensor scores-tensor total-score]
        (compiled-fn init-flat noise)
        _ (mx/materialize! outputs-tensor scores-tensor total-score)
        {:keys [choices states step-scores]}
        (unpack-fused-outputs outputs-tensor scores-tensor n addr-order state-keys)]
    (tag-joint
      (with-meta
        (tr/make-trace {:gen-fn this :args args
                        :choices choices :retval states :score total-score})
        {::step-scores step-scores ::compiled-path true ::fused true}))))

(defn- unfold-simulate-compiled
  "Compiled Unfold simulate: call compiled-simulate per step."
  [this args kernel n init-state extra csim]
  (let [init-key (rng/fresh-key)]
    (loop [t 0 state init-state key init-key
           choices cm/EMPTY score ZERO
           states [] step-scores []]
      (if (>= t n)
        (tag-joint
          (with-meta
            (tr/make-trace {:gen-fn this :args args
                            :choices choices :retval states :score score})
            {::step-scores step-scores ::compiled-path true}))
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
         choices cm/EMPTY score ZERO
         states [] step-scores [] st :joint]
    (if (>= t n)
      (tr/with-score-type
        (with-meta
          (tr/make-trace {:gen-fn this :args args
                          :choices choices :retval states :score score})
          {::step-scores step-scores})
        st)
      (let [trace (p/simulate kernel (into [t state] extra))
            new-state (:retval trace)]
        (recur (inc t)
               new-state
               (cm/set-choice choices [t] (:choices trace))
               (mx/add score (:score trace))
               (conj states new-state)
               (conj step-scores (:score trace))
               (tr/combine-score-types st (tr/score-type trace)))))))

(defrecord UnfoldCombinator [kernel fused-cache]
  p/IGenerativeFunction
  (simulate [this args]
    (let [[n init-state & extra] args]
      (if-let [_fused (get-or-build-fused-unfold fused-cache kernel n extra)]
        (unfold-simulate-fused this args kernel fused-cache n init-state extra)
        (if-let [csim (cops/get-compiled-simulate kernel)]
          (unfold-simulate-compiled this args kernel n init-state extra csim)
          (unfold-simulate-handler this args kernel n init-state extra)))))

  p/IGenerate
  (generate [this args constraints]
    (let [[n init-state & extra] args]
      (if-let [cgen (cops/get-compiled-generate kernel)]
        ;; Compiled path
        (let [init-key (rng/fresh-key)]
          (loop [t 0 state init-state key init-key
                 choices cm/EMPTY score ZERO weight ZERO
                 states [] step-scores []]
            (if (>= t n)
              {:trace (tag-joint
                        (with-meta
                          (tr/make-trace {:gen-fn this :args args
                                          :choices choices :retval states :score score})
                          {::step-scores step-scores ::compiled-path true}))
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
               choices cm/EMPTY score ZERO weight ZERO
               states [] step-scores [] st :joint]
          (if (>= t n)
            {:trace (tr/with-score-type
                      (with-meta
                        (tr/make-trace {:gen-fn this :args args
                                        :choices choices :retval states :score score})
                        {::step-scores step-scores})
                      st)
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
                     (conj step-scores (:score (:trace result)))
                     (tr/combine-score-types st (tr/score-type trace))))))))))

(extend-type UnfoldCombinator
  p/IUpdate
  (update [this trace constraints]
    (let [trace (ensure-joint-self this :update trace)
          kern (:kernel this)
          cupd (cops/get-compiled-update kern)
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
        {:trace trace :weight ZERO :discard cm/EMPTY}
        ;; Build prefix from old trace (steps 0..first-changed-1)
        (let [prefix-choices (if (pos? first-changed)
                               (reduce (fn [cm t]
                                         (cm/set-choice cm [t] (cm/get-submap choices t)))
                                       cm/EMPTY (range first-changed))
                               cm/EMPTY)
              prefix-score (if (pos? first-changed)
                             (reduce (fn [acc t] (mx/add acc (nth old-step-scores t)))
                                     ZERO (range first-changed))
                             ZERO)
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
          ;; Execute steps first-changed..n-1. `nf` accumulates the non-fresh
          ;; score: prefix steps are retained verbatim (their non-fresh score
          ;; is the recorded prefix score); compiled steps never sample fresh
          ;; (full new score counts); fallback steps recover it from the
          ;; child's thesis weight, nonfresh_t = w_t + constructed_old_t.
          ;; Final thesis weight = nf - old_total, exact with or without
          ;; step-score metadata (the constructed old scores cancel).
          (loop [t first-changed state start-state key init-key
                 new-choices prefix-choices score prefix-score
                 nf prefix-score
                 discard cm/EMPTY
                 states prefix-states step-scores prefix-step-scores
                 st :joint]
            (if (>= t n)
              {:trace (tr/with-score-type
                        (with-meta
                          (tr/make-trace {:gen-fn this :args args
                                          :choices new-choices :retval states :score score})
                          (cond-> {::step-scores step-scores}
                            cupd (assoc ::compiled-path true)))
                        st)
               :weight (mx/subtract nf (:score trace)) :discard discard}
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
                         (mx/add nf (:score result))
                         (if (seq (:discard result))
                           (cm/set-choice discard [t] step-discard)
                           discard)
                         (conj states new-state)
                         (conj step-scores (:score result))
                         st))
                ;; Fallback: handler path
                (let [old-sub-choices (cm/get-submap choices t)
                      kernel-args (into [t state] extra)
                      constructed-old (if old-step-scores (nth old-step-scores t) ZERO)
                      old-trace (tr/make-trace
                                 {:gen-fn kern :args kernel-args
                                  :choices old-sub-choices
                                  :retval nil :score constructed-old})
                      result (p/update kern old-trace (cm/get-submap constraints t))
                      new-trace (:trace result)
                      new-state (:retval new-trace)]
                  (recur (inc t)
                         new-state nil
                         (cm/set-choice new-choices [t] (:choices new-trace))
                         (mx/add score (:score new-trace))
                         (mx/add nf (mx/add (:weight result) constructed-old))
                         (if (:discard result)
                           (cm/set-choice discard [t] (:discard result))
                           discard)
                         (conj states new-state)
                         (conj step-scores (:score new-trace))
                         (tr/combine-score-types st (tr/score-type new-trace)))))))))))

  p/IRegenerate
  ;; Per-step old scores come from ::step-scores metadata; when it is lost
  ;; (splice-boundary reconstruction, serialize round-trip) the loop runs
  ;; against ZERO olds and the summed weight is too high by the old total —
  ;; recorded on (:score trace). Correct at the end (genmlx-v740).
  (regenerate [this trace selection]
    (let [trace (ensure-joint-self this :regenerate trace)
          kern (:kernel this)
          cregen (cops/get-compiled-regenerate kern)
          {:keys [args choices]} trace
          [n init-state & extra] args
          old-step-scores (::step-scores (meta trace))
          init-key (when cregen (rng/fresh-key))]
      (loop [t 0 state init-state key init-key
             new-choices cm/EMPTY score ZERO weight ZERO
             states [] step-scores [] st :joint]
        (if (>= t n)
          {:trace (tr/with-score-type
                    (with-meta
                      (tr/make-trace {:gen-fn this :args args
                                      :choices new-choices :retval states :score score})
                      (cond-> {::step-scores step-scores}
                        cregen (assoc ::compiled-path true)))
                    st)
           :weight (if old-step-scores
                     weight
                     (mx/subtract weight (:score trace)))}
          (let [old-sub-choices (cm/get-submap choices t)
                kernel-args (into [t state] extra)
                old-score (if old-step-scores (nth old-step-scores t) ZERO)]
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
                       (conj step-scores (:score result))
                       st))
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
                       (conj step-scores (:score new-trace))
                       (tr/combine-score-types st (tr/score-type new-trace)))))))))))

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

(extend-type UnfoldCombinator
  p/IBatchedSplice
  (batched-splice [this state addr args]
    (let [kern (:kernel this)]
      ;; Fall back to the per-particle slow path when the kernel is not a
      ;; DynamicGF, OR in update/regenerate mode (:old-choices present): the fast
      ;; path below only runs batched simulate/generate and would RESAMPLE every
      ;; step instead of replaying/regenerating the retained kernel sites
      ;; (genmlx-llpt). combinator-batched-fallback consults old-choices/selection.
      (if (or (not (:body-fn kern)) (contains? state :old-choices))
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
                    [transition init-sub]
                    (batched-sub state step-constraints sk batch-size ZERO)
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
    (tag-joint
      (with-meta
        (tr/make-trace {:gen-fn unfold-gf :args args
                        :choices cm/EMPTY :retval [] :score ZERO})
        {::step-scores []}))))

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
    {:trace (tag-from-traces
              (with-meta
                (tr/make-trace {:gen-fn unfold-gf :args new-args
                                :choices new-choices :retval new-retval :score new-score})
                {::step-scores new-step-scores})
              [trace step-trace])
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
      (if-let [csim (cops/get-compiled-simulate branch)]
        ;; L1-M5: compiled path
        (let [key (rng/fresh-key)
              result (csim key (vec branch-args))
              choices (cm/from-flat-map (:values result))]
          (tag-joint
            (with-meta
              (tr/make-trace {:gen-fn this :args args
                              :choices choices
                              :retval (:retval result)
                              :score (:score result)})
              {::switch-idx idx ::compiled-path true})))
        ;; L0: handler path
        (let [trace (p/simulate branch (vec branch-args))]
          (tag-from-traces
            (with-meta
              (tr/make-trace {:gen-fn this :args args
                              :choices (:choices trace)
                              :retval (:retval trace)
                              :score (:score trace)})
              {::switch-idx idx})
            [trace])))))

  p/IGenerate
  (generate [this args constraints]
    (let [[idx & branch-args] args
          branch (nth branches idx)]
      (if-let [cgen (cops/get-compiled-generate branch)]
        ;; Compiled path
        (let [key (rng/fresh-key)
              result (cgen key (vec branch-args) constraints)
              choices (values->choices (:values result))]
          {:trace (tag-joint
                    (with-meta
                      (tr/make-trace {:gen-fn this :args args
                                      :choices choices
                                      :retval (:retval result)
                                      :score (:score result)})
                      {::switch-idx idx ::compiled-path true}))
           :weight (:weight result)})
        ;; Fallback: handler path
        (let [{:keys [trace weight]} (p/generate branch (vec branch-args) constraints)]
          {:trace (tag-from-traces
                    (with-meta
                      (tr/make-trace {:gen-fn this :args args
                                      :choices (:choices trace)
                                      :retval (:retval trace)
                                      :score (:score trace)})
                      {::switch-idx idx})
                    [trace])
           :weight weight})))))

(extend-type SwitchCombinator
  p/IUpdate
  (update [this trace constraints]
    (let [trace (ensure-joint-self this :update trace)
          orig-args (:args trace)
          [new-idx & branch-args] orig-args
          old-idx (or (::switch-idx (meta trace)) new-idx)]
      (if (= old-idx new-idx)
        ;; Same branch: update in place
        (let [branch (nth (:branches this) new-idx)
              cupd (cops/get-compiled-update branch)]
          (if cupd
            ;; WP-8: compiled update path
            (let [key (rng/fresh-key)
                  result (cupd key (vec branch-args) constraints (:choices trace))
                  new-choices (values->choices (:values result))
                  new-discard (values->choices (:discard result))]
              {:trace (tag-joint
                        (with-meta
                          (tr/make-trace {:gen-fn this :args orig-args
                                          :choices new-choices
                                          :retval (:retval result)
                                          :score (:score result)})
                          {::switch-idx new-idx ::compiled-path true}))
               :weight (mx/subtract (:score result) (:score trace))
               :discard new-discard})
            ;; Fallback: handler path
            (let [old-branch-trace (tr/make-trace
                                    {:gen-fn branch :args (vec branch-args)
                                     :choices (:choices trace)
                                     :retval (:retval trace) :score (:score trace)})
                  result (p/update branch old-branch-trace constraints)
                  new-branch-trace (:trace result)]
              {:trace (tag-from-traces
                        (with-meta
                          (tr/make-trace {:gen-fn this :args orig-args
                                          :choices (:choices new-branch-trace)
                                          :retval (:retval new-branch-trace)
                                          :score (:score new-branch-trace)})
                          {::switch-idx new-idx})
                        [new-branch-trace])
               :weight (:weight result) :discard (:discard result)})))
        ;; Different branch: generate new branch from scratch.
        ;; Thesis weight: the generate weight counts only constrained sites
        ;; (fresh ones cancel against the internal proposal); the removed old
        ;; branch is charged via its recorded score.
        (let [new-branch (nth (:branches this) new-idx)
              gen-result (p/generate new-branch (vec branch-args) constraints)
              new-branch-trace (:trace gen-result)
              new-score (:score new-branch-trace)]
          {:trace (tag-from-traces
                    (with-meta
                      (tr/make-trace {:gen-fn this :args orig-args
                                      :choices (:choices new-branch-trace)
                                      :retval (:retval new-branch-trace)
                                      :score new-score})
                      {::switch-idx new-idx})
                    [new-branch-trace])
           :weight (mx/subtract (:weight gen-result) (:score trace))
           :discard (:choices trace)}))))

  p/IRegenerate
  (regenerate [this trace selection]
    (let [trace (ensure-joint-self this :regenerate trace)
          orig-args (:args trace)
          [idx & branch-args] orig-args
          branch (nth (:branches this) idx)]
      (if-let [cregen (cops/get-compiled-regenerate branch)]
        ;; WP-9A: compiled regenerate path
        (let [key (rng/fresh-key)
              result (cregen key (vec branch-args) (:choices trace) selection)
              new-choices (values->choices (:values result))
              ;; compiled-regenerate returns proposal_ratio in :weight
              ;; weight = new_score - old_score - proposal_ratio
              weight (mx/subtract (mx/subtract (:score result) (:score trace))
                                  (:weight result))]
          {:trace (tag-joint
                    (with-meta
                      (tr/make-trace {:gen-fn this :args orig-args
                                      :choices new-choices
                                      :retval (:retval result)
                                      :score (:score result)})
                      {::switch-idx idx ::compiled-path true}))
           :weight weight})
        ;; Fallback: handler path
        (let [old-branch-trace (tr/make-trace
                                {:gen-fn branch :args (vec branch-args)
                                 :choices (:choices trace)
                                 :retval (:retval trace) :score (:score trace)})
              result (p/regenerate branch old-branch-trace selection)
              new-branch-trace (:trace result)]
          {:trace (tag-from-traces
                    (with-meta
                      (tr/make-trace {:gen-fn this :args orig-args
                                      :choices (:choices new-branch-trace)
                                      :retval (:retval new-branch-trace)
                                      :score (:score new-branch-trace)})
                      {::switch-idx idx})
                    [new-branch-trace])
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
   Where mask is true, use cm-true's value; otherwise cm-false's.
   Operates on the UNION of leaf addresses: an address present on only
   one side keeps that side's value for all particles — every branch
   runs once in batched mode, so a branch-only address holds that
   branch's sampled values, and score masking already excludes the
   particles that selected another branch (genmlx-v740)."
  [mask cm-true cm-false]
  (let [addrs-t (cm/addresses cm-true)
        union (into addrs-t (remove (set addrs-t)) (cm/addresses cm-false))]
    (reduce
     (fn [acc addr-path]
       (let [node-t (reduce cm/get-submap cm-true addr-path)
             node-f (reduce cm/get-submap cm-false addr-path)]
         (cm/set-choice acc addr-path
                        (cond
                          (not (cm/has-value? node-f)) (cm/get-value node-t)
                          (not (cm/has-value? node-t)) (cm/get-value node-f)
                          :else (mx/where mask (cm/get-value node-t)
                                          (cm/get-value node-f))))))
     cm/EMPTY union)))

(defn- idx-mask
  "Boolean mask selecting particles whose [N]-shaped index equals branch i."
  [index i]
  (mx/equal index (mx/scalar i mx/int32)))

(defn- where-combine
  "Mask-select per-branch values into one [N]-shaped result.
   value-fn extracts the value contributed by branch i's result; where the
   [N]-shaped index equals i, that value wins over the running accumulator."
  [index init results value-fn]
  (reduce-kv
   (fn [acc i r]
     (mx/where (idx-mask index i) (value-fn r) acc))
   init results))

(defn- mask-combinable?
  "Can the per-branch values vs be mask-selected into one per-particle
   value? True for: all nil, all MLX arrays/numbers, maps with identical
   key sets whose values are combinable, same-length vectors combinable
   element-wise, or identical values across branches."
  [vs]
  (cond
    (every? nil? vs) true
    (every? #(or (mx/array? %) (number? %)) vs) true
    (and (every? map? vs) (apply = (map (comp set keys) vs)))
    (every? (fn [k] (mask-combinable? (mapv #(get % k) vs)))
            (keys (first vs)))
    (and (every? vector? vs) (apply = (map count vs)))
    (every? mask-combinable? (apply mapv vector vs))
    :else (apply = vs)))

(defn- mask-combine-vals
  "Mask-select per-branch return values into one per-particle value.
   Arrays and numbers combine via mx/where on the [N]-shaped index; maps
   and vectors combine recursively (mirrors vectorized merge-state-by-mask);
   identical non-numeric values pass through. Call mask-combinable? first —
   on uncombinable input the equal-values fallthrough is meaningless."
  [index vs]
  (cond
    (every? nil? vs) nil
    (every? #(or (mx/array? %) (number? %)) vs)
    (where-combine index (first vs) vs identity)
    (map? (first vs))
    (into {} (map (fn [k] [k (mask-combine-vals index (mapv #(get % k) vs))]))
          (keys (first vs)))
    (vector? (first vs))
    (mapv #(mask-combine-vals index %) (apply mapv vector vs))
    :else (first vs)))

(defn- combine-retvals
  "Combine per-branch return values per-particle, or throw an honest
   error when they cannot be represented in shape-batched mode. The old
   behavior returned nil for any non-array retval — silently wrong for
   every model that uses the splice's return value (genmlx-v740)."
  [combinator addr index rvs]
  (when-not (mask-combinable? rvs)
    (throw (ex-info (str combinator " batched-splice at " addr ": branch return "
                         "values cannot be combined per-particle in shape-batched "
                         "mode. Branches must return MLX arrays, numbers, maps/"
                         "vectors of those with matching structure, or identical "
                         "values. Use scalar-mode inference for this model.")
                    {:addr addr
                     :retval-types (mapv #(cond (nil? %) :nil
                                                (mx/array? %) :array
                                                (number? %) :number
                                                (map? %) :map
                                                (vector? %) :vector
                                                :else (type %))
                                         rvs)})))
  (mask-combine-vals index rvs))

(extend-type SwitchCombinator
  p/IBatchedSplice
  (batched-splice [this state addr args]
    (let [brs (:branches this)
          all-dynamic? (every? :body-fn brs)]
      ;; genmlx-llpt: fall back when not all branches are DynamicGF, OR in
      ;; update/regenerate mode (:old-choices present) — the fast path would
      ;; resample the chosen branch instead of replaying/regenerating it.
      (if (or (not all-dynamic?) (contains? state :old-choices))
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
                        [transition init-sub]
                        (batched-sub state sub-constraints bk batch-size batch-zero)
                        result (rt/run-handler transition init-sub
                                               (fn [rt] (apply (:body-fn (nth brs i)) rt
                                                               (vec branch-args))))]
                    (recur (inc i) (conj results result) nk))))
              ;; Combine results using mx/where based on [N]-shaped index
              combined-score
              (where-combine index batch-zero branch-results :score)
              combined-weight
              (when (contains? state :weight)
                (where-combine index batch-zero branch-results
                               #(or (:weight %) batch-zero)))
              combined-choices
              (reduce-kv
               (fn [acc i br]
                 (if (zero? i)
                   (:choices br)
                   (where-select-choicemap (idx-mask index i) (:choices br) acc)))
               cm/EMPTY branch-results)
              combined-retval
              (combine-retvals "Switch" addr index (mapv :retval branch-results))
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
          (tag-from-traces
            (tr/make-trace {:gen-fn this :args args
                            :choices (:choices trace)
                            :retval (:retval trace)
                            :score (:score trace)})
            [trace]))
        (tag-joint
          (tr/make-trace {:gen-fn this :args args
                          :choices cm/EMPTY
                          :retval nil
                          :score ZERO})))))

  p/IGenerate
  (generate [this args constraints]
    (let [[active? & inner-args] args]
      (if active?
        (let [{:keys [trace weight]} (p/generate inner (vec inner-args) constraints)]
          {:trace (tag-from-traces
                    (tr/make-trace {:gen-fn this :args args
                                    :choices (:choices trace)
                                    :retval (:retval trace)
                                    :score (:score trace)})
                    [trace])
           :weight weight})
        {:trace (tag-joint
                  (tr/make-trace {:gen-fn this :args args
                                  :choices cm/EMPTY
                                  :retval nil
                                  :score ZERO}))
         :weight ZERO}))))

(defn mask-combinator
  "Create a Mask combinator that gates execution of an inner GF.
   First argument to the masked GF is a boolean active? flag."
  [inner]
  (->MaskCombinator inner))

(extend-type MaskCombinator
  p/IUpdate
  (update [this trace constraints]
    (let [trace (ensure-joint-self this :update trace)
          [active? & inner-args] (:args trace)]
      (if active?
        (let [inner (:inner this)
              old-inner-trace (tr/make-trace
                               {:gen-fn inner :args (vec inner-args)
                                :choices (:choices trace)
                                :retval (:retval trace) :score (:score trace)})
              result (p/update inner old-inner-trace constraints)
              new-trace (:trace result)]
          {:trace (tag-from-traces
                    (tr/make-trace {:gen-fn this :args (:args trace)
                                    :choices (:choices new-trace)
                                    :retval (:retval new-trace)
                                    :score (:score new-trace)})
                    [new-trace])
           :weight (:weight result) :discard (:discard result)})
        {:trace trace :weight ZERO :discard cm/EMPTY})))

  p/IRegenerate
  (regenerate [this trace selection]
    (let [trace (ensure-joint-self this :regenerate trace)
          [active? & inner-args] (:args trace)]
      (if active?
        (let [inner (:inner this)
              old-inner-trace (tr/make-trace
                               {:gen-fn inner :args (vec inner-args)
                                :choices (:choices trace)
                                :retval (:retval trace) :score (:score trace)})
              result (p/regenerate inner old-inner-trace selection)
              new-trace (:trace result)]
          {:trace (tag-from-traces
                    (tr/make-trace {:gen-fn this :args (:args trace)
                                    :choices (:choices new-trace)
                                    :retval (:retval new-trace)
                                    :score (:score new-trace)})
                    [new-trace])
           :weight (:weight result)})
        {:trace trace :weight ZERO}))))

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
      (tag-from-traces
        (tr/make-trace {:gen-fn this :args args
                        :choices (:choices trace)
                        :retval (:retval trace)
                        :score (:score trace)})
        [trace])))

  p/IGenerate
  (generate [this args constraints]
    (let [gf (maker this)
          {:keys [trace weight]} (p/generate gf args constraints)]
      {:trace (tag-from-traces
                (tr/make-trace {:gen-fn this :args args
                                :choices (:choices trace)
                                :retval (:retval trace)
                                :score (:score trace)})
                [trace])
       :weight weight})))

(extend-type RecurseCombinator
  p/IUpdate
  (update [this trace constraints]
    (let [trace (ensure-joint-self this :update trace)
          gf ((:maker this) this)
          old-inner-trace (tr/make-trace
                           {:gen-fn gf :args (:args trace)
                            :choices (:choices trace)
                            :retval (:retval trace) :score (:score trace)})
          result (p/update gf old-inner-trace constraints)
          new-trace (:trace result)]
      {:trace (tag-from-traces
                (tr/make-trace {:gen-fn this :args (:args trace)
                                :choices (:choices new-trace)
                                :retval (:retval new-trace)
                                :score (:score new-trace)})
                [new-trace])
       :weight (:weight result) :discard (:discard result)}))

  p/IRegenerate
  (regenerate [this trace selection]
    (let [trace (ensure-joint-self this :regenerate trace)
          gf ((:maker this) this)
          old-inner-trace (tr/make-trace
                           {:gen-fn gf :args (:args trace)
                            :choices (:choices trace)
                            :retval (:retval trace) :score (:score trace)})
          result (p/regenerate gf old-inner-trace selection)
          new-trace (:trace result)]
      {:trace (tag-from-traces
                (tr/make-trace {:gen-fn this :args (:args trace)
                                :choices (:choices new-trace)
                                :retval (:retval new-trace)
                                :score (:score new-trace)})
                [new-trace])
       :weight (:weight result)}))

  p/IProject
  (project [this trace selection]
    (let [trace (ensure-joint-self this :project trace)
          gf ((:maker this) this)
          inner-trace (tr/make-trace
                       {:gen-fn gf :args (:args trace)
                        :choices (:choices trace)
                        :retval (:retval trace) :score (:score trace)})]
      (p/project gf inner-trace selection)))

  p/IUpdateWithDiffs
  (update-with-diffs [this trace constraints argdiffs]
    (if (and (diff/no-change? argdiffs) (= constraints cm/EMPTY))
      {:trace trace :weight ZERO :discard cm/EMPTY}
      (p/update this trace constraints)))

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
        leaf? (cm/has-value? first-choices)]
    {:choices (if leaf?
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
        leaf? (cm/has-value? first-choices)
        ;; Build combined choices
        ;; Note: reduce-kv over full vector (not rest) so indices match branch indices
        combined-choices
        (if leaf?
          ;; Distribution branches: combine leaf values
          (let [vals (mapv #(cm/get-value (:choices %)) branch-data)
                combined (reduce-kv
                          (fn [acc i v]
                            (if (zero? i) acc
                                (mx/where (idx-mask index i) v acc)))
                          (first vals) vals)]
            (cm/->Value combined))
          ;; GF branches: combine per-address
          (let [all-addrs (into #{} (mapcat #(cm/addresses (:choices %)) branch-data))]
            (reduce
             (fn [cm addr-path]
               (let [vals (mapv #(cm/get-choice (:choices %) addr-path)
                                branch-data)
                     combined (reduce-kv
                               (fn [acc i v]
                                 (if (or (zero? i) (nil? v)) acc
                                     (mx/where (idx-mask index i) v acc)))
                               (or (first vals) (mx/zeros [n-val]))
                               vals)]
                 (cm/set-choice cm addr-path combined)))
             cm/EMPTY all-addrs)))
        ;; Combine scores using where
        combined-score (reduce-kv
                        (fn [acc i bd]
                          (if (zero? i)
                            (:score bd)
                            (mx/where (idx-mask index i) (:score bd) acc)))
                        ZERO
                        (vec branch-data))
        ;; Combine retvals
        combined-retval (let [rvs (mapv :retval branch-data)]
                          (if (and (mx/array? (first rvs)) (> n-branches 1))
                            (reduce-kv
                             (fn [acc i rv]
                               (if (or (zero? i) (nil? rv)) acc
                                   (mx/where (idx-mask index i) rv acc)))
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
  (when (and (pos? T) (cops/fusable-kernel? kernel))
    (let [cached (get @cache T)]
      (if cached
        cached
        (when-let [fused (cops/make-fused-scan-simulate
                          (:schema kernel) (:source kernel) T)]
          (swap! cache assoc T fused)
          fused)))))

(defn- scan-simulate-fused
  "Fused Scan simulate: compiled fn over all T steps."
  [this args kernel fused-cache init-carry inputs n]
  (let [{:keys [compiled-fn addr-order noise-site-types]}  ; noise-dim unused (genmlx-21kt)
        (get-or-build-fused-scan fused-cache kernel n)
        key (rng/fresh-key)
        noise (cops/generate-noise-matrix key n noise-site-types)
        [outputs-tensor scores-tensor total-score]
        (compiled-fn (mx/ensure-array init-carry)
                     (mx/stack (mapv mx/ensure-array inputs))
                     noise)
        _ (mx/materialize! outputs-tensor scores-tensor total-score)
        {:keys [choices carries outputs step-scores]}
        (unpack-fused-scan-outputs outputs-tensor scores-tensor n addr-order)]
    (tag-joint
      (with-meta
        (tr/make-trace {:gen-fn this :args args
                        :choices choices
                        :retval {:carry (last carries) :outputs outputs}
                        :score total-score})
        {::step-scores step-scores ::step-carries carries
         ::compiled-path true ::fused true}))))

(defn- scan-simulate-compiled
  "Compiled Scan simulate: call compiled-simulate per step."
  [this args kernel init-carry inputs n csim]
  (let [init-key (rng/fresh-key)]
    (loop [t 0 carry init-carry key init-key
           choices cm/EMPTY score ZERO
           outputs [] step-scores [] step-carries []]
      (if (>= t n)
        (tag-joint
          (with-meta
            (tr/make-trace {:gen-fn this :args args
                            :choices choices
                            :retval {:carry carry :outputs outputs}
                            :score score})
            {::step-scores step-scores ::step-carries step-carries
             ::compiled-path true}))
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
         choices cm/EMPTY score ZERO
         outputs [] step-scores [] step-carries [] st :joint]
    (if (>= t n)
      (tr/with-score-type
        (with-meta
          (tr/make-trace {:gen-fn this :args args
                          :choices choices
                          :retval {:carry carry :outputs outputs}
                          :score score})
          {::step-scores step-scores ::step-carries step-carries})
        st)
      (let [trace (p/simulate kernel [carry (nth inputs t)])
            [new-carry output] (:retval trace)]
        (recur (inc t)
               new-carry
               (cm/set-choice choices [t] (:choices trace))
               (mx/add score (:score trace))
               (conj outputs output)
               (conj step-scores (:score trace))
               (conj step-carries new-carry)
               (tr/combine-score-types st (tr/score-type trace)))))))

(defrecord ScanCombinator [kernel fused-cache]
  p/IGenerativeFunction
  (simulate [this args]
    (let [[init-carry inputs] args
          n (count inputs)]
      (if-let [_fused (get-or-build-fused-scan fused-cache kernel n)]
        (scan-simulate-fused this args kernel fused-cache init-carry inputs n)
        (if-let [csim (cops/get-compiled-simulate kernel)]
          (scan-simulate-compiled this args kernel init-carry inputs n csim)
          (scan-simulate-handler this args kernel init-carry inputs n)))))

  p/IGenerate
  (generate [this args constraints]
    (let [[init-carry inputs] args
          n (count inputs)]
      (if-let [cgen (cops/get-compiled-generate kernel)]
        ;; Compiled path
        (let [init-key (rng/fresh-key)]
          (loop [t 0 carry init-carry key init-key
                 choices cm/EMPTY score ZERO weight ZERO
                 outputs [] step-scores [] step-carries []]
            (if (>= t n)
              {:trace (tag-joint
                        (with-meta
                          (tr/make-trace {:gen-fn this :args args
                                          :choices choices
                                          :retval {:carry carry :outputs outputs}
                                          :score score})
                          {::step-scores step-scores ::step-carries step-carries
                           ::compiled-path true}))
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
               choices cm/EMPTY score ZERO weight ZERO
               outputs [] step-scores [] step-carries [] st :joint]
          (if (>= t n)
            {:trace (tr/with-score-type
                      (with-meta
                        (tr/make-trace {:gen-fn this :args args
                                        :choices choices
                                        :retval {:carry carry :outputs outputs}
                                        :score score})
                        {::step-scores step-scores ::step-carries step-carries})
                      st)
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
                     (conj step-carries new-carry)
                     (tr/combine-score-types st (tr/score-type trace))))))))))

(extend-type ScanCombinator
  p/IUpdate
  (update [this trace constraints]
    (let [trace (ensure-joint-self this :update trace)
          kern (:kernel this)
          cupd (cops/get-compiled-update kern)
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
        {:trace trace :weight ZERO :discard cm/EMPTY}
        ;; Build prefix from old trace (steps 0..first-changed-1)
        (let [prefix-choices (if (pos? first-changed)
                               (reduce (fn [cm t]
                                         (cm/set-choice cm [t] (cm/get-submap choices t)))
                                       cm/EMPTY (range first-changed))
                               cm/EMPTY)
              prefix-score (if (pos? first-changed)
                             (reduce (fn [acc t] (mx/add acc (nth old-step-scores t)))
                                     ZERO (range first-changed))
                             ZERO)
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
          ;; Execute steps first-changed..n-1. `nf` accumulates the non-fresh
          ;; score (see Unfold update for the convention); final thesis
          ;; weight = nf - old_total.
          (loop [t first-changed carry start-carry key init-key
                 new-choices prefix-choices score prefix-score
                 nf prefix-score
                 discard cm/EMPTY
                 outputs prefix-outputs
                 step-scores prefix-step-scores step-carries prefix-step-carries
                 st :joint]
            (if (>= t n)
              {:trace (tr/with-score-type
                        (with-meta
                          (tr/make-trace {:gen-fn this :args args
                                          :choices new-choices
                                          :retval {:carry carry :outputs outputs}
                                          :score score})
                          (cond-> {::step-scores step-scores ::step-carries step-carries}
                            cupd (assoc ::compiled-path true)))
                        st)
               :weight (mx/subtract nf (:score trace)) :discard discard}
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
                         (mx/add nf (:score result))
                         (if (seq (:discard result))
                           (cm/set-choice discard [t] step-discard)
                           discard)
                         (conj outputs output)
                         (conj step-scores (:score result))
                         (conj step-carries new-carry)
                         st))
                ;; Fallback: handler path
                (let [old-sub-choices (cm/get-submap choices t)
                      constructed-old (if old-step-scores (nth old-step-scores t) ZERO)
                      old-trace (tr/make-trace
                                 {:gen-fn kern :args [carry (nth inputs t)]
                                  :choices old-sub-choices
                                  :retval nil :score constructed-old})
                      result (p/update kern old-trace (cm/get-submap constraints t))
                      new-trace (:trace result)
                      [new-carry output] (:retval new-trace)]
                  (recur (inc t)
                         new-carry nil
                         (cm/set-choice new-choices [t] (:choices new-trace))
                         (mx/add score (:score new-trace))
                         (mx/add nf (mx/add (:weight result) constructed-old))
                         (if (:discard result)
                           (cm/set-choice discard [t] (:discard result))
                           discard)
                         (conj outputs output)
                         (conj step-scores (:score new-trace))
                         (conj step-carries new-carry)
                         (tr/combine-score-types st (tr/score-type new-trace)))))))))))

  p/IRegenerate
  ;; Same ::step-scores-loss correction as Unfold regenerate (genmlx-v740).
  (regenerate [this trace selection]
    (let [trace (ensure-joint-self this :regenerate trace)
          kern (:kernel this)
          cregen (cops/get-compiled-regenerate kern)
          {:keys [args choices]} trace
          [init-carry inputs] args
          n (count inputs)
          old-step-scores (::step-scores (meta trace))
          init-key (when cregen (rng/fresh-key))]
      (loop [t 0 carry init-carry key init-key
             new-choices cm/EMPTY score ZERO weight ZERO
             outputs [] step-scores [] step-carries [] st :joint]
        (if (>= t n)
          {:trace (tr/with-score-type
                    (with-meta
                      (tr/make-trace {:gen-fn this :args args
                                      :choices new-choices
                                      :retval {:carry carry :outputs outputs}
                                      :score score})
                      (cond-> {::step-scores step-scores ::step-carries step-carries}
                        cregen (assoc ::compiled-path true)))
                    st)
           :weight (if old-step-scores
                     weight
                     (mx/subtract weight (:score trace)))}
          (let [old-sub-choices (cm/get-submap choices t)
                kernel-args [carry (nth inputs t)]
                old-score (if old-step-scores (nth old-step-scores t) ZERO)]
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
                       (conj step-carries new-carry)
                       st))
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
                       (conj step-carries new-carry)
                       (tr/combine-score-types st (tr/score-type new-trace)))))))))))

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
      ;; genmlx-llpt: fall back in update/regenerate mode so retained kernel
      ;; steps are replayed/regenerated, not resampled.
      (if (or (not (:body-fn kern)) (contains? state :old-choices))
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
                    [transition init-sub]
                    (batched-sub state step-constraints sk batch-size batch-zero)
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
      (tr/with-score-type
        (tr/make-trace {:gen-fn this :args args
                        :choices (:choices trace)
                        :retval (:retval trace)
                        :score (:score trace)})
        (tr/score-type trace))))

  p/IGenerate
  (generate [this args constraints]
    (let [transformed-args (f args)
          {:keys [trace weight]} (p/generate inner transformed-args constraints)]
      {:trace (tr/with-score-type
                (tr/make-trace {:gen-fn this :args args
                                :choices (:choices trace)
                                :retval (:retval trace)
                                :score (:score trace)})
                (tr/score-type trace))
       :weight weight})))

(defrecord MapRetvalGF [inner g]
  ;; Transform return value from inner GF
  p/IGenerativeFunction
  (simulate [this args]
    (let [trace (p/simulate inner args)]
      (tr/with-score-type
        (tr/make-trace {:gen-fn this :args args
                        :choices (:choices trace)
                        :retval (g (:retval trace))
                        :score (:score trace)})
        (tr/score-type trace))))

  p/IGenerate
  (generate [this args constraints]
    (let [{:keys [trace weight]} (p/generate inner args constraints)]
      {:trace (tr/with-score-type
                (tr/make-trace {:gen-fn this :args args
                                :choices (:choices trace)
                                :retval (g (:retval trace))
                                :score (:score trace)})
                (tr/score-type trace))
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
          inner-trace (tr/with-score-type
                        (tr/make-trace {:gen-fn (:inner this) :args transformed-args
                                        :choices (:choices trace)
                                        :retval (:retval trace) :score (:score trace)})
                        (tr/score-type trace))
          result (p/update (:inner this) inner-trace constraints)]
      {:trace (tr/with-score-type
                (tr/make-trace {:gen-fn this :args (:args trace)
                                :choices (:choices (:trace result))
                                :retval (:retval (:trace result))
                                :score (:score (:trace result))})
                (tr/score-type (:trace result)))
       :weight (:weight result) :discard (:discard result)}))

  p/IRegenerate
  (regenerate [this trace selection]
    (let [transformed-args ((:f this) (:args trace))
          inner-trace (tr/with-score-type
                        (tr/make-trace {:gen-fn (:inner this) :args transformed-args
                                        :choices (:choices trace)
                                        :retval (:retval trace) :score (:score trace)})
                        (tr/score-type trace))
          result (p/regenerate (:inner this) inner-trace selection)]
      {:trace (tr/with-score-type
                (tr/make-trace {:gen-fn this :args (:args trace)
                                :choices (:choices (:trace result))
                                :retval (:retval (:trace result))
                                :score (:score (:trace result))})
                (tr/score-type (:trace result)))
       :weight (:weight result)})))

(extend-type MapRetvalGF
  p/IUpdate
  (update [this trace constraints]
    (let [inner-trace (tr/with-score-type
                        (tr/make-trace {:gen-fn (:inner this) :args (:args trace)
                                        :choices (:choices trace)
                                        :retval nil :score (:score trace)})
                        (tr/score-type trace))
          result (p/update (:inner this) inner-trace constraints)]
      {:trace (tr/with-score-type
                (tr/make-trace {:gen-fn this :args (:args trace)
                                :choices (:choices (:trace result))
                                :retval ((:g this) (:retval (:trace result)))
                                :score (:score (:trace result))})
                (tr/score-type (:trace result)))
       :weight (:weight result) :discard (:discard result)}))

  p/IRegenerate
  (regenerate [this trace selection]
    (let [inner-trace (tr/with-score-type
                        (tr/make-trace {:gen-fn (:inner this) :args (:args trace)
                                        :choices (:choices trace)
                                        :retval nil :score (:score trace)})
                        (tr/score-type trace))
          result (p/regenerate (:inner this) inner-trace selection)]
      {:trace (tr/with-score-type
                (tr/make-trace {:gen-fn this :args (:args trace)
                                :choices (:choices (:trace result))
                                :retval ((:g this) (:retval (:trace result)))
                                :score (:score (:trace result))})
                (tr/score-type (:trace result)))
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
      (if-let [csim (cops/get-compiled-simulate component)]
        ;; L1-M5: compiled path
        (let [key (rng/fresh-key)
              result (csim key (vec args))
              choices (cm/from-flat-map (:values result))
              choices (cm/set-choice choices [:component-idx]
                                     (mx/scalar (int idx) mx/int32))]
          (tag-joint
            (with-meta
              (tr/make-trace {:gen-fn this :args args
                              :choices choices
                              :retval (:retval result)
                              :score (mx/add (:score result) (:score idx-trace))})
              {::compiled-path true})))
        ;; L0: handler path
        (let [comp-trace (p/simulate component args)]
          (tag-from-traces
            (tr/make-trace {:gen-fn this :args args
                            :choices (cm/set-choice (:choices comp-trace)
                                                    [:component-idx]
                                                    (mx/scalar (int idx) mx/int32))
                            :retval (:retval comp-trace)
                            :score (mx/add (:score comp-trace) (:score idx-trace))})
            [comp-trace])))))

  p/IGenerate
  (generate [this args constraints]
    (let [log-w (log-weights-fn args)
          ;; Check if component index is constrained
          idx-constraint (cm/get-submap constraints :component-idx)
          idx-result (if (cm/has-value? idx-constraint)
                       (let [d (dc/->Distribution :categorical {:logits log-w})]
                         (dc/dist-generate d idx-constraint))
                       (let [d (dc/->Distribution :categorical {:logits log-w})]
                         {:trace (dc/dist-simulate d) :weight ZERO}))
          idx (mx/item (cm/get-value (:choices (:trace idx-result))))
          component (nth components (int idx))
          comp-constraints (without-component-idx constraints)]
      (if-let [cgen (cops/get-compiled-generate component)]
        ;; Compiled path — only the component generate is compiled
        (let [key (rng/fresh-key)
              result (cgen key (vec args) comp-constraints)
              comp-choices (values->choices (:values result))]
          {:trace (tag-joint
                    (with-meta
                      (tr/make-trace {:gen-fn this :args args
                                      :choices (cm/set-choice comp-choices
                                                              [:component-idx]
                                                              (mx/scalar (int idx) mx/int32))
                                      :retval (:retval result)
                                      :score (mx/add (:score result)
                                                     (:score (:trace idx-result)))})
                      {::compiled-path true}))
           :weight (mx/add (:weight result) (:weight idx-result))})
        ;; Fallback: handler path
        (let [{:keys [trace weight]} (p/generate component args comp-constraints)]
          {:trace (tag-from-traces
                    (tr/make-trace {:gen-fn this :args args
                                    :choices (cm/set-choice (:choices trace)
                                                            [:component-idx]
                                                            (mx/scalar (int idx) mx/int32))
                                    :retval (:retval trace)
                                    :score (mx/add (:score trace)
                                                   (:score (:trace idx-result)))})
                    [trace])
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
      ;; genmlx-llpt: fall back in update/regenerate mode so retained component
      ;; sites are replayed/regenerated, not resampled.
      (if (or (not all-dynamic?) (contains? state :old-choices))
        (h/combinator-batched-fallback state addr this (vec args))
        ;; Fast path
        (let [batch-size (:batch-size state)
              log-w ((:log-weights-fn this) args)
              sub-constraints (cm/get-submap (:constraints state) addr)
              ;; Check if component-idx is constrained
              idx-constraint (when (and sub-constraints
                                        (not= sub-constraints cm/EMPTY))
                               (cm/get-submap sub-constraints :component-idx))
              ;; Sample or constrain [N]-shaped component indices.
              ;; Three-way split: k-idx samples the indices, k-comps drives
              ;; the component runs, k-next continues the parent. The parent
              ;; used to continue with the SAME key that sampled the indices,
              ;; correlating every downstream site with index sampling
              ;; (genmlx-njaq; Switch/Unfold already keep these disjoint).
              cat-dist (dc/->Distribution :categorical {:logits log-w})
              [k-next k-comps k-idx] (rng/split-n (:key state) 3)
              [idx-vals idx-score idx-weight]
              (if (and idx-constraint (cm/has-value? idx-constraint))
                ;; Constrained: fixed value, weight = log-prob
                (let [v (cm/get-value idx-constraint)
                      lp (dc/dist-log-prob cat-dist v)]
                  [v lp lp])
                ;; Unconstrained: sample [N] indices
                (let [sampled (dc/dist-sample-n cat-dist k-idx batch-size)
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
              (loop [i 0 results [] key k-comps]
                (if (>= i (count comps))
                  results
                  (let [[ck nk] (rng/split key)
                        [transition init-sub]
                        (batched-sub state inner-constraints ck batch-size batch-zero)
                        result (rt/run-handler transition init-sub
                                               (fn [rt] (apply (:body-fn (nth comps i)) rt
                                                               (vec args))))]
                    (recur (inc i) (conj results result) nk))))
              ;; Combine per-particle results using mx/where on idx-vals
              combined-score
              (where-combine idx-vals batch-zero comp-results :score)
              combined-weight
              (when (contains? state :weight)
                (where-combine idx-vals batch-zero comp-results
                               #(or (:weight %) batch-zero)))
              combined-choices
              (reduce-kv
               (fn [acc i cr]
                 (if (zero? i)
                   (:choices cr)
                   (where-select-choicemap (idx-mask idx-vals i) (:choices cr) acc)))
               cm/EMPTY comp-results)
              combined-retval
              (combine-retvals "Mix" addr idx-vals (mapv :retval comp-results))
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
                         (assoc :key k-next)
                         (h/merge-sub-result addr sub-result))]
          [state' combined-retval])))))

(extend-type MixCombinator
  p/IUpdate
  (update [this trace constraints]
    (let [trace (ensure-joint-self this :update trace)
          old-choices (:choices trace)
          _ (when-not (cm/has-value? (cm/get-submap old-choices :component-idx))
              (throw (ex-info "Mix combinator requires :component-idx in choices"
                              {:operation :update})))
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
          inner-old-choices (without-component-idx old-choices)
          inner-constraints (without-component-idx constraints)]
      (if (= new-idx old-idx)
        ;; Same component: update inner only
        (let [component (nth (:components this) old-idx)
              cupd (cops/get-compiled-update component)]
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
              {:trace (tag-joint
                        (with-meta
                          (tr/make-trace {:gen-fn this :args args
                                          :choices new-choices
                                          :retval (:retval result)
                                          :score new-score})
                          {::compiled-path true}))
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
              {:trace (tag-from-traces
                        (tr/make-trace {:gen-fn this :args args
                                        :choices (cm/set-choice (:choices new-inner-trace)
                                                                [:component-idx]
                                                                (mx/scalar old-idx mx/int32))
                                        :retval (:retval new-inner-trace)
                                        :score new-score})
                        [new-inner-trace])
               ;; Same component: the inner update weight is the combinator
               ;; weight (the unchanged index score cancels). Using the raw
               ;; score delta over-counts any fresh sites the inner update
               ;; sampled on a nested structural change (genmlx-zek9).
               :weight (:weight result)
               :discard (:discard result)})))
        ;; Different component: generate new component from scratch.
        ;; Thesis weight: the generate weight counts only constrained inner
        ;; sites (fresh unconstrained sites cancel against the internal
        ;; proposal); add the new index score and charge the removed old
        ;; component via the recorded old total. Using (:score new-inner-trace)
        ;; here would double-count the new component's fresh latents
        ;; (genmlx-zek9). The new component is keyed so its unconstrained
        ;; latents can be sampled (else p/generate throws 'No PRNG key').
        (let [new-component (ensure-kernel-key (nth (:components this) new-idx))
              new-idx-score (dc/dist-log-prob idx-dist (mx/scalar new-idx mx/int32))
              gen-result (p/generate new-component args inner-constraints)
              new-inner-trace (:trace gen-result)
              new-score (mx/add (:score new-inner-trace) new-idx-score)]
          {:trace (tag-from-traces
                    (tr/make-trace {:gen-fn this :args args
                                    :choices (cm/set-choice (:choices new-inner-trace)
                                                            [:component-idx]
                                                            (mx/scalar new-idx mx/int32))
                                    :retval (:retval new-inner-trace)
                                    :score new-score})
                    [new-inner-trace])
           :weight (mx/subtract (mx/add (:weight gen-result) new-idx-score)
                                (:score trace))
           :discard old-choices}))))

  p/IRegenerate
  (regenerate [this trace selection]
    (let [trace (ensure-joint-self this :regenerate trace)
          old-choices (:choices trace)
          _ (when-not (cm/has-value? (cm/get-submap old-choices :component-idx))
              (throw (ex-info "Mix combinator requires :component-idx in choices"
                              {:operation :regenerate})))
          old-idx (int (mx/item (cm/get-choice old-choices [:component-idx])))
          args (:args trace)
          log-w ((:log-weights-fn this) args)
          idx-dist (dc/->Distribution :categorical {:logits log-w})
          old-idx-score (dc/dist-log-prob idx-dist (mx/scalar old-idx mx/int32))
          idx-selected? (sel/selected? selection :component-idx)]
      (if idx-selected?
        ;; Resample the component index. Only :component-idx is selected, so
        ;; the inner component sites are UNSELECTED and must be retained
        ;; whenever the structure is unchanged (genmlx-zek9).
        (let [new-idx-trace (dc/dist-simulate idx-dist)
              new-idx (int (mx/item (cm/get-value (:choices new-idx-trace))))
              new-idx-score (:score new-idx-trace)]
          (if (= new-idx old-idx)
            ;; Same component resampled: retain the unselected inner choices.
            ;; Nothing under the component moves, so the retained-only weight
            ;; is 0 (and the selected index resample cancels its own delta).
            (let [component (nth (:components this) old-idx)
                  inner-old-choices (without-component-idx old-choices)
                  inner-old-score (mx/subtract (:score trace) old-idx-score)
                  inner-old-trace (tr/make-trace {:gen-fn component :args args
                                                  :choices inner-old-choices
                                                  :retval (:retval trace)
                                                  :score inner-old-score})
                  new-score (mx/add inner-old-score new-idx-score)]
              {:trace (tag-from-traces
                        (tr/make-trace {:gen-fn this :args args
                                        :choices (cm/set-choice inner-old-choices
                                                                [:component-idx]
                                                                (mx/scalar new-idx mx/int32))
                                        :retval (:retval trace)
                                        :score new-score})
                        [inner-old-trace])
               :weight ZERO})
            ;; Different component: EVERYTHING under the Mix is freshly drawn
            ;; from the prior, so the regenerate MH weight is exactly 0: the
            ;; score delta cancels against the forward/backward prior-proposal
            ;; ratio (W = dS - proposal_ratio, proposal_ratio = dS here). The
            ;; old scoreDelta return un-cancelled the whole subtree score at
            ;; parent splices and skewed top-level MH acceptance (genmlx-v740).
            (let [new-component (ensure-kernel-key (nth (:components this) new-idx))
                  new-comp-trace (p/simulate new-component args)
                  new-score (mx/add (:score new-comp-trace) new-idx-score)]
              {:trace (tag-from-traces
                        (tr/make-trace {:gen-fn this :args args
                                        :choices (cm/set-choice (:choices new-comp-trace)
                                                                [:component-idx]
                                                                (mx/scalar new-idx mx/int32))
                                        :retval (:retval new-comp-trace)
                                        :score new-score})
                        [new-comp-trace])
               :weight ZERO})))
        ;; Same component: regenerate within the component
        (let [component (nth (:components this) old-idx)
              inner-old-score (mx/subtract (:score trace) old-idx-score)
              inner-old-choices (without-component-idx old-choices)]
          (if-let [cregen (cops/get-compiled-regenerate component)]
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
              {:trace (tag-joint
                        (with-meta
                          (tr/make-trace {:gen-fn this :args args
                                          :choices new-choices
                                          :retval (:retval result)
                                          :score new-score})
                          {::compiled-path true}))
               :weight weight})
            ;; Fallback: handler path
            (let [inner-old-trace (tr/make-trace {:gen-fn component :args args
                                                  :choices inner-old-choices
                                                  :retval (:retval trace) :score inner-old-score})
                  result (p/regenerate component inner-old-trace selection)
                  new-inner-trace (:trace result)
                  new-score (mx/add (:score new-inner-trace) old-idx-score)]
              {:trace (tag-from-traces
                        (tr/make-trace {:gen-fn this :args args
                                        :choices (cm/set-choice (:choices new-inner-trace)
                                                                [:component-idx]
                                                                (mx/scalar old-idx mx/int32))
                                        :retval (:retval new-inner-trace)
                                        :score new-score})
                        [new-inner-trace])
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
    (let [trace (ensure-joint-self this :update trace)
          old-choices (:choices trace)
          args (:args trace)
          n (count (first args))
          old-element-scores (::element-scores (meta trace))
          has-constraints (not= constraints cm/EMPTY)]
      (cond
        ;; No changes to args and no new constraints: return trace unchanged
        (and (diff/no-change? argdiffs) (not has-constraints))
        {:trace trace :weight ZERO :discard cm/EMPTY}

        ;; vector-diff with stored element scores: optimize
        (and (or (diff/no-change? argdiffs)
                 (= (:diff-type argdiffs) :vector-diff))
             old-element-scores)
        (let [changed-set (if (diff/no-change? argdiffs)
                            #{}
                            (:changed argdiffs))
              kernel (ensure-kernel-key (:kernel this))
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
                                             :retval nil :score (if old-element-scores (nth old-element-scores i) ZERO)})]
                             (p/update kernel old-trace (cm/get-submap constraints i)))
                            ;; Element unchanged: reuse old choices and score
                           {:trace (tr/make-trace
                                    {:gen-fn kernel
                                     :args (mapv #(nth % i) args)
                                     :choices (cm/get-submap old-choices i)
                                     :retval (nth (:retval trace) i nil)
                                     :score (nth old-element-scores i)})
                            :weight ZERO
                            :discard cm/EMPTY}))
                       (range n))
              choices (assemble-choices results (comp :choices :trace))
              retvals (mapv (comp :retval :trace) results)
              score (sum-field results (comp :score :trace))
              weight (mx/subtract score (:score trace))
              discard (assemble-indexed-discards results)
              element-scores (mapv (comp :score :trace) results)]
          {:trace (tag-from-traces
                    (with-meta
                      (tr/make-trace {:gen-fn this :args args
                                      :choices choices :retval retvals :score score})
                      {::element-scores element-scores})
                    (map :trace results))
           :weight weight :discard discard})

        ;; Unknown change: fall back to full update
        :else (p/update this trace constraints)))))

(extend-type UnfoldCombinator
  p/IUpdateWithDiffs
  (update-with-diffs [this trace constraints argdiffs]
    (cond
      ;; No changes at all: fast path
      (and (diff/no-change? argdiffs) (= constraints cm/EMPTY))
      {:trace trace :weight ZERO :discard cm/EMPTY}

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
      {:trace trace :weight ZERO :discard cm/EMPTY}
      (p/update this trace constraints))))

(extend-type ScanCombinator
  p/IUpdateWithDiffs
  (update-with-diffs [this trace constraints argdiffs]
    (cond
      ;; No changes at all: fast path
      (and (diff/no-change? argdiffs) (= constraints cm/EMPTY))
      {:trace trace :weight ZERO :discard cm/EMPTY}

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
      {:trace trace :weight ZERO :discard cm/EMPTY}
      (p/update this trace constraints))))

(extend-type ContramapGF
  p/IUpdateWithDiffs
  (update-with-diffs [this trace constraints argdiffs]
    (if (and (diff/no-change? argdiffs) (= constraints cm/EMPTY))
      {:trace trace :weight ZERO :discard cm/EMPTY}
      (p/update this trace constraints))))

(extend-type MapRetvalGF
  p/IUpdateWithDiffs
  (update-with-diffs [this trace constraints argdiffs]
    (if (and (diff/no-change? argdiffs) (= constraints cm/EMPTY))
      {:trace trace :weight ZERO :discard cm/EMPTY}
      (p/update this trace constraints))))

(extend-type MixCombinator
  p/IUpdateWithDiffs
  (update-with-diffs [this trace constraints argdiffs]
    (if (and (diff/no-change? argdiffs) (= constraints cm/EMPTY))
      {:trace trace :weight ZERO :discard cm/EMPTY}
      (p/update this trace constraints))))

;; ---------------------------------------------------------------------------
;; IUpdateWithArgs implementations (genmlx-s8e8)
;; ---------------------------------------------------------------------------
;; Sequence combinators share one weight convention with their p/update
;; loops: accumulate `nf` (the non-fresh score under the NEW args — child
;; thesis weights re-based by their constructed old scores, verbatim prefix
;; steps by their recorded true scores, fresh steps by their generate
;; weights) and return W = nf - old_total. Elements/steps dropped by the
;; new args never enter nf, so the old-total subtraction charges them; their
;; choices go to the discard. Exact with or without per-element metadata
;; (the constructed old scores cancel inside the child weights).

(defn- throw-update-with-args-unsupported
  [gf]
  (throw (ex-info (str "update-with-args is not supported by "
                       (pr-str (type gf))
                       " — open a bean if you need it; plain update covers"
                       " unchanged args")
                  {:genmlx/error :update-with-args-unsupported
                   :gf-type (type gf)})))

(extend-type MapCombinator
  p/IUpdateWithArgs
  (update-with-args [this trace new-args argdiffs constraints]
    (let [trace (ensure-joint-self this :update trace)
          kern (ensure-kernel-key (:kernel this))
          old-args (:args trace)
          old-choices (:choices trace)
          old-n (count (first old-args))
          new-n (count (first new-args))
          old-element-scores (::element-scores (meta trace))
          changed (when (diff/vector-diff? argdiffs) (:changed argdiffs))
          ;; Verbatim retention is valid only when the caller asserts the
          ;; element unchanged (vector-diff), it is unconstrained, and the
          ;; true element score is recorded.
          verbatim? (fn [i]
                      (and old-element-scores changed
                           (not (contains? changed i))
                           (= (cm/get-submap constraints i) cm/EMPTY)))]
      (loop [i 0
             choices cm/EMPTY score ZERO nf ZERO
             discard cm/EMPTY
             retvals [] element-scores []
             st :joint]
        (if (>= i new-n)
          (let [discard (reduce (fn [d j]
                                  (cm/set-choice d [j] (cm/get-submap old-choices j)))
                                discard (range new-n old-n))]
            {:trace (tr/with-score-type
                      (with-meta
                        (tr/make-trace {:gen-fn this :args new-args
                                        :choices choices :retval retvals :score score})
                        {::element-scores element-scores})
                      st)
             :weight (mx/subtract nf (:score trace))
             :discard discard})
          (let [elem-args-new (mapv #(nth % i) new-args)
                sub-constraints (cm/get-submap constraints i)]
            (cond
              ;; Retained verbatim
              (and (< i old-n) (verbatim? i))
              (let [sub (cm/get-submap old-choices i)
                    s-i (nth old-element-scores i)]
                (recur (inc i)
                       (cm/set-choice choices [i] sub)
                       (mx/add score s-i)
                       (mx/add nf s-i)
                       discard
                       (conj retvals (nth (:retval trace) i))
                       (conj element-scores s-i)
                       st))

              ;; Kept: re-execute under the new element args
              (< i old-n)
              (let [elem-args-old (mapv #(nth % i) old-args)
                    c-old (if old-element-scores (nth old-element-scores i) ZERO)
                    old-sub (tr/make-trace
                              {:gen-fn kern :args elem-args-old
                               :choices (cm/get-submap old-choices i)
                               :retval nil :score c-old})
                    result (p/update-with-args kern old-sub elem-args-new
                                               :unknown sub-constraints)
                    ntr (:trace result)]
                (recur (inc i)
                       (cm/set-choice choices [i] (:choices ntr))
                       (mx/add score (:score ntr))
                       (mx/add nf (mx/add (:weight result) c-old))
                       (if (and (:discard result) (not= (:discard result) cm/EMPTY))
                         (cm/set-choice discard [i] (:discard result))
                         discard)
                       (conj retvals (:retval ntr))
                       (conj element-scores (:score ntr))
                       (tr/combine-score-types st (tr/score-type ntr))))

              ;; Fresh element
              :else
              (let [result (p/generate kern elem-args-new sub-constraints)
                    ntr (:trace result)]
                (recur (inc i)
                       (cm/set-choice choices [i] (:choices ntr))
                       (mx/add score (:score ntr))
                       (mx/add nf (:weight result))
                       discard
                       (conj retvals (:retval ntr))
                       (conj element-scores (:score ntr))
                       (tr/combine-score-types st (tr/score-type ntr)))))))))))

(extend-type UnfoldCombinator
  p/IUpdateWithArgs
  (update-with-args [this trace new-args argdiffs constraints]
    (let [trace (ensure-joint-self this :update trace)
          kern (ensure-kernel-key (:kernel this))
          old-args (:args trace)
          [old-n old-init & old-extra] old-args
          [new-n new-init & new-extra] new-args
          choices (:choices trace)
          old-states (:retval trace)
          old-step-scores (::step-scores (meta trace))
          ;; Host = is the prefix-enabler: cheap and verifiable for numbers/
          ;; vectors, reference-conservative for MLX arrays (same object ✓,
          ;; numerically-equal copy → full sweep, still exact).
          shared-config? (and (= old-init new-init) (= old-extra new-extra))
          kept-n (min old-n new-n)
          boundary (if (and shared-config? old-step-scores)
                     (loop [t 0]
                       (cond
                         (>= t kept-n) kept-n
                         (not= (cm/get-submap constraints t) cm/EMPTY) t
                         :else (recur (inc t))))
                     0)
          prefix-choices (if (pos? boundary)
                           (reduce (fn [cm t]
                                     (cm/set-choice cm [t] (cm/get-submap choices t)))
                                   cm/EMPTY (range boundary))
                           cm/EMPTY)
          prefix-score (if (pos? boundary)
                         (reduce (fn [acc t] (mx/add acc (nth old-step-scores t)))
                                 ZERO (range boundary))
                         ZERO)
          prefix-states (if (pos? boundary) (subvec old-states 0 boundary) [])
          prefix-step-scores (if (pos? boundary)
                               (subvec (vec old-step-scores) 0 boundary)
                               [])
          start-state (if (pos? boundary)
                        (nth old-states (dec boundary))
                        new-init)
          old-state-at (fn [t threaded]
                         (cond
                           (zero? t) old-init
                           (vector? old-states) (nth old-states (dec t))
                           :else threaded))]
      (loop [t boundary state start-state
             new-choices prefix-choices score prefix-score nf prefix-score
             discard cm/EMPTY
             states prefix-states step-scores prefix-step-scores
             st :joint]
        (if (>= t new-n)
          (let [discard (reduce (fn [d j]
                                  (cm/set-choice d [j] (cm/get-submap choices j)))
                                discard (range new-n old-n))]
            {:trace (tr/with-score-type
                      (with-meta
                        (tr/make-trace {:gen-fn this :args new-args
                                        :choices new-choices :retval states :score score})
                        {::step-scores step-scores})
                      st)
             :weight (mx/subtract nf (:score trace))
             :discard discard})
          (let [kernel-args (into [t state] new-extra)
                sub-constraints (cm/get-submap constraints t)]
            (if (< t old-n)
              ;; Kept step: child update-with-args under the new threaded state
              (let [c-old (if old-step-scores (nth old-step-scores t) ZERO)
                    old-sub (tr/make-trace
                              {:gen-fn kern
                               :args (into [t (old-state-at t state)] old-extra)
                               :choices (cm/get-submap choices t)
                               :retval nil :score c-old})
                    result (p/update-with-args kern old-sub kernel-args
                                               :unknown sub-constraints)
                    ntr (:trace result)
                    new-state (:retval ntr)]
                (recur (inc t) new-state
                       (cm/set-choice new-choices [t] (:choices ntr))
                       (mx/add score (:score ntr))
                       (mx/add nf (mx/add (:weight result) c-old))
                       (if (and (:discard result) (not= (:discard result) cm/EMPTY))
                         (cm/set-choice discard [t] (:discard result))
                         discard)
                       (conj states new-state)
                       (conj step-scores (:score ntr))
                       (tr/combine-score-types st (tr/score-type ntr))))
              ;; Fresh step: child generate
              (let [result (p/generate kern kernel-args sub-constraints)
                    ntr (:trace result)
                    new-state (:retval ntr)]
                (recur (inc t) new-state
                       (cm/set-choice new-choices [t] (:choices ntr))
                       (mx/add score (:score ntr))
                       (mx/add nf (:weight result))
                       discard
                       (conj states new-state)
                       (conj step-scores (:score ntr))
                       (tr/combine-score-types st (tr/score-type ntr)))))))))))

(extend-type ScanCombinator
  p/IUpdateWithArgs
  (update-with-args [this trace new-args argdiffs constraints]
    (let [trace (ensure-joint-self this :update trace)
          kern (ensure-kernel-key (:kernel this))
          [old-init-carry old-inputs] (:args trace)
          [new-init-carry new-inputs] new-args
          choices (:choices trace)
          old-n (count old-inputs)
          new-n (count new-inputs)
          old-step-scores (::step-scores (meta trace))
          old-step-carries (::step-carries (meta trace))
          kept-n (min old-n new-n)
          ;; Prefix-skip up to the first constrained step or first changed
          ;; input — both verified, not asserted (host = on inputs).
          boundary (if (and (= old-init-carry new-init-carry)
                            old-step-scores old-step-carries)
                     (loop [t 0]
                       (cond
                         (>= t kept-n) kept-n
                         (not= (cm/get-submap constraints t) cm/EMPTY) t
                         (not= (nth old-inputs t) (nth new-inputs t)) t
                         :else (recur (inc t))))
                     0)
          prefix-choices (if (pos? boundary)
                           (reduce (fn [cm t]
                                     (cm/set-choice cm [t] (cm/get-submap choices t)))
                                   cm/EMPTY (range boundary))
                           cm/EMPTY)
          prefix-score (if (pos? boundary)
                         (reduce (fn [acc t] (mx/add acc (nth old-step-scores t)))
                                 ZERO (range boundary))
                         ZERO)
          prefix-outputs (if (pos? boundary)
                           (subvec (:outputs (:retval trace)) 0 boundary)
                           [])
          prefix-step-scores (if (pos? boundary)
                               (subvec (vec old-step-scores) 0 boundary)
                               [])
          prefix-step-carries (if (pos? boundary)
                                (subvec (vec old-step-carries) 0 boundary)
                                [])
          start-carry (if (pos? boundary)
                        (nth old-step-carries (dec boundary))
                        new-init-carry)
          old-carry-at (fn [t threaded]
                         (cond
                           (zero? t) old-init-carry
                           old-step-carries (nth old-step-carries (dec t))
                           :else threaded))]
      (loop [t boundary carry start-carry
             new-choices prefix-choices score prefix-score nf prefix-score
             discard cm/EMPTY
             outputs prefix-outputs
             step-scores prefix-step-scores step-carries prefix-step-carries
             st :joint]
        (if (>= t new-n)
          (let [discard (reduce (fn [d j]
                                  (cm/set-choice d [j] (cm/get-submap choices j)))
                                discard (range new-n old-n))]
            {:trace (tr/with-score-type
                      (with-meta
                        (tr/make-trace {:gen-fn this :args new-args
                                        :choices new-choices
                                        :retval {:carry carry :outputs outputs}
                                        :score score})
                        {::step-scores step-scores ::step-carries step-carries})
                      st)
             :weight (mx/subtract nf (:score trace))
             :discard discard})
          (let [kernel-args [carry (nth new-inputs t)]
                sub-constraints (cm/get-submap constraints t)]
            (if (< t old-n)
              ;; Kept step
              (let [c-old (if old-step-scores (nth old-step-scores t) ZERO)
                    old-sub (tr/make-trace
                              {:gen-fn kern
                               :args [(old-carry-at t carry) (nth old-inputs t)]
                               :choices (cm/get-submap choices t)
                               :retval nil :score c-old})
                    result (p/update-with-args kern old-sub kernel-args
                                               :unknown sub-constraints)
                    ntr (:trace result)
                    [new-carry output] (:retval ntr)]
                (recur (inc t) new-carry
                       (cm/set-choice new-choices [t] (:choices ntr))
                       (mx/add score (:score ntr))
                       (mx/add nf (mx/add (:weight result) c-old))
                       (if (and (:discard result) (not= (:discard result) cm/EMPTY))
                         (cm/set-choice discard [t] (:discard result))
                         discard)
                       (conj outputs output)
                       (conj step-scores (:score ntr))
                       (conj step-carries new-carry)
                       (tr/combine-score-types st (tr/score-type ntr))))
              ;; Fresh step
              (let [result (p/generate kern kernel-args sub-constraints)
                    ntr (:trace result)
                    [new-carry output] (:retval ntr)]
                (recur (inc t) new-carry
                       (cm/set-choice new-choices [t] (:choices ntr))
                       (mx/add score (:score ntr))
                       (mx/add nf (:weight result))
                       discard
                       (conj outputs output)
                       (conj step-scores (:score ntr))
                       (conj step-carries new-carry)
                       (tr/combine-score-types st (tr/score-type ntr)))))))))))

(extend-type SwitchCombinator
  p/IUpdateWithArgs
  (update-with-args [this trace new-args argdiffs constraints]
    (let [trace (ensure-joint-self this :update trace)
          [new-idx & new-bargs] new-args
          old-idx (or (::switch-idx (meta trace)) (first (:args trace)))]
      (if (= old-idx new-idx)
        ;; Same branch: delegate with the TRUE old score, so the child's
        ;; thesis weight is the Switch weight directly.
        (let [branch (ensure-kernel-key (nth (:branches this) new-idx))
              old-branch-trace (tr/make-trace
                                 {:gen-fn branch :args (vec (rest (:args trace)))
                                  :choices (:choices trace)
                                  :retval (:retval trace) :score (:score trace)})
              result (p/update-with-args branch old-branch-trace (vec new-bargs)
                                         :unknown constraints)
              nbt (:trace result)]
          {:trace (tag-from-traces
                    (with-meta
                      (tr/make-trace {:gen-fn this :args new-args
                                      :choices (:choices nbt)
                                      :retval (:retval nbt)
                                      :score (:score nbt)})
                      {::switch-idx new-idx})
                    [nbt])
           :weight (:weight result) :discard (:discard result)})
        ;; Branch flip: generate the new branch; the removed branch is
        ;; charged via its recorded score and discarded whole.
        (let [new-branch (ensure-kernel-key (nth (:branches this) new-idx))
              gen-result (p/generate new-branch (vec new-bargs) constraints)
              nbt (:trace gen-result)]
          {:trace (tag-from-traces
                    (with-meta
                      (tr/make-trace {:gen-fn this :args new-args
                                      :choices (:choices nbt)
                                      :retval (:retval nbt)
                                      :score (:score nbt)})
                      {::switch-idx new-idx})
                    [nbt])
           :weight (mx/subtract (:weight gen-result) (:score trace))
           :discard (:choices trace)})))))

(extend-type ContramapGF
  p/IUpdateWithArgs
  (update-with-args [this trace new-args argdiffs constraints]
    ;; argdiffs describe the PRE-transform args; after f they are :unknown.
    (let [inner (ensure-kernel-key (:inner this))
          old-inner-args ((:f this) (:args trace))
          new-inner-args ((:f this) new-args)
          inner-trace (tr/with-score-type
                        (tr/make-trace {:gen-fn inner :args old-inner-args
                                        :choices (:choices trace)
                                        :retval (:retval trace) :score (:score trace)})
                        (tr/score-type trace))
          result (p/update-with-args inner inner-trace new-inner-args
                                     :unknown constraints)]
      {:trace (tr/with-score-type
                (tr/make-trace {:gen-fn this :args new-args
                                :choices (:choices (:trace result))
                                :retval (:retval (:trace result))
                                :score (:score (:trace result))})
                (tr/score-type (:trace result)))
       :weight (:weight result) :discard (:discard result)})))

(extend-type MapRetvalGF
  p/IUpdateWithArgs
  (update-with-args [this trace new-args argdiffs constraints]
    (let [inner (ensure-kernel-key (:inner this))
          inner-trace (tr/with-score-type
                        (tr/make-trace {:gen-fn inner :args (:args trace)
                                        :choices (:choices trace)
                                        :retval (:retval trace) :score (:score trace)})
                        (tr/score-type trace))
          result (p/update-with-args inner inner-trace new-args
                                     argdiffs constraints)]
      {:trace (tr/with-score-type
                (tr/make-trace {:gen-fn this :args new-args
                                :choices (:choices (:trace result))
                                :retval ((:g this) (:retval (:trace result)))
                                :score (:score (:trace result))})
                (tr/score-type (:trace result)))
       :weight (:weight result) :discard (:discard result)})))

(extend-type MaskCombinator
  p/IUpdateWithArgs
  (update-with-args [this trace new-args argdiffs constraints]
    (throw-update-with-args-unsupported this)))

(extend-type MixCombinator
  p/IUpdateWithArgs
  (update-with-args [this trace new-args argdiffs constraints]
    (throw-update-with-args-unsupported this)))

(extend-type RecurseCombinator
  p/IUpdateWithArgs
  (update-with-args [this trace new-args argdiffs constraints]
    (throw-update-with-args-unsupported this)))

;; ---------------------------------------------------------------------------
;; IProject implementations
;; ---------------------------------------------------------------------------

(extend-type MapCombinator
  p/IProject
  (project [this trace selection]
    (let [trace (ensure-joint-self this :project trace)
          old-choices (:choices trace)
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
                                  :score (if old-element-scores (nth old-element-scores i) ZERO)})
                      w (p/project kernel sub-trace
                                   (sel/get-subselection selection i))]
                  (mx/add acc w)))
              ZERO
              (range n)))))

(extend-type UnfoldCombinator
  p/IProject
  (project [this trace selection]
    (let [kern (:kernel this)
          {:keys [args choices]} trace
          [n init-state & extra] args]
      (loop [t 0 state init-state
             weight ZERO]
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
    (let [trace (ensure-joint-self this :project trace)
          [idx & branch-args] (:args trace)
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
             weight ZERO]
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
    (let [trace (ensure-joint-self this :project trace)
          [active? & inner-args] (:args trace)]
      (if active?
        (let [inner-trace (tr/make-trace
                           {:gen-fn (:inner this) :args (vec inner-args)
                            :choices (:choices trace)
                            :retval (:retval trace) :score (:score trace)})]
          (p/project (:inner this) inner-trace selection))
        ZERO))))

(extend-type MixCombinator
  p/IProject
  (project [this trace selection]
    (let [trace (ensure-joint-self this :project trace)
          old-choices (:choices trace)
          _ (when-not (cm/has-value? (cm/get-submap old-choices :component-idx))
              (throw (ex-info "Mix combinator requires :component-idx in choices"
                              {:operation :project})))
          old-idx (int (mx/item (cm/get-choice old-choices [:component-idx])))
          args (:args trace)
          log-w ((:log-weights-fn this) args)
          idx-dist (dc/->Distribution :categorical {:logits log-w})
          ;; Project the component-idx if selected
          idx-weight (if (sel/selected? selection :component-idx)
                       (dc/dist-log-prob idx-dist (mx/scalar old-idx mx/int32))
                       ZERO)
          ;; Project the inner component
          component (nth (:components this) old-idx)
          old-idx-score (dc/dist-log-prob idx-dist (mx/scalar old-idx mx/int32))
          inner-old-choices (without-component-idx old-choices)
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
          inner-trace (tr/with-score-type
                        (tr/make-trace {:gen-fn (:inner this) :args transformed-args
                                        :choices (:choices trace)
                                        :retval (:retval trace) :score (:score trace)})
                        (tr/score-type trace))]
      (p/project (:inner this) inner-trace selection))))

(extend-type MapRetvalGF
  p/IProject
  (project [this trace selection]
    (let [inner-trace (tr/with-score-type
                        (tr/make-trace {:gen-fn (:inner this) :args (:args trace)
                                        :choices (:choices trace)
                                        :retval nil :score (:score trace)})
                        (tr/score-type trace))]
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
      (loop [t 0 state init-state weight ZERO states []]
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
             choices cm/EMPTY weight ZERO states []]
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
      (loop [t 0 carry init-carry weight ZERO outputs []]
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
             choices cm/EMPTY weight ZERO outputs []]
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
        {:retval nil :weight ZERO})))

  p/IPropose
  (propose [this args]
    (let [[active? & inner-args] args]
      (if active?
        (p/propose (:inner this) (vec inner-args))
        {:choices cm/EMPTY :weight ZERO :retval nil}))))

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
    (when-not (cm/has-value? (cm/get-submap choices :component-idx))
      (throw (ex-info "Mix combinator requires :component-idx in choices"
                      {:operation :assess})))
    (let [log-w ((:log-weights-fn this) args)
          idx-val (cm/get-choice choices [:component-idx])
          idx (int (mx/item idx-val))
          idx-dist (dc/->Distribution :categorical {:logits log-w})
          idx-weight (dc/dist-log-prob idx-dist (mx/scalar idx mx/int32))
          component (nth (:components this) idx)
          inner-choices (without-component-idx choices)
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
