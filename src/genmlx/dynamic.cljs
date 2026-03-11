(ns genmlx.dynamic
  "DynamicDSLFunction — implements the full GFI by delegating to
   handler-based execution, with optional compiled fast path for
   static models (Level 1)."
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
            [genmlx.conjugacy :as conjugacy]
            [genmlx.rewrite :as rewrite]
            [genmlx.inference.auto-analytical :as auto-analytical]
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

(defrecord DynamicGF [body-fn source schema]
  p/IGenerativeFunction
  (simulate [this args]
    (let [key (let [k (::key (meta this))]
              (cond
                (= k auto-key-sentinel) (rng/fresh-key)
                k k
                :else (throw (ex-info "No PRNG key on gen-fn. Use (dyn/with-key gf key) or (dyn/auto-key gf)."
                                      {:gen-fn (.-source this)}))))
          _ (rng/seed! key)
          result
          (if-let [compiled-sim (:compiled-simulate schema)]
            ;; L1-M2: full compiled path → build trace from result
            (let [result (compiled-sim key (vec args))
                  choices (cm/from-flat-map (:values result))]
              (tr/make-trace
                {:gen-fn this :args args
                 :choices choices
                 :retval  (:retval result)
                 :score   (:score result)}))
            (if-let [compiled-pfx (:compiled-prefix schema)]
              ;; L1-M3: compiled prefix + replay handler
              (let [result (compiled-pfx key (vec args))
                    replay (compiled/make-replay-simulate-transition (:values result))
                    handler-result (rt/run-handler replay
                                     {:choices cm/EMPTY :score (:score result) :key key
                                      :executor execute-sub
                                      :param-store (::param-store (meta this))}
                                     (fn [rt] (apply body-fn rt args)))
                    trace (tr/make-trace
                            {:gen-fn this :args args
                             :choices (:choices handler-result)
                             :retval  (:retval handler-result)
                             :score   (:score handler-result)})]
                (if-let [ss (:splice-scores handler-result)]
                  (with-meta trace {::splice-scores ss})
                  trace))
              ;; L0: handler path
              (let [result (rt/run-handler h/simulate-transition
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
                  trace))))]
      (mx/sweep-dead-arrays!)
      result))

  p/IGenerate
  (generate [this args constraints]
    (let [key (let [k (::key (meta this))]
              (cond
                (= k auto-key-sentinel) (rng/fresh-key)
                k k
                :else (throw (ex-info "No PRNG key on gen-fn. Use (dyn/with-key gf key) or (dyn/auto-key gf)."
                                      {:gen-fn (.-source this)}))))
          _ (rng/seed! key)
          result
          (if (and (:auto-handlers schema)
                   (auto-analytical/some-conjugate-obs-constrained?
                     (:conjugate-pairs schema) constraints))
            ;; L3: auto-analytical handler (address-based conjugate elimination)
            (let [transition (auto-analytical/make-address-dispatch
                               h/generate-transition (:auto-handlers schema))
                  result (rt/run-handler transition
                           {:choices cm/EMPTY :score SCORE-ZERO
                            :weight SCORE-ZERO
                            :key key :constraints constraints
                            :auto-posteriors {}
                            :auto-kalman-beliefs {}
                            :auto-kalman-noise-vars {}
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
                  result-map)))
            (if-let [compiled-gen (:compiled-generate schema)]
              ;; WP-1/M4: compiled generate path (static or branch-rewritten)
              (let [result (compiled-gen key (vec args) constraints)
                    choices (cm/from-flat-map (:values result))
                    trace (tr/make-trace
                            {:gen-fn this :args args
                             :choices choices
                             :retval  (:retval result)
                             :score   (:score result)})]
                {:trace trace :weight (:weight result)})
              (if-let [compiled-pfx-gen (:compiled-prefix-generate schema)]
                ;; WP-2/M3: compiled prefix generate + replay handler
                (let [result (compiled-pfx-gen key (vec args) constraints)
                      replay (compiled/make-replay-generate-transition (:values result))
                      handler-result (rt/run-handler replay
                                       {:choices cm/EMPTY :score (:score result)
                                        :weight (:weight result)
                                        :key key :constraints constraints
                                        :executor execute-sub
                                        :param-store (::param-store (meta this))}
                                       (fn [rt] (apply body-fn rt args)))
                      trace (tr/make-trace
                              {:gen-fn this :args args
                               :choices (:choices handler-result)
                               :retval  (:retval handler-result)
                               :score   (:score handler-result)})]
                  (let [result-map {:trace (if-let [ss (:splice-scores handler-result)]
                                            (with-meta trace {::splice-scores ss})
                                            trace)
                                    :weight (:weight handler-result)}]
                    (if-let [unused (find-unused-constraints constraints (:choices handler-result))]
                      (assoc result-map :unused-constraints unused)
                      result-map)))
                ;; L0: handler path
                (let [result (rt/run-handler h/generate-transition
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
                      result-map))))))]
      (mx/sweep-dead-arrays!)
      result))

  p/IUpdate
  (update [this trace constraints]
    (let [key (let [k (::key (meta this))]
              (cond
                (= k auto-key-sentinel) (rng/fresh-key)
                k k
                :else (throw (ex-info "No PRNG key on gen-fn. Use (dyn/with-key gf key) or (dyn/auto-key gf)."
                                      {:gen-fn (.-source this)}))))
          _ (rng/seed! key)]
      (if-let [compiled-upd (:compiled-update schema)]
        ;; WP-3/M4: compiled update path (static or branch-rewritten)
        (let [result (compiled-upd key (vec (:args trace)) constraints (:choices trace))
              choices (cm/from-flat-map (:values result))
              discard (cm/from-flat-map (:discard result))
              new-trace (tr/make-trace
                          {:gen-fn this :args (:args trace)
                           :choices choices
                           :retval  (:retval result)
                           :score   (:score result)})]
          {:trace new-trace
           :weight (mx/subtract (:score result) (:score trace))
           :discard discard})
        (if-let [compiled-pfx-upd (:compiled-prefix-update schema)]
          ;; WP-4/M3: compiled prefix update + replay handler
          (let [result (compiled-pfx-upd key (vec (:args trace)) constraints (:choices trace))
                prefix-discard (cm/from-flat-map (:discard result))
                replay (compiled/make-replay-update-transition (:values result))
                handler-result (rt/run-handler replay
                                 {:choices cm/EMPTY :score (:score result)
                                  :weight SCORE-ZERO
                                  :key key :constraints constraints
                                  :old-choices (:choices trace)
                                  :discard prefix-discard
                                  :executor execute-sub
                                  :param-store (::param-store (meta this))}
                                 (fn [rt] (apply body-fn rt (:args trace))))
                new-trace (tr/make-trace
                            {:gen-fn this :args (:args trace)
                             :choices (:choices handler-result)
                             :retval  (:retval handler-result)
                             :score   (:score handler-result)})]
            (let [result-map {:trace (if-let [ss (:splice-scores handler-result)]
                                      (with-meta new-trace {::splice-scores ss})
                                      new-trace)
                              :weight (mx/subtract (:score handler-result) (:score trace))
                              :discard (:discard handler-result)}]
              (if-let [unused (find-unused-constraints constraints (:choices handler-result))]
                (assoc result-map :unused-constraints unused)
                result-map)))
          ;; L0: handler path
          (let [result (rt/run-handler h/update-transition
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
                result-map)))))))

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
          ;; L3.5 Opt 3: O(1) check for precomputed transition
          has-regen-analytical? (boolean (:auto-regenerate-transition schema))]
      (if has-regen-analytical?
        ;; L3.5: auto-analytical regenerate (Case B: prior not selected)
        ;; Opt 1: use precomputed transition (no per-step make-address-dispatch)
        ;; Opt 2b: no regen-constraints map — handlers read from :old-choices directly
        (let [result (rt/run-handler (:auto-regenerate-transition schema)
                       {:choices cm/EMPTY :score SCORE-ZERO
                        :weight SCORE-ZERO
                        :key key :selection selection
                        :old-choices (:choices trace)
                        :auto-posteriors {}
                        :auto-kalman-beliefs {}
                        :auto-kalman-noise-vars {}
                        :old-splice-scores (::splice-scores (meta trace))
                        :executor execute-sub
                        :param-store (::param-store (meta this))}
                       (fn [rt] (apply body-fn rt (:args trace))))
              new-score (:score result)
              proposal-ratio (:weight result)
              weight (mx/subtract (mx/subtract new-score old-score) proposal-ratio)
              new-trace (tr/make-trace
                          {:gen-fn this :args (:args trace)
                           :choices (:choices result)
                           :retval  (:retval result)
                           :score   new-score})]
          {:trace (if-let [ss (:splice-scores result)]
                    (with-meta new-trace {::splice-scores ss})
                    new-trace)
           :weight weight})
        (if-let [cregen (:compiled-regenerate schema)]
          ;; WP-6: fully compiled regenerate (M2/M4)
          (let [result (cregen key (vec (:args trace)) (:choices trace) selection)
                new-score (:score result)
                proposal-ratio (:weight result)
                weight (mx/subtract (mx/subtract new-score old-score) proposal-ratio)
                choices (cm/from-flat-map (:values result))
                new-trace (tr/make-trace
                            {:gen-fn this :args (:args trace)
                             :choices choices
                             :retval  (:retval result)
                             :score   new-score})]
            {:trace new-trace :weight weight})
          (if-let [prefix-regen (:compiled-prefix-regenerate schema)]
            ;; WP-6: M3 prefix regenerate + replay handler
            (let [prefix-result (prefix-regen key (vec (:args trace))
                                              (:choices trace) selection)
                  replay (compiled/make-replay-regenerate-transition (:values prefix-result))
                  handler-result (rt/run-handler replay
                                   {:choices cm/EMPTY :score (:score prefix-result)
                                    :weight (:weight prefix-result)
                                    :key key :selection selection
                                    :old-choices (:choices trace)
                                    :executor execute-sub
                                    :param-store (::param-store (meta this))}
                                   (fn [rt] (apply body-fn rt (:args trace))))
                  new-score (:score handler-result)
                  proposal-ratio (:weight handler-result)
                  weight (mx/subtract (mx/subtract new-score old-score) proposal-ratio)
                  new-trace (tr/make-trace
                              {:gen-fn this :args (:args trace)
                               :choices (:choices handler-result)
                               :retval  (:retval handler-result)
                               :score   new-score})]
              {:trace (if-let [ss (:splice-scores handler-result)]
                        (with-meta new-trace {::splice-scores ss})
                        new-trace)
               :weight weight})
            ;; Handler fallback
            (let [result (rt/run-handler h/regenerate-transition
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
                  weight (mx/subtract (mx/subtract new-score old-score) proposal-ratio)
                  new-trace (tr/make-trace
                              {:gen-fn this :args (:args trace)
                               :choices (:choices result)
                               :retval  (:retval result)
                               :score   new-score})]
              {:trace (if-let [ss (:splice-scores result)]
                        (with-meta new-trace {::splice-scores ss})
                        new-trace)
               :weight weight}))))))

  p/IAssess
  (assess [this args choices]
    (if (and (:auto-handlers schema)
             (auto-analytical/some-conjugate-obs-constrained?
               (:conjugate-pairs schema) choices))
      ;; L3.5: auto-analytical handler (address-based conjugate elimination)
      (let [key (let [k (::key (meta this))]
                  (cond
                    (= k auto-key-sentinel) (rng/fresh-key)
                    k k
                    :else (throw (ex-info "No PRNG key on gen-fn." {:gen-fn (.-source this)}))))
            _ (rng/seed! key)
            transition (auto-analytical/make-address-dispatch
                         h/assess-transition (:auto-handlers schema))
            result (rt/run-handler transition
                     {:choices cm/EMPTY :score SCORE-ZERO
                      :weight SCORE-ZERO
                      :key key :constraints choices
                      :auto-posteriors {}
                      :auto-kalman-beliefs {}
                      :auto-kalman-noise-vars {}
                      :executor execute-sub-assess
                      :param-store (::param-store (meta this))}
                     (fn [rt] (apply body-fn rt args)))]
        {:retval (:retval result)
         :weight (:score result)})
      (if-let [cassess (:compiled-assess schema)]
        ;; WP-5: fully compiled assess (M2/M4)
        (let [result (cassess (vec args) choices)]
          {:retval (:retval result)
           :weight (:score result)})
        (if-let [prefix-assess (:compiled-prefix-assess schema)]
          ;; WP-5: M3 prefix assess + replay
          (let [key (let [k (::key (meta this))]
                      (cond
                        (= k auto-key-sentinel) (rng/fresh-key)
                        k k
                        :else (throw (ex-info "No PRNG key on gen-fn." {:gen-fn (.-source this)}))))
                _ (rng/seed! key)
                prefix-result (prefix-assess (vec args) choices)
                replay-transition (compiled/make-replay-assess-transition (:values prefix-result))
                result (rt/run-handler replay-transition
                         {:choices cm/EMPTY :score (:score prefix-result)
                          :weight (:score prefix-result)
                          :key key :constraints choices
                          :executor execute-sub-assess
                          :param-store (::param-store (meta this))}
                         (fn [rt] (apply body-fn rt args)))]
            {:retval (:retval result)
             :weight (:score result)})
          ;; Handler fallback
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
             :weight (:score result)})))))

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
    (if-let [cproj (:compiled-project schema)]
      ;; WP-5: fully compiled project (M2/M4)
      (cproj (vec (:args trace)) (:choices trace) selection)
      (if-let [prefix-proj (:compiled-prefix-project schema)]
        ;; WP-5: M3 prefix project + replay
        (let [key (let [k (::key (meta this))]
                    (cond
                      (= k auto-key-sentinel) (rng/fresh-key)
                      k k
                      :else (throw (ex-info "No PRNG key on gen-fn." {:gen-fn (.-source this)}))))
              _ (rng/seed! key)
              prefix-result (prefix-proj (vec (:args trace)) (:choices trace) selection)
              replay-transition (compiled/make-replay-project-transition (:values prefix-result))
              result (rt/run-handler replay-transition
                       {:choices cm/EMPTY :score SCORE-ZERO
                        :weight (:weight prefix-result)
                        :key key :selection selection
                        :old-choices (:choices trace)
                        :constraints cm/EMPTY
                        :executor execute-sub-project
                        :param-store (::param-store (meta this))}
                       (fn [rt] (apply body-fn rt (:args trace))))]
          (:weight result))
        ;; Handler fallback
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
          (:weight result)))))

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
  [[:compiled-simulate   compiled/make-compiled-simulate]
   [:compiled-generate   compiled/make-compiled-generate]
   [:compiled-update     compiled/make-compiled-update]
   [:compiled-assess     compiled/make-compiled-assess]
   [:compiled-project    compiled/make-compiled-project]
   [:compiled-regenerate compiled/make-compiled-regenerate]])

(def ^:private branch-ops
  [[:compiled-simulate   compiled/make-branch-rewritten-simulate]
   [:compiled-generate   compiled/make-branch-rewritten-generate]
   [:compiled-update     compiled/make-branch-rewritten-update]
   [:compiled-assess     compiled/make-branch-rewritten-assess]
   [:compiled-project    compiled/make-branch-rewritten-project]
   [:compiled-regenerate compiled/make-branch-rewritten-regenerate]])

(def ^:private prefix-ops
  [[:compiled-prefix             compiled/make-compiled-prefix            :compiled-prefix-addrs]
   [:compiled-prefix-generate    compiled/make-compiled-prefix-generate   nil]
   [:compiled-prefix-update      compiled/make-compiled-prefix-update     nil]
   [:compiled-prefix-assess      compiled/make-compiled-prefix-assess     nil]
   [:compiled-prefix-project     compiled/make-compiled-prefix-project    nil]
   [:compiled-prefix-regenerate  compiled/make-compiled-prefix-regenerate nil]])

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
                                           (:conjugate-pairs augmented))
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
