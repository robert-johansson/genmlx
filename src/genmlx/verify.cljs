(ns genmlx.verify
  "Static validation for generative functions.

   Implements the five DML restrictions from Cusumano-Towner 2020 PhD
   thesis [T], Chapter 2.2.1, p.63:

     1. Halts with probability 1          (check-halts, runtime trials)
     2. Addresses must be unique           (validate-transition, duplicate detection)
     3. No external randomness             (check-no-external-randomness, source analysis)
     4. No mutation                        (check-no-mutation, source analysis)
     5. No HOF passing of gen fns          (check-no-hof-gen-fns, source analysis)

   Plus structural checks beyond the DML restrictions:
     - Non-finite scores (e.g. zero-sigma gaussian)
     - Empty models (no trace calls)
     - Materialization in body (eval!/item breaks vectorized execution)

   (validate-gen-fn gf args)  => {:valid? bool :violations [...] :trace trace}

   Multiple trials can catch conditional duplicates that only appear in
   some execution paths."
  (:require [genmlx.runtime :as rt]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist.core :as dc]
            [genmlx.trace :as tr]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [clojure.string :as str]))

;; ---------------------------------------------------------------------------
;; Source analysis (no execution needed)
;; ---------------------------------------------------------------------------

(defn- check-source-materialization
  "Walk the source form tree looking for eval! or item symbols."
  [source]
  (when source
    (->> (tree-seq coll? seq source)
         (filter symbol?)
         (filter #(#{"eval!" "item"} (name %)))
         (mapv (fn [s] {:type :materialization-in-body
                        :severity :warning
                        :message (str "Found " s " in model body — breaks vectorized execution")})))))

(defn- external-random?
  "True if sym refers to an untraced source of randomness."
  [sym]
  (or (and (nil? (namespace sym))
           (#{"rand" "rand-int" "rand-nth"} (name sym)))
      (and (= "js" (namespace sym))
           (str/includes? (name sym) "random"))
      (and (= "Math" (namespace sym))
           (= "random" (name sym)))))

(defn check-no-external-randomness
  "Walk the source form tree for untraced randomness symbols.
   DML restriction 3: no external randomness affecting control flow."
  [source]
  (when source
    (->> (tree-seq coll? seq source)
         (filter symbol?)
         (filter external-random?)
         (mapv (fn [s]
                 {:type :external-randomness
                  :severity :warning
                  :message (str "Found " s " in model body — untraced randomness violates DML restriction 3")})))))

(def ^:private mutation-syms
  "Symbols that indicate mutation in model body."
  #{"set!" "aset" "reset!" "swap!" "vreset!" "vswap!" "volatile!" "atom"})

(defn check-no-mutation
  "Walk the source form tree for mutation symbols.
   DML restriction 4: no mutation in model body."
  [source]
  (when source
    (->> (tree-seq coll? seq source)
         (filter symbol?)
         (filter #(mutation-syms (name %)))
         (mapv (fn [s]
                 {:type :mutation
                  :severity :warning
                  :message (str "Found " s " in model body — mutation violates DML restriction 4")})))))

(def ^:private hof-syms
  "Higher-order function symbols that should not receive gen fns."
  #{"map" "mapv" "filter" "filterv" "reduce" "keep" "some" "remove" "mapcat" "every?"})

(defn- gen-form?
  "True if form looks like a gen fn expression: (gen ...)."
  [form]
  (and (seq? form) (symbol? (first form)) (= "gen" (name (first form)))))

(defn check-no-hof-gen-fns
  "Walk the source form tree for gen fns passed to higher-order functions.
   DML restriction 5: use combinators (Map, Unfold, etc.) instead."
  [source]
  (when source
    (->> (tree-seq coll? seq source)
         (filter seq?)
         (filter (fn [form]
                   (and (symbol? (first form))
                        (hof-syms (name (first form)))
                        (some gen-form? (rest form)))))
         (mapv (fn [form]
                 {:type :hof-gen-fn
                  :severity :warning
                  :message (str "Found gen fn passed to " (first form)
                                " — use combinators (Map, Unfold, etc.) instead")})))))

;; ---------------------------------------------------------------------------
;; Pure validation transition (replaces validate-handler)
;; ---------------------------------------------------------------------------

(defn- validate-transition
  "Pure state transition for validation: like simulate-transition but
   tracks seen addresses for duplicate detection."
  [state addr dist]
  (let [seen (:seen-addrs state)
        violations (:violations state)
        violations' (if (contains? seen addr)
                      (conj violations {:type :duplicate-address
                                        :severity :error
                                        :message (str "Address " addr " traced more than once")
                                        :addr addr})
                      violations)
        [k1 k2] (rng/split (:key state))
        value (dc/dist-sample dist k2)
        lp (dc/dist-log-prob dist value)]
    [value (-> state
               (assoc :key k1)
               (update :choices cm/set-value addr value)
               (update :score #(mx/add % lp))
               (assoc :seen-addrs (conj seen addr))
               (assoc :violations violations'))]))

;; ---------------------------------------------------------------------------
;; Single-trial validation
;; ---------------------------------------------------------------------------

(defn- run-validation-trial
  "Run one validation trial. Returns {:violations [...] :trace trace} or
   {:violations [{:type :execution-error ...}]} on exception."
  [gf args key]
  (try
    (let [result (rt/run-handler validate-transition
                                 {:choices cm/EMPTY
                                  :score (mx/scalar 0.0)
                                  :key key
                                  :seen-addrs #{}
                                  :violations []
                                  :executor nil}
                                 (fn [rt] (apply (:body-fn gf) rt args)))
          violations (:violations result)
          trace (tr/make-trace {:gen-fn gf :args args
                                :choices (:choices result)
                                :retval (:retval result)
                                :score (:score result)})
          ;; Check score finiteness
          _ (mx/materialize! (:score result))
          score-val (mx/item (:score result))
          violations (if (js/Number.isFinite score-val)
                       violations
                       (conj violations {:type :non-finite-score
                                         :severity :error
                                         :message (str "Model score is " score-val)}))
          ;; Check empty model
          violations (if (= (:choices result) cm/EMPTY)
                       (conj violations {:type :empty-model
                                         :severity :warning
                                         :message "Model body contains no trace calls"})
                       violations)]
      {:violations violations :trace trace})
    (catch :default e
      {:violations [{:type :execution-error
                     :severity :error
                     :message (str "Model execution failed: " (.-message e))}]
       :trace nil})))

(defn- check-halts
  "Run n-trials simulates to test that the model terminates.
   DML restriction 1: model must halt with probability 1.
   Returns violations vector (empty if all trials succeed)."
  [gf args n-trials key]
  (try
    (let [keys (rng/split-n (rng/ensure-key key) n-trials)]
      (doseq [k keys]
        (p/simulate (dyn/with-key gf k) args))
      [])
    (catch :default e
      [{:type :non-termination
        :severity :warning
        :message (str "Model failed during halting test: " (.-message e))}])))

;; ---------------------------------------------------------------------------
;; Public API
;; ---------------------------------------------------------------------------

(defn validate-gen-fn
  "Validate a generative function for structural correctness.

   Checks all 5 DML restrictions from [T] §2.2.1:
     1. Halts with probability 1 (runtime trials)
     2. Unique addresses (runtime, per-trial)
     3. No external randomness (source analysis)
     4. No mutation (source analysis)
     5. No HOF gen fns (source analysis)

   Plus structural checks:
     - Non-finite scores, empty models, materialization in body

   Options:
     :key       - PRNG key (default: fresh)
     :n-trials  - number of independent runs (default: 1)

   Returns {:valid? bool :violations [...] :trace trace-from-last-trial}"
  ([gf args] (validate-gen-fn gf args {}))
  ([gf args {:keys [key n-trials] :or {n-trials 1}}]
   (let [source (:source gf)
         ;; Source analysis — static, no execution needed
         source-violations (into []
                                 (concat (check-source-materialization source)
                                         (check-no-external-randomness source)
                                         (check-no-mutation source)
                                         (check-no-hof-gen-fns source)))
         ;; Runtime trials — execution-based checks
         base-key (or key (rng/fresh-key))
         trial-keys (rng/split-n (rng/ensure-key base-key) n-trials)
         trial-results (mapv #(run-validation-trial gf args %) trial-keys)
         trial-violations (->> (mapcat :violations trial-results)
                               (into [] (distinct)))
         ;; Halting check — DML restriction 1
         halts-violations (check-halts gf args n-trials base-key)
         ;; Merge all violations
         all-violations (-> (vec source-violations)
                            (into trial-violations)
                            (into halts-violations))
         last-trace (:trace (peek trial-results))
         has-error? (some #(= :error (:severity %)) all-violations)]
     {:valid? (not has-error?)
      :violations all-violations
      :trace last-trace})))
