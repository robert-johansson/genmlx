(ns genmlx.verify
  "Static validation for generative functions.

   (validate-gen-fn gf args)  => {:valid? bool :violations [...] :trace trace}

   Runs a generative function through a validation handler that checks for:
   - Execution errors (model throws)
   - Duplicate addresses (same address traced twice)
   - Non-finite scores (e.g. zero-sigma gaussian)
   - Empty models (no trace calls)
   - Materialization in body (eval!/item breaks vectorized execution)

   Multiple trials can catch conditional duplicates that only appear in
   some execution paths."
  (:require [genmlx.handler :as h]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist.core :as dc]
            [genmlx.trace :as tr]))

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

;; ---------------------------------------------------------------------------
;; Validation handler (wraps simulate logic + address tracking)
;; ---------------------------------------------------------------------------

(defn- validate-handler
  "Like simulate-handler but tracks seen addresses for duplicate detection."
  [addr dist]
  (let [state @h/*state*
        seen (:seen-addrs state)
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
    (vreset! h/*state*
      (-> state
          (assoc :key k1)
          (update :choices cm/set-value addr value)
          (update :score #(mx/add % lp))
          (assoc :seen-addrs (conj seen addr))
          (assoc :violations violations')))
    value))

;; ---------------------------------------------------------------------------
;; Single-trial validation
;; ---------------------------------------------------------------------------

(defn- run-validation-trial
  "Run one validation trial. Returns {:violations [...] :trace trace} or
   {:violations [{:type :execution-error ...}]} on exception."
  [gf args key]
  (try
    (let [result (h/run-handler validate-handler
                   {:choices cm/EMPTY
                    :score (mx/scalar 0.0)
                    :key key
                    :seen-addrs #{}
                    :violations []
                    :executor nil}
                   #(apply (:body-fn gf) args))
          violations (:violations result)
          trace (tr/make-trace {:gen-fn gf :args args
                                :choices (:choices result)
                                :retval (:retval result)
                                :score (:score result)})
          ;; Check score finiteness
          _ (mx/eval! (:score result))
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

;; ---------------------------------------------------------------------------
;; Public API
;; ---------------------------------------------------------------------------

(defn validate-gen-fn
  "Validate a generative function for structural correctness.

   Options:
     :key       - PRNG key (default: fresh)
     :n-trials  - number of independent runs (default: 1)

   Returns {:valid? bool :violations [...] :trace trace-from-last-trial}"
  ([gf args] (validate-gen-fn gf args {}))
  ([gf args {:keys [key n-trials] :or {n-trials 1}}]
   (let [;; Source analysis (static, no execution)
         source-violations (check-source-materialization (:source gf))
         ;; Run trials
         base-key (or key (rng/next-key))
         trial-keys (rng/split-n (rng/ensure-key base-key) n-trials)
         trial-results (mapv #(run-validation-trial gf args %) trial-keys)
         ;; Merge violations across trials (dedupe by type+addr)
         trial-violations (->> (mapcat :violations trial-results)
                               (into [] (distinct)))
         all-violations (into (vec source-violations) trial-violations)
         last-trace (:trace (peek trial-results))
         has-error? (some #(= :error (:severity %)) all-violations)]
     {:valid? (not has-error?)
      :violations all-violations
      :trace last-trace})))
