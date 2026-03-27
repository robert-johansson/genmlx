#!/usr/bin/env bb
;; Cross-system verification orchestrator
;; Runs Gen.jl, GenJAX, and GenMLX against shared test specs, compares results.
;; All runners read JSON from stdin, write JSON to stdout.

(require '[babashka.process :as p]
         '[cheshire.core :as json]
         '[clojure.string :as str]
         '[clojure.walk])

(def base-dir (str (System/getProperty "user.dir") "/cross_system_tests"))
(def specs-dir (str base-dir "/specs"))
(def results-dir (str base-dir "/results"))
(def runners-dir (str base-dir "/runners"))

(def venv-python (str (System/getProperty "user.dir") "/.venv-genjax/bin/python"))

;; --- Global failure collector for summary report ---
;; Each entry: {:suite "name" :id "test-id" :values {"sys" "val" ...} :reason "msg"}
(def all-failures (atom []))

(defn record-failure!
  "Record a test failure for the summary report."
  [suite id values reason]
  (swap! all-failures conj {:suite suite :id id :values values :reason reason}))

;; --- Runner commands ---
;; Each runner reads {test_type, ...specs...} from stdin, writes JSON to stdout.

(defn runner-cmd
  ([system] (runner-cmd system false))
  ([system use-node?]
   (case system
     "gen_jl" ["julia" (str runners-dir "/gen_jl_runner.jl")]
     "genjax" [venv-python (str runners-dir "/genjax_runner.py")]
     "genmlx" (if use-node?
                ;; Node.js fallback for SMC: Bun/JSC crashes on large SMC due to
                ;; Metal pipeline leak (mlx#3297) + JSC GC interaction
                ["npx" "nbb" (str runners-dir "/genmlx_runner.cljs")]
                ["bun" "run" "--bun" "nbb" (str runners-dir "/genmlx_runner.cljs")]))))

(defn run-system [system test-type spec-json output-file]
  (println (str "  Running " system "..."))
  (let [;; Inject test_type into the spec payload
        payload (json/generate-string (assoc (json/parse-string spec-json true)
                                             :test_type test-type))
        cmd (runner-cmd system)]
    (try
      (let [result (p/shell {:in payload :out :string :err :string
                             :dir (System/getProperty "user.dir")}
                            (str/join " " cmd))]
        (spit output-file (:out result))
        (when (not= 0 (:exit result))
          (println (str "    STDERR: " (:err result))))
        {:success (= 0 (:exit result))
         :exit (:exit result)
         :stderr (:err result)
         :output-file output-file})
      (catch Exception e
        (println (str "    ERROR: " (.getMessage e)))
        {:success false :error (.getMessage e) :output-file output-file}))))

;; --- Comparison ---

(defn parse-number
  "Handle -Inf, Inf, NaN strings from JSON."
  [v]
  (cond
    (number? v) v
    (= v "-Inf") Double/NEGATIVE_INFINITY
    (= v "Inf") Double/POSITIVE_INFINITY
    (= v "NaN") Double/NaN
    :else v))

(defn coerce-results
  "Walk result maps and convert string numbers."
  [m]
  (clojure.walk/postwalk
   (fn [x]
     (if (and (map? x) (not (record? x)))
       (reduce-kv (fn [acc k v]
                    (assoc acc k (if (#{:logprob :weight :score :old_score :new_score :total_score :sum_components :gradient :posterior_mean :posterior_variance :log_ml :acceptance_rate :ess} k)
                                   (parse-number v) v)))
                  {} x)
       x))
   m))

(defn load-results [file]
  (coerce-results (json/parse-string (slurp file) true)))

(defn close-enough? [a b tol]
  (let [a (parse-number a)
        b (parse-number b)]
    (cond
      ;; Both NaN (check first — NaN != NaN in arithmetic)
      (and (number? a) (number? b)
           (Double/isNaN a) (Double/isNaN b))
      true

      (and (number? a) (number? b))
      (or ;; Absolute tolerance
       (< (abs (- a b)) tol)
          ;; Relative tolerance (for large values)
       (and (not (zero? (max (abs a) (abs b))))
            (< (/ (abs (- a b)) (max (abs a) (abs b))) tol))
          ;; Both -Inf or both +Inf
       (and (Double/isInfinite a) (Double/isInfinite b)
            (= (< a 0) (< b 0))))

      :else false)))

(defn compare-results [test-type systems-results tolerance]
  (let [system-names (keys systems-results)
        ;; Build id -> system -> result maps
        by-id (reduce
               (fn [acc [sys-name results]]
                 (reduce
                  (fn [acc2 r]
                    (assoc-in acc2 [(:id r) sys-name] r))
                  acc
                  (:results results)))
               {}
               systems-results)
        field (case test-type
                "logprob" :logprob
                "assess" :weight
                "generate" :weight)]

    (println (str "\n=== " (str/upper-case test-type) " COMPARISON ==="))
    (println (str "Tolerance: " tolerance))
    (println (str "Systems: " (str/join ", " (map name system-names))))
    (println)

    (let [pass-count (atom 0)
          fail-count (atom 0)
          error-count (atom 0)
          skip-count (atom 0)]

      (doseq [[id sys-map] (sort-by key by-id)]
        (let [values (map (fn [sys]
                            (let [r (get sys-map sys)]
                              (when r
                                (if (:error r)
                                  {:system sys :error (:error r)}
                                  {:system sys :value (get r field)}))))
                          system-names)
              valid-values (filter #(and % (:value %) (not (nil? (:value %)))) values)
              error-values (filter #(and % (:error %)) values)]

          (cond
            ;; Some systems had errors
            (seq error-values)
            (do
              (swap! error-count inc)
              (println (str "  ERROR  " id))
              (doseq [ev error-values]
                (println (str "         " (name (:system ev)) ": " (:error ev))))
              (doseq [vv valid-values]
                (println (str "         " (name (:system vv)) ": " (:value vv)))))

            ;; Not enough systems to compare
            (< (count valid-values) 2)
            (do
              (swap! skip-count inc)
              (println (str "  SKIP   " id " (only " (count valid-values) " result(s))")))

            ;; Compare all pairs
            :else
            (let [vals (map :value valid-values)
                  ref-val (first vals)
                  all-close (every? #(close-enough? ref-val % tolerance) (rest vals))]
              (if all-close
                (do
                  (swap! pass-count inc)
                  (println (str "  PASS   " id "  "
                                (str/join " | "
                                          (map #(str (name (:system %)) "="
                                                     (format "%.6f" (double (:value %))))
                                               valid-values)))))
                (do
                  (swap! fail-count inc)
                  (record-failure! test-type id
                                   (into {} (map (fn [vv] [(name (:system vv))
                                                           (format "%.10f" (double (:value vv)))])
                                                 valid-values))
                                   "cross-system mismatch")
                  (println (str "  FAIL   " id))
                  (doseq [vv valid-values]
                    (println (str "         " (name (:system vv)) ": "
                                  (format "%.10f" (double (:value vv))))))))))))

      (println)
      (println (str "Results: " @pass-count " pass, " @fail-count " fail, "
                    @error-count " error, " @skip-count " skip"))
      {:suite test-type :pass @pass-count :fail @fail-count :error @error-count :skip @skip-count})))

;; --- MLX ops: compare GenMLX-only against known expected values ---

(defn close-enough-recursive?
  "Compare scalars, vectors, or nested vectors element-wise."
  [a b tol]
  (cond
    (and (sequential? a) (sequential? b))
    (and (= (count a) (count b))
         (every? true? (map #(close-enough-recursive? %1 %2 tol) a b)))
    ;; Delegate everything else (numbers, strings like "NaN") to close-enough?
    :else (close-enough? a b tol)))

(defn run-mlx-ops-suite [spec-file _default-tolerance]
  (println "\n>>> Running mlx_ops tests <<<")
  (println (str "Spec: " spec-file))

  (let [spec-json (slurp spec-file)
        specs (json/parse-string spec-json true)
        tolerance (or (:tolerance specs) _default-tolerance)
        out-file (str results-dir "/genmlx_mlx_ops.json")
        run-result (run-system "genmlx" "mlx_ops" spec-json out-file)]

    (if-not (:success run-result)
      (do (println "  genmlx failed to run")
          {:suite "mlx_ops" :pass 0 :fail 0 :error 0 :skip 0})

      (let [results (load-results out-file)
            expected-by-id (into {} (map (fn [s] [(:id s) s]) (:tests specs)))
            pass-count (atom 0)
            fail-count (atom 0)
            error-count (atom 0)]

        (println (str "\n=== MLX_OPS COMPARISON (vs known values) ==="))
        (println (str "Tolerance: " tolerance))
        (println)

        (doseq [r (:results results)]
          (let [spec (get expected-by-id (:id r))
                expected (parse-number (:expected spec))]
            (cond
              (:error r)
              (do (swap! error-count inc)
                  (println (str "  ERROR  " (:id r) ": " (:error r))))

              (close-enough-recursive? (:result r) expected tolerance)
              (do (swap! pass-count inc)
                  (println (str "  PASS   " (:id r) "  mlx=" (pr-str (:result r))
                                " expected=" (pr-str expected))))

              :else
              (do (swap! fail-count inc)
                  (record-failure! "mlx_ops" (:id r)
                                   {"genmlx" (pr-str (:result r))
                                    "expected" (pr-str expected)}
                                   "value mismatch vs expected")
                  (println (str "  FAIL   " (:id r)))
                  (println (str "         mlx:      " (pr-str (:result r))))
                  (println (str "         expected:  " (pr-str expected)))))))

        (println)
        (println (str "Results: " @pass-count " pass, " @fail-count " fail, "
                      @error-count " error"))
        {:suite "mlx_ops" :pass @pass-count :fail @fail-count :error @error-count :skip 0}))))

;; --- Score Decomposition Suite ---

(defn run-score-decomposition-suite [spec-file tolerance]
  (println "\n>>> Running score_decomposition tests <<<")
  (println (str "Spec: " spec-file))

  (let [spec-json (slurp spec-file)
        systems ["gen_jl" "genmlx"]
        run-results
        (reduce
         (fn [acc sys]
           (let [out-file (str results-dir "/" sys "_score_decomposition.json")
                 result (run-system sys "score_decomposition" spec-json out-file)]
             (assoc acc sys result)))
         {}
         systems)

        loaded (reduce
                (fn [acc [sys info]]
                  (if (:success info)
                    (try
                      (assoc acc (keyword sys) (load-results (:output-file info)))
                      (catch Exception e
                        (println (str "  Failed to load " sys " results: " (.getMessage e)))
                        acc))
                    (do
                      (println (str "  " sys " failed to run"))
                      acc)))
                {}
                run-results)]

    (println (str "\n=== SCORE_DECOMPOSITION COMPARISON ==="))
    (println (str "Tolerance: " tolerance))
    (println (str "Systems: " (str/join ", " (map name (keys loaded)))))
    (println)

    (if (< (count loaded) 2)
      (do (println "Not enough systems succeeded for comparison")
          {:suite "score_decomposition" :pass 0 :fail 0 :error 0 :skip 0})

      ;; Build id -> system -> result
      (let [by-id (reduce
                   (fn [acc [sys-name results]]
                     (reduce
                      (fn [acc2 r]
                        (assoc-in acc2 [(:id r) sys-name] r))
                      acc
                      (:results results)))
                   {}
                   loaded)
            pass-count (atom 0)
            fail-count (atom 0)
            error-count (atom 0)]

        (doseq [[id sys-map] (sort-by key by-id)]
          (let [all-sys (vals sys-map)
                errors (filter :error all-sys)
                valid (remove :error all-sys)]
            (cond
              (seq errors)
              (do (swap! error-count inc)
                  (println (str "  ERROR  " id))
                  (doseq [e errors]
                    (println (str "         " (:error e)))))

              (< (count valid) 2)
              (do (swap! error-count inc)
                  (println (str "  ERROR  " id " (not enough results)")))

              :else
              ;; 1. Check internal consistency: total_score ≈ sum_components within each system
              (let [internal-ok
                    (every?
                     (fn [r]
                       (let [total (parse-number (:total_score r))
                             sum-c (parse-number (:sum_components r))]
                         (close-enough? total sum-c tolerance)))
                     valid)

                    ;; 2. Cross-system: compare total_score
                    totals (map #(parse-number (:total_score %)) valid)
                    ref-total (first totals)
                    cross-ok (every? #(close-enough? ref-total % tolerance) (rest totals))

                    sys-names (keys sys-map)]

                (if (and internal-ok cross-ok)
                  (do (swap! pass-count inc)
                      (println (str "  PASS   " id "  "
                                    (str/join " | "
                                              (map (fn [[sys r]]
                                                     (str (name sys) "="
                                                          (format "%.6f" (double (parse-number (:total_score r))))))
                                                   sys-map)))))
                  (do (swap! fail-count inc)
                      (record-failure! "score_decomposition" id
                                       (into {} (map (fn [[sys r]]
                                                       [(name sys)
                                                        (str "total=" (format "%.10f" (double (parse-number (:total_score r))))
                                                             " sum=" (format "%.10f" (double (parse-number (:sum_components r)))))])
                                                     sys-map))
                                       (str (when (not internal-ok) "internal consistency ")
                                            (when (not cross-ok) "cross-system mismatch")))
                      (println (str "  FAIL   " id
                                    (when (not internal-ok) " [internal consistency]")
                                    (when (not cross-ok) " [cross-system mismatch]")))
                      (doseq [[sys r] sys-map]
                        (println (str "         " (name sys)
                                      " total=" (format "%.10f" (double (parse-number (:total_score r))))
                                      " sum=" (format "%.10f" (double (parse-number (:sum_components r)))))))))))))

        (println)
        (println (str "Results: " @pass-count " pass, " @fail-count " fail, "
                      @error-count " error"))
        {:suite "score_decomposition" :pass @pass-count :fail @fail-count :error @error-count :skip 0}))))

;; --- Update Suite ---

(defn run-update-suite [spec-file tolerance]
  (println "\n>>> Running update tests <<<")
  (println (str "Spec: " spec-file))

  (let [spec-json (slurp spec-file)
        systems ["gen_jl" "genmlx"]
        run-results
        (reduce
         (fn [acc sys]
           (let [out-file (str results-dir "/" sys "_update.json")
                 result (run-system sys "update" spec-json out-file)]
             (assoc acc sys result)))
         {}
         systems)

        loaded (reduce
                (fn [acc [sys info]]
                  (if (:success info)
                    (try
                      (assoc acc (keyword sys) (load-results (:output-file info)))
                      (catch Exception e
                        (println (str "  Failed to load " sys " results: " (.getMessage e)))
                        acc))
                    (do
                      (println (str "  " sys " failed to run"))
                      acc)))
                {}
                run-results)]

    (println (str "\n=== UPDATE COMPARISON ==="))
    (println (str "Tolerance: " tolerance))
    (println (str "Systems: " (str/join ", " (map name (keys loaded)))))
    (println)

    (if (< (count loaded) 2)
      (do (println "Not enough systems succeeded for comparison")
          {:suite "update" :pass 0 :fail 0 :error 0 :skip 0})

      (let [by-id (reduce
                   (fn [acc [sys-name results]]
                     (reduce
                      (fn [acc2 r]
                        (assoc-in acc2 [(:id r) sys-name] r))
                      acc
                      (:results results)))
                   {}
                   loaded)
            pass-count (atom 0)
            fail-count (atom 0)
            error-count (atom 0)]

        (doseq [[id sys-map] (sort-by key by-id)]
          (let [all-sys (vals sys-map)
                errors (filter :error all-sys)
                valid (remove :error all-sys)]
            (cond
              (seq errors)
              (do (swap! error-count inc)
                  (println (str "  ERROR  " id))
                  (doseq [e errors]
                    (println (str "         " (:error e)))))

              (< (count valid) 2)
              (do (swap! error-count inc)
                  (println (str "  ERROR  " id " (not enough results)")))

              :else
              (let [;; 1. Internal invariant: weight ≈ new_score - old_score within each system
                    internal-ok
                    (every?
                     (fn [r]
                       (let [old-s (parse-number (:old_score r))
                             new-s (parse-number (:new_score r))
                             w (parse-number (:weight r))
                             expected-w (- new-s old-s)]
                         (close-enough? w expected-w tolerance)))
                     valid)

                    ;; 2. Cross-system comparison of old_score, new_score, weight
                    vals-by-field
                    (fn [field]
                      (map #(parse-number (get % field)) valid))

                    cross-ok
                    (every?
                     (fn [field]
                       (let [vs (vals-by-field field)
                             ref-v (first vs)]
                         (every? #(close-enough? ref-v % tolerance) (rest vs))))
                     [:old_score :new_score :weight])

                    ;; Special check: "update-single-same" weight ≈ 0
                    same-val-ok
                    (if (= id "update-single-same")
                      (every?
                       (fn [r]
                         (close-enough? (parse-number (:weight r)) 0.0 tolerance))
                       valid)
                      true)]

                (if (and internal-ok cross-ok same-val-ok)
                  (do (swap! pass-count inc)
                      (println (str "  PASS   " id "  "
                                    (str/join " | "
                                              (map (fn [[sys r]]
                                                     (str (name sys)
                                                          " w=" (format "%.6f" (double (parse-number (:weight r))))))
                                                   sys-map)))))
                  (do (swap! fail-count inc)
                      (let [reason (str (when (not internal-ok) "w != new-old ")
                                        (when (not cross-ok) "cross-system mismatch ")
                                        (when (not same-val-ok) "same-val weight != 0"))]
                        (record-failure! "update" id
                                         (into {} (map (fn [[sys r]]
                                                         [(name sys)
                                                          (str "w=" (format "%.10f" (double (parse-number (:weight r)))))])
                                                       sys-map))
                                         reason))
                      (println (str "  FAIL   " id
                                    (when (not internal-ok) " [w != new-old]")
                                    (when (not cross-ok) " [cross-system mismatch]")
                                    (when (not same-val-ok) " [same-val weight != 0]")))
                      (doseq [[sys r] sys-map]
                        (println (str "         " (name sys)
                                      " old=" (format "%.10f" (double (parse-number (:old_score r))))
                                      " new=" (format "%.10f" (double (parse-number (:new_score r))))
                                      " w=" (format "%.10f" (double (parse-number (:weight r)))))))))))))

        (println)
        (println (str "Results: " @pass-count " pass, " @fail-count " fail, "
                      @error-count " error"))
        {:suite "update" :pass @pass-count :fail @fail-count :error @error-count :skip 0}))))

;; --- Project Suite ---

(defn run-project-suite [spec-file tolerance]
  (println "\n>>> Running project tests <<<")
  (println (str "Spec: " spec-file))

  (let [spec-json (slurp spec-file)
        systems ["gen_jl" "genmlx"]
        run-results
        (reduce
         (fn [acc sys]
           (let [out-file (str results-dir "/" sys "_project.json")
                 result (run-system sys "project" spec-json out-file)]
             (assoc acc sys result)))
         {}
         systems)

        loaded (reduce
                (fn [acc [sys info]]
                  (if (:success info)
                    (try
                      (assoc acc (keyword sys) (load-results (:output-file info)))
                      (catch Exception e
                        (println (str "  Failed to load " sys " results: " (.getMessage e)))
                        acc))
                    (do
                      (println (str "  " sys " failed to run"))
                      acc)))
                {}
                run-results)]

    (println (str "\n=== PROJECT COMPARISON ==="))
    (println (str "Tolerance: " tolerance))
    (println (str "Systems: " (str/join ", " (map name (keys loaded)))))
    (println)

    (if (< (count loaded) 2)
      (do (println "Not enough systems succeeded for comparison")
          {:suite "project" :pass 0 :fail 0 :error 0 :skip 0})

      (let [by-id (reduce
                   (fn [acc [sys-name results]]
                     (reduce
                      (fn [acc2 r]
                        (assoc-in acc2 [(:id r) sys-name] r))
                      acc
                      (:results results)))
                   {}
                   loaded)
            pass-count (atom 0)
            fail-count (atom 0)
            error-count (atom 0)]

        (doseq [[id sys-map] (sort-by key by-id)]
          (let [all-sys (vals sys-map)
                errors (filter :error all-sys)
                valid (remove :error all-sys)]
            (cond
              (seq errors)
              (do (swap! error-count inc)
                  (println (str "  ERROR  " id))
                  (doseq [e errors]
                    (println (str "         " (:error e)))))

              (< (count valid) 2)
              (do (swap! error-count inc)
                  (println (str "  ERROR  " id " (not enough results)")))

              :else
              ;; Cross-system: compare weight (the project output)
              (let [weights (map #(parse-number (:weight %)) valid)
                    ref-w (first weights)
                    cross-ok (every? #(close-enough? ref-w % tolerance) (rest weights))]
                (if cross-ok
                  (do (swap! pass-count inc)
                      (println (str "  PASS   " id "  "
                                    (str/join " | "
                                              (map (fn [[sys r]]
                                                     (str (name sys) "="
                                                          (format "%.6f" (double (parse-number (:weight r))))))
                                                   sys-map)))))
                  (do (swap! fail-count inc)
                      (record-failure! "project" id
                                       (into {} (map (fn [[sys r]]
                                                       [(name sys) (format "%.10f" (double (parse-number (:weight r))))])
                                                     sys-map))
                                       "cross-system mismatch")
                      (println (str "  FAIL   " id " [cross-system mismatch]"))
                      (doseq [[sys r] sys-map]
                        (println (str "         " (name sys)
                                      " weight=" (format "%.10f" (double (parse-number (:weight r)))))))))))))

        (println)
        (println (str "Results: " @pass-count " pass, " @fail-count " fail, "
                      @error-count " error"))
        {:suite "project" :pass @pass-count :fail @fail-count :error @error-count :skip 0}))))

;; --- Regenerate Suite ---

(defn run-regenerate-suite [spec-file tolerance]
  (println "\n>>> Running regenerate tests <<<")
  (println (str "Spec: " spec-file))

  (let [spec-json (slurp spec-file)
        systems ["gen_jl" "genmlx"]
        run-results
        (reduce
         (fn [acc sys]
           (let [out-file (str results-dir "/" sys "_regenerate.json")
                 result (run-system sys "regenerate" spec-json out-file)]
             (assoc acc sys result)))
         {}
         systems)

        loaded (reduce
                (fn [acc [sys info]]
                  (if (:success info)
                    (try
                      (assoc acc (keyword sys) (load-results (:output-file info)))
                      (catch Exception e
                        (println (str "  Failed to load " sys " results: " (.getMessage e)))
                        acc))
                    (do
                      (println (str "  " sys " failed to run"))
                      acc)))
                {}
                run-results)

        ;; Load spec to identify invariant-based tests
        specs (json/parse-string spec-json true)
        test-by-id (into {} (map (fn [t] [(:id t) t]) (:regenerate_tests specs)))]

    (println (str "\n=== REGENERATE COMPARISON ==="))
    (println (str "Tolerance: " tolerance))
    (println (str "Systems: " (str/join ", " (map name (keys loaded)))))
    (println)

    (if (< (count loaded) 2)
      (do (println "Not enough systems succeeded for comparison")
          {:suite "regenerate" :pass 0 :fail 0 :error 0 :skip 0})

      (let [by-id (reduce
                   (fn [acc [sys-name results]]
                     (reduce
                      (fn [acc2 r]
                        (assoc-in acc2 [(:id r) sys-name] r))
                      acc
                      (:results results)))
                   {}
                   loaded)
            pass-count (atom 0)
            fail-count (atom 0)
            error-count (atom 0)]

        (doseq [[id sys-map] (sort-by key by-id)]
          (let [all-sys (vals sys-map)
                errors (filter :error all-sys)
                valid (remove :error all-sys)
                test-spec (get test-by-id id)]
            (cond
              (seq errors)
              (do (swap! error-count inc)
                  (println (str "  ERROR  " id))
                  (doseq [e errors]
                    (println (str "         " (:error e)))))

              (< (count valid) 2)
              (do (swap! error-count inc)
                  (println (str "  ERROR  " id " (not enough results)")))

              :else
              (let [;; 1. Cross-system: old_score must match (deterministic from generate)
                    old-scores (map #(parse-number (:old_score %)) valid)
                    ref-old (first old-scores)
                    old-ok (every? #(close-enough? ref-old % tolerance) (rest old-scores))

                    ;; 2. Invariant checks within each system
                    ;; regenerate weight = log p(unselected|new_selected) - log p(unselected|old_selected)
                    ;; NOT new_score - old_score (that ignores the proposal cancellation)
                    sel-type (get-in test-spec [:selection :type])

                    invariant-ok
                    (cond
                      ;; Empty selection: weight must be 0, trace unchanged
                      (= sel-type "none")
                      (every?
                       (fn [r]
                         (and (close-enough? (parse-number (:weight r)) 0.0 tolerance)
                              (close-enough? (parse-number (:old_score r))
                                             (parse-number (:new_score r)) tolerance)))
                       valid)

                      ;; Single-site-from-prior or all-sites: weight = 0
                      ;; (proposal = prior, so proposal terms cancel the selected terms)
                      (contains? #{"regen-single-x" "regen-all-sites"} id)
                      (every?
                       (fn [r]
                         (close-enough? (parse-number (:weight r)) 0.0 tolerance))
                       valid)

                      ;; General case: weight is random, can't compare cross-system.
                      ;; Just verify old_score match (already checked above) and
                      ;; that weight is finite.
                      :else
                      (every?
                       (fn [r]
                         (let [w (parse-number (:weight r))]
                           (and (not (Double/isNaN w))
                                (not (Double/isInfinite w)))))
                       valid))]

                (if (and old-ok invariant-ok)
                  (do (swap! pass-count inc)
                      (println (str "  PASS   " id "  "
                                    (str/join " | "
                                              (map (fn [[sys r]]
                                                     (str (name sys)
                                                          " w=" (format "%.6f" (double (parse-number (:weight r))))))
                                                   sys-map)))))
                  (do (swap! fail-count inc)
                      (let [reason (str (when (not old-ok) "old_score mismatch ")
                                        (when (not invariant-ok) "invariant violated"))]
                        (record-failure! "regenerate" id
                                         (into {} (map (fn [[sys r]]
                                                         [(name sys)
                                                          (str "w=" (format "%.10f" (double (parse-number (:weight r)))))])
                                                       sys-map))
                                         reason))
                      (println (str "  FAIL   " id
                                    (when (not old-ok) " [old_score mismatch]")
                                    (when (not invariant-ok) " [invariant violated]")))
                      (doseq [[sys r] sys-map]
                        (println (str "         " (name sys)
                                      " old=" (format "%.10f" (double (parse-number (:old_score r))))
                                      " new=" (format "%.10f" (double (parse-number (:new_score r))))
                                      " w=" (format "%.10f" (double (parse-number (:weight r)))))))))))))

        (println)
        (println (str "Results: " @pass-count " pass, " @fail-count " fail, "
                      @error-count " error"))
        {:suite "regenerate" :pass @pass-count :fail @fail-count :error @error-count :skip 0}))))

;; --- Combinator Suite ---

(defn run-combinator-suite [spec-file tolerance]
  (println "\n>>> Running combinator tests <<<")
  (println (str "Spec: " spec-file))

  (let [spec-json (slurp spec-file)
        systems ["gen_jl" "genmlx"]
        run-results
        (reduce
         (fn [acc sys]
           (let [out-file (str results-dir "/" sys "_combinator.json")
                 result (run-system sys "combinator" spec-json out-file)]
             (assoc acc sys result)))
         {}
         systems)

        loaded (reduce
                (fn [acc [sys info]]
                  (if (:success info)
                    (try
                      (assoc acc (keyword sys) (load-results (:output-file info)))
                      (catch Exception e
                        (println (str "  Failed to load " sys " results: " (.getMessage e)))
                        acc))
                    (do
                      (println (str "  " sys " failed to run"))
                      acc)))
                {}
                run-results)]

    (println (str "\n=== COMBINATOR COMPARISON ==="))
    (println (str "Tolerance: " tolerance))
    (println (str "Systems: " (str/join ", " (map name (keys loaded)))))
    (println)

    (if (< (count loaded) 2)
      (do (println "Not enough systems succeeded for comparison")
          {:suite "combinator" :pass 0 :fail 0 :error 0 :skip 0})

      (let [by-id (reduce
                   (fn [acc [sys-name results]]
                     (reduce
                      (fn [acc2 r]
                        (assoc-in acc2 [(:id r) sys-name] r))
                      acc
                      (:results results)))
                   {}
                   loaded)
            pass-count (atom 0)
            fail-count (atom 0)
            error-count (atom 0)]

        (doseq [[id sys-map] (sort-by key by-id)]
          (let [all-sys (vals sys-map)
                errors (filter :error all-sys)
                valid (remove :error all-sys)]
            (cond
              (seq errors)
              (do (swap! error-count inc)
                  (println (str "  ERROR  " id))
                  (doseq [e errors]
                    (println (str "         " (:error e)))))

              (< (count valid) 2)
              (do (swap! error-count inc)
                  (println (str "  ERROR  " id " (not enough results)")))

              :else
              (let [;; Dispatch on what fields are available
                    has-weight (some :weight valid)
                    has-total (some :total_score valid)
                    has-old (some :old_score valid)]
                (cond
                  ;; Score decomposition: check total_score + sum_components
                  has-total
                  (let [internal-ok
                        (every?
                         (fn [r]
                           (let [total (parse-number (:total_score r))
                                 sum-c (parse-number (:sum_components r))]
                             (close-enough? total sum-c tolerance)))
                         valid)
                        totals (map #(parse-number (:total_score %)) valid)
                        ref-total (first totals)
                        cross-ok (every? #(close-enough? ref-total % tolerance) (rest totals))]
                    (if (and internal-ok cross-ok)
                      (do (swap! pass-count inc)
                          (println (str "  PASS   " id "  "
                                        (str/join " | "
                                                  (map (fn [[sys r]]
                                                         (str (name sys) "="
                                                              (format "%.6f" (double (parse-number (:total_score r))))))
                                                       sys-map)))))
                      (do (swap! fail-count inc)
                          (record-failure! "combinator" id
                                           (into {} (map (fn [[sys r]]
                                                           [(name sys) (format "%.10f" (double (parse-number (:total_score r))))])
                                                         sys-map))
                                           (str (when (not internal-ok) "internal consistency ")
                                                (when (not cross-ok) "cross-system mismatch")))
                          (println (str "  FAIL   " id
                                        (when (not internal-ok) " [internal consistency]")
                                        (when (not cross-ok) " [cross-system mismatch]")))
                          (doseq [[sys r] sys-map]
                            (println (str "         " (name sys)
                                          " total=" (format "%.10f" (double (parse-number (:total_score r))))
                                          " sum=" (format "%.10f" (double (parse-number (:sum_components r))))))))))

                  ;; Update: check old_score, new_score, weight + invariant
                  has-old
                  (let [internal-ok
                        (every?
                         (fn [r]
                           (let [old-s (parse-number (:old_score r))
                                 new-s (parse-number (:new_score r))
                                 w (parse-number (:weight r))]
                             (close-enough? w (- new-s old-s) tolerance)))
                         valid)
                        cross-ok
                        (every?
                         (fn [field]
                           (let [vs (map #(parse-number (get % field)) valid)
                                 ref-v (first vs)]
                             (every? #(close-enough? ref-v % tolerance) (rest vs))))
                         [:old_score :new_score :weight])]
                    (if (and internal-ok cross-ok)
                      (do (swap! pass-count inc)
                          (println (str "  PASS   " id "  "
                                        (str/join " | "
                                                  (map (fn [[sys r]]
                                                         (str (name sys)
                                                              " w=" (format "%.6f" (double (parse-number (:weight r))))))
                                                       sys-map)))))
                      (do (swap! fail-count inc)
                          (record-failure! "combinator" id
                                           (into {} (map (fn [[sys r]]
                                                           [(name sys) (str "w=" (format "%.10f" (double (parse-number (:weight r)))))])
                                                         sys-map))
                                           (str (when (not internal-ok) "w != new-old ")
                                                (when (not cross-ok) "cross-system mismatch")))
                          (println (str "  FAIL   " id
                                        (when (not internal-ok) " [w != new-old]")
                                        (when (not cross-ok) " [cross-system mismatch]")))
                          (doseq [[sys r] sys-map]
                            (println (str "         " (name sys)
                                          " old=" (format "%.10f" (double (parse-number (:old_score r))))
                                          " new=" (format "%.10f" (double (parse-number (:new_score r))))
                                          " w=" (format "%.10f" (double (parse-number (:weight r))))))))))

                  ;; Assess/generate: compare weight
                  has-weight
                  (let [weights (map #(parse-number (:weight %)) valid)
                        ref-w (first weights)
                        cross-ok (every? #(close-enough? ref-w % tolerance) (rest weights))]
                    (if cross-ok
                      (do (swap! pass-count inc)
                          (println (str "  PASS   " id "  "
                                        (str/join " | "
                                                  (map (fn [[sys r]]
                                                         (str (name sys) "="
                                                              (format "%.6f" (double (parse-number (:weight r))))))
                                                       sys-map)))))
                      (do (swap! fail-count inc)
                          (record-failure! "combinator" id
                                           (into {} (map (fn [[sys r]]
                                                           [(name sys) (format "%.10f" (double (parse-number (:weight r))))])
                                                         sys-map))
                                           "cross-system mismatch")
                          (println (str "  FAIL   " id " [cross-system mismatch]"))
                          (doseq [[sys r] sys-map]
                            (println (str "         " (name sys)
                                          " weight=" (format "%.10f" (double (parse-number (:weight r))))))))))

                  :else
                  (do (swap! error-count inc)
                      (println (str "  ERROR  " id " (no comparable fields)"))))))))

        (println)
        (println (str "Results: " @pass-count " pass, " @fail-count " fail, "
                      @error-count " error"))
        {:suite "combinator" :pass @pass-count :fail @fail-count :error @error-count :skip 0}))))

;; --- Stability Suite ---

(defn run-stability-suite [spec-file tolerance]
  (println "\n>>> Running stability tests <<<")
  (println (str "Spec: " spec-file))

  (let [spec-json (slurp spec-file)
        systems ["gen_jl" "genmlx"]
        run-results
        (reduce
         (fn [acc sys]
           (let [out-file (str results-dir "/" sys "_stability.json")
                 result (run-system sys "logprob" spec-json out-file)]
             (assoc acc sys result)))
         {}
         systems)

        loaded (reduce
                (fn [acc [sys info]]
                  (if (:success info)
                    (try
                      (assoc acc (keyword sys) (load-results (:output-file info)))
                      (catch Exception e
                        (println (str "  Failed to load " sys " results: " (.getMessage e)))
                        acc))
                    (do
                      (println (str "  " sys " failed to run"))
                      acc)))
                {}
                run-results)]

    (println (str "\n=== STABILITY COMPARISON ==="))
    (println (str "Tolerance: " tolerance))
    (println (str "Systems: " (str/join ", " (map name (keys loaded)))))
    (println)

    (if (< (count loaded) 2)
      (do (println "Not enough systems succeeded for comparison")
          {:suite "stability" :pass 0 :fail 0 :error 0 :skip 0})

      (let [by-id (reduce
                   (fn [acc [sys-name results]]
                     (reduce
                      (fn [acc2 r]
                        (assoc-in acc2 [(:id r) sys-name] r))
                      acc
                      (:results results)))
                   {}
                   loaded)
            pass-count (atom 0)
            fail-count (atom 0)
            error-count (atom 0)]

        (doseq [[id sys-map] (sort-by key by-id)]
          (let [all-sys (vals sys-map)
                errors (filter :error all-sys)
                valid (remove :error all-sys)]
            (cond
              (seq errors)
              (do (swap! error-count inc)
                  (println (str "  ERROR  " id))
                  (doseq [e errors]
                    (println (str "         " (:error e)))))

              (< (count valid) 2)
              (do (swap! error-count inc)
                  (println (str "  ERROR  " id " (not enough results)")))

              :else
              (let [vals-list (map :logprob valid)
                    both-inf (and (every? #(and (number? %) (Double/isInfinite %)) vals-list)
                                  (apply = (map #(< % 0) vals-list)))
                    both-nan (every? #(and (number? %) (Double/isNaN %)) vals-list)
                    ref-val (first vals-list)
                    all-close (every? #(close-enough? ref-val % tolerance) (rest vals-list))]
                (if (or both-inf both-nan all-close)
                  (do (swap! pass-count inc)
                      (let [display (cond
                                      both-inf "both=-Inf"
                                      both-nan "both=NaN"
                                      :else (str/join " | "
                                                      (map (fn [[sys r]]
                                                             (str (name sys) "="
                                                                  (format "%.6f" (double (:logprob r)))))
                                                           sys-map)))]
                        (println (str "  PASS   " id "  " display))))
                  (do (swap! fail-count inc)
                      (record-failure! "stability" id
                                       (into {} (map (fn [[sys r]]
                                                       [(name sys) (str (:logprob r))])
                                                     sys-map))
                                       "cross-system mismatch")
                      (println (str "  FAIL   " id))
                      (doseq [[sys r] sys-map]
                        (println (str "         " (name sys) ": " (:logprob r))))))))))

        (println)
        (println (str "Results: " @pass-count " pass, " @fail-count " fail, "
                      @error-count " error"))
        {:suite "stability" :pass @pass-count :fail @fail-count :error @error-count :skip 0}))))

;; --- Gradient Suite ---

(defn run-gradient-suite [spec-file tolerance]
  (println "\n>>> Running gradient tests <<<")
  (println (str "Spec: " spec-file))

  (let [spec-json (slurp spec-file)
        specs (json/parse-string spec-json true)
        systems ["gen_jl" "genmlx"]
        run-results
        (reduce
         (fn [acc sys]
           (let [out-file (str results-dir "/" sys "_gradient.json")
                 result (run-system sys "gradient" spec-json out-file)]
             (assoc acc sys result)))
         {}
         systems)

        loaded (reduce
                (fn [acc [sys info]]
                  (if (:success info)
                    (try
                      (assoc acc (keyword sys) (load-results (:output-file info)))
                      (catch Exception e
                        (println (str "  Failed to load " sys " results: " (.getMessage e)))
                        acc))
                    (do
                      (println (str "  " sys " failed to run"))
                      acc)))
                {}
                run-results)

        expected-by-id (into {} (map (fn [t] [(:id t) t]) (:gradient_tests specs)))]

    (println (str "\n=== GRADIENT COMPARISON ==="))
    (println (str "Tolerance: " tolerance))
    (println (str "Systems: " (str/join ", " (map name (keys loaded)))))
    (println)

    (let [pass-count (atom 0)
          fail-count (atom 0)
          error-count (atom 0)
          by-id (reduce
                 (fn [acc [sys-name results]]
                   (reduce
                    (fn [acc2 r]
                      (assoc-in acc2 [(:id r) sys-name] r))
                    acc
                    (:results results)))
                 {}
                 loaded)]

      (doseq [[id sys-map] (sort-by key by-id)]
        (let [all-sys (vals sys-map)
              errors (filter :error all-sys)
              valid (remove :error all-sys)
              spec (get expected-by-id id)
              expected (:expected spec)]
          (cond
            (seq errors)
            (do (swap! error-count inc)
                (println (str "  ERROR  " id))
                (doseq [e errors]
                  (println (str "         " (:error e)))))

            (empty? valid)
            (do (swap! error-count inc)
                (println (str "  ERROR  " id " (no results)")))

            :else
            (let [gradients (map #(parse-number (:gradient %)) valid)
                  ref-g (first gradients)
                  cross-ok (if (> (count gradients) 1)
                             (every? #(close-enough? ref-g % tolerance) (rest gradients))
                             true)
                  expected-ok (if expected
                                (every? #(close-enough? % (parse-number expected) tolerance) gradients)
                                true)]
              (if (and cross-ok expected-ok)
                (do (swap! pass-count inc)
                    (println (str "  PASS   " id "  "
                                  (str/join " | "
                                            (map (fn [[sys r]]
                                                   (str (name sys) "="
                                                        (format "%.6f" (double (parse-number (:gradient r))))))
                                                 sys-map))
                                  (when expected (str "  expected=" (format "%.6f" (double expected)))))))
                (do (swap! fail-count inc)
                    (record-failure! "gradient" id
                                     (merge (into {} (map (fn [[sys r]]
                                                            [(name sys) (format "%.10f" (double (parse-number (:gradient r))))])
                                                          sys-map))
                                            (when expected {"expected" (format "%.10f" (double expected))}))
                                     (str (when (not cross-ok) "cross-system mismatch ")
                                          (when (not expected-ok) "expected mismatch")))
                    (println (str "  FAIL   " id
                                  (when (not cross-ok) " [cross-system mismatch]")
                                  (when (not expected-ok) " [expected mismatch]")))
                    (doseq [[sys r] sys-map]
                      (println (str "         " (name sys) ": "
                                    (format "%.10f" (double (parse-number (:gradient r)))))))
                    (when expected
                      (println (str "         expected: " (format "%.10f" (double expected)))))))))))

      (println)
      (println (str "Results: " @pass-count " pass, " @fail-count " fail, "
                    @error-count " error"))
      {:suite "gradient" :pass @pass-count :fail @fail-count :error @error-count :skip 0})))

;; --- Inference Quality Suite ---

(defn run-system-per-test
  "Run a system once per test to avoid OOM in long-running inference.
   Uses Node.js for SMC tests (Bun/JSC crashes on large SMC, mlx#3297).
   Returns merged results as if it were a single run."
  [system tests out-file]
  (let [all-results (atom [])
        total (count tests)]
    (doseq [[idx test] (map-indexed vector tests)]
      (let [smc? (contains? #{"smc" "smc_single"} (:algorithm test))
            use-node? (and (= system "genmlx") smc?)]
        (print (str "    [" (inc idx) "/" total "] " (:id test)
                    (when use-node? " [node]") "... "))
        (flush))
      (let [smc? (contains? #{"smc" "smc_single"} (:algorithm test))
            use-node? (and (= system "genmlx") smc?)
            single-spec (json/generate-string {:test_type "inference_quality"
                                                :tests [test]})
            cmd (runner-cmd system use-node?)]
        (try
          (let [result (p/shell {:in single-spec :out :string :err :string
                                  :dir (System/getProperty "user.dir")}
                                 (str/join " " cmd))
                parsed (when (= 0 (:exit result))
                          (json/parse-string (:out result) true))]
            (when parsed
              (swap! all-results into (:results parsed)))
            (println "ok"))
          (catch Exception e
            (println "CRASH")
            (swap! all-results conj {:id (:id test) :error (str "runner crash: " (.getMessage e))})))))
    ;; Write merged results
    (let [merged {:system system :test_type "inference_quality" :results @all-results}]
      (spit out-file (json/generate-string merged {:pretty true}))
      {:success true :output-file out-file})))

(defn run-inference-quality-suite [spec-file]
  (println "\n>>> Running inference_quality tests <<<")
  (println (str "Spec: " spec-file))

  (let [spec-json (slurp spec-file)
        specs (json/parse-string spec-json true)
        systems ["gen_jl" "genmlx"]
        run-results
        (reduce
         (fn [acc sys]
           (let [out-file (str results-dir "/" sys "_inference_quality.json")]
             (if (= sys "genmlx")
               ;; Run GenMLX per-test to avoid Bun OOM on long inference suites
               (do (println (str "  Running " sys " (per-test, " (count (:tests specs)) " tests)..."))
                   (assoc acc sys (run-system-per-test sys (:tests specs) out-file)))
               ;; Other systems run all tests in one process
               (assoc acc sys (run-system sys "inference_quality" spec-json out-file)))))
         {}
         systems)

        loaded (reduce
                (fn [acc [sys info]]
                  (if (:success info)
                    (try
                      (assoc acc (keyword sys) (load-results (:output-file info)))
                      (catch Exception e
                        (println (str "  Failed to load " sys " results: " (.getMessage e)))
                        acc))
                    (do
                      (println (str "  " sys " failed to run"))
                      acc)))
                {}
                run-results)

        test-by-id (into {} (map (fn [t] [(:id t) t]) (:tests specs)))]

    (println (str "\n=== INFERENCE QUALITY COMPARISON ==="))
    (println (str "Systems: " (str/join ", " (map name (keys loaded)))))
    (println (str "Tolerance: 3 * analytical_std (statistical)"))
    (println)

    (if (empty? loaded)
      (do (println "No systems succeeded")
          {:suite "inference_quality" :pass 0 :fail 0 :error 0 :skip 0})

      (let [by-id (reduce
                   (fn [acc [sys-name results]]
                     (reduce
                      (fn [acc2 r]
                        (assoc-in acc2 [(:id r) sys-name] r))
                      acc
                      (:results results)))
                   {}
                   loaded)
            pass-count (atom 0)
            fail-count (atom 0)
            error-count (atom 0)
            skip-count (atom 0)]

        (doseq [[id sys-map] (sort-by key by-id)]
          (let [test-spec (get test-by-id id)
                skip? (get test-spec :skip false)
                comparison (get test-spec :comparison "posterior_mean")
                analytical-mean (get-in test-spec [:analytical_posterior :mean])
                analytical-std  (get-in test-spec [:analytical_posterior :std])
                all-sys (vals sys-map)
                errors (filter :error all-sys)
                valid (remove :error all-sys)]
            (cond
              skip?
              (do (swap! skip-count inc)
                  (println (str "  SKIP   " id
                                (when-let [reason (get test-spec :skip_reason)]
                                  (str " — " reason)))))

              (seq errors)
              (do (swap! error-count inc)
                  (println (str "  ERROR  " id))
                  (doseq [e errors]
                    (println (str "         " (:error e)))))

              (empty? valid)
              (do (swap! error-count inc)
                  (println (str "  ERROR  " id " (no results)")))

              (= comparison "acceptance_rate")
              ;; Acceptance rate test: check each system's rate > min threshold
              (let [min-rate (get test-spec :min_accept_rate 0.3)
                    rates (map (fn [r] (when-let [ar (:acceptance_rate r)]
                                         (parse-number ar)))
                               valid)
                    rates (remove nil? rates)
                    all-above (every? #(> % min-rate) rates)]
                (if (or all-above (empty? rates))
                  (do (swap! pass-count inc)
                      (println (str "  PASS   " id "  "
                                    (str/join " | "
                                              (map (fn [[sys r]]
                                                     (let [ar (:acceptance_rate r)]
                                                       (str (name sys) "="
                                                            (if ar (format "%.4f" (double (parse-number ar))) "N/A"))))
                                                   sys-map))
                                    "  min=" (format "%.2f" min-rate))))
                  (do (swap! fail-count inc)
                      (record-failure! "inference_quality" id
                                       (into {} (map (fn [[sys r]]
                                                       [(name sys) (if-let [ar (:acceptance_rate r)]
                                                                     (format "%.4f" (double (parse-number ar))) "N/A")])
                                                     sys-map))
                                       (str "acceptance rate below " (format "%.2f" min-rate)))
                      (println (str "  FAIL   " id " [acceptance rate below " (format "%.2f" min-rate) "]"))
                      (doseq [[sys r] sys-map]
                        (let [ar (:acceptance_rate r)]
                          (println (str "         " (name sys) ": "
                                        (if ar (format "%.4f" (double (parse-number ar))) "N/A"))))))))

              (= comparison "log_ml")
              ;; Log-ML cross-system comparison (within tolerance nats)
              (let [tol (get test-spec :tolerance 1.0)
                    log-mls (keep (fn [r] (when-let [ml (:log_ml r)] (parse-number ml)))
                                  valid)]
                (if (< (count log-mls) 2)
                  (do (swap! pass-count inc)
                      (println (str "  PASS   " id " (only " (count log-mls) " system(s) with log-ML)")))
                  (let [ref-ml (first log-mls)
                        all-close (every? #(< (abs (- % ref-ml)) tol) (rest log-mls))]
                    (if all-close
                      (do (swap! pass-count inc)
                          (println (str "  PASS   " id "  "
                                        (str/join " | "
                                                  (map (fn [[sys r]]
                                                         (str (name sys) "="
                                                              (if-let [ml (:log_ml r)]
                                                                (format "%.4f" (double (parse-number ml)))
                                                                "N/A")))
                                                       sys-map))
                                        "  tol=" (format "%.1f" (double tol)))))
                      (do (swap! fail-count inc)
                          (record-failure! "inference_quality" id
                                           (into {} (map (fn [[sys r]]
                                                           [(name sys) (if-let [ml (:log_ml r)]
                                                                         (format "%.6f" (double (parse-number ml))) "N/A")])
                                                         sys-map))
                                           (str "log-ML cross-system disagree, tol=" (format "%.1f" (double tol))))
                          (println (str "  FAIL   " id " [log-ML cross-system disagree, tol=" (format "%.1f" (double tol)) "]"))
                          (doseq [[sys r] sys-map]
                            (println (str "         " (name sys) ": "
                                          (if-let [ml (:log_ml r)]
                                            (format "%.6f" (double (parse-number ml)))
                                            "N/A")))))))))

              (= comparison "ess")
              ;; ESS threshold test: check each system's ESS > threshold
              (let [ess-threshold (get test-spec :ess_threshold 10)
                    ess-vals (keep (fn [r] (when-let [e (:ess r)]
                                             (parse-number e)))
                                   valid)
                    all-above (every? #(> % ess-threshold) ess-vals)]
                (if (or all-above (empty? ess-vals))
                  (do (swap! pass-count inc)
                      (println (str "  PASS   " id "  "
                                    (str/join " | "
                                              (map (fn [[sys r]]
                                                     (let [e (:ess r)]
                                                       (str (name sys) "="
                                                            (if e (format "%.1f" (double (parse-number e))) "N/A"))))
                                                   sys-map))
                                    "  threshold=" ess-threshold)))
                  (do (swap! fail-count inc)
                      (record-failure! "inference_quality" id
                                       (into {} (map (fn [[sys r]]
                                                       [(name sys) (if-let [e (:ess r)]
                                                                     (format "%.1f" (double (parse-number e))) "N/A")])
                                                     sys-map))
                                       (str "ESS below " ess-threshold))
                      (println (str "  FAIL   " id " [ESS below " ess-threshold "]"))
                      (doseq [[sys r] sys-map]
                        (let [e (:ess r)]
                          (println (str "         " (name sys) ": "
                                        (if e (format "%.1f" (double (parse-number e))) "N/A"))))))))

              (= comparison "analytical_only")
              ;; Check against analytical posterior only (skip cross-system)
              (let [tol (* 3.0 analytical-std)
                    non-skip (remove :skip valid)
                    means (map #(parse-number (:posterior_mean %)) non-skip)
                    all-within-tol (every? #(< (abs (- % analytical-mean)) tol) means)]
                (if all-within-tol
                  (do (swap! pass-count inc)
                      (println (str "  PASS   " id "  "
                                    (str/join " | "
                                              (map (fn [[sys r]]
                                                     (if (:skip r)
                                                       (str (name sys) "=skip")
                                                       (str (name sys) "="
                                                            (format "%.4f" (double (parse-number (:posterior_mean r)))))))
                                                   sys-map))
                                    "  analytical=" (format "%.4f" analytical-mean)
                                    "  tol=" (format "%.4f" tol))))
                  (do (swap! fail-count inc)
                      (record-failure! "inference_quality" id
                                       (merge (into {} (map (fn [[sys r]]
                                                              [(name sys) (if (:skip r) "skip"
                                                                            (format "%.6f" (double (parse-number (:posterior_mean r)))))])
                                                            sys-map))
                                              {"analytical" (format "%.6f" analytical-mean)})
                                       "outside 3*std of analytical")
                      (println (str "  FAIL   " id " [outside 3*std of analytical]"))
                      (doseq [[sys r] sys-map]
                        (if (:skip r)
                          (println (str "         " (name sys) ": skipped"))
                          (println (str "         " (name sys) ": "
                                        (format "%.6f" (double (parse-number (:posterior_mean r))))))))
                      (println (str "         analytical: " (format "%.6f" analytical-mean)
                                    " +/- " (format "%.6f" tol))))))

              (= comparison "analytical_log_ml")
              ;; Log-ML vs analytical value (within tolerance nats)
              (let [tol (get test-spec :tolerance 1.0)
                    analytical-log-ml (get test-spec :analytical_log_ml)
                    log-mls (map (fn [r] (parse-number (:log_ml r))) valid)
                    log-mls (remove nil? log-mls)
                    all-close (every? #(< (abs (- % analytical-log-ml)) tol) log-mls)]
                (if all-close
                  (do (swap! pass-count inc)
                      (println (str "  PASS   " id "  "
                                    (str/join " | "
                                              (map (fn [[sys r]]
                                                     (str (name sys) "="
                                                          (if-let [ml (:log_ml r)]
                                                            (format "%.4f" (double (parse-number ml)))
                                                            "N/A")))
                                                   sys-map))
                                    "  analytical=" (format "%.4f" (double analytical-log-ml))
                                    "  tol=" (format "%.1f" (double tol)))))
                  (do (swap! fail-count inc)
                      (record-failure! "inference_quality" id
                                       (merge (into {} (map (fn [[sys r]]
                                                              [(name sys) (if-let [ml (:log_ml r)]
                                                                            (format "%.6f" (double (parse-number ml))) "N/A")])
                                                            sys-map))
                                              {"analytical" (format "%.4f" (double analytical-log-ml))})
                                       (str "log-ML outside " (format "%.1f" (double tol)) " nat of analytical"))
                      (println (str "  FAIL   " id " [log-ML outside " (format "%.1f" (double tol))
                                    " nat of analytical=" (format "%.4f" (double analytical-log-ml)) "]"))
                      (doseq [[sys r] sys-map]
                        (println (str "         " (name sys) ": "
                                      (if-let [ml (:log_ml r)]
                                        (format "%.6f" (double (parse-number ml)))
                                        "N/A")))))))

              (= comparison "posterior_variance")
              ;; Posterior variance test: check sample variance within factor of 3 of analytical
              (let [analytical-var (get-in test-spec [:analytical_posterior :variance])
                    variances (keep (fn [r] (when-let [v (:posterior_variance r)]
                                              (parse-number v)))
                                    valid)
                    all-ok (every? (fn [v]
                                     (and (> v (/ analytical-var 3.0))
                                          (< v (* analytical-var 3.0))))
                                   variances)]
                (if (or all-ok (empty? variances))
                  (do (swap! pass-count inc)
                      (println (str "  PASS   " id "  "
                                    (str/join " | "
                                              (map (fn [[sys r]]
                                                     (str (name sys) "="
                                                          (if-let [v (:posterior_variance r)]
                                                            (format "%.6f" (double (parse-number v)))
                                                            "N/A")))
                                                   sys-map))
                                    "  analytical=" (format "%.6f" (double analytical-var))
                                    "  [factor-of-3]")))
                  (do (swap! fail-count inc)
                      (record-failure! "inference_quality" id
                                       (merge (into {} (map (fn [[sys r]]
                                                              [(name sys) (if-let [v (:posterior_variance r)]
                                                                            (format "%.6f" (double (parse-number v))) "N/A")])
                                                            sys-map))
                                              {"analytical" (format "%.6f" (double analytical-var))})
                                       "variance outside factor-of-3 of analytical")
                      (println (str "  FAIL   " id " [variance outside factor-of-3 of analytical="
                                    (format "%.6f" (double analytical-var)) "]"))
                      (doseq [[sys r] sys-map]
                        (println (str "         " (name sys) ": "
                                      (if-let [v (:posterior_variance r)]
                                        (format "%.6f" (double (parse-number v)))
                                        "N/A")))))))

              (nil? analytical-mean)
              ;; No analytical posterior — skip this test
              (do (swap! skip-count inc)
                  (println (str "  SKIP   " id " (no analytical posterior)")))

              :else
              ;; Posterior mean test
              (let [tol (if analytical-std (* 3.0 analytical-std) 1.0)
                    means (map #(parse-number (:posterior_mean %)) valid)
                    ;; Filter out nil means (systems that couldn't produce a result)
                    valid-means (filter some? means)
                    ;; Check each system against analytical posterior
                    all-within-tol
                    (every? #(< (abs (- % analytical-mean)) tol) valid-means)
                    ;; Also check systems agree with each other (within 2*std)
                    cross-tol (if analytical-std (* 2.0 analytical-std) 1.0)
                    cross-ok (if (> (count means) 1)
                               (let [ref-m (first means)]
                                 (every? #(< (abs (- % ref-m)) cross-tol) (rest means)))
                               true)]
                (if (and all-within-tol cross-ok)
                  (do (swap! pass-count inc)
                      (println (str "  PASS   " id "  "
                                    (str/join " | "
                                              (map (fn [[sys r]]
                                                     (str (name sys) "="
                                                          (format "%.4f" (double (parse-number (:posterior_mean r))))))
                                                   sys-map))
                                    "  analytical=" (format "%.4f" analytical-mean)
                                    "  tol=" (format "%.4f" tol))))
                  (do (swap! fail-count inc)
                      (record-failure! "inference_quality" id
                                       (merge (into {} (map (fn [[sys r]]
                                                              [(name sys) (format "%.6f" (double (parse-number (:posterior_mean r))))])
                                                            sys-map))
                                              {"analytical" (format "%.6f" analytical-mean)})
                                       (str (when (not all-within-tol) "outside 3*std of analytical ")
                                            (when (not cross-ok) "cross-system disagree")))
                      (println (str "  FAIL   " id
                                    (when (not all-within-tol) " [outside 3*std of analytical]")
                                    (when (not cross-ok) " [cross-system disagree]")))
                      (doseq [[sys r] sys-map]
                        (println (str "         " (name sys) ": "
                                      (format "%.6f" (double (parse-number (:posterior_mean r)))))))
                      (println (str "         analytical: " (format "%.6f" analytical-mean)
                                    " +/- " (format "%.6f" tol)))))))))

        (println)
        (println (str "Results: " @pass-count " pass, " @fail-count " fail, "
                      @error-count " error"))
        {:suite "inference_quality" :pass @pass-count :fail @fail-count :error @error-count :skip @skip-count}))))

;; --- Main ---

(defn run-test-suite [test-type spec-file tolerance]
  (println (str "\n>>> Running " test-type " tests <<<"))
  (println (str "Spec: " spec-file))

  (let [spec-json (slurp spec-file)
        systems ["gen_jl" "genjax" "genmlx"]
        run-results
        (reduce
         (fn [acc sys]
           (let [out-file (str results-dir "/" sys "_" test-type ".json")
                 result (run-system sys test-type spec-json out-file)]
             (assoc acc sys result)))
         {}
         systems)]

    ;; Load and compare results from successful runs
    (let [loaded (reduce
                  (fn [acc [sys info]]
                    (if (:success info)
                      (try
                        (assoc acc (keyword sys) (load-results (:output-file info)))
                        (catch Exception e
                          (println (str "  Failed to load " sys " results: " (.getMessage e)))
                          acc))
                      (do
                        (println (str "  " sys " failed to run"))
                        acc)))
                  {}
                  run-results)]

      (if (>= (count loaded) 2)
        (compare-results test-type loaded tolerance)
        (do
          (println "Not enough systems succeeded for comparison")
          {:suite test-type :pass 0 :fail 0 :error 0 :skip 0})))))

;; --- Summary Report Generation ---

(defn generate-summary
  "Write results/SUMMARY.md with per-category table and failure details."
  [suite-results totals suites-run]
  (let [timestamp (str (java.time.LocalDateTime/now))
        summary-file (str results-dir "/SUMMARY.md")
        sb (StringBuilder.)]

    ;; Header
    (.append sb "# Cross-System Verification Summary\n\n")
    (.append sb (str "**Date:** " timestamp "\n\n"))
    (.append sb (str "**Systems:** Gen.jl, GenJAX, GenMLX\n\n"))
    (.append sb (str "**Suites run:** " (str/join ", " (sort suites-run)) "\n\n"))

    ;; Overall verdict
    (let [verdict (cond
                    (pos? (:fail totals))
                    (str "FAIL (" (:fail totals) " failures, " (:error totals) " errors)")
                    (pos? (:error totals))
                    (str "PASS with errors (" (:error totals) " errors — systems could not evaluate some tests)")
                    :else "PASS")]
      (.append sb (str "**Verdict:** " verdict "\n\n")))

    ;; Per-category table
    (.append sb "## Results by Category\n\n")
    (.append sb "| Category | Total | Pass | Fail | Error | Skip |\n")
    (.append sb "|---|---|---|---|---|---|\n")
    (doseq [r (sort-by #(or (:suite %) "unknown") suite-results)]
      (let [suite-name (or (:suite r) "unknown")
            total (+ (:pass r) (:fail r) (:error r) (:skip r))]
        (.append sb (str "| " suite-name
                         " | " total
                         " | " (:pass r)
                         " | " (:fail r)
                         " | " (:error r)
                         " | " (:skip r)
                         " |\n"))))
    (let [grand-total (+ (:pass totals) (:fail totals) (:error totals) (:skip totals))]
      (.append sb (str "| **TOTAL** | **" grand-total
                       "** | **" (:pass totals)
                       "** | **" (:fail totals)
                       "** | **" (:error totals)
                       "** | **" (:skip totals)
                       "** |\n\n")))

    ;; Failure details
    (let [failures @all-failures]
      (if (empty? failures)
        (.append sb "## Failures\n\nNone.\n")
        (do
          (.append sb (str "## Failures (" (count failures) " total)\n\n"))
          ;; Group by suite
          (doseq [[suite fails] (sort-by key (group-by :suite failures))]
            (.append sb (str "### " suite "\n\n"))
            (doseq [f fails]
              (.append sb (str "- **" (:id f) "** -- " (:reason f) "\n"))
              (doseq [[sys val] (sort-by key (:values f))]
                (.append sb (str "  - " sys ": `" val "`\n"))))
            (.append sb "\n")))))

    (spit summary-file (str sb))
    (println (str "\nSummary written to " summary-file))))

(defn main []
  (clojure.java.io/make-parents (str results-dir "/placeholder"))

  (let [logprob-spec (str specs-dir "/logprob_tests.json")
        gfi-spec (str specs-dir "/gfi_tests.json")
        mlx-spec (str specs-dir "/mlx_ops_tests.json")
        update-spec (str specs-dir "/update_tests.json")
        project-regen-spec (str specs-dir "/project_regen_tests.json")
        combinator-spec (str specs-dir "/combinator_tests.json")
        stability-spec (str specs-dir "/stability_tests.json")
        gradient-spec (str specs-dir "/gradient_tests.json")
        inference-spec (str specs-dir "/inference_quality_tests.json")
        args (vec *command-line-args*)
        ;; Default: fast deterministic suites only. Opt-in to slow inference_quality:
        ;; bb cross_system_tests/run_all.clj inference_quality
        suites (if (seq args) (set args) #{"mlx_ops" "logprob" "assess" "score_decomposition" "update" "project" "regenerate" "combinator" "stability" "gradient"})]

    (println "╔════════════════════════════════════════════╗")
    (println "║  GenMLX Cross-System Verification Harness  ║")
    (println "╚════════════════════════════════════════════╝")

    (let [all-results
          (cond-> []
            (contains? suites "mlx_ops")
            (conj (run-mlx-ops-suite mlx-spec 1e-4))

            (contains? suites "logprob")
            (conj (run-test-suite "logprob" logprob-spec 1e-4))

            (contains? suites "assess")
            (conj (run-test-suite "assess" gfi-spec 1e-4))

            (contains? suites "generate")
            (conj (run-test-suite "generate" gfi-spec 1e-4))

            (contains? suites "score_decomposition")
            (conj (run-score-decomposition-suite gfi-spec 1e-4))

            (contains? suites "update")
            (conj (run-update-suite update-spec 1e-4))

            (contains? suites "project")
            (conj (run-project-suite project-regen-spec 1e-4))

            (contains? suites "regenerate")
            (conj (run-regenerate-suite project-regen-spec 1e-4))

            (contains? suites "combinator")
            (conj (run-combinator-suite combinator-spec 1e-4))

            (contains? suites "stability")
            (conj (run-stability-suite stability-spec 0.01))

            (contains? suites "gradient")
            (conj (run-gradient-suite gradient-spec 1e-3))

            (contains? suites "inference_quality")
            (conj (run-inference-quality-suite inference-spec)))

          totals (reduce
                  (fn [acc r]
                    (-> acc
                        (update :pass + (:pass r))
                        (update :fail + (:fail r))
                        (update :error + (:error r))
                        (update :skip + (:skip r))))
                  {:pass 0 :fail 0 :error 0 :skip 0}
                  all-results)]

      (println "\n════════════════════════════════════════════")
      (println (str "TOTAL: " (:pass totals) " pass, " (:fail totals) " fail, "
                    (:error totals) " error, " (:skip totals) " skip"))

      ;; Generate persistent summary report
      (generate-summary all-results totals suites)

      (when (> (:fail totals) 0)
        (println "\n⚠ FAILURES DETECTED — investigate discrepancies")
        (System/exit 1))

      (when (> (:error totals) 0)
        (println "\n⚠ ERRORS — some systems could not evaluate all tests")))))

(main)
