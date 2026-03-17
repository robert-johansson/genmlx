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

;; --- Runner commands ---
;; Each runner reads {test_type, ...specs...} from stdin, writes JSON to stdout.

(defn runner-cmd [system]
  (case system
    "gen_jl" ["julia" (str runners-dir "/gen_jl_runner.jl")]
    "genjax" [venv-python (str runners-dir "/genjax_runner.py")]
    "genmlx" ["bun" "run" "--bun" "nbb" (str runners-dir "/genmlx_runner.cljs")]))

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
                    (assoc acc k (if (#{:logprob :weight :score :old_score :new_score :total_score :sum_components} k)
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
                  (println (str "  FAIL   " id))
                  (doseq [vv valid-values]
                    (println (str "         " (name (:system vv)) ": "
                                  (format "%.10f" (double (:value vv))))))))))))

      (println)
      (println (str "Results: " @pass-count " pass, " @fail-count " fail, "
                    @error-count " error, " @skip-count " skip"))
      {:pass @pass-count :fail @fail-count :error @error-count :skip @skip-count})))

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
          {:pass 0 :fail 0 :error 0 :skip 0})

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
                  (println (str "  FAIL   " (:id r)))
                  (println (str "         mlx:      " (pr-str (:result r))))
                  (println (str "         expected:  " (pr-str expected)))))))

        (println)
        (println (str "Results: " @pass-count " pass, " @fail-count " fail, "
                      @error-count " error"))
        {:pass @pass-count :fail @fail-count :error @error-count :skip 0}))))

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
          {:pass 0 :fail 0 :error 0 :skip 0})

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
        {:pass @pass-count :fail @fail-count :error @error-count :skip 0}))))

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
          {:pass 0 :fail 0 :error 0 :skip 0})

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
        {:pass @pass-count :fail @fail-count :error @error-count :skip 0}))))

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
          {:pass 0 :fail 0 :error 0 :skip 0})

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
                      (println (str "  FAIL   " id " [cross-system mismatch]"))
                      (doseq [[sys r] sys-map]
                        (println (str "         " (name sys)
                                      " weight=" (format "%.10f" (double (parse-number (:weight r)))))))))))))

        (println)
        (println (str "Results: " @pass-count " pass, " @fail-count " fail, "
                      @error-count " error"))
        {:pass @pass-count :fail @fail-count :error @error-count :skip 0}))))

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
          {:pass 0 :fail 0 :error 0 :skip 0})

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
        {:pass @pass-count :fail @fail-count :error @error-count :skip 0}))))

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
          {:pass 0 :fail 0 :error 0 :skip 0})

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
        {:pass @pass-count :fail @fail-count :error @error-count :skip 0}))))

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
          {:pass 0 :fail 0 :error 0 :skip 0})))))

(defn main []
  (clojure.java.io/make-parents (str results-dir "/placeholder"))

  (let [logprob-spec (str specs-dir "/logprob_tests.json")
        gfi-spec (str specs-dir "/gfi_tests.json")
        mlx-spec (str specs-dir "/mlx_ops_tests.json")
        update-spec (str specs-dir "/update_tests.json")
        project-regen-spec (str specs-dir "/project_regen_tests.json")
        combinator-spec (str specs-dir "/combinator_tests.json")
        args (vec *command-line-args*)
        suites (if (seq args) (set args) #{"mlx_ops" "logprob" "assess" "score_decomposition" "update" "project" "regenerate" "combinator"})]

    (println "╔════════════════════════════════════════════╗")
    (println "║  GenMLX Cross-System Verification Harness  ║")
    (println "╚════════════════════════════════════════════╝")

    (let [all-results
          (cond-> []
            (contains? suites "mlx_ops")
            (conj (run-mlx-ops-suite mlx-spec 1e-4))

            (contains? suites "logprob")
            (conj (run-test-suite "logprob" logprob-spec 1e-5))

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
            (conj (run-combinator-suite combinator-spec 1e-4)))

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

      (when (> (:fail totals) 0)
        (println "\n⚠ FAILURES DETECTED — investigate discrepancies")
        (System/exit 1))

      (when (> (:error totals) 0)
        (println "\n⚠ ERRORS — some systems could not evaluate all tests")))))

(main)
