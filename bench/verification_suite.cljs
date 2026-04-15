(ns bench.verification-suite
  "Verification Suite — runs all core test suites and reports pass rates.

   Executes each test suite as a child process, parses cljs.test output
   for assertion/failure/error counts, and writes a summary to data.json.

   Output: $GENMLX_RESULTS_DIR/data.json or results/verification-suite/data.json

   Usage: bun run --bun nbb bench/verification_suite.cljs")

;; ---------------------------------------------------------------------------
;; Node.js interop
;; ---------------------------------------------------------------------------

(def fs (js/require "fs"))
(def path-mod (js/require "path"))
(def child-process (js/require "child_process"))

(def bun-path "/Users/robert/.bun/bin/bun")

(def out-dir
  (or (aget (.-env js/process) "GENMLX_RESULTS_DIR")
      (.resolve path-mod (js/process.cwd) "results/verification-suite")))

(defn ensure-dir [dir]
  (when-not (.existsSync fs dir)
    (.mkdirSync fs dir #js {:recursive true})))

(defn write-json [filename data]
  (ensure-dir out-dir)
  (let [filepath (str out-dir "/" filename)]
    (.writeFileSync fs filepath (js/JSON.stringify (clj->js data) nil 2))
    (println (str "  wrote: " filepath))))

(defn perf-now [] (js/performance.now))

;; ---------------------------------------------------------------------------
;; Test suite definitions
;; ---------------------------------------------------------------------------

(def suites
  [{:name "L0 certification"
    :file "test/genmlx/level0_certification_test.cljs"
    :expected 68}
   {:name "L1-M1 schema"
    :file "test/genmlx/schema_test.cljs"
    :expected 174}
   {:name "L1-M2 compiled simulate"
    :file "test/genmlx/compiled_simulate_test.cljs"
    :expected 82}
   {:name "L1-M3 partial compile"
    :file "test/genmlx/partial_compile_test.cljs"
    :expected 92}
   {:name "L1-M5 combinator compile"
    :file "test/genmlx/combinator_compile_test.cljs"
    :expected 90}
   {:name "L4 certification"
    :file "test/genmlx/l4_certification_test.cljs"
    :expected 41}
   {:name "Gen.clj compat"
    :file "test/genmlx/gen_clj_compat_test.cljs"
    :expected 165}
   {:name "GenJAX compat"
    :file "test/genmlx/genjax_compat_test.cljs"
    :expected 73}])

;; ---------------------------------------------------------------------------
;; Output parsing
;; ---------------------------------------------------------------------------

(defn parse-cljs-test-output
  "Parse cljs.test run-tests output.

   Looks for the summary line:
     'Ran N tests containing M assertions.'
     'X failures, Y errors.'

   Also counts individual FAIL lines as a cross-check."
  [stdout]
  (let [lines (.split stdout "\n")
        ;; Parse the summary line: 'X failures, Y errors.'
        summary-re #"(\d+)\s+failures?,\s+(\d+)\s+errors?"
        assertion-re #"Ran\s+(\d+)\s+tests?\s+containing\s+(\d+)\s+assertions?"
        fail-line-count (count (filter #(re-find #"^FAIL\s" %) (array-seq lines)))
        summary-match (some #(re-find summary-re %) (array-seq lines))
        assertion-match (some #(re-find assertion-re %) (array-seq lines))]
    (if (and summary-match assertion-match)
      (let [total-assertions (js/parseInt (nth assertion-match 2) 10)
            failures (js/parseInt (nth summary-match 1) 10)
            errors (js/parseInt (nth summary-match 2) 10)
            pass (- total-assertions failures errors)]
        {:pass pass
         :fail failures
         :error errors
         :total-assertions total-assertions
         :fail-lines fail-line-count})
      ;; Fallback: count PASS/FAIL lines if cljs.test summary not found
      (let [pass-count (count (filter #(re-find #"PASS" %) (array-seq lines)))
            fail-count fail-line-count]
        {:pass pass-count
         :fail fail-count
         :error 0
         :total-assertions (+ pass-count fail-count)
         :fail-lines fail-count
         :parse-fallback true}))))

;; ---------------------------------------------------------------------------
;; Suite runner
;; ---------------------------------------------------------------------------

(defn run-suite
  "Run a single test suite as a child process. Returns result map."
  [{:keys [name file expected]}]
  (println (str "\n  running: " name " (" file ")"))
  (let [t0 (perf-now)]
    (try
      (let [cmd (str bun-path " run --bun nbb " file)
            stdout (.toString
                     (.execSync child-process cmd
                                #js {:timeout 120000
                                     :maxBuffer (* 10 1024 1024)
                                     :encoding "utf8"
                                     :stdio #js ["pipe" "pipe" "pipe"]}))
            duration-ms (- (perf-now) t0)
            parsed (parse-cljs-test-output stdout)
            pass-rate (if (pos? (:total-assertions parsed))
                        (/ (:pass parsed) (:total-assertions parsed))
                        0)]
        (println (str "    " (:pass parsed) "/" (:total-assertions parsed)
                      " assertions passed"
                      (when (pos? (:fail parsed))
                        (str ", " (:fail parsed) " failures"))
                      (when (pos? (:error parsed))
                        (str ", " (:error parsed) " errors"))
                      " (" (.toFixed duration-ms 0) "ms)"
                      (if (= (:total-assertions parsed) expected)
                        " [OK]"
                        (str " [EXPECTED " expected "]"))))
        {:name name
         :file file
         :pass (:pass parsed)
         :fail (:fail parsed)
         :error (:error parsed)
         :total-assertions (:total-assertions parsed)
         :expected expected
         :duration-ms (js/Math.round duration-ms)
         :pass-rate pass-rate
         :status (if (and (zero? (:fail parsed))
                          (zero? (:error parsed))
                          (= (:total-assertions parsed) expected))
                   "pass"
                   "degraded")})
      (catch :default e
        (let [duration-ms (- (perf-now) t0)
              msg (or (.-message e) (str e))
              ;; execSync throws on non-zero exit; stderr/stdout may be on the error
              stdout (or (some-> (.-stdout e) .toString) "")
              parsed (parse-cljs-test-output stdout)]
          (println (str "    ERROR: " (subs msg 0 (min (count msg) 200))
                        " (" (.toFixed duration-ms 0) "ms)"))
          (when (pos? (:total-assertions parsed))
            (println (str "    partial: " (:pass parsed) "/" (:total-assertions parsed)
                          " passed, " (:fail parsed) " failures, " (:error parsed) " errors")))
          {:name name
           :file file
           :pass (:pass parsed)
           :fail (:fail parsed)
           :error (max 1 (:error parsed))
           :total-assertions (:total-assertions parsed)
           :expected expected
           :duration-ms (js/Math.round duration-ms)
           :pass-rate 0
           :status "error"
           :error-message (subs msg 0 (min (count msg) 500))})))))

;; ---------------------------------------------------------------------------
;; Main
;; ---------------------------------------------------------------------------

(println "\n=== GenMLX Verification Suite ===")
(println (str "  output: " out-dir))

(let [t0 (perf-now)
      results (mapv run-suite suites)
      total-duration-ms (- (perf-now) t0)
      totals {:pass (reduce + (map :pass results))
              :fail (reduce + (map :fail results))
              :error (reduce + (map :error results))
              :total-assertions (reduce + (map :total-assertions results))
              :expected (reduce + (map :expected results))
              :duration-ms (js/Math.round total-duration-ms)
              :suites-passed (count (filter #(= "pass" (:status %)) results))
              :suites-total (count results)}
      data {:experiment "verification-suite"
            :suites results
            :totals totals}]

  (println "\n=== Summary ===")
  (println (str "  suites: " (:suites-passed totals) "/" (:suites-total totals) " passed"))
  (println (str "  assertions: " (:pass totals) "/" (:total-assertions totals)
                " passed, " (:fail totals) " failures, " (:error totals) " errors"))
  (println (str "  expected total: " (:expected totals)))
  (println (str "  duration: " (.toFixed (/ total-duration-ms 1000) 1) "s"))

  (write-json "data.json" data)

  (println "\n=== Done ==="))
