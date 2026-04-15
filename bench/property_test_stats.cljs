(ns bench.property-test-stats
  "Property Test Stats — runs all property test files and collects statistics.

   Discovers test/genmlx/*_property_test.cljs files dynamically, executes each
   as a child process, parses cljs.test output, and writes a summary to data.json.

   Output: $GENMLX_RESULTS_DIR/data.json or results/property-test-stats/data.json

   Usage: bun run --bun nbb bench/property_test_stats.cljs")

;; ---------------------------------------------------------------------------
;; Node.js interop
;; ---------------------------------------------------------------------------

(def fs (js/require "fs"))
(def path-mod (js/require "path"))
(def child-process (js/require "child_process"))

(def bun-path "/Users/robert/.bun/bin/bun")

(def out-dir
  (or (aget (.-env js/process) "GENMLX_RESULTS_DIR")
      (.resolve path-mod (js/process.cwd) "results/property-test-stats")))

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
;; Dynamic test file discovery
;; ---------------------------------------------------------------------------

(defn discover-property-test-files
  "Scan test/genmlx/ for *_property_test.cljs files and return suite definitions."
  []
  (let [test-dir (.resolve path-mod (js/process.cwd) "test/genmlx")
        files (array-seq (.readdirSync fs test-dir))
        property-files (->> files
                            (filter #(re-find #"_property_test\.cljs$" %))
                            (sort))]
    (mapv (fn [filename]
            (let [name (.replace filename ".cljs" "")]
              {:name name
               :file (str "test/genmlx/" filename)}))
          property-files)))

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

(defn timeout-for-suite
  "Return timeout in ms for a given suite. SMC property tests get extra time."
  [name]
  (if (= name "smc_property_test")
    300000
    120000))

(defn run-suite
  "Run a single test suite as a child process. Returns result map."
  [{:keys [name file]}]
  (println (str "\n  running: " name " (" file ")"))
  (let [t0 (perf-now)
        timeout (timeout-for-suite name)]
    (try
      (let [cmd (str bun-path " run --bun nbb " file)
            stdout (.toString
                     (.execSync child-process cmd
                                #js {:timeout timeout
                                     :maxBuffer (* 10 1024 1024)
                                     :encoding "utf8"
                                     :stdio #js ["pipe" "pipe" "pipe"]}))
            duration-ms (- (perf-now) t0)
            parsed (parse-cljs-test-output stdout)]
        (println (str "    " (:pass parsed) "/" (:total-assertions parsed)
                      " assertions passed"
                      (when (pos? (:fail parsed))
                        (str ", " (:fail parsed) " failures"))
                      (when (pos? (:error parsed))
                        (str ", " (:error parsed) " errors"))
                      " (" (.toFixed duration-ms 0) "ms)"))
        {:name name
         :file file
         :pass (:pass parsed)
         :fail (:fail parsed)
         :error (:error parsed)
         :total-assertions (:total-assertions parsed)
         :duration-ms (js/Math.round duration-ms)
         :status (if (and (zero? (:fail parsed))
                          (zero? (:error parsed)))
                   "pass"
                   "fail")})
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
           :duration-ms (js/Math.round duration-ms)
           :status "error"
           :error-message (subs msg 0 (min (count msg) 500))})))))

;; ---------------------------------------------------------------------------
;; Main
;; ---------------------------------------------------------------------------

(println "\n=== GenMLX Property Test Stats ===")
(println (str "  output: " out-dir))

(let [suites (discover-property-test-files)
      _ (println (str "  discovered: " (count suites) " property test files"))
      t0 (perf-now)
      results (mapv run-suite suites)
      total-duration-ms (- (perf-now) t0)
      totals {:pass (reduce + (map :pass results))
              :fail (reduce + (map :fail results))
              :error (reduce + (map :error results))
              :total-assertions (reduce + (map :total-assertions results))
              :duration-ms (js/Math.round total-duration-ms)
              :suites-passed (count (filter #(= "pass" (:status %)) results))
              :suites-total (count results)}
      data {:experiment "property-test-stats"
            :suites results
            :totals totals}]

  (println "\n=== Summary ===")
  (println (str "  suites: " (:suites-passed totals) "/" (:suites-total totals) " passed"))
  (println (str "  assertions: " (:pass totals) "/" (:total-assertions totals)
                " passed, " (:fail totals) " failures, " (:error totals) " errors"))
  (println (str "  duration: " (.toFixed (/ total-duration-ms 1000) 1) "s"))

  (write-json "data.json" data)

  (println "\n=== Done ==="))
