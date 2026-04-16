(ns bench.gfi-law-verification
  "GFI Law Verification — runs 5 GFI law test suites and reports pass rates.

   Executes each test suite as a child process, parses cljs.test output
   for assertion/failure/error counts, and writes a summary to data.json.

   Output: $GENMLX_RESULTS_DIR/data.json or results/gfi-law-verification/data.json

   Usage: bun run --bun nbb bench/gfi_law_verification.cljs")

;; ---------------------------------------------------------------------------
;; Node.js interop
;; ---------------------------------------------------------------------------

(def fs (js/require "fs"))
(def path-mod (js/require "path"))
(def child-process (js/require "child_process"))

(def bun-path "/Users/robert/.bun/bin/bun")

(def out-dir
  (or (aget (.-env js/process) "GENMLX_RESULTS_DIR")
      (.resolve path-mod (js/process.cwd) "results/gfi-law-verification")))

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
  [{:name "GFI Laws (1/10)"
    :file "test/genmlx/gfi_laws_test_p1.cljs"
    :expected nil}
   {:name "GFI Laws (2/10)"
    :file "test/genmlx/gfi_laws_test_p2.cljs"
    :expected nil}
   {:name "GFI Laws (3/10)"
    :file "test/genmlx/gfi_laws_test_p3.cljs"
    :expected nil}
   {:name "GFI Laws (4/10)"
    :file "test/genmlx/gfi_laws_test_p4.cljs"
    :expected nil}
   {:name "GFI Laws (5/10)"
    :file "test/genmlx/gfi_laws_test_p5.cljs"
    :expected nil}
   {:name "GFI Laws (6/10)"
    :file "test/genmlx/gfi_laws_test_p6.cljs"
    :expected nil}
   {:name "GFI Laws (7/10)"
    :file "test/genmlx/gfi_laws_test_p7.cljs"
    :expected nil}
   {:name "GFI Laws (8/10)"
    :file "test/genmlx/gfi_laws_test_p8.cljs"
    :expected nil}
   {:name "GFI Laws (9/10)"
    :file "test/genmlx/gfi_laws_test_p9.cljs"
    :expected nil}
   {:name "GFI Laws (10/10)"
    :file "test/genmlx/gfi_laws_test_p10.cljs"
    :expected nil}
   {:name "GFI Universal"
    :file "test/genmlx/gfi_universal_test.cljs"
    :expected nil}
   {:name "GFI Compositional"
    :file "test/genmlx/gfi_compositional_test.cljs"
    :expected nil}
   {:name "GFI Gradient"
    :file "test/genmlx/gfi_gradient_test.cljs"
    :expected nil}
   {:name "GFI Gaps"
    :file "test/genmlx/gfi_gaps_test.cljs"
    :expected nil}])

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
                                #js {:timeout 300000
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
                      (cond
                        (nil? expected)
                        (if (and (zero? (:fail parsed)) (zero? (:error parsed)))
                          " [OK]"
                          " [DEGRADED]")

                        (= (:total-assertions parsed) expected)
                        " [OK]"

                        :else
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
                          (or (nil? expected)
                              (= (:total-assertions parsed) expected)))
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

(println "\n=== GenMLX GFI Law Verification ===")
(println (str "  output: " out-dir))

(let [t0 (perf-now)
      results (mapv run-suite suites)
      total-duration-ms (- (perf-now) t0)
      totals {:pass (reduce + (map :pass results))
              :fail (reduce + (map :fail results))
              :error (reduce + (map :error results))
              :total-assertions (reduce + (map :total-assertions results))
              :duration-ms (js/Math.round total-duration-ms)
              :suites-passed (count (filter #(= "pass" (:status %)) results))
              :suites-total (count results)}
      data {:experiment "gfi-law-verification"
            :suites results
            :totals totals}]

  (println "\n=== Summary ===")
  (println (str "  suites: " (:suites-passed totals) "/" (:suites-total totals) " passed"))
  (println (str "  assertions: " (:pass totals) "/" (:total-assertions totals)
                " passed, " (:fail totals) " failures, " (:error totals) " errors"))
  (println (str "  duration: " (.toFixed (/ total-duration-ms 1000) 1) "s"))

  (write-json "data.json" data)

  (println "\n=== Done ==="))
