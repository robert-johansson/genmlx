(ns run-experiments
  "GenMLX Experiment Orchestrator.

   Runs benchmark experiments defined in experiments.edn, writes results
   to results/<name>/data.json with metadata for reproducibility.

   Usage:
     bun run --bun nbb run_experiments.cljs                    ;; run all
     bun run --bun nbb run_experiments.cljs --only name1,name2 ;; run subset
     bun run --bun nbb run_experiments.cljs --category perf    ;; run by category
     bun run --bun nbb run_experiments.cljs --changed          ;; only if sources changed
     bun run --bun nbb run_experiments.cljs --figures          ;; regenerate figures after run
     bun run --bun nbb run_experiments.cljs --list             ;; list experiments
     bun run --bun nbb run_experiments.cljs --dry-run          ;; show what would run"
  (:require [clojure.edn :as edn]
            [clojure.string :as str]))

;; ---------------------------------------------------------------------------
;; Node.js interop
;; ---------------------------------------------------------------------------

(def fs (js/require "fs"))
(def path-mod (js/require "path"))
(def crypto (js/require "crypto"))
(def child-process (js/require "child_process"))
(def os (js/require "os"))

(defn read-file [path]
  (when (.existsSync fs path)
    (.toString (.readFileSync fs path "utf-8"))))

(defn write-file [path content]
  (let [dir (.dirname path-mod path)]
    (when-not (.existsSync fs dir)
      (.mkdirSync fs dir #js {:recursive true}))
    (.writeFileSync fs path content)))

(defn file-exists? [path] (.existsSync fs path))

;; ---------------------------------------------------------------------------
;; Hashing for change detection
;; ---------------------------------------------------------------------------

(defn hash-file [path]
  (when (file-exists? path)
    (let [h (.createHash crypto "sha256")]
      (.update h (.readFileSync fs path))
      (.digest h "hex"))))

(defn hash-sources
  "Hash a list of source files + the script itself into a single digest."
  [script sources]
  (let [h (.createHash crypto "sha256")
        all-files (cons script sources)]
    (doseq [f all-files]
      (let [abs (.resolve path-mod (js/process.cwd) f)]
        (when (file-exists? abs)
          (.update h (.readFileSync fs abs)))))
    (.digest h "hex")))

;; ---------------------------------------------------------------------------
;; Hardware / environment info
;; ---------------------------------------------------------------------------

(defn collect-metadata [experiment duration-ms source-hash]
  (let [git-sha (try
                  (str/trim (.toString (.execSync child-process "git rev-parse --short HEAD")))
                  (catch :default _ "unknown"))]
    {:experiment  (:name experiment)
     :git_sha     git-sha
     :timestamp   (.toISOString (js/Date.))
     :hardware    {:platform (.platform os)
                   :arch     (.arch os)
                   :cpus     (count (.cpus os))
                   :memory   (str (js/Math.round (/ (.totalmem os) 1073741824)) "GB")}
     :runtime     {:engine  "bun+nbb"
                   :node    (.-version js/process)}
     :source_hash source-hash
     :duration_ms duration-ms
     :script      (:script experiment)}))

;; ---------------------------------------------------------------------------
;; Experiment execution
;; ---------------------------------------------------------------------------

(defn results-dir [experiment-name]
  (.resolve path-mod (js/process.cwd) "results" experiment-name))

(defn metadata-path [experiment-name]
  (str (results-dir experiment-name) "/metadata.json"))

(defn should-run?
  "Check if experiment needs re-running (source hash changed or no results)."
  [experiment]
  (let [meta-file (metadata-path (:name experiment))
        source-hash (hash-sources (:script experiment) (:sources experiment))]
    (if-not (file-exists? meta-file)
      {:run? true :reason "no previous results"}
      (let [prev-meta (js->clj (js/JSON.parse (read-file meta-file)) :keywordize-keys true)]
        (if (= source-hash (:source_hash prev-meta))
          {:run? false :reason (str "unchanged since " (:timestamp prev-meta))}
          {:run? true :reason "sources changed"})))))

(defn run-experiment!
  "Run a single experiment via child process. Returns {:success? :duration-ms}."
  [experiment]
  (let [script-path (.resolve path-mod (js/process.cwd) (:script experiment))
        out-dir (results-dir (:name experiment))
        _ (when-not (.existsSync fs out-dir)
            (.mkdirSync fs out-dir #js {:recursive true}))
        t0 (js/Date.now)
        env (js/Object.assign
              #js {}
              (.-env js/process)
              #js {:GENMLX_RESULTS_DIR out-dir})]
    (try
      (let [bun-path (or (aget (.-env js/process) "BUN_PATH")
                         (try (str/trim (.toString (.execSync child-process "which bun")))
                              (catch :default _ "bun")))
            result (.execSync child-process
                     (str bun-path " run --bun nbb " script-path)
                     #js {:cwd      (js/process.cwd)
                          :env      env
                          :timeout  (or (:timeout-ms experiment) 300000)
                          :maxBuffer (* 50 1024 1024)  ;; 50MB
                          :stdio    "pipe"})
            duration-ms (- (js/Date.now) t0)
            output (.toString result)]
        ;; Save stdout
        (write-file (str out-dir "/output.txt") output)
        ;; Save metadata
        (let [source-hash (hash-sources (:script experiment) (:sources experiment))
              meta (collect-metadata experiment duration-ms source-hash)]
          (write-file (str out-dir "/metadata.json")
                      (js/JSON.stringify (clj->js meta) nil 2)))
        {:success? true :duration-ms duration-ms})
      (catch :default e
        (let [duration-ms (- (js/Date.now) t0)]
          (write-file (str out-dir "/error.txt")
                      (str "Error running " (:name experiment) "\n"
                           (.-message e) "\n"
                           (or (.-stderr e) "")))
          {:success? false :duration-ms duration-ms :error (.-message e)})))))

;; ---------------------------------------------------------------------------
;; Figure generation
;; ---------------------------------------------------------------------------

(def ^:private figure-scripts
  ["paper/viz/fig_performance.py"
   "paper/viz/fig_inference.py"
   "paper/viz/fig_analytical.py"
   "paper/viz/fig_system.py"])

(def ^:private python-bin ".venv-genjax/bin/python")

(defn generate-figures!
  "Run all 4 figure scripts against current results/. Returns count of successes."
  []
  (println "── Regenerating figures ──────────────────────")
  (if-not (file-exists? python-bin)
    (do (println (str "   ✗ python venv not found at " python-bin))
        0)
    (let [results (atom [])]
      (doseq [script figure-scripts]
        (try
          (.execSync child-process
            (str python-bin " " script)
            #js {:cwd (js/process.cwd) :stdio "pipe"})
          (println (str "   ✓ " script))
          (swap! results conj true)
          (catch :default e
            (println (str "   ✗ " script " — " (.-message e)))
            (swap! results conj false))))
      (count (filter identity @results)))))

;; ---------------------------------------------------------------------------
;; CLI argument parsing
;; ---------------------------------------------------------------------------

(defn parse-args [argv]
  (let [args (vec (drop 2 argv))]  ;; drop "nbb" and script path
    (loop [i 0 opts {}]
      (if (>= i (count args))
        opts
        (let [arg (nth args i)]
          (cond
            (= arg "--list")     (recur (inc i) (assoc opts :list? true))
            (= arg "--dry-run")  (recur (inc i) (assoc opts :dry-run? true))
            (= arg "--changed")  (recur (inc i) (assoc opts :changed? true))
            (= arg "--figures")  (recur (inc i) (assoc opts :figures? true))
            (= arg "--only")     (recur (+ i 2)
                                   (assoc opts :only
                                     (set (str/split (nth args (inc i)) #","))))
            (= arg "--category") (recur (+ i 2)
                                   (assoc opts :category
                                     (keyword (nth args (inc i)))))
            :else                (recur (inc i) opts)))))))

;; ---------------------------------------------------------------------------
;; Main
;; ---------------------------------------------------------------------------

(defn -main []
  (let [registry (edn/read-string (read-file "experiments.edn"))
        experiments (:experiments registry)
        opts (parse-args (js->clj (.-argv js/process)))]

    ;; --list: just print experiments and exit
    (when (:list? opts)
      (println "\nGenMLX Experiments:")
      (println (str "  " (count experiments) " experiments in registry\n"))
      (doseq [exp experiments]
        (let [status (should-run? exp)]
          (println (str "  " (if (:run? status) "○" "●") " "
                        (:name exp)
                        " [" (name (:category exp)) "]"
                        (when-not (:run? status)
                          (str " — " (:reason status)))))))
      (println)
      (js/process.exit 0))

    ;; Filter experiments
    (let [selected (cond->> experiments
                     (:only opts)     (filter #(contains? (:only opts) (:name %)))
                     (:category opts) (filter #(= (:category opts) (:category %)))
                     (:changed? opts) (filter #(:run? (should-run? %))))]

      (when (empty? selected)
        (if (:figures? opts)
          (do (println "No experiments to run. Regenerating figures only.\n")
              (generate-figures!)
              (js/process.exit 0))
          (do (println "No experiments to run.")
              (js/process.exit 0))))

      (println (str "\n=== GenMLX Experiment Runner ==="))
      (println (str "    " (count selected) " of " (count experiments) " experiments selected\n"))

      ;; --dry-run: show what would run
      (when (:dry-run? opts)
        (doseq [exp selected]
          (let [status (should-run? exp)]
            (println (str "  would run: " (:name exp)
                          " (" (:reason status) ")"))))
        (js/process.exit 0))

      ;; Run experiments
      (let [results (atom [])
            t-total (js/Date.now)]
        (doseq [exp selected]
          (println (str "── " (:name exp) " ──────────────────────────────"))
          (println (str "   " (:description exp)))
          (println (str "   script: " (:script exp)))
          (let [result (run-experiment! exp)]
            (swap! results conj (assoc result :name (:name exp)))
            (if (:success? result)
              (println (str "   ✓ done in " (js/Math.round (/ (:duration-ms result) 1000)) "s"))
              (println (str "   ✗ FAILED: " (:error result))))
            (println)))

        ;; Summary
        (let [total-ms (- (js/Date.now) t-total)
              successes (filter :success? @results)
              failures (remove :success? @results)]
          (println "=== Summary ===")
          (println (str "  " (count successes) "/" (count @results) " succeeded"))
          (when (seq failures)
            (println "  Failed:")
            (doseq [f failures]
              (println (str "    ✗ " (:name f)))))
          (println (str "  Total time: " (js/Math.round (/ total-ms 1000)) "s"))

          ;; Write run summary
          (write-file "results/_last_run.json"
            (js/JSON.stringify
              (clj->js {:timestamp (.toISOString (js/Date.))
                         :total_ms total-ms
                         :experiments (mapv #(select-keys % [:name :success? :duration-ms])
                                            @results)})
              nil 2))

          ;; Regenerate figures if requested
          (when (:figures? opts)
            (println)
            (generate-figures!)))))))

(-main)
