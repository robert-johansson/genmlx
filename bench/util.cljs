(ns bench.util
  "Shared infrastructure for benchmark scripts.

   Provides: benchmark, write-json, stats helpers, output-dir resolution.")

;; ---------------------------------------------------------------------------
;; Node.js interop
;; ---------------------------------------------------------------------------

(def fs (js/require "fs"))
(def path-mod (js/require "path"))

(defn perf-now [] (js/performance.now))

(defn out-dir
  "Resolve output directory from GENMLX_RESULTS_DIR env var or fallback."
  [fallback-name]
  (or (aget (.-env js/process) "GENMLX_RESULTS_DIR")
      (.resolve path-mod (js/process.cwd) (str "results/" fallback-name))))

(defn ensure-dir [dir]
  (when-not (.existsSync fs dir)
    (.mkdirSync fs dir #js {:recursive true})))

(defn write-json
  "Write data as pretty-printed JSON to dir/filename."
  [dir filename data]
  (ensure-dir dir)
  (let [filepath (str dir "/" filename)]
    (.writeFileSync fs filepath (js/JSON.stringify (clj->js data) nil 2))
    (println (str "  wrote: " filepath))))

;; ---------------------------------------------------------------------------
;; Benchmarking
;; ---------------------------------------------------------------------------

(defn benchmark
  "Run f repeatedly, return timing statistics.
   outer-n independent runs (with cache clear between), each takes min of inner-n.
   Requires genmlx.mlx as mx in the calling namespace."
  [label f mx-ns & {:keys [warmup-n outer-n inner-n]
                    :or {warmup-n 10 outer-n 5 inner-n 10}}]
  (let [materialize! (.-materialize! mx-ns)
        clear-cache! (.-clear-cache! mx-ns)]
    (println (str "\n  [" label "] warming up..."))
    (dotimes [_ warmup-n] (f) (materialize!))
    (clear-cache!)
    (let [outer-times
          (vec (for [_ (range outer-n)]
                 (let [inner-times
                       (vec (for [_ (range inner-n)]
                              (let [t0 (perf-now)]
                                (f)
                                (materialize!)
                                (- (perf-now) t0))))]
                   (clear-cache!)
                   (apply min inner-times))))
          mean-ms (/ (reduce + outer-times) (count outer-times))
          std-ms  (js/Math.sqrt (/ (reduce + (map #(* (- % mean-ms) (- % mean-ms))
                                                    outer-times))
                                     (max 1 (dec (count outer-times)))))]
      (println (str "  [" label "] " (.toFixed mean-ms 3) " ± " (.toFixed std-ms 3) " ms"))
      {:label label :mean-ms mean-ms :std-ms std-ms
       :min-ms (apply min outer-times) :max-ms (apply max outer-times)
       :raw outer-times})))

;; ---------------------------------------------------------------------------
;; Statistics
;; ---------------------------------------------------------------------------

(defn mean [xs] (/ (reduce + xs) (count xs)))

(defn variance [xs]
  (let [m (mean xs) n (count xs)]
    (/ (reduce + (map #(let [d (- % m)] (* d d)) xs)) n)))

(defn std [xs] (js/Math.sqrt (variance xs)))

(defn ess-from-log-weights
  "Effective sample size from unnormalized log-weights."
  [log-ws]
  (let [max-w (apply max log-ws)
        ws (map #(js/Math.exp (- % max-w)) log-ws)
        s (reduce + ws)
        nw (map #(/ % s) ws)]
    (/ 1.0 (reduce + (map #(* % %) nw)))))

(defn log-ml-from-log-weights
  "Log marginal likelihood from unnormalized log-weights."
  [log-ws]
  (let [n (count log-ws)
        max-w (apply max log-ws)]
    (+ max-w (- (js/Math.log (reduce + (map #(js/Math.exp (- % max-w)) log-ws)))
                (js/Math.log n)))))

(defn timing-map
  "Extract a clean timing summary from a benchmark result."
  [result]
  (select-keys result [:label :mean-ms :std-ms :min-ms :max-ms]))
