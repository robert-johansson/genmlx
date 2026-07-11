(ns genmlx.world.distill-sandbox
  "Process-isolation layer for the distillation filter (genmlx-8d15).

   WHY. The filter evaluates UNTRUSTED teacher code (SCI eval + apply, and model
   scoring). Such code can NON-TERMINATE — the real qwen3.5-4b smoke emitted a
   factorial `(reduce * (iterate identity n))` (an infinite lazy seq) and a swapped-arg
   gcd `recur` — or crash the process natively. Single-threaded JS cannot preempt a
   synchronous infinite loop, so the Step-3-spec'd 'sandbox + timeout' MUST live in a
   separate process.

   HOW. A single RESUMABLE worker child (scripts/distill_check.cljs) evaluates rows
   from `--start` to the end, appending one EDN verdict line per row (flushed). The
   parent here WATCHDOGS it asynchronously (promesa; the event loop must stay free so
   the worker's exit fires and its writes flush) by polling the verdict file's line
   count:
     - progress (line count grows)  -> reset the stall timer;
     - stalled past `timeout-ms`     -> the row at the current line count is looping:
                                        kill the worker's PROCESS GROUP, record a
                                        :timeout verdict for it, respawn from the next row;
     - worker exits non-zero/signal  -> it crashed on the current row: record :crashed,
                                        respawn from the next row;
     - worker exits 0                -> all rows done.

   Two subprocess-lifecycle subtleties this handles (learned the hard way):
     * `bun run --bun nbb` spawns a wrapper whose nbb/node GRANDCHILD runs the loop, so
       we spawn `detached` and kill the whole PROCESS GROUP (`process.kill (- pid)`) —
       killing just the wrapper would orphan a CPU-spinning grandchild.
     * the watchdog is ASYNC (never blocks the event loop), so the worker's 'exit'
       event is actually delivered and clean completion is detected.

   The expensive GenMLX + native load is paid ONCE in the common case (and once more
   per killed candidate). evaluate-candidate stays PURE in genmlx.world.distill;
   isolation lives only here. collect-verdicts returns a PROMISE of the verdict vector
   (reassembled by row :index, deduped, sentinels dropped)."
  (:require [genmlx.world.distill :as d]
            [genmlx.world.distill-tasks :as t]
            [genmlx.world.t1-battery :as t1]
            [clojure.string :as str]
            [cljs.reader :as reader]
            [promesa.core :as p]))

(def ^:private fs (js/require "fs"))
(def ^:private cp (js/require "child_process"))

(def ^:private batteries
  "Task-battery registry for the :battery opt (and the worker's matching
   --battery arg): which in-tree task set resolves candidate task ids. \"t1\" is
   the T1 bake-off battery (genmlx-8lm2) — the distill seed set plus the lifted
   MSA tasks. The worker (scripts/distill_check.cljs) mirrors this mapping."
  {"distill" t/tasks-by-id
   "t1"      t1/tasks-by-id})

(defn- count-lines [path]
  (if (.existsSync fs path)
    (->> (str/split-lines (.readFileSync fs path "utf8")) (remove str/blank?) count)
    0))

(defn read-candidates
  "Read raw_candidates.jsonl into a vector of keywordized maps (malformed lines
   skipped). The parent and worker read it identically, so row indices agree."
  [path]
  (->> (str/split-lines (.readFileSync fs path "utf8"))
       (remove str/blank?)
       (keep (fn [l] (try (js->clj (js/JSON.parse l) :keywordize-keys true)
                          (catch :default _ nil))))
       vec))

(defn- spawn-worker [candidates-file out-path start eval-opts battery]
  (let [args (cond-> ["run" "--bun" "nbb" "scripts/distill_check.cljs"
                      "--candidates" candidates-file "--out" out-path "--start" (str start)
                      "--n-particles" (str (:n-particles eval-opts 50))]
               battery                (conj "--battery" battery)
               (:min-log-ml eval-opts) (conj "--min-log-ml" (str (:min-log-ml eval-opts))))]
    ;; detached -> the worker leads its own process group, so we can kill the whole
    ;; tree (wrapper + nbb/node grandchild). stdout ignored (wrapper is noisy; verdicts
    ;; go to the file); stderr inherited so a worker error surfaces on our stderr.
    (.spawn cp "bun" (clj->js args)
            #js {:stdio #js ["ignore" "ignore" "inherit"] :detached true})))

(defn- kill-group! [worker]
  (try (js/process.kill (- (.-pid worker)) "SIGKILL")
       (catch :default _ (try (.kill worker "SIGKILL") (catch :default _ nil)))))

(defn- watch-worker
  "Async watchdog over one worker. Resolves to
   {:status :done} | {:status :stall :index k} | {:status :crash :index k}.
   k = the current line count = the next-unwritten row = the one that stalled/crashed."
  [worker out-path timeout-ms poll-ms]
  (p/create
    (fn [resolve _reject]
      (let [last-lines (volatile! (count-lines out-path))
            last-prog  (volatile! (js/Date.now))
            iv         (volatile! nil)
            settled?   (volatile! false)
            finish     (fn [m] (when-not @settled?
                                 (vreset! settled? true)
                                 (when @iv (js/clearInterval @iv))
                                 (resolve m)))]
        (.on worker "exit"
             (fn [code signal]
               (if (and (= 0 code) (nil? signal))
                 (finish {:status :done})
                 (finish {:status :crash :index (count-lines out-path)}))))
        (vreset! iv
          (js/setInterval
            (fn []
              (when-not @settled?
                (let [lines (count-lines out-path)]
                  (if (> lines @last-lines)
                    (do (vreset! last-lines lines) (vreset! last-prog (js/Date.now)))
                    (when (> (- (js/Date.now) @last-prog) timeout-ms)
                      (kill-group! worker)
                      (finish {:status :stall :index lines}))))))
            poll-ms))))))

(defn- assemble
  "Read the accumulated verdict lines, dedupe by :index (prefer a real verdict over a
   :timeout/:crashed/unknown sentinel if a race wrote both), drop unknown-task rows,
   strip :index, and return verdicts ordered by row index."
  [out-path]
  (let [sentinel? #(or (:unknown-task? %) (contains? #{:timeout :crashed} (:reason %)))]
    (->> (str/split-lines (.readFileSync fs out-path "utf8"))
         (remove str/blank?)
         (map reader/read-string)
         (group-by :index)
         vals
         (map (fn [vs] (or (first (remove sentinel? vs)) (first vs))))
         (remove :unknown-task?)
         (sort-by :index)
         (mapv #(dissoc % :index)))))

(defn collect-verdicts
  "Evaluate every candidate in `candidates-file` under process isolation. Returns a
   PROMISE of the verdict vector (sorted by row index). A non-terminating row gets a
   :timeout verdict, a process-crashing row gets :crashed, and the batch ALWAYS
   completes.

   opts:
     :out-path    scratch EDN file accumulating verdict lines (required)
     :eval-opts   {:n-particles :min-log-ml} forwarded to evaluate-candidate
     :battery     task-set name resolving candidate task ids (see `batteries`;
                  default \"distill\" — the pre-8lm2 behavior, unchanged)
     :timeout-ms  per-candidate stall budget before the worker group is killed
                  (default 15000; must exceed the worker's cold-start ~2-4s)
     :poll-ms     watchdog poll interval (default 400)
     :verbose?    print per-worker / per-kill progress"
  [candidates-file {:keys [out-path eval-opts battery timeout-ms poll-ms verbose?]
                    :or   {timeout-ms 15000 poll-ms 400}}]
  (let [rows  (read-candidates candidates-file)
        n     (count rows)
        by-id (get batteries battery t/tasks-by-id)]
    (.writeFileSync fs out-path "")
    (letfn [(step [start]
              (if (>= start n)
                (p/resolved (assemble out-path))
                (do (when verbose? (println (str "  [sandbox] worker resuming at row " start "/" n)))
                    (-> (watch-worker (spawn-worker candidates-file out-path start eval-opts battery)
                                      out-path timeout-ms poll-ms)
                        (p/then (fn [{:keys [status index]}]
                                  (if (= :done status)
                                    (assemble out-path)
                                    (let [reason (if (= :stall status) :timeout :crashed)
                                          {:keys [task-id sample-idx]} (d/candidate->fields (nth rows index))
                                          task   (or (get by-id task-id) {:id task-id :kind nil})]
                                      (when verbose?
                                        (println (str "  [sandbox] row " index " (" task-id " #" sample-idx
                                                      ") -> " reason "; resuming")))
                                      (.appendFileSync fs out-path
                                                       (str (pr-str (assoc (d/timeout-verdict task sample-idx reason)
                                                                           :index index)) "\n"))
                                      (step (inc index))))))))))]
      (step 0))))
