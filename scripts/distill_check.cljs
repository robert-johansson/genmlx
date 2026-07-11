(ns genmlx.distill-check
  "Sandbox WORKER for the distillation filter (genmlx-8d15).

   A RESUMABLE leaf process: evaluates candidates from `--start` to the end of the
   candidates file, appending one EDN verdict line per row (flushed) to `--out`. Its
   point is to be DISPOSABLE — if a candidate is non-terminating (a sync infinite loop
   that single-threaded JS cannot preempt) or crashes the process natively, the PARENT
   watchdog (genmlx.world.distill-sandbox) kills this worker and respawns it past the
   bad row. Because verdicts are appended incrementally, the rows completed before the
   kill are never lost, and `--start` resumes exactly where the parent says.

   Usage (invoked by the parent, not by hand):
     bun run --bun nbb scripts/distill_check.cljs \\
       --candidates <raw_candidates.jsonl> --out <verdicts.edn> --start <i> \\
       [--n-particles N] [--min-log-ml F] [--battery distill|t1]

   --battery selects which in-tree task set resolves task ids (default the
   distill seed set; \"t1\" adds the lifted MSA tasks of the T1 bake-off battery,
   genmlx-8lm2). Mirrors the `batteries` registry in genmlx.world.distill-sandbox."
  (:require [genmlx.world.distill :as d]
            [genmlx.world.distill-tasks :as t]
            [genmlx.world.t1-battery :as t1]
            [clojure.string :as str]))

(def fs (js/require "fs"))

(defn- parse-args [argv]
  (loop [args (seq argv), m {}]
    (if-let [a (first args)]
      (if (str/starts-with? a "--")
        (let [v (second args)]
          (if (and v (not (str/starts-with? v "--")))
            (recur (nnext args) (assoc m (keyword (subs a 2)) v))
            (recur (next args) (assoc m (keyword (subs a 2)) true))))
        (recur (next args) m))
      m)))

(defn- read-candidates [path]
  (->> (str/split-lines (.readFileSync fs path "utf8"))
       (remove str/blank?)
       (keep (fn [l] (try (js->clj (js/JSON.parse l) :keywordize-keys true)
                          (catch :default _ nil))))
       vec))

(let [opts   (parse-args (vec (drop 2 (.-argv js/process))))
      cf     (:candidates opts)
      out    (:out opts)
      start  (js/parseInt (or (:start opts) "0") 10)
      n-part (js/parseInt (or (:n-particles opts) "50") 10)
      min-ml (let [m (and (string? (:min-log-ml opts)) (js/parseFloat (:min-log-ml opts)))]
               (when (and m (not (js/isNaN m))) m))
      eopts  (cond-> {:n-particles n-part} min-ml (assoc :min-log-ml min-ml))
      by-id  (case (:battery opts) "t1" t1/tasks-by-id t/tasks-by-id)
      rows   (read-candidates cf)]
  (doseq [i (range start (count rows))]
    (let [{:keys [task-id sample-idx raw-text]} (d/candidate->fields (nth rows i))
          task    (get by-id task-id)
          verdict (if task
                    (assoc (d/evaluate-candidate (merge task eopts) raw-text sample-idx) :index i)
                    {:index i :unknown-task? true :task-id task-id :sample-idx sample-idx})]
      ;; appendFileSync is synchronous + flushed, so the parent watchdog sees each
      ;; completed row immediately and never loses work when it kills this worker.
      (.appendFileSync fs out (str (pr-str verdict) "\n")))))
