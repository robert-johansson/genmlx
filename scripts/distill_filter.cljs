(ns genmlx.distill-filter
  "Runner for the cljs-coder distillation oracle-filter (genmlx-j0d6).

   The thin I/O shell around the pure core (genmlx.world.distill): reads the teacher's
   raw candidates, runs the oracle gate ladder against the in-tree seed tasks, and
   writes an oracle-validated SFT corpus + a stats report. No teacher LLM is loaded
   here — candidates arrive over a file.

   TWO MODES

     1. Export tasks for the teacher (step 1->2 handoff):
          bun run --bun nbb scripts/distill_filter.cljs --export-tasks <tasks.jsonl>
        Writes one {task_id, kind, system_prompt, prompt} line per seed task. The
        held-out oracle signal (observations / transitions / test-cases) is NOT
        exported — it can never leak into a teacher prompt.

     2. Filter the teacher's candidates (steps 3-4):
          bun run --bun nbb scripts/distill_filter.cljs \\
            --candidates <raw_candidates.jsonl> --out <dir> \\
            [--top-k 1] [--n-particles 50] [--min-log-ml <float>] \\
            [--timeout-ms 15000] [--no-sandbox]
        --top-k      best candidates kept per prompt (default 1 — one best exemplar)
        --min-log-ml optional absolute model-evidence floor for :program candidates
        --timeout-ms per-candidate evaluation budget; a non-terminating teacher
                     candidate is killed and recorded :timeout (default 15000)
        --no-sandbox evaluate in-process (faster, but a non-terminating candidate
                     hangs the whole run — use only on trusted input)
        raw_candidates.jsonl lines: {task_id, sample_idx, raw_text} (aliases accepted:
        completion / text for raw_text). Writes into <dir>:
          distill_sft.jsonl  — Qwen3 chat rows (the corpus)
          verdicts.jsonl     — every candidate's verdict (the audit trail)
          stats.json         — parse/eval/test-pass rates, mean log-ML, per-prompt yield

   Outputs default to an EXTERNAL scratch dir ($TMPDIR/genmlx-distill) so nothing
   lands in the repo (the repo intentionally has no catch-all gitignore)."
  (:require [genmlx.world.distill :as d]
            [genmlx.world.distill-tasks :as t]
            [genmlx.world.distill-sandbox :as sb]
            [clojure.string :as str]
            [promesa.core :as p]))

(def fs (js/require "fs"))
(def path-mod (js/require "path"))

(defn- ensure-dir [dir]
  (when-not (.existsSync fs dir)
    (.mkdirSync fs dir #js {:recursive true})))

(defn- default-out-dir []
  (let [tmp (or (.. js/process -env -TMPDIR) "/tmp")]
    (.join path-mod tmp "genmlx-distill")))

;; ---------------------------------------------------------------------------
;; Tiny CLI arg parsing: --key value pairs + bare --flags.
;; ---------------------------------------------------------------------------

(defn- parse-args [argv]
  (loop [args (seq argv), m {}]
    (if-let [a (first args)]
      (if (str/starts-with? a "--")
        (let [k (keyword (subs a 2))
              v (second args)]
          (if (and v (not (str/starts-with? v "--")))
            (recur (nnext args) (assoc m k v))
            (recur (next args) (assoc m k true))))
        (recur (next args) m))
      m)))

;; ---------------------------------------------------------------------------
;; JSONL read/write.
;; ---------------------------------------------------------------------------

(defn- read-jsonl
  "Read a JSONL file into {:rows [...] :skipped n}. A malformed/truncated line is
   skipped and counted — one bad line never aborts the whole filter run."
  [path]
  (let [lines  (remove str/blank? (str/split-lines (.readFileSync fs path "utf8")))
        parsed (map (fn [l] (try (js->clj (js/JSON.parse l) :keywordize-keys true)
                                 (catch :default _ ::bad)))
                    lines)]
    {:rows    (vec (remove #(= ::bad %) parsed))
     :skipped (count (filter #(= ::bad %) parsed))}))

(defn- pos-int-or-nil
  "Parse s as a positive integer, or nil (a bare flag arrives as boolean true ->
   nil, so the caller can reject it instead of silently using a garbage value)."
  [s]
  (let [n (js/parseInt (str s) 10)]
    (when-not (or (js/isNaN n) (< n 1)) n)))

(defn- write-jsonl [path rows]
  (.writeFileSync fs path (str (str/join "\n" (map #(js/JSON.stringify (clj->js %)) rows))
                               (when (seq rows) "\n"))))

(defn- write-json [path data]
  (.writeFileSync fs path (js/JSON.stringify (clj->js data) nil 2)))

;; ---------------------------------------------------------------------------
;; Mode 1 — export teacher-facing task prompts.
;; ---------------------------------------------------------------------------

(defn- export-tasks! [path]
  (ensure-dir (.dirname path-mod path))
  (write-jsonl path (map t/task->prompt-record t/tasks))
  (println (str "Exported " (count t/tasks) " task prompts -> " path))
  (println "  (held-out oracle signal NOT included — no test leakage)"))

;; ---------------------------------------------------------------------------
;; Mode 2 — filter candidates into an SFT corpus.
;; ---------------------------------------------------------------------------

;; In-process scoring (the --no-sandbox path): fast, but a non-terminating candidate
;; hangs the whole run. The default path isolates each candidate in a worker process.
(defn- score-in-process [rows eval-opts note]
  (vec (for [c rows
             :let [{:keys [task-id sample-idx raw-text]} (d/candidate->fields c)
                   task (get t/tasks-by-id task-id)]
             :when task]
         (do (note (str "scoring " task-id " #" sample-idx))
             (let [v (d/evaluate-candidate (merge task eval-opts) raw-text sample-idx)]
               (note (str "  -> " (:reason v) (when (:log-ml v) (str " log-ml=" (:log-ml v)))))
               v)))))

(defn- filter! [{:keys [candidates out top-k n-particles min-log-ml timeout-ms no-sandbox]}]
  (let [out-dir   (or out (default-out-dir))
        top-k     (pos-int-or-nil (or top-k 1))
        n-part    (pos-int-or-nil (or n-particles 50))
        tmo       (pos-int-or-nil (or timeout-ms 15000))
        min-ml    (let [m (and (string? min-log-ml) (js/parseFloat min-log-ml))]
                    (when (and m (not (js/isNaN m))) m))]
    (cond
      (or (nil? top-k) (nil? n-part) (nil? tmo))
      (do (println "ERROR: --top-k, --n-particles and --timeout-ms must be positive integers")
          (set! (.-exitCode js/process) 1))

      :else
      (let [{:keys [rows skipped]} (read-jsonl candidates)
            _        (ensure-dir out-dir)
            _        (println (str "Read " (count rows) " candidates from " candidates
                                   (when (pos? skipped)
                                     (str " (" skipped " malformed lines skipped)"))
                                   (if no-sandbox "  [in-process]"
                                       (str "  [sandboxed, timeout " tmo "ms]"))))
            ;; attach scoring opts to each task at evaluate time via assoc/merge
            eval-opts (cond-> {:n-particles n-part} min-ml (assoc :min-log-ml min-ml))
            verbose? (.. js/process -env -DISTILL_VERBOSE)
            prog-log (.join path-mod out-dir "progress.log")
            _        (when verbose? (.writeFileSync fs prog-log ""))
            note     (fn [s] (when verbose? (println s) (.appendFileSync fs prog-log (str s "\n"))))
            unknown  (count (remove #(get t/tasks-by-id (:task-id (d/candidate->fields %))) rows))]
        (p/let [verdicts (if no-sandbox
                           (p/resolved (score-in-process rows eval-opts note))
                           (sb/collect-verdicts candidates
                                                {:out-path   (.join path-mod out-dir "_sandbox_verdicts.edn")
                                                 :eval-opts  eval-opts :timeout-ms tmo
                                                 :poll-ms    400 :verbose? verbose?}))]
          (let [n-timeout (count (filter #(contains? #{:timeout :crashed} (:reason %)) verdicts))
                selected (d/rank-and-select verdicts top-k)
                records  (d/build-sft-records t/tasks-by-id selected)
                stats    (assoc (d/verdicts->stats t/tasks verdicts selected)
                                :top-k top-k :n-particles n-part :min-log-ml min-ml
                                :sandboxed? (not no-sandbox) :timeout-ms tmo
                                :n-timed-out n-timeout
                                :malformed-lines-skipped skipped
                                :candidates-with-unknown-task unknown)]
            (write-jsonl (.join path-mod out-dir "distill_sft.jsonl") records)
            (write-jsonl (.join path-mod out-dir "verdicts.jsonl") verdicts)
            (write-json  (.join path-mod out-dir "stats.json") stats)
            (println "\n== distillation stats ==")
            (doseq [k [:n-candidates :n-tasks :n-tasks-attempted :n-kept :n-selected
                       :parse-rate :eval-rate :program-pass-rate :function-pass-rate
                       :mean-log-ml :yield-per-prompt :task-space-coverage
                       :n-prompts-covered :n-selected-noisy-is :n-timed-out :drop-reasons]]
              (println (str "  " (name k) ": " (get stats k))))
            (when (pos? n-timeout)
              (println (str "  NOTE: " n-timeout
                            " candidate(s) were non-terminating/crashing and recorded :timeout/:crashed")))
            (when (pos? (:n-selected-noisy-is stats))
              (println (str "  WARNING: " (:n-selected-noisy-is stats)
                            " selected program(s) ranked by non-reproducible importance sampling")))
            (when (pos? unknown)
              (println (str "  WARNING: " unknown " candidates referenced an unknown task_id (skipped)")))
            (println (str "\nWrote:\n  " (.join path-mod out-dir "distill_sft.jsonl")
                          "  (" (count records) " SFT rows)\n  "
                          (.join path-mod out-dir "verdicts.jsonl") "\n  "
                          (.join path-mod out-dir "stats.json")))))))))

;; ---------------------------------------------------------------------------

(let [opts (parse-args (vec (drop 2 (.-argv js/process))))]
  (cond
    (:export-tasks opts)
    (export-tasks! (if (string? (:export-tasks opts))
                     (:export-tasks opts)
                     (.join path-mod (default-out-dir) "tasks.jsonl")))

    (:candidates opts)
    (-> (p/resolved nil)
        (p/then (fn [_] (filter! opts)))
        (p/catch (fn [e] (println "ERROR:" (.-message e))
                   (set! (.-exitCode js/process) 1))))

    :else
    (do (println "usage:")
        (println "  --export-tasks <tasks.jsonl>")
        (println "  --candidates <raw_candidates.jsonl> [--out <dir>] [--top-k 1] [--n-particles 50] [--min-log-ml <float>] [--timeout-ms 15000] [--no-sandbox]")
        (set! (.-exitCode js/process) 1))))
