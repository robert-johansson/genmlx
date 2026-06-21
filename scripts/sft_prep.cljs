(ns genmlx.sft-prep
  "Data-prep runner for the cljs-coder SFT step (genmlx-o8w9). The thin I/O shell around
   the pure core (genmlx.world.sft) — twin of scripts/distill_filter.cljs.

   THREE MODES

     1. Export the TRAIN tasks for the teacher (the split happens BEFORE generation, so
        the corpus is built only over train tasks and the eval tasks stay truly held-out):
          bun run --bun nbb scripts/sft_prep.cljs --export-train-tasks <tasks.jsonl>

     2. Export the held-out EVAL tasks for student generation at eval time:
          bun run --bun nbb scripts/sft_prep.cljs --export-eval-tasks <eval_tasks.jsonl>

        Both exports write the TEACHER-FACING projection only ({task_id, kind,
        system_prompt, prompt}) — the held-out oracle signal never leaves the tree.

     3. Build an mlx-lm LoRA data dir from a distilled corpus:
          bun run --bun nbb scripts/sft_prep.cljs \\
            --corpus <distill_sft.jsonl> --out <dir> \\
            [--pairs <training_pairs.jsonl>] [--blend N] [--valid-frac 0.15]
        --corpus      the oracle-validated rows from distill_filter (step 3 output)
        --pairs       optional volume corpus (already {messages} chat rows) to blend in
        --blend N     how many volume rows to append behind the distilled rows (default 0)
        --valid-frac  fraction of the train rows held out as the mlx-lm validation set
                      (loss monitoring / early-stopping — NOT the held-out pass@1 eval;
                      default 0.15)
        Writes <dir>/{train.jsonl, valid.jsonl}, rows = {\"messages\":[...]}. ASSERTS that
        no held-out eval-task row entered training (aborts with a non-zero exit on leak),
        and logs row/task counts.

   Outputs default to an EXTERNAL scratch dir ($TMPDIR/genmlx-sft) so nothing lands in
   the repo."
  (:require [genmlx.world.sft :as sft]
            [genmlx.world.distill-tasks :as t]
            [clojure.string :as str]))

(def fs (js/require "fs"))
(def path-mod (js/require "path"))

(defn- ensure-dir [dir]
  (when-not (.existsSync fs dir)
    (.mkdirSync fs dir #js {:recursive true})))

(defn- default-out-dir []
  (let [tmp (or (.. js/process -env -TMPDIR) "/tmp")]
    (.join path-mod tmp "genmlx-sft")))

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

(defn- read-jsonl
  "Read a JSONL file into a vector of keywordized maps; a malformed line is skipped
   and counted. Returns {:rows [...] :skipped n}."
  [path]
  (let [lines  (remove str/blank? (str/split-lines (.readFileSync fs path "utf8")))
        parsed (map (fn [l] (try (js->clj (js/JSON.parse l) :keywordize-keys true)
                                 (catch :default _ ::bad)))
                    lines)]
    {:rows    (vec (remove #(= ::bad %) parsed))
     :skipped (count (filter #(= ::bad %) parsed))}))

(defn- write-jsonl [path rows]
  (.writeFileSync fs path (str (str/join "\n" (map #(js/JSON.stringify (clj->js %)) rows))
                               (when (seq rows) "\n"))))

(defn- pos-int-or [s d]
  (if (string? s)
    (let [n (js/parseInt s 10)] (if (or (js/isNaN n) (neg? n)) d n))
    d))

(defn- frac-or [s d]
  (if (string? s)
    (let [x (js/parseFloat s)] (if (js/isNaN x) d x))
    d))

;; ---------------------------------------------------------------------------
;; Mode 1/2 — export teacher-facing task prompts for one side of the split.
;; ---------------------------------------------------------------------------

(defn- export-side! [path which tasks]
  (ensure-dir (.dirname path-mod path))
  (write-jsonl path (map t/task->prompt-record tasks))
  (println (str "Exported " (count tasks) " " (name which) " task prompts -> " path))
  (println (str "  ids: " (str/join ", " (map :id tasks))))
  (println "  (held-out oracle signal NOT included — no test leakage)"))

;; ---------------------------------------------------------------------------
;; Mode 3 — build the mlx-lm LoRA data dir.
;; ---------------------------------------------------------------------------

(defn- build-data-dir! [{:keys [corpus out pairs blend valid-frac]}]
  (let [out-dir (or out (default-out-dir))
        n-blend (pos-int-or blend 0)
        vfrac   (frac-or valid-frac 0.15)
        {:keys [rows skipped]} (read-jsonl corpus)
        ;; 1. partition by task: drop any held-out eval-task row, report it.
        {:keys [train-rows dropped-eval eval-task-ids-present train-task-ids]}
        (sft/partition-corpus rows)
        ;; 2. HARD leakage guard — abort if any eval row survived into training.
        _ (sft/assert-train-disjoint! train-rows)
        ;; 3. strip provenance -> bare {messages}; optionally blend volume rows.
        distilled (mapv sft/row->messages train-rows)
        vol       (when (and pairs (pos? n-blend)) (:rows (read-jsonl pairs)))
        blended   (sft/blend distilled (vec vol) n-blend)
        ;; 4. carve a validation slice for mlx-lm loss monitoring.
        {:keys [train valid]} (sft/valid-split blended vfrac)]
    (ensure-dir out-dir)
    (write-jsonl (.join path-mod out-dir "train.jsonl") train)
    (write-jsonl (.join path-mod out-dir "valid.jsonl") valid)
    (println (str "Read " (count rows) " distilled rows from " corpus
                  (when (pos? skipped) (str " (" skipped " malformed skipped)"))))
    (when (seq dropped-eval)
      (println (str "  DROPPED " (count dropped-eval) " held-out eval-task row(s): "
                    (str/join ", " eval-task-ids-present) " (never trained on)")))
    (when (and pairs (pos? n-blend))
      (println (str "  Blended " (min n-blend (count vol)) " volume rows from " pairs)))
    (println (str "  train tasks present: " (str/join ", " train-task-ids)))
    (println (str "\nWrote mlx-lm data dir -> " out-dir))
    (println (str "  train.jsonl: " (count train) " rows"))
    (println (str "  valid.jsonl: " (count valid) " rows"))
    (println (str "  (" (count distilled) " distilled + "
                  (max 0 (- (count blended) (count distilled))) " volume = "
                  (count blended) " total, split " vfrac " to valid)"))
    ;; Honesty: partition-corpus + assert-train-disjoint! guarantee the DISTILLED rows are
    ;; disjoint from the held-out eval tasks. Blended volume rows carry no :task-id, so the
    ;; task-id guard cannot see them — say so rather than print an unconditional ✓ (genmlx-o8w9 review).
    (if (and pairs (pos? n-blend))
      (println (str "\n  LEAKAGE GUARD: distilled rows are disjoint from held-out eval tasks ["
                    (str/join ", " (sort sft/eval-task-ids)) "] ✓"
                    "\n  NOTE: " (min n-blend (count vol)) " blended volume rows are NOT task-id-screened"
                    " (general-cljs corpus, no :task-id) — assumed task-neutral."))
      (println (str "\n  LEAKAGE GUARD: held-out eval tasks ["
                    (str/join ", " (sort sft/eval-task-ids)) "] are absent from training ✓")))))

;; ---------------------------------------------------------------------------

(let [opts (parse-args (vec (drop 2 (.-argv js/process))))
      {:keys [train eval]} (sft/split-tasks t/tasks)]
  (cond
    (:export-train-tasks opts)
    (export-side! (if (string? (:export-train-tasks opts)) (:export-train-tasks opts)
                      (.join path-mod (default-out-dir) "train_tasks.jsonl"))
                  :train train)

    (:export-eval-tasks opts)
    (export-side! (if (string? (:export-eval-tasks opts)) (:export-eval-tasks opts)
                      (.join path-mod (default-out-dir) "eval_tasks.jsonl"))
                  :eval eval)

    (:corpus opts)
    (try (build-data-dir! opts)
         (catch :default e
           (println "ERROR:" (.-message e))
           (set! (.-exitCode js/process) 1)))

    :else
    (do (println "usage:")
        (println "  --export-train-tasks <tasks.jsonl>")
        (println "  --export-eval-tasks <eval_tasks.jsonl>")
        (println "  --corpus <distill_sft.jsonl> --out <dir> [--pairs <pairs.jsonl>] [--blend N] [--valid-frac 0.15]")
        (set! (.-exitCode js/process) 1))))
