(ns genmlx.gen-tasks
  "Runner for the scaled task/curriculum generator (genmlx-7473), twin of
   scripts/distill_filter.cljs but over genmlx.world.distill-gen instead of the 12 seeds.

   MODES

     --validate
        Run EVERY generated task's :reference through the SAME oracle that grades the
        student (genmlx.world.distill/evaluate-candidate). Reports per-task kept?/reason/
        method and FAILS (non-zero exit) if any reference is not admitted — the grounding
        guarantee: a task whose own reference cannot pass its oracle is broken and must not
        ship. Use this before every teacher run.

     --export-train <prompts.jsonl>   teacher-facing prompts for the TRAIN tasks
     --export-eval  <prompts.jsonl>   teacher-facing prompts for the held-out EVAL tasks
        (both drop the oracle signal AND the reference — no leakage)

     --export-refs  <candidates.jsonl>
        One reference-solution candidate row per TRAIN task (sample_idx -1) — seed rows for
        the SFT corpus (oracle-validated, cold-start insurance). Eval tasks are excluded.

     --write-edn <gen_tasks.edn>      the full task set (with oracle signal + references)

   Default (no mode): print the generated-set summary.

   Outputs default under $TMPDIR/genmlx-gen."
  (:require [genmlx.world.distill-gen :as g]
            [genmlx.world.distill :as d]
            [clojure.string :as str]))

(def fs (js/require "fs"))
(def path-mod (js/require "path"))

(defn- ensure-dir [dir]
  (when-not (.existsSync fs dir) (.mkdirSync fs dir #js {:recursive true})))

(defn- default-out-dir []
  (.join path-mod (or (.. js/process -env -TMPDIR) "/tmp") "genmlx-gen"))

(defn- parse-args [argv]
  (loop [args (seq argv), m {}]
    (if-let [a (first args)]
      (if (str/starts-with? a "--")
        (let [k (keyword (subs a 2)), v (second args)]
          (if (and v (not (str/starts-with? v "--")))
            (recur (nnext args) (assoc m k v))
            (recur (next args) (assoc m k true))))
        (recur (next args) m))
      m)))

(defn- write-jsonl [path rows]
  (ensure-dir (.dirname path-mod path))
  (.writeFileSync fs path (str (str/join "\n" (map #(js/JSON.stringify (clj->js %)) rows))
                               (when (seq rows) "\n"))))

(defn- write-text [path s]
  (ensure-dir (.dirname path-mod path))
  (.writeFileSync fs path s))

(defn- pos-int-or [s d]
  (if (string? s) (let [n (js/parseInt s 10)] (if (or (js/isNaN n) (< n 1)) d n)) d))

;; ---------------------------------------------------------------------------

(defn- validate! [{:keys [n-particles]}]
  (let [n-part (pos-int-or n-particles 50)
        verdicts (mapv (fn [t]
                         (let [v (d/evaluate-candidate (assoc t :n-particles n-part)
                                                       (:reference t) -1)]
                           (assoc v :family (:family t) :split (:split t))))
                       g/all-tasks)
        failed   (remove :kept? verdicts)
        progs    (filter #(= :program (:kind %)) verdicts)
        is-progs (filter #(and (= :program (:kind %))
                               (not (contains? #{:exact :kalman} (:method %)))) progs)]
    (println (str "Validating " (count g/all-tasks) " references through the oracle "
                  "(n-particles " n-part ") ...\n"))
    (doseq [v verdicts]
      (println (str "  " (if (:kept? v) "ok  " "FAIL")
                    " " (.padEnd (str (:task-id v)) 22)
                    " " (.padEnd (str (name (:kind v))) 9)
                    " " (.padEnd (str (:reason v)) 14)
                    (when (:method v) (str "method=" (name (:method v)) " "))
                    (when (:log-ml v) (str "log-ml=" (.toFixed (js/Number (:log-ml v)) 2))))))
    (println (str "\n== validation =="))
    (println (str "  kept: " (count (filter :kept? verdicts)) "/" (count verdicts)))
    (println (str "  programs scored by EXACT marginal: " (- (count progs) (count is-progs))
                  "/" (count progs)
                  (when (seq is-progs) (str "  (IS-scored: " (str/join ", " (map :task-id is-progs)) ")"))))
    (when (seq failed)
      (println (str "\n  FAILED references (" (count failed) "):"))
      (doseq [v failed]
        (println (str "    " (:task-id v) " -> " (:reason v)
                      (when (:error v) (str " : " (:error v)))))))
    (if (seq failed)
      (do (println "\nGROUNDING FAILURE: some references are not admitted by their own oracle.")
          (set! (.-exitCode js/process) 1))
      (println "\nAll references admitted — task set is grounded. ✓"))))

(defn- export-prompts! [path tasks which]
  (write-jsonl path (map g/task->prompt-record tasks))
  (println (str "Exported " (count tasks) " " which " task prompts -> " path))
  (println "  (oracle signal + reference NOT included — no leakage)"))

(defn- export-refs! [path]
  (let [rows (map g/reference-record g/train-tasks)]
    (write-jsonl path rows)
    (println (str "Exported " (count rows) " reference candidate rows (train only) -> " path))))

(defn- write-edn! [path]
  (write-text path (with-out-str (pr g/all-tasks)))
  (println (str "Wrote full task set (" (count g/all-tasks) " tasks) -> " path)))

(defn- print-summary []
  (println "Generated task set summary:")
  (doseq [[k v] g/summary] (println (str "  " (name k) ": " (pr-str v))))
  (println "\nModes: --validate | --export-train <f> | --export-eval <f> | --export-refs <f> | --write-edn <f>"))

;; ---------------------------------------------------------------------------

(let [opts (parse-args (vec (drop 2 (.-argv js/process))))
      od   (default-out-dir)]
  (cond
    (:validate opts) (validate! opts)
    (:export-train opts) (export-prompts! (if (string? (:export-train opts)) (:export-train opts)
                                              (.join path-mod od "train_prompts.jsonl"))
                                          g/train-tasks "train")
    (:export-eval opts) (export-prompts! (if (string? (:export-eval opts)) (:export-eval opts)
                                             (.join path-mod od "eval_prompts.jsonl"))
                                         g/eval-tasks "eval")
    (:export-refs opts) (export-refs! (if (string? (:export-refs opts)) (:export-refs opts)
                                          (.join path-mod od "reference_candidates.jsonl")))
    (:write-edn opts) (write-edn! (if (string? (:write-edn opts)) (:write-edn opts)
                                      (.join path-mod od "gen_tasks.edn")))
    :else (print-summary)))
