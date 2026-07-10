(ns distill-teacher
  "IN-PROCESS teacher generation for the distillation/GRPO loop (genmlx-j0d6 step 2,
   genmlx-fqvu teacher wiring) — the CUDA-native CLJS twin of scripts/distill_teacher.py.

   The python teacher rides mlx-lm, which is METAL-ONLY; on the Thor (CUDA) the j0d6
   recipe said 'use vLLM or transformers'. That external dependency is gone: GenMLX's
   own resident forward (native or owned CLJS, llm/load-model picks) now runs the
   qwen3_5_moe 35B teachers on CUDA, so the teacher batch is a GenMLX script like
   everything else — the whole teacher→oracle→student loop lives in ONE system (the
   closure thesis). The interface contract is unchanged and engine-agnostic: read the
   exported tasks file, write raw_candidates.jsonl; the oracle filter
   (scripts/distill_filter.cljs) cannot tell which engine produced the file.

   Same two roles as the python twin, same prompt rendering rule:
     - TEACHER (corpus build): a strong model over TRAIN tasks -> raw_candidates.jsonl
     - STUDENT (eval): the small student over held-out EVAL tasks -> candidates for
       sft_eval.cljs
   Rendering: generate-text-raw's ChatML build injects the think-skip scaffold for
   qwen3/3.5/3.5-moe families (backend.cljs render-chat), matching the python side's
   enable_thinking=False — the model emits the code form directly, and the oracle
   strips residual <think> defensively anyway.

   LIFECYCLE (Thor discipline): ONE GPU process at a time — run this ALONE (no
   student training in parallel), prefer ~/genmlx-guarded-run.sh for the 35B.
   Candidates append to --out incrementally (line-buffered JSONL), so a killed run
   keeps everything already generated; re-run with --skip-done to resume.

   Run (from repo root):
     bunx --bun nbb@1.4.208 scripts/distill_teacher.cljs -- \\
       --tasks /tmp/genmlx-distill/tasks.jsonl --out /tmp/genmlx-distill/raw_candidates.jsonl \\
       [--model <dir>] [--n 8] [--max-tokens 512] [--temp 0.8] [--limit K] \\
       [--greedy-first] [--skip-done]

   --model defaults to the Ornith-1.0-35B-4bit snapshot (an image-text fine-tune of
   the same Qwen3.5-35B-A3B family as the Ornith-1.0-9B student — teacher and student
   share a lineage, which is the fqvu construction: big Ornith teaches small Ornith)."
  (:require [genmlx.llm.backend :as llm]
            [genmlx.mlx :as mx]
            [clojure.string :as str]
            [promesa.core :as p]))

(def fs   (js/require "fs"))
(def os   (js/require "os"))
(def path (js/require "path"))

(defn- env [k d] (or (aget (.-env js/process) k) d))

(def default-model
  (.join path (.homedir os)
         ".cache" "huggingface" "hub" "models--mlx-community--Ornith-1.0-35B-4bit"
         "snapshots" "781f91090809411b7fc07449817f398a99feb188"))

;; --- tiny CLI: --key value pairs + bare --flags (same shape as distill_filter) ---
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

(def args (parse-args (drop 1 (.slice js/process.argv 2))))

(def tasks-file  (or (:tasks args) (env "TASKS" nil)))
(def out-file    (or (:out args) (env "OUT" nil)))
(def model-dir   (or (:model args) (env "MODEL_DIR" default-model)))
(def n-samples   (js/parseInt (or (:n args) "8") 10))
(def max-tokens  (js/parseInt (or (:max-tokens args) "512") 10))
(def temp        (js/parseFloat (or (:temp args) "0.8")))
(def limit       (some-> (:limit args) (js/parseInt 10)))
(def greedy-first? (boolean (:greedy-first args)))
(def skip-done?    (boolean (:skip-done args)))

(defn- read-jsonl [p]
  (->> (str/split-lines (.readFileSync fs p "utf8"))
       (remove str/blank?)
       (mapv (fn [l] (js->clj (js/JSON.parse l) :keywordize-keys true)))))

(defn- append-jsonl! [p row]
  (.appendFileSync fs p (str (js/JSON.stringify (clj->js row)) "\n")))

(defn- done-keys
  "#{[task_id sample_idx] ...} already present in the out file (for --skip-done)."
  []
  (if (and skip-done? (.existsSync fs out-file))
    (into #{} (map (juxt :task_id :sample_idx)) (read-jsonl out-file))
    #{}))

(defn -main []
  (when-not (and tasks-file out-file)
    (println "usage: distill_teacher.cljs --tasks <tasks.jsonl> --out <raw_candidates.jsonl> [opts]")
    (js/process.exit 1))
  (let [tasks (cond->> (read-jsonl tasks-file) limit (take limit))
        done  (done-keys)
        t0    (.now js/Date)]
    (println "== distill_teacher (GenMLX in-process) ==")
    (println "  model :" model-dir)
    (println "  tasks :" (count tasks) " samples/task:" n-samples
             " temp:" temp " max-tokens:" max-tokens
             (str (when greedy-first? " greedy-first") (when skip-done? " skip-done")))
    (p/let [m (llm/load-model model-dir)]
      (println "  loaded:" (name (:type m)) "in" (js/Math.round (/ (- (.now js/Date) t0) 1000)) "s")
      (p/loop [ts (seq tasks), n-gen 0]
        (if-not ts
          (println (str "== done: " n-gen " new candidates -> " out-file " =="))
          (let [{:keys [task_id system_prompt prompt]} (first ts)]
            (p/let
              [made
               (p/loop [i 0, made 0]
                 (if (= i n-samples)
                   (p/resolved made)
                   (if (contains? done [task_id i])
                     (p/recur (inc i) made)
                     (let [greedy? (and greedy-first? (zero? i))
                           t1 (.now js/Date)]
                       (p/let [text (llm/generate-text-raw
                                     m prompt
                                     {:max-tokens  max-tokens
                                      ;; sample i seeds the PRNG -> a killed+resumed
                                      ;; run regenerates the SAME candidate set
                                      :temperature (if greedy? 0 temp)
                                      :seed        (inc i)
                                      :system-prompt (or system_prompt
                                                         "You are a ClojureScript code generator.")})]
                         (append-jsonl! out-file {:task_id task_id :sample_idx i
                                                  :raw_text text
                                                  :gen_ms (- (.now js/Date) t1)
                                                  :greedy greedy?})
                         (println (str "  " task_id "[" i "] "
                                       (- (.now js/Date) t1) " ms, "
                                       (count text) " chars"))
                         ;; dead decode graphs are dark pages on Tegra (R4 lesson,
                         ;; genmlx-h3p5) — sweep between samples on a 35B teacher
                         (mx/force-gc!)
                         (p/recur (inc i) (inc made)))))))]
              (p/recur (next ts) (+ n-gen made)))))))))

(-> (-main)
    (p/catch (fn [e]
               (println "UNCAUGHT:" (.-message e))
               (println (.-stack e))
               (set! (.-exitCode js/process) 1))))
