(ns genmlx.sft-eval
  "Held-out evaluation runner for the cljs-coder SFT step (genmlx-o8w9). The thin I/O
   shell that grades a BASELINE student and an SFT'd student on the held-out eval tasks
   with the EXACT j0d6 oracle, then reports the lift.

   It reuses, unchanged:
     - the oracle gate ladder    genmlx.world.distill/evaluate-candidate
     - the timeout/process sandbox genmlx.world.distill-sandbox/collect-verdicts
       (student code is untrusted — a non-terminating completion is killed and recorded
        :timeout, exactly as for teacher candidates)
     - the in-tree tasks          genmlx.world.distill-tasks (the eval tasks live here;
       they are held out from TRAINING, never from the oracle)
     - the pure report core       genmlx.world.sft/eval-report (pass@k + cold-start)

   USAGE
     bun run --bun nbb scripts/sft_eval.cljs \\
       --baseline <baseline_candidates.jsonl> --sft <sft_candidates.jsonl> \\
       --out <dir> [--k 4] [--n-particles 50] [--timeout-ms 15000]

   Each candidates file holds the student's completions for the held-out EVAL tasks
   (produced by scripts/distill_teacher.py pointed at the student model, --greedy-first
   so sample_idx 0 is the greedy/temperature-0 sample and 1.. are temperature-sampled):
     {task_id, sample_idx, raw_text}

   Writes <dir>/{eval_report.json}, and prints a baseline-vs-SFT pass@1/pass@k table
   with a per-kind breakdown and a cold-start flag (baseline pass@k == 0 → unreachable;
   neither SFT nor GRPO can lift it, so it bounds the achievable ceiling)."
  (:require [genmlx.world.sft :as sft]
            [genmlx.world.distill-tasks :as t]
            [genmlx.world.distill-gen :as g]
            [genmlx.world.distill-sandbox :as sb]
            [clojure.string :as str]
            [promesa.core :as p]))

(def fs (js/require "fs"))
(def path-mod (js/require "path"))

(defn- ensure-dir [dir]
  (when-not (.existsSync fs dir) (.mkdirSync fs dir #js {:recursive true})))

(defn- default-out-dir []
  (.join path-mod (or (.. js/process -env -TMPDIR) "/tmp") "genmlx-sft"))

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

(defn- pos-int-or [s d]
  (if (string? s) (let [n (js/parseInt s 10)] (if (or (js/isNaN n) (< n 1)) d n)) d))

(defn- float-or-nil [s]
  (when (string? s) (let [x (js/parseFloat s)] (when-not (js/isNaN x) x))))

(defn- write-json [path data]
  (.writeFileSync fs path (js/JSON.stringify (clj->js data) nil 2)))

(defn- fmt [x] (.toFixed (js/Number x) 3))

(defn- grade
  "Sandbox-grade one candidates file (a student's eval completions). Returns a promise
   of the verdict vector."
  [label candidates out-dir eval-opts timeout-ms]
  (println (str "  grading " label " (" candidates ") ..."))
  (sb/collect-verdicts candidates
                       {:out-path   (.join path-mod out-dir (str "_eval_" label ".edn"))
                        :eval-opts  eval-opts
                        :timeout-ms timeout-ms :poll-ms 400 :verbose? false}))

(defn- print-table [report]
  (let [k (:k report)]
    (println (str "\n== held-out eval: baseline vs SFT (pass@1 greedy / pass@" k ") =="))
    (println (str (.padEnd "task" 26) (.padEnd "kind" 10)
                  (.padEnd "base p@1" 10) (.padEnd "sft p@1" 10)
                  (.padEnd (str "base p@" k) 10) (.padEnd (str "sft p@" k) 10) "cold?"))
    (println (str "  (base-cold = baseline pass@" k " 0; sft-cold = SFT pass@" k " 0 → GRPO-unreachable)"))
    (doseq [r (:tasks report)]
      (println (str (.padEnd (str (:task-id r)) 26) (.padEnd (str (:kind r)) 10)
                    (.padEnd (fmt (:pass1-greedy (:baseline r))) 10)
                    (.padEnd (fmt (:pass1-greedy (:sft r))) 10)
                    (.padEnd (fmt (:passk (:baseline r))) 10)
                    (.padEnd (fmt (:passk (:sft r))) 10)
                    (str (when (:cold-start? r) "base-cold ")
                         (when (:sft-cold? r) "SFT-COLD")))))
    (let [a (:aggregate report)]
      (println (str "\n  aggregate pass@1: baseline " (fmt (:baseline-pass1 a))
                    " -> SFT " (fmt (:sft-pass1 a))
                    "   (Δ " (fmt (:delta-pass1 a)) ")"))
      (println (str "  aggregate pass@" k ": baseline " (fmt (:baseline-passk a))
                    " -> SFT " (fmt (:sft-passk a))
                    "   (Δ " (fmt (:delta-passk a)) ")"))
      (println (str "  base-cold (baseline pass@" k " = 0): " (:n-cold-start a) "/" (:n-tasks a)
                    "  — SFT must generalize from train-task demos to lift these"))
      (println (str "  SFT-cold  (SFT pass@" k " = 0): " (:n-sft-cold a) "/" (:n-tasks a)
                    "  — the GRPO step (genmlx-2ctu) has no reward signal here; this bounds the loop's ceiling")))
    (println "\n  per-kind:")
    (doseq [[kind m] (:by-kind report)]
      (println (str "    " (.padEnd (str kind) 10)
                    "pass@1 " (fmt (:baseline-pass1 m)) "->" (fmt (:sft-pass1 m))
                    "  pass@" k " " (fmt (:baseline-passk m)) "->" (fmt (:sft-passk m)))))))

(defn- warn-non-eval!
  "Symmetric to the corpus-side assert-train-disjoint!: warn loudly if any graded
   candidate references a NON-held-out (train) task — its pass@k is then not a held-out
   measurement and would inflate the reported lift (genmlx-o8w9 review). `eval?` is the
   held-out predicate on a task-id (the seed set's or the scaled distill-gen set's)."
  [verdicts eval?]
  (let [non-eval (distinct (remove eval? (map :task-id verdicts)))]
    (when (seq non-eval)
      (println (str "  WARNING: graded candidates reference NON-held-out task(s): "
                    (str/join ", " non-eval)
                    "\n           these are TRAIN tasks — their pass@k is NOT a held-out"
                    " measurement (did you point --baseline/--sft at the wrong file? use"
                    " sft_prep --export-eval-tasks).")))))

(defn- run! [{:keys [baseline sft out k n-particles timeout-ms min-log-ml gen]}]
  (let [tasks-by-id (if gen g/tasks-by-id t/tasks-by-id)
        eval?       (if gen #(contains? g/eval-task-ids %) sft/eval-task?)
        out-dir (or out (default-out-dir))
        kk      (pos-int-or k 4)
        n-part  (pos-int-or n-particles 50)
        tmo     (pos-int-or timeout-ms 15000)
        ;; thread the SAME model-evidence floor the corpus was filtered with, so eval
        ;; grading and corpus validation use the identical oracle gate (genmlx-o8w9 review).
        min-ml  (float-or-nil min-log-ml)
        eopts   (cond-> {:n-particles n-part} min-ml (assoc :min-log-ml min-ml))]
    (ensure-dir out-dir)
    (when min-ml (println (str "  applying model-evidence floor --min-log-ml " min-ml
                               " (must match the corpus build)")))
    (p/let [b-verdicts (grade "baseline" baseline out-dir eopts tmo)
            s-verdicts (grade "sft" sft out-dir eopts tmo)]
      (warn-non-eval! (concat b-verdicts s-verdicts))
      (let [report (assoc (sft/eval-report b-verdicts s-verdicts kk t/tasks-by-id)
                          :min-log-ml min-ml :n-particles n-part)]
        (write-json (.join path-mod out-dir "eval_report.json") report)
        (print-table report)
        (println (str "\nWrote " (.join path-mod out-dir "eval_report.json")))))))

;; ---------------------------------------------------------------------------

(let [opts (parse-args (vec (drop 2 (.-argv js/process))))]
  (if (and (:baseline opts) (:sft opts))
    (-> (p/resolved nil)
        (p/then (fn [_] (run! opts)))
        (p/catch (fn [e] (println "ERROR:" (.-message e))
                   (set! (.-exitCode js/process) 1))))
    (do (println "usage:")
        (println "  --baseline <baseline_candidates.jsonl> --sft <sft_candidates.jsonl> --out <dir> [--k 4] [--n-particles 50] [--timeout-ms 15000] [--min-log-ml F]")
        (set! (.-exitCode js/process) 1))))
