(ns grpo-repl
  "Phase 3 (genmlx-oexl), done-means #3: GRPO-sharpen the SFT'd 0.8B proposer with the
   EXACT ORACLE EVIDENCE as the verifiable per-step reward. The validated real-checkpoint
   path (world_train_reward_test Part B): Qwen35Model.load -> with-trainer -> train-step!
   with tr/model-evidence-reward. genmlx-o94r blocks only SHARDED 3GB+ checkpoints; the
   single-file fused 0.8B loads fine.

   PRINCIPLE: GRPO sharpens the SAME decision distribution the SFT learned. The training
   prompts ARE the harvested corpus rows' (system,user) turns (the step-prompts the loop
   visited), grouped by task so each train-step!'s reward matches that task's observations.
   The reward = model-evidence-reward (oracle log marginal likelihood of the policy's
   proposed model vs the task data; floored, coverage-guarded). kl-coef regularizes toward
   the SFT base (Phase-1.5 genmlx-65d5) so GRPO refines rather than drifts.

   LIFECYCLE: the 35B worker MUST be down (32GB; this loads the 0.8B natively + trains).

   Run:  CORPUS=~/genmlx-loop-artifacts/harvest/full-r0-greedy/train_rows.jsonl \\
          MODEL_DIR=~/.cache/models/qwen3.5-0.8b-mlx-bf16-cljs-sft-fused \\
          ROUND=0 INSTANCES=12 EPOCHS=2 \\
          bun run --bun nbb scripts/grpo_repl.cljs
   Env:  CORPUS MODEL_DIR OUT_DIR ROUND INSTANCES EPOCHS STEP_BUDGET GROUP_SIZE
         MAX_COMPLETION LR KL_COEF NP REWARD_FLOOR SEED."
  (:require [genmlx.world.train :as train]
            [genmlx.world.train-reward :as tr]
            [genmlx.world.curriculum :as cur]
            [clojure.string :as str]
            [promesa.core :as p]))

(def os    (js/require "os"))
(def path  (js/require "path"))
(def fs    (js/require "fs"))
(def gcore (js/require "@genmlx/core"))
(defn home [& xs] (apply (.-join path) (.homedir os) xs))
(defn- env  [k d] (or (aget (.-env js/process) k) d))
(defn- envi [k d] (let [v (env k nil)] (if v (js/parseInt v 10) d)))
(defn- envf [k d] (let [v (env k nil)] (if v (js/parseFloat v) d)))
(defn fx [x] (if (and x (js/isFinite x)) (.toFixed (js/Number x) 3) (str x)))

(def corpus-file (env "CORPUS" (home "genmlx-loop-artifacts" "harvest" "full-r0-greedy" "train_rows.jsonl")))
(def model-dir   (env "MODEL_DIR" (home ".cache" "models" "qwen3.5-0.8b-mlx-bf16-cljs-sft-fused")))
(def out-dir     (env "OUT_DIR" (home ".cache" "models" "qwen3.5-0.8b-cljs-sft-grpo")))
(def round       (envi "ROUND" 0))
(def instances   (envi "INSTANCES" 12))
(def epochs      (envi "EPOCHS" 2))
(def step-budget (envi "STEP_BUDGET" 0))          ;; 0 = no cap (epochs * n-tasks steps)
(def group-size  (envi "GROUP_SIZE" 8))
(def max-comp    (envi "MAX_COMPLETION" 220))
(def lr          (envf "LR" 2e-6))
(def kl-coef     (envf "KL_COEF" 0.1))
(def np          (envi "NP" 50))
(def reward-floor (envf "REWARD_FLOOR" -20.0))
(def seed        (envi "SEED" 1))

;; Verified GRPO config (world_train_reward_test Part B trend config).
(def grpo-cfg
  {:learning-rate lr :temperature 0.9 :gradient-clip-norm 0.5
   :kl-coef kl-coef :loss-type :grpo :enable-thinking false
   :lm-head-chunk-size 2 :forward-chunk-size 4
   :group-size group-size :max-completion-length max-comp})

;; ---------------------------------------------------------------------------
;; 1. Build the per-task training prompts from the harvested corpus.
;; ---------------------------------------------------------------------------
(defn- read-jsonl [p]
  (->> (str/split-lines (.readFileSync fs p "utf8"))
       (remove str/blank?)
       (mapv (fn [l] (js->clj (js/JSON.parse l) :keywordize-keys true)))))

(def C (cur/generate-curriculum {:round round :instances-per-family instances}))
(def task-by-id (into {} (map (fn [t] [(name (:id t)) t])) (:tasks C)))

(def rows (read-jsonl corpus-file))
;; A training prompt = the row's (system,user) turns (drop the assistant target — GRPO
;; generates its own completions and scores them by the oracle reward).
(defn- row->prompt [row]
  (->> (:messages row)
       (filter #(#{"system" "user"} (:role %)))
       (mapv #(select-keys % [:role :content]))))

;; Group prompts by task-id (the reward is task-specific, so one task per train-step!).
;; PROMPTS_PER_TASK caps rollouts/step (speed + peak memory): each step does
;; (#prompts * group-size) generations. 1 = just the first decision state per task.
(def prompts-per-task (envi "PROMPTS_PER_TASK" 0))   ;; 0 = all of a task's rows
(def by-task
  (->> rows
       (group-by :task-id)
       (keep (fn [[tid rs]]
               (when-let [task (task-by-id tid)]
                 (let [ps (mapv row->prompt rs)]
                   {:task-id tid :task task
                    :prompts (if (pos? prompts-per-task) (vec (take prompts-per-task ps)) ps)}))))
       vec))

(println (str "\n### GRPO  model=" (.basename path model-dir)))
(println (str "  corpus " corpus-file " -> " (count rows) " rows over " (count by-task) " train tasks"))
(println (str "  cfg: lr=" lr " kl=" kl-coef " group-size=" group-size " max-comp=" max-comp
              " epochs=" epochs (when (pos? step-budget) (str " step-budget=" step-budget))))
(println (str "  out -> " out-dir))
(when (empty? by-task) (println "  ABORT: no usable corpus rows") (js/process.exit 1))
(when-not (.existsSync fs (.join path model-dir "config.json"))
  (println "  ABORT: no config.json at" model-dir "(point MODEL_DIR at the fused SFT model)")
  (js/process.exit 1))

;; ---------------------------------------------------------------------------
;; 2. The GRPO loop — load the model, train, save weights, copy aux files.
;; ---------------------------------------------------------------------------
(def aux-files ["config.json" "tokenizer.json" "tokenizer_config.json"
                "special_tokens_map.json" "vocab.json" "merges.txt"
                "generation_config.json" "added_tokens.json" "chat_template.jinja"])

(defn- copy-aux! [src dst]
  (doseq [f aux-files :let [s (.join path src f)] :when (.existsSync fs s)]
    (.copyFileSync fs s (.join path dst f))))

(-> (p/let [model (.load (.-Qwen35Model gcore) model-dir)
            _     (println "  policy model loaded; starting GRPO ...")
            order (vec (sort-by :task-id by-task))     ;; deterministic; seed varies rollouts
            ;; FLAT step plan: each (epoch, task) pair is ONE train-step! (one task per step,
            ;; so the prompt-agnostic reward matches that task's observations). A single
            ;; p/loop with p/recur (promesa needs p/recur, and recur can't cross a p/let).
            steps-plan (vec (for [ep (range epochs) t order] {:epoch ep :item t}))
            history
            (train/with-trainer model grpo-cfg
              (fn [trainer]
                (p/loop [i 0, hist []]
                  (if (or (= i (count steps-plan))
                          (and (pos? step-budget) (>= i step-budget)))
                    (p/resolved hist)
                    (let [{:keys [epoch item]} (nth steps-plan i)
                          {:keys [task-id task prompts]} item
                          reward-fn (tr/model-evidence-reward
                                     task {:reward-floor reward-floor :n-particles np})]
                      (p/let [m (train/train-step! trainer prompts reward-fn)]
                        (let [rs   (:rewards m)
                              vrat (when (seq rs)
                                     (/ (count (filter #(> % reward-floor) rs)) (double (count rs))))]
                          (println (str "  step" i " ep" epoch " " task-id
                                        " reward-mean=" (fx (:reward-mean m))
                                        " valid-rate=" (fx vrat)
                                        " loss=" (fx (:loss m))
                                        " adv-mean=" (fx (:advantage-mean m))))
                          (p/recur (inc i) (conj hist {:epoch epoch :task-id task-id
                                                       :reward-mean (:reward-mean m)
                                                       :valid-rate vrat :loss (:loss m)})))))))))
            _ (do (.mkdirSync fs out-dir #js {:recursive true})
                  (println "  GRPO done; saving weights ..."))
            _ (.saveModel model out-dir)
            _ (copy-aux! model-dir out-dir)]
      (let [first-rm (->> history (keep :reward-mean) first)
            last-rm  (->> history (keep :reward-mean) last)
            first-vr (->> history (keep :valid-rate) first)
            last-vr  (->> history (keep :valid-rate) last)
            report {:model-dir model-dir :out-dir out-dir :corpus corpus-file
                    :config grpo-cfg :epochs epochs :n-tasks (count by-task)
                    :n-steps (count history)
                    :reward-mean-first first-rm :reward-mean-last last-rm
                    :valid-rate-first first-vr :valid-rate-last last-vr
                    :history history}]
        (.writeFileSync fs (.join path out-dir "grpo_report.json")
                        (js/JSON.stringify (clj->js report) nil 2))
        (println (str "\n### GRPO DONE  " (count history) " steps"))
        (println (str "  reward-mean " (fx first-rm) " -> " (fx last-rm)
                      "   valid-rate " (fx first-vr) " -> " (fx last-vr)))
        (println (str "  saved sharpened model -> " out-dir))
        (println (str "  next: bring up a worker on " out-dir
                      " and run scripts/inloop_eval.cljs (TIER=sft-grpo-0.8b)"))))
    (p/catch (fn [e] (println "  GRPO ERROR:" (.-message e)) (js/process.exit 1))))
