(ns grpo-student
  "genmlx-fqvu: GRPO a REAL student policy with the Bayesian-evidence reward —
   the §9/§10 'close the training loop' run, sized for the students that
   actually fit Thor's 128 GB.

   The construction: the policy LLM writes a GenMLX probabilistic program; the
   reward is the program's Bayesian model evidence against fixed observations
   (world/train_reward.cljs, the Phase-1 seam); GRPO climbs it. This script is
   the REAL-student runner: same validated loop as world_train_reward_test
   Part B, parameterized by checkpoint. Default student = Ornith-1.0-9B-bf16
   (dense qwen3_5, 9.41B). The 35B-A3B MoE trains via the FROZEN-experts
   path (genmlx-n32r): the non-expert stack (~3B params) dequantizes to bf16
   masters while the ~32B of packed experts stay frozen behind gather_qmm —
   full expert dequantize remains arithmetically infeasible (~64 GB masters
   + ~64 GB grads). The model family is auto-detected from config.json.

   OPTIMIZER note (Thor 128 GB): AdamW's fp32 moments are 8 bytes/param —
   ~69 GB for the 9B text stack, peak ~110 GB: inside the dark-page danger
   zone (genmlx-h3p5). Default here is SGD (no optimizer state, peak ~45 GB).
   The 0.8B fits AdamW comfortably (OPTIMIZER=adamw LR=2e-6).

   LIFECYCLE (Thor discipline): ONE GPU process at a time; run through
   ~/genmlx-guarded-run.sh. Metrics stream to OUT_JSONL (line-buffered) so a
   killed run keeps every completed step.

   Run (from repo root):
     MODEL_DIR=~/.cache/huggingface/hub/models--mlx-community--Ornith-1.0-9B-bf16/snapshots/<hash> \\
     STEPS=10 GROUP_SIZE=8 MAX_COMPLETION=220 LR=1e-3 OPTIMIZER=sgd \\
       bunx --bun nbb@1.4.208 scripts/grpo_student.cljs
   Env: MODEL_DIR STEPS GROUP_SIZE MAX_COMPLETION LR OPTIMIZER (sgd|adamw)
        KL_COEF NP REWARD_FLOOR SEED TEMPERATURE OUT_JSONL SAVE_DIR"
  (:require [genmlx.world.train :as train]
            [genmlx.world.train-reward :as tr]
            [clojure.string :as str]
            [promesa.core :as p]))

(def os    (js/require "os"))
(def path  (js/require "path"))
(def fs    (js/require "fs"))
(def gcore (js/require "@genmlx/core"))

(defn- env  [k d] (or (aget (.-env js/process) k) d))
(defn- envi [k d] (let [v (env k nil)] (if v (js/parseInt v 10) d)))
(defn- envf [k d] (let [v (env k nil)] (if v (js/parseFloat v) d)))
(defn- fx [x] (if (and (number? x) (js/isFinite x)) (.toFixed (js/Number x) 3) (str x)))

(defn- model-family
  "Detect the training family from config.json's model_type (genmlx-n32r):
   qwen3_5_moe -> {:loader Qwen35MoeModel :family :qwen35-moe}, else the
   dense default."
  [dir]
  (let [cfg (js/JSON.parse (.readFileSync fs (.join path dir "config.json") "utf8"))
        mt  (or (.-model_type cfg) "qwen3_5")]
    (if (str/starts-with? mt "qwen3_5_moe")
      {:loader (.-Qwen35MoeModel gcore) :family :qwen35-moe :model-type mt}
      {:loader (.-Qwen35Model gcore) :family :qwen35 :model-type mt})))

(def default-9b
  (.join path (.homedir os)
         ".cache" "huggingface" "hub" "models--mlx-community--Ornith-1.0-9B-bf16"
         "snapshots" "5944bc88218397fe6879d5b350b26297673f80a7"))

(def model-dir   (env "MODEL_DIR" default-9b))
(def n-steps     (envi "STEPS" 10))
(def group-size  (envi "GROUP_SIZE" 8))
(def max-comp    (envi "MAX_COMPLETION" 220))
(def optimizer   (env "OPTIMIZER" "sgd"))
(def lr          (envf "LR" (if (= optimizer "sgd") 1e-3 2e-6)))
(def kl-coef     (envf "KL_COEF" 0.0))
(def np          (envi "NP" 50))
(def train-floor (envf "REWARD_FLOOR" -20.0))
(def seed        (envi "SEED" 1))
;; NOT "TEMP" — Bun honors TEMP as its temp-dir
(def temp        (envf "TEMPERATURE" 0.9))
(def out-jsonl   (env "OUT_JSONL" nil))
(def save-dir    (env "SAVE_DIR" nil))
;; Backward-phase memory diet (genmlx-y7qe): both chunk knobs count SEQUENCES
;; per chunk over the batch dim. lm-head 1 = one [1,T,V] f32 logit block live
;; at a time; forward 2 = value_and_grad over 2 sequences per pass with grad
;; accumulation across passes. Defaults sized so group-8/completion-220 on the
;; 9B stays above the 25 GB MemAvailable floor.
(def lm-head-chunk (envi "LM_HEAD_CHUNK" 1))
(def forward-chunk (envi "FORWARD_CHUNK" 2))

(def grpo-cfg
  {:learning-rate lr :temperature temp :gradient-clip-norm 0.5
   :kl-coef kl-coef :loss-type :grpo :enable-thinking false
   :lm-head-chunk-size lm-head-chunk :forward-chunk-size forward-chunk
   :group-size group-size :max-completion-length max-comp :seed seed
   :raw {:optimizerType optimizer}})

(defn- mem-mb []
  (let [s (.readFileSync fs "/proc/meminfo" "utf8")]
    (some-> (re-find #"MemAvailable:\s+(\d+)" s) second (js/parseInt 10) (/ 1024) js/Math.round)))

(defn- log-step! [row]
  (when out-jsonl
    (.appendFileSync fs out-jsonl (str (js/JSON.stringify (clj->js row)) "\n"))))

(defn- mean [xs] (if (seq xs) (/ (reduce + xs) (count xs)) 0.0))

(defn -main []
  (println "== grpo_student (genmlx-fqvu) ==")
  (println "  model     :" model-dir)
  (println "  optimizer :" optimizer " lr:" lr " steps:" n-steps
           " group:" group-size " max-completion:" max-comp " seed:" seed)
  (println "  mem-diet  : lm-head-chunk:" lm-head-chunk " forward-chunk:" forward-chunk)
  (let [t0 (.now js/Date)
        {:keys [loader family model-type]} (model-family model-dir)]
    (println "  family    :" (name family) " (model_type" model-type ")")
    (p/let [model (.load loader model-dir)]
      (println "  loaded in" (js/Math.round (/ (- (.now js/Date) t0) 1000)) "s;"
               "MemAvailable" (mem-mb) "MB")
      (p/let
        [reward-fn (tr/model-evidence-reward tr/gaussian-mean-task
                                             {:reward-floor train-floor :n-particles np})
         prompts   (tr/task->prompts tr/gaussian-mean-task)
         history
         (train/with-trainer model grpo-cfg {:family family}
           (fn [trainer]
             (p/loop [step 0, hist []]
               (if (= step n-steps)
                 (p/resolved hist)
                 (p/let [r (train/train-step! trainer prompts reward-fn)]
                   (let [rs   (:rewards r)
                         vrat (/ (count (filter #(> % train-floor) rs)) (double (count rs)))
                         row  {:step step :reward-mean (:reward-mean r) :valid-rate vrat
                               :loss (:loss r) :applied (:gradients-applied? r)
                               :gen-ms (:generation-ms r) :train-ms (:training-ms r)
                               :peak-mem-mb (:peak-memory-mb r) :mem-avail-mb (mem-mb)
                               :rewards rs}]
                     (log-step! row)
                     (println (str "  step " step
                                   "  reward-mean=" (fx (:reward-mean r))
                                   "  valid-rate=" (fx vrat)
                                   "  loss=" (fx (:loss r))
                                   "  applied=" (:gradients-applied? r)
                                   "  gen=" (js/Math.round (:generation-ms r)) "ms"
                                   "  train=" (js/Math.round (:training-ms r)) "ms"
                                   "  MemAvail=" (mem-mb) "MB")))
                   (p/recur (inc step) (conj hist r)))))))]
        (let [k       3
              means   (map :reward-mean history)
              applied (count (filter :gradients-applied? history))
              first-k (mean (take k means))
              last-k  (mean (take-last k means))]
          (println "\n== summary ==")
          (println "  reward-means:" (mapv #(js/Number (fx %)) means))
          (println (str "  applied " applied "/" n-steps
                        "  mean(first " k ")=" (fx first-k)
                        "  mean(last " k ")=" (fx last-k)
                        (if (> last-k first-k) "  TREND UP" "  no trend")))
          (when save-dir
            (println "  saving weights ->" save-dir)
            (p/do (.saveModel model save-dir)
                  (println "  saved."))))))))

(-> (-main)
    (p/catch (fn [e]
               (println "UNCAUGHT:" (.-message e))
               (println (.-stack e))
               (set! (.-exitCode js/process) 1))))
