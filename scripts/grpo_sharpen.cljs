(ns genmlx.grpo-sharpen
  "GRPO sharpening of the SFT'd cljs-coder student (genmlx-2ctu, loop step 4/5).

   Sharpens an SFT'd qwen3.5-0.8b toward its own pass@k ceiling: RLVR (Yue 2025) lifts
   pass@1 without raising pass@k beyond the SFT base, so this runs AFTER SFT on a band of
   TRAIN tasks where the student SOMETIMES succeeds (the only band with reward variance —
   an all-pass or all-fail group has 0 advantage and teaches nothing). The reward is a
   GROUNDED GFI quantity (the SAME oracle the corpus + held-out eval use): Bayesian model
   evidence for :program, behavioral accuracy for :function. Held-out eval tasks are NEVER
   in the band (leakage), so the lift is measured on tasks the student never trained on.

   Reuses, unchanged: genmlx.world.train (the native GRPO engine membrane), the
   train_reward floor/extraction, and the distill-gen task set.

   ENV
     STEPS         number of GRPO steps (default 4 = smoke)
     BAND_SIZE     train tasks per step (default 8; group-size completions each)
     GROUP_SIZE    completions per prompt (default 6)
     MAXLEN        max completion length (default 128 — fn forms are short)
     LR            learning rate (default 2e-6, the Phase-1-stable value)
     KL            kl-coef to base (default 0.0; >0 = KL-to-base via genmlx-65d5)
     MODEL         student model dir name under ~/.cache/models (default the SFT-600 fuse)
     SAVE_TO       if set, saveModel the sharpened weights to ~/.cache/models/<SAVE_TO>"
  (:require [genmlx.world.train :as train]
            [genmlx.world.train-reward :as reward]
            [genmlx.world.distill-gen :as g]
            [genmlx.llm.msa-score :as score]
            [genmlx.codegen.eval :as ce]
            [clojure.string :as str]
            [promesa.core :as p]))

(def os   (js/require "os"))
(def path (js/require "path"))
(def fs   (js/require "fs"))
(def gcore (js/require "@genmlx/core"))

(defn- env [k d] (or (aget (.-env js/process) k) d))
(defn- envi [k d] (let [v (aget (.-env js/process) k)] (if v (js/parseInt v 10) d)))
(defn- envf [k d] (let [v (aget (.-env js/process) k)] (if v (js/parseFloat v) d)))

(def train-floor -20.0)

;; ---------------------------------------------------------------------------
;; The band — TRAIN tasks only (never held-out eval). A diverse spread across the
;; function families (where the held-out learnable tasks live) plus a couple of
;; programs for a stable always-some-signal anchor.
;; ---------------------------------------------------------------------------

(defn- pick-band
  "MEDIUM-difficulty train FUNCTIONS — the reward-variance sweet spot. Easy functions the
   SFT student always solves (no variance) and hard ones it never solves (all floored) teach
   nothing; medium ones are the band where it SOMETIMES succeeds, the only band GRPO learns
   from. Programs are excluded (the scaffold makes them near-always covering = low variance).
   Deterministic even stride across the family spread."
  [n]
  (let [med (filterv #(and (= :function (:kind %)) (= :medium (:difficulty %))) g/train-tasks)
        stride (max 1 (quot (count med) (max 1 n)))]
    (vec (take n (take-nth stride med)))))

;; ---------------------------------------------------------------------------
;; The reward dispatcher — identify the task from the prompt, apply its oracle.
;; ---------------------------------------------------------------------------

(defn- run-tests [f test-cases]
  (let [tot (count test-cases)
        ok  (count (filter (fn [{:keys [args expected]}]
                             (try (= expected (apply f args)) (catch :default _ false)))
                           test-cases))]
    (/ ok (max 1 tot))))

(defn- first-form
  "Isolate the FIRST complete cljs form from a raw completion and re-emit it, DROPPING any
   trailing junk. The native GRPO engine does not stop at EOS (mlx-lm does), so a completion
   is `(valid form)` + `\\n!\\n!...` filler; eval'ing the whole string errors. parse-form reads
   one form; pr-str re-emits it — preserving a defn NAME (so self-recursion survives, unlike
   extract-program which canonicalizes defn->anonymous fn)."
  [completion]
  (let [raw  (ce/extract-code (reward/strip-think completion))
        form (ce/parse-form raw)]
    (if form (pr-str form) raw)))

(defn- function-reward
  "Behavioral accuracy in [0,1] (floored): garbage -> floor, runnable-but-wrong -> partial,
   correct -> 1. Mirrors the distill gate but returns a scalar reward, not a verdict."
  [task completion floor]
  (let [code (first-form completion)]
    (cond
      (str/blank? code)        floor
      (not (ce/valid-cljs? code)) floor
      (seq (:transitions task))
      (let [{:keys [accuracy error]} (ce/verify-transition-fn code (:transitions task))]
        (if error floor accuracy))
      (seq (:test-cases task))
      (let [{f :fn err :error} (ce/eval-fn code)]
        (if err floor (run-tests f (:test-cases task))))
      :else floor)))

(defn- program-reward [task completion floor]
  (let [code (reward/extract-program completion)
        gf   (score/eval-model code)]
    (if (or (nil? gf) (not (reward/covered-observations? gf (:observations task))))
      floor
      (reward/clamp-floor floor (score/score-model gf (:observations task) {:n-particles 30})))))

(defn- make-reward [band floor]
  (let [by-prompt (into {} (map (juxt :prompt identity)) band)]
    (fn [prompt completion]
      (try
        (let [user (some #(when (= :user (:role %)) (:content %)) prompt)
              task (get by-prompt user)]
          (cond
            (nil? task)              floor
            (= :program (:kind task)) (program-reward task completion floor)
            :else                     (reward/clamp-floor floor (function-reward task completion floor))))
        (catch :default _ floor)))))

;; ---------------------------------------------------------------------------

(defn- mean [xs] (if (seq xs) (/ (reduce + xs) (count xs)) 0.0))
(defn- fx [x] (.toFixed (js/Number x) 3))

(defn run! []
  (let [steps  (envi "STEPS" 4)
        bsize  (envi "BAND_SIZE" 8)
        gsize  (envi "GROUP_SIZE" 6)
        maxlen (envi "MAXLEN" 128)
        lr     (envf "LR" 2e-6)
        kl     (envf "KL" 0.0)
        model-name (env "MODEL" "qwen3.5-0.8b-cljs-sft600")
        save-to (aget (.-env js/process) "SAVE_TO")
        model-dir (.join path (.homedir os) ".cache" "models" model-name)
        plog-path (env "PROGRESS" "/tmp/genmlx-loop/grpo_progress.log")
        plog   (fn [s] (.appendFileSync fs plog-path (str s "\n")))
        band   (pick-band bsize)
        prompts (mapv #(into [] (cond-> []
                                  (:system-prompt %) (conj {:role :system :content (:system-prompt %)})
                                  true (conj {:role :user :content (:prompt %)})))
                      band)
        reward-fn (make-reward band train-floor)
        cfg    {:learning-rate lr :temperature 0.9 :gradient-clip-norm 0.5
                :kl-coef kl :loss-type :grpo :enable-thinking false
                :group-size gsize :max-completion-length maxlen
                :lm-head-chunk-size 2 :forward-chunk-size 4}]
    (println (str "GRPO sharpen: model=" model-name " steps=" steps " band=" (count band)
                  " group=" gsize " maxlen=" maxlen " lr=" lr " kl=" kl))
    (println (str "  band tasks: " (str/join ", " (map :id band))))
    (cond
      (not (.existsSync fs (.join path model-dir "tokenizer.json")))
      (do (println (str "  ABORT — no model at " model-dir)) (set! (.-exitCode js/process) 1) (p/resolved nil))

      ;; DIAGNOSE: generate one completion per band task and print it raw + its reward,
      ;; to see what the native engine actually emits (format vs capability).
      (aget (.-env js/process) "DIAGNOSE")
      (p/let [model (.load (.-Qwen35Model gcore) model-dir)
              _ (println "  policy model loaded:" (some? model))
              comps (train/with-trainer model cfg
                      (fn [trainer] (train/generate-batch trainer (vec (take 4 prompts)))))]
        (doseq [[i c] (map-indexed vector comps)]
          (let [task (nth band i)
                rw   (reward-fn (nth prompts i) c)]
            (println (str "\n=== [" i "] " (:id task) "  reward=" (fx rw) " ==="))
            (println (str "RAW (" (count c) " chars): " (pr-str (subs c 0 (min 500 (count c))))))))
        (p/resolved nil))

      :else
      (p/let [model (.load (.-Qwen35Model gcore) model-dir)
              _ (println "  policy model loaded:" (some? model))
              history (train/with-trainer model cfg
                        (fn [trainer]
                          (p/loop [step 0, hist []]
                            (if (= step steps)
                              (p/resolved hist)
                              (p/let [r (train/train-step! trainer prompts reward-fn)]
                                (let [rs (:rewards r)
                                      valid (/ (count (filter #(> % train-floor) rs)) (double (count rs)))
                                      ;; per-prompt reward variance present (= learnable groups)
                                      groups (partition gsize rs)
                                      varied (count (filter #(> (apply max %) (apply min %)) groups))
                                      line (str "    step " step
                                                "  reward-mean=" (fx (:reward-mean r))
                                                "  valid-rate=" (fx valid)
                                                "  learnable-groups=" varied "/" (count groups)
                                                "  applied=" (:gradients-applied? r))]
                                  (println line)
                                  (plog line))
                                (p/recur (inc step) (conj hist r)))))))]
        (let [means (map :reward-mean history)
              applied (count (filter :gradients-applied? history))]
          (println (str "\n  reward-means: " (mapv #(js/Number (fx %)) means)))
          (println (str "  steps that applied gradients: " applied "/" steps))
          (when (>= (count means) 2)
            (println (str "  trend: first=" (fx (first means)) " -> last=" (fx (last means))
                          "  (" (if (> (last means) (first means)) "UP" "flat/down") ")")))
          (if save-to
            (p/let [out (.join path (.homedir os) ".cache" "models" save-to)
                    _   (.saveModel model out)]
              (println (str "  saved sharpened weights -> " out)))
            (p/resolved (println "  (no SAVE_TO — weights not persisted)"))))))))

(-> (p/resolved nil)
    (p/then run!)
    (p/catch (fn [e] (println "ERROR:" (.-message e)) (println (.-stack e))
               (set! (.-exitCode js/process) 1))))
