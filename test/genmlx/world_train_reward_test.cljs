;; @tier slow
(ns genmlx.world-train-reward-test
  "Phase-1 acceptance (bean genmlx-ugkv) for genmlx.world.train-reward — GRPO where
   the REWARD is a GFI quantity (Bayesian model evidence).

   TWO PARTS, one serial process (run: `bun run --bun nbb test/genmlx/world_train_reward_test.cljs`):

   PART A — CORRECTNESS GATES (deterministic, no policy LLM; uses GenMLX inference).
     The reward builders are exercised on CRAFTED completions:
       * a valid model scores its Bayesian evidence, verified against an INDEPENDENT
         closed-form gaussian-gaussian marginal oracle (NOT score-model — no circular
         test) and an independent importance-sampling oracle;
       * invalid / non-function / data-ignoring completions score the FINITE floor,
         never ##-Inf (the GRPO-poison guard);
       * the reward-integrity coverage guard (a latent-only program that ignores the
         data would otherwise beat a correct model with weight 0);
       * reward purity (pure (prompt, completion) -> number, prompt-insensitive,
         deterministic, never touches a policy model);
       * transition-fn-reward (lightly), extract-program robustness, the shaping knobs.

   PART B — THE HEADLINE (slow, GPU, serial). Load the REAL qwen3.5-0.8b policy, run
     N KL-free GRPO steps whose reward is msa/score-model log-evidence through the
     Phase-0 train-step! bridge, and assert the loop closes, every reward is finite &
     >= floor, and reward-mean TRENDS UP (mean(last k) > mean(first k)) — the policy
     demonstrably learns to write better-fitting probabilistic programs. Skips cleanly
     when the model is absent."
  (:require [genmlx.world.train-reward :as tr]
            [genmlx.world.train :as train]
            [genmlx.llm.msa-score :as msa]
            [promesa.core :as p]
            [clojure.string :as str]))

;; Node builtins + the native core handle are needed only by Part B (the real
;; GRPO run). They are require'd lazily there, NOT at top level, so the Part-A
;; correctness gates load and run without touching the policy-model native path.

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))

(defn assert-true [label v]
  (if v
    (do (swap! pass inc) (println "  PASS" label))
    (do (swap! fail inc) (println "  FAIL" label))))

(defn assert-close [label expected actual tol]
  (let [ok (and (tr/finite? actual) (<= (js/Math.abs (- expected actual)) tol))]
    (if ok
      (do (swap! pass inc) (println "  PASS" label (str "(" actual " ~ " expected ")")))
      (do (swap! fail inc) (println "  FAIL" label (str "(" actual " vs " expected " tol " tol ")"))))))

;; ===========================================================================
;; Independent oracle: closed-form gaussian-gaussian marginal log-evidence.
;;   mu ~ N(m0, s0^2),  y_i | mu ~ N(mu, sn^2) iid.
;;   y ~ N(m0*1, sn^2 I + s0^2 11^T)  =>  via Sherman-Morrison:
;;   log p(y) = -n/2 log 2pi - 1/2[log(sn^2+n s0^2)+(n-1)log sn^2]
;;              - 1/2 (1/sn^2)[ sum d_i^2 - s0^2/(sn^2+n s0^2) (sum d_i)^2 ],  d=y-m0.
;; This is DERIVED independently of GenMLX (the function under test).
;; ===========================================================================

(defn gaussian-gaussian-marginal [ys m0 s0 sn]
  (let [n   (count ys)
        s0² (* s0 s0)
        sn² (* sn sn)
        ds  (map #(- % m0) ys)
        sd  (reduce + ds)
        sd² (reduce + (map #(* % %) ds))
        denom (+ sn² (* n s0²))
        logdet (+ (js/Math.log denom) (* (dec n) (js/Math.log sn²)))
        quad   (* (/ 1.0 sn²) (- sd² (* (/ s0² denom) (* sd sd))))]
    (- (* -0.5 n (js/Math.log (* 2 js/Math.PI)))
       (* 0.5 logdet)
       (* 0.5 quad))))

;; A crafted, VALID model completion (shared-mean gaussian) wrapped in noise the
;; extractor must peel: a <think> block, a markdown fence, and trailing prose.
(def ^:private valid-model-code
  "(fn [trace] (let [mu (trace :mu (dist/gaussian 0 5))] {:y0 (trace :y0 (dist/gaussian mu 1)) :y1 (trace :y1 (dist/gaussian mu 1)) :y2 (trace :y2 (dist/gaussian mu 1)) :y3 (trace :y3 (dist/gaussian mu 1))}))")

(def ^:private noisy-valid-completion
  (str "<think>\nThe user wants a shared-mean model. (this paren ( fools naive extractors)\n</think>\n\n"
       "```clojure\n" valid-model-code "\n```\n\nThis model uses a shared latent mean."))

;; A program that declares a latent but IGNORES the data (traces none of :y0..:y3).
(def ^:private data-ignoring-completion
  "(fn [trace] (let [mu (trace :mu (dist/gaussian 0 5))] {:result mu}))")

;; A program that covers only SOME observed addresses.
(def ^:private partial-coverage-completion
  "(fn [trace] (let [mu (trace :mu (dist/gaussian 0 5))] {:y0 (trace :y0 (dist/gaussian mu 1))}))")

(def ^:private obs (:observations tr/gaussian-mean-task))

;; ===========================================================================
;; PART A — correctness gates
;; ===========================================================================

(defn part-a []
  (println "\n== PART A: reward correctness gates (no policy LLM) ==")

  ;; ---- pure helpers ----
  (assert-true "finite? rejects ##-Inf/NaN/nil, accepts a number"
               (and (tr/finite? 1.0) (not (tr/finite? ##-Inf))
                    (not (tr/finite? js/NaN)) (not (tr/finite? nil))))
  (assert-true "clamp-floor maps non-finite to the floor"
               (= -100.0 (tr/clamp-floor -100.0 ##-Inf)))
  (assert-true "clamp-floor maps below-floor to the floor"
               (= -100.0 (tr/clamp-floor -100.0 -250.0)))
  (assert-true "clamp-floor passes a finite above-floor value through"
               (= -6.0 (tr/clamp-floor -100.0 -6.0)))

  ;; ---- extract-program robustness ----
  (assert-true "extract-program peels <think>, fence and trailing prose to the fn form"
               (let [c (tr/extract-program noisy-valid-completion)]
                 (and (str/starts-with? c "(fn") (str/includes? c ":y3"))))
  (assert-true "extract-program normalizes a (defn name [..] ..) to a fn form"
               (str/starts-with?
                 (tr/extract-program "(defn model [trace] (trace :y0 (dist/gaussian 0 1)))")
                 "(fn"))
  (assert-true "extract-program on garbage returns a non-fn string (-> floor downstream)"
               (not (str/starts-with? (str/trim (tr/extract-program "hello world no code"))
                                      "(fn")))
  (assert-true "completion->gf builds a GF from a valid (noisy) completion"
               (some? (tr/completion->gf noisy-valid-completion)))
  (assert-true "completion->gf returns nil on garbage"
               (nil? (tr/completion->gf "not code at all")))

  ;; ---- the headline: evidence reward matches the independent closed-form oracle ----
  (let [reward     (tr/model-evidence-reward tr/gaussian-mean-task {:n-particles 400})
        ys         (map obs [:y0 :y1 :y2 :y3])
        oracle     (gaussian-gaussian-marginal ys 0.0 5.0 1.0)
        r-valid    (reward "ignored-prompt" noisy-valid-completion)
        ;; second independent oracle: direct importance-sampling estimate
        gf         (msa/eval-model valid-model-code)
        is-oracle  (msa/score-model gf obs {:n-particles 800})
        method     (:method (msa/score-model* gf obs))]
    (println "    closed-form marginal =" oracle "| reward =" r-valid
             "| IS oracle =" is-oracle "| score method =" method)
    (assert-true "evidence reward is finite and above the floor"
                 (and (tr/finite? r-valid) (> r-valid -100.0)))
    (assert-close "evidence reward ~ closed-form gaussian-gaussian marginal (independent oracle)"
                  oracle r-valid 1.0)
    (assert-close "closed-form oracle ~ independent IS estimate (cross-check)"
                  oracle is-oracle 1.5))

  ;; ---- invalid / non-function completions -> finite floor (no ##-Inf) ----
  (let [reward (tr/model-evidence-reward tr/gaussian-mean-task)]
    (assert-true "unparseable completion -> exactly the finite floor"
                 (= -100.0 (reward "p" "this is not clojure (((")))
    (assert-true "a completion that evals to a NON-function -> floor"
                 (= -100.0 (reward "p" "42")))
    (assert-true "empty completion -> floor"
                 (= -100.0 (reward "p" "")))
    (assert-true "no reward is ever ##-Inf or NaN (GRPO-poison guard)"
                 (every? tr/finite?
                         (map #(reward "p" %)
                              [noisy-valid-completion "garbage" "42" "" data-ignoring-completion]))))

  ;; ---- reward-integrity: the coverage guard ----
  (let [guarded   (tr/model-evidence-reward tr/gaussian-mean-task)
        unguarded (tr/model-evidence-reward tr/gaussian-mean-task {:require-coverage? false})]
    (assert-true "coverage guard: a data-IGNORING program scores the floor (not a hackable 0)"
                 (= -100.0 (guarded "p" data-ignoring-completion)))
    (assert-true "coverage guard: a PARTIAL-coverage program scores the floor"
                 (= -100.0 (guarded "p" partial-coverage-completion)))
    ;; demonstrate WHY the guard exists: without it the data-ignoring program scores
    ;; weight ~ 0, which BEATS a correct model's (negative) evidence — reward hacking.
    (let [hack (unguarded "p" data-ignoring-completion)
          good (unguarded "p" noisy-valid-completion)]
      (println "    without guard: data-ignoring =" hack " correct =" good)
      (assert-true "WITHOUT the guard the data-ignoring program out-scores the correct one (the hole the guard closes)"
                   (> hack good))))

  ;; ---- reward purity ----
  (let [reward (tr/model-evidence-reward tr/gaussian-mean-task {:n-particles 1})]
    (assert-true "reward is prompt-insensitive (depends only on the completion)"
                 (= (reward "prompt-A" data-ignoring-completion)
                    (reward "totally-different-prompt-B" data-ignoring-completion)))
    (assert-true "reward is deterministic on the floor path (same completion -> same value)"
                 (= (reward "p" "garbage") (reward "p" "garbage")))
    (assert-true "reward-fn is a 2-arg pure fn (no model handle in its closure)"
                 (fn? reward)))

  ;; ---- on-error observability (genmlx-oi07) ----
  ;; A THROWN exception inside the reward path (as opposed to an expected bad
  ;; candidate, which floors via nil-gf/coverage WITHOUT throwing) must floor
  ;; AND be reported, so a systematic scorer fault is distinguishable from
  ;; reward saturation in a training run log.
  (let [seen   (atom [])
        hooked (tr/model-evidence-reward tr/gaussian-mean-task
                                         {:on-error (fn [kind e]
                                                      (swap! seen conj [kind (ex-message e)]))})
        silent (tr/model-evidence-reward tr/gaussian-mean-task {:on-error nil})]
    (with-redefs [msa/score-model (fn [& _] (throw (ex-info "injected scorer fault" {})))]
      (assert-true "a thrown scorer fault still floors (never throws into the training step)"
                   (= -100.0 (hooked "p" noisy-valid-completion)))
      (assert-true "the :on-error hook saw the swallowed fault (kind + message)"
                   (= [[:model-evidence "injected scorer fault"]] @seen))
      (assert-true ":on-error nil silences reporting without changing the floor"
                   (= -100.0 (silent "p" noisy-valid-completion)))))

  ;; ---- shaping knobs ----
  (let [gf (tr/completion->gf valid-model-code)]
    (assert-true "compilation-level reports a keyword for a valid model"
                 (keyword? (tr/compilation-level gf)))
    (assert-true "compilation-bonus is a non-negative integer"
                 (and (integer? (tr/compilation-bonus gf)) (>= (tr/compilation-bonus gf) 0)))
    (assert-true "form-size counts nodes (a bigger form scores larger)"
                 (> (tr/form-size '(fn [trace] (let [a (trace :a (dist/gaussian 0 1))] {:a a})))
                    (tr/form-size '(fn [x] x))))
    (let [base    ((tr/model-evidence-reward tr/gaussian-mean-task {:n-particles 200}) "p" valid-model-code)
          with-λk ((tr/model-evidence-reward tr/gaussian-mean-task {:n-particles 200 :lambda-k 1.0}) "p" valid-model-code)]
      (assert-true "lambda-k complexity penalty lowers the reward (Occam knob)"
                   (< with-λk base))))

  ;; ---- transition-fn-reward (lightly) ----
  (let [transitions [{:state {:n 0} :action :inc :expected {:n 1}}
                     {:state {:n 5} :action :inc :expected {:n 6}}
                     {:state {:n 3} :action :dec :expected {:n 2}}]
        reward (tr/transition-fn-reward {:transitions transitions})
        correct "(fn [state action] (case action :inc (update state :n inc) :dec (update state :n dec) state))"
        wrong   "(fn [state action] state)"
        garbage "(((not code"]
    (assert-true "transition reward: a correct fn scores accuracy 1.0"
                 (= 1.0 (reward "p" correct)))
    (assert-true "transition reward: a runnable-but-WRONG fn scores 0.0 (finite, above floor)"
                 (= 0.0 (reward "p" wrong)))
    (assert-true "transition reward: an un-evaluable fn scores the floor"
                 (= -100.0 (reward "p" garbage)))
    (assert-true "transition reward orders garbage < wrong < correct (a GRPO gradient)"
                 (< (reward "p" garbage) (reward "p" wrong) (reward "p" correct)))))

;; ===========================================================================
;; PART B — the headline real-model GRPO trend
;; ===========================================================================

;; The FULL 10-step GRPO trend (~25 min) exceeds every CI tier cap (slow = 600s), so
;; it is OPT-IN via PHASE1_FULL_TREND=1. It was verified manually 2026-06-20:
;; reward-mean -13.4 -> -9.0, valid-rate 0.50 -> 0.88, mean(last 3) > mean(first 3).
;; The DEFAULT in-suite run is a SHORT GRPO smoke (fits the cap) that drives the SAME
;; reward bridge through the REAL model and asserts the loop MECHANICS deterministically
;; (loop closes, every reward finite & >= floor). The learning TREND is the full-mode claim.
(def ^:private full-trend? (= "1" (.. js/process -env -PHASE1_FULL_TREND)))
(def ^:private n-steps (if full-trend? 10 2))
(def ^:private trend-k 3)           ; compare mean(first k) vs mean(last k)
;; Training-time floor: a MODEST gap above the valid range (~-5..-11) instead of the
;; -100.0 default. GRPO advantages scale with the reward spread; the full -100 gap
;; produced huge advantages that destabilized the 0.8B in one step (valid-rate
;; 0.75 -> 0.13). -20 keeps a clear valid/invalid contrast with a stable gradient.
;; (Still finite — the only hard requirement; the floor value is a documented knob.)
(def ^:private train-floor -20.0)
(def ^:private grpo-cfg
  (merge {:learning-rate 2e-6 :temperature 0.9 :gradient-clip-norm 0.5
          :kl-coef 0.0 :loss-type :grpo
          ;; thinking OFF so the 0.8B emits the code form directly (not a truncated
          ;; reasoning block); chunk sizes cap peak memory for the 248k-vocab model.
          ;; (repetition-penalty left at the native 1.1 default — higher hurts code.)
          :enable-thinking false :lm-head-chunk-size 2 :forward-chunk-size 4}
         (if full-trend?
           {:group-size 8 :max-completion-length 220}   ; the verified trend config
           {:group-size 4 :max-completion-length 96})))  ; short smoke, fits the cap

(defn- mean [xs] (if (seq xs) (/ (reduce + xs) (count xs)) 0.0))

(defn part-b []
  (println "\n== PART B: real qwen3.5-0.8b GRPO trend (Bayesian evidence as the reward) ==")
  (let [os    (js/require "os")
        path  (js/require "path")
        fs    (js/require "fs")
        gcore (js/require "@genmlx/core")
        model-dir (.join path (.homedir os) ".cache" "models" "qwen3.5-0.8b-mlx-bf16")]
   (cond
    ;; PHASE1_SKIP_TRAIN=1 runs just the fast Part-A correctness gates (no GPU train).
    (= "1" (.. js/process -env -PHASE1_SKIP_TRAIN))
    (do (println "  SKIP real GRPO trend — PHASE1_SKIP_TRAIN=1") (p/resolved nil))

    (not (.existsSync fs (.join path model-dir "tokenizer.json")))
    (do (println "  SKIP real GRPO trend — no qwen3.5-0.8b at" model-dir)
        (p/resolved nil))

    :else
    (p/let [model     (.load (.-Qwen35Model gcore) model-dir)
            _         (assert-true "qwen3.5-0.8b policy model loads" (some? model))
            reward-fn (tr/model-evidence-reward tr/gaussian-mean-task {:reward-floor train-floor})
            prompts   (tr/task->prompts tr/gaussian-mean-task)
            history   (train/with-trainer model grpo-cfg
                        (fn [trainer]
                          (p/loop [step 0, hist []]
                            (if (= step n-steps)
                              (p/resolved hist)
                              (p/let [r (train/train-step! trainer prompts reward-fn)]
                                (let [rs   (:rewards r)
                                      vrat (/ (count (filter #(> % train-floor) rs)) (double (count rs)))]
                                  (println (str "    step " step
                                                "  reward-mean=" (.toFixed (:reward-mean r) 3)
                                                "  valid-rate=" (.toFixed vrat 2)
                                                "  applied=" (:gradients-applied? r))))
                                (p/recur (inc step) (conj hist r)))))))]
      (let [means    (map :reward-mean history)
            applied  (filter :gradients-applied? history)
            all-rs   (mapcat :rewards history)
            first-k  (mean (take trend-k means))
            last-k   (mean (take-last trend-k means))]
        (println "    reward-means:" (mapv #(js/Number (.toFixed % 3)) means))
        ;; --- deterministic loop mechanics (BOTH modes): the reward bridge drives a
        ;;     REAL GRPO step and every reward is a finite, floored GFI quantity. ---
        (assert-true (str "the GRPO loop closed: all " n-steps " steps returned metrics with a finite reward-mean")
                     (and (= n-steps (count history)) (every? #(tr/finite? (:reward-mean %)) history)))
        (assert-true "every reward is finite (no ##-Inf/NaN poisoned the group)"
                     (every? tr/finite? all-rs))
        (assert-true (str "every reward respects the finite floor (>= " train-floor ")")
                     (every? #(>= % train-floor) all-rs))
        (assert-true "reward bridge invoked once per completion across all steps"
                     (= (* n-steps (:group-size grpo-cfg)) (count all-rs)))
        ;; --- the headline LEARNING claim: only asserted under the full opt-in run
        ;;     (the 0.8B trend needs ~10 steps; a 3-step smoke is too short/noisy). ---
        (if full-trend?
          (do (println (str "    mean(first " trend-k ")=" (.toFixed first-k 3)
                            "  mean(last " trend-k ")=" (.toFixed last-k 3)))
              (assert-true "at least half the steps applied gradients (the weight-update effect fired)"
                           (>= (count applied) (quot n-steps 2)))
              (assert-true (str "reward-mean TRENDS UP: mean(last " trend-k ") > mean(first " trend-k ") "
                                "(policy learns to write better-fitting programs)")
                           (> last-k first-k)))
          (do
            ;; genmlx-li1p regression guard: the engine once cached garbage
            ;; old-logprobs (unnormalized vocab-id-0 logit) and recomputed
            ;; completion logprobs off by one — every real-checkpoint step
            ;; NaN'd and SKIPPED silently (gradients-applied?=false), and this
            ;; smoke stayed green because nothing asserted the apply. The one
            ;; honest signal that the weight-update effect actually fired:
            (assert-true "every smoke step APPLIED gradients (genmlx-li1p: a skip means NaN grads / garbage ratios)"
                         (= n-steps (count applied)))
            (println (str "    [smoke] strict reward-mean trend + gradient-progress asserted only under "
                          "PHASE1_FULL_TREND=1 (verified manually 2026-06-20: -13.4 -> -9.0, valid-rate 0.50 -> 0.88); "
                          "applied=" (count applied) "/" n-steps)))))))))

;; ===========================================================================

(defn- summary []
  (println (str "\n== world.train-reward Phase 1: " @pass " passed, " @fail " failed =="))
  (when (pos? @fail) (set! (.-exitCode js/process) 1)))

(-> (p/do (part-a) (part-b))
    (p/then (fn [_] (summary)))
    (p/catch (fn [e]
               (swap! fail inc)
               (println "  FAIL (uncaught)" (.-message e))
               (println (.-stack e))
               (summary))))
