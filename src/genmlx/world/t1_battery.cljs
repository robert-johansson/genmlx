(ns genmlx.world.t1-battery
  "The unified task battery for the Tier-1 instruction-prompted bake-off
   (genmlx-8lm2).

   tasks = the 12 canonical distillation seed tasks (genmlx.world.distill-tasks:
   4 conjugate :program + 3 state-machine :function + 5 test-case :function)
   ++ 5 :program tasks lifted from the inline MSA task maps in
   examples/msa_matching_demo.cljs and examples/qwen36_*.cljs (msi /
   hyperprior / model_selection / structural), converted to the distill task
   format: {:id :kind :system-prompt :prompt} + the held-out oracle
   :observations. Same holdout discipline as distill-tasks — the oracle signal
   lives HERE, never in the prompt text, so a completion that scores well is
   real generalization. Same prompt style too: a complete (fn [trace] ...)
   template over the REAL observation addresses (so a faithful completion
   clears the coverage guard) with deliberately loose numbers, and the model is
   asked to ADAPT them.

   The lifted tasks (string ids, matching distill-tasks' convention — a JSONL
   round-trip stringifies ids, and tasks-by-id must join on what comes back):
     msa-1  matching-to-sample, one Bernoulli trial with a Beta-prior ability
            (examples/msa_matching_demo.cljs, conjugate)
     msa-2  beta-bernoulli coin, 8 heads / 2 tails over :flip0..:flip9
            (examples/qwen36_msi.cljs / qwen36_hyperprior.cljs, conjugate)
     msa-3  noisy LINEAR regression, latents :slope :intercept, obs :y0..:y6
            (examples/qwen36_structural.cljs linear-task, linear-Gaussian)
     msa-4  noisy QUADRATIC regression, latents :a :b :c, obs :y0..:y6
            (examples/qwen36_structural.cljs quadratic-task, linear-in-params)
     msa-5  matching-to-sample with ability AND difficulty latents behind a
            sigmoid link (the msa_matching_demo task lifted faithfully —
            non-conjugate, exercises the IS-scoring path)

   exemplars maps each lifted id to its prompt-embedded template string — the
   known-good completion the STUB arm of scripts/t1_bakeoff.cljs replays, so a
   stub run validates the full oracle path (coverage guard, exact/IS scoring)
   over every lifted task without drift between prompt and stub."
  (:require [genmlx.world.distill-tasks :as dt]
            [clojure.string :as str]))

;; ===========================================================================
;; Template builders for the lifted :program tasks
;; ===========================================================================

(def ^:private coin-flips
  "Eight heads, two tails (examples/qwen36_msi.cljs `observations`)."
  [1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 0.0 0.0])

(defn- coin-site [i]
  (let [k (str ":flip" i)]
    (str k " (trace " k " (dist/bernoulli theta))")))

(def ^:private coin-template
  (str "(fn [trace] (let [theta (trace :theta (dist/beta 1 1))] {"
       (str/join " " (map coin-site (range (count coin-flips))))
       "}))"))

(def ^:private linreg-xs [-3.0 -2.0 -1.0 0.0 1.0 2.0 3.0])

(def ^:private linreg-ys
  "examples/qwen36_structural.cljs linear-task (truth y = 2x + 1)."
  [-5.16 -3.04 -0.86 1.21 3.05 4.95 7.13])

(def ^:private quad-ys
  "examples/qwen36_structural.cljs quadratic-task (truth y = 0.5x^2 + 1.5x + 2)."
  [1.65 0.84 1.32 2.18 4.05 6.78 11.14])

(defn- linreg-site [i x]
  (let [k (str ":y" i)]
    (str k " (trace " k " (dist/gaussian (mx/add (mx/multiply slope " x
         ") intercept) 0.5))")))

(def ^:private linreg-template
  (str "(fn [trace] (let [slope (trace :slope (dist/gaussian 0 10))"
       " intercept (trace :intercept (dist/gaussian 0 10))] {"
       (str/join " " (map-indexed linreg-site linreg-xs))
       "}))"))

(defn- quad-site [i x]
  (let [k (str ":y" i)]
    (str k " (trace " k " (dist/gaussian (mx/add (mx/add (mx/multiply a " (* x x)
         ") (mx/multiply b " x ")) c) 0.5))")))

(def ^:private quad-template
  (str "(fn [trace] (let [a (trace :a (dist/gaussian 0 10))"
       " b (trace :b (dist/gaussian 0 10))"
       " c (trace :c (dist/gaussian 0 10))] {"
       (str/join " " (map-indexed quad-site linreg-xs))
       "}))"))

(def ^:private mts-template
  (str "(fn [trace] (let [ability (trace :ability (dist/beta 1 1))]"
       " {:correct (trace :correct (dist/bernoulli ability))}))"))

(def ^:private mts2-template
  (str "(fn [trace] (let [ability (trace :ability (dist/gaussian 0 1))"
       " difficulty (trace :difficulty (dist/gaussian 0 1))"
       " p (mx/divide 1 (mx/add 1 (mx/exp (mx/subtract difficulty ability))))]"
       " {:correct (trace :correct (dist/bernoulli p))}))"))

(defn- data-table [xs ys]
  (str "  x: [" (str/join ", " xs) "]\n"
       "  y: [" (str/join ", " ys) "]"))

;; ===========================================================================
;; The lifted tasks
;; ===========================================================================

(def lifted-tasks
  [{:id "msa-1"
    :kind :program
    :system-prompt dt/program-system-prompt
    :prompt (str "A subject does a matching-to-sample perceptual task: they see a\n"
                 "sample stimulus and must pick the matching one. We observed ONE\n"
                 "trial: they matched correctly (site :correct, value 1). Model their\n"
                 "perceptual ability as a latent probability :ability with a Beta\n"
                 "prior; the trial is Bernoulli(ability). Start from this and ADAPT\n"
                 "the two Beta-prior numbers so the observation is well explained:\n\n"
                 mts-template "\n\n"
                 "Output ONLY your completed (fn [trace] ...) form, nothing else.")
    :observations {:correct 1.0}}

   {:id "msa-2"
    :kind :program
    :system-prompt dt/program-system-prompt
    :prompt (str "Write a GenMLX model for a possibly-biased coin observed flipping\n"
                 "1 1 1 1 1 1 1 1 0 0 (ten flips, sites :flip0..:flip9, 1 = heads).\n"
                 "One latent bias :theta with a Beta prior; each flip is\n"
                 "Bernoulli(theta). Start from this and ADAPT the two Beta-prior\n"
                 "numbers so the data is well explained:\n\n"
                 coin-template "\n\n"
                 "Output ONLY your completed (fn [trace] ...) form, nothing else.")
    :observations (into {} (map-indexed (fn [i v] [(keyword (str "flip" i)) v])
                                        coin-flips))}

   {:id "msa-3"
    :kind :program
    :system-prompt dt/program-system-prompt
    :prompt (str "Write a GenMLX model of these (x, y) data points as a noisy LINEAR\n"
                 "function y = slope*x + intercept + noise:\n"
                 (data-table linreg-xs linreg-ys) "\n"
                 "Latents :slope and :intercept with Gaussian priors; one Gaussian\n"
                 "observation site per data point (:y0..:y6, the x values are baked\n"
                 "in as constants). Start from this and ADAPT the numbers (the prior\n"
                 "means, the prior stds, and the observation noise) so the data is\n"
                 "well explained:\n\n"
                 linreg-template "\n\n"
                 "Output ONLY your completed (fn [trace] ...) form, nothing else.")
    :observations (into {} (map-indexed (fn [i v] [(keyword (str "y" i)) v])
                                        linreg-ys))}

   {:id "msa-4"
    :kind :program
    :system-prompt dt/program-system-prompt
    :prompt (str "Write a GenMLX model of these (x, y) data points as a noisy\n"
                 "QUADRATIC function y = a*x^2 + b*x + c + noise:\n"
                 (data-table linreg-xs quad-ys) "\n"
                 "Latents :a :b :c with Gaussian priors; one Gaussian observation\n"
                 "site per data point (:y0..:y6, the x and x^2 values are baked in\n"
                 "as constants). Start from this and ADAPT the numbers (the prior\n"
                 "means, the prior stds, and the observation noise) so the data is\n"
                 "well explained:\n\n"
                 quad-template "\n\n"
                 "Output ONLY your completed (fn [trace] ...) form, nothing else.")
    :observations (into {} (map-indexed (fn [i v] [(keyword (str "y" i)) v])
                                        quad-ys))}

   {:id "msa-5"
    :kind :program
    :system-prompt dt/program-system-prompt
    :prompt (str "A subject does a matching-to-sample perceptual task. They have some\n"
                 "perceptual ability (higher is better); the task has some difficulty.\n"
                 "Their probability of a correct match is sigmoid(ability - difficulty).\n"
                 "We observed ONE trial: correct (site :correct, value 1). Latents\n"
                 ":ability and :difficulty with Gaussian priors. Start from this and\n"
                 "ADAPT the four Gaussian-prior numbers so the observation is well\n"
                 "explained:\n\n"
                 mts2-template "\n\n"
                 "Output ONLY your completed (fn [trace] ...) form, nothing else.")
    :observations {:correct 1.0}}])

;; ===========================================================================
;; The assembled battery
;; ===========================================================================

(def tasks
  "The full T1 battery (17): the 12 distill seed tasks ++ the 5 lifted MSA tasks."
  (vec (concat dt/tasks lifted-tasks)))

(def tasks-by-id
  "Map of :id -> task, for joining verdicts/candidates back to their prompts."
  (into {} (map (juxt :id identity)) tasks))

(def exemplars
  "Lifted-task id -> the prompt's own template string (a known-good completion).
   Consumed by the STUB arm of scripts/t1_bakeoff.cljs so stub validation
   exercises the real oracle path on every lifted task with zero prompt/stub
   drift. NOT used by real arms."
  {"msa-1" mts-template
   "msa-2" coin-template
   "msa-3" linreg-template
   "msa-4" quad-template
   "msa-5" mts2-template})
