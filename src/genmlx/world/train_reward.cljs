(ns genmlx.world.train-reward
  "Phase 1 (bean genmlx-ugkv) — GRPO where the REWARD is a GFI quantity.

   THE IDEA (the highest-leverage novelty). GenMLX already computes exactly the
   scalar that policy-gradient RL wants as a reward. An LLM (the policy) writes a
   PROBABILISTIC PROGRAM; GenMLX evaluates it and returns **Bayesian model
   evidence** — the marginal log-likelihood `log p(data | program)` — and GRPO uses
   THAT as the reward to update the policy weights. As far as we know this is the
   first system where Bayesian model evidence is the RL signal for an LLM: the
   *wake phase* of wake/sleep operating at the level of program space. The
   estimator family is the same REINFORCE-with-baseline already in
   `inference/adev.cljs` — GRPO's group-mean baseline IS the variance-reduction
   baseline.

   WHAT THIS NAMESPACE IS. Pure reward-fn builders. Each returns a pure
       (fn [prompt completion] -> number)
   closure to hand to the Phase-0 `genmlx.world.train/train-step!` bridge. There is
   NO new Rust, NO new autograd, and NO dependency on `world.train` — this ns only
   PRODUCES a reward-fn; the caller passes it to the training effect. The reward-fn
   itself is deterministic given a completion and (critically) never forward-passes
   the *policy* model: `score-model` runs GenMLX inference over the *generated*
   program (a fresh GF) and `verify` runs the generated fn — neither touches the
   policy LLM, so model ownership stays with the training thread.

   THE FINITE FLOOR (load-bearing for GRPO). GRPO normalizes advantages by the
   group std. A single non-finite reward (`##-Inf` from a nil/erroring GF, or NaN)
   poisons the whole group: NaN advantages -> the native step is silently skipped
   (`gradients_applied=false`, the Phase-0 lesson). So EVERY path here clamps to a
   FINITE floor (`default-reward-floor`, -100.0): a parse/eval failure, a non-finite
   evidence, or a completion that does not even trace the observed sites all map to
   the floor, never to `##-Inf`. The floor also creates the dominant learning
   contrast (a valid, data-explaining program scores ~ -1..-30; garbage scores the
   floor), which is what GRPO climbs first.

   REWARD INTEGRITY (the coverage guard). `p/generate` only charges weight for sites
   that are BOTH traced and constrained (handler.cljs `generate-transition`). So a
   program that declares latents but never traces the observed addresses scores
   weight 0 — HIGHER than a correct model's negative log-evidence. That is reward
   hacking. `model-evidence-reward` defends against it: a static program must declare
   every observed address as a trace site, else it scores the floor (`:require-
   coverage?`, on by default). For the Phase-1 task the synthesized models are static
   with literal addresses, so legitimate solutions pass and data-ignoring ones floor.

   KL-FREE. The native autograd path rejects `klCoef > 0` (reference-model logprobs
   not yet wired through autograd; grpo/autograd.rs). So Phase-1 training runs
   KL-free (`:kl-coef 0.0`). Wiring ref-logprobs through autograd for a true
   KL-to-base penalty is Phase 1.5 (a native change + a clean upstream PR),
   explicitly deferred.

   Sections:
     1. The finite floor
     2. Program extraction  (completion text -> a clean (fn [trace] ...) string)
     3. Reward shaping knobs (compilation bonus, complexity, coverage guard)
     4. The reward builders  (model-evidence-reward, transition-fn-reward)
     5. The Phase-1 demo task (gaussian-mean) + prompt/batch builders"
  (:require [genmlx.llm.msa-score :as score]
            [genmlx.codegen.eval :as ce]
            [genmlx.inspect :as inspect]
            [clojure.string :as str]))

;; ===========================================================================
;; 1. The finite floor
;; ===========================================================================

(def default-reward-floor
  "The finite reward assigned to an invalid / non-finite / data-ignoring
   completion. NOT `##-Inf`: a non-finite value in a GRPO group produces NaN
   advantages and the native step is silently skipped (the Phase-0 lesson)."
  -100.0)

(defn finite?
  "True iff x is a finite JS number (rejects nil, NaN, ##Inf, ##-Inf)."
  [x]
  (and (number? x) (js/isFinite x)))

(defn clamp-floor
  "Clamp x to the finite floor: a non-finite x (or x < floor) becomes `floor`."
  [floor x]
  (if (finite? x) (max x floor) floor))

;; ===========================================================================
;; 2. Program extraction — completion text -> a clean (fn [trace] ...) string
;;
;; A raw LLM completion may carry a Qwen `<think>...</think>` block, markdown
;; fences, surrounding prose, and trailing text after the program. score/eval-model
;; (sci/eval-string) evaluates the WHOLE string and returns nil if anything after
;; the program fails to eval, which would unfairly floor a VALID program. So we
;; isolate the FIRST function form and pr-str it back to a clean code string.
;; ===========================================================================

(defn- strip-think
  "Drop everything up to and including the final `</think>` (Qwen reasoning),
   whose contents are full of parens that would fool a first-paren extractor."
  [s]
  (let [s (str s)]
    (if-let [idx (str/last-index-of s "</think>")]
      (subs s (+ idx (count "</think>")))
      s)))

(defn- canonical-fn-form
  "Normalize a parsed top-level form to a `(fn [args] body...)` form, or nil.
   Passes `(fn ...)`/`(fn* ...)` through; rewrites `(defn name [args] body...)`
   and `(defn- ...)` (the form the LLM sometimes emits) to an anonymous fn."
  [form]
  (when (seq? form)
    (let [head (first form)]
      (cond
        (contains? #{'fn 'fn*} head) form
        (contains? #{'defn 'defn-} head)
        (let [more (rest form)
              argv (first (filter vector? more))
              body (rest (drop-while #(not= % argv) more))]
          (when argv (list* 'fn argv body)))
        :else nil))))

(defn extract-program
  "Extract a clean `(fn [trace] ...)` program string from a raw LLM completion.
   Strips a `<think>` block, peels markdown fences / prose (ce/extract-code),
   reads the first form, and — when it is a function form — pr-strs JUST that form
   (dropping trailing junk). Returns a code string (possibly empty)."
  [completion]
  (let [code  (ce/extract-code (strip-think completion))
        form  (ce/parse-form code)
        fform (canonical-fn-form form)]
    (if fform (pr-str fform) code)))

(defn completion->gf
  "Completion text -> a DynamicGF (via the MSA SCI sandbox), or nil on a parse /
   eval failure / non-function result."
  [completion]
  (score/eval-model (extract-program completion)))

;; ===========================================================================
;; 3. Reward shaping knobs — all OFF by default (plain log-ML is the clean proof)
;; ===========================================================================

(def ^:private compilation-rank
  "A structural prior toward GenMLX-idiomatic models: a higher compilation level
   (reported by inspect/inspect) earns a larger bonus. M2 (full static
   compilation) ranks highest; M3 (prefix) / M4 (branch-rewrite) above L0."
  {:L0 0 :L1-M3 1 :L1-M4 1 :L1-M2 2})

(defn compilation-level
  "The compilation level inspect/inspect reports for gf (:L0/:L1-M2/:L1-M3/:L1-M4),
   or :L0 when unavailable."
  [gf]
  (or (:compilation (inspect/inspect gf)) :L0))

(defn compilation-bonus
  "Integer compilation-bonus magnitude for gf (see compilation-rank). Multiplied by
   the λc knob in model-evidence-reward (λc = 0 by default => no bonus)."
  [gf]
  (get compilation-rank (compilation-level gf) 0))

(defn form-size
  "Total node count of a form tree (an Occam complexity measure). Multiplied by the
   λk knob (λk = 0 by default => no penalty)."
  [form]
  (cond
    (map? form)        (reduce + 1 (map form-size (mapcat identity form)))
    (sequential? form) (reduce + 1 (map form-size form))
    :else 1))

(defn covered-observations?
  "True iff EVERY observed address is declared as a (static, literal) trace site of
   gf — the reward-integrity guard (a data-ignoring program traces none of them and
   scores weight 0, which would beat a correct model). For a non-static program the
   literal sites are unknown; we conservatively report false so it scores the floor
   rather than risk a hackable 0. The Phase-1 task's models are static."
  [gf observations]
  (let [addrs (set (map :addr (:trace-sites (:schema gf))))]
    (every? addrs (keys observations))))

;; ===========================================================================
;; 4. The reward builders — pure (fn [prompt completion] -> number)
;; ===========================================================================

(defn model-evidence-reward
  "Build the HEADLINE Phase-1 reward: a pure `(fn [prompt completion] -> number)`
   whose value is the program's Bayesian model evidence against the task's fixed
   `observations`.

       completion --extract--> (fn [trace] ...) --SCI eval--> GF
                  --score-model--> log p(observations | program)   [floored]

   `task` = {:observations {:addr value ...} ...}. opts:
     :reward-floor      finite floor for invalid/non-finite/data-ignoring (default -100.0)
     :n-particles       importance samples for the non-conjugate IS fallback (default 50)
     :require-coverage? require every observed addr be a trace site (default true)
     :lambda-c          weight on the compilation bonus (default 0.0, off)
     :lambda-k          weight on the form-size complexity penalty (default 0.0, off)

   Pure, deterministic given a completion, and never forward-passes the policy model
   (it scores the *generated* GF). Eval errors are caught and mapped to the floor —
   the reward-fn never throws into the training step."
  ([task] (model-evidence-reward task {}))
  ([{:keys [observations]}
    {:keys [reward-floor n-particles require-coverage? lambda-c lambda-k]
     :or   {reward-floor default-reward-floor n-particles 50
            require-coverage? true lambda-c 0.0 lambda-k 0.0}}]
   (fn [_prompt completion]
     (try
       (let [code (extract-program completion)
             gf   (score/eval-model code)]
         (if (or (nil? gf)
                 (and require-coverage? (not (covered-observations? gf observations))))
           reward-floor
           (let [base (score/score-model gf observations {:n-particles n-particles})]
             (if (finite? base)
               (clamp-floor reward-floor
                            (cond-> base
                              (pos? lambda-c) (+ (* lambda-c (compilation-bonus gf)))
                              (pos? lambda-k) (- (* lambda-k (form-size (or (ce/parse-form code) code))))))
               reward-floor))))
       (catch :default _ reward-floor)))))

(defn transition-fn-reward
  "Build the program-CORRECTNESS reward (implemented per the spec but exercised only
   lightly): a pure `(fn [prompt completion] -> number)` = fraction of held
   transitions the generated `(fn [state action] -> state)` reproduces.

       completion --extract--> (fn [state action] ...) --verify--> :accuracy in [0,1]

   `task` = {:transitions [{:state map :action kw :expected map} ...]}. opts:
     :reward-floor   finite floor for an un-parseable / un-evaluable program (default -100.0)
     :lambda-s       weight on the (idiomaticity) structural score (default 0.0, off)

   A program that evals but is simply WRONG scores its accuracy (>= 0.0, finite);
   only a program that fails to eval at all (verify's :error) scores the floor, so
   GRPO sees a gradient between garbage, runnable-but-wrong, and correct."
  ([task] (transition-fn-reward task {}))
  ([{:keys [transitions]}
    {:keys [reward-floor lambda-s]
     :or   {reward-floor default-reward-floor lambda-s 0.0}}]
   (fn [_prompt completion]
     (try
       (let [code (extract-program completion)
             {:keys [accuracy error]} (ce/verify-transition-fn code transitions)]
         (if error
           reward-floor
           (clamp-floor reward-floor
                        (cond-> accuracy
                          (pos? lambda-s)
                          (+ (* lambda-s (if-let [f (ce/parse-form code)]
                                           (ce/score-structure f)
                                           -10)))))))
       (catch :default _ reward-floor)))))

;; ===========================================================================
;; 5. The Phase-1 demo task — Bayesian model synthesis (gaussian mean)
;;
;; The policy is asked to write a GenMLX `gen`-style model that explains a small
;; fixed dataset; the reward is the model's marginal log-evidence. The prompt shows
;; a COMPLETE shared-mean model over the real observation addresses (so a faithful
;; completion traces the observed sites and clears the coverage guard) but with
;; deliberately suboptimal numbers, and asks the policy to ADAPT the prior mean,
;; prior std, and noise — leaving headroom for the reward to climb (valid-rate first,
;; then fit). The dataset is fixed CLJS data (no I/O).
;; ===========================================================================

(def gaussian-mean-system-prompt
  (str "You are a ClojureScript code generator for the GenMLX probabilistic "
       "programming system. Reply with ONLY a single (fn [trace] ...) form and "
       "nothing else — no prose, no markdown, no comments. Begin your reply with "
       "the characters (fn [trace]."))

(defn gaussian-mean-prompt
  "Build the user prompt for a gaussian-mean dataset. It gives a COMPLETE, single
   shared-mean model over the real observation addresses and asks the policy to
   ADAPT the three numbers — the prior mean, prior std, and observation noise — to
   best fit the data. Using the real addresses means a faithful completion traces
   the observed sites (passing the coverage guard); the numbers are where the
   marginal-likelihood gradient — the thing GRPO climbs — lives."
  [observations]
  (let [obs  (sort-by (comp name key) observations)
        data (str/join ", " (map (fn [[k v]] (str (name k) " = " v)) obs))
        site (fn [k] (str k " (trace " k " (dist/gaussian mu 1))"))
        body (str/join " " (map (comp site str key) obs))]
    (str "Write a GenMLX probabilistic model that explains this data: " data ".\n"
         "It has one shared latent mean :mu and one Gaussian observation site per\n"
         "data point (:y0 :y1 :y2 :y3). Start from this example and ADAPT the three\n"
         "numbers (the prior mean, the prior std, and the observation noise) so the\n"
         "data is well explained — a higher marginal likelihood is better:\n\n"
         "(fn [trace] (let [mu (trace :mu (dist/gaussian 0 10))] {" body "}))\n\n"
         "Output ONLY your completed (fn [trace] ...) form, nothing else.")))

(def gaussian-mean-task
  "The canonical Phase-1 model-synthesis task: explain four observations near 2.0
   with a shared-mean Gaussian model; reward = marginal log-evidence."
  (let [observations {:y0 2.0 :y1 2.3 :y2 1.7 :y3 2.1}]
    {:name          "gaussian-mean"
     :observations  observations
     :system-prompt gaussian-mean-system-prompt
     :prompt        (gaussian-mean-prompt observations)}))

(defn task->chat
  "Task -> a single native-ready chat conversation: an optional :system turn then
   the :user turn (a vector of {:role :content} maps, which world.train marshals)."
  [{:keys [system-prompt prompt]}]
  (cond-> []
    system-prompt (conj {:role :system :content system-prompt})
    true          (conj {:role :user :content prompt})))

(defn task->prompts
  "Task -> a one-element prompt batch for `train/train-step!` (the engine samples
   group-size completions per prompt)."
  [task]
  [(task->chat task)])
