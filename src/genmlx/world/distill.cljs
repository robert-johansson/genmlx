(ns genmlx.world.distill
  "Offline oracle-filter for teacher->student ClojureScript distillation (genmlx-j0d6).

   THE OFFLINE TWIN of `genmlx.world.train-reward`. Where train-reward turns ONE
   completion into a scalar GRPO reward online (the *wake phase* signal), this
   namespace BATCH-FILTERS a strong teacher's many candidate completions into an
   oracle-VALIDATED SFT corpus — using the SAME native-free oracle: `codegen.eval`
   (edamame reader + SCI) and `msa-score` (Bayesian model evidence). No new native
   code; the policy/teacher LLM is never loaded here — candidates arrive over a file
   (`raw_candidates.jsonl`) from the teacher's offline batch (step 2 of the recipe).

   Because the filter and the reward share the exact extraction (`extract-program`,
   `strip-think`) and the exact integrity guard (`covered-observations?`), the
   distilled corpus and the GRPO reward can never silently disagree about what a
   'good' completion is.

   THE GATE LADDER (steps 3-4 of the j0d6 recipe):

     candidate ──extract──► code ──parse?──► eval?──► {behavioral gate}──► rank
       :program  ─► coverage guard ─► score-model log-evidence   (rank by evidence)
       :function ─► verify-transition-fn  OR  held-out test-cases (rank by accuracy)

   Keep the top-k valid+highest-ranked completions per prompt; emit Qwen3 chat-format
   SFT records + a stats report. Per-prompt YIELD (fraction of prompts that produced
   >=1 kept candidate) measures the teacher's coverage of the task space.

   Sections:
     1. Drop reasons + small helpers
     2. evaluate-candidate  (the per-completion gate ladder, both kinds)
     3. rank-and-select     (top-k per prompt)
     4. verdicts->stats     (corpus-quality report)
     5. build-sft-records   (kept candidates -> Qwen3 chat JSONL rows)"
  (:require [genmlx.world.train-reward :as reward]
            [genmlx.llm.msa-score :as score]
            [genmlx.codegen.eval :as ce]
            [clojure.string :as str]))

;; ===========================================================================
;; 1. Drop reasons + helpers
;; ===========================================================================

(def drop-reasons
  "Human-readable explanation per terminal :reason a verdict can carry."
  {:kept         "passed all gates"
   :empty        "no code extracted from the completion"
   :unparseable  "extracted text is not a complete, valid ClojureScript form"
   :eval-error   "evaluated to an error / not a model (or function)"
   :uncovered    "program ignores an observed address (the reward-integrity guard)"
   :degenerate   "program uses a point-mass (delta) at an observed site — a degenerate reward-hack fit"
   :nonfinite-ml "model evidence was non-finite (-Inf / NaN)"
   :low-evidence "model evidence below the configured :min-log-ml floor"
   :test-fail    "failed at least one held-out test"
   :no-oracle    "task carries no held-out oracle (no :transitions and no :test-cases) — misconfigured"
   :unknown-kind "task :kind is neither :program nor :function"})

(defn- safe-div
  "n/d as a float, or 0.0 when d is 0 — for rate stats over possibly-empty sets."
  [n d]
  (if (zero? d) 0.0 (/ n d)))

(defn- mean
  "Arithmetic mean of a non-empty seq of numbers, or nil if empty."
  [xs]
  (when (seq xs) (/ (reduce + xs) (count xs))))

(defn- structure-of
  "Idiomaticity score of a code string (ce/score-structure over its parsed form),
   or nil when the code does not parse to a form."
  [code]
  (when-let [f (ce/parse-form code)]
    (ce/score-structure f)))

(defn- form-uses-delta?
  "Does the parsed form reference a `delta` distribution anywhere? A delta (point
   mass) at an OBSERVED site is a degenerate reward-hack: it asserts the observation
   is deterministic, scoring log-evidence ~0 (it beats any honest noisy model). We
   reject any program that mentions delta — none of the seed tasks legitimately need
   one. (coll? is false for strings in CLJS, so the walk never recurses into chars.)"
  [form]
  (cond
    (symbol? form) (= "delta" (name form))
    (coll? form)   (boolean (some form-uses-delta? form))
    :else          false))

(defn- degenerate-program?
  "True iff the program's source uses a point-mass delta (see form-uses-delta?)."
  [code]
  (boolean (when-let [f (ce/parse-form code)] (form-uses-delta? f))))

;; ===========================================================================
;; 2. evaluate-candidate — the per-completion gate ladder
;; ===========================================================================

(defn- gate-program
  "Gate a :program completion via the MSA / model-evidence oracle. Mirrors
   train-reward/model-evidence-reward's ladder exactly (same extract, same coverage
   guard, same scorer) but RETURNS A VERDICT instead of a floored scalar — a corpus
   filter wants to know *why* a candidate was dropped, and to RANK survivors by
   evidence rather than collapse failures to a floor."
  [{:keys [observations n-particles require-coverage? min-log-ml]
    :or   {n-particles 50 require-coverage? true}}
   completion]
  (let [code (reward/extract-program completion)]
    (cond
      (str/blank? code)
      {:code code :parse? false :kept? false :reason :empty}

      (not (ce/valid-cljs? code))
      {:code code :parse? false :kept? false :reason :unparseable}

      :else
      (let [gf (score/eval-model code)]
        (cond
          (nil? gf)
          {:code code :parse? true :eval? false :kept? false :reason :eval-error}

          (and require-coverage? (not (reward/covered-observations? gf observations)))
          {:code code :parse? true :eval? true :covered? false
           :kept? false :reason :uncovered}

          (degenerate-program? code)
          {:code code :parse? true :eval? true :covered? true
           :kept? false :reason :degenerate}

          :else
          (let [{:keys [log-ml method]}
                (score/score-model* gf observations {:n-particles n-particles})]
            (cond
              (not (reward/finite? log-ml))
              {:code code :parse? true :eval? true :covered? true
               :log-ml log-ml :method method :kept? false :reason :nonfinite-ml}

              (and min-log-ml (< log-ml min-log-ml))
              {:code code :parse? true :eval? true :covered? true
               :log-ml log-ml :method method :kept? false :reason :low-evidence}

              :else
              {:code code :parse? true :eval? true :covered? true
               :log-ml log-ml :method method :structure (structure-of code)
               :rank-key log-ml :kept? true :reason :kept})))))))

(defn- run-test-cases
  "Apply f to each test-case's :args and compare to :expected, returning the same
   {:accuracy :total :correct :failures} shape as ce/verify-transition-fn so the
   two behavioral gates fold into one code path."
  [f test-cases]
  (let [total   (count test-cases)
        results (map-indexed
                  (fn [i {:keys [args expected]}]
                    (try
                      (let [actual (apply f args)]
                        (if (= expected actual)
                          {:correct true}
                          {:correct false :index i :args args
                           :expected expected :actual actual}))
                      (catch :default e
                        {:correct false :index i :args args
                         :expected expected :actual (str "ERROR: " (.-message e))})))
                  test-cases)
        correct (count (filter :correct results))]
    {:accuracy (if (zero? total) 1.0 (/ correct total))
     :total    total
     :correct  correct
     :failures (vec (remove :correct results))}))

(defn- gate-function
  "Gate a :function completion behaviorally. The task carries EITHER :transitions
   (scored by ce/verify-transition-fn, the state x action -> state oracle) XOR
   :test-cases ([{:args [..] :expected v}], scored by applying the fn). The held-out
   tests are NEVER shown in the prompt, so passing them is real generalization.
   Kept iff accuracy >= :pass-threshold (default 1.0 — only fully-correct exemplars
   belong in an SFT corpus)."
  [{:keys [transitions test-cases pass-threshold]
    :or   {pass-threshold 1.0}}
   completion]
  ;; Keep the original form (named defn / named fn) — unlike extract-program, do NOT
  ;; canonicalize defn -> anonymous fn, which would break self-recursive solutions
  ;; (factorial, gcd) that call themselves by name.
  (let [code (ce/extract-code (reward/strip-think completion))]
    (cond
      ;; A :function task with NO held-out oracle would otherwise keep EVERY fn that
      ;; evals (accuracy defaults to 1.0 over zero checks) — silent corpus poisoning.
      ;; Treat it as a task misconfiguration, never a pass.
      (not (or (seq transitions) (seq test-cases)))
      {:code code :parse? false :kept? false :reason :no-oracle}

      (str/blank? code)
      {:code code :parse? false :kept? false :reason :empty}

      (not (ce/valid-cljs? code))
      {:code code :parse? false :kept? false :reason :unparseable}

      :else
      (let [{:keys [accuracy total correct error]}
            (if transitions
              (ce/verify-transition-fn code transitions)
              (let [{f :fn err :error} (ce/eval-fn code)]
                (if err {:accuracy 0.0 :error err} (run-test-cases f test-cases))))]
        (cond
          error
          {:code code :parse? true :eval? false :kept? false :reason :eval-error}

          (< accuracy pass-threshold)
          {:code code :parse? true :eval? true :accuracy accuracy
           :total total :correct correct :kept? false :reason :test-fail}

          :else
          {:code code :parse? true :eval? true :accuracy accuracy
           :total total :correct correct :structure (structure-of code)
           :rank-key accuracy :kept? true :reason :kept})))))

(defn evaluate-candidate
  "Run the full oracle gate ladder on ONE raw teacher completion against its task.

   `task`       — a seed task map (see genmlx.world.distill-tasks): {:id :kind ...}
                  plus :observations (for :program) or :transitions/:test-cases
                  (for :function).
   `completion` — the raw teacher output string (may carry <think>, fences, prose).
   `sample-idx` — which of the N teacher samples this is (for provenance).

   Returns a verdict map that ALWAYS has :task-id :sample-idx :kind :kept? :reason,
   plus :code (the gated string) and path-specific fields (:log-ml/:method for
   :program, :accuracy for :function, :structure for survivors). Never throws — an
   unexpected error is caught and reported as an :eval-error verdict."
  [task completion sample-idx]
  (let [base {:task-id (:id task) :sample-idx sample-idx :kind (:kind task)}]
    (merge base
           (try
             (case (:kind task)
               :program  (gate-program task completion)
               :function (gate-function task completion)
               {:code "" :parse? false :kept? false :reason :unknown-kind})
             (catch :default e
               {:code "" :parse? false :kept? false
                :reason :eval-error :error (.-message e)})))))

;; ===========================================================================
;; 3. rank-and-select — top-k survivors per prompt
;; ===========================================================================

(defn- deterministic-score?
  "True iff a verdict's :rank-key is reproducible: any :function verdict (accuracy is
   exact) or a :program scored by an EXACT analytical method. A non-conjugate program
   scored by importance sampling (:handler-is/:smc/:hmc/:vi) is non-reproducible and
   must never out-rank a deterministically-scored peer."
  [v]
  (or (= :function (:kind v))
      (contains? #{:exact :kalman} (:method v))))

(defn- select-key
  "Sort key for ranking KEPT verdicts within a task (ascending vector of comparables):
   deterministically-scored candidates first (an exact marginal / a function accuracy
   never loses to a noisy IS estimate), then higher :rank-key (evidence/accuracy),
   then — for :function only, where idiomaticity is meaningful — higher structure
   score, then shorter code (Occam tie-break)."
  [v]
  [(if (deterministic-score? v) 0 1)
   (- (:rank-key v))
   (if (= :function (:kind v)) (- (or (:structure v) 0)) 0)
   (count (:code v))])

(defn rank-and-select
  "Group KEPT verdicts by :task-id and keep the top-`top-k` per task, ranked by
   model evidence / test accuracy (idiomaticity & brevity break ties). Returns the
   flat seq of selected verdicts across all tasks (input order of tasks preserved)."
  [verdicts top-k]
  (->> verdicts
       (filter :kept?)
       (group-by :task-id)
       vals
       (mapcat (fn [vs] (take top-k (sort-by select-key vs))))))

;; ===========================================================================
;; 4. verdicts->stats — corpus-quality report
;; ===========================================================================

(defn verdicts->stats
  "Aggregate per-candidate verdicts (and the selected subset) into a corpus-quality
   report: parse / eval / test-pass rates, mean model evidence over kept programs,
   per-prompt yield (coverage of the task space), and a drop-reason histogram.

   `tasks` — the seed tasks that were filtered (for n-tasks / yield denominators).
   `verdicts` — every candidate's verdict. `selected` — the kept top-k subset."
  [tasks verdicts selected]
  (let [n         (count verdicts)
        n-tasks   (count tasks)
        prog?     #(= :program (:kind %))
        fn?       #(= :function (:kind %))
        progs     (filter prog? verdicts)
        fns       (filter fn? verdicts)
        kept      (filter :kept? verdicts)
        kept-prog (filter :kept? progs)
        attempted (count (distinct (map :task-id verdicts)))
        with-kept (count (distinct (map :task-id kept)))]
    {:n-candidates      n
     :n-tasks           n-tasks
     :n-tasks-attempted attempted
     :n-kept            (count kept)
     :n-selected        (count selected)
     :parse-rate        (safe-div (count (filter :parse? verdicts)) n)
     :eval-rate         (safe-div (count (filter :eval? verdicts)) n)
     :program-pass-rate (safe-div (count kept-prog) (count progs))
     :function-pass-rate (safe-div (count (filter :kept? fns)) (count fns))
     :mean-log-ml       (mean (keep :log-ml kept-prog))
     ;; yield = fraction of ATTEMPTED prompts that produced >=1 kept candidate
     ;; (the teacher's hit-rate over the prompts it was actually given).
     :yield-per-prompt  (safe-div with-kept attempted)
     ;; coverage = fraction of the FULL seed task space that got >=1 kept candidate.
     :task-space-coverage (safe-div with-kept n-tasks)
     :n-prompts-covered with-kept
     ;; how many SELECTED programs were scored by non-reproducible importance sampling
     ;; (a non-conjugate model): >0 means that prompt's pick is not deterministic.
     :n-selected-noisy-is (count (filter #(and (= :program (:kind %))
                                               (not (contains? #{:exact :kalman} (:method %))))
                                         selected))
     :drop-reasons      (into (sorted-map) (frequencies (map :reason verdicts)))}))

;; ===========================================================================
;; 5. build-sft-records — kept candidates -> Qwen3 chat JSONL rows
;; ===========================================================================

(defn task->messages
  "Build a Qwen3 instruct/chat message vector for an SFT pair: the task's optional
   system turn, its user prompt, then the validated completion as the assistant
   turn. Roles are STRINGS (\"system\"/\"user\"/\"assistant\") for JSONL fidelity."
  [task code]
  (cond-> []
    (:system-prompt task) (conj {:role "system" :content (:system-prompt task)})
    true                  (conj {:role "user" :content (:prompt task)})
    true                  (conj {:role "assistant" :content code})))

(defn build-sft-records
  "Turn selected verdicts into SFT corpus rows. `tasks-by-id` maps :task-id -> task
   (so the prompt/system turns come from the canonical in-tree task, never from a
   file that could leak hidden tests). Each row carries provenance (:task-id :kind
   :rank-key) alongside the :messages the trainer consumes."
  [tasks-by-id selected]
  (for [v selected
        :let [task (get tasks-by-id (:task-id v))]
        :when task]
    {:task-id  (:task-id v)
     :kind     (name (:kind v))
     :rank-key (:rank-key v)
     :messages (task->messages task (:code v))}))
