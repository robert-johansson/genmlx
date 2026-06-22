(ns genmlx.world.repl-corpus
  "Phase 3 (genmlx-oexl), done-means #1: harvest the GFI/REPL trace of a SUCCESSFUL loop
   run into an SFT corpus that teaches the *propose-eval-revise POLICY* — given a partial
   model + the verifier's feedback, produce the good next edit — NOT whole programs (the
   distinction the north star rests on: §12 showed the loop's value is the policy, and a
   cheap 0.8B SFT'd on whole programs can't drive the loop; this corpus is what fixes that).

   THE TRACE OF A GOOD RUN *IS* THE CORPUS. Each accepted step transition (model_i,
   feedback_i) → model_{i+1} becomes one chat row whose user turn is exactly the step-prompt
   the proposer saw and whose assistant turn is the edit that improved the exact evidence.
   So the student learns to imitate the loop's accepted moves (including, on the harder
   models, the structure→refine sequence the 35B teacher produced).

   NATIVE-FREE: reuses the proposer's prompt builders (`genmlx.world.llm-proposer`) and the
   four-level `check` node (`genmlx.world.synth`) — no model loads here. Rows match
   `genmlx.world.sft`'s chat-row shape (`{:task-id :kind :messages}`), so sft_prep /
   sft_train / sft_eval consume them unchanged, and the SAME leakage-guarded task split
   applies (the rrps lesson: split at the TASK level, before training).

   Sections:
     1. trajectory->rows — one successful run's accepted transitions → SFT rows
     2. build-corpus — many (task, trajectory) runs → rows + a leakage-safe report"
  (:require [genmlx.world.synth :as syn]
            [genmlx.world.llm-proposer :as lp]
            [genmlx.world.sft :as sft]))

;; ===========================================================================
;; 1. trajectory->rows
;; ===========================================================================

(defn trajectory->rows
  "A successful run's trajectory (a vector of accepted states, each at least `{:code ...}`,
   step 0 = the crude init) + its `task` ({:id :task-desc :observations}) → SFT rows, one
   per step TRANSITION i→i+1:
     system    = the DSL prompt (genmlx.world.llm-proposer/default-system)
     user      = the step-prompt the proposer SAW at step i: task + data + model_i + the
                 verifier's feedback on model_i (re-derived by `check`, so it is exactly
                 what conditioned the real proposal)
     assistant = the model ACCEPTED at step i+1 (fenced — what extract-form expects)

   ORACLE FILTER: a transition is kept only when model_{i+1} scores AND strictly improves
   model_i's evidence (or model_i was unscored and model_{i+1} scores). An accepted step is
   improving by construction, but re-verifying makes the corpus self-consistent and lets
   this run over ANY trajectory source (e.g. a re-loaded artifact) — never trusting a label."
  ([task trajectory] (trajectory->rows task trajectory {}))
  ([{:keys [id task-desc observations]} trajectory {:keys [n-particles] :or {n-particles 2000}}]
   (let [steps (vec trajectory)
         chk   (fn [code] (when code (syn/check code observations {:n-particles n-particles})))]
     (vec
      (for [i (range (dec (count steps)))
            :let [code-i  (:code (nth steps i))
                  code-i1 (:code (nth steps (inc i)))
                  fb-i    (chk code-i)
                  fb-i1   (chk code-i1)]
            :when (and code-i code-i1 fb-i fb-i1
                       (syn/scored? fb-i1)
                       (or (not (syn/scored? fb-i))
                           (> (:evidence fb-i1) (:evidence fb-i))))]
        {:task-id  (name id)
         :kind     :repl-edit
         :rank-key (:evidence fb-i1)
         :messages [{:role "system"    :content lp/default-system}
                    {:role "user"      :content (lp/step-prompt task-desc observations code-i fb-i)}
                    {:role "assistant" :content (str "```clojure\n" code-i1 "\n```")}]})))))

;; ===========================================================================
;; 2. build-corpus
;; ===========================================================================

(defn build-corpus
  "Harvest many successful runs into one REPL-edit corpus. `runs` is a seq of
   `{:task {:id :task-desc :observations} :trajectory [...]}`. Returns
     {:rows [...]                 ; ALL harvested transition rows
      :train-rows [...]           ; rows whose task is NOT held out (sft/eval-task-ids or
                                  ;   the supplied `eval-ids`) — the leakage-safe training set
      :dropped-eval [...]         ; rows dropped because their task is held out
      :n-runs n :n-rows n :per-task {task-id count} :train-task-ids #{...}}

   The held-out split is applied at the TASK level via genmlx.world.sft/partition-corpus,
   so a student trained on `:train-rows` is graded only on tasks it never saw — the same
   leakage guarantee the distill→SFT pipeline enforces."
  ([runs] (build-corpus runs {}))
  ([runs {:keys [n-particles eval-ids] :or {n-particles 2000}}]
   (let [rows (vec (mapcat (fn [{:keys [task trajectory]}]
                             (trajectory->rows task trajectory {:n-particles n-particles}))
                           runs))
         part (if eval-ids (sft/partition-corpus rows eval-ids) (sft/partition-corpus rows))]
     {:rows rows
      :train-rows (:train-rows part)
      :dropped-eval (:dropped-eval part)
      :n-runs (count runs)
      :n-rows (count rows)
      :per-task (frequencies (map :task-id rows))
      :train-task-ids (:train-task-ids part)})))
