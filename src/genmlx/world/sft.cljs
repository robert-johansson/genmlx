(ns genmlx.world.sft
  "Pure core for the cljs-coder SFT step (genmlx-o8w9) — loop step 3 of 5
   (corpus → **SFT** → GRPO → iterate).

   THE OFFLINE-PREP + EVAL TWIN of `genmlx.world.distill`. Where distill BUILDS the
   oracle-validated SFT corpus from a teacher's candidates, this namespace (a) SPLITS
   that corpus into a training set and a leakage-guarded held-out eval set, (b) shapes
   the training rows for mlx-lm LoRA, and (c) scores the SFT'd student on the held-out
   tasks with the EXACT SAME oracle (`genmlx.world.distill/evaluate-candidate`, via the
   `distill-sandbox` timeout worker). Corpus-validation and eval-grading therefore use
   ONE oracle — they can never silently disagree about what a 'good' completion is, the
   same invariant distill enforces between the filter and the GRPO reward.

   This file is PURE: data shaping + pass@k arithmetic only. No model loading, no file
   or process I/O (mlx-lm trains/generates; the thin shells scripts/sft_prep.cljs and
   scripts/sft_eval.cljs do the I/O and call the sandbox grader). That is why the whole
   leakage guarantee, the corpus split, and the pass@k math are unit-testable here with
   synthetic rows — no GPU, no LLM.

   Sections:
     1. The canonical TRAIN / EVAL task split (held-out, leakage-guarded)
     2. Corpus prep — distill_sft rows → mlx-lm {messages} train/valid rows
     3. pass@k arithmetic (Chen et al. 2021 unbiased estimator)
     4. Eval reporting — baseline-vs-SFT table + cold-start guard"
  (:require [clojure.string :as str]))

;; ===========================================================================
;; 1. The canonical TRAIN / EVAL task split
;; ===========================================================================

(def eval-task-ids
  "The held-out EVAL task ids — ONE per oracle family, chosen so pass@1 covers BOTH
   oracle paths (model-evidence for :program, behavioral for :function) while leaving
   enough same-family train signal:

     gaussian-mean-negshift  :program  — train sees gaussian-mean-near2 + beta-bernoulli
                                          + gamma-poisson (conjugate-program transfer)
     traffic-light           :function — train sees counter-machine + toggle-switch
                                          (state-machine transfer)
     palindrome?             :function — train sees factorial/fizzbuzz/gcd/sum-evens
                                          (pure-function/test-case transfer)

   These tasks' completions MUST NEVER enter the training corpus — the rrps test-split
   leakage lesson (a validation-on-train win evaporated on a disjoint test split). The
   split is at the TASK level, before generation: the teacher generates the corpus over
   TRAIN tasks only, and the student is graded on EVAL tasks it was never trained on."
  #{"gaussian-mean-negshift"
    "traffic-light"
    "palindrome?"})

(defn eval-task?
  "True iff a seed task (or anything with an :id) is in the held-out eval set."
  [task-or-id]
  (contains? eval-task-ids (if (map? task-or-id) (:id task-or-id) task-or-id)))

(defn train-task?
  "True iff a seed task is NOT held out (its completions may train the student)."
  [task-or-id]
  (not (eval-task? task-or-id)))

(defn split-tasks
  "Partition a seed task seq into {:train [...] :eval [...]} by `eval-ids`
   (default `eval-task-ids`). Input order is preserved within each side."
  ([tasks] (split-tasks tasks eval-task-ids))
  ([tasks eval-ids]
   (let [in-eval? #(contains? eval-ids (:id %))]
     {:train (vec (remove in-eval? tasks))
      :eval  (vec (filter in-eval? tasks))})))

;; ===========================================================================
;; 2. Corpus prep — distill_sft rows → mlx-lm chat rows
;; ===========================================================================

(defn row->messages
  "Strip a distill_sft row {:task-id :kind :rank-key :messages [...]} to the bare
   mlx-lm chat record {:messages [...]} — the trainer must see ONLY the conversation,
   never the provenance keys (which could otherwise be mistaken for input features)."
  [row]
  {:messages (:messages row)})

(defn assert-train-disjoint!
  "Defensive post-condition: throw if any row belongs to a held-out EVAL task. The actual
   leakage prevention is done by partition-corpus (which DROPS eval-task rows by construction);
   this re-checks that the training set partition-corpus produced is genuinely clean, and also
   serves as a real load-bearing guard when applied to RAW (un-partitioned) corpus rows. Returns
   `rows` unchanged on success so it can wrap a pipeline."
  ([rows] (assert-train-disjoint! rows eval-task-ids))
  ([rows eval-ids]
   (let [leaked (filter #(contains? eval-ids (:task-id %)) rows)]
     (when (seq leaked)
       (throw (ex-info (str "LEAKAGE: " (count leaked)
                            " corpus row(s) belong to held-out eval task(s) "
                            (str/join ", " (distinct (map :task-id leaked))))
                       {:eval-ids eval-ids
                        :leaked-task-ids (vec (distinct (map :task-id leaked)))})))
     rows)))

(defn partition-corpus
  "Split distill_sft `rows` into the training set, DROPPING (never silently keeping)
   any held-out eval-task rows. Disjointness is enforced here by construction and the
   dropped set is reported, not folded in. Returns:
     {:train-rows           [...]   ; rows whose :task-id is a TRAIN task
      :dropped-eval         [...]   ; rows that belonged to a held-out eval task
      :train-task-ids       #{...}  ; distinct train task ids actually present
      :eval-task-ids-present #{...}} ; held-out ids that appeared (should be empty if
                                       the teacher only generated over train tasks)"
  ([rows] (partition-corpus rows eval-task-ids))
  ([rows eval-ids]
   ;; group on `eval-ids` directly (NOT the hardcoded train-task?), so a custom
   ;; held-out set — e.g. the scaled genmlx.world.distill-gen eval ids — is honored.
   (let [train? (fn [r] (not (contains? eval-ids (:task-id r))))
         {train true dropped false} (group-by train? rows)]
     {:train-rows            (vec train)
      :dropped-eval          (vec dropped)
      :train-task-ids        (into (sorted-set) (map :task-id) train)
      :eval-task-ids-present (into (sorted-set) (map :task-id) dropped)})))

(defn valid-split
  "Deterministically carve a validation slice out of `rows` for mlx-lm early-stopping /
   loss monitoring. Every `stride`-th row (stride derived from `frac`) goes to valid,
   the rest to train. Validation draws from TRAIN tasks — it monitors training loss and
   is NOT the held-out pass@1 eval (a different, task-disjoint set). Guarantees a
   non-empty train AND valid whenever `rows` has ≥2 rows (mlx-lm requires both)."
  [rows frac]
  (let [n (count rows)]
    (if (< n 2)
      {:train (vec rows) :valid (vec rows)} ; degenerate: reuse the row(s) so both files exist
      (let [stride (max 2 (js/Math.round (/ 1.0 (max frac 1e-9))))
            tagged (map-indexed vector rows)
            v?     (fn [[i _]] (zero? (mod (inc i) stride)))
            valid  (mapv second (filter v? tagged))
            train  (mapv second (remove v? tagged))]
        {:train (if (seq train) train (vec (butlast rows)))
         :valid (if (seq valid) valid [(last rows)])}))))

(defn blend
  "Interleave volume rows into the high-value distilled rows for LoRA volume. The
   distilled `head` rows lead (highest signal); up to `n` `volume` rows fill behind
   them. Deterministic (takes the first `n` volume rows — the caller pre-shuffles if a
   random draw is wanted). Both inputs are already {:messages ...} records."
  [head volume n]
  (vec (concat head (take (max 0 n) volume))))

;; ===========================================================================
;; 3. pass@k arithmetic — Chen et al. (2021) unbiased estimator
;; ===========================================================================

(defn pass-at-k
  "Unbiased pass@k estimate (Chen et al. 2021, the HumanEval estimator): given `n`
   i.i.d. samples of which `c` are correct, the probability that `k` uniformly-drawn
   samples contain ≥1 correct is

       1 − C(n−c, k) / C(n, k)  =  1 − ∏_{i=n−c+1}^{n} (1 − k/i)

   The product form is numerically stable (no big binomials). Edge cases: 0.0 when
   c=0 (no correct sample can be drawn) or n≤0; 1.0 when n−c < k (every draw must hit a
   correct one). `k` is clamped to ≤ n by the (n−c<k ⇒ 1.0) branch."
  [n c k]
  (cond
    (<= n 0)     0.0
    (<= c 0)     0.0
    (< (- n c) k) 1.0
    :else        (- 1.0 (reduce * (for [i (range (inc (- n c)) (inc n))]
                                    (- 1.0 (/ k i)))))))

(defn passk-of
  "Pass-rates for ONE task's sample verdicts. `verdicts` are that task's graded samples
   (each carrying :kept? and :sample-idx). By the generation convention sample-idx 0 is
   the GREEDY (temperature-0) sample and 1.. are temperature-sampled. Returns:
     {:n :c                  ; total / kept count over the SAMPLED (idx≥1) verdicts
      :pass1-greedy         ; 1.0/0.0 — did the greedy (idx 0) sample pass?
      :passk}               ; unbiased pass@k over the sampled verdicts at `k`
   When there is no idx-0 verdict (greedy not generated), :pass1-greedy falls back to
   pass-at-k over the sampled set at k=1."
  [verdicts k]
  (let [greedy   (first (filter #(= 0 (:sample-idx %)) verdicts))
        sampled  (filter #(pos? (or (:sample-idx %) 0)) verdicts)
        n        (count sampled)
        c        (count (filter :kept? sampled))
        kk       (min k (max 1 n))]
    {:n n
     :c c
     :pass1-greedy (cond
                      greedy        (if (:kept? greedy) 1.0 0.0)
                      (pos? n)      (pass-at-k n c 1)
                      :else         0.0)
     :passk (pass-at-k n c kk)}))

;; ===========================================================================
;; 4. Eval reporting — baseline-vs-SFT table + cold-start guard
;; ===========================================================================

(defn- by-task [verdicts] (group-by :task-id verdicts))

(defn- mean [xs] (if (seq xs) (/ (reduce + xs) (count xs)) 0.0))

(defn eval-report
  "Build the baseline-vs-SFT comparison from graded eval verdicts. `baseline` and `sft`
   are verdict seqs over the SAME held-out eval tasks (each verdict {:task-id :kind
   :sample-idx :kept? ...}). `tasks-by-id` maps task-id → seed task (for :kind / order).
   Returns:
     {:k k
      :tasks [{:task-id :kind
               :baseline {:n :c :pass1-greedy :passk}
               :sft      {...}
               :delta-pass1 :delta-passk
               :cold-start?      ; baseline pass@k 0 AND greedy fails → the base policy never
                                 ;   reaches a correct solution at all. SFT may still lift it by
                                 ;   generalizing from train-task demos; a reward-only method could not.
               :sft-cold?}]      ; SFT pass@k 0 AND greedy fails → even after SFT it is unreachable,
                                 ;   so the downstream GRPO step (genmlx-2ctu) has no reward signal to
                                 ;   sharpen — THIS is what bounds the loop's achievable ceiling.
      :aggregate {:baseline-pass1 :sft-pass1 :delta-pass1
                  :baseline-passk :sft-passk :delta-passk
                  :n-cold-start :n-sft-cold :n-tasks}
      :by-kind {<kind> {:baseline-pass1 :sft-pass1 :delta-pass1 ...}}}"
  [baseline sft k tasks-by-id]
  (let [b-by   (by-task baseline)
        s-by   (by-task sft)
        ;; Preserve first-appearance order from the verdicts. The candidates are generated
        ;; in task-export order and collect-verdicts returns them sorted by row index, so
        ;; first-appearance IS the canonical task order. Do NOT sort by (vals tasks-by-id):
        ;; a >8-key CLJS map iterates in hash order, not insertion order (genmlx-o8w9 review).
        ids    (distinct (concat (map :task-id baseline) (map :task-id sft)))
        rows   (for [id ids
                     :let [bk (passk-of (get b-by id) k)
                           sk (passk-of (get s-by id) k)
                           kind (name (:kind (get tasks-by-id id) :unknown))]]
                 {:task-id      id
                  :kind         kind
                  :baseline     bk
                  :sft          sk
                  :delta-pass1 (- (:pass1-greedy sk) (:pass1-greedy bk))
                  :delta-passk (- (:passk sk) (:passk bk))
                  ;; "cold" = genuinely unreachable: NEITHER the temperature samples (passk)
                  ;; NOR the greedy sample reach a correct solution. A greedy-only success is
                  ;; reachable (just not by the sampled set), so it is NOT cold (genmlx-o8w9 review).
                  :cold-start?  (and (zero? (:passk bk)) (zero? (:pass1-greedy bk)))
                  :sft-cold?    (and (zero? (:passk sk)) (zero? (:pass1-greedy sk)))})
        agg-fn (fn [sel rs] (mean (map sel rs)))]
    {:k k
     :tasks (vec rows)
     :aggregate
     {:n-tasks         (count rows)
      :baseline-pass1 (agg-fn (comp :pass1-greedy :baseline) rows)
      :sft-pass1      (agg-fn (comp :pass1-greedy :sft) rows)
      :delta-pass1    (agg-fn :delta-pass1 rows)
      :baseline-passk (agg-fn (comp :passk :baseline) rows)
      :sft-passk      (agg-fn (comp :passk :sft) rows)
      :delta-passk    (agg-fn :delta-passk rows)
      :n-cold-start    (count (filter :cold-start? rows))
      :n-sft-cold      (count (filter :sft-cold? rows))}
     :by-kind
     (into {}
           (for [[kind rs] (group-by :kind rows)]
             [kind {:n-tasks         (count rs)
                    :baseline-pass1 (agg-fn (comp :pass1-greedy :baseline) rs)
                    :sft-pass1      (agg-fn (comp :pass1-greedy :sft) rs)
                    :delta-pass1    (agg-fn :delta-pass1 rs)
                    :baseline-passk (agg-fn (comp :passk :baseline) rs)
                    :sft-passk      (agg-fn (comp :passk :sft) rs)
                    :delta-passk    (agg-fn :delta-passk rs)}]))}))
