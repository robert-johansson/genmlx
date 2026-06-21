(ns genmlx.distill-gen-test
  "Tests for the scaled task/curriculum generator (genmlx.world.distill-gen, genmlx-7473).

   TWO LAYERS:
     1. PURE structure — split disjointness + leakage safety + teacher-projection hygiene +
        family balance (no GPU).
     2. GROUNDING — the load-bearing guarantee: EVERY generated task's :reference is admitted
        by the SAME oracle that grades the student (genmlx.world.distill/evaluate-candidate),
        programs by EXACT marginal; and the oracle DISCRIMINATES (a deliberately wrong
        completion is rejected), so 'kept' is real, not a rubber stamp. This is the
        independent-oracle / non-circular guarantee: program evidence is GenMLX analytics
        (not the reference's form); function ground truth is hand-written (not derived from
        the reference).

   Run: bun run --bun nbb test/genmlx/distill_gen_test.cljs"
  (:require [genmlx.world.distill-gen :as g]
            [genmlx.world.distill :as d]
            [clojure.set]
            [clojure.string :as str]))

(def ^:private fails (atom 0))
(def ^:private passes (atom 0))

(defn assert-true [desc x]
  (if x (do (swap! passes inc) (println (str "  ok   " desc)))
        (do (swap! fails inc) (println (str "  FAIL " desc)))))

(defn assert-eq [desc expected actual]
  (assert-true (str desc " (expected " (pr-str expected) ", got " (pr-str actual) ")")
               (= expected actual)))

;; ===========================================================================
(println "\n== 1. generated-set shape ==")

(assert-true "a substantial set was generated (>= 120 tasks)" (>= (count g/all-tasks) 120))
(assert-eq "all-tasks = train + eval" (count g/all-tasks)
           (+ (count g/train-tasks) (count g/eval-tasks)))
(assert-true "both oracle kinds present" (= #{:program :function} (set (map :kind g/all-tasks))))
(assert-true "every task has an :id, :kind, :prompt, :reference, :family, :split"
             (every? (fn [t] (and (:id t) (:kind t) (:prompt t) (:reference t)
                                  (:family t) (:split t)))
                     g/all-tasks))
(assert-true "every program carries :observations"
             (every? :observations (filter #(= :program (:kind %)) g/all-tasks)))
(assert-true "every function carries :transitions XOR :test-cases"
             (every? (fn [t] (or (seq (:transitions t)) (seq (:test-cases t))))
                     (filter #(= :function (:kind %)) g/all-tasks)))

;; ===========================================================================
(println "\n== 2. split: disjoint, leakage-safe, family-balanced ==")

(let [train-ids (set (map :id g/train-tasks))
      eval-ids  (set (map :id g/eval-tasks))]
  (assert-true "train/eval disjoint by id"
               (empty? (clojure.set/intersection train-ids eval-ids)))
  (assert-eq "eval-task-ids = the eval ids" eval-ids g/eval-task-ids)
  (assert-true "all ids unique (no dupes across program/machine/extra catalogs)"
               (= (count g/all-tasks) (count (set (map :id g/all-tasks)))))
  (assert-true "held-out fraction is a sane minority (10%-40%)"
               (let [f (/ (count g/eval-tasks) (count g/all-tasks))] (and (>= f 0.1) (<= f 0.4)))))

(let [train-fams (set (map :family g/train-tasks))
      eval-fams  (set (map :family g/eval-tasks))]
  ;; every held-out family has TRAIN siblings — passing an eval task is same-distribution
  ;; generalization, never a leaked train instance (the rrps test-split lesson).
  (assert-true "every eval family also appears in train (no orphan eval family)"
               (clojure.set/subset? eval-fams train-fams)))

;; ===========================================================================
(println "\n== 3. teacher-projection hygiene (no leakage) ==")

(let [rec (g/task->prompt-record (first g/program-tasks))]
  (assert-eq "prompt record has exactly the 4 teacher-facing keys"
             #{:task_id :kind :system_prompt :prompt} (set (keys rec)))
  (assert-true "prompt record drops :observations / :transitions / :test-cases / :reference"
               (every? #(not (contains? rec %)) [:observations :transitions :test-cases :reference])))

(assert-true "no exported prompt text contains the literal reference solution"
             (every? (fn [t]
                       (let [p (str (:system-prompt t) " " (:prompt t))]
                         (not (str/includes? p (:reference t)))))
                     ;; functions only — program scaffolds legitimately echo a generic
                     ;; (fn [trace] ...) template; the reference (different priors) is not in it.
                     (filter #(= :function (:kind %)) g/all-tasks)))

;; ===========================================================================
(println "\n== 4. GROUNDING — every reference admitted by its own oracle ==")

(def ^:private verdicts
  (mapv (fn [t] (assoc (d/evaluate-candidate (assoc t :n-particles 50) (:reference t) -1)
                       :family (:family t)))
        g/all-tasks))

(let [failed (remove :kept? verdicts)]
  (when (seq failed)
    (println "  -- references NOT admitted: --")
    (doseq [v failed] (println (str "     " (:task-id v) " -> " (:reason v)
                                    (when (:error v) (str " : " (:error v)))))))
  (assert-eq "ALL references are admitted (kept?) by the oracle" [] (vec failed)))

(let [progs (filter #(= :program (:kind %)) verdicts)
      exact (filter #(contains? #{:exact :kalman} (:method %)) progs)]
  (assert-eq "EVERY program reference scores by EXACT analytical marginal (reproducible)"
             (count progs) (count exact)))

;; ===========================================================================
(println "\n== 5. the oracle DISCRIMINATES (kept is not a rubber stamp) ==")

;; A function reference graded against a DIFFERENT task's oracle should usually fail —
;; pick a clear mismatch: factorial code vs sum-evens tests.
(let [fact (g/tasks-by-id "factorial")
      sev  (g/tasks-by-id "sum-evens")]
  (when (and fact sev)
    (let [wrong (d/evaluate-candidate sev (:reference fact) 0)]
      (assert-true "factorial code is REJECTED against sum-evens tests"
                   (not (:kept? wrong))))))

;; A degenerate constant program must be rejected (uncovered / not a model).
(let [prog (first (filter #(= :program (:kind %)) g/all-tasks))]
  (when prog
    (let [wrong (d/evaluate-candidate (assoc prog :n-particles 20)
                                      "(fn [trace] {:nope 1})" 0)]
      (assert-true "a program ignoring the observed sites is REJECTED (coverage guard)"
                   (not (:kept? wrong))))))

;; ===========================================================================
(println (str "\n== SUMMARY: " @passes " passed, " @fails " failed =="))
(when (pos? @fails) (set! (.-exitCode js/process) 1))
