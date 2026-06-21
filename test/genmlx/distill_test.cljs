;; @tier slow
(ns genmlx.distill-test
  "Acceptance for genmlx.world.distill — the OFFLINE oracle-filter (the batch twin of
   genmlx.world.train-reward) that turns teacher completions into an SFT corpus (genmlx-j0d6).

   Deterministic, no teacher LLM (conjugate programs score the EXACT analytical
   marginal; functions are checked behaviorally). Run:
     bun run --bun nbb test/genmlx/distill_test.cljs

   PARTS:
     A — the gate ladder discriminates: valid+correct kept, garbage / non-function /
         data-ignoring / wrong dropped with the right :reason; program evidence ranks
         a good fit above a bad fit and matches an INDEPENDENT closed-form oracle.
     B — rank-and-select keeps the top-k per prompt by evidence/accuracy.
     C — verdicts->stats computes the corpus-quality rates; build-sft-records emits
         well-formed Qwen3 chat rows whose assistant turn is the validated code.
     D — every one of the 12 seed tasks is SOLVABLE (a reference solution is kept) and
         DISCRIMINATING (a trivial wrong/empty answer is dropped) — the tasks are real."
  (:require [genmlx.world.distill :as d]
            [genmlx.world.distill-tasks :as t]
            [genmlx.world.train-reward :as tr]
            [genmlx.llm.msa-score :as msa]
            [clojure.string :as str]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))

(defn assert-true [label v]
  (if v
    (do (swap! pass inc) (println "  PASS" label))
    (do (swap! fail inc) (println "  FAIL" label))))

(defn assert-close [label expected actual tol]
  (let [ok (and (number? actual) (js/isFinite actual) (<= (js/Math.abs (- expected actual)) tol))]
    (if ok
      (do (swap! pass inc) (println "  PASS" label (str "(" actual " ~ " expected ")")))
      (do (swap! fail inc) (println "  FAIL" label (str "(" actual " vs " expected " tol " tol ")"))))))

;; Independent closed-form gaussian-gaussian marginal (copied from the reward test —
;; DERIVED outside GenMLX, so the program-evidence check is not circular).
(defn gaussian-gaussian-marginal [ys m0 s0 sn]
  (let [n (count ys) s0² (* s0 s0) sn² (* sn sn)
        ds (map #(- % m0) ys) sd (reduce + ds) sd² (reduce + (map #(* % %) ds))
        denom (+ sn² (* n s0²))
        logdet (+ (js/Math.log denom) (* (dec n) (js/Math.log sn²)))
        quad (* (/ 1.0 sn²) (- sd² (* (/ s0² denom) (* sd sd))))]
    (- (* -0.5 n (js/Math.log (* 2 js/Math.PI))) (* 0.5 logdet) (* 0.5 quad))))

;; ===========================================================================
;; PART A — the gate ladder discriminates
;; ===========================================================================

(def ^:private gm-task (t/tasks-by-id "gaussian-mean-near2"))
(def ^:private gm-obs (:observations gm-task))
;; a well-fit shared-mean model (prior near the data, tight noise), wrapped in the
;; noise an extractor must peel (think block + fence + trailing prose).
(def ^:private good-model
  "(fn [trace] (let [mu (trace :mu (dist/gaussian 2 1))] {:y0 (trace :y0 (dist/gaussian mu 0.5)) :y1 (trace :y1 (dist/gaussian mu 0.5)) :y2 (trace :y2 (dist/gaussian mu 0.5)) :y3 (trace :y3 (dist/gaussian mu 0.5))}))")
(def ^:private good-completion
  (str "<think> the data sits near 2 ( a stray paren</think>\n```clojure\n" good-model "\n```\nDone."))
;; a covered-but-badly-fit model (prior mean 50, data near 2): finite, kept, low evidence.
(def ^:private bad-model
  "(fn [trace] (let [mu (trace :mu (dist/gaussian 50 1))] {:y0 (trace :y0 (dist/gaussian mu 0.5)) :y1 (trace :y1 (dist/gaussian mu 0.5)) :y2 (trace :y2 (dist/gaussian mu 0.5)) :y3 (trace :y3 (dist/gaussian mu 0.5))}))")
;; declares a latent but IGNORES the data (traces none of :y0..:y3).
(def ^:private ignoring-model "(fn [trace] (let [mu (trace :mu (dist/gaussian 0 5))] {:result mu}))")

(defn part-a []
  (println "\n== PART A: gate ladder discrimination ==")

  ;; ---- program path ----
  (let [v (d/evaluate-candidate gm-task good-completion 0)]
    (assert-true "good program: kept" (:kept? v))
    (assert-true "good program: reason :kept" (= :kept (:reason v)))
    (assert-true "good program: parsed + evaluated + covered"
                 (and (:parse? v) (:eval? v) (:covered? v)))
    (assert-true "good program: finite model evidence as :log-ml (= :rank-key)"
                 (and (js/isFinite (:log-ml v)) (= (:log-ml v) (:rank-key v))))
    (assert-true "good program: extracted code is the (fn ...) form (noise peeled)"
                 (and (str/starts-with? (:code v) "(fn") (str/includes? (:code v) ":y3")))
    ;; independent oracle: gaussian-gaussian marginal with m0=2 s0=1 sn=0.5
    (let [oracle (gaussian-gaussian-marginal (map gm-obs [:y0 :y1 :y2 :y3]) 2.0 1.0 0.5)]
      (println "    program evidence =" (:log-ml v) " method=" (:method v) " oracle=" oracle)
      (assert-close "good program evidence ~ independent closed-form marginal"
                    oracle (:log-ml v) 1.0)))

  (let [vbad (d/evaluate-candidate gm-task bad-model 0)
        vgood (d/evaluate-candidate gm-task good-model 0)]
    (assert-true "bad-fit program is still kept (covered + finite)" (:kept? vbad))
    (assert-true "good-fit program out-ranks bad-fit (higher model evidence)"
                 (> (:rank-key vgood) (:rank-key vbad))))

  (let [v (d/evaluate-candidate gm-task ignoring-model 0)]
    (assert-true "data-ignoring program dropped (coverage guard)"
                 (and (not (:kept? v)) (= :uncovered (:reason v)))))
  (assert-true "garbage program dropped as :unparseable"
               (= :unparseable (:reason (d/evaluate-candidate gm-task "this is not (((" 0))))
  (assert-true "empty completion dropped as :empty"
               (= :empty (:reason (d/evaluate-candidate gm-task "" 0))))
  (assert-true "a non-model (parses + evaluates, but not a fn) dropped as :eval-error"
               (= :eval-error (:reason (d/evaluate-candidate gm-task "(+ 1 2)" 0))))

  ;; ---- function path: transitions ----
  (let [task (t/tasks-by-id "counter-machine")
        correct "(fn [state action] (case action :inc (update state :count inc) :dec (update state :count dec) :reset (assoc state :count 0) state))"
        wrong   "(fn [state action] state)"
        vgood (d/evaluate-candidate task correct 0)
        vbad  (d/evaluate-candidate task wrong 0)]
    (assert-true "correct transition fn: kept, accuracy 1.0"
                 (and (:kept? vgood) (= 1.0 (:accuracy vgood)) (= 1.0 (:rank-key vgood))))
    (assert-true "wrong transition fn: dropped :test-fail, accuracy < 1"
                 (and (not (:kept? vbad)) (= :test-fail (:reason vbad)) (< (:accuracy vbad) 1.0)))
    (assert-true "un-evaluable transition fn: dropped :eval-error"
                 (= :eval-error (:reason (d/evaluate-candidate task "(fn [state action] (boom!))" 0)))))

  ;; ---- function path: test-cases, incl. a NAMED self-recursive solution ----
  (let [task (t/tasks-by-id "factorial")
        recursive "(defn factorial [n] (if (<= n 1) 1 (* n (factorial (dec n)))))"
        wrong     "(fn [n] (* n n))"
        vrec (d/evaluate-candidate task recursive 0)]
    (assert-true "named self-recursive factorial: kept (the defn name is preserved, not anonymized)"
                 (:kept? vrec))
    (assert-true "wrong factorial: dropped :test-fail"
                 (= :test-fail (:reason (d/evaluate-candidate task wrong 0))))))

;; ===========================================================================
;; PART B — rank-and-select
;; ===========================================================================

(defn part-b []
  (println "\n== PART B: rank-and-select top-k per prompt ==")
  ;; three candidates for one task: good, bad-fit, and garbage; top-1 = good.
  (let [verdicts [(d/evaluate-candidate gm-task good-model 0)
                  (d/evaluate-candidate gm-task bad-model 1)
                  (d/evaluate-candidate gm-task "nonsense" 2)]
        top1 (d/rank-and-select verdicts 1)
        top2 (d/rank-and-select verdicts 2)]
    (assert-true "top-1 selects exactly one survivor" (= 1 (count top1)))
    (assert-true "top-1 is the higher-evidence (good-fit) candidate"
                 (= 0 (:sample-idx (first top1))))
    (assert-true "top-2 keeps both valid survivors (garbage excluded)"
                 (and (= 2 (count top2)) (every? :kept? top2))))
  ;; selection is grouped per task: two tasks, top-1 each -> 2 selected.
  (let [ct (t/tasks-by-id "counter-machine")
        verdicts [(d/evaluate-candidate gm-task good-model 0)
                  (d/evaluate-candidate ct "(fn [state action] (case action :inc (update state :count inc) :dec (update state :count dec) :reset (assoc state :count 0) state))" 0)]
        sel (d/rank-and-select verdicts 1)]
    (assert-true "top-1-per-task across 2 tasks -> 2 selected"
                 (= 2 (count (distinct (map :task-id sel)))))))

;; ===========================================================================
;; PART C — stats + SFT records
;; ===========================================================================

(defn part-c []
  (println "\n== PART C: stats + SFT records ==")
  (let [tasks [{:id "p1" :kind :program} {:id "p2" :kind :program}
               {:id "f1" :kind :function}]
        verdicts [{:task-id "p1" :kind :program :parse? true :eval? true :kept? true :reason :kept :log-ml -5.0}
                  {:task-id "p1" :kind :program :parse? true :eval? true :kept? true :reason :kept :log-ml -7.0}
                  {:task-id "p2" :kind :program :parse? true :eval? true :kept? false :reason :uncovered}
                  {:task-id "f1" :kind :function :parse? true :eval? true :kept? true :reason :kept}
                  {:task-id "f1" :kind :function :parse? false :kept? false :reason :unparseable}]
        selected (filter :kept? verdicts)
        s (d/verdicts->stats tasks verdicts selected)]
    (assert-true "n-candidates counted" (= 5 (:n-candidates s)))
    (assert-true "n-tasks counted" (= 3 (:n-tasks s)))
    (assert-close "parse-rate = 4/5" 0.8 (:parse-rate s) 1e-9)
    (assert-close "eval-rate = 4/5" 0.8 (:eval-rate s) 1e-9)
    (assert-close "program-pass-rate = 2/3 (kept programs / program candidates)"
                  (/ 2 3) (:program-pass-rate s) 1e-9)
    (assert-close "function-pass-rate = 1/2" 0.5 (:function-pass-rate s) 1e-9)
    (assert-close "mean-log-ml over kept programs = -6.0" -6.0 (:mean-log-ml s) 1e-9)
    (assert-close "yield-per-prompt = 2/3 prompts produced >=1 kept" (/ 2 3) (:yield-per-prompt s) 1e-9)
    (assert-true "drop-reasons histogram present (3 kept: p1x2 + f1)"
                 (= 3 (get (:drop-reasons s) :kept))))

  ;; build-sft-records: chat shape + assistant turn = the validated code.
  (let [v (d/evaluate-candidate gm-task good-model 0)
        recs (d/build-sft-records t/tasks-by-id [v])
        r (first recs)
        msgs (:messages r)]
    (assert-true "one SFT record per selected verdict" (= 1 (count recs)))
    (assert-true "record carries provenance (task-id + kind)"
                 (and (= "gaussian-mean-near2" (:task-id r)) (= "program" (:kind r))))
    (assert-true "messages are [system user assistant] with string roles"
                 (= ["system" "user" "assistant"] (mapv :role msgs)))
    (assert-true "user turn is the task prompt"
                 (= (:prompt gm-task) (:content (nth msgs 1))))
    (assert-true "assistant turn is the ORACLE-VALIDATED code (not the raw completion)"
                 (= (:code v) (:content (nth msgs 2))))))

;; ===========================================================================
;; PART D — every seed task is solvable + discriminating
;; ===========================================================================

(def ^:private reference-solutions
  {"gaussian-mean-near2"  "(fn [trace] (let [mu (trace :mu (dist/gaussian 2 1))] {:y0 (trace :y0 (dist/gaussian mu 0.5)) :y1 (trace :y1 (dist/gaussian mu 0.5)) :y2 (trace :y2 (dist/gaussian mu 0.5)) :y3 (trace :y3 (dist/gaussian mu 0.5))}))"
   "gaussian-mean-negshift" "(fn [trace] (let [mu (trace :mu (dist/gaussian -3 1))] {:y0 (trace :y0 (dist/gaussian mu 0.5)) :y1 (trace :y1 (dist/gaussian mu 0.5)) :y2 (trace :y2 (dist/gaussian mu 0.5)) :y3 (trace :y3 (dist/gaussian mu 0.5))}))"
   "beta-bernoulli-coin"  "(fn [trace] (let [p (trace :p (dist/beta 4 2))] {:f0 (trace :f0 (dist/bernoulli p)) :f1 (trace :f1 (dist/bernoulli p)) :f2 (trace :f2 (dist/bernoulli p)) :f3 (trace :f3 (dist/bernoulli p)) :f4 (trace :f4 (dist/bernoulli p)) :f5 (trace :f5 (dist/bernoulli p))}))"
   "gamma-poisson-counts" "(fn [trace] (let [rate (trace :rate (dist/gamma 4 1))] {:c0 (trace :c0 (dist/poisson rate)) :c1 (trace :c1 (dist/poisson rate)) :c2 (trace :c2 (dist/poisson rate)) :c3 (trace :c3 (dist/poisson rate)) :c4 (trace :c4 (dist/poisson rate))}))"
   "counter-machine"      "(fn [state action] (case action :inc (update state :count inc) :dec (update state :count dec) :reset (assoc state :count 0) state))"
   "traffic-light"        "(fn [state action] (assoc state :light (case (:light state) :red :green :green :yellow :yellow :red)))"
   "toggle-switch"        "(fn [state action] (if (= action :flip) (update state :on not) state))"
   "factorial"            "(defn factorial [n] (if (<= n 1) 1 (* n (factorial (dec n)))))"
   "fizzbuzz"             "(fn [n] (cond (zero? (mod n 15)) \"FizzBuzz\" (zero? (mod n 3)) \"Fizz\" (zero? (mod n 5)) \"Buzz\" :else (str n)))"
   "gcd"                  "(fn [a b] (if (zero? b) a (recur b (mod a b))))"
   "palindrome?"          "(fn [s] (= s (apply str (reverse s))))"
   "sum-evens"            "(fn [coll] (reduce + (filter even? coll)))"})

;; a trivially-wrong answer per task (covered/parseable where needed, but incorrect),
;; to prove each task's hidden oracle actually DISCRIMINATES (rejects a plausible dud).
(def ^:private wrong-answers
  {"counter-machine" "(fn [state action] state)"
   "traffic-light"   "(fn [state action] state)"
   "toggle-switch"   "(fn [state action] (assoc state :on true))"
   "factorial"       "(fn [n] n)"
   "fizzbuzz"        "(fn [n] (str n))"
   "gcd"             "(fn [a b] 1)"
   "palindrome?"     "(fn [s] true)"
   "sum-evens"       "(fn [coll] (reduce + coll))"})

(defn part-d []
  (println "\n== PART D: every seed task is solvable + discriminating ==")
  (assert-true "seed set has 12 tasks" (= 12 (count t/tasks)))
  (assert-true "task->prompt-record drops the held-out oracle signal (no test leakage)"
               (let [r (t/task->prompt-record (t/tasks-by-id "factorial"))]
                 (and (contains? r :prompt) (not (contains? r :test-cases))
                      (not (contains? r :transitions)) (not (contains? r :observations)))))
  (doseq [task t/tasks
          :let [id (:id task)
                v (d/evaluate-candidate task (reference-solutions id) 0)]]
    (assert-true (str "[" id "] reference solution is KEPT (task is solvable)") (:kept? v)))
  (doseq [[id ans] wrong-answers
          :let [v (d/evaluate-candidate (t/tasks-by-id id) ans 0)]]
    (assert-true (str "[" id "] a wrong answer is DROPPED (oracle discriminates)")
                 (not (:kept? v)))))

;; ===========================================================================
;; PART E — corpus-poisoning guards (from the adversarial review)
;; ===========================================================================

;; a delta point-mass model: covers every observed address but asserts the data is
;; deterministic (log-evidence ~0, would out-rank any honest noisy model) — a hack.
(def ^:private delta-hack
  "(fn [trace] {:y0 (trace :y0 (dist/delta 2.0)) :y1 (trace :y1 (dist/delta 2.3)) :y2 (trace :y2 (dist/delta 1.7)) :y3 (trace :y3 (dist/delta 2.1))})")

(defn part-e []
  (println "\n== PART E: corpus-poisoning guards ==")

  ;; ---- :no-oracle — a :function task with no held-out tests keeps NOTHING ----
  (let [bad-task {:id "no-oracle" :kind :function :prompt "p"}
        v (d/evaluate-candidate bad-task "(fn [state action] state)" 0)]
    (assert-true "a :function task with neither transitions nor test-cases keeps no candidate"
                 (and (not (:kept? v)) (= :no-oracle (:reason v))))
    (assert-true "empty :test-cases [] is also treated as no-oracle (not a vacuous pass)"
                 (= :no-oracle (:reason (d/evaluate-candidate
                                          {:id "e" :kind :function :test-cases []} "(fn [n] n)" 0)))))

  ;; ---- :degenerate — a delta point-mass program is rejected ----
  (let [v (d/evaluate-candidate gm-task delta-hack 0)]
    (assert-true "delta point-mass program rejected (covered, but degenerate)"
                 (and (not (:kept? v)) (= :degenerate (:reason v)))))

  ;; ---- :low-evidence — the optional absolute floor drops a covered-but-awful fit ----
  (let [vgood (d/evaluate-candidate (assoc gm-task :min-log-ml -10.0) good-model 0)
        vbad  (d/evaluate-candidate (assoc gm-task :min-log-ml -10.0) bad-model 0)]
    (assert-true "min-log-ml floor keeps a good fit (evidence above floor)" (:kept? vgood))
    (assert-true "min-log-ml floor drops a covered-but-awful fit as :low-evidence"
                 (and (not (:kept? vbad)) (= :low-evidence (:reason vbad)))))

  ;; ---- exact-over-IS ranking tier: a deterministic exact marginal must win the
  ;;      top-k slot over a higher-but-noisy importance-sampling estimate ----
  (let [verdicts [{:task-id "p" :kind :program :kept? true :method :handler-is :rank-key -3.0 :code "noisy"}
                  {:task-id "p" :kind :program :kept? true :method :exact :rank-key -5.0 :code "exact"}]
        top1 (d/rank-and-select verdicts 1)]
    (assert-true "kept program records its scoring :method (= :exact for a conjugate model)"
                 (= :exact (:method (d/evaluate-candidate gm-task good-model 0))))
    (assert-true "top-1 prefers the EXACT-scored program over a higher-but-noisy-IS one"
                 (= :exact (:method (first top1)))))

  ;; ---- stats expose task-space coverage alongside the per-prompt hit-rate ----
  (let [tasks [{:id "p1" :kind :program} {:id "p2" :kind :program} {:id "f1" :kind :function}]
        verdicts [{:task-id "p1" :kind :program :kept? true :reason :kept :log-ml -5.0 :method :exact}
                  {:task-id "f1" :kind :function :kept? true :reason :kept}]
        s (d/verdicts->stats tasks verdicts (filter :kept? verdicts))]
    (assert-close "task-space-coverage = 2/3 of the full seed set" (/ 2 3) (:task-space-coverage s) 1e-9)
    (assert-true "n-selected-noisy-is is 0 when no IS-scored program was selected"
                 (= 0 (:n-selected-noisy-is s)))))

;; ===========================================================================

(defn- summary []
  (println (str "\n== distill (genmlx-j0d6): " @pass " passed, " @fail " failed =="))
  (when (pos? @fail) (set! (.-exitCode js/process) 1)))

(part-a)
(part-b)
(part-c)
(part-d)
(part-e)
(summary)
