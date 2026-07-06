;; @tier medium
(ns genmlx.curriculum-test
  "Well-posedness + integration tests for the REPL-synthesis curriculum generator
   (genmlx.world.curriculum, bean genmlx-ilna). NATIVE-FREE: uses the exact oracle
   (synth/check) but never the policy LLM.

   Every assertion is one the curriculum's value depends on:
     - PRNG correctness (the signed-shift / NaN bugs the critique flagged)
     - per-task well-posedness (crude<bar<gold, exact+reproducible oracle, no structure
       leak in the task-desc, ground-truth re-scores to gold)
     - the complexity SPREAD (the resource-rational training signal)
     - leakage-safe split, proven END-TO-END through repl-corpus (a planted held-out
       trajectory is actually DROPPED — the rrps no-op-guard failure mode)
     - consumer compatibility (->probe-task; a multi-step family-proposer harvest)"
  (:require [genmlx.world.curriculum :as cur]
            [genmlx.world.synth :as syn]
            [genmlx.world.repl-corpus :as rc]
            [clojure.set :as set]
            [clojure.string :as str]))

;; ---------------------------------------------------------------------------
;; Minimal assertion helpers (project convention — no framework).
;; ---------------------------------------------------------------------------
(def ^:dynamic *fails* (atom 0))
(def ^:dynamic *passes* (atom 0))
(defn assert-true [msg x]
  (if x (swap! *passes* inc)
      (do (swap! *fails* inc) (println "  FAIL:" msg))))
(defn assert-close [msg expected actual tol]
  (assert-true (str msg " (exp " expected " got " actual ")")
               (and (number? actual) (< (js/Math.abs (- expected actual)) tol))))

;; ===========================================================================
;; 1. PRNG correctness — the signed-shift / NaN failure modes.
;; ===========================================================================
(println "\n-- PRNG --")
(let [us (cur/uniforms 7 5000)
      ns (cur/normals 9 5000)
      mean (fn [xs] (/ (reduce + xs) (count xs)))
      std  (fn [xs] (let [m (mean xs)] (js/Math.sqrt (mean (map #(* (- % m) (- % m)) xs)))))]
  (assert-true "uniforms all in [0,1) (catches signed-shift negatives)"
               (every? #(and (>= % 0.0) (< % 1.0)) us))
  (assert-true "uniforms reproducible (same seed)" (= (cur/uniforms 123 200) (cur/uniforms 123 200)))
  (assert-true "uniforms differ across seeds" (not= (cur/uniforms 1 50) (cur/uniforms 2 50)))
  (assert-close "uniform mean ~ 0.5 (catches skew from arithmetic shift)" 0.5 (mean us) 0.03)
  (assert-true "normals all finite (catches Box-Muller log(0) NaN)" (every? js/isFinite ns))
  (assert-true "normals reproducible" (= (cur/normals 55 200) (cur/normals 55 200)))
  (assert-close "normal mean ~ 0" 0.0 (mean ns) 0.06)
  (assert-close "normal std ~ 1" 1.0 (std ns) 0.06))
(let [s1 (cur/mix-seed 0 1 2 3) s2 (cur/mix-seed 0 1 2 4) s3 (cur/mix-seed 0 1 2 3)]
  (assert-true "mix-seed deterministic" (= s1 s3))
  (assert-true "mix-seed distinguishes tuples" (not= s1 s2))
  (assert-true "mix-seed is a 32-bit int" (and (integer? s1) (<= (js/Math.abs s1) 4294967296))))
(assert-close "round1 rounds down below .05" 1.0 (cur/round1 1.0499999) 1e-9)
(assert-close "round1 rounds up at/above .05" 1.1 (cur/round1 1.06) 1e-9)

;; ===========================================================================
;; 2. Generate a curriculum exercising BOTH eval cohorts.
;; ===========================================================================
(println "\n-- generating curriculum (6/family, stride 3) --")
(def C (cur/generate-curriculum {:round 0 :instances-per-family 6 :eval-stride 3}))
(def tasks (:tasks C))
(println "  " (pr-str (:summary C)))
(assert-true "curriculum is non-empty" (pos? (count tasks)))
(assert-true "covers all 6 families" (= 6 (count (group-by :family tasks))))
(assert-true "both eval cohorts present"
             (and (pos? (:eval-within (:summary C))) (pos? (:eval-family (:summary C)))))

;; ===========================================================================
;; 3. Per-task well-posedness (the core spec).
;; ===========================================================================
(println "\n-- per-task well-posedness --")
(def banned-structure
  ["slope" "intercept" "linear" "regress" "coefficient" "autoreg" "markov" "pool"
   "hierarch" "latent" "prior" "gaussian" "distribution" "variance" "covariance"
   "mean" "noise" "scale" "drift"])
(defn rounded-1dp? [x] (< (js/Math.abs (- x (cur/round1 x))) 1e-9))

(doseq [t tasks]
  (let [{:keys [id family crude crude-tuned gold solve-bar gap struct-gap exact? method
                task-desc observations ground-truth-code complexity structural?]} t
        struct? (get (cur/family-by-key family) :structural? true)]
    ;; oracle quality
    (assert-true (str id " exact? true") exact?)
    (assert-true (str id " method exact/kalman") (contains? #{:exact :kalman} method))
    ;; bar strictly brackets, with margin
    (assert-true (str id " crude < solve-bar < gold") (and (< crude solve-bar) (< solve-bar gold)))
    (assert-true (str id " gap >= min-gap") (>= gap cur/default-min-gap))
    (when struct?
      (assert-true (str id " struct-gap >= min (structure beats best structureless)")
                   (>= struct-gap cur/default-min-struct-gap))
      ;; the core admissibility guarantee: the best STRUCTURELESS noise-tuned model must
      ;; NOT clear the bar, so 'solved' provably means found-the-structure (not noise tuning).
      (assert-true (str id " best structureless model does NOT clear the bar")
                   (< crude-tuned solve-bar)))
    ;; ground-truth code re-scores to gold (tolerance, float32) + parses + covers
    (let [fb (syn/check ground-truth-code observations {:n-particles 0})]
      (assert-true (str id " ground-truth parses+covers+scored") (syn/scored? fb))
      (assert-close (str id " ground-truth re-scores to gold") gold (:evidence fb)
                    (+ 1e-3 (* 1e-4 (js/Math.abs gold))))
      (assert-true (str id " ground-truth clears its own bar") (>= (:evidence fb) solve-bar)))
    ;; task-desc semantics-only (no structural vocabulary)
    (let [low (str/lower-case task-desc)]
      (assert-true (str id " task-desc omits structural words")
                   (not-any? #(str/includes? low %) banned-structure)))
    ;; observations rounded; id is a stable string
    (assert-true (str id " observations rounded to 1dp") (every? rounded-1dp? (vals observations)))
    (assert-true (str id " :id is a string") (string? id))))

;; global id uniqueness
(assert-true "all task ids unique" (= (count tasks) (count (distinct (map :id tasks)))))

;; ===========================================================================
;; 4. The complexity SPREAD — the resource-rational training signal.
;; ===========================================================================
(println "\n-- complexity spread --")
(let [bc (:by-complexity C)
      cs (sort (keys bc))
      n-lats (map #(:mean-n-latents (bc %)) cs)]
  (assert-true "spans >= 3 complexity bands" (>= (count cs) 3))
  (assert-true "mean n-latents monotone non-decreasing in complexity"
               (apply <= n-lats))
  (assert-true "hardest band has a bigger mean gap than the easiest band"
               (> (:mean-gap (bc (last cs))) (:mean-gap (bc (first cs)))))
  (doseq [c cs]
    (println "  c" c " n=" (:n (bc c)) " mean-gap=" (.toFixed (:mean-gap (bc c)) 2)
             " mean-n-lat=" (.toFixed (:mean-n-latents (bc c)) 2))))

;; ===========================================================================
;; 5. Reproducibility — same opts ⇒ byte-identical observations.
;; ===========================================================================
(println "\n-- reproducibility --")
(let [C2 (cur/generate-curriculum {:round 0 :instances-per-family 6 :eval-stride 3})
      by-id  (into {} (map (juxt :id :observations) tasks))
      by-id2 (into {} (map (juxt :id :observations) (:tasks C2)))]
  (assert-true "same ids regenerated" (= (set (keys by-id)) (set (keys by-id2))))
  (assert-true "observations byte-identical across runs" (= by-id by-id2)))
(let [Cr1 (cur/generate-curriculum {:round 1 :instances-per-family 3})]
  (assert-true "ReST-EM round changes the data (fresh batch)"
               (not= (map :observations (:tasks Cr1))
                     (take (count (:tasks Cr1)) (map :observations tasks)))))

;; ===========================================================================
;; 6. Leakage-safe split — proven END-TO-END through repl-corpus.
;; ===========================================================================
(println "\n-- leakage-safe split (end-to-end) --")
(let [eval-ids (:eval-task-ids C)
      train-ids (into #{} (map (comp name :id)) (:train-tasks C))]
  (assert-true "eval-task-ids are strings ((name id) form)"
               (every? string? eval-ids))
  (assert-true "eval-task-ids = (name id) of every eval task"
               (= eval-ids (into #{} (map (comp name :id)) (:eval-tasks C))))
  (assert-true "train/eval task-id sets are disjoint"
               (empty? (set/intersection eval-ids train-ids)))
  (assert-true "held-out family (:segmented) is entirely eval"
               (every? #(= :eval (:split %)) (filter #(= :segmented (:family %)) tasks))))

;; Build a real trajectory for one held-out (eval) task + one train task, harvest with
;; the curriculum's eval-ids, and assert the planted eval rows are DROPPED (not silently
;; kept) — the no-op-guard failure the rrps lesson is about.
(defn harvest-run [task]
  (let [obs (:observations task)
        res (syn/synthesize {:init-spec (cur/crude-spec obs) :observations obs
                             :propose (cur/family-proposer task) :max-steps 8})]
    {:task task :trajectory (:trajectory res) :steps (:steps res)
     :final (get-in res [:feedback :evidence])}))

(let [eval-task  (first (filter #(= :segmented (:family %)) (:eval-tasks C)))  ; held-out FAMILY
      train-task (first (filter #(= :varying-slopes (:family %)) (:train-tasks C)))
      runs (mapv harvest-run [eval-task train-task])
      corpus (rc/build-corpus runs {:eval-ids (:eval-task-ids C)})]
  (println "  eval task" (:id eval-task) "trajectory steps:" (:steps (first runs)))
  (println "  train task" (:id train-task) "trajectory steps:" (:steps (second runs)))
  (println "  corpus:" (pr-str (select-keys corpus [:n-runs :n-rows :train-task-ids])))
  (assert-true "harvest produced rows" (pos? (:n-rows corpus)))
  (assert-true "planted EVAL-task rows are DROPPED (not silently kept)"
               (seq (:dropped-eval corpus)))
  (assert-true "no train-row belongs to a held-out eval task"
               (not-any? #(contains? (:eval-task-ids C) (:task-id %)) (:train-rows corpus)))
  (assert-true "the train task's rows ARE in train-rows"
               (some #(= (name (:id train-task)) (:task-id %)) (:train-rows corpus)))
  ;; close the multi-step harvest claim THROUGH build-corpus (not just at trajectory level):
  ;; the >1-step train trajectory must yield >1 SFT row.
  (assert-true "multi-step trajectory harvests >1 row for the train task"
               (> (count (filter #(= (name (:id train-task)) (:task-id %)) (:train-rows corpus))) 1)))

;; ===========================================================================
;; 7. Consumer adapter + multi-step family-proposer harvest.
;; ===========================================================================
(println "\n-- consumers --")
(let [t (first (filter #(= :linear (:family %)) tasks))
      p (cur/->probe-task t)]
  (assert-true "->probe-task :obs == :observations" (= (:observations t) (:obs p)))
  (assert-true "->probe-task carries probe knobs"
               (and (= 0 (:np p)) (= (:solve-bar t) (:solve-bar p))
                    (= (:crude t) (:crude p)) (= (:gold t) (:gold p)))))

;; the family-proposer must yield a MULTI-step trajectory (structure + noise refinement)
;; on a structural task, and the loop must SOLVE it (final evidence clears the bar) —
;; proving the curriculum is consumable by the loop without the LLM.
(let [t   (first (filter #(= :varying-slopes (:family %)) tasks))
      res (syn/synthesize {:init-spec (cur/crude-spec (:observations t))
                           :observations (:observations t)
                           :propose (cur/family-proposer t) :max-steps 8})]
  (println "  vslope family-proposer: steps=" (:steps res) " final=" (.toFixed (get-in res [:feedback :evidence]) 2)
           " bar=" (.toFixed (:solve-bar t) 2))
  (assert-true "family-proposer trajectory is multi-step (>1 accepted edit)" (> (:steps res) 1))
  (assert-true "family-proposer solves the task (clears the bar)"
               (>= (get-in res [:feedback :evidence]) (:solve-bar t))))

;; ===========================================================================
(println (str "\n=== curriculum_test: " @*passes* " passed, " @*fails* " failed ==="))
(when (pos? @*fails*) (js/process.exit 1))
