(ns genmlx.bugfix-test
  "Tests for three bug fixes:
   1. Update weight correctness for dependent variables
   2. Vectorized switch produces distinct samples
   3. Conditional SMC uses the reference trace"
  (:require [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dynamic :as dyn]
            [genmlx.gen :refer [gen]]
            [genmlx.dist :as dist]
            [genmlx.combinators :as comb]
            [genmlx.inference.smc :as smc]))

;; ---------------------------------------------------------------------------
;; Test helpers
;; ---------------------------------------------------------------------------

(def ^:dynamic *failures* (atom 0))

(defn assert-true [msg pred]
  (if pred
    (println (str "  PASS: " msg))
    (do (println (str "  FAIL: " msg))
        (swap! *failures* inc))))

(defn assert-close [msg expected actual tol]
  (let [ok (< (js/Math.abs (- actual expected)) tol)]
    (if ok
      (println (str "  PASS: " msg " (" actual " ≈ " expected ")"))
      (do (println (str "  FAIL: " msg " (got " actual ", expected " expected ", tol " tol ")"))
          (swap! *failures* inc)))))

(println "\n=== Bug Fix Tests ===\n")

;; =========================================================================
;; Test 1: Update weight correctness
;; =========================================================================
;; The old bug: update-transition only tracked weight changes at constrained
;; addresses. When unconstrained addresses depend on constrained ones, the
;; likelihood change was missing from the weight.
;;
;; Model: mu ~ N(0, 10), obs ~ N(mu, 1)
;; If we update mu from old_mu to new_mu, the weight should include:
;;   log N(obs; new_mu, 1) - log N(obs; old_mu, 1)
;; (the likelihood change at the obs address)

(println "-- Update weight: dependent variables --")

(let [;; Model: mu drawn, then obs depends on mu
      model (gen [obs-val]
              (let [mu (dyn/trace :mu (dist/gaussian 0 10))]
                (dyn/trace :obs (dist/gaussian mu 1))
                mu))

      ;; Generate a trace with mu=0 and obs=5
      obs-val 5.0
      init-constraints (cm/choicemap :mu (mx/scalar 0.0) :obs (mx/scalar obs-val))
      {:keys [trace]} (p/generate model [obs-val] init-constraints)
      old-score (:score trace)

      ;; Update: change mu from 0 to 5 (closer to obs=5)
      new-constraints (cm/choicemap :mu (mx/scalar 5.0))
      {:keys [trace weight]} (p/update model trace new-constraints)
      new-score (:score trace)

      ;; The correct weight = new_score - old_score
      ;; This should include the likelihood change at :obs
      expected-weight (- (mx/realize new-score) (mx/realize old-score))
      actual-weight (mx/realize weight)]

  (assert-close "update weight = new_score - old_score"
                expected-weight actual-weight 0.001)

  ;; Verify the weight is positive: moving mu from 0 to 5 when obs=5
  ;; should increase the likelihood (obs is closer to mu)
  (assert-true "update weight > 0 (mu moved toward obs)"
               (> actual-weight 0)))

;; Test with MapCombinator: same fix should propagate
(println "\n-- Update weight: MapCombinator --")

(let [kernel (gen [x]
              (let [mu (dyn/trace :mu (dist/gaussian 0 10))]
                (dyn/trace :obs (dist/gaussian mu 1))
                mu))
      model (comb/map-combinator kernel)
      args [[0.0 0.0]]  ;; 2 elements
      ;; Generate with specific values
      constraints (cm/choicemap
                    0 (cm/choicemap :mu (mx/scalar 1.0) :obs (mx/scalar 5.0))
                    1 (cm/choicemap :mu (mx/scalar 2.0) :obs (mx/scalar 5.0)))
      {:keys [trace]} (p/generate model args constraints)
      old-score (:score trace)

      ;; Update: change mu[0] from 1 to 5, mu[1] from 2 to 5
      new-constraints (cm/choicemap
                        0 (cm/choicemap :mu (mx/scalar 5.0))
                        1 (cm/choicemap :mu (mx/scalar 5.0)))
      {:keys [trace weight]} (p/update model trace new-constraints)
      new-score (:score trace)

      expected-weight (- (mx/realize new-score) (mx/realize old-score))
      actual-weight (mx/realize weight)]

  (assert-close "map update weight = new_score - old_score"
                expected-weight actual-weight 0.001)
  (assert-true "map update weight > 0 (mu moved toward obs)"
               (> actual-weight 0)))

;; Test with ScanCombinator
(println "\n-- Update weight: ScanCombinator --")

(let [kernel (gen [carry input]
              (let [x (dyn/trace :x (dist/gaussian carry 1))]
                (dyn/trace :obs (dist/gaussian x 0.5))
                [x x]))
      model (comb/scan-combinator kernel)
      args [(mx/scalar 0.0) [1 2]]  ;; init carry=0, 2 steps
      constraints (cm/choicemap
                    0 (cm/choicemap :x (mx/scalar 1.0) :obs (mx/scalar 3.0))
                    1 (cm/choicemap :x (mx/scalar 2.0) :obs (mx/scalar 3.0)))
      {:keys [trace]} (p/generate model args constraints)
      old-score (:score trace)

      ;; Update: change x[0] from 1 to 3, closer to obs=3
      new-constraints (cm/choicemap
                        0 (cm/choicemap :x (mx/scalar 3.0)))
      {:keys [trace weight]} (p/update model trace new-constraints)
      new-score (:score trace)

      expected-weight (- (mx/realize new-score) (mx/realize old-score))
      actual-weight (mx/realize weight)]

  (assert-close "scan update weight = new_score - old_score"
                expected-weight actual-weight 0.001))

;; =========================================================================
;; Test 2: Vectorized switch produces distinct samples
;; =========================================================================
;; The old bug: p/simulate was called once per branch (scalar), so all N
;; particles selecting the same branch got identical values.

(println "\n-- Vectorized switch: distinct samples --")

(let [branches [(dist/gaussian 0 1) (dist/gaussian 10 1)]
      n 20
      ;; All particles select branch 0
      index (mx/zeros [n] mx/int32)
      result (comb/vectorized-switch branches index [])
      values (cm/get-value (:choices result))
      ;; Extract individual values
      vals-list (mx/->clj values)]

  ;; With N=20 draws from N(0,1), they should NOT all be identical
  (assert-true "vectorized switch: values are [N]-shaped"
               (= (mx/shape values) [n]))

  ;; Check that values are distinct (not all the same)
  (let [unique-count (count (set vals-list))]
    (assert-true "vectorized switch: values are distinct (not identical)"
                 (> unique-count 1)))

  ;; Mean should be near 0 (branch 0 = N(0,1))
  (let [mean-val (mx/realize (mx/mean values))]
    (assert-true "vectorized switch: mean near 0 for branch 0"
                 (< (js/Math.abs mean-val) 2.0))))

;; Mixed branch indices
(println "\n-- Vectorized switch: mixed branch selection --")

(let [branches [(dist/gaussian 0 0.1) (dist/gaussian 100 0.1)]
      n 10
      ;; First 5 particles select branch 0, last 5 select branch 1
      index (mx/array [0 0 0 0 0 1 1 1 1 1] mx/int32)
      result (comb/vectorized-switch branches index [])
      values (cm/get-value (:choices result))
      vals-list (mx/->clj values)
      branch0-vals (take 5 vals-list)
      branch1-vals (drop 5 vals-list)]

  ;; Branch 0 values should be near 0
  (assert-true "branch 0 values near 0"
               (< (/ (reduce + branch0-vals) 5) 5.0))

  ;; Branch 1 values should be near 100
  (assert-true "branch 1 values near 100"
               (> (/ (reduce + branch1-vals) 5) 90.0))

  ;; Scores should be [N]-shaped
  (assert-true "vectorized switch: scores are [N]-shaped"
               (= (mx/shape (:score result)) [n])))

;; =========================================================================
;; Test 3: Conditional SMC uses reference trace
;; =========================================================================
;; The old bug: csmc generated a fresh trace instead of using the provided
;; reference-trace. The reference trace was completely ignored.

(println "\n-- Conditional SMC: reference trace used --")

(let [;; Simple model
      model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 10))]
                (dyn/trace :obs (dist/gaussian x 1))
                x))

      ;; Create a reference trace with a distinctive x value
      ref-x 42.0
      ref-constraints (cm/choicemap :x (mx/scalar ref-x) :obs (mx/scalar 42.5))
      {:keys [trace]} (p/generate model [] ref-constraints)
      reference-trace trace

      ;; Run cSMC with this reference trace
      obs (cm/choicemap :obs (mx/scalar 42.5))
      result (smc/csmc {:particles 10 :key (rng/fresh-key)}
                       model [] [obs] reference-trace)

      ;; The reference particle (index 0) should have x = 42.0
      ref-particle-trace (first (:traces result))
      ref-x-val (mx/realize (cm/get-choice (:choices ref-particle-trace) [:x]))]

  (assert-close "cSMC: reference particle x = 42.0"
                ref-x ref-x-val 0.001)

  ;; Verify other particles have different x values (sampled from prior)
  (let [other-xs (mapv (fn [t]
                         (mx/realize (cm/get-choice (:choices t) [:x])))
                       (rest (:traces result)))
        ;; At least some should differ significantly from 42
        different-count (count (filter #(> (js/Math.abs (- % ref-x)) 5) other-xs))]
    (assert-true "cSMC: other particles differ from reference"
                 (> different-count 0))))

;; =========================================================================
;; Summary
;; =========================================================================

(println (str "\n=== Bug Fix Test Results ==="))
(println (str "  Failures: " @*failures*))
(when (pos? @*failures*)
  (js/process.exit 1))
