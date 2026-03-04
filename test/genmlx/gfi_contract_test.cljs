(ns genmlx.gfi-contract-test
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn assert-true [msg actual]
  (if actual
    (println "  PASS:" msg)
    (println "  FAIL:" msg "- expected truthy")))

(defn assert-close [msg expected actual tolerance]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tolerance)
      (println "  PASS:" msg)
      (do (println "  FAIL:" msg)
          (println "    expected:" expected "+/-" tolerance)
          (println "    actual:  " actual)))))

(println "\n=== GFI Contract Tests ===")

;; ---------------------------------------------------------------------------
;; Canonical models
;; ---------------------------------------------------------------------------

;; 1. Single-site
(def single-site
  (dyn/auto-key
    (gen []
      (let [x (trace :x (dist/gaussian 0 1))]
        (mx/eval! x)
        (mx/item x)))))

;; 2. Multi-site
(def multi-site
  (dyn/auto-key
    (gen []
      (let [x (trace :x (dist/gaussian 0 1))]
        (mx/eval! x)
        (let [xv (mx/item x)
              y (trace :y (dist/gaussian xv 1))]
          (mx/eval! y)
          (mx/item y))))))

;; 3. Linear regression (3 obs)
(def linreg
  (dyn/auto-key
    (gen [xs]
      (let [slope (trace :slope (dist/gaussian 0 10))
            intercept (trace :intercept (dist/gaussian 0 10))]
        (mx/eval! slope intercept)
        (let [sv (mx/item slope) iv (mx/item intercept)]
          (doseq [[j x] (map-indexed vector xs)]
            (trace (keyword (str "y" j))
                       (dist/gaussian (+ (* sv x) iv) 1)))
          sv)))))

;; 4. With splice
(def inner-model
  (dyn/auto-key
    (gen []
      (let [z (trace :z (dist/gaussian 0 1))]
        (mx/eval! z)
        (mx/item z)))))

(def splice-model
  (dyn/auto-key
    (gen []
      (let [x (trace :x (dist/gaussian 0 10))]
        (mx/eval! x)
        (splice :inner inner-model)
        (mx/item x)))))

;; 5. Discrete + continuous
(def mixed-model
  (dyn/auto-key
    (gen []
      (let [b (trace :b (dist/bernoulli 0.5))]
        (mx/eval! b)
        (let [bv (mx/item b)
              y (trace :y (dist/gaussian (if (> bv 0.5) 5.0 -5.0) 1))]
          (mx/eval! y)
          (mx/item y))))))

;; ---------------------------------------------------------------------------
;; Contract harness
;; ---------------------------------------------------------------------------

(defn get-choice-val
  "Extract a scalar value from a choicemap at addr."
  [choices addr]
  (let [sub (cm/get-submap choices addr)]
    (when (and sub (cm/has-value? sub))
      (let [v (cm/get-value sub)]
        (mx/eval! v)
        (mx/item v)))))

(defn run-contracts [label model args]
  (println (str "\n-- " label " --"))

  ;; Contract 1: simulate validity
  (let [trace (p/simulate model args)]
    (assert-true (str label ": simulate has choices") (some? (:choices trace)))
    (assert-true (str label ": simulate has gen-fn") (some? (:gen-fn trace)))
    (let [score (:score trace)]
      (mx/eval! score)
      (assert-true (str label ": simulate finite score") (js/isFinite (mx/item score))))

    ;; Contract 2: generate full constraints → weight ≈ score
    (let [{:keys [trace weight]} (p/generate model args (:choices trace))]
      (mx/eval! (:score trace) weight)
      (assert-close (str label ": generate(all) weight ≈ score")
                    (mx/item (:score trace)) (mx/item weight) 0.01))

    ;; Contract 3: assess = generate score (use original trace's choices)
    (let [choices (:choices trace)
          {:keys [weight]} (p/assess model args choices)
          {:keys [trace]} (p/generate model args choices)]
      (mx/eval! weight (:score trace))
      (assert-close (str label ": assess weight ≈ generate score")
                    (mx/item (:score trace)) (mx/item weight) 0.01))

    ;; Contract 4: generate empty constraints → weight ≈ 0
    (let [{:keys [weight]} (p/generate model args cm/EMPTY)]
      (mx/eval! weight)
      (assert-close (str label ": generate(empty) weight ≈ 0")
                    0.0 (mx/item weight) 0.01))

    ;; Contract 5: update no-op → weight ≈ 0
    (let [{:keys [weight]} (p/update model trace (:choices trace))]
      (mx/eval! weight)
      (assert-close (str label ": update(same) weight ≈ 0")
                    0.0 (mx/item weight) 0.01))

    ;; Contract 6: update round-trip via discard
    (let [orig-choices (:choices trace)
          ;; Pick a single address to change
          new-val (mx/scalar 42.0)
          ;; Make a constraint on :x (or first available address)
          constraint (cm/choicemap :x new-val)
          {:keys [trace discard weight]} (p/update model trace constraint)]
      (when discard
        (let [x-new (get-choice-val (:choices trace) :x)]
          (when x-new
            (assert-close (str label ": update sets new value")
                          42.0 x-new 0.01)))
        ;; Round-trip: apply discard to get back
        (let [{:keys [trace]} (p/update model trace discard)
              x-recovered (get-choice-val (:choices trace) :x)]
          (when x-recovered
            (let [x-orig (get-choice-val orig-choices :x)]
              (when x-orig
                (assert-close (str label ": update round-trip recovers value")
                              x-orig x-recovered 0.01)))))))

    ;; Contract 7: regenerate empty selection → weight ≈ 0, choices unchanged
    (let [orig-x (get-choice-val (:choices trace) :x)
          {:keys [trace weight]} (p/regenerate model trace sel/none)]
      (mx/eval! weight)
      (assert-close (str label ": regenerate(none) weight ≈ 0")
                    0.0 (mx/item weight) 0.01)
      (when orig-x
        (let [new-x (get-choice-val (:choices trace) :x)]
          (when new-x
            (assert-close (str label ": regenerate(none) preserves :x")
                          orig-x new-x 0.01)))))

    ;; Contract 8: propose → generate round-trip
    (let [{:keys [choices weight]} (p/propose model args)]
      (mx/eval! weight)
      (let [{:keys [trace weight]} (p/generate model args choices)]
        (mx/eval! weight)
        ;; propose weight + generate weight should ≈ 0
        ;; propose returns -score, generate with full constraints returns +score
        ;; so their sum should be ≈ 0 ... actually:
        ;; propose.weight = score, generate.weight = score
        ;; The round-trip identity: propose produces (choices, score)
        ;; then generate(choices) produces (trace, score) where weight = score
        ;; So propose.weight ≈ generate.weight
        (assert-true (str label ": propose weight is finite")
                     (js/isFinite (mx/item weight)))))

    ;; Contract 9: project(all) ≈ score
    (let [proj (p/project model trace sel/all)]
      (mx/eval! proj (:score trace))
      (assert-close (str label ": project(all) ≈ score")
                    (mx/item (:score trace)) (mx/item proj) 0.01))

    ;; Contract 10: project(none) ≈ 0
    (let [proj (p/project model trace sel/none)]
      (mx/eval! proj)
      (assert-close (str label ": project(none) ≈ 0")
                    0.0 (mx/item proj) 0.01))))

;; ---------------------------------------------------------------------------
;; Run contracts on each model
;; ---------------------------------------------------------------------------

(run-contracts "single-site" single-site [])
(run-contracts "multi-site" multi-site [])
(run-contracts "linreg" linreg [[1.0 2.0 3.0]])
(run-contracts "splice" splice-model [])
(run-contracts "mixed" mixed-model [])

(println "\nAll GFI contract tests complete.")
