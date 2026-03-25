(ns genmlx.gfi-contract-test
  "GFI contract tests across canonical models."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Canonical models
;; ---------------------------------------------------------------------------

(def single-site
  (dyn/auto-key
    (gen []
      (let [x (trace :x (dist/gaussian 0 1))]
        (mx/eval! x)
        (mx/item x)))))

(def multi-site
  (dyn/auto-key
    (gen []
      (let [x (trace :x (dist/gaussian 0 1))]
        (mx/eval! x)
        (let [xv (mx/item x)
              y (trace :y (dist/gaussian xv 1))]
          (mx/eval! y)
          (mx/item y))))))

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

(defn run-contract-tests [label model args]
  ;; Contract 1: simulate validity
  (let [trace (p/simulate model args)]
    (is (some? (:choices trace)) (str label ": simulate has choices"))
    (is (some? (:gen-fn trace)) (str label ": simulate has gen-fn"))
    (let [score (:score trace)]
      (mx/eval! score)
      (is (js/isFinite (mx/item score)) (str label ": simulate finite score")))

    ;; Contract 2: generate full constraints -> weight ~ score
    (let [{:keys [trace weight]} (p/generate model args (:choices trace))]
      (mx/eval! (:score trace) weight)
      (is (h/close? (mx/item (:score trace)) (mx/item weight) 0.01)
          (str label ": generate(all) weight ~ score")))

    ;; Contract 3: assess = generate score
    (let [choices (:choices trace)
          {:keys [weight]} (p/assess model args choices)
          {:keys [trace]} (p/generate model args choices)]
      (mx/eval! weight (:score trace))
      (is (h/close? (mx/item (:score trace)) (mx/item weight) 0.01)
          (str label ": assess weight ~ generate score")))

    ;; Contract 4: generate empty constraints -> weight ~ 0
    (let [{:keys [weight]} (p/generate model args cm/EMPTY)]
      (mx/eval! weight)
      (is (h/close? 0.0 (mx/item weight) 0.01)
          (str label ": generate(empty) weight ~ 0")))

    ;; Contract 5: update no-op -> weight ~ 0
    (let [{:keys [weight]} (p/update model trace (:choices trace))]
      (mx/eval! weight)
      (is (h/close? 0.0 (mx/item weight) 0.01)
          (str label ": update(same) weight ~ 0")))

    ;; Contract 6: update round-trip via discard
    (let [orig-choices (:choices trace)
          new-val (mx/scalar 42.0)
          constraint (cm/choicemap :x new-val)
          {:keys [trace discard weight]} (p/update model trace constraint)]
      (when discard
        (let [x-new (get-choice-val (:choices trace) :x)]
          (when x-new
            (is (h/close? 42.0 x-new 0.01)
                (str label ": update sets new value"))))
        (let [{:keys [trace]} (p/update model trace discard)
              x-recovered (get-choice-val (:choices trace) :x)]
          (when x-recovered
            (let [x-orig (get-choice-val orig-choices :x)]
              (when x-orig
                (is (h/close? x-orig x-recovered 0.01)
                    (str label ": update round-trip recovers value"))))))))

    ;; Contract 7: regenerate empty selection -> weight ~ 0, choices unchanged
    (let [orig-x (get-choice-val (:choices trace) :x)
          {:keys [trace weight]} (p/regenerate model trace sel/none)]
      (mx/eval! weight)
      (is (h/close? 0.0 (mx/item weight) 0.01)
          (str label ": regenerate(none) weight ~ 0"))
      (when orig-x
        (let [new-x (get-choice-val (:choices trace) :x)]
          (when new-x
            (is (h/close? orig-x new-x 0.01)
                (str label ": regenerate(none) preserves :x"))))))

    ;; Contract 8: propose -> generate round-trip
    (let [{:keys [choices weight]} (p/propose model args)]
      (mx/eval! weight)
      (let [{:keys [trace weight]} (p/generate model args choices)]
        (mx/eval! weight)
        (is (js/isFinite (mx/item weight))
            (str label ": propose weight is finite"))))

    ;; Contract 9: project(all) ~ score
    (let [proj (p/project model trace sel/all)]
      (mx/eval! proj (:score trace))
      (is (h/close? (mx/item (:score trace)) (mx/item proj) 0.01)
          (str label ": project(all) ~ score")))

    ;; Contract 10: project(none) ~ 0
    (let [proj (p/project model trace sel/none)]
      (mx/eval! proj)
      (is (h/close? 0.0 (mx/item proj) 0.01)
          (str label ": project(none) ~ 0")))))

;; ---------------------------------------------------------------------------
;; Run contracts on each model
;; ---------------------------------------------------------------------------

(deftest single-site-contracts
  (testing "single-site"
    (run-contract-tests "single-site" single-site [])))

(deftest multi-site-contracts
  (testing "multi-site"
    (run-contract-tests "multi-site" multi-site [])))

(deftest linreg-contracts
  (testing "linreg"
    (run-contract-tests "linreg" linreg [[1.0 2.0 3.0]])))

(deftest splice-contracts
  (testing "splice"
    (run-contract-tests "splice" splice-model [])))

(deftest mixed-contracts
  (testing "mixed"
    (run-contract-tests "mixed" mixed-model [])))

(cljs.test/run-tests)
