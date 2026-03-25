(ns genmlx.custom-gradient-test
  "CustomGradientGF tests."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.custom-gradient :as cg]
            [genmlx.gradients :as grad])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Basic GFI operations
;; ---------------------------------------------------------------------------

(deftest basic-gfi-operations
  (testing "basic GFI operations"
    (let [gf (cg/custom-gradient-gf
               {:forward (fn [x y] (mx/add (mx/multiply x y) (mx/scalar 1.0)))
                :has-argument-grads [true true]})
          x (mx/scalar 3.0)
          y (mx/scalar 5.0)
          args [x y]]

      ;; simulate
      (let [trace (p/simulate gf args)]
        (mx/eval! (:retval trace) (:score trace))
        (is (h/close? 16.0 (mx/item (:retval trace)) 1e-5) "simulate retval = x*y+1 = 16")
        (is (h/close? 0.0 (mx/item (:score trace)) 1e-5) "simulate score = 0")
        (is (= (:choices trace) cm/EMPTY) "simulate choices are empty"))

      ;; generate
      (let [{:keys [trace weight]} (p/generate gf args cm/EMPTY)]
        (mx/eval! (:retval trace) weight)
        (is (h/close? 16.0 (mx/item (:retval trace)) 1e-5) "generate retval = 16")
        (is (h/close? 0.0 (mx/item weight) 1e-5) "generate weight = 0"))

      ;; assess
      (let [{:keys [retval weight]} (p/assess gf args cm/EMPTY)]
        (mx/eval! retval weight)
        (is (h/close? 16.0 (mx/item retval) 1e-5) "assess retval = 16")
        (is (h/close? 0.0 (mx/item weight) 1e-5) "assess weight = 0"))

      ;; propose
      (let [{:keys [choices weight retval]} (p/propose gf args)]
        (mx/eval! retval weight)
        (is (h/close? 16.0 (mx/item retval) 1e-5) "propose retval = 16")
        (is (h/close? 0.0 (mx/item weight) 1e-5) "propose weight = 0")
        (is (= choices cm/EMPTY) "propose choices are empty")))))

;; ---------------------------------------------------------------------------
;; has-argument-grads protocol
;; ---------------------------------------------------------------------------

(deftest has-argument-grads-protocol
  (testing "has-argument-grads protocol"
    (let [gf-with-grads (cg/custom-gradient-gf
                          {:forward (fn [x] x)
                           :has-argument-grads [true false true]})]
      (is (= [true false true] (p/has-argument-grads gf-with-grads)) "CustomGradientGF returns arg-grads vector")
      (is (cg/accepts-arg-grads? gf-with-grads) "accepts-arg-grads? is true"))

    (let [gf-no-grads (cg/custom-gradient-gf
                         {:forward (fn [x] x)})]
      (is (nil? (p/has-argument-grads gf-no-grads)) "CustomGradientGF with no arg-grads returns nil")
      (is (not (cg/accepts-arg-grads? gf-no-grads)) "accepts-arg-grads? is false when nil"))

    ;; DynamicGF returns nil
    (let [dgf (gen [x] x)]
      (is (nil? (p/has-argument-grads dgf)) "DynamicGF has-argument-grads returns nil")
      (is (not (cg/accepts-arg-grads? dgf)) "accepts-arg-grads? is false for DynamicGF"))

    ;; Distribution returns nil
    (let [d (dist/gaussian 0 1)]
      (is (nil? (p/has-argument-grads d)) "Distribution has-argument-grads returns nil")
      (is (not (cg/accepts-arg-grads? d)) "accepts-arg-grads? is false for Distribution"))))

;; ---------------------------------------------------------------------------
;; Custom gradient function
;; ---------------------------------------------------------------------------

(deftest custom-gradient-function
  (testing "custom gradient function"
    (let [gf (cg/custom-gradient-gf
               {:forward (fn [x] (mx/multiply x x))
                :gradient (fn [args _retval cotangent]
                            [(mx/multiply (mx/multiply (mx/scalar 2.0) (first args))
                                          cotangent)])
                :has-argument-grads [true]})
          grad-result ((:gradient-fn gf) [(mx/scalar 4.0)] (mx/scalar 16.0) (mx/scalar 1.0))]
      (mx/eval! (first grad-result))
      (is (h/close? 8.0 (mx/item (first grad-result)) 1e-5) "custom gradient at x=4 is 2*4=8"))))

;; ---------------------------------------------------------------------------
;; Use in model via splice -- gradient flow
;; ---------------------------------------------------------------------------

(deftest gradient-flow-through-splice
  (testing "gradient flow through spliced CustomGradientGF"
    (let [transform (cg/custom-gradient-gf
                      {:forward (fn [x] (mx/add (mx/multiply (mx/scalar 2.0) x)
                                                (mx/scalar 3.0)))
                       :has-argument-grads [true]})
          model (dyn/auto-key (gen [x]
                  (let [tx (splice :transform transform x)]
                    (trace :y (dist/gaussian tx 1))
                    tx)))
          x (mx/scalar 5.0)
          obs (cm/choicemap :y (mx/scalar 13.0))]
      (let [{:keys [trace weight]} (p/generate model [x] obs)]
        (mx/eval! (:retval trace) weight)
        (is (h/close? 13.0 (mx/item (:retval trace)) 1e-5) "spliced model retval = 2*5+3 = 13")
        (is (h/close? -0.9189 (mx/item weight) 0.01) "generate weight = log gaussian(13; 13, 1)")))))

;; ---------------------------------------------------------------------------
;; Score gradient with spliced CustomGradientGF
;; ---------------------------------------------------------------------------

(deftest score-gradient-with-custom-gf
  (testing "score gradient with CustomGradientGF in model"
    (let [multiply-gf (cg/custom-gradient-gf
                        {:forward (fn [a b] (mx/multiply a b))
                         :has-argument-grads [true true]})
          model (dyn/auto-key (gen [x]
                  (let [slope (trace :slope (dist/gaussian 0 10))
                        pred  (splice :mul multiply-gf slope x)]
                    (trace :y (dist/gaussian pred 1))
                    pred)))
          x (mx/scalar 2.0)
          obs (cm/choicemap :y (mx/scalar 6.0))
          {:keys [trace weight]} (p/generate model [x] obs)
          _ (mx/eval! (:retval trace) weight)
          grads (grad/choice-gradients model trace [:slope])]
      (is (contains? grads :slope) "choice gradient for :slope exists")
      (let [g (mx/item (get grads :slope))]
        (is (> (js/Math.abs g) 0.001) "gradient for :slope is nonzero")))))

(cljs.test/run-tests)
