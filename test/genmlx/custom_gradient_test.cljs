(ns genmlx.custom-gradient-test
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.custom-gradient :as cg]
            [genmlx.gradients :as grad])
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

(println "\n=== CustomGradientGF Tests ===\n")

;; ---------------------------------------------------------------------------
;; Basic GFI operations
;; ---------------------------------------------------------------------------

(println "-- Basic GFI operations --")

(let [;; f(x, y) = x * y + 1
      gf (cg/custom-gradient-gf
           {:forward (fn [x y] (mx/add (mx/multiply x y) (mx/scalar 1.0)))
            :has-argument-grads [true true]})
      x (mx/scalar 3.0)
      y (mx/scalar 5.0)
      args [x y]]

  ;; simulate
  (let [trace (p/simulate gf args)]
    (mx/eval! (:retval trace) (:score trace))
    (assert-close "simulate retval = x*y+1 = 16"
                  16.0 (mx/item (:retval trace)) 1e-5)
    (assert-close "simulate score = 0"
                  0.0 (mx/item (:score trace)) 1e-5)
    (assert-true "simulate choices are empty"
                 (= (:choices trace) cm/EMPTY)))

  ;; generate
  (let [{:keys [trace weight]} (p/generate gf args cm/EMPTY)]
    (mx/eval! (:retval trace) weight)
    (assert-close "generate retval = 16"
                  16.0 (mx/item (:retval trace)) 1e-5)
    (assert-close "generate weight = 0"
                  0.0 (mx/item weight) 1e-5))

  ;; assess
  (let [{:keys [retval weight]} (p/assess gf args cm/EMPTY)]
    (mx/eval! retval weight)
    (assert-close "assess retval = 16"
                  16.0 (mx/item retval) 1e-5)
    (assert-close "assess weight = 0"
                  0.0 (mx/item weight) 1e-5))

  ;; propose
  (let [{:keys [choices weight retval]} (p/propose gf args)]
    (mx/eval! retval weight)
    (assert-close "propose retval = 16"
                  16.0 (mx/item retval) 1e-5)
    (assert-close "propose weight = 0"
                  0.0 (mx/item weight) 1e-5)
    (assert-true "propose choices are empty"
                 (= choices cm/EMPTY))))

;; ---------------------------------------------------------------------------
;; has-argument-grads protocol
;; ---------------------------------------------------------------------------

(println "\n-- has-argument-grads protocol --")

(let [gf-with-grads (cg/custom-gradient-gf
                      {:forward (fn [x] x)
                       :has-argument-grads [true false true]})]
  (assert-true "CustomGradientGF returns arg-grads vector"
               (= [true false true] (p/has-argument-grads gf-with-grads)))
  (assert-true "accepts-arg-grads? is true"
               (cg/accepts-arg-grads? gf-with-grads)))

(let [gf-no-grads (cg/custom-gradient-gf
                     {:forward (fn [x] x)})]
  (assert-true "CustomGradientGF with no arg-grads returns nil"
               (nil? (p/has-argument-grads gf-no-grads)))
  (assert-true "accepts-arg-grads? is false when nil"
               (not (cg/accepts-arg-grads? gf-no-grads))))

;; DynamicGF returns nil
(let [dgf (gen [x] x)]
  (assert-true "DynamicGF has-argument-grads returns nil"
               (nil? (p/has-argument-grads dgf)))
  (assert-true "accepts-arg-grads? is false for DynamicGF"
               (not (cg/accepts-arg-grads? dgf))))

;; Distribution returns nil
(let [d (dist/gaussian 0 1)]
  (assert-true "Distribution has-argument-grads returns nil"
               (nil? (p/has-argument-grads d)))
  (assert-true "accepts-arg-grads? is false for Distribution"
               (not (cg/accepts-arg-grads? d))))

;; ---------------------------------------------------------------------------
;; Custom gradient function
;; ---------------------------------------------------------------------------

(println "\n-- Custom gradient function --")

(let [;; f(x) = x^2, with explicit gradient 2*x
      gf (cg/custom-gradient-gf
           {:forward (fn [x] (mx/multiply x x))
            :gradient (fn [args _retval cotangent]
                        ;; d/dx(x^2) = 2x, scaled by cotangent
                        [(mx/multiply (mx/multiply (mx/scalar 2.0) (first args))
                                      cotangent)])
            :has-argument-grads [true]})
      x (mx/scalar 4.0)
      ;; Call the gradient function directly
      grad-result ((:gradient-fn gf) [(mx/scalar 4.0)] (mx/scalar 16.0) (mx/scalar 1.0))]
  (mx/eval! (first grad-result))
  (assert-close "custom gradient at x=4 is 2*4=8"
                8.0 (mx/item (first grad-result)) 1e-5))

;; ---------------------------------------------------------------------------
;; Use in model via splice — gradient flow
;; ---------------------------------------------------------------------------

(println "\n-- Gradient flow through spliced CustomGradientGF --")

(let [;; Deterministic transform: f(x) = 2*x + 3
      transform (cg/custom-gradient-gf
                  {:forward (fn [x] (mx/add (mx/multiply (mx/scalar 2.0) x)
                                            (mx/scalar 3.0)))
                   :has-argument-grads [true]})
      ;; Model that uses the transform
      model (gen [x]
              (let [tx (dyn/splice :transform transform x)]
                ;; Observe y ~ gaussian(transform(x), 1)
                (dyn/trace :y (dist/gaussian tx 1))
                tx))
      x (mx/scalar 5.0)
      obs (cm/choicemap :y (mx/scalar 13.0))]  ;; 2*5+3 = 13

  ;; Generate should work
  (let [{:keys [trace weight]} (p/generate model [x] obs)]
    (mx/eval! (:retval trace) weight)
    (assert-close "spliced model retval = 2*5+3 = 13"
                  13.0 (mx/item (:retval trace)) 1e-5)
    ;; Weight should be log p(y=13 | mu=13, sigma=1) = log gaussian(13; 13, 1)
    ;; = -0.5*log(2*pi) - 0.5*(0)^2 = -0.9189...
    (assert-close "generate weight = log gaussian(13; 13, 1)"
                  -0.9189 (mx/item weight) 0.01)))

;; ---------------------------------------------------------------------------
;; Score gradient with spliced CustomGradientGF
;; ---------------------------------------------------------------------------

(println "\n-- Score gradient with CustomGradientGF in model --")

(let [;; Model: slope ~ gaussian(0, 10), y ~ gaussian(slope * x, 1)
      ;; But slope*x is computed via a CustomGradientGF
      multiply-gf (cg/custom-gradient-gf
                    {:forward (fn [a b] (mx/multiply a b))
                     :has-argument-grads [true true]})
      model (gen [x]
              (let [slope (dyn/trace :slope (dist/gaussian 0 10))
                    pred  (dyn/splice :mul multiply-gf slope x)]
                (dyn/trace :y (dist/gaussian pred 1))
                pred))
      x (mx/scalar 2.0)
      obs (cm/choicemap :y (mx/scalar 6.0))
      ;; Compute choice gradient w.r.t. :slope
      {:keys [trace weight]} (p/generate model [x] obs)
      _ (mx/eval! (:retval trace) weight)
      grads (grad/choice-gradients model trace [:slope])]
  (assert-true "choice gradient for :slope exists"
               (contains? grads :slope))
  (let [g (mx/item (get grads :slope))]
    ;; Gradient should be nonzero (slope affects y through multiply-gf)
    (assert-true "gradient for :slope is nonzero"
                 (> (js/Math.abs g) 0.001))
    (println "    slope gradient =" g)))

(println "\n=== All CustomGradientGF Tests Complete ===")
