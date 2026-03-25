(ns genmlx.loop-obs-test
  "Tests for loop-obs and merge-obs convenience helpers."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.vectorized :as vec])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Shared models
;; ---------------------------------------------------------------------------

(def linreg-model
  (gen [xs]
    (let [slope (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j))
               (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                      intercept) 1)))
      slope)))

;; ---------------------------------------------------------------------------
;; 1. loop-obs basic
;; ---------------------------------------------------------------------------

(deftest loop-obs-basic
  (testing "loop-obs basic"
    (let [obs (dyn/loop-obs "y" [1.0 2.0 3.0])]
      (is (instance? cm/Node obs) "loop-obs returns Node")
      (is (= 1 (cm/get-value (cm/get-submap obs :y0))) "y0 value")
      (is (= 2 (cm/get-value (cm/get-submap obs :y1))) "y1 value")
      (is (= 3 (cm/get-value (cm/get-submap obs :y2))) "y2 value"))))

;; ---------------------------------------------------------------------------
;; 2. merge-obs
;; ---------------------------------------------------------------------------

(deftest merge-obs-test
  (testing "merge-obs"
    (let [static (cm/choicemap :slope 2.0 :intercept 1.0)
          loop-c (dyn/loop-obs "y" [10.0 20.0 30.0])
          merged (dyn/merge-obs static loop-c)]
      (is (= 2 (cm/get-value (cm/get-submap merged :slope))) "merged has slope")
      (is (= 10 (cm/get-value (cm/get-submap merged :y0))) "merged has y0")
      (is (= 30 (cm/get-value (cm/get-submap merged :y2))) "merged has y2"))))

;; ---------------------------------------------------------------------------
;; 3. vgenerate + loop model
;; ---------------------------------------------------------------------------

(deftest vgenerate-loop-model
  (testing "vgenerate + loop model"
    (let [xs [1.0 2.0 3.0]
          obs (dyn/loop-obs "y" [10.0 20.0 30.0])
          key (rng/fresh-key)
          vt (dyn/vgenerate linreg-model [xs] obs 10 key)]
      (is (= 10 (:n-particles vt)) "n-particles")
      (is (= [10] (mx/shape (:weight vt))) "weight shape")
      (is (= [10] (mx/shape (:score vt))) "score shape")
      (let [y0-val (cm/get-value (cm/get-submap (:choices vt) :y0))]
        (is (= [] (mx/shape y0-val)) "y0 is constrained scalar")))))

;; ---------------------------------------------------------------------------
;; 4. Equivalence: loop-obs == manual choicemap
;; ---------------------------------------------------------------------------

(deftest loop-obs-equivalence
  (testing "Equivalence: loop-obs vs manual choicemap"
    (let [xs [1.0 2.0 3.0]
          manual (cm/choicemap :y0 10.0 :y1 20.0 :y2 30.0)
          loop-c (dyn/loop-obs "y" [10.0 20.0 30.0])
          key (rng/fresh-key)
          vt-manual (dyn/vgenerate linreg-model [xs] manual 100 key)
          vt-loop (dyn/vgenerate linreg-model [xs] loop-c 100 key)
          w-manual (mx/item (mx/mean (:weight vt-manual)))
          w-loop (mx/item (mx/mean (:weight vt-loop)))]
      (is (h/close? w-manual w-loop 0.01) "mean weights match")
      (is (= (:n-particles vt-manual) (:n-particles vt-loop)) "same n-particles"))))

;; ---------------------------------------------------------------------------
;; 5. merge-obs + vgenerate end-to-end
;; ---------------------------------------------------------------------------

(deftest merge-obs-vgenerate-e2e
  (testing "merge-obs + vgenerate end-to-end"
    (let [xs [1.0 2.0 3.0]
          static (cm/choicemap :slope 2.0 :intercept 1.0)
          loop-c (dyn/loop-obs "y" [3.0 5.0 7.0])
          merged (dyn/merge-obs static loop-c)
          key (rng/fresh-key)
          vt (dyn/vgenerate linreg-model [xs] merged 50 key)]
      (is (= 50 (:n-particles vt)) "vt n-particles")
      (let [slope-val (cm/get-value (cm/get-submap (:choices vt) :slope))
            slope-num (if (mx/array? slope-val) (mx/item slope-val) slope-val)]
        (is (h/close? 2.0 slope-num 0.001) "slope constrained")))))

(cljs.test/run-tests)
