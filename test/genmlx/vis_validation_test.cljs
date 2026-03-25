(ns genmlx.vis-validation-test
  "VIS (Vectorized Inference for Dynamic-Address Models) validation.
   Documents that vsimulate/vgenerate handle dynamic-address models
   correctly, with significant speedup over scalar execution."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.gen :refer [gen]]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p]))

;; ---------------------------------------------------------------------------
;; Test models
;; ---------------------------------------------------------------------------

(def simple-loop-model
  "Model with doseq loop generating dynamic addresses."
  (gen [xs]
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j))
               (dist/gaussian mu 1)))
      mu)))

(def linreg-model
  "Linear regression with doseq loop."
  (gen [xs]
    (let [slope     (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j))
               (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                      intercept) 1)))
      slope)))

(deftest vis-vsimulate
  (testing "vsimulate with dynamic-address models"
    (let [key (rng/fresh-key)
          model (dyn/auto-key simple-loop-model)
          vt (dyn/vsimulate model [(range 5)] 100 key)
          inner (:m (:choices vt))]
      (is (= [100] (mx/shape (:v (get inner :mu)))) "vsimulate :mu shape")
      (is (= [100] (mx/shape (:v (get inner :y0)))) "vsimulate :y0 shape")
      (is (= [100] (mx/shape (:v (get inner :y4)))) "vsimulate :y4 shape")
      (is (= [100] (mx/shape (:retval vt))) "vsimulate retval shape")
      (is (= [100] (mx/shape (:score vt))) "vsimulate score shape"))))

(deftest vis-vgenerate
  (testing "vgenerate with flat constraints"
    (let [key (rng/fresh-key)
          model (dyn/auto-key simple-loop-model)
          obs (cm/choicemap :y0 1.0 :y1 2.0 :y2 3.0 :y3 4.0 :y4 5.0)
          vt (dyn/vgenerate model [(range 5)] obs 1000 key)
          w (:weight vt) r (:retval vt)
          wn (let [e (mx/exp (mx/subtract w (mx/amax w)))] (mx/divide e (mx/sum e)))
          mu-est (mx/item (mx/sum (mx/multiply wn r)))]
      (is (= [1000] (mx/shape w)) "vgenerate weight shape")
      (is (h/close? 3.0 mu-est 0.5) "posterior mean mu ~ 3.0"))))

(deftest vis-linreg-posterior
  (testing "linreg posterior"
    (let [key (rng/fresh-key)
          model (dyn/auto-key linreg-model)
          xs [1.0 2.0 3.0 4.0 5.0]
          obs (cm/choicemap :y0 3.1 :y1 4.9 :y2 7.2 :y3 8.8 :y4 11.1)
          vt (dyn/vgenerate model [xs] obs 5000 key)
          w (:weight vt) r (:retval vt)
          wn (let [e (mx/exp (mx/subtract w (mx/amax w)))] (mx/divide e (mx/sum e)))
          slope-est (mx/item (mx/sum (mx/multiply wn r)))]
      (is (h/close? 2.0 slope-est 0.5) "linreg posterior slope ~ 2.0"))))

(deftest vis-performance
  (testing "performance: batched vs scalar"
    (let [model (dyn/auto-key linreg-model)
          xs (mapv double (range 20))
          obs (apply cm/choicemap
                     (mapcat (fn [j] [(keyword (str "y" j)) (+ (* 2.0 j) 1.0)])
                             (range 20)))]
      (let [t0 (.now js/Date)
            _ (dyn/vgenerate model [xs] obs 1000 (rng/fresh-key))
            t1 (.now js/Date)
            batch-ms (- t1 t0)
            t2 (.now js/Date)
            _ (dotimes [_ 10] (p/generate model [xs] obs))
            t3 (.now js/Date)
            scalar-per (/ (- t3 t2) 10.0)
            scalar-1000 (* scalar-per 1000)
            speedup (/ scalar-1000 (max batch-ms 1))]
        (is (> speedup 50) (str "speedup > 50x (got " (.toFixed speedup 0) "x)"))))))

(deftest vis-scaling
  (testing "scaling with N particles"
    (let [model (dyn/auto-key linreg-model)
          xs (mapv double (range 20))
          obs (apply cm/choicemap
                     (mapcat (fn [j] [(keyword (str "y" j)) (+ (* 2.0 j) 1.0)])
                             (range 20)))]
      (doseq [n [100 1000 10000]]
        (let [t0 (.now js/Date)
              _ (dyn/vgenerate model [xs] obs n (rng/fresh-key))
              t1 (.now js/Date)]
          ;; Just check it completes
          (is true (str "N=" n " completed in " (- t1 t0) "ms")))))))

(cljs.test/run-tests)
