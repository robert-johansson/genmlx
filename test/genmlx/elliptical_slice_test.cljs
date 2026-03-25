(ns genmlx.elliptical-slice-test
  "Tests for elliptical slice sampling."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.mcmc :as mcmc])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; 1. Convergence — single Gaussian prior
;; ---------------------------------------------------------------------------

(deftest ess-single-address-convergence
  (testing "ESS convergence: single address"
    (let [prior-std 5.0
          obs-val 3.0
          posterior-mean (/ (* obs-val prior-std prior-std) (+ (* prior-std prior-std) 1))
          model (gen []
                  (let [x (trace :x (dist/gaussian 0 prior-std))]
                    (mx/eval! x)
                    (trace :obs (dist/gaussian (mx/item x) 1))
                    (mx/item x)))
          observations (cm/choicemap :obs (mx/scalar obs-val))
          traces (mcmc/elliptical-slice
                   {:samples 200 :burn 50 :selection [:x] :prior-std prior-std}
                   model [] observations)
          x-vals (mapv (fn [t]
                         (mx/realize (cm/get-value (cm/get-submap (:choices t) :x))))
                       traces)
          x-mean (/ (reduce + x-vals) (count x-vals))]
      (is (= 200 (count traces)) "ESS: 200 samples")
      (is (h/close? posterior-mean x-mean 1.0) "ESS: posterior mean")
      (let [ar (:acceptance-rate (meta traces))]
        (is (h/close? 1.0 ar 0.01) "ESS: acceptance rate = 1.0")))))

;; ---------------------------------------------------------------------------
;; 2. Multiple addresses
;; ---------------------------------------------------------------------------

(deftest ess-two-address-convergence
  (testing "ESS convergence: two addresses"
    (let [prior-std 5.0
          model (gen []
                  (let [x (trace :x (dist/gaussian 0 prior-std))
                        z (trace :z (dist/gaussian 0 prior-std))]
                    (mx/eval! x z)
                    (trace :obs-x (dist/gaussian (mx/item x) 1))
                    (trace :obs-z (dist/gaussian (mx/item z) 1))
                    [(mx/item x) (mx/item z)]))
          observations (cm/merge-cm
                         (cm/choicemap :obs-x (mx/scalar 3.0))
                         (cm/choicemap :obs-z (mx/scalar -2.0)))
          traces (mcmc/elliptical-slice
                   {:samples 200 :burn 50 :selection [:x :z] :prior-std prior-std}
                   model [] observations)
          x-vals (mapv (fn [t]
                         (mx/realize (cm/get-value (cm/get-submap (:choices t) :x))))
                       traces)
          z-vals (mapv (fn [t]
                         (mx/realize (cm/get-value (cm/get-submap (:choices t) :z))))
                       traces)
          x-mean (/ (reduce + x-vals) (count x-vals))
          z-mean (/ (reduce + z-vals) (count z-vals))
          post-mean-x (/ (* 3.0 prior-std prior-std) (+ (* prior-std prior-std) 1))
          post-mean-z (/ (* -2.0 prior-std prior-std) (+ (* prior-std prior-std) 1))]
      (is (h/close? post-mean-x x-mean 1.0) "ESS: x posterior mean near analytical")
      (is (h/close? post-mean-z z-mean 1.0) "ESS: z posterior mean near analytical"))))

(cljs.test/run-tests)
