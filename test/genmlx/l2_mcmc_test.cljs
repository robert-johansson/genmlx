(ns genmlx.l2-mcmc-test
  "Level 2 WP-1 tests: compiled MCMC with tensor-native score."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.gen :refer [gen]]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.tensor-trace :as tt]
            [genmlx.inference.util :as iu]
            [genmlx.inference.mcmc :as mcmc]))

;; ---------------------------------------------------------------------------
;; Test models
;; ---------------------------------------------------------------------------

(def linear-model
  (gen [x]
    (let [slope (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 5))
          y-pred (mx/add (mx/multiply slope (mx/ensure-array x)) intercept)]
      (trace :y (dist/gaussian y-pred 1))
      slope)))

(def simple-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 1))]
      (trace :x (dist/gaussian mu 0.1))
      mu)))

;; ---------------------------------------------------------------------------
;; Tests
;; ---------------------------------------------------------------------------

(deftest prepare-mcmc-score-test
  (testing "prepare-mcmc-score returns tensor-native for static models"
    (let [model (dyn/auto-key linear-model)
          obs (cm/choicemap :y (mx/scalar 5.0))
          {:keys [trace]} (p/generate model [2.0] obs)
          result (iu/prepare-mcmc-score linear-model [2.0] obs [:slope :intercept] trace)]
      (is (:tensor-native? result) "tensor-native? for static model")
      (is (fn? (:score-fn result)) "has score-fn")
      (is (mx/array? (:init-params result)) "has init-params")
      (is (= 2 (:n-params result)) "n-params is 2")
      (let [s ((:score-fn result) (:init-params result))]
        (mx/eval! s)
        (is (js/isFinite (mx/item s)) "score-fn returns finite value")))))

(deftest compiled-mh-test
  (testing "compiled-mh with tensor-score"
    (let [obs (cm/choicemap :y (mx/scalar 5.0))
          samples (mcmc/compiled-mh
                    {:samples 50 :burn 200 :addresses [:slope :intercept]
                     :proposal-std 0.5 :compile? true}
                    linear-model [2.0] obs)]
      (is (= 50 (count samples)) "compiled-mh returns 50 samples")
      (is (= 2 (count (first samples))) "each sample has 2 elements")
      (let [slopes (mapv first samples)
            mean-slope (/ (reduce + slopes) (count slopes))]
        (is (js/isFinite mean-slope) "mean slope is finite")
        (is (< -20 mean-slope 20) "mean slope in reasonable range"))))

  (testing "non-compiled mh"
    (let [obs (cm/choicemap :y (mx/scalar 5.0))
          samples (mcmc/compiled-mh
                    {:samples 10 :burn 50 :addresses [:slope :intercept]
                     :proposal-std 0.5 :compile? false}
                    linear-model [2.0] obs)]
      (is (= 10 (count samples)) "non-compiled mh returns 10 samples"))))

(deftest mala-test
  (testing "MALA with tensor-score"
    (let [obs (cm/choicemap :x (mx/scalar 0.5))
          samples (mcmc/mala
                    {:samples 20 :burn 100 :addresses [:mu]
                     :step-size 0.1 :compile? true}
                    simple-model [] obs)]
      (is (= 20 (count samples)) "MALA returns 20 samples")
      (let [mus (mapv first samples)
            mean-mu (/ (reduce + mus) (count mus))]
        (is (h/close? 0.5 mean-mu 0.3) "MALA posterior mean near 0.5")))))

(deftest hmc-test
  (testing "HMC with tensor-score"
    (let [obs (cm/choicemap :x (mx/scalar 0.5))
          samples (mcmc/hmc
                    {:samples 20 :burn 100 :addresses [:mu]
                     :step-size 0.05 :n-leapfrog 10 :compile? true}
                    simple-model [] obs)]
      (is (= 20 (count samples)) "HMC returns 20 samples")
      (let [mus (mapv first samples)
            mean-mu (/ (reduce + mus) (count mus))]
        (is (h/close? 0.5 mean-mu 0.3) "HMC posterior mean near 0.5")))))

(deftest map-optimize-test
  (testing "MAP with tensor-score"
    (let [obs (cm/choicemap :x (mx/scalar 0.5))
          result (mcmc/map-optimize
                   {:iterations 500 :addresses [:mu]
                    :learning-rate 0.05 :compile? true}
                   simple-model [] obs)]
      (is (some? (:trace result)) "MAP returns trace")
      (is (js/isFinite (:score result)) "MAP score is finite")
      (is (pos? (count (:score-history result))) "MAP has score-history")
      (let [mu-val (first (:params result))]
        (is (h/close? 0.5 mu-val 0.2) "MAP mu near 0.5")))))

(deftest gfi-fallback-test
  (testing "GFI fallback for non-static models"
    (let [dynamic-model (gen [n]
                          (dotimes [i n]
                            (trace (keyword (str "x" i)) (dist/gaussian 0 1)))
                          nil)
          model (dyn/auto-key dynamic-model)
          obs cm/EMPTY
          {:keys [trace]} (p/generate model [3] (cm/choicemap :x0 (mx/scalar 0.0)
                                                               :x1 (mx/scalar 0.0)
                                                               :x2 (mx/scalar 0.0)))
          result (iu/prepare-mcmc-score dynamic-model [3]
                                         (cm/choicemap :x0 (mx/scalar 0.0)
                                                        :x1 (mx/scalar 0.0)
                                                        :x2 (mx/scalar 0.0))
                                         [] trace)]
      (is (not (:tensor-native? result)) "non-static model not tensor-native"))))

(deftest score-consistency-test
  (testing "Score consistency: tensor-score matches GFI"
    (let [model (dyn/auto-key linear-model)
          obs (cm/choicemap :y (mx/scalar 5.0))
          {:keys [trace]} (p/generate model [2.0] obs)
          tensor-result (iu/prepare-mcmc-score linear-model [2.0] obs [:slope :intercept] trace)
          layout (iu/compute-param-layout trace [:slope :intercept])
          gfi-fn (iu/make-score-fn model [2.0] obs [:slope :intercept] layout)
          gfi-params (iu/extract-params trace [:slope :intercept] layout)
          tensor-params (:init-params tensor-result)]
      (let [ts ((:score-fn tensor-result) tensor-params)
            gs (gfi-fn gfi-params)]
        (mx/eval! ts)
        (mx/eval! gs)
        (is (h/close? (mx/item gs) (mx/item ts) 1e-4)
            "tensor and GFI scores match for same trace")))))

(cljs.test/run-tests)
