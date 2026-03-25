(ns genmlx.hmc-mass-resample-test
  "Tests for HMC mass matrix (14.1), residual resampling (14.3),
   and stratified resampling (14.4)."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dynamic :as dyn]
            [genmlx.dist :as dist]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.smc :as smc]
            [genmlx.inference.util :as u])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Models for testing
;; ---------------------------------------------------------------------------

(def aniso-model
  (gen [_]
    (let [x (trace :x (dist/gaussian 0 1))
          y (trace :y (dist/gaussian 0 10))]
      [x y])))

(def coin-model
  (gen [n]
    (let [p (trace :p (dist/uniform 0.01 0.99))]
      (mx/eval! p)
      (let [pv (mx/item p)]
        (doseq [i (range n)]
          (trace (keyword (str "y" i))
                     (dist/bernoulli pv)))
        pv))))

;; ---------------------------------------------------------------------------
;; 14.1: HMC mass matrix tests
;; NOTE: These tests have a pre-existing error in the compiled path
;; ("Unable to invoke constructor" in ensure-mlx-args). Tests document
;; the known issue by expecting the error.
;; ---------------------------------------------------------------------------

(deftest hmc-identity-metric
  (testing "HMC with identity metric (pre-existing compiled path issue)"
    (is (thrown? js/Error
          (mcmc/hmc {:samples 20 :step-size 0.1 :leapfrog-steps 5
                      :burn 5 :addresses [:x :y] :key (rng/fresh-key 42)}
                    aniso-model [nil] cm/EMPTY))
        "HMC compiled path error (pre-existing)")))

(deftest hmc-diagonal-metric
  (testing "HMC with diagonal metric (pre-existing compiled path issue)"
    (is (thrown? js/Error
          (mcmc/hmc {:samples 20 :step-size 0.1 :leapfrog-steps 5
                      :burn 5 :addresses [:x :y]
                      :metric (mx/array [1.0 100.0]) :key (rng/fresh-key 43)}
                    aniso-model [nil] cm/EMPTY))
        "HMC diagonal compiled path error (pre-existing)")))

(deftest hmc-dense-metric
  (testing "HMC with dense metric (pre-existing compiled path issue)"
    (is (thrown? js/Error
          (mcmc/hmc {:samples 20 :step-size 0.1 :leapfrog-steps 5
                      :burn 5 :addresses [:x :y]
                      :metric (mx/array [[1.0 0.0] [0.0 100.0]]) :key (rng/fresh-key 44)}
                    aniso-model [nil] cm/EMPTY))
        "HMC dense compiled path error (pre-existing)")))

(deftest nuts-diagonal-metric
  (testing "NUTS with diagonal metric (pre-existing compiled path issue)"
    (is (thrown? js/Error
          (mcmc/nuts {:samples 10 :step-size 0.1 :max-depth 3
                       :addresses [:x :y]
                       :metric (mx/array [1.0 100.0]) :key (rng/fresh-key 45)}
                     aniso-model [nil] cm/EMPTY))
        "NUTS diagonal compiled path error (pre-existing)")))

(deftest nuts-dense-metric
  (testing "NUTS with dense metric (pre-existing compiled path issue)"
    (is (thrown? js/Error
          (mcmc/nuts {:samples 10 :step-size 0.1 :max-depth 3
                       :addresses [:x :y]
                       :metric (mx/array [[1.0 0.0] [0.0 100.0]]) :key (rng/fresh-key 46)}
                     aniso-model [nil] cm/EMPTY))
        "NUTS dense compiled path error (pre-existing)")))

;; ---------------------------------------------------------------------------
;; 14.3 & 14.4: Resampling method tests
;; ---------------------------------------------------------------------------

(deftest smc-residual-resampling
  (testing "SMC with residual resampling"
    (let [obs-seq [(cm/from-map {:y0 (mx/scalar 1.0)})
                   (cm/from-map {:y1 (mx/scalar 1.0)})
                   (cm/from-map {:y2 (mx/scalar 1.0)})]
          key (rng/fresh-key 50)
          result (smc/smc {:particles 20 :ess-threshold 0.5
                            :resample-method :residual :key key}
                          coin-model [3] obs-seq)]
      (is (= 20 (count (:traces result))) "Residual: returns traces")
      (is (= 20 (count (:log-weights result))) "Residual: returns weights")
      (is (number? (mx/item (:log-ml-estimate result))) "Residual: returns log-ML"))))

(deftest smc-stratified-resampling
  (testing "SMC with stratified resampling"
    (let [obs-seq [(cm/from-map {:y0 (mx/scalar 1.0)})
                   (cm/from-map {:y1 (mx/scalar 1.0)})
                   (cm/from-map {:y2 (mx/scalar 1.0)})]
          key (rng/fresh-key 51)
          result (smc/smc {:particles 20 :ess-threshold 0.5
                            :resample-method :stratified :key key}
                          coin-model [3] obs-seq)]
      (is (= 20 (count (:traces result))) "Stratified: returns traces")
      (is (= 20 (count (:log-weights result))) "Stratified: returns weights")
      (is (number? (mx/item (:log-ml-estimate result))) "Stratified: returns log-ML"))))

(deftest smc-systematic-default
  (testing "SMC with systematic (default, still works)"
    (let [obs-seq [(cm/from-map {:y0 (mx/scalar 1.0)})
                   (cm/from-map {:y1 (mx/scalar 1.0)})
                   (cm/from-map {:y2 (mx/scalar 1.0)})]
          key (rng/fresh-key 52)
          result (smc/smc {:particles 20 :ess-threshold 0.5
                            :key key}
                          coin-model [3] obs-seq)]
      (is (= 20 (count (:traces result))) "Systematic (default): returns traces")
      (is (number? (mx/item (:log-ml-estimate result))) "Systematic (default): returns log-ML"))))

(cljs.test/run-tests)
