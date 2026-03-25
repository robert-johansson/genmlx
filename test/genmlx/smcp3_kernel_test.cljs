(ns genmlx.smcp3-kernel-test
  "Tests for SMCP3 kernel interface."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.smcp3 :as smcp3]
            [genmlx.mlx.random :as rng])
  (:require-macros [genmlx.gen :refer [gen]]))

;; Model: x ~ N(0, 1), y ~ N(x, 0.5)
(def model
  (dyn/auto-key (gen []
    (let [x (trace :x (dist/gaussian 0 1))]
      (mx/eval! x)
      (trace :y (dist/gaussian (mx/item x) 0.5))
      (mx/item x)))))

;; Forward kernel: random-walk propose new :x near current :x
(def fwd-kernel
  (dyn/auto-key (gen [choices]
    (let [cur-x (mx/realize (cm/get-choice choices [:x]))]
      (trace :x (dist/gaussian cur-x 0.5))))))

;; Backward kernel: symmetric (same structure)
(def bwd-kernel
  (dyn/auto-key (gen [choices]
    (let [new-x (mx/realize (cm/get-choice choices [:x]))]
      (trace :x (dist/gaussian new-x 0.5))))))

;; Two observation steps (step 1 triggers kernel path)
(def obs-seq [(cm/choicemap :y (mx/scalar 3.0))
              (cm/choicemap :y (mx/scalar 3.0))])

;; ---------------------------------------------------------------------------
;; Tests
;; ---------------------------------------------------------------------------

(deftest smcp3-runs-test
  (testing "SMCP3 with kernels runs"
    (let [result (smcp3/smcp3
                   {:particles 10
                    :forward-kernel fwd-kernel
                    :backward-kernel bwd-kernel
                    :key (rng/fresh-key)}
                   model [] obs-seq)]
      (is (some? result) "smcp3 returns result"))))

(deftest structure-test
  (testing "correct keys and particle count"
    (let [n 10
          result (smcp3/smcp3
                   {:particles n
                    :forward-kernel fwd-kernel
                    :backward-kernel bwd-kernel
                    :key (rng/fresh-key)}
                   model [] obs-seq)]
      (is (some? (:traces result)) "result has :traces")
      (is (some? (:log-weights result)) "result has :log-weights")
      (is (some? (:log-ml-estimate result)) "result has :log-ml-estimate")
      (is (= n (count (:traces result))) "correct particle count for traces")
      (is (= n (count (:log-weights result))) "correct particle count for weights"))))

(deftest traces-valid-test
  (testing "each trace has finite :x and :y matches obs"
    (let [result (smcp3/smcp3
                   {:particles 10
                    :forward-kernel fwd-kernel
                    :backward-kernel bwd-kernel
                    :key (rng/fresh-key)}
                   model [] obs-seq)
          traces (:traces result)
          all-x-finite (every? (fn [t]
                                 (let [x (mx/realize (cm/get-choice (:choices t) [:x]))]
                                   (js/isFinite x)))
                               traces)
          all-y-correct (every? (fn [t]
                                  (let [y (mx/realize (cm/get-choice (:choices t) [:y]))]
                                    (< (js/Math.abs (- y 3.0)) 1e-6)))
                                traces)]
      (is all-x-finite "all traces have finite :x")
      (is all-y-correct "all traces have :y = 3.0"))))

(deftest weights-finite-test
  (testing "weights are finite"
    (let [result (smcp3/smcp3
                   {:particles 10
                    :forward-kernel fwd-kernel
                    :backward-kernel bwd-kernel
                    :key (rng/fresh-key)}
                   model [] obs-seq)
          all-w-finite (every? (fn [w] (js/isFinite (mx/realize w)))
                               (:log-weights result))
          ml-finite (js/isFinite (mx/realize (:log-ml-estimate result)))]
      (is all-w-finite "all log-weights finite")
      (is ml-finite "log-ml-estimate finite"))))

(deftest log-ml-reasonable-test
  (testing "log-ML in plausible range"
    (let [result (smcp3/smcp3
                   {:particles 50
                    :forward-kernel fwd-kernel
                    :backward-kernel bwd-kernel
                    :key (rng/fresh-key)}
                   model [] obs-seq)
          log-ml (mx/realize (:log-ml-estimate result))]
      (is (and (> log-ml -30) (< log-ml 0)) "log-ML in plausible range [-30, 0]"))))

(deftest direct-init-step-test
  (testing "direct smcp3-init + smcp3-step with kernels"
    (let [n 10
          key (rng/fresh-key)
          [k1 k2] (rng/split-n (rng/ensure-key key) 2)
          init-result (smcp3/smcp3-init model [] (first obs-seq) nil n k1)
          step-result (smcp3/smcp3-step
                        (:traces init-result)
                        (:log-weights init-result)
                        model (second obs-seq)
                        fwd-kernel bwd-kernel
                        n 0.5 nil k2)]
      (is (some? (:traces init-result)) "init has :traces")
      (is (some? (:log-weights init-result)) "init has :log-weights")
      (is (some? (:log-ml-increment init-result)) "init has :log-ml-increment")
      (is (some? (:traces step-result)) "step has :traces")
      (is (some? (:log-weights step-result)) "step has :log-weights")
      (is (some? (:log-ml-increment step-result)) "step has :log-ml-increment")
      (is (some? (:ess step-result)) "step has :ess")
      (is (contains? step-result :resampled?) "step has :resampled?")
      (is (= n (count (:traces step-result))) "step traces count")
      (is (= n (count (:log-weights step-result))) "step weights count"))))

(cljs.test/run-tests)
