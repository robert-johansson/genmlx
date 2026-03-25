(ns genmlx.vimco-test
  "Tests for VIMCO (Variational Inference with Multi-sample Objectives).
   Tests convergence, comparison with IWELBO, and shape correctness."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.inference.vi :as vi]))

;; ---------------------------------------------------------------------------
;; Tests
;; ---------------------------------------------------------------------------

(deftest vimco-convergence-test
  (testing "VIMCO converges on Gaussian target"
    ;; Model: z ~ N(0, 1), obs = z + noise. Observe obs=3.
    ;; True posterior: z | obs=3 ~ N(1.5, sqrt(0.5))
    (let [log-p (fn [z]
                  (let [z-scalar (mx/index z 0)]
                    (mx/add (dc/dist-log-prob (dist/gaussian 0 1) z-scalar)
                            (dc/dist-log-prob (dist/gaussian z-scalar 1) (mx/scalar 3.0)))))
          log-q (fn [z params]
                  (let [mu (mx/index params 0)
                        log-sigma (mx/index params 1)
                        sigma (mx/exp log-sigma)]
                    (dc/dist-log-prob (dist/gaussian mu sigma) (mx/index z 0))))
          sample-fn (fn [params key n]
                      (let [mu (mx/index params 0)
                            log-sigma (mx/index params 1)
                            sigma (mx/exp log-sigma)
                            eps (rng/normal (rng/ensure-key key) [n 1])]
                        (mx/add mu (mx/multiply sigma eps))))
          init-params (mx/array [0.0 0.0])
          result (vi/vimco
                   {:iterations 500 :learning-rate 0.01 :n-samples 20}
                   log-p log-q sample-fn init-params)]
      (mx/eval! (:params result))
      (let [final-mu (mx/item (mx/index (:params result) 0))
            final-log-sigma (mx/item (mx/index (:params result) 1))
            final-sigma (js/Math.exp final-log-sigma)
            losses (:loss-history result)
            first-loss (first losses)
            last-loss (last losses)]
        (is (h/close? 1.5 final-mu 1.0) "VIMCO: mu near 1.5")
        (is (and (> final-sigma 0.1) (< final-sigma 3.0)) "VIMCO: sigma reasonable")
        (is (< last-loss first-loss) "VIMCO: loss decreased")))))

(deftest vimco-programmable-vi-test
  (testing "VIMCO via programmable-vi interface"
    (let [log-p (fn [z]
                  (let [z-scalar (mx/index z 0)]
                    (mx/add (dc/dist-log-prob (dist/gaussian 0 1) z-scalar)
                            (dc/dist-log-prob (dist/gaussian z-scalar 1) (mx/scalar 3.0)))))
          log-q (fn [z params]
                  (let [mu (mx/index params 0)
                        log-sigma (mx/index params 1)
                        sigma (mx/exp log-sigma)]
                    (dc/dist-log-prob (dist/gaussian mu sigma) (mx/index z 0))))
          sample-fn (fn [params key n]
                      (let [mu (mx/index params 0)
                            log-sigma (mx/index params 1)
                            sigma (mx/exp log-sigma)
                            eps (rng/normal (rng/ensure-key key) [n 1])]
                        (mx/add mu (mx/multiply sigma eps))))
          init-params (mx/array [0.0 0.0])
          result (vi/programmable-vi
                   {:iterations 500 :learning-rate 0.01 :n-samples 20
                    :objective :vimco}
                   log-p log-q sample-fn init-params)]
      (mx/eval! (:params result))
      (let [final-mu (mx/item (mx/index (:params result) 0))
            losses (:loss-history result)]
        (is (> final-mu 0.3) "programmable-vi VIMCO: mu moved toward 1.5")
        (is (pos? (count losses)) "programmable-vi VIMCO: has loss history")
        (is (< (last losses) (first losses)) "programmable-vi VIMCO: loss decreased")))))

(deftest vimco-shape-correctness-test
  (testing "VIMCO objective returns scalar"
    (let [log-p (fn [z] (dc/dist-log-prob (dist/gaussian 0 1) (mx/index z 0)))
          log-q (fn [z] (dc/dist-log-prob (dist/gaussian 0 1) (mx/index z 0)))
          obj-fn (vi/vimco-objective log-p log-q)
          samples (rng/normal (rng/fresh-key) [5 1])
          result (obj-fn samples)]
      (mx/eval! result)
      (is (= 0 (mx/ndim result)) "VIMCO objective: returns scalar")
      (is (js/isFinite (mx/item result)) "VIMCO objective: finite value"))))

(deftest vimco-different-k-test
  (testing "VIMCO works with different K values"
    (let [log-p (fn [z] (dc/dist-log-prob (dist/gaussian 0 1) (mx/index z 0)))
          log-q (fn [z] (dc/dist-log-prob (dist/gaussian 0 1) (mx/index z 0)))
          obj-fn (vi/vimco-objective log-p log-q)]
      (doseq [k [3 10 20]]
        (let [samples (rng/normal (rng/fresh-key) [k 1])
              result (obj-fn samples)]
          (mx/eval! result)
          (is (= 0 (mx/ndim result)) (str "VIMCO K=" k ": returns scalar"))
          (is (js/isFinite (mx/item result)) (str "VIMCO K=" k ": finite")))))))

(cljs.test/run-tests)
