(ns genmlx.new-dist-test
  "Tests for new distributions: broadcasted-normal, beta-uniform-mixture,
   piecewise-uniform, wishart, inverse-wishart."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm])
  (:require-macros [genmlx.gen :refer [gen]]))

;; =========================================================================
;; Broadcasted Normal
;; =========================================================================

(deftest broadcasted-normal-sample
  (testing "sample shape and finiteness"
    (let [mu (mx/array [1.0 2.0 3.0])
          sigma (mx/array [0.1 0.2 0.3])
          d (dist/broadcasted-normal mu sigma)
          v (dist/sample d)]
      (mx/eval! v)
      (is (= [3] (mx/shape v)) "broadcasted-normal sample shape is [3]")
      (is (every? js/isFinite (mx/->clj v)) "broadcasted-normal sample values are finite"))))

(deftest broadcasted-normal-log-prob
  (testing "log-prob at zeros"
    (let [mu (mx/array [0.0 0.0])
          sigma (mx/array [1.0 1.0])
          d (dist/broadcasted-normal mu sigma)
          v (mx/array [0.0 0.0])
          lp (dist/log-prob d v)]
      (mx/eval! lp)
      (is (h/close? -1.8379 (mx/item lp) 0.01) "broadcasted-normal log-prob at zeros"))))

(deftest broadcasted-normal-sample-n
  (testing "sample-n produces [N, ...shape]"
    (let [mu (mx/array [1.0 2.0])
          sigma (mx/array [0.5 0.5])
          d (dist/broadcasted-normal mu sigma)
          samples (dc/dist-sample-n d nil 50)]
      (mx/eval! samples)
      (is (= [50 2] (mx/shape samples)) "broadcasted-normal sample-n shape is [50,2]"))))

(deftest broadcasted-normal-statistics
  (testing "sample mean near mu"
    (let [mu (mx/array [5.0 -3.0])
          sigma (mx/array [0.5 0.5])
          d (dist/broadcasted-normal mu sigma)
          samples (dc/dist-sample-n d nil 1000)
          sample-mean (mx/mean samples [0])]
      (mx/eval! sample-mean)
      (let [means (mx/->clj sample-mean)]
        (is (h/close? 5.0 (first means) 0.2) "broadcasted-normal mean[0] ~ 5")
        (is (h/close? -3.0 (second means) 0.2) "broadcasted-normal mean[1] ~ -3")))))

(deftest broadcasted-normal-gfi
  (testing "works inside gen body"
    (let [model (dyn/auto-key (gen []
                  (trace :x (dist/broadcasted-normal (mx/array [0.0 0.0])
                                                          (mx/array [1.0 1.0])))))
          trace (p/simulate model [])]
      (is (some? trace) "broadcasted-normal works in gen body"))))

;; =========================================================================
;; Beta-Uniform Mixture
;; =========================================================================

(deftest beta-uniform-mixture-sample
  (testing "sample in [0,1]"
    (let [d (dist/beta-uniform-mixture 0.5 2.0 5.0)
          v (dist/sample d)]
      (mx/eval! v)
      (let [val (mx/item v)]
        (is (and (>= val 0.0) (<= val 1.0)) "beta-uniform-mixture sample in [0,1]")))))

(deftest beta-uniform-mixture-log-prob
  (testing "log-prob is finite for values in (0,1)"
    (let [d (dist/beta-uniform-mixture 0.5 2.0 5.0)
          lp (dist/log-prob d (mx/scalar 0.3))]
      (mx/eval! lp)
      (is (js/isFinite (mx/item lp)) "beta-uniform-mixture log-prob is finite"))))

(deftest beta-uniform-mixture-statistics
  (testing "mixture mean ~ 0.393"
    (let [d (dist/beta-uniform-mixture 0.5 2.0 5.0)
          samples (mapv (fn [_]
                          (let [v (dist/sample d)]
                            (mx/eval! v) (mx/item v)))
                        (range 2000))
          mean (/ (reduce + samples) (count samples))]
      (is (h/close? 0.393 mean 0.05) "beta-uniform-mixture mean ~ 0.39"))))

(deftest beta-uniform-mixture-gfi
  (testing "works inside gen body"
    (let [model (dyn/auto-key (gen []
                  (trace :p (dist/beta-uniform-mixture 0.5 2.0 5.0))))
          trace (p/simulate model [])]
      (is (some? trace) "beta-uniform-mixture works in gen body"))))

;; =========================================================================
;; Piecewise Uniform
;; =========================================================================

(deftest piecewise-uniform-sample
  (testing "sample in valid range"
    (let [bounds (mx/array [0.0 1.0 3.0 5.0])
          probs (mx/array [1.0 2.0 1.0])
          d (dist/piecewise-uniform bounds probs)
          v (dist/sample d)]
      (mx/eval! v)
      (let [val (mx/item v)]
        (is (and (>= val 0.0) (< val 5.0)) "piecewise-uniform sample in [0,5)")))))

(deftest piecewise-uniform-log-prob
  (testing "log-prob in bins"
    (let [bounds (mx/array [0.0 1.0 3.0 5.0])
          probs (mx/array [1.0 2.0 1.0])
          d (dist/piecewise-uniform bounds probs)]
      (let [lp (dist/log-prob d (mx/scalar 0.5))]
        (mx/eval! lp)
        (is (h/close? (js/Math.log 0.25) (mx/item lp) 0.01) "piecewise-uniform log-prob in bin 0"))
      (let [lp (dist/log-prob d (mx/scalar 2.0))]
        (mx/eval! lp)
        (is (h/close? (js/Math.log 0.25) (mx/item lp) 0.01) "piecewise-uniform log-prob in bin 1")))))

(deftest piecewise-uniform-out-of-bounds
  (testing "out of bounds returns -Inf"
    (let [bounds (mx/array [0.0 1.0 3.0 5.0])
          probs (mx/array [1.0 2.0 1.0])
          d (dist/piecewise-uniform bounds probs)
          lp (dist/log-prob d (mx/scalar 6.0))]
      (mx/eval! lp)
      (is (= ##-Inf (mx/item lp)) "piecewise-uniform log-prob out of bounds is -Inf"))))

(deftest piecewise-uniform-gfi
  (testing "works inside gen body"
    (let [model (dyn/auto-key (gen []
                  (trace :x (dist/piecewise-uniform (mx/array [0.0 1.0 2.0])
                                                         (mx/array [1.0 1.0])))))
          trace (p/simulate model [])]
      (is (some? trace) "piecewise-uniform works in gen body"))))

;; =========================================================================
;; Wishart
;; =========================================================================

(deftest wishart-sample
  (mx/clear-cache!)
  (testing "sample shape and symmetry"
    (let [V (mx/eye 2)
          d (dist/wishart 5 V)
          W (dist/sample d)]
      (mx/eval! W)
      (is (= [2 2] (mx/shape W)) "wishart sample shape is [2,2]")
      (let [vals (mx/->clj W)]
        (is (h/close? (get-in vals [0 1]) (get-in vals [1 0]) 0.001)
            "wishart sample is symmetric (W[0,1] ~ W[1,0])")))))

(deftest wishart-log-prob
  (mx/clear-cache!)
  (testing "log-prob is finite for valid PD matrix"
    (let [V (mx/eye 2)
          d (dist/wishart 5 V)
          W (dist/sample d)
          _ (mx/eval! W)
          lp (dist/log-prob d W)]
      (mx/eval! lp)
      (is (js/isFinite (mx/item lp)) "wishart log-prob is finite"))))

(deftest wishart-statistics
  (mx/clear-cache!)
  (testing "E[W] = df * V"
    (let [V (mx/eye 2)
          d (dist/wishart 5 V)
          n 500
          samples (mapv (fn [_]
                          (let [W (dist/sample d)]
                            (mx/eval! W) (mx/->clj W)))
                        (range n))
          mean-00 (/ (reduce + (map #(get-in % [0 0]) samples)) n)
          mean-11 (/ (reduce + (map #(get-in % [1 1]) samples)) n)
          mean-01 (/ (reduce + (map #(get-in % [0 1]) samples)) n)]
      (is (h/close? 5.0 mean-00 1.0) "wishart E[W[0,0]] ~ 5")
      (is (h/close? 5.0 mean-11 1.0) "wishart E[W[1,1]] ~ 5")
      (is (h/close? 0.0 mean-01 0.5) "wishart E[W[0,1]] ~ 0"))))

;; =========================================================================
;; Inverse Wishart
;; =========================================================================

(deftest inv-wishart-sample
  (mx/clear-cache!)
  (testing "sample shape and symmetry"
    (let [Psi (mx/eye 2)
          d (dist/inv-wishart 5 Psi)
          X (dist/sample d)]
      (mx/eval! X)
      (is (= [2 2] (mx/shape X)) "inv-wishart sample shape is [2,2]")
      (let [vals (mx/->clj X)]
        (is (h/close? (get-in vals [0 1]) (get-in vals [1 0]) 0.001)
            "inv-wishart sample is symmetric")))))

(deftest inv-wishart-log-prob
  (mx/clear-cache!)
  (testing "log-prob is finite"
    (let [Psi (mx/eye 2)
          d (dist/inv-wishart 5 Psi)
          X (dist/sample d)
          _ (mx/eval! X)
          lp (dist/log-prob d X)]
      (mx/eval! lp)
      (is (js/isFinite (mx/item lp)) "inv-wishart log-prob is finite"))))

(deftest inv-wishart-statistics
  (mx/clear-cache!)
  (testing "E[X] = Psi / (df - k - 1)"
    (let [Psi (mx/eye 2)
          d (dist/inv-wishart 5 Psi)
          n 500
          samples (mapv (fn [_]
                          (let [X (dist/sample d)]
                            (mx/eval! X) (mx/->clj X)))
                        (range n))
          mean-00 (/ (reduce + (map #(get-in % [0 0]) samples)) n)
          mean-11 (/ (reduce + (map #(get-in % [1 1]) samples)) n)
          mean-01 (/ (reduce + (map #(get-in % [0 1]) samples)) n)]
      (is (h/close? 0.5 mean-00 0.15) "inv-wishart E[X[0,0]] ~ 0.5")
      (is (h/close? 0.5 mean-11 0.15) "inv-wishart E[X[1,1]] ~ 0.5")
      (is (h/close? 0.0 mean-01 0.1) "inv-wishart E[X[0,1]] ~ 0"))))

(cljs.test/run-tests)
