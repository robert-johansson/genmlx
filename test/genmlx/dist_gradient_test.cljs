(ns genmlx.dist-gradient-test
  "Phase 1.7: Gradient flow tests for reparameterized distributions.
   For each reparameterizable distribution, verify that mx/grad of log-prob
   w.r.t. the sampled value produces analytically correct gradients.

   Gradient derivations:
   - Gaussian: d/dv log N(v;mu,sigma) = -(v-mu)/sigma^2
   - Laplace:  d/dv log Lap(v;loc,scale) = -sign(v-loc)/scale
   - LogNormal: d/dv log LN(v;mu,sigma) = -(1 + (log(v)-mu)/sigma^2) / v
   - Cauchy:   d/dv log C(v;loc,scale) = -2*(v-loc) / (scale^2 + (v-loc)^2)
   - Exponential: d/dv log Exp(v;rate) = -rate (for v>0)
   - Uniform:  d/dv log U(v;a,b) = 0 (constant inside support)

   Tolerance: 1e-4 (float32 gradient accumulation)."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.dist :as dist]
            [genmlx.mlx :as mx]
            [genmlx.test-helpers :as h]))

;; ==========================================================================
;; Gaussian gradient
;; ==========================================================================
;; d/dv log N(v; mu, sigma) = -(v - mu) / sigma^2

(deftest gaussian-gradient
  (testing "d/dv log N(v;0,1) at v=1 = -(1-0)/1 = -1"
    (let [d (dist/gaussian 0 1)
          grad-fn (mx/grad (fn [v] (dist/log-prob d v)))
          g (grad-fn (mx/scalar 1.0))]
      (mx/eval! g)
      (is (js/isFinite (mx/item g)) "gradient is finite")
      (is (h/close? -1.0 (mx/item g) 1e-4) "analytically correct")))

  (testing "d/dv log N(v;0,1) at v=0 = 0"
    (let [d (dist/gaussian 0 1)
          grad-fn (mx/grad (fn [v] (dist/log-prob d v)))
          g (grad-fn (mx/scalar 0.0))]
      (mx/eval! g)
      (is (h/close? 0.0 (mx/item g) 1e-4))))

  (testing "d/dv log N(v;3,2) at v=5 = -(5-3)/4 = -0.5"
    (let [d (dist/gaussian 3 2)
          grad-fn (mx/grad (fn [v] (dist/log-prob d v)))
          g (grad-fn (mx/scalar 5.0))]
      (mx/eval! g)
      (is (h/close? -0.5 (mx/item g) 1e-4))))

  (testing "d/dv log N(v;3,2) at v=1 = -(1-3)/4 = 0.5"
    (let [d (dist/gaussian 3 2)
          grad-fn (mx/grad (fn [v] (dist/log-prob d v)))
          g (grad-fn (mx/scalar 1.0))]
      (mx/eval! g)
      (is (h/close? 0.5 (mx/item g) 1e-4)))))

;; ==========================================================================
;; Laplace gradient
;; ==========================================================================
;; d/dv log Lap(v; loc, scale) = -sign(v - loc) / scale
;; (undefined at v=loc, but MLX gives 0 for sign(0))

(deftest laplace-gradient
  (testing "d/dv log Lap(v;0,1) at v=1 = -sign(1)/1 = -1"
    (let [d (dist/laplace 0 1)
          grad-fn (mx/grad (fn [v] (dist/log-prob d v)))
          g (grad-fn (mx/scalar 1.0))]
      (mx/eval! g)
      (is (h/close? -1.0 (mx/item g) 1e-4))))

  (testing "d/dv log Lap(v;0,1) at v=-1 = -sign(-1)/1 = 1"
    (let [d (dist/laplace 0 1)
          grad-fn (mx/grad (fn [v] (dist/log-prob d v)))
          g (grad-fn (mx/scalar -1.0))]
      (mx/eval! g)
      (is (h/close? 1.0 (mx/item g) 1e-4))))

  (testing "d/dv log Lap(v;5,2) at v=7 = -sign(2)/2 = -0.5"
    (let [d (dist/laplace 5 2)
          grad-fn (mx/grad (fn [v] (dist/log-prob d v)))
          g (grad-fn (mx/scalar 7.0))]
      (mx/eval! g)
      (is (h/close? -0.5 (mx/item g) 1e-4)))))

;; ==========================================================================
;; Cauchy gradient
;; ==========================================================================
;; d/dv log C(v; loc, scale) = -2*(v-loc) / (scale^2 + (v-loc)^2)

(deftest cauchy-gradient
  (testing "d/dv log C(v;0,1) at v=0 = 0 (at mode)"
    (let [d (dist/cauchy 0 1)
          grad-fn (mx/grad (fn [v] (dist/log-prob d v)))
          g (grad-fn (mx/scalar 0.0))]
      (mx/eval! g)
      (is (h/close? 0.0 (mx/item g) 1e-4))))

  (testing "d/dv log C(v;0,1) at v=1 = -2*1/(1+1) = -1"
    (let [d (dist/cauchy 0 1)
          grad-fn (mx/grad (fn [v] (dist/log-prob d v)))
          g (grad-fn (mx/scalar 1.0))]
      (mx/eval! g)
      (is (h/close? -1.0 (mx/item g) 1e-4))))

  (testing "d/dv log C(v;0,1) at v=-1 = 2*1/(1+1) = 1 (antisymmetric)"
    (let [d (dist/cauchy 0 1)
          grad-fn (mx/grad (fn [v] (dist/log-prob d v)))
          g (grad-fn (mx/scalar -1.0))]
      (mx/eval! g)
      (is (h/close? 1.0 (mx/item g) 1e-4)))))

;; ==========================================================================
;; Exponential gradient
;; ==========================================================================
;; d/dv log Exp(v; rate) = -rate (constant gradient in support)

(deftest exponential-gradient
  (testing "d/dv log Exp(v;1) at v=1 = -1"
    (let [d (dist/exponential 1)
          grad-fn (mx/grad (fn [v] (dist/log-prob d v)))
          g (grad-fn (mx/scalar 1.0))]
      (mx/eval! g)
      (is (h/close? -1.0 (mx/item g) 1e-4))))

  (testing "d/dv log Exp(v;2) at v=3 = -2"
    (let [d (dist/exponential 2)
          grad-fn (mx/grad (fn [v] (dist/log-prob d v)))
          g (grad-fn (mx/scalar 3.0))]
      (mx/eval! g)
      (is (h/close? -2.0 (mx/item g) 1e-4))))

  (testing "d/dv log Exp(v;0.5) at v=0.1 = -0.5"
    (let [d (dist/exponential 0.5)
          grad-fn (mx/grad (fn [v] (dist/log-prob d v)))
          g (grad-fn (mx/scalar 0.1))]
      (mx/eval! g)
      (is (h/close? -0.5 (mx/item g) 1e-4)))))

;; ==========================================================================
;; Log-Normal gradient
;; ==========================================================================
;; d/dv log LN(v; mu, sigma) = -(1 + (log(v)-mu)/sigma^2) / v
;; At v=1, mu=0, sigma=1: -(1 + 0)/1 = -1

(deftest log-normal-gradient
  (testing "d/dv log LN(v;0,1) at v=1 = -1"
    (let [d (dist/log-normal 0 1)
          grad-fn (mx/grad (fn [v] (dist/log-prob d v)))
          g (grad-fn (mx/scalar 1.0))]
      (mx/eval! g)
      (is (h/close? -1.0 (mx/item g) 1e-4))))

  (testing "d/dv log LN(v;0,1) at v=e = -(1+1)/e = -2/e = -0.73576"
    ;; log(e)=1, so -(1+1/1)/e = -2/e
    (let [d (dist/log-normal 0 1)
          grad-fn (mx/grad (fn [v] (dist/log-prob d v)))
          g (grad-fn (mx/scalar js/Math.E))]
      (mx/eval! g)
      (is (h/close? (/ -2.0 js/Math.E) (mx/item g) 1e-3)))))

;; ==========================================================================
;; Uniform gradient (constant log-prob inside support → gradient = 0)
;; ==========================================================================

(deftest uniform-gradient
  (testing "d/dv log U(v;0,1) at v=0.5 = 0 (constant density)"
    (let [d (dist/uniform 0 1)
          grad-fn (mx/grad (fn [v] (dist/log-prob d v)))
          g (grad-fn (mx/scalar 0.5))]
      (mx/eval! g)
      (is (h/close? 0.0 (mx/item g) 1e-4)))))

;; ==========================================================================
;; Gradient finiteness for all reparameterizable distributions
;; ==========================================================================

(deftest gradient-finiteness
  (testing "all reparameterizable distributions produce finite gradients"
    (doseq [[name d v]
            [["gaussian" (dist/gaussian 0 1) 0.5]
             ["laplace" (dist/laplace 0 1) 0.5]
             ["cauchy" (dist/cauchy 0 1) 0.5]
             ["exponential" (dist/exponential 1) 0.5]
             ["log-normal" (dist/log-normal 0 1) 1.0]
             ["uniform" (dist/uniform 0 1) 0.5]]]
      (let [grad-fn (mx/grad (fn [x] (dist/log-prob d x)))
            g (grad-fn (mx/scalar v))]
        (mx/eval! g)
        (is (js/isFinite (mx/item g))
            (str name " produces finite gradient at v=" v))))))

;; ==========================================================================
;; Run tests
;; ==========================================================================

(cljs.test/run-tests)
