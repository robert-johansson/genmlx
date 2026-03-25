(ns genmlx.dist-statistics-test
  "Distribution statistics tests: E[X], Var[X], and discrete PMF sums."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]))

(def N 10000)
(def N-small 1000)

(defn sample-moments
  "Sample N values from distribution, return [mean variance]."
  ([d] (sample-moments d N))
  ([d n]
   (let [key (rng/fresh-key)
         samples (dc/dist-sample-n d key n)]
     (mx/eval! samples)
     (let [m (mx/mean samples)
           v (mx/variance samples)]
       (mx/eval! m v)
       [(mx/item m) (mx/item v)]))))

;; ---------------------------------------------------------------------------
;; 21.1 -- E[X] and Var[X]
;; ---------------------------------------------------------------------------

(deftest gaussian-moments
  (testing "gaussian(2, 3): E=2, Var=9"
    (let [[m v] (sample-moments (dist/gaussian 2 3))]
      (is (h/close? 2.0 m 0.3) "E[gaussian(2,3)] ~ 2")
      (is (h/close? 9.0 v 1.5) "Var[gaussian(2,3)] ~ 9"))))

(deftest uniform-moments
  (testing "uniform(1, 5): E=3, Var=16/12"
    (let [[m v] (sample-moments (dist/uniform 1 5))]
      (is (h/close? 3.0 m 0.2) "E[uniform(1,5)] ~ 3")
      (is (h/close? (/ 16.0 12.0) v 0.3) "Var[uniform(1,5)] ~ 1.333"))))

(deftest bernoulli-moments
  (testing "bernoulli(0.3): E=0.3, Var=0.21"
    (let [[m v] (sample-moments (dist/bernoulli 0.3))]
      (is (h/close? 0.3 m 0.05) "E[bernoulli(0.3)] ~ 0.3")
      (is (h/close? 0.21 v 0.05) "Var[bernoulli(0.3)] ~ 0.21"))))

(deftest beta-moments
  (testing "beta(2, 5): E=2/7, Var=10/343"
    (let [[m v] (sample-moments (dist/beta-dist 2 5))]
      (is (h/close? (/ 2.0 7.0) m 0.05) "E[beta(2,5)] ~ 0.286")
      (is (h/close? (/ 10.0 343.0) v 0.01) "Var[beta(2,5)] ~ 0.029"))))

(deftest gamma-moments
  (testing "gamma(3, 2): E=3/2, Var=3/4"
    (let [[m v] (sample-moments (dist/gamma-dist 3 2))]
      (is (h/close? 1.5 m 0.2) "E[gamma(3,2)] ~ 1.5")
      (is (h/close? 0.75 v 0.2) "Var[gamma(3,2)] ~ 0.75"))))

(deftest exponential-moments
  (testing "exponential(2): E=0.5, Var=0.25"
    (let [[m v] (sample-moments (dist/exponential 2))]
      (is (h/close? 0.5 m 0.1) "E[exponential(2)] ~ 0.5")
      (is (h/close? 0.25 v 0.1) "Var[exponential(2)] ~ 0.25"))))

(deftest poisson-moments
  (testing "poisson(4): E=4, Var=4"
    (let [[m v] (sample-moments (dist/poisson 4) N-small)]
      (is (h/close? 4.0 m 0.5) "E[poisson(4)] ~ 4")
      (is (h/close? 4.0 v 1.5) "Var[poisson(4)] ~ 4"))))

(deftest laplace-moments
  (testing "laplace(1, 2): E=1, Var=8"
    (let [[m v] (sample-moments (dist/laplace 1 2))]
      (is (h/close? 1.0 m 0.3) "E[laplace(1,2)] ~ 1")
      (is (h/close? 8.0 v 1.5) "Var[laplace(1,2)] ~ 8"))))

(deftest log-normal-moments
  (testing "log-normal(0, 0.5): E=e^0.125, Var=(e^0.25-1)*e^0.25"
    (let [[m v] (sample-moments (dist/log-normal 0 0.5))
          expected-mean (js/Math.exp 0.125)
          expected-var (* (- (js/Math.exp 0.25) 1) (js/Math.exp 0.25))]
      (is (h/close? expected-mean m 0.1) "E[log-normal(0,0.5)] ~ e^0.125")
      (is (h/close? expected-var v 0.1) "Var[log-normal(0,0.5)]"))))

(deftest geometric-moments
  (testing "geometric(0.3): E=7/3, Var=70/9"
    (let [[m v] (sample-moments (dist/geometric 0.3))]
      (is (h/close? (/ 7.0 3.0) m 0.3) "E[geometric(0.3)] ~ 2.333")
      (is (h/close? (/ 70.0 9.0) v 2.0) "Var[geometric(0.3)] ~ 7.778"))))

(deftest neg-binomial-moments
  (testing "neg-binomial(5, 0.4): E=7.5, Var=18.75"
    (let [[m v] (sample-moments (dist/neg-binomial 5 0.4) N-small)]
      (is (h/close? 7.5 m 1.0) "E[neg-binomial(5,0.4)] ~ 7.5")
      (is (h/close? 18.75 v 6.0) "Var[neg-binomial(5,0.4)] ~ 18.75"))))

(deftest binomial-moments
  (testing "binomial(20, 0.3): E=6, Var=4.2"
    (let [[m v] (sample-moments (dist/binomial 20 0.3))]
      (is (h/close? 6.0 m 0.5) "E[binomial(20,0.3)] ~ 6")
      (is (h/close? 4.2 v 1.0) "Var[binomial(20,0.3)] ~ 4.2"))))

(deftest discrete-uniform-moments
  (testing "discrete-uniform(1, 6): E=3.5, Var=35/12"
    (let [[m v] (sample-moments (dist/discrete-uniform 1 6))]
      (is (h/close? 3.5 m 0.2) "E[discrete-uniform(1,6)] ~ 3.5")
      (is (h/close? (/ 35.0 12.0) v 0.6) "Var[discrete-uniform(1,6)] ~ 2.917"))))

(deftest truncated-normal-moments
  (testing "truncated-normal(0, 1, -2, 2): E=0, Var~0.774"
    (let [[m v] (sample-moments (dist/truncated-normal 0 1 -2 2))]
      (is (h/close? 0.0 m 0.1) "E[truncated-normal(0,1,-2,2)] ~ 0")
      (is (h/close? 0.774 v 0.1) "Var[truncated-normal(0,1,-2,2)] ~ 0.774"))))

(deftest student-t-moments
  (testing "student-t(5, 0, 1): E=0, Var=5/3"
    (let [[m v] (sample-moments (dist/student-t 5 0 1))]
      (is (h/close? 0.0 m 0.2) "E[student-t(5,0,1)] ~ 0")
      (is (h/close? (/ 5.0 3.0) v 0.5) "Var[student-t(5,0,1)] ~ 1.667"))))

(deftest inv-gamma-moments
  (testing "inv-gamma(4, 2): E=2/3, Var=2/9"
    (let [[m v] (sample-moments (dist/inv-gamma 4 2))]
      (is (h/close? (/ 2.0 3.0) m 0.1) "E[inv-gamma(4,2)] ~ 0.667")
      (is (h/close? (/ 2.0 9.0) v 0.1) "Var[inv-gamma(4,2)] ~ 0.222"))))

(deftest mvn-component-means
  (testing "multivariate-normal component means"
    (let [mu (mx/array [1.0 2.0 3.0])
          cov (mx/array [[1.0 0.0 0.0] [0.0 1.0 0.0] [0.0 0.0 1.0]])
          d (dist/multivariate-normal mu cov)
          key (rng/fresh-key)
          samples (dc/dist-sample-n d key N)
          _ (mx/eval! samples)
          means (mx/mean samples [0])]
      (mx/eval! means)
      (let [m (mx/->clj means)]
        (is (h/close? 1.0 (nth m 0) 0.2) "E[MVN][0] ~ 1")
        (is (h/close? 2.0 (nth m 1) 0.2) "E[MVN][1] ~ 2")
        (is (h/close? 3.0 (nth m 2) 0.2) "E[MVN][2] ~ 3")))))

(deftest dirichlet-component-means
  (testing "dirichlet component means"
    (let [alpha (mx/array [2.0 3.0 5.0])
          d (dist/dirichlet alpha)
          key (rng/fresh-key)
          samples (dc/dist-sample-n d key N)
          _ (mx/eval! samples)
          means (mx/mean samples [0])]
      (mx/eval! means)
      (let [m (mx/->clj means)]
        (is (h/close? 0.2 (nth m 0) 0.05) "E[dirichlet][0] ~ 0.2")
        (is (h/close? 0.3 (nth m 1) 0.05) "E[dirichlet][1] ~ 0.3")
        (is (h/close? 0.5 (nth m 2) 0.05) "E[dirichlet][2] ~ 0.5")))))

;; ---------------------------------------------------------------------------
;; 21.2 -- Discrete PMF sums to 1
;; ---------------------------------------------------------------------------

(defn pmf-sum
  "Sum exp(log-prob(v)) for values in vs."
  [d vs]
  (let [total (reduce (fn [acc v]
                        (let [lp (dc/dist-log-prob d (mx/scalar v))]
                          (mx/eval! lp)
                          (+ acc (js/Math.exp (mx/item lp)))))
                      0.0 vs)]
    total))

(deftest bernoulli-pmf-sum
  (testing "bernoulli PMF sums to 1"
    (is (h/close? 1.0 (pmf-sum (dist/bernoulli 0.3) [0 1]) 0.001) "bernoulli(0.3) PMF sums to 1")
    (is (h/close? 1.0 (pmf-sum (dist/bernoulli 0.7) [0 1]) 0.001) "bernoulli(0.7) PMF sums to 1")))

(deftest categorical-pmf-sum
  (testing "categorical PMF sums to 1"
    (let [logits (mx/log (mx/array [0.2 0.3 0.5]))
          s (pmf-sum (dist/categorical logits) [0 1 2])]
      (is (h/close? 1.0 s 0.001) "categorical([0.2,0.3,0.5]) PMF sums to 1"))))

(deftest binomial-pmf-sum
  (testing "binomial PMF sums to 1"
    (let [s (pmf-sum (dist/binomial 10 0.4) (range 11))]
      (is (h/close? 1.0 s 0.001) "binomial(10,0.4) PMF sums to 1"))))

(deftest discrete-uniform-pmf-sum
  (testing "discrete-uniform PMF sums to 1"
    (let [s (pmf-sum (dist/discrete-uniform 3 8) (range 3 9))]
      (is (h/close? 1.0 s 0.001) "discrete-uniform(3,8) PMF sums to 1"))))

(deftest geometric-pmf-sum
  (testing "geometric PMF sum > 0.999"
    (let [s (pmf-sum (dist/geometric 0.3) (range 31))]
      (is (> s 0.999) "geometric(0.3) PMF sum > 0.999"))))

(deftest poisson-pmf-sum
  (testing "poisson PMF sum > 0.999"
    (let [s (pmf-sum (dist/poisson 3) (range 21))]
      (is (> s 0.999) "poisson(3) PMF sum > 0.999"))))

(deftest neg-binomial-pmf-sum
  (testing "neg-binomial PMF sum > 0.99"
    (let [s (pmf-sum (dist/neg-binomial 5 0.4) (range 41))]
      (is (> s 0.99) "neg-binomial(5,0.4) PMF sum > 0.99"))))

(cljs.test/run-tests)
