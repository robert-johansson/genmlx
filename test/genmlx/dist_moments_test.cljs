(ns genmlx.dist-moments-test
  "Phase 1.3: Sampling moment tests for distributions with finite moments.
   For each distribution, sample N=5000 values and verify:
   - Sample mean converges to analytical E[X]
   - Sample variance converges to analytical Var[X]

   Tolerance derivation:
   For N=5000 samples, standard error of mean = sigma/sqrt(5000).
   z-test at z=3.5 gives P(false positive) < 0.0005.
   The z-test helper computes this automatically from the samples.
   Variance tolerance: empirical, ~0.5 for most distributions."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.test-helpers :as h]))

(def ^:private N 5000)

(defn- sample-n-realize
  "Sample N values from dist, return as seq of JS numbers."
  [d n]
  (let [key (h/deterministic-key)
        samples (dc/dist-sample-n d key n)]
    (mx/eval! samples)
    (mx/->clj samples)))

;; ==========================================================================
;; Gaussian
;; ==========================================================================
;; E[X] = mu, Var[X] = sigma^2

(deftest gaussian-moments
  (testing "N(5, 2): mean=5, variance=4"
    ;; Analytical: E[X]=5, Var[X]=4
    ;; z-test tolerance: 3.5*2/sqrt(5000) = 0.099
    (let [samples (sample-n-realize (dist/gaussian 5 2) N)]
      (is (h/z-test-passes? 5.0 samples) "mean converges to 5")
      (is (h/close? 4.0 (h/sample-variance samples) 0.5) "variance converges to 4")))

  (testing "N(0, 1): mean=0, variance=1"
    (let [samples (sample-n-realize (dist/gaussian 0 1) N)]
      (is (h/z-test-passes? 0.0 samples) "mean converges to 0")
      (is (h/close? 1.0 (h/sample-variance samples) 0.15) "variance converges to 1")))

  (testing "N(-3, 0.5): mean=-3, variance=0.25"
    (let [samples (sample-n-realize (dist/gaussian -3 0.5) N)]
      (is (h/z-test-passes? -3.0 samples) "mean converges to -3")
      (is (h/close? 0.25 (h/sample-variance samples) 0.05) "variance converges to 0.25"))))

;; ==========================================================================
;; Uniform
;; ==========================================================================
;; E[X] = (lo+hi)/2, Var[X] = (hi-lo)^2/12

(deftest uniform-moments
  (testing "Uniform(0,1): mean=0.5, variance=1/12"
    ;; Analytical: E[X]=0.5, Var[X]=1/12=0.08333
    (let [samples (sample-n-realize (dist/uniform 0 1) N)]
      (is (h/z-test-passes? 0.5 samples) "mean converges to 0.5")
      (is (h/close? 0.08333 (h/sample-variance samples) 0.01) "variance converges to 1/12")))

  (testing "Uniform(2,5): mean=3.5, variance=0.75"
    ;; Analytical: E[X]=3.5, Var[X]=(3)^2/12=0.75
    (let [samples (sample-n-realize (dist/uniform 2 5) N)]
      (is (h/z-test-passes? 3.5 samples) "mean converges to 3.5")
      (is (h/close? 0.75 (h/sample-variance samples) 0.1) "variance converges to 0.75"))))

;; ==========================================================================
;; Bernoulli
;; ==========================================================================
;; E[X] = p, Var[X] = p*(1-p)

(deftest bernoulli-moments
  (testing "Bernoulli(0.7): mean=0.7, variance=0.21"
    ;; Analytical: E[X]=0.7, Var[X]=0.7*0.3=0.21
    (let [samples (sample-n-realize (dist/bernoulli 0.7) N)]
      (is (h/z-test-passes? 0.7 samples) "mean converges to 0.7")
      (is (h/close? 0.21 (h/sample-variance samples) 0.03) "variance converges to 0.21")))

  (testing "Bernoulli(0.5): mean=0.5, variance=0.25"
    (let [samples (sample-n-realize (dist/bernoulli 0.5) N)]
      (is (h/z-test-passes? 0.5 samples) "mean converges to 0.5"))))

;; ==========================================================================
;; Exponential
;; ==========================================================================
;; E[X] = 1/rate, Var[X] = 1/rate^2

(deftest exponential-moments
  (testing "Exponential(2): mean=0.5, variance=0.25"
    ;; Analytical: E[X]=1/2=0.5, Var[X]=1/4=0.25
    (let [samples (sample-n-realize (dist/exponential 2) N)]
      (is (h/z-test-passes? 0.5 samples) "mean converges to 0.5")
      (is (h/close? 0.25 (h/sample-variance samples) 0.05) "variance converges to 0.25")))

  (testing "Exponential(1): mean=1, variance=1"
    (let [samples (sample-n-realize (dist/exponential 1) N)]
      (is (h/z-test-passes? 1.0 samples) "mean converges to 1"))))

;; ==========================================================================
;; Beta
;; ==========================================================================
;; E[X] = a/(a+b), Var[X] = ab/((a+b)^2*(a+b+1))

(deftest beta-moments
  (testing "Beta(2,5): mean=2/7, variance=10/(49*8)=0.02551"
    ;; Analytical: E[X]=2/7=0.28571, Var[X]=10/392=0.02551
    (let [samples (sample-n-realize (dist/beta-dist 2 5) N)]
      (is (h/z-test-passes? 0.28571 samples) "mean converges to 2/7")
      (is (h/close? 0.02551 (h/sample-variance samples) 0.005) "variance converges")))

  (testing "Beta(1,1): mean=0.5, variance=1/12"
    ;; Beta(1,1) = Uniform(0,1)
    (let [samples (sample-n-realize (dist/beta-dist 1 1) N)]
      (is (h/z-test-passes? 0.5 samples) "mean converges to 0.5"))))

;; ==========================================================================
;; Gamma
;; ==========================================================================
;; E[X] = shape/rate, Var[X] = shape/rate^2

(deftest gamma-moments
  (testing "Gamma(2,1): mean=2, variance=2"
    ;; Analytical: E[X]=2/1=2, Var[X]=2/1=2
    (let [samples (sample-n-realize (dist/gamma-dist 2 1) N)]
      (is (h/z-test-passes? 2.0 samples) "mean converges to 2")
      (is (h/close? 2.0 (h/sample-variance samples) 0.3) "variance converges to 2")))

  (testing "Gamma(3,2): mean=1.5, variance=0.75"
    ;; Analytical: E[X]=3/2=1.5, Var[X]=3/4=0.75
    (let [samples (sample-n-realize (dist/gamma-dist 3 2) N)]
      (is (h/z-test-passes? 1.5 samples) "mean converges to 1.5")
      (is (h/close? 0.75 (h/sample-variance samples) 0.15) "variance converges to 0.75"))))

;; ==========================================================================
;; Poisson
;; ==========================================================================
;; E[X] = lambda, Var[X] = lambda

(deftest poisson-moments
  (testing "Poisson(3): mean=3, variance=3"
    ;; Analytical: E[X]=3, Var[X]=3
    ;; Note: Poisson uses sequential scalar sampling, so use fewer samples
    (let [d (dist/poisson 3)
          key (h/deterministic-key)
          keys (rng/split-n key 2000)
          samples (mapv #(h/realize (dist/sample d %)) keys)]
      (is (h/z-test-passes? 3.0 samples) "mean converges to 3")
      (is (h/close? 3.0 (h/sample-variance samples) 0.5) "variance converges to 3"))))

;; ==========================================================================
;; Laplace
;; ==========================================================================
;; E[X] = loc, Var[X] = 2*scale^2

(deftest laplace-moments
  (testing "Laplace(0,1): mean=0, variance=2"
    ;; Analytical: E[X]=0, Var[X]=2*1=2
    (let [samples (sample-n-realize (dist/laplace 0 1) N)]
      (is (h/z-test-passes? 0.0 samples) "mean converges to 0")
      (is (h/close? 2.0 (h/sample-variance samples) 0.4) "variance converges to 2")))

  (testing "Laplace(5,2): mean=5, variance=8"
    ;; Analytical: E[X]=5, Var[X]=2*4=8
    (let [samples (sample-n-realize (dist/laplace 5 2) N)]
      (is (h/z-test-passes? 5.0 samples) "mean converges to 5")
      (is (h/close? 8.0 (h/sample-variance samples) 1.5) "variance converges to 8"))))

;; ==========================================================================
;; Log-Normal
;; ==========================================================================
;; E[X] = exp(mu + sigma^2/2), Var[X] = (exp(sigma^2)-1)*exp(2*mu+sigma^2)

(deftest log-normal-moments
  (testing "LogNormal(0,1): mean=exp(0.5)=1.6487, variance=(e-1)*e=4.6710"
    ;; Analytical: E[X]=exp(0.5)=1.6487, Var[X]=(e-1)*e=4.6710
    (let [samples (sample-n-realize (dist/log-normal 0 1) N)]
      (is (h/z-test-passes? 1.6487 samples) "mean converges to exp(0.5)")
      ;; Variance tolerance is large due to heavy tail
      (is (h/close? 4.6710 (h/sample-variance samples) 2.0) "variance roughly correct")))

  (testing "LogNormal(0,0.5): mean=exp(0.125)=1.1331"
    ;; Analytical: E[X]=exp(0.125)=1.1331
    (let [samples (sample-n-realize (dist/log-normal 0 0.5) N)]
      (is (h/z-test-passes? 1.1331 samples) "mean converges to exp(0.125)"))))

;; ==========================================================================
;; Geometric
;; ==========================================================================
;; E[X] = (1-p)/p, Var[X] = (1-p)/p^2

(deftest geometric-moments
  (testing "Geometric(0.5): mean=1, variance=2"
    ;; Analytical: E[X]=(1-0.5)/0.5=1, Var[X]=0.5/0.25=2
    (let [d (dist/geometric 0.5)
          key (h/deterministic-key)
          keys (rng/split-n key 3000)
          samples (mapv #(h/realize (dist/sample d %)) keys)]
      (is (h/z-test-passes? 1.0 samples) "mean converges to 1")
      (is (h/close? 2.0 (h/sample-variance samples) 0.5) "variance converges to 2"))))

;; ==========================================================================
;; Binomial
;; ==========================================================================
;; E[X] = n*p, Var[X] = n*p*(1-p)

(deftest binomial-moments
  (testing "Binomial(10,0.5): mean=5, variance=2.5"
    ;; Analytical: E[X]=5, Var[X]=10*0.5*0.5=2.5
    (let [samples (sample-n-realize (dist/binomial 10 0.5) N)]
      (is (h/z-test-passes? 5.0 samples) "mean converges to 5")
      (is (h/close? 2.5 (h/sample-variance samples) 0.3) "variance converges to 2.5"))))

;; ==========================================================================
;; Discrete Uniform
;; ==========================================================================
;; E[X] = (lo+hi)/2, Var[X] = ((hi-lo+1)^2-1)/12

(deftest discrete-uniform-moments
  (testing "DiscreteUniform(1,6): mean=3.5, variance=35/12=2.917"
    ;; Analytical: E[X]=(1+6)/2=3.5, Var[X]=((6)^2-1)/12=35/12=2.917
    (let [samples (sample-n-realize (dist/discrete-uniform 1 6) N)]
      (is (h/z-test-passes? 3.5 samples) "mean converges to 3.5")
      (is (h/close? 2.917 (h/sample-variance samples) 0.4) "variance converges"))))

;; ==========================================================================
;; Inverse Gamma
;; ==========================================================================
;; E[X] = scale/(shape-1) for shape>1, Var[X] = scale^2/((shape-1)^2*(shape-2)) for shape>2

(deftest inv-gamma-moments
  (testing "InvGamma(3,1): mean=1/(3-1)=0.5, variance=1/(4*1)=0.25"
    ;; Analytical: E[X]=1/2=0.5, Var[X]=1/(2^2*1)=0.25
    (let [samples (sample-n-realize (dist/inv-gamma 3 1) N)]
      (is (h/z-test-passes? 0.5 samples) "mean converges to 0.5")
      (is (h/close? 0.25 (h/sample-variance samples) 0.1) "variance converges to 0.25"))))

;; ==========================================================================
;; Neg-Binomial
;; ==========================================================================
;; E[X] = r*(1-p)/p, Var[X] = r*(1-p)/p^2

(deftest neg-binomial-moments
  (testing "NegBinomial(5,0.6): mean=5*0.4/0.6=3.333, variance=5*0.4/0.36=5.556"
    ;; Analytical: E[X]=5*0.4/0.6=3.333, Var[X]=5*0.4/0.36=5.556
    ;; NegBinomial sampling is slow (scalar), use fewer samples
    (let [d (dist/neg-binomial 5 0.6)
          key (h/deterministic-key)
          keys (rng/split-n key 2000)
          samples (mapv #(h/realize (dist/sample d %)) keys)]
      (is (h/z-test-passes? 3.333 samples) "mean converges to 3.333")
      (is (h/close? 5.556 (h/sample-variance samples) 1.5) "variance roughly correct"))))

;; ==========================================================================
;; Von Mises — circular mean
;; ==========================================================================
;; E[cos(X-mu)] = I1(kappa)/I0(kappa) (circular mean resultant length)
;; For kappa=1: I1(1)/I0(1) = 0.44606/1.26607 ≈ 0.35221
;; But for simplicity, we test that the circular mean direction ≈ mu

(deftest von-mises-moments
  (testing "VonMises(0,5): circular mean ≈ 0 (high concentration)"
    ;; With kappa=5, samples cluster tightly around mu=0
    ;; Circular mean = atan2(mean(sin(x)), mean(cos(x))) ≈ 0
    (let [d (dist/von-mises 0 5)
          key (h/deterministic-key)
          keys (rng/split-n key 3000)
          samples (mapv #(h/realize (dist/sample d %)) keys)
          sin-mean (h/sample-mean (map #(js/Math.sin %) samples))
          cos-mean (h/sample-mean (map #(js/Math.cos %) samples))
          circ-mean (js/Math.atan2 sin-mean cos-mean)]
      (is (h/close? 0.0 circ-mean 0.1) "circular mean ≈ 0"))))

;; ==========================================================================
;; Run tests
;; ==========================================================================

(cljs.test/run-tests)
