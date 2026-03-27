
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
  (:require [cljs.test :refer [deftest is testing use-fixtures]]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.test-helpers :as h]))

(def ^:private N 5000)

(use-fixtures :each
  {:after (fn [] (mx/force-gc!))})

(defn- sample-n-realize
  "Sample N values from dist, return as seq of JS numbers."
  [d n]
  (let [key (h/deterministic-key)
        samples (dc/dist-sample-n d key n)]
    (mx/eval! samples)
    (mx/->clj samples)))

(defn- circular-mean
  "Circular mean direction from a seq of angle samples."
  [samples]
  (let [S (h/sample-mean (map #(js/Math.sin %) samples))
        C (h/sample-mean (map #(js/Math.cos %) samples))]
    (js/Math.atan2 S C)))

(defn- mean-resultant-length
  "Mean resultant length R from a seq of angle samples.
   R = sqrt(mean(cos)^2 + mean(sin)^2).
   R = 1 means all samples identical, R = 0 means uniform on circle."
  [samples]
  (let [S (h/sample-mean (map #(js/Math.sin %) samples))
        C (h/sample-mean (map #(js/Math.cos %) samples))]
    (js/Math.sqrt (+ (* S S) (* C C)))))

(defn- sample-covariance
  "Unbiased sample covariance between two sequences."
  [xs ys]
  (let [mu-x (h/sample-mean xs)
        mu-y (h/sample-mean ys)
        n (count xs)]
    (/ (reduce + (map (fn [x y] (* (- x mu-x) (- y mu-y))) xs ys))
       (dec n))))

(defn- sample-skewness
  "Sample skewness (Fisher). Uses biased central moments m2 and m3."
  [xs]
  (let [mu (h/sample-mean xs)
        n (count xs)
        m2 (/ (reduce + (map #(let [d (- % mu)] (* d d)) xs)) n)
        m3 (/ (reduce + (map #(let [d (- % mu)] (* d d d)) xs)) n)]
    (/ m3 (js/Math.pow m2 1.5))))

(defn- sample-excess-kurtosis
  "Sample excess kurtosis. Uses biased central moments m2 and m4."
  [xs]
  (let [mu (h/sample-mean xs)
        n (count xs)
        m2 (/ (reduce + (map #(let [d (- % mu)] (* d d)) xs)) n)
        m4 (/ (reduce + (map #(let [d (- % mu)] (* d d d d)) xs)) n)]
    (- (/ m4 (* m2 m2)) 3.0)))

(defn- sample-n-realize-mv
  "Sample N vectors/matrices from dist, return as nested Clojure vectors.
   For [N,k] arrays returns [[v0 v1 ...] [v0 v1 ...] ...].
   For [N,k,k] arrays returns [[[v00 v01] [v10 v11]] ...]."
  [d n]
  (let [key (h/deterministic-key)
        samples (dc/dist-sample-n d key n)]
    (mx/eval! samples)
    (mx/->clj samples)))

(defn- extract-element
  "Extract per-element sequences from nested sample vectors.
   Returns a seq of N scalar values at the given index path."
  ([data i]
   (mapv #(nth % i) data))
  ([data i j]
   (mapv #(nth (nth % i) j) data)))

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
      (is (h/z-test-passes? 0.5 samples) "mean converges to 0.5")))

  (testing "Beta(2,5): skewness=0.5963"
    ;; Analytical: skew = 2*(5-2)*sqrt(8) / (9*sqrt(10)) = 0.5963
    (let [samples (sample-n-realize (dist/beta-dist 2 5) N)]
      (is (h/close? 0.5963 (sample-skewness samples) 0.12) "skewness converges to 0.5963"))))

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
      (is (h/close? 0.75 (h/sample-variance samples) 0.15) "variance converges to 0.75")))

  (testing "Gamma(4,1): skewness=1.0"
    ;; Analytical: skew = 2/sqrt(shape) = 2/sqrt(4) = 1.0
    (let [samples (sample-n-realize (dist/gamma-dist 4 1) N)]
      (is (h/close? 1.0 (sample-skewness samples) 0.12) "skewness converges to 1.0")))

  (testing "Gamma(9,2): skewness=0.6667"
    ;; Analytical: skew = 2/sqrt(9) = 2/3 = 0.6667
    (let [samples (sample-n-realize (dist/gamma-dist 9 2) N)]
      (is (h/close? 0.6667 (sample-skewness samples) 0.12) "skewness converges to 0.6667"))))

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
      (is (h/close? 4.6710 (h/sample-variance samples) 3.5) "variance roughly correct")))

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
      (is (h/close? 0.0 circ-mean 0.1) "circular mean ≈ 0")))

  (testing "VonMises(0,2): circular mean and R"
    ;; Analytical: circular mean = 0, R = I_1(2)/I_0(2) = 0.6978
    (let [d (dist/von-mises 0 2)
          key (h/deterministic-key)
          keys (rng/split-n key 2000)
          samples (mapv #(h/realize (dist/sample d %)) keys)
          R (mean-resultant-length samples)]
      (is (h/close? 0.0 (circular-mean samples) 0.1) "circular mean ≈ 0")
      ;; Analytical: R = I_1(2)/I_0(2) = 0.6978
      ;; Tolerance: 0.05 (N=2000, SE ~ 1/sqrt(N) ~ 0.022, 3.5*SE ~ 0.08)
      (is (h/close? 0.6978 R 0.05) "R converges to I1/I0")))

  (testing "VonMises(pi/4,5): circular mean and R"
    ;; Analytical: circular mean = pi/4 = 0.7854, R = I_1(5)/I_0(5) = 0.8934
    (let [d (dist/von-mises (/ js/Math.PI 4) 5)
          key (h/deterministic-key)
          keys (rng/split-n key 2000)
          samples (mapv #(h/realize (dist/sample d %)) keys)
          R (mean-resultant-length samples)]
      (is (h/close? 0.7854 (circular-mean samples) 0.1) "circular mean ≈ pi/4")
      ;; Analytical: R = I_1(5)/I_0(5) = 0.8934
      ;; Tolerance: 0.04 (N=2000, SE ~ 0.01 for concentrated dist, 3.5*SE ~ 0.035)
      (is (h/close? 0.8934 R 0.04) "R converges to I1/I0"))))

;; ==========================================================================
;; Student-t
;; ==========================================================================
;; E[X] = loc (df > 1), Var[X] = scale^2 * df/(df-2) (df > 2)

(deftest student-t-moments
  (testing "StudentT(5,2,1.5): mean=2.0, variance=3.75"
    ;; Analytical: E[X]=2.0, Var[X]=1.5^2 * 5/3 = 3.75
    ;; Excess kurtosis = 6/(5-4) = 6 (very heavy tails), so S^2 has high variance.
    (let [samples (sample-n-realize (dist/student-t 5 2 1.5) N)]
      (is (h/z-test-passes? 2.0 samples) "mean converges to 2")
      (is (h/close? 3.75 (h/sample-variance samples) 1.5) "variance converges to 3.75")))

  (testing "StudentT(10,0,1): mean=0.0, variance=1.25"
    ;; Analytical: E[X]=0.0, Var[X]=1^2 * 10/8 = 1.25
    (let [samples (sample-n-realize (dist/student-t 10 0 1) N)]
      (is (h/z-test-passes? 0.0 samples) "mean converges to 0")
      (is (h/close? 1.25 (h/sample-variance samples) 0.15) "variance converges to 1.25")))

  (testing "StudentT(10,0,1): excess kurtosis=1.0"
    ;; Analytical: excess kurtosis = 6/(df-4) = 6/6 = 1.0
    ;; Sample kurtosis estimator has high variance for heavy-tailed distributions.
    (let [samples (sample-n-realize (dist/student-t 10 0 1) N)]
      (is (h/close? 1.0 (sample-excess-kurtosis samples) 1.0) "excess kurtosis converges to 1.0"))))

;; ==========================================================================
;; Truncated Normal
;; ==========================================================================
;; E[X] = mu + sigma*(phi(a)-phi(b))/Z, Var[X] = sigma^2 * [1 + (a*phi(a)-b*phi(b))/Z - ((phi(a)-phi(b))/Z)^2]

(deftest truncated-normal-moments
  (testing "TruncatedNormal(0,1,-2,2): symmetric, mean=0, variance=0.7737"
    ;; Analytical: E[X]=0.0 (by symmetry), Var[X]=0.7737
    (let [samples (sample-n-realize (dist/truncated-normal 0 1 -2 2) N)]
      (is (h/z-test-passes? 0.0 samples) "mean converges to 0")
      (is (h/close? 0.7737 (h/sample-variance samples) 0.1) "variance converges to 0.7737")))

  (testing "TruncatedNormal(1,2,0,5): asymmetric, mean=1.8915, variance=1.5062"
    ;; Analytical: E[X]=1.8915, Var[X]=1.5062
    (let [samples (sample-n-realize (dist/truncated-normal 1 2 0 5) N)]
      (is (h/z-test-passes? 1.8915 samples) "mean converges to 1.8915")
      (is (h/close? 1.5062 (h/sample-variance samples) 0.2) "variance converges to 1.5062"))))

;; ==========================================================================
;; Categorical
;; ==========================================================================
;; E[X] = sum_k k*p_k, Var[X] = sum_k k^2*p_k - (E[X])^2

(deftest categorical-moments
  (testing "Categorical([0,0,0]): uniform, mean=1.0, variance=0.6667"
    ;; Analytical: p=[1/3,1/3,1/3], E[X]=1.0, Var[X]=2/3
    (let [samples (sample-n-realize (dist/categorical (mx/array [0 0 0])) N)]
      (is (h/z-test-passes? 1.0 samples) "mean converges to 1.0")
      (is (h/close? 0.6667 (h/sample-variance samples) 0.1) "variance converges to 2/3")))

  (testing "Categorical([log(.1),log(.3),log(.6)]): mean=1.5, variance=0.45"
    ;; Analytical: p=[0.1,0.3,0.6], E[X]=1.5, Var[X]=0.45
    (let [logits (mx/array [(js/Math.log 0.1) (js/Math.log 0.3) (js/Math.log 0.6)])
          samples (sample-n-realize (dist/categorical logits) N)]
      (is (h/z-test-passes? 1.5 samples) "mean converges to 1.5")
      (is (h/close? 0.45 (h/sample-variance samples) 0.1) "variance converges to 0.45"))))

;; ==========================================================================
;; Dirichlet (marginal moments)
;; ==========================================================================
;; E[X_k] = alpha_k/alpha_0, Var[X_k] = alpha_k*(alpha_0-alpha_k)/(alpha_0^2*(alpha_0+1))

(deftest dirichlet-moments
  (testing "Dirichlet([2,5,3]): marginal means and variances"
    ;; Analytical: alpha_0=10
    ;; E[X_0]=0.2, E[X_1]=0.5, E[X_2]=0.3
    ;; Var[X_0]=0.01455, Var[X_1]=0.02273, Var[X_2]=0.01909
    (let [data (sample-n-realize-mv (dist/dirichlet (mx/array [2 5 3])) N)
          x0 (extract-element data 0)
          x1 (extract-element data 1)
          x2 (extract-element data 2)]
      (is (h/z-test-passes? 0.2 x0) "E[X_0] = 0.2")
      (is (h/z-test-passes? 0.5 x1) "E[X_1] = 0.5")
      (is (h/z-test-passes? 0.3 x2) "E[X_2] = 0.3")
      (is (h/close? 0.01455 (h/sample-variance x0) 0.005) "Var[X_0] converges")
      (is (h/close? 0.02273 (h/sample-variance x1) 0.005) "Var[X_1] converges")
      (is (h/close? 0.01909 (h/sample-variance x2) 0.005) "Var[X_2] converges")))

  (testing "Dirichlet([2,5,3]): off-diagonal covariance"
    ;; Analytical: Cov[X_0,X_1] = -2*5/(100*11) = -0.00909
    (let [data (sample-n-realize-mv (dist/dirichlet (mx/array [2 5 3])) N)
          x0 (extract-element data 0)
          x1 (extract-element data 1)]
      (is (h/close? -0.00909 (sample-covariance x0 x1) 0.005)
          "Cov[X_0,X_1] converges to -0.00909"))))

;; ==========================================================================
;; Multivariate Normal (full covariance)
;; ==========================================================================
;; E[X] = mean-vec, Cov[X] = cov-matrix

(deftest multivariate-normal-moments
  (let [mu (mx/array [1.0 -1.0 0.5])
        sigma (mx/array [[2.0 0.5 -0.3]
                         [0.5 1.0 0.2]
                         [-0.3 0.2 1.5]])
        data (sample-n-realize-mv (dist/multivariate-normal mu sigma) N)
        x0 (extract-element data 0)
        x1 (extract-element data 1)
        x2 (extract-element data 2)]

    (testing "MVN(3d): per-component means"
      (is (h/z-test-passes? 1.0 x0) "E[X_0] = 1.0")
      (is (h/z-test-passes? -1.0 x1) "E[X_1] = -1.0")
      (is (h/z-test-passes? 0.5 x2) "E[X_2] = 0.5"))

    (testing "MVN(3d): per-component variances"
      (is (h/close? 2.0 (h/sample-variance x0) 0.3) "Var[X_0] = 2.0")
      (is (h/close? 1.0 (h/sample-variance x1) 0.15) "Var[X_1] = 1.0")
      (is (h/close? 1.5 (h/sample-variance x2) 0.2) "Var[X_2] = 1.5"))

    (testing "MVN(3d): off-diagonal covariances"
      (is (h/close? 0.5 (sample-covariance x0 x1) 0.08) "Cov[X_0,X_1] = 0.5")
      (is (h/close? -0.3 (sample-covariance x0 x2) 0.08) "Cov[X_0,X_2] = -0.3")
      (is (h/close? 0.2 (sample-covariance x1 x2) 0.08) "Cov[X_1,X_2] = 0.2"))))

;; ==========================================================================
;; Gaussian-Vec (independent per-element)
;; ==========================================================================
;; E[X_i] = mu_i, Var[X_i] = sigma_i^2, Cov[X_i,X_j] = 0

(deftest gaussian-vec-moments
  (let [data (sample-n-realize-mv
              (dist/gaussian-vec (mx/array [1.0 -2.0 3.0])
                                 (mx/array [0.5 1.0 2.0])) N)
        x0 (extract-element data 0)
        x1 (extract-element data 1)
        x2 (extract-element data 2)]

    (testing "GaussianVec([1,-2,3],[.5,1,2]): per-component means"
      (is (h/z-test-passes? 1.0 x0) "E[X_0] = 1.0")
      (is (h/z-test-passes? -2.0 x1) "E[X_1] = -2.0")
      (is (h/z-test-passes? 3.0 x2) "E[X_2] = 3.0"))

    (testing "GaussianVec: per-component variances"
      (is (h/close? 0.25 (h/sample-variance x0) 0.05) "Var[X_0] = 0.25")
      (is (h/close? 1.0 (h/sample-variance x1) 0.15) "Var[X_1] = 1.0")
      (is (h/close? 4.0 (h/sample-variance x2) 0.3) "Var[X_2] = 4.0"))

    (testing "GaussianVec: independence (Cov[X_0,X_1] ~ 0)"
      (is (< (js/Math.abs (sample-covariance x0 x1)) 0.08)
          "Cov[X_0,X_1] ≈ 0 (independence)"))))

;; ==========================================================================
;; Broadcasted Normal (per-element independent, matrix-shaped)
;; ==========================================================================
;; E[X_{ij}] = mu_{ij}, Var[X_{ij}] = sigma_{ij}^2

(deftest broadcasted-normal-moments
  (let [data (sample-n-realize-mv
              (dist/broadcasted-normal (mx/array [[1 2] [3 4]])
                                       (mx/array [[0.5 1] [1.5 2]])) N)
        x00 (extract-element data 0 0)
        x01 (extract-element data 0 1)
        x10 (extract-element data 1 0)
        x11 (extract-element data 1 1)]

    (testing "BroadcastedNormal(2x2): per-element means"
      (is (h/z-test-passes? 1.0 x00) "E[X_{00}] = 1.0")
      (is (h/z-test-passes? 2.0 x01) "E[X_{01}] = 2.0")
      (is (h/z-test-passes? 3.0 x10) "E[X_{10}] = 3.0")
      (is (h/z-test-passes? 4.0 x11) "E[X_{11}] = 4.0"))

    (testing "BroadcastedNormal(2x2): per-element variances"
      (is (h/close? 0.25 (h/sample-variance x00) 0.05) "Var[X_{00}] = 0.25")
      (is (h/close? 1.0 (h/sample-variance x01) 0.15) "Var[X_{01}] = 1.0")
      (is (h/close? 2.25 (h/sample-variance x10) 0.3) "Var[X_{10}] = 2.25")
      (is (h/close? 4.0 (h/sample-variance x11) 0.5) "Var[X_{11}] = 4.0"))))

;; ==========================================================================
;; Wishart (element-wise means + symmetry)
;; ==========================================================================
;; E[W] = df * V

(deftest wishart-moments
  (let [N-wish 2000
        data (sample-n-realize-mv
              (dist/wishart 10 (mx/array [[2.0 0.5] [0.5 1.0]])) N-wish)
        w00 (extract-element data 0 0)
        w01 (extract-element data 0 1)
        w10 (extract-element data 1 0)
        w11 (extract-element data 1 1)]

    (testing "Wishart(10, 2x2): element means"
      ;; Analytical: E[W] = 10 * V = [[20, 5], [5, 10]]
      (is (h/close? 20.0 (h/sample-mean w00) 1.0) "E[W_{00}] = 20.0")
      (is (h/close? 5.0 (h/sample-mean w01) 0.5) "E[W_{01}] = 5.0")
      (is (h/close? 10.0 (h/sample-mean w11) 1.0) "E[W_{11}] = 10.0"))

    (testing "Wishart(10, 2x2): symmetry W_{01} == W_{10}"
      ;; Each sample matrix must be symmetric (float32 tolerance)
      (is (h/all-close? w01 w10 1e-5) "W_{01} == W_{10} for all samples"))))

;; ==========================================================================
;; Inverse Wishart (element-wise means)
;; ==========================================================================
;; E[X] = Psi / (df - k - 1)

(deftest inv-wishart-moments
  (let [N-iw 2000
        data (sample-n-realize-mv
              (dist/inv-wishart 8 (mx/array [[4.0 1.0] [1.0 2.0]])) N-iw)
        x00 (extract-element data 0 0)
        x01 (extract-element data 0 1)
        x11 (extract-element data 1 1)]

    (testing "InvWishart(8, 2x2): element means"
      ;; Analytical: E[X] = Psi/(8-2-1) = Psi/5 = [[0.8, 0.2], [0.2, 0.4]]
      (is (h/close? 0.8 (h/sample-mean x00) 0.15) "E[X_{00}] = 0.8")
      (is (h/close? 0.2 (h/sample-mean x01) 0.08) "E[X_{01}] = 0.2")
      (is (h/close? 0.4 (h/sample-mean x11) 0.15) "E[X_{11}] = 0.4"))))

;; ==========================================================================
;; IID (independent copies of a base distribution)
;; ==========================================================================
;; E[X_i] = E[base], Var[X_i] = Var[base], Cov[X_i,X_j] = 0

(deftest iid-moments
  (let [data (sample-n-realize-mv (dist/iid (dist/gaussian 3 2) 4) N)
        x0 (extract-element data 0)
        x1 (extract-element data 1)]

    (testing "IID(N(3,2), t=4): per-element mean and variance"
      ;; Analytical: E[X_i]=3.0, Var[X_i]=4.0
      (is (h/z-test-passes? 3.0 x0) "E[X_0] = 3.0")
      (is (h/close? 4.0 (h/sample-variance x0) 0.5) "Var[X_0] = 4.0"))

    (testing "IID(N(3,2), t=4): independence"
      ;; SE(Cov) = Var/sqrt(N) = 4/sqrt(5000) = 0.057. Tol = 3.5*SE ≈ 0.20.
      (is (< (js/Math.abs (sample-covariance x0 x1)) 0.20)
          "Cov[X_0,X_1] ≈ 0 (independence)"))))

;; ==========================================================================
;; IID-Gaussian
;; ==========================================================================
;; Scalar mu: E[X_i] = mu, Var[X_i] = sigma^2
;; Vector mu: E[X_i] = mu_i, Var[X_i] = sigma^2

(deftest iid-gaussian-moments
  (testing "IIDGaussian(2, 1.5, 3): scalar mu, per-element mean and variance"
    ;; Analytical: E[X_i]=2.0, Var[X_i]=2.25
    (let [data (sample-n-realize-mv (dist/iid-gaussian 2 1.5 3) N)
          x0 (extract-element data 0)]
      (is (h/z-test-passes? 2.0 x0) "E[X_0] = 2.0")
      (is (h/close? 2.25 (h/sample-variance x0) 0.3) "Var[X_0] = 2.25")))

  (testing "IIDGaussian([1,-1,3], 0.5, 3): vector mu, per-element means"
    ;; Analytical: E[X_0]=1.0, E[X_1]=-1.0, E[X_2]=3.0, Var[X_i]=0.25
    (let [data (sample-n-realize-mv (dist/iid-gaussian (mx/array [1 -1 3]) 0.5 3) N)
          x0 (extract-element data 0)
          x1 (extract-element data 1)
          x2 (extract-element data 2)]
      (is (h/z-test-passes? 1.0 x0) "E[X_0] = 1.0")
      (is (h/z-test-passes? -1.0 x1) "E[X_1] = -1.0")
      (is (h/z-test-passes? 3.0 x2) "E[X_2] = 3.0")
      (is (h/close? 0.25 (h/sample-variance x0) 0.05) "Var[X_0] = 0.25"))))

;; ==========================================================================
;; Wrapped Cauchy (circular moments)
;; ==========================================================================
;; circular mean = mu, R = rho

(deftest wrapped-cauchy-moments
  (testing "WrappedCauchy(1, 0.6): circular mean=1.0, R=0.6"
    (let [d (dist/wrapped-cauchy 1 0.6)
          key (h/deterministic-key)
          keys (rng/split-n key 2000)
          samples (mapv #(h/realize (dist/sample d %)) keys)]
      (is (h/close? 1.0 (circular-mean samples) 0.1) "circular mean ≈ 1.0")
      (is (h/close? 0.6 (mean-resultant-length samples) 0.05) "R ≈ 0.6")))

  (testing "WrappedCauchy(-pi/2, 0.8): circular mean=-1.5708, R=0.8"
    (let [d (dist/wrapped-cauchy (- (/ js/Math.PI 2)) 0.8)
          key (h/deterministic-key)
          keys (rng/split-n key 2000)
          samples (mapv #(h/realize (dist/sample d %)) keys)]
      (is (h/close? -1.5708 (circular-mean samples) 0.1) "circular mean ≈ -pi/2")
      (is (h/close? 0.8 (mean-resultant-length samples) 0.05) "R ≈ 0.8"))))

;; ==========================================================================
;; Wrapped Normal (circular moments)
;; ==========================================================================
;; circular mean = mu, R = exp(-sigma^2/2)

(deftest wrapped-normal-moments
  (testing "WrappedNormal(0, 0.5): circular mean=0, R=0.8825"
    ;; Analytical: R = exp(-0.25/2) = exp(-0.125) = 0.8825
    (let [samples (sample-n-realize (dist/wrapped-normal 0 0.5) N)]
      (is (h/close? 0.0 (circular-mean samples) 0.1) "circular mean ≈ 0")
      (is (h/close? 0.8825 (mean-resultant-length samples) 0.03) "R ≈ 0.8825")))

  (testing "WrappedNormal(2, 1): circular mean=2.0, R=0.6065"
    ;; Analytical: R = exp(-1.0/2) = exp(-0.5) = 0.6065
    (let [samples (sample-n-realize (dist/wrapped-normal 2 1) N)]
      (is (h/close? 2.0 (circular-mean samples) 0.1) "circular mean ≈ 2.0")
      (is (h/close? 0.6065 (mean-resultant-length samples) 0.03) "R ≈ 0.6065"))))

;; ==========================================================================
;; Piecewise Uniform
;; ==========================================================================
;; E[X] = sum_k w_k*(b_k+b_{k+1})/2

(deftest piecewise-uniform-moments
  (testing "PiecewiseUniform([0,1,3,5],[2,1,1]): mean=1.75, variance=2.2708"
    ;; Analytical: w=[0.5,0.25,0.25], E[X]=1.75, Var[X]=2.2708
    (let [samples (sample-n-realize
                   (dist/piecewise-uniform (mx/array [0 1 3 5]) (mx/array [2 1 1])) N)]
      (is (h/z-test-passes? 1.75 samples) "mean converges to 1.75")
      (is (h/close? 2.2708 (h/sample-variance samples) 0.3) "variance converges to 2.2708"))))

;; ==========================================================================
;; Mixture
;; ==========================================================================
;; E[X] = sum_k w_k*E[d_k], Var[X] = sum_k w_k*(Var[d_k]+E[d_k]^2) - E[X]^2

(deftest mixture-moments
  (testing "Mixture(50/50 N(0,1)+N(4,1)): mean=2.0, variance=5.0"
    ;; Analytical: E[X]=2.0, Var[X]=0.5*(1+0)+0.5*(1+16)-4=5.0
    (let [samples (sample-n-realize
                   (dc/mixture [(dist/gaussian 0 1) (dist/gaussian 4 1)]
                               (mx/array [0 0])) N)]
      (is (h/z-test-passes? 2.0 samples) "mean converges to 2.0")
      (is (h/close? 5.0 (h/sample-variance samples) 0.8) "variance converges to 5.0"))))

;; ==========================================================================
;; Delta (trivial sanity check)
;; ==========================================================================
;; E[X] = v, Var[X] = 0

(deftest delta-moments
  (testing "Delta(3.5): all samples exactly 3.5"
    (let [samples (sample-n-realize (dist/delta 3.5) 100)]
      (is (every? #(== 3.5 %) samples) "all 100 samples are exactly 3.5"))))

;; ==========================================================================
;; Run tests
;; ==========================================================================

(cljs.test/run-tests)
