(ns genmlx.dist-statistics-test
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]))

(defn assert-true [msg actual]
  (if actual
    (println "  PASS:" msg)
    (println "  FAIL:" msg "- expected truthy")))

(defn assert-close [msg expected actual tolerance]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tolerance)
      (println "  PASS:" msg)
      (do (println "  FAIL:" msg)
          (println "    expected:" expected "+/-" tolerance)
          (println "    actual:  " actual)))))

(println "\n=== Distribution Statistics Tests ===")

;; ---------------------------------------------------------------------------
;; 21.1 — E[X] and Var[X] for all distributions
;; ---------------------------------------------------------------------------

(def N 10000)
(def N-small 1000) ;; For distributions without native batch sampling

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

(println "\n-- 21.1: E[X] and Var[X] --")

;; Gaussian(2, 3): E=2, Var=9
(println "\n  gaussian(2, 3)")
(let [[m v] (sample-moments (dist/gaussian 2 3))]
  (assert-close "E[gaussian(2,3)] ≈ 2" 2.0 m 0.3)
  (assert-close "Var[gaussian(2,3)] ≈ 9" 9.0 v 1.5))

;; Uniform(1, 5): E=3, Var=16/12
(println "\n  uniform(1, 5)")
(let [[m v] (sample-moments (dist/uniform 1 5))]
  (assert-close "E[uniform(1,5)] ≈ 3" 3.0 m 0.2)
  (assert-close "Var[uniform(1,5)] ≈ 1.333" (/ 16.0 12.0) v 0.3))

;; Bernoulli(0.3): E=0.3, Var=0.21
(println "\n  bernoulli(0.3)")
(let [[m v] (sample-moments (dist/bernoulli 0.3))]
  (assert-close "E[bernoulli(0.3)] ≈ 0.3" 0.3 m 0.05)
  (assert-close "Var[bernoulli(0.3)] ≈ 0.21" 0.21 v 0.05))

;; Beta(2, 5): E=2/7, Var=10/343
(println "\n  beta(2, 5)")
(let [[m v] (sample-moments (dist/beta-dist 2 5))]
  (assert-close "E[beta(2,5)] ≈ 0.286" (/ 2.0 7.0) m 0.05)
  (assert-close "Var[beta(2,5)] ≈ 0.029" (/ 10.0 343.0) v 0.01))

;; Gamma(3, 2): E=3/2, Var=3/4
(println "\n  gamma(3, 2)")
(let [[m v] (sample-moments (dist/gamma-dist 3 2))]
  (assert-close "E[gamma(3,2)] ≈ 1.5" 1.5 m 0.2)
  (assert-close "Var[gamma(3,2)] ≈ 0.75" 0.75 v 0.2))

;; Exponential(2): E=0.5, Var=0.25
(println "\n  exponential(2)")
(let [[m v] (sample-moments (dist/exponential 2))]
  (assert-close "E[exponential(2)] ≈ 0.5" 0.5 m 0.1)
  (assert-close "Var[exponential(2)] ≈ 0.25" 0.25 v 0.1))

;; Poisson(4): E=4, Var=4 (uses sequential fallback, smaller N)
(println "\n  poisson(4)")
(let [[m v] (sample-moments (dist/poisson 4) N-small)]
  (assert-close "E[poisson(4)] ≈ 4" 4.0 m 0.5)
  (assert-close "Var[poisson(4)] ≈ 4" 4.0 v 1.5))

;; Laplace(1, 2): E=1, Var=8
(println "\n  laplace(1, 2)")
(let [[m v] (sample-moments (dist/laplace 1 2))]
  (assert-close "E[laplace(1,2)] ≈ 1" 1.0 m 0.3)
  (assert-close "Var[laplace(1,2)] ≈ 8" 8.0 v 1.5))

;; Log-normal(0, 0.5): E=e^0.125, Var=(e^0.25-1)*e^0.25
(println "\n  log-normal(0, 0.5)")
(let [[m v] (sample-moments (dist/log-normal 0 0.5))
      expected-mean (js/Math.exp 0.125)
      expected-var (* (- (js/Math.exp 0.25) 1) (js/Math.exp 0.25))]
  (assert-close "E[log-normal(0,0.5)] ≈ e^0.125" expected-mean m 0.1)
  (assert-close "Var[log-normal(0,0.5)]" expected-var v 0.1))

;; Geometric(0.3): E=7/3, Var=70/9
(println "\n  geometric(0.3)")
(let [[m v] (sample-moments (dist/geometric 0.3))]
  (assert-close "E[geometric(0.3)] ≈ 2.333" (/ 7.0 3.0) m 0.3)
  (assert-close "Var[geometric(0.3)] ≈ 7.778" (/ 70.0 9.0) v 2.0))

;; Neg-binomial(5, 0.4): E=7.5, Var=18.75 (uses sequential fallback, smaller N)
(println "\n  neg-binomial(5, 0.4)")
(let [[m v] (sample-moments (dist/neg-binomial 5 0.4) N-small)]
  (assert-close "E[neg-binomial(5,0.4)] ≈ 7.5" 7.5 m 1.0)
  (assert-close "Var[neg-binomial(5,0.4)] ≈ 18.75" 18.75 v 6.0))

;; Binomial(20, 0.3): E=6, Var=4.2
(println "\n  binomial(20, 0.3)")
(let [[m v] (sample-moments (dist/binomial 20 0.3))]
  (assert-close "E[binomial(20,0.3)] ≈ 6" 6.0 m 0.5)
  (assert-close "Var[binomial(20,0.3)] ≈ 4.2" 4.2 v 1.0))

;; Discrete-uniform(1, 6): E=3.5, Var=35/12
(println "\n  discrete-uniform(1, 6)")
(let [[m v] (sample-moments (dist/discrete-uniform 1 6))]
  (assert-close "E[discrete-uniform(1,6)] ≈ 3.5" 3.5 m 0.2)
  (assert-close "Var[discrete-uniform(1,6)] ≈ 2.917" (/ 35.0 12.0) v 0.6))

;; Truncated-normal(0, 1, -2, 2): E=0, Var≈0.774
(println "\n  truncated-normal(0, 1, -2, 2)")
(let [[m v] (sample-moments (dist/truncated-normal 0 1 -2 2))]
  (assert-close "E[truncated-normal(0,1,-2,2)] ≈ 0" 0.0 m 0.1)
  (assert-close "Var[truncated-normal(0,1,-2,2)] ≈ 0.774" 0.774 v 0.1))

;; Student-t(5, 0, 1): E=0, Var=5/3
(println "\n  student-t(5, 0, 1)")
(let [[m v] (sample-moments (dist/student-t 5 0 1))]
  (assert-close "E[student-t(5,0,1)] ≈ 0" 0.0 m 0.2)
  (assert-close "Var[student-t(5,0,1)] ≈ 1.667" (/ 5.0 3.0) v 0.5))

;; Inv-gamma(4, 2): E=2/3, Var=2/9
(println "\n  inv-gamma(4, 2)")
(let [[m v] (sample-moments (dist/inv-gamma 4 2))]
  (assert-close "E[inv-gamma(4,2)] ≈ 0.667" (/ 2.0 3.0) m 0.1)
  (assert-close "Var[inv-gamma(4,2)] ≈ 0.222" (/ 2.0 9.0) v 0.1))

;; MVN: component-wise means
(println "\n  multivariate-normal (component means)")
(let [mu (mx/array [1.0 2.0 3.0])
      cov (mx/array [[1.0 0.0 0.0] [0.0 1.0 0.0] [0.0 0.0 1.0]])
      d (dist/multivariate-normal mu cov)
      key (rng/fresh-key)
      samples (dc/dist-sample-n d key N)
      _ (mx/eval! samples)
      means (mx/mean samples [0])]
  (mx/eval! means)
  (let [m (mx/->clj means)]
    (assert-close "E[MVN][0] ≈ 1" 1.0 (nth m 0) 0.2)
    (assert-close "E[MVN][1] ≈ 2" 2.0 (nth m 1) 0.2)
    (assert-close "E[MVN][2] ≈ 3" 3.0 (nth m 2) 0.2)))

;; Dirichlet: component-wise means = alpha_i / sum(alpha)
(println "\n  dirichlet (component means)")
(let [alpha (mx/array [2.0 3.0 5.0])
      d (dist/dirichlet alpha)
      key (rng/fresh-key)
      samples (dc/dist-sample-n d key N)
      _ (mx/eval! samples)
      means (mx/mean samples [0])]
  (mx/eval! means)
  (let [m (mx/->clj means)]
    (assert-close "E[dirichlet][0] ≈ 0.2" 0.2 (nth m 0) 0.05)
    (assert-close "E[dirichlet][1] ≈ 0.3" 0.3 (nth m 1) 0.05)
    (assert-close "E[dirichlet][2] ≈ 0.5" 0.5 (nth m 2) 0.05)))

;; ---------------------------------------------------------------------------
;; 21.2 — Discrete PMF sums to 1
;; ---------------------------------------------------------------------------

(println "\n-- 21.2: Discrete PMF sums to 1 --")

(defn pmf-sum
  "Sum exp(log-prob(v)) for values in vs."
  [d vs]
  (let [total (reduce (fn [acc v]
                        (let [lp (dc/dist-log-prob d (mx/scalar v))]
                          (mx/eval! lp)
                          (+ acc (js/Math.exp (mx/item lp)))))
                      0.0 vs)]
    total))

;; Bernoulli(0.3): {0, 1}
(let [s (pmf-sum (dist/bernoulli 0.3) [0 1])]
  (assert-close "bernoulli(0.3) PMF sums to 1" 1.0 s 0.001))

;; Bernoulli(0.7): {0, 1}
(let [s (pmf-sum (dist/bernoulli 0.7) [0 1])]
  (assert-close "bernoulli(0.7) PMF sums to 1" 1.0 s 0.001))

;; Categorical([0.2, 0.3, 0.5]): {0, 1, 2}
(let [logits (mx/log (mx/array [0.2 0.3 0.5]))
      s (pmf-sum (dist/categorical logits) [0 1 2])]
  (assert-close "categorical([0.2,0.3,0.5]) PMF sums to 1" 1.0 s 0.001))

;; Binomial(10, 0.4): {0..10}
(let [s (pmf-sum (dist/binomial 10 0.4) (range 11))]
  (assert-close "binomial(10,0.4) PMF sums to 1" 1.0 s 0.001))

;; Discrete-uniform(3, 8): {3..8}
(let [s (pmf-sum (dist/discrete-uniform 3 8) (range 3 9))]
  (assert-close "discrete-uniform(3,8) PMF sums to 1" 1.0 s 0.001))

;; Geometric(0.3): {0..30} (truncated, verify sum > 0.999)
(let [s (pmf-sum (dist/geometric 0.3) (range 31))]
  (assert-true "geometric(0.3) PMF sum > 0.999" (> s 0.999)))

;; Poisson(3): {0..20} (truncated, verify sum > 0.999)
(let [s (pmf-sum (dist/poisson 3) (range 21))]
  (assert-true "poisson(3) PMF sum > 0.999" (> s 0.999)))

;; Neg-binomial(5, 0.4): {0..40} (truncated)
(let [s (pmf-sum (dist/neg-binomial 5 0.4) (range 41))]
  (assert-true "neg-binomial(5,0.4) PMF sum > 0.99" (> s 0.99)))

(println "\nAll distribution statistics tests complete.")
