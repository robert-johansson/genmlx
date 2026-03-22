(ns genmlx.dist-normalization-test
  "Phase 1.6: Normalization tests for discrete distributions.
   For every discrete distribution with finite support, verify that
   probabilities sum to 1 (logsumexp of log-probs ≈ 0).
   Tolerance: 1e-4 (float32 accumulation in logsumexp)."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.dist :as dist]
            [genmlx.mlx :as mx]
            [genmlx.test-helpers :as h]))

(defn- logsumexp
  "Numerically stable logsumexp over a seq of log-probs."
  [log-probs]
  (reduce (fn [acc lp]
            (let [m (max acc lp)]
              (+ m (js/Math.log (+ (js/Math.exp (- acc m))
                                   (js/Math.exp (- lp m)))))))
          ##-Inf
          log-probs))

;; ==========================================================================
;; Bernoulli: support {0, 1}
;; ==========================================================================
;; p(0) + p(1) = (1-p) + p = 1
;; logsumexp(log(1-p), log(p)) = 0

(deftest bernoulli-normalization
  (testing "Bernoulli(0.7) probabilities sum to 1"
    (let [d (dist/bernoulli 0.7)
          lps (mapv #(h/realize (dist/log-prob d (mx/scalar %))) [0.0 1.0])]
      (is (h/close? 0.0 (logsumexp lps) 1e-4)
          "logsumexp ≈ 0")))

  (testing "Bernoulli(0.01) probabilities sum to 1"
    (let [d (dist/bernoulli 0.01)
          lps (mapv #(h/realize (dist/log-prob d (mx/scalar %))) [0.0 1.0])]
      (is (h/close? 0.0 (logsumexp lps) 1e-4))))

  (testing "Bernoulli(0.99) probabilities sum to 1"
    (let [d (dist/bernoulli 0.99)
          lps (mapv #(h/realize (dist/log-prob d (mx/scalar %))) [0.0 1.0])]
      (is (h/close? 0.0 (logsumexp lps) 1e-4)))))

;; ==========================================================================
;; Categorical: support {0, 1, ..., K-1}
;; ==========================================================================
;; softmax(logits) sums to 1 by construction

(deftest categorical-normalization
  (testing "Categorical([1,2,3]) probabilities sum to 1"
    (let [d (dist/categorical (mx/array [1.0 2.0 3.0]))
          lps (mapv #(h/realize (dist/log-prob d (mx/scalar % mx/int32))) (range 3))]
      (is (h/close? 0.0 (logsumexp lps) 1e-4)
          "logsumexp ≈ 0")))

  (testing "Categorical([0,0,0,0,0]) uniform 5-way"
    (let [d (dist/categorical (mx/array [0.0 0.0 0.0 0.0 0.0]))
          lps (mapv #(h/realize (dist/log-prob d (mx/scalar % mx/int32))) (range 5))]
      (is (h/close? 0.0 (logsumexp lps) 1e-4))))

  (testing "Categorical with extreme logits"
    (let [d (dist/categorical (mx/array [10.0 0.0 -10.0]))
          lps (mapv #(h/realize (dist/log-prob d (mx/scalar % mx/int32))) (range 3))]
      (is (h/close? 0.0 (logsumexp lps) 1e-4)))))

;; ==========================================================================
;; Discrete Uniform: support {lo, lo+1, ..., hi}
;; ==========================================================================
;; K values, each with probability 1/K → logsumexp = 0

(deftest discrete-uniform-normalization
  (testing "DiscreteUniform(1,6): 6 values sum to 1"
    (let [d (dist/discrete-uniform 1 6)
          lps (mapv #(h/realize (dist/log-prob d (mx/scalar (double %)))) (range 1 7))]
      (is (h/close? 0.0 (logsumexp lps) 1e-4)
          "logsumexp ≈ 0")))

  (testing "DiscreteUniform(0,1): 2 values sum to 1"
    (let [d (dist/discrete-uniform 0 1)
          lps (mapv #(h/realize (dist/log-prob d (mx/scalar (double %)))) (range 2))]
      (is (h/close? 0.0 (logsumexp lps) 1e-4)))))

;; ==========================================================================
;; Binomial: support {0, 1, ..., n}
;; ==========================================================================
;; sum_{k=0}^{n} C(n,k) p^k (1-p)^{n-k} = 1

(deftest binomial-normalization
  (testing "Binomial(5, 0.5) probabilities sum to 1"
    (let [d (dist/binomial 5 0.5)
          lps (mapv #(h/realize (dist/log-prob d (mx/scalar (double %)))) (range 6))]
      (is (h/close? 0.0 (logsumexp lps) 1e-3)
          "logsumexp ≈ 0")))

  (testing "Binomial(10, 0.3) probabilities sum to 1"
    (let [d (dist/binomial 10 0.3)
          lps (mapv #(h/realize (dist/log-prob d (mx/scalar (double %)))) (range 11))]
      (is (h/close? 0.0 (logsumexp lps) 1e-3)))))

;; ==========================================================================
;; Poisson: support {0, 1, 2, ...} — truncated sum
;; ==========================================================================
;; sum_{k=0}^{K} p(k) → 1 as K → ∞
;; For lambda=3, truncating at k=20 captures > 0.9999999 of mass

(deftest poisson-normalization
  (testing "Poisson(3) truncated sum ≈ 1 (up to k=20)"
    ;; Analytical: sum_{k=0}^{20} e^{-3} 3^k / k! ≈ 1 - 1e-10
    (let [d (dist/poisson 3)
          lps (mapv #(h/realize (dist/log-prob d (mx/scalar (double %)))) (range 21))]
      (is (h/close? 0.0 (logsumexp lps) 1e-3)
          "logsumexp ≈ 0 (truncated at k=20)")))

  (testing "Poisson(1) truncated sum ≈ 1 (up to k=15)"
    (let [d (dist/poisson 1)
          lps (mapv #(h/realize (dist/log-prob d (mx/scalar (double %)))) (range 16))]
      (is (h/close? 0.0 (logsumexp lps) 1e-3)))))

;; ==========================================================================
;; Geometric: support {0, 1, 2, ...} — truncated sum
;; ==========================================================================
;; sum_{k=0}^{K} (1-p)^k * p → 1 as K → ∞
;; For p=0.5, truncating at k=30 captures > 1 - 2^{-31} of mass

(deftest geometric-normalization
  (testing "Geometric(0.5) truncated sum ≈ 1 (up to k=30)"
    (let [d (dist/geometric 0.5)
          lps (mapv #(h/realize (dist/log-prob d (mx/scalar (double %)))) (range 31))]
      (is (h/close? 0.0 (logsumexp lps) 1e-3)
          "logsumexp ≈ 0 (truncated at k=30)")))

  (testing "Geometric(0.3) truncated sum ≈ 1 (up to k=50)"
    (let [d (dist/geometric 0.3)
          lps (mapv #(h/realize (dist/log-prob d (mx/scalar (double %)))) (range 51))]
      (is (h/close? 0.0 (logsumexp lps) 1e-3)))))

;; ==========================================================================
;; Run tests
;; ==========================================================================

(cljs.test/run-tests)
