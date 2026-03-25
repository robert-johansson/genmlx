(ns genmlx.entropy-property-test
  "Entropy verification: for distributions with known closed-form entropy,
   the empirical mean of -log p(x) over N=2000 samples should converge
   to the analytical entropy H(d).

   Tested distributions (10):
     gaussian, uniform, bernoulli, exponential, laplace, log-normal,
     cauchy, delta, geometric, categorical"
  (:require [cljs.test :as t :refer [deftest is testing]]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.test-helpers :as h]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(def ^:private N 2000)

(defn- s [v] (mx/scalar (double v)))

(defn- empirical-entropy
  "Draw N samples from d, compute mean of -log p(x_i).
   Returns a JS number."
  [d n]
  (let [key (rng/fresh-key)
        samples (dc/dist-sample-n d key n)
        lps (dc/dist-log-prob d samples)
        neg-lps (mx/negative lps)
        mean-neg-lp (mx/divide (mx/sum neg-lps) (s n))]
    (mx/eval! mean-neg-lp)
    (mx/item mean-neg-lp)))

(defn- entropy-close?
  "True if |empirical - analytical| <= tol."
  [empirical analytical tol]
  (<= (js/Math.abs (- empirical analytical)) tol))

;; ---------------------------------------------------------------------------
;; Analytical entropy formulas
;; ---------------------------------------------------------------------------

(def ^:private LOG-2PI (js/Math.log (* 2.0 js/Math.PI)))

(defn- gaussian-entropy [sigma]
  (+ 0.5 (* 0.5 LOG-2PI) (js/Math.log sigma)))

(defn- uniform-entropy [lo hi]
  (js/Math.log (- hi lo)))

(defn- bernoulli-entropy [p]
  (if (or (== p 0.0) (== p 1.0))
    0.0
    (- (+ (* p (js/Math.log p))
           (* (- 1.0 p) (js/Math.log (- 1.0 p)))))))

(defn- exponential-entropy [rate]
  (- 1.0 (js/Math.log rate)))

(defn- laplace-entropy [scale]
  (+ 1.0 (js/Math.log (* 2.0 scale))))

(defn- log-normal-entropy [mu sigma]
  (+ mu 0.5 (* 0.5 LOG-2PI) (js/Math.log sigma)))

(defn- cauchy-entropy [scale]
  (js/Math.log (* 4.0 js/Math.PI scale)))

(defn- geometric-entropy [p]
  (/ (+ (- (* (- 1.0 p) (js/Math.log (- 1.0 p))))
        (- (* p (js/Math.log p))))
     p))

(defn- categorical-entropy-from-logits
  "H = -sum(softmax(logits) * log(softmax(logits)))
     = -sum(p * (logits - logsumexp(logits)))
     = logsumexp(logits) - sum(p * logits)"
  [logits]
  (mx/eval! logits)
  (let [log-probs (mx/subtract logits (mx/logsumexp logits))
        probs (mx/exp log-probs)
        h (mx/negative (mx/sum (mx/multiply probs log-probs)))]
    (mx/eval! h)
    (mx/item h)))

;; ---------------------------------------------------------------------------
;; Tests
;; ---------------------------------------------------------------------------

(deftest gaussian-entropy-test
  (testing "gaussian(0,1): H = 0.5 + 0.5*log(2pi) + log(1)"
    (let [d (dist/gaussian (s 0) (s 1))
          expected (gaussian-entropy 1.0)
          empirical (empirical-entropy d N)]
      (is (entropy-close? empirical expected 0.15)
          (str "gaussian(0,1) entropy: expected=" expected " got=" empirical))))
  (testing "gaussian(3,0.5): H = 0.5 + 0.5*log(2pi) + log(0.5)"
    (let [d (dist/gaussian (s 3) (s 0.5))
          expected (gaussian-entropy 0.5)
          empirical (empirical-entropy d N)]
      (is (entropy-close? empirical expected 0.15)
          (str "gaussian(3,0.5) entropy: expected=" expected " got=" empirical))))
  (testing "gaussian(-2,5): H = 0.5 + 0.5*log(2pi) + log(5)"
    (let [d (dist/gaussian (s -2) (s 5))
          expected (gaussian-entropy 5.0)
          empirical (empirical-entropy d N)]
      (is (entropy-close? empirical expected 0.15)
          (str "gaussian(-2,5) entropy: expected=" expected " got=" empirical)))))

(deftest uniform-entropy-test
  (testing "uniform(0,1): H = log(1) = 0"
    (let [d (dist/uniform (s 0) (s 1))
          expected (uniform-entropy 0.0 1.0)
          empirical (empirical-entropy d N)]
      (is (entropy-close? empirical expected 0.15)
          (str "uniform(0,1) entropy: expected=" expected " got=" empirical))))
  (testing "uniform(-5,5): H = log(10)"
    (let [d (dist/uniform (s -5) (s 5))
          expected (uniform-entropy -5.0 5.0)
          empirical (empirical-entropy d N)]
      (is (entropy-close? empirical expected 0.15)
          (str "uniform(-5,5) entropy: expected=" expected " got=" empirical)))))

(deftest bernoulli-entropy-test
  (testing "bernoulli(0.5): H = log(2)"
    (let [d (dist/bernoulli (s 0.5))
          expected (bernoulli-entropy 0.5)
          empirical (empirical-entropy d N)]
      (is (entropy-close? empirical expected 0.1)
          (str "bernoulli(0.5) entropy: expected=" expected " got=" empirical))))
  (testing "bernoulli(0.1): H = -0.1*log(0.1) - 0.9*log(0.9)"
    (let [d (dist/bernoulli (s 0.1))
          expected (bernoulli-entropy 0.1)
          empirical (empirical-entropy d N)]
      (is (entropy-close? empirical expected 0.1)
          (str "bernoulli(0.1) entropy: expected=" expected " got=" empirical))))
  (testing "bernoulli(0.9): H = -0.9*log(0.9) - 0.1*log(0.1)"
    (let [d (dist/bernoulli (s 0.9))
          expected (bernoulli-entropy 0.9)
          empirical (empirical-entropy d N)]
      (is (entropy-close? empirical expected 0.1)
          (str "bernoulli(0.9) entropy: expected=" expected " got=" empirical)))))

(deftest exponential-entropy-test
  (testing "exponential(1): H = 1 - log(1) = 1"
    (let [d (dist/exponential (s 1))
          expected (exponential-entropy 1.0)
          empirical (empirical-entropy d N)]
      (is (entropy-close? empirical expected 0.15)
          (str "exponential(1) entropy: expected=" expected " got=" empirical))))
  (testing "exponential(0.5): H = 1 - log(0.5)"
    (let [d (dist/exponential (s 0.5))
          expected (exponential-entropy 0.5)
          empirical (empirical-entropy d N)]
      (is (entropy-close? empirical expected 0.15)
          (str "exponential(0.5) entropy: expected=" expected " got=" empirical)))))

(deftest laplace-entropy-test
  (testing "laplace(0,1): H = 1 + log(2)"
    (let [d (dist/laplace (s 0) (s 1))
          expected (laplace-entropy 1.0)
          empirical (empirical-entropy d N)]
      (is (entropy-close? empirical expected 0.15)
          (str "laplace(0,1) entropy: expected=" expected " got=" empirical))))
  (testing "laplace(2,3): H = 1 + log(6)"
    (let [d (dist/laplace (s 2) (s 3))
          expected (laplace-entropy 3.0)
          empirical (empirical-entropy d N)]
      (is (entropy-close? empirical expected 0.15)
          (str "laplace(2,3) entropy: expected=" expected " got=" empirical)))))

(deftest log-normal-entropy-test
  (testing "log-normal(0,1): H = 0 + 0.5 + 0.5*log(2pi) + log(1)"
    (let [d (dist/log-normal (s 0) (s 1))
          expected (log-normal-entropy 0.0 1.0)
          empirical (empirical-entropy d N)]
      (is (entropy-close? empirical expected 0.15)
          (str "log-normal(0,1) entropy: expected=" expected " got=" empirical))))
  (testing "log-normal(1,0.5): H = 1 + 0.5 + 0.5*log(2pi) + log(0.5)"
    (let [d (dist/log-normal (s 1) (s 0.5))
          expected (log-normal-entropy 1.0 0.5)
          empirical (empirical-entropy d N)]
      (is (entropy-close? empirical expected 0.15)
          (str "log-normal(1,0.5) entropy: expected=" expected " got=" empirical)))))

(deftest cauchy-entropy-test
  (testing "cauchy(0,1): H = log(4*pi)"
    (let [d (dist/cauchy (s 0) (s 1))
          expected (cauchy-entropy 1.0)
          empirical (empirical-entropy d N)]
      ;; Cauchy has heavy tails — high variance in -log p(x). Use generous tolerance.
      (is (entropy-close? empirical expected 0.3)
          (str "cauchy(0,1) entropy: expected=" expected " got=" empirical)))))

(deftest delta-entropy-test
  (testing "delta(3.14): H = 0 (all mass on one point, log-prob = 0)"
    (let [d (dist/delta (s 3.14))
          empirical (empirical-entropy d N)]
      (is (entropy-close? empirical 0.0 1e-6)
          (str "delta(3.14) entropy: expected=0 got=" empirical))))
  (testing "delta(-1): H = 0"
    (let [d (dist/delta (s -1))
          empirical (empirical-entropy d N)]
      (is (entropy-close? empirical 0.0 1e-6)
          (str "delta(-1) entropy: expected=0 got=" empirical)))))

(deftest geometric-entropy-test
  (testing "geometric(0.8): H = (-(1-p)log(1-p) - p*log(p)) / p"
    ;; p=0.8 has low variance (most mass near 0), safe to test
    (let [d (dist/geometric (s 0.8))
          expected (geometric-entropy 0.8)
          empirical (empirical-entropy d N)]
      (is (entropy-close? empirical expected 0.15)
          (str "geometric(0.8) entropy: expected=" expected " got=" empirical))))
  (testing "geometric(0.3): higher variance but still testable"
    (let [d (dist/geometric (s 0.3))
          expected (geometric-entropy 0.3)
          empirical (empirical-entropy d N)]
      (is (entropy-close? empirical expected 0.25)
          (str "geometric(0.3) entropy: expected=" expected " got=" empirical)))))

(deftest categorical-entropy-test
  (testing "categorical([0,0,0]) uniform 3-way: H = log(3)"
    (let [logits (mx/array [0.0 0.0 0.0])
          d (dist/categorical logits)
          expected (categorical-entropy-from-logits logits)
          empirical (empirical-entropy d N)]
      (is (entropy-close? empirical expected 0.1)
          (str "categorical(uniform-3) entropy: expected=" expected " got=" empirical))))
  (testing "categorical([1,0,-1]) skewed 3-way"
    (let [logits (mx/array [1.0 0.0 -1.0])
          d (dist/categorical logits)
          expected (categorical-entropy-from-logits logits)
          empirical (empirical-entropy d N)]
      (is (entropy-close? empirical expected 0.1)
          (str "categorical(skewed-3) entropy: expected=" expected " got=" empirical))))
  (testing "categorical([0,0]) binary uniform: H = log(2)"
    (let [logits (mx/array [0.0 0.0])
          d (dist/categorical logits)
          expected (categorical-entropy-from-logits logits)
          empirical (empirical-entropy d N)]
      (is (entropy-close? empirical expected 0.1)
          (str "categorical(uniform-2) entropy: expected=" expected " got=" empirical)))))

(t/run-tests)
