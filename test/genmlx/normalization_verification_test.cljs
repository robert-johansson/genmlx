(ns genmlx.normalization-verification-test
  "Verify that all distributions normalize: discrete PMFs sum to 1,
   continuous PDFs integrate to 1 via Simpson's rule.

   Uses vectorized log-prob (pass an array of x-values in one shot)
   to avoid Metal resource exhaustion from per-point scalar allocation."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- s [v] (mx/scalar (double v)))

(defn discrete-pmf-sum
  "Sum exp(log-prob(d, k)) for k = 0..upper, using vectorized log-prob.
   Passes a [K] array of int32 values to dist-log-prob, gets back [K] log-probs."
  [d upper]
  (let [ks (mx/array (vec (range (inc upper))) mx/int32)
        lps (dc/dist-log-prob d ks)]
    (mx/eval! lps)
    (reduce + (map #(js/Math.exp %) (mx/->clj lps)))))

(defn discrete-pmf-sum-float
  "Sum exp(log-prob(d, k)) for k = 0..upper, using vectorized log-prob.
   Passes a [K] array of float32 values for distributions that expect floats."
  [d upper]
  (let [ks (mx/array (vec (range (inc upper))))
        lps (dc/dist-log-prob d ks)]
    (mx/eval! lps)
    (reduce + (map #(js/Math.exp %) (mx/->clj lps)))))

(defn simpsons-integrate
  "Vectorized Simpson's rule: create [n+1] array of x-values, compute
   log-prob in one shot, extract densities, apply Simpson weights in JS.
   n must be even."
  [d a b n]
  (let [h (/ (- b a) n)
        ;; Build x-values as a JS array, then convert to MLX
        xs-vec (mapv #(+ a (* % h)) (range (inc n)))
        xs (mx/array xs-vec)
        lps (dc/dist-log-prob d xs)]
    (mx/eval! lps)
    (let [ys (mapv #(js/Math.exp %) (mx/->clj lps))
          sum (+ (nth ys 0)
                 (nth ys n)
                 (* 4 (reduce + (map #(nth ys %) (range 1 n 2))))
                 (* 2 (reduce + (map #(nth ys %) (range 2 (dec n) 2)))))]
      (* (/ h 3) sum))))

;; ---------------------------------------------------------------------------
;; Discrete distributions: PMF sums to 1
;; ---------------------------------------------------------------------------

(deftest bernoulli-normalization
  (testing "bernoulli PMF sums to 1"
    (doseq [[label p] [["p=0.3" 0.3] ["p=0.5" 0.5] ["p=0.7" 0.7]
                        ["p=0.01" 0.01] ["p=0.99" 0.99]]]
      (let [d (dist/bernoulli (s p))
            total (discrete-pmf-sum-float d 1)]
        (is (h/close? total 1.0 1e-5)
            (str "bernoulli(" label ") PMF sum = " total))))))

(deftest categorical-normalization
  (testing "categorical PMF sums to 1"
    (doseq [[label logits] [["3-way" [0.2 0.3 0.5]]
                             ["2-way" [0.5 0.5]]
                             ["4-way" [0.1 0.1 0.1 0.7]]
                             ["uniform-4" [0.25 0.25 0.25 0.25]]
                             ["large-spread" [-2.0 0.0 1.0 3.0 -1.0]]]]
      (let [d (dist/categorical (mx/array logits))
            K (count logits)
            total (discrete-pmf-sum d (dec K))]
        (is (h/close? total 1.0 1e-5)
            (str "categorical(" label ") PMF sum = " total))))))

(deftest binomial-normalization
  (testing "binomial PMF sums to 1"
    (doseq [[label n p] [["n=5,p=0.5" 5 0.5]
                          ["n=10,p=0.3" 10 0.3]
                          ["n=20,p=0.7" 20 0.7]
                          ["n=1,p=0.5" 1 0.5]]]
      (let [d (dist/binomial (s n) (s p))
            total (discrete-pmf-sum-float d n)]
        (is (h/close? total 1.0 1e-4)
            (str "binomial(" label ") PMF sum = " total))))))

(deftest delta-normalization
  (testing "delta PMF at its value is 1"
    (doseq [[label v] [["v=3.14" 3.14] ["v=0" 0.0] ["v=-1" -1.0]]]
      (let [d (dist/delta (s v))
            lp (dc/dist-log-prob d (s v))]
        (mx/eval! lp)
        (let [prob-at-v (js/Math.exp (mx/item lp))]
          (is (h/close? prob-at-v 1.0 1e-5)
              (str "delta(" label ") P(v) = " prob-at-v)))))))

(deftest poisson-normalization
  (testing "poisson PMF sums to ~1 over truncated support"
    (doseq [[label rate] [["rate=1" 1.0]
                           ["rate=5" 5.0]
                           ["rate=10" 10.0]
                           ["rate=0.5" 0.5]]]
      (let [d (dist/poisson (s rate))
            upper (int (+ rate (* 6 (js/Math.sqrt rate)) 30))
            total (discrete-pmf-sum-float d upper)]
        (is (h/close? total 1.0 1e-4)
            (str "poisson(" label ") PMF sum = " total))))))

(deftest geometric-normalization
  (testing "geometric PMF sums to ~1 over truncated support"
    (doseq [[label p] [["p=0.3" 0.3]
                        ["p=0.5" 0.5]
                        ["p=0.8" 0.8]]]
      (let [d (dist/geometric (s p))
            upper (int (+ (js/Math.ceil (/ -10 (js/Math.log (- 1.0 p)))) 20))
            total (discrete-pmf-sum-float d upper)]
        (is (h/close? total 1.0 1e-4)
            (str "geometric(" label ") PMF sum = " total))))))

(deftest neg-binomial-normalization
  (testing "neg-binomial PMF sums to ~1 over truncated support"
    (doseq [[label r p] [["r=3,p=0.5" 3 0.5]
                          ["r=5,p=0.3" 5 0.3]
                          ["r=1,p=0.7" 1 0.7]]]
      (let [d (dist/neg-binomial (s r) (s p))
            mean-val (/ (* r (- 1.0 p)) p)
            std-val (js/Math.sqrt (/ (* r (- 1.0 p)) (* p p)))
            upper (int (+ mean-val (* 8 std-val) 30))
            total (discrete-pmf-sum-float d upper)]
        (is (h/close? total 1.0 1e-3)
            (str "neg-binomial(" label ") PMF sum = " total))))))

;; ---------------------------------------------------------------------------
;; Continuous distributions: PDF integrates to 1 via vectorized Simpson's rule
;; ---------------------------------------------------------------------------

(deftest gaussian-normalization
  (testing "gaussian PDF integrates to 1"
    (doseq [[label mu sigma] [["standard" 0 1]
                               ["shifted" 3 0.5]
                               ["wide" -2 5]]]
      (let [d (dist/gaussian (s mu) (s sigma))
            a (- mu (* 8 sigma))
            b (+ mu (* 8 sigma))
            integral (simpsons-integrate d a b 2000)]
        (is (h/close? integral 1.0 1e-3)
            (str "gaussian(" label ") integral = " integral))))))

(deftest uniform-normalization
  (testing "uniform PDF integrates to 1"
    (doseq [[label lo hi] [["0-1" 0 1]
                            ["-5-5" -5 5]
                            ["2-7" 2 7]]]
      (let [d (dist/uniform (s lo) (s hi))
            integral (simpsons-integrate d lo hi 100)]
        (is (h/close? integral 1.0 1e-5)
            (str "uniform(" label ") integral = " integral))))))

(deftest exponential-normalization
  (testing "exponential PDF integrates to 1"
    (doseq [[label rate] [["rate=1" 1]
                           ["rate=0.5" 0.5]
                           ["rate=3" 3]]]
      (let [d (dist/exponential (s rate))
            b (/ 30.0 rate)
            integral (simpsons-integrate d 0 b 2000)]
        (is (h/close? integral 1.0 1e-3)
            (str "exponential(" label ") integral = " integral))))))

(deftest laplace-normalization
  (testing "laplace PDF integrates to 1"
    (doseq [[label loc scale] [["standard" 0 1]
                                ["shifted" 2 3]
                                ["tight" -1 0.5]]]
      (let [d (dist/laplace (s loc) (s scale))
            a (- loc (* 15 scale))
            b (+ loc (* 15 scale))
            integral (simpsons-integrate d a b 2000)]
        (is (h/close? integral 1.0 1e-3)
            (str "laplace(" label ") integral = " integral))))))

(deftest beta-normalization
  (testing "beta PDF integrates to 1"
    (doseq [[label a b] [["symmetric" 2 2]
                          ["left-skew" 5 1]
                          ["right-skew" 1 5]
                          ["U-shape" 0.5 0.5]]]
      (let [d (dist/beta-dist (s a) (s b))
            integral (simpsons-integrate d 1e-4 (- 1.0 1e-4) 4000)]
        (is (h/close? integral 1.0 0.02)
            (str "beta(" label ") integral = " integral))))))

(deftest gamma-normalization
  (testing "gamma PDF integrates to 1"
    (doseq [[label k r] [["k=2,r=1" 2 1]
                          ["k=0.5,r=2" 0.5 2]
                          ["k=5,r=0.5" 5 0.5]]]
      (let [d (dist/gamma-dist (s k) (s r))
            upper (+ (/ k r) (* 10 (/ (js/Math.sqrt k) r)))
            integral (simpsons-integrate d 1e-4 upper 4000)]
        (is (h/close? integral 1.0 0.02)
            (str "gamma(" label ") integral = " integral))))))

(defn simpsons-integrate-logscale
  "Simpson's rule in log-space for distributions on (0, inf).
   Substitution x = exp(t), dx = exp(t) dt. Integrates over t in [log(a), log(b)]
   with n quadrature points. The integrand is f(exp(t)) * exp(t)."
  [d a b n]
  (let [log-a (js/Math.log a)
        log-b (js/Math.log b)
        h (/ (- log-b log-a) n)
        ts-vec (mapv #(+ log-a (* % h)) (range (inc n)))
        xs-vec (mapv #(js/Math.exp %) ts-vec)
        xs (mx/array xs-vec)
        lps (dc/dist-log-prob d xs)]
    (mx/eval! lps)
    (let [lp-vals (mx/->clj lps)
          ;; f(exp(t)) * exp(t) = exp(log-prob(exp(t)) + t)
          ys (mapv (fn [lp t] (js/Math.exp (+ lp t))) lp-vals ts-vec)
          sum (+ (nth ys 0)
                 (nth ys n)
                 (* 4 (reduce + (map #(nth ys %) (range 1 n 2))))
                 (* 2 (reduce + (map #(nth ys %) (range 2 (dec n) 2)))))]
      (* (/ h 3) sum))))

(deftest log-normal-normalization
  (testing "log-normal PDF integrates to 1"
    ;; Log-normal has support (0, inf). Integrate in log-space for accuracy:
    ;; x = exp(t) transforms the peaked density into a Gaussian-like shape.
    ;; Bounds: exp(mu +/- 6*sigma) captures ~all mass.
    (doseq [[label mu sigma] [["standard" 0 1]
                               ["shifted" 1 0.5]
                               ["wide" -1 2]]]
      (let [d (dist/log-normal (s mu) (s sigma))
            lower (js/Math.exp (- mu (* 6 sigma)))
            upper (js/Math.exp (+ mu (* 6 sigma)))
            integral (simpsons-integrate-logscale d lower upper 4000)]
        (is (h/close? integral 1.0 0.02)
            (str "log-normal(" label ") integral = " integral))))))

(deftest student-t-normalization
  (testing "student-t PDF integrates to 1"
    (doseq [[label df loc scale] [["df=3" 3 0 1]
                                   ["df=10" 10 0 1]
                                   ["df=1(cauchy-like)" 1 2 0.5]]]
      (let [d (dist/student-t (s df) (s loc) (s scale))
            a (- loc (* 50 scale))
            b (+ loc (* 50 scale))
            integral (simpsons-integrate d a b 4000)]
        (is (h/close? integral 1.0 0.02)
            (str "student-t(" label ") integral = " integral))))))

(deftest cauchy-normalization
  (testing "cauchy PDF integrates to 1"
    (doseq [[label loc scale] [["standard" 0 1]
                                ["shifted" 2 0.5]]]
      (let [d (dist/cauchy (s loc) (s scale))
            a (- loc (* 1000 scale))
            b (+ loc (* 1000 scale))
            integral (simpsons-integrate d a b 10000)]
        (is (h/close? integral 1.0 0.02)
            (str "cauchy(" label ") integral = " integral))))))

(deftest inv-gamma-normalization
  (testing "inv-gamma PDF integrates to 1"
    (doseq [[label a b] [["a=2,b=1" 2 1]
                          ["a=3,b=2" 3 2]
                          ["a=5,b=1" 5 1]]]
      (let [d (dist/inv-gamma (s a) (s b))
            upper (+ (/ b a) (* 15 (/ b a)))
            integral (simpsons-integrate d 1e-4 upper 4000)]
        (is (h/close? integral 1.0 0.02)
            (str "inv-gamma(" label ") integral = " integral))))))

(cljs.test/run-tests)
