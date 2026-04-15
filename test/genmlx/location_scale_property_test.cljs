(ns genmlx.location-scale-property-test
  "Property-based tests for location-scale family distributions.
   For location-scale families (gaussian, laplace, cauchy, student-t),
   shifting the location parameter by delta should shift the sample mean
   by approximately delta (over many samples).

   Log-normal is tested separately: shifting mu by delta multiplies the
   median by exp(delta), so we verify the log-space mean shift instead."
  (:require [cljs.test :as t]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.test-helpers :as h])
  (:require-macros [clojure.test.check.clojure-test :refer [defspec]]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- sample-n-values
  "Sample n values from a distribution, return a vec of JS numbers.
   Uses mx/->clj for bulk extraction (avoids N intermediate mx/index arrays)."
  [d n]
  (let [key (rng/fresh-key)
        samples (dc/dist-sample-n d key n)]
    (mx/eval! samples)
    (mx/->clj samples)))

(defn- sample-mean [xs]
  (/ (reduce + xs) (count xs)))

;; ---------------------------------------------------------------------------
;; Property 1: Gaussian location shift
;; gaussian(loc, scale): E[X] = loc.
;; Shifting loc by delta shifts E[X] by delta.
;; ---------------------------------------------------------------------------

(def gaussian-shift-specs
  [{:scale 1.0 :delta 3.0 :label "N(s=1,d=3)"}
   {:scale 0.5 :delta 5.0 :label "N(s=0.5,d=5)"}
   {:scale 2.0 :delta -4.0 :label "N(s=2,d=-4)"}
   {:scale 1.0 :delta 10.0 :label "N(s=1,d=10)"}
   {:scale 0.1 :delta 1.0 :label "N(s=0.1,d=1)"}])

(defspec gaussian-location-shift 30
  (prop/for-all [spec (gen/elements gaussian-shift-specs)]
    (let [scale (:scale spec)
          delta (:delta spec)
          n 500
          d0 (dist/gaussian (mx/scalar 0.0) (mx/scalar scale))
          d1 (dist/gaussian (mx/scalar delta) (mx/scalar scale))
          samples0 (sample-n-values d0 n)
          samples1 (sample-n-values d1 n)
          mean0 (sample-mean samples0)
          mean1 (sample-mean samples1)
          ;; mean1 - mean0 should be close to delta
          observed-shift (- mean1 mean0)
          ;; Tolerance: 4 * SE of the difference.
          ;; SE(mean1 - mean0) = scale * sqrt(2/n)
          se (* scale (js/Math.sqrt (/ 2.0 n)))
          tol (* 4.0 se)]
      (h/close? delta observed-shift tol))))

;; ---------------------------------------------------------------------------
;; Property 2: Laplace location shift
;; laplace(loc, scale): E[X] = loc.
;; ---------------------------------------------------------------------------

(def laplace-shift-specs
  [{:scale 1.0 :delta 3.0 :label "Lap(s=1,d=3)"}
   {:scale 0.5 :delta 5.0 :label "Lap(s=0.5,d=5)"}
   {:scale 2.0 :delta -4.0 :label "Lap(s=2,d=-4)"}
   {:scale 1.0 :delta 10.0 :label "Lap(s=1,d=10)"}])

(defspec laplace-location-shift 30
  (prop/for-all [spec (gen/elements laplace-shift-specs)]
    (let [scale (:scale spec)
          delta (:delta spec)
          n 500
          d0 (dist/laplace (mx/scalar 0.0) (mx/scalar scale))
          d1 (dist/laplace (mx/scalar delta) (mx/scalar scale))
          samples0 (sample-n-values d0 n)
          samples1 (sample-n-values d1 n)
          mean0 (sample-mean samples0)
          mean1 (sample-mean samples1)
          observed-shift (- mean1 mean0)
          ;; Laplace variance = 2 * scale^2, so SE = scale * sqrt(2*2/n)
          se (* scale (js/Math.sqrt (/ 4.0 n)))
          tol (* 4.0 se)]
      (h/close? delta observed-shift tol))))

;; ---------------------------------------------------------------------------
;; Property 3: Student-t location shift (df >= 3 for finite variance)
;; student-t(df, loc, scale): E[X] = loc (for df > 1).
;; Var[X] = scale^2 * df/(df-2) for df > 2.
;; ---------------------------------------------------------------------------

(def student-t-shift-specs
  [{:df 5.0 :scale 1.0 :delta 3.0 :label "t(5,s=1,d=3)"}
   {:df 10.0 :scale 0.5 :delta 5.0 :label "t(10,s=0.5,d=5)"}
   {:df 30.0 :scale 1.0 :delta -2.0 :label "t(30,s=1,d=-2)"}
   {:df 5.0 :scale 2.0 :delta 4.0 :label "t(5,s=2,d=4)"}])

(defspec student-t-location-shift 30
  (prop/for-all [spec (gen/elements student-t-shift-specs)]
    (let [df (:df spec)
          scale (:scale spec)
          delta (:delta spec)
          n 500
          d0 (dist/student-t (mx/scalar df) (mx/scalar 0.0) (mx/scalar scale))
          d1 (dist/student-t (mx/scalar df) (mx/scalar delta) (mx/scalar scale))
          samples0 (sample-n-values d0 n)
          samples1 (sample-n-values d1 n)
          mean0 (sample-mean samples0)
          mean1 (sample-mean samples1)
          observed-shift (- mean1 mean0)
          ;; Var = scale^2 * df/(df-2)
          var (* scale scale (/ df (- df 2.0)))
          se (js/Math.sqrt (/ (* 2.0 var) n))
          tol (* 4.0 se)]
      (h/close? delta observed-shift tol))))

;; ---------------------------------------------------------------------------
;; Property 4: Log-normal location shift (in log space)
;; log-normal(mu, sigma): log(X) ~ N(mu, sigma).
;; Shifting mu by delta shifts E[log X] by delta.
;; We test in log space to get the location-scale property.
;; ---------------------------------------------------------------------------

(def log-normal-shift-specs
  [{:sigma 0.5 :delta 1.0 :label "LN(s=0.5,d=1)"}
   {:sigma 0.3 :delta 2.0 :label "LN(s=0.3,d=2)"}
   {:sigma 1.0 :delta -1.0 :label "LN(s=1,d=-1)"}
   {:sigma 0.5 :delta 3.0 :label "LN(s=0.5,d=3)"}])

(defspec log-normal-log-space-location-shift 30
  (prop/for-all [spec (gen/elements log-normal-shift-specs)]
    (let [sigma (:sigma spec)
          delta (:delta spec)
          n 500
          d0 (dist/log-normal (mx/scalar 0.0) (mx/scalar sigma))
          d1 (dist/log-normal (mx/scalar delta) (mx/scalar sigma))
          samples0 (sample-n-values d0 n)
          samples1 (sample-n-values d1 n)
          ;; Take logs to get back to location-scale
          log-samples0 (map js/Math.log samples0)
          log-samples1 (map js/Math.log samples1)
          mean0 (sample-mean log-samples0)
          mean1 (sample-mean log-samples1)
          observed-shift (- mean1 mean0)
          ;; In log space, variance is sigma^2, so SE = sigma * sqrt(2/n)
          se (* sigma (js/Math.sqrt (/ 2.0 n)))
          tol (* 4.0 se)]
      (h/close? delta observed-shift tol))))

;; ---------------------------------------------------------------------------
;; Property 5: Gaussian scale invariance
;; gaussian(0, scale): Var[X] = scale^2.
;; Doubling scale should approximately double the standard deviation.
;; ---------------------------------------------------------------------------

(def gaussian-scale-specs
  [{:scale1 1.0 :scale2 2.0 :label "N(1->2)"}
   {:scale1 0.5 :scale2 1.5 :label "N(0.5->1.5)"}
   {:scale1 1.0 :scale2 5.0 :label "N(1->5)"}])

(defspec gaussian-scale-changes-std 20
  (prop/for-all [spec (gen/elements gaussian-scale-specs)]
    (let [s1 (:scale1 spec)
          s2 (:scale2 spec)
          n 500
          d1 (dist/gaussian (mx/scalar 0.0) (mx/scalar s1))
          d2 (dist/gaussian (mx/scalar 0.0) (mx/scalar s2))
          samples1 (sample-n-values d1 n)
          samples2 (sample-n-values d2 n)
          std1 (js/Math.sqrt (h/sample-variance samples1))
          std2 (js/Math.sqrt (h/sample-variance samples2))
          ;; ratio of stds should be close to ratio of scales
          expected-ratio (/ s2 s1)
          observed-ratio (/ std2 std1)
          ;; Tolerance: 30% relative error (sampling noise for 500 samples)
          rel-error (js/Math.abs (/ (- observed-ratio expected-ratio) expected-ratio))]
      (< rel-error 0.30))))

;; ---------------------------------------------------------------------------
;; Property 6: Laplace scale invariance
;; laplace(0, scale): Var[X] = 2*scale^2.
;; ---------------------------------------------------------------------------

(defspec laplace-scale-changes-variance 20
  (prop/for-all [spec (gen/elements [{:scale1 1.0 :scale2 2.0}
                                      {:scale1 0.5 :scale2 1.5}
                                      {:scale1 1.0 :scale2 3.0}])]
    (let [s1 (:scale1 spec)
          s2 (:scale2 spec)
          n 500
          d1 (dist/laplace (mx/scalar 0.0) (mx/scalar s1))
          d2 (dist/laplace (mx/scalar 0.0) (mx/scalar s2))
          samples1 (sample-n-values d1 n)
          samples2 (sample-n-values d2 n)
          var1 (h/sample-variance samples1)
          var2 (h/sample-variance samples2)
          ;; Var = 2*scale^2, so ratio of variances = (s2/s1)^2
          expected-ratio (* (/ s2 s1) (/ s2 s1))
          observed-ratio (/ var2 var1)
          rel-error (js/Math.abs (/ (- observed-ratio expected-ratio) expected-ratio))]
      (< rel-error 0.35))))

;; ---------------------------------------------------------------------------
;; Property 7: Cauchy location shift (median-based)
;; Cauchy has no finite mean, but the median equals the location parameter.
;; Shifting location by delta shifts the median by delta.
;; ---------------------------------------------------------------------------

(defn- sample-median [xs]
  (let [sorted (sort xs)
        n (count sorted)
        mid (quot n 2)]
    (if (odd? n)
      (nth sorted mid)
      (/ (+ (nth sorted (dec mid)) (nth sorted mid)) 2.0))))

(def cauchy-shift-specs
  [{:scale 1.0 :delta 5.0 :label "C(s=1,d=5)"}
   {:scale 0.5 :delta 10.0 :label "C(s=0.5,d=10)"}
   {:scale 2.0 :delta -5.0 :label "C(s=2,d=-5)"}])

(defspec cauchy-location-shift-median 20
  (prop/for-all [spec (gen/elements cauchy-shift-specs)]
    (let [scale (:scale spec)
          delta (:delta spec)
          n 1000 ;; Need more samples for Cauchy (heavy tails)
          d0 (dist/cauchy (mx/scalar 0.0) (mx/scalar scale))
          d1 (dist/cauchy (mx/scalar delta) (mx/scalar scale))
          samples0 (sample-n-values d0 n)
          samples1 (sample-n-values d1 n)
          med0 (sample-median samples0)
          med1 (sample-median samples1)
          observed-shift (- med1 med0)
          ;; Median SE for Cauchy: pi*scale/(2*sqrt(n))
          se (/ (* js/Math.PI scale) (* 2.0 (js/Math.sqrt n)))
          tol (* 5.0 se)]
      (h/close? delta observed-shift tol))))

;; ---------------------------------------------------------------------------
;; Property 8: Exponential rate scaling
;; exponential(rate): E[X] = 1/rate.
;; Doubling the rate halves the mean.
;; ---------------------------------------------------------------------------

(def exponential-rate-specs
  [{:rate1 1.0 :rate2 2.0 :label "Exp(1->2)"}
   {:rate1 0.5 :rate2 1.0 :label "Exp(0.5->1)"}
   {:rate1 1.0 :rate2 5.0 :label "Exp(1->5)"}])

(defspec exponential-rate-scales-mean 20
  (prop/for-all [spec (gen/elements exponential-rate-specs)]
    (let [r1 (:rate1 spec)
          r2 (:rate2 spec)
          n 500
          d1 (dist/exponential (mx/scalar r1))
          d2 (dist/exponential (mx/scalar r2))
          samples1 (sample-n-values d1 n)
          samples2 (sample-n-values d2 n)
          mean1 (sample-mean samples1)
          mean2 (sample-mean samples2)
          ;; E[X] = 1/rate, so mean1/mean2 should be close to rate2/rate1
          expected-ratio (/ r2 r1)
          observed-ratio (/ mean1 mean2)
          rel-error (js/Math.abs (/ (- observed-ratio expected-ratio) expected-ratio))]
      (< rel-error 0.25))))

;; ---------------------------------------------------------------------------
;; Property 9: Zero shift is identity
;; For any location-scale distribution, shifting location by 0
;; should not change the sample mean.
;; ---------------------------------------------------------------------------

(defspec zero-shift-is-identity 30
  (prop/for-all [_dummy (gen/return nil)]
    (let [n 500
          ;; Two independent samples from the same distribution
          d (dist/gaussian (mx/scalar 3.0) (mx/scalar 1.0))
          samples1 (sample-n-values d n)
          samples2 (sample-n-values d n)
          mean1 (sample-mean samples1)
          mean2 (sample-mean samples2)
          ;; Both means should be close to 3.0
          se (/ 1.0 (js/Math.sqrt n))
          tol (* 4.0 se)]
      (and (h/close? 3.0 mean1 tol)
           (h/close? 3.0 mean2 tol)))))

(t/run-tests)
