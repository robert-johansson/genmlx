(ns genmlx.dist-test
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]))

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

(defn test-dist [name d expected-range]
  (println (str "\n-- " name " --"))
  ;; Test sampling
  (let [v (dist/sample d)]
    (mx/eval! v)
    (let [val (mx/item v)]
      (assert-true (str name " sample is number") (number? val))
      (when expected-range
        (let [[lo hi] expected-range]
          (assert-true (str name " sample in range")
                       (and (>= val lo) (<= val hi)))))))
  ;; Test log-prob
  (let [v (dist/sample d)
        lp (dist/log-prob d v)]
    (mx/eval! lp)
    (let [lp-val (mx/item lp)]
      (assert-true (str name " log-prob is number") (number? lp-val))
      (assert-true (str name " log-prob is finite") (js/isFinite lp-val)))))

(println "\n=== Distribution Tests ===")

;; Gaussian
(test-dist "Gaussian(0,1)" (dist/gaussian 0 1) nil)
(let [d (dist/gaussian 0 1)
      lp (dist/log-prob d (mx/scalar 0.0))]
  (mx/eval! lp)
  ;; log(1/sqrt(2*pi)) ≈ -0.9189
  (assert-close "Gaussian(0,1) log-prob at 0" -0.9189 (mx/item lp) 0.01))

;; Uniform
(test-dist "Uniform(0,1)" (dist/uniform 0 1) [0 1])
(let [d (dist/uniform 0 1)
      lp (dist/log-prob d (mx/scalar 0.5))]
  (mx/eval! lp)
  (assert-close "Uniform(0,1) log-prob" 0.0 (mx/item lp) 0.001))

;; Bernoulli
(test-dist "Bernoulli(0.5)" (dist/bernoulli 0.5) [0 1])
(let [d (dist/bernoulli 0.7)
      lp1 (dist/log-prob d (mx/scalar 1.0))
      lp0 (dist/log-prob d (mx/scalar 0.0))]
  (mx/eval! lp1 lp0)
  (assert-close "Bernoulli log-prob(1)" (js/Math.log 0.7) (mx/item lp1) 0.001)
  (assert-close "Bernoulli log-prob(0)" (js/Math.log 0.3) (mx/item lp0) 0.001))

;; Exponential
(test-dist "Exponential(1)" (dist/exponential 1) [0 js/Infinity])
(let [d (dist/exponential 2)
      lp (dist/log-prob d (mx/scalar 1.0))]
  (mx/eval! lp)
  ;; log(2) - 2*1 = 0.693 - 2 = -1.307
  (assert-close "Exponential log-prob" -1.307 (mx/item lp) 0.01))

;; Beta
(test-dist "Beta(2,5)" (dist/beta-dist 2 5) [0 1])

;; Gamma
(test-dist "Gamma(2,1)" (dist/gamma-dist 2 1) [0 js/Infinity])

;; Laplace
(test-dist "Laplace(0,1)" (dist/laplace 0 1) nil)

;; Log-Normal
(test-dist "LogNormal(0,1)" (dist/log-normal 0 1) [0 js/Infinity])

;; Poisson
(test-dist "Poisson(3)" (dist/poisson 3) [0 js/Infinity])

;; Delta
(println "\n-- Delta --")
(let [d (dist/delta 5.0)
      v (dist/sample d)]
  (mx/eval! v)
  (assert-close "Delta always returns value" 5.0 (mx/item v) 0.001))

;; Categorical
(println "\n-- Categorical --")
(let [logits (mx/array [(js/Math.log 0.2) (js/Math.log 0.3) (js/Math.log 0.5)])
      d (dist/categorical logits)
      v (dist/sample d)]
  (mx/eval! v)
  (let [val (mx/item v)]
    (assert-true "Categorical sample is integer-valued" (and (>= val 0) (<= val 2)))))

;; Test GFI bridge
(println "\n-- GFI bridge --")
(let [d (dist/gaussian 0 1)
      trace (genmlx.protocols/simulate d [])]
  (assert-true "dist implements GFI simulate" (some? trace)))

;; Test statistical properties with many samples
(println "\n-- Statistical validation --")
(let [d (dist/gaussian 5 2)
      samples (mapv (fn [_]
                      (let [v (dist/sample d)]
                        (mx/eval! v)
                        (mx/item v)))
                    (range 1000))
      mean (/ (reduce + samples) (count samples))
      variance (/ (reduce + (map #(let [d (- % mean)] (* d d)) samples))
                  (count samples))]
  (assert-close "Gaussian mean ≈ 5" 5.0 mean 0.3)
  (assert-close "Gaussian variance ≈ 4" 4.0 variance 1.0))

(println "\nAll distribution tests complete.")
