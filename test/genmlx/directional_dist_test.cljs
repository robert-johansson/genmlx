(ns genmlx.directional-dist-test
  "Tests for directional statistics distributions: von-mises, wrapped-cauchy, wrapped-normal."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]))

(defn- assert-true [desc pred]
  (if pred
    (println (str "  PASS: " desc))
    (println (str "  FAIL: " desc))))

(defn- assert-close [desc expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (println (str "  PASS: " desc " (expected=" expected " actual=" actual ")"))
      (println (str "  FAIL: " desc " (expected=" expected " actual=" actual " diff=" diff " tol=" tol ")")))))

;; ---------------------------------------------------------------------------
;; Circular mean helper
;; ---------------------------------------------------------------------------

(defn- circular-mean
  "Compute circular mean of a vector of angles."
  [angles]
  (let [n (count angles)
        s (reduce + (map js/Math.sin angles))
        c (reduce + (map js/Math.cos angles))]
    (js/Math.atan2 (/ s n) (/ c n))))

(defn- circular-variance
  "Compute circular variance (1 - R̄) of a vector of angles."
  [angles]
  (let [n (count angles)
        s (reduce + (map js/Math.sin angles))
        c (reduce + (map js/Math.cos angles))
        r-bar (/ (js/Math.sqrt (+ (* s s) (* c c))) n)]
    (- 1.0 r-bar)))

;; ---------------------------------------------------------------------------
;; Von Mises tests
;; ---------------------------------------------------------------------------

(println "\n== Von Mises distribution ==")

(let [d (dist/von-mises 0 5)
      key (rng/fresh-key)
      samples (mapv (fn [k] (mx/item (dc/dist-sample d k)))
                    (rng/split-n key 2000))
      cm (circular-mean samples)]
  (assert-close "mean direction recovery (mu=0, kappa=5)" 0.0 cm 0.15)
  (assert-true "all samples in [-π, π)"
    (every? #(and (>= % (- js/Math.PI)) (< % js/Math.PI)) samples)))

(mx/clear-cache!)

(let [d (dist/von-mises 2.0 10)
      key (rng/fresh-key)
      samples (mapv (fn [k] (mx/item (dc/dist-sample d k)))
                    (rng/split-n key 2000))
      cm (circular-mean samples)]
  (assert-close "mean direction recovery (mu=2, kappa=10)" 2.0 cm 0.15))

(mx/clear-cache!)

(let [d-lo (dist/von-mises 0 1)
      d-hi (dist/von-mises 0 20)
      key (rng/fresh-key)
      [k1 k2] (rng/split key)
      s-lo (mapv (fn [k] (mx/item (dc/dist-sample d-lo k))) (rng/split-n k1 1000))
      s-hi (mapv (fn [k] (mx/item (dc/dist-sample d-hi k))) (rng/split-n k2 1000))
      cv-lo (circular-variance s-lo)
      cv-hi (circular-variance s-hi)]
  (assert-true "higher kappa → lower circular variance"
    (> cv-lo cv-hi)))

(mx/clear-cache!)

(let [d (dist/von-mises 1.0 3.0)
      lp (mx/item (dist/log-prob d (mx/scalar 1.0)))]
  (assert-true "log-prob is finite" (js/isFinite lp)))

(println "\n-- Von Mises dist-sample-n --")
(let [d (dist/von-mises 0 5)
      key (rng/fresh-key)
      batch (dc/dist-sample-n d key 50)]
  (assert-true "batch shape is [50]" (= [50] (mx/shape batch))))

(mx/clear-cache!)

;; ---------------------------------------------------------------------------
;; Wrapped Cauchy tests
;; ---------------------------------------------------------------------------

(println "\n== Wrapped Cauchy distribution ==")

(let [d (dist/wrapped-cauchy 0 0.5)
      key (rng/fresh-key)
      samples (mapv (fn [k] (mx/item (dc/dist-sample d k)))
                    (rng/split-n key 2000))
      cm (circular-mean samples)]
  (assert-close "mean direction recovery (mu=0, rho=0.5)" 0.0 cm 0.15)
  (assert-true "all samples in [-π, π)"
    (every? #(and (>= % (- js/Math.PI)) (< % js/Math.PI)) samples)))

(mx/clear-cache!)

(let [d (dist/wrapped-cauchy 1.5 0.8)
      key (rng/fresh-key)
      samples (mapv (fn [k] (mx/item (dc/dist-sample d k)))
                    (rng/split-n key 2000))
      cm (circular-mean samples)]
  (assert-close "mean direction recovery (mu=1.5, rho=0.8)" 1.5 cm 0.15))

(mx/clear-cache!)

(let [d-lo (dist/wrapped-cauchy 0 0.2)
      d-hi (dist/wrapped-cauchy 0 0.9)
      key (rng/fresh-key)
      [k1 k2] (rng/split key)
      s-lo (mapv (fn [k] (mx/item (dc/dist-sample d-lo k))) (rng/split-n k1 1000))
      s-hi (mapv (fn [k] (mx/item (dc/dist-sample d-hi k))) (rng/split-n k2 1000))
      cv-lo (circular-variance s-lo)
      cv-hi (circular-variance s-hi)]
  (assert-true "higher rho → lower circular variance"
    (> cv-lo cv-hi)))

(mx/clear-cache!)

(let [d (dist/wrapped-cauchy 0 0.5)
      lp (mx/item (dist/log-prob d (mx/scalar 0.5)))]
  (assert-true "log-prob is finite" (js/isFinite lp)))

(println "\n-- Wrapped Cauchy dist-sample-n --")
(let [d (dist/wrapped-cauchy 0 0.5)
      key (rng/fresh-key)
      batch (dc/dist-sample-n d key 50)]
  (assert-true "batch shape is [50]" (= [50] (mx/shape batch))))

(mx/clear-cache!)

;; ---------------------------------------------------------------------------
;; Wrapped Normal tests
;; ---------------------------------------------------------------------------

(println "\n== Wrapped Normal distribution ==")

(let [d (dist/wrapped-normal 0 1.0)
      key (rng/fresh-key)
      samples (mapv (fn [k] (mx/item (dc/dist-sample d k)))
                    (rng/split-n key 2000))
      cm (circular-mean samples)]
  (assert-close "mean direction recovery (mu=0, sigma=1)" 0.0 cm 0.15)
  (assert-true "all samples in [-π, π)"
    (every? #(and (>= % (- js/Math.PI)) (< % js/Math.PI)) samples)))

(mx/clear-cache!)

(let [d (dist/wrapped-normal 2.0 0.5)
      key (rng/fresh-key)
      samples (mapv (fn [k] (mx/item (dc/dist-sample d k)))
                    (rng/split-n key 2000))
      cm (circular-mean samples)]
  (assert-close "mean direction recovery (mu=2, sigma=0.5)" 2.0 cm 0.15))

(mx/clear-cache!)

(let [d-lo (dist/wrapped-normal 0 0.3)
      d-hi (dist/wrapped-normal 0 2.0)
      key (rng/fresh-key)
      [k1 k2] (rng/split key)
      s-lo (mapv (fn [k] (mx/item (dc/dist-sample d-lo k))) (rng/split-n k1 1000))
      s-hi (mapv (fn [k] (mx/item (dc/dist-sample d-hi k))) (rng/split-n k2 1000))
      cv-lo (circular-variance s-lo)
      cv-hi (circular-variance s-hi)]
  (assert-true "smaller sigma → lower circular variance"
    (< cv-lo cv-hi)))

(mx/clear-cache!)

(let [d (dist/wrapped-normal 0 1.0)
      lp (mx/item (dist/log-prob d (mx/scalar 0.5)))]
  (assert-true "log-prob is finite" (js/isFinite lp)))

(println "\n-- Wrapped Normal dist-sample-n --")
(let [d (dist/wrapped-normal 0 1.0)
      key (rng/fresh-key)
      batch (dc/dist-sample-n d key 50)]
  (assert-true "batch shape is [50]" (= [50] (mx/shape batch))))

(mx/clear-cache!)

;; ---------------------------------------------------------------------------
;; Parameter validation tests
;; ---------------------------------------------------------------------------

(println "\n== Parameter validation ==")

(println "\n-- von-mises rejects invalid kappa --")
(let [threw-zero? (try (dist/von-mises 0 0) false (catch :default _ true))
      threw-neg? (try (dist/von-mises 0 -1) false (catch :default _ true))]
  (assert-true "von-mises rejects kappa=0" threw-zero?)
  (assert-true "von-mises rejects kappa<0" threw-neg?))

(println "\n-- wrapped-cauchy rejects invalid rho --")
(let [threw-zero? (try (dist/wrapped-cauchy 0 0) false (catch :default _ true))
      threw-one? (try (dist/wrapped-cauchy 0 1) false (catch :default _ true))
      threw-neg? (try (dist/wrapped-cauchy 0 -0.5) false (catch :default _ true))]
  (assert-true "wrapped-cauchy rejects rho=0" threw-zero?)
  (assert-true "wrapped-cauchy rejects rho=1" threw-one?)
  (assert-true "wrapped-cauchy rejects rho<0" threw-neg?))

(println "\n-- wrapped-normal rejects invalid sigma --")
(let [threw-zero? (try (dist/wrapped-normal 0 0) false (catch :default _ true))
      threw-neg? (try (dist/wrapped-normal 0 -1) false (catch :default _ true))]
  (assert-true "wrapped-normal rejects sigma=0" threw-zero?)
  (assert-true "wrapped-normal rejects sigma<0" threw-neg?))

;; ---------------------------------------------------------------------------
;; Edge case tests
;; ---------------------------------------------------------------------------

(println "\n== Edge cases ==")

(println "\n-- von-mises: very small kappa (0.01) --")
(let [d (dist/von-mises 0 0.01)
      key (rng/fresh-key)
      s (mx/item (dc/dist-sample d key))]
  (assert-true "sample is finite with small kappa" (js/isFinite s)))

(mx/clear-cache!)

(println "\n-- von-mises: very large kappa (100) --")
(let [d (dist/von-mises 1.0 100)
      key (rng/fresh-key)
      samples (mapv (fn [k] (mx/item (dc/dist-sample d k)))
                    (rng/split-n key 200))
      cm (circular-mean samples)]
  (assert-close "concentrated near mu=1 with large kappa" 1.0 cm 0.1))

(mx/clear-cache!)

(println "\n-- wrapped-cauchy: rho near 0 and near 1 --")
(let [d-low (dist/wrapped-cauchy 0 0.01)
      d-high (dist/wrapped-cauchy 0 0.99)
      key (rng/fresh-key)
      [k1 k2] (rng/split key)
      s1 (mx/item (dc/dist-sample d-low k1))
      s2 (mx/item (dc/dist-sample d-high k2))]
  (assert-true "sample finite with rho near 0" (js/isFinite s1))
  (assert-true "sample finite with rho near 1" (js/isFinite s2)))

(println "\n== All directional distribution tests complete ==")
