(ns genmlx.directional-dist-test
  "Tests for directional statistics distributions: von-mises, wrapped-cauchy, wrapped-normal."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.gradients :as grad]
            [genmlx.vectorized :as vec])
  (:require-macros [genmlx.gen :refer [gen]]))

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

;; ---------------------------------------------------------------------------
;; Von Mises log-prob numerical accuracy
;; ---------------------------------------------------------------------------

(println "\n== Von Mises log-prob numerical accuracy ==")

;; Reference: log p(x|μ,κ) = κ cos(x-μ) - log(2π) - log(I₀(κ))
;; For μ=0, κ=2, x=1:
;;   2*cos(1) - log(2π) - log(I₀(2))
;;   I₀(2) ≈ 2.27958530... → log ≈ 0.82408...
;;   = 1.08060... - 1.83788... - 0.82408... ≈ -1.58136

(let [d (dist/von-mises 0 2)
      lp (mx/item (dist/log-prob d (mx/scalar 1.0)))
      ;; Compute reference in JS
      i0-2 (+ 1.0 (* (/ 4.0 4.0) (+ 1.0 (* (/ 1.0 4.0) (+ 1.0 (* (/ 1.0 9.0)
              (+ 1.0 (* (/ 1.0 16.0) (+ 1.0 (* (/ 1.0 25.0) (+ 1.0 (/ 1.0 36.0))))))))))))
      ;; Better: use the series sum directly
      ;; I₀(κ) = Σ (κ/2)^(2k) / (k!)²
      ;; For κ=2: terms = 1, 1, 0.25, 0.02778, 0.001736, 0.0000694...
      ref-i0 (+ 1.0 1.0 0.25 (/ 1.0 36.0) (/ 1.0 576.0) (/ 1.0 14400.0) (/ 1.0 518400.0))
      ref-lp (- (* 2.0 (js/Math.cos 1.0)) (js/Math.log (* 2.0 js/Math.PI)) (js/Math.log ref-i0))]
  (assert-close "von-mises log-prob at x=1, mu=0, kappa=2" ref-lp lp 0.01))

;; At the mode (x=μ), log-prob should be maximal: κ - log(2π) - log(I₀(κ))
(let [d (dist/von-mises 0 5)
      lp-mode (mx/item (dist/log-prob d (mx/scalar 0.0)))
      lp-off  (mx/item (dist/log-prob d (mx/scalar 1.0)))]
  (assert-true "log-prob at mode > log-prob off mode" (> lp-mode lp-off)))

;; Symmetry: log-prob(μ+δ) = log-prob(μ-δ)
(let [d (dist/von-mises 1.0 3.0)
      lp-plus  (mx/item (dist/log-prob d (mx/scalar 1.5)))
      lp-minus (mx/item (dist/log-prob d (mx/scalar 0.5)))]
  (assert-close "log-prob symmetric around mu" lp-plus lp-minus 1e-5))

(mx/clear-cache!)

;; ---------------------------------------------------------------------------
;; Von Mises GFI integration (generate, update, assess)
;; ---------------------------------------------------------------------------

(println "\n== Von Mises GFI integration ==")

(let [model (dyn/auto-key (gen []
              (let [angle (trace :angle (dist/von-mises 0 5))]
                angle)))
      ;; simulate
      tr (p/simulate model [])
      angle-val (cm/get-value (cm/get-submap (:choices tr) :angle))]
  (mx/eval! angle-val)
  (assert-true "simulate: angle is finite" (js/isFinite (mx/item angle-val)))
  (assert-true "simulate: score is finite" (js/isFinite (mx/item (:score tr)))))

(mx/clear-cache!)

(let [model (dyn/auto-key (gen []
              (let [angle (trace :angle (dist/von-mises 0 5))]
                angle)))
      ;; generate with constraint
      constraints (cm/choicemap :angle (mx/scalar 1.0))
      {:keys [trace weight]} (p/generate model [] constraints)]
  (let [v (mx/item (cm/get-value (cm/get-submap (:choices trace) :angle)))]
    (assert-close "generate: constrained angle = 1.0" 1.0 v 1e-5))
  (mx/eval! weight)
  (assert-true "generate: weight is finite" (js/isFinite (mx/item weight))))

(mx/clear-cache!)

(let [model (dyn/auto-key (gen []
              (let [angle (trace :angle (dist/von-mises 0 5))]
                angle)))
      ;; assess
      choices (cm/choicemap :angle (mx/scalar 0.5))
      {:keys [weight]} (p/assess model [] choices)]
  (mx/eval! weight)
  (assert-true "assess: weight is finite" (js/isFinite (mx/item weight))))

(mx/clear-cache!)

(let [model (dyn/auto-key (gen []
              (let [angle (trace :angle (dist/von-mises 0 5))]
                angle)))
      ;; update: change constraint
      constraints (cm/choicemap :angle (mx/scalar 1.0))
      {:keys [trace]} (p/generate model [] constraints)
      new-constraints (cm/choicemap :angle (mx/scalar 2.0))
      {:keys [trace weight discard]} (p/update model trace new-constraints)]
  (let [v (mx/item (cm/get-value (cm/get-submap (:choices trace) :angle)))]
    (assert-close "update: angle changed to 2.0" 2.0 v 1e-5))
  (mx/eval! weight)
  (assert-true "update: weight is finite" (js/isFinite (mx/item weight)))
  (assert-true "update: discard returned" (some? discard)))

(mx/clear-cache!)

;; ---------------------------------------------------------------------------
;; Von Mises gradient test
;; ---------------------------------------------------------------------------

(println "\n== Von Mises gradients ==")

;; At the mode (x=μ), d(log p)/dx = -κ sin(x-μ) = 0
(let [model (dyn/auto-key (gen []
              (trace :angle (dist/von-mises 0 5))))
      {:keys [trace]} (p/generate model [] (cm/choicemap :angle (mx/scalar 0.0)))
      grads (grad/choice-gradients model trace [:angle])
      g (mx/item (:angle grads))]
  (assert-close "gradient at mode ≈ 0" 0.0 g 0.1))

(mx/clear-cache!)

;; Off-mode: gradient should be non-zero and finite
(let [model (dyn/auto-key (gen []
              (trace :angle (dist/von-mises 0 5))))
      {:keys [trace]} (p/generate model [] (cm/choicemap :angle (mx/scalar 1.0)))
      grads (grad/choice-gradients model trace [:angle])
      g (mx/item (:angle grads))]
  (assert-true "gradient off-mode is finite" (js/isFinite g))
  (assert-true "gradient off-mode is non-zero" (> (js/Math.abs g) 0.1)))

(mx/clear-cache!)

;; ---------------------------------------------------------------------------
;; Von Mises vectorized inference (vsimulate, vgenerate)
;; ---------------------------------------------------------------------------

(println "\n== Von Mises vectorized inference ==")

(let [model (gen []
              (trace :angle (dist/von-mises 0 5))
              nil)
      n 50
      key (rng/fresh-key)
      vtrace (dyn/vsimulate model [] n key)]
  (assert-true "vsimulate: returns VectorizedTrace"
               (instance? vec/VectorizedTrace vtrace))
  (let [v (cm/get-value (cm/get-submap (:choices vtrace) :angle))]
    (mx/eval! v)
    (assert-true "vsimulate: angle shape is [N]" (= [n] (mx/shape v))))
  (let [score (:score vtrace)]
    (mx/eval! score)
    (assert-true "vsimulate: score shape is [N]" (= [n] (mx/shape score)))))

(mx/clear-cache!)

(let [model (gen []
              (let [angle (trace :angle (dist/von-mises 0 5))]
                (trace :obs (dist/gaussian (mx/cos angle) 0.1))
                nil))
      n 50
      key (rng/fresh-key)
      obs (cm/choicemap :obs (mx/scalar 0.5))
      vtrace (dyn/vgenerate model [] obs n key)]
  (assert-true "vgenerate: returns VectorizedTrace"
               (instance? vec/VectorizedTrace vtrace))
  (let [v (cm/get-value (cm/get-submap (:choices vtrace) :angle))]
    (mx/eval! v)
    (assert-true "vgenerate: angle shape is [N]" (= [n] (mx/shape v))))
  (let [w (:weight vtrace)]
    (mx/eval! w)
    (assert-true "vgenerate: weight shape is [N]" (= [n] (mx/shape w)))))

(mx/clear-cache!)

(println "\n== All directional distribution tests complete ==")
