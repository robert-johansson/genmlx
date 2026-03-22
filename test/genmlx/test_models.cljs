(ns genmlx.test-models
  "Canonical test models for the GenMLX test suite.
   Shared across GFI, inference, and property-based tests."
  (:require [genmlx.dist :as dist]
            [genmlx.mlx :as mx]
            [genmlx.dynamic :as dyn])
  (:require-macros [genmlx.gen :refer [gen]]))

;; Model 1: Single gaussian (simplest possible)
(def single-gaussian
  (dyn/auto-key
   (gen []
        (trace :x (dist/gaussian 0 1)))))

;; Model 2: Two independent gaussians (test score additivity)
(def two-gaussians
  (dyn/auto-key
   (gen []
        (let [x (trace :x (dist/gaussian 0 1))
              y (trace :y (dist/gaussian 0 1))]
          (mx/add x y)))))

;; Model 3: Dependent model (y depends on x)
(def dependent-model
  (dyn/auto-key
   (gen []
        (let [x (trace :x (dist/gaussian 0 1))]
          (trace :y (dist/gaussian x 1))))))

;; Model 4: Linear regression (for inference tests)
(def linear-regression
  (dyn/auto-key
   (gen [xs]
        (let [slope (trace :slope (dist/gaussian 0 10))
              intercept (trace :intercept (dist/gaussian 0 10))]
          (doseq [[j x] (map-indexed vector xs)]
            (trace (keyword (str "y" j))
                   (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                          intercept) 1)))
          slope))))

;; Model 5: Coin flip (for IS tests)
(def coin-model
  (dyn/auto-key
   (gen []
        (let [p (trace :p (dist/beta-dist 1 1))]
          (doseq [i (range 10)]
            (trace (keyword (str "flip" i)) (dist/bernoulli p)))
          p))))

;; Model 6: Multi-distribution (tests variety)
(def multi-dist-model
  (dyn/auto-key
   (gen []
        (let [a (trace :a (dist/gaussian 0 1))
              b (trace :b (dist/exponential 1))
              c (trace :c (dist/bernoulli 0.5))]
          [a b c]))))

;; Model 7: Branching (tests dynamic control flow)
(def branching-model
  (dyn/auto-key
   (gen []
        (let [coin (trace :coin (dist/bernoulli 0.5))]
          (mx/eval! coin)
          (if (pos? (mx/item coin))
            (trace :heads (dist/gaussian 10 1))
            (trace :tails (dist/gaussian -10 1)))))))
