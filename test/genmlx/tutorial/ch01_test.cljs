(ns genmlx.tutorial.ch01-test
  "Test file for Tutorial Chapter 1: Your First Model.
   Every code listing in the chapter has a corresponding test here."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn])
  (:require-macros [genmlx.gen :refer [gen]]))

(def pass (atom 0))
(def fail (atom 0))

(defn assert-true [msg pred]
  (if pred
    (do (swap! pass inc) (println "  PASS:" msg))
    (do (swap! fail inc) (println "  FAIL:" msg))))

(defn assert-close [msg expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (do (swap! pass inc) (println "  PASS:" msg))
      (do (swap! fail inc) (println "  FAIL:" msg (str "expected=" expected " actual=" actual))))))

;; ============================================================
;; Listing 1.1: The gen macro and trace
;; A simple model that samples a single value from a Gaussian
;; ============================================================
(println "\n== Listing 1.1: gen macro and trace ==")

(def simple-model
  (gen []
    (trace :x (dist/gaussian 0 1))))

(assert-true "simple-model is a DynamicGF" (some? (:body-fn simple-model)))

(let [model (dyn/auto-key simple-model)
      trace (p/simulate model [])]
  (assert-true "simulate returns a trace" (some? trace))
  (assert-true "trace has :choices" (some? (:choices trace)))
  (assert-true "trace has :retval" (some? (:retval trace)))
  (assert-true "trace has :score" (some? (:score trace)))
  (assert-true "retval is an MLX array" (mx/array? (:retval trace)))
  (assert-true "score is finite" (js/Number.isFinite (mx/item (:score trace)))))

;; ============================================================
;; Listing 1.2: Inspecting traces and choice maps
;; ============================================================
(println "\n== Listing 1.2: inspecting traces ==")

(let [model (dyn/auto-key simple-model)
      trace (p/simulate model [])
      choices (:choices trace)]
  (assert-true "choices has :x" (cm/has-value? (cm/get-submap choices :x)))
  (let [x-val (mx/item (cm/get-choice choices [:x]))]
    (assert-true "x value is a number" (number? x-val))
    (assert-true "x value is finite" (js/Number.isFinite x-val)))
  (let [score (mx/item (:score trace))]
    (assert-true "score is negative (log-prob of Gaussian)" (< score 0.5))))

;; ============================================================
;; Listing 1.3: Coin flip model (Beta prior + Bernoulli flips)
;; ============================================================
(println "\n== Listing 1.3: coin flip model ==")

(def coin-model
  (gen []
    (let [bias (trace :bias (dist/beta-dist 2 2))]
      (trace :flip1 (dist/bernoulli bias))
      (trace :flip2 (dist/bernoulli bias))
      (trace :flip3 (dist/bernoulli bias))
      bias)))

(let [model (dyn/auto-key coin-model)
      trace (p/simulate model [])]
  (assert-true "coin model simulates" (some? trace))
  (let [choices (:choices trace)]
    (assert-true "has :bias" (cm/has-value? (cm/get-submap choices :bias)))
    (assert-true "has :flip1" (cm/has-value? (cm/get-submap choices :flip1)))
    (assert-true "has :flip2" (cm/has-value? (cm/get-submap choices :flip2)))
    (assert-true "has :flip3" (cm/has-value? (cm/get-submap choices :flip3))))
  (let [bias-val (mx/item (cm/get-choice (:choices trace) [:bias]))]
    (assert-true "bias is between 0 and 1" (and (>= bias-val 0) (<= bias-val 1))))
  (let [flip1-val (mx/item (cm/get-choice (:choices trace) [:flip1]))]
    (assert-true "flip is 0 or 1" (or (= flip1-val 0.0) (= flip1-val 1.0))))
  (assert-true "retval equals bias" (mx/array? (:retval trace)))
  (assert-true "score is finite" (js/Number.isFinite (mx/item (:score trace)))))

;; ============================================================
;; Listing 1.4: Linear regression model (running example)
;; ============================================================
(println "\n== Listing 1.4: linear regression ==")

(def linear-model
  (gen [xs]
    (let [slope     (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j))
               (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                      intercept) 1)))
      slope)))

(let [model (dyn/auto-key linear-model)
      xs [1.0 2.0 3.0]
      trace (p/simulate model [xs])]
  (assert-true "linear model simulates" (some? trace))
  (assert-true "has :slope" (cm/has-value? (cm/get-submap (:choices trace) :slope)))
  (assert-true "has :intercept" (cm/has-value? (cm/get-submap (:choices trace) :intercept)))
  (assert-true "has :y0" (cm/has-value? (cm/get-submap (:choices trace) :y0)))
  (assert-true "has :y1" (cm/has-value? (cm/get-submap (:choices trace) :y1)))
  (assert-true "has :y2" (cm/has-value? (cm/get-submap (:choices trace) :y2)))
  (assert-true "retval is slope (MLX array)" (mx/array? (:retval trace)))
  (assert-true "score is finite" (js/Number.isFinite (mx/item (:score trace)))))

;; ============================================================
;; Listing 1.5: Running simulate multiple times
;; ============================================================
(println "\n== Listing 1.5: multiple simulations ==")

(let [model (dyn/auto-key linear-model)
      xs [1.0 2.0 3.0]
      traces (mapv (fn [_] (p/simulate model [xs])) (range 5))
      slopes (mapv #(mx/item (cm/get-choice (:choices %) [:slope])) traces)]
  (assert-true "got 5 traces" (= 5 (count traces)))
  (assert-true "slopes are different" (not= (first slopes) (second slopes)))
  (assert-true "all scores finite" (every? #(js/Number.isFinite (mx/item (:score %))) traces)))

;; ============================================================
;; Listing 1.6: What the score means
;; ============================================================
(println "\n== Listing 1.6: score = log p(all choices) ==")

(let [model (dyn/auto-key simple-model)
      trace (p/simulate model [])
      x-val (cm/get-choice (:choices trace) [:x])
      ;; Manually compute log-prob of x under N(0,1)
      manual-lp (mx/item (dc/dist-log-prob (dist/gaussian 0 1) x-val))
      trace-score (mx/item (:score trace))]
  (assert-close "score equals log-prob of the choice"
                manual-lp trace-score 0.0001))

;; ============================================================
;; Listing 1.7: MLX array operations
;; ============================================================
(println "\n== Listing 1.7: MLX arrays ==")

(let [a (mx/scalar 3.0)
      b (mx/scalar 2.0)]
  (assert-true "scalar creates MLX array" (mx/array? a))
  (assert-close "item extracts JS number" 3.0 (mx/item a) 0.001)
  (assert-close "add works" 5.0 (mx/item (mx/add a b)) 0.001)
  (assert-close "multiply works" 6.0 (mx/item (mx/multiply a b)) 0.001)
  (assert-close "subtract works" 1.0 (mx/item (mx/subtract a b)) 0.001)
  (assert-close "divide works" 1.5 (mx/item (mx/divide a b)) 0.001))

(let [arr (mx/array [1.0 2.0 3.0])]
  (assert-true "array from vector" (mx/array? arr))
  (assert-true "shape is [3]" (= [3] (mx/shape arr))))

;; ============================================================
;; Listing 1.8: Distribution catalog (sample of 6)
;; ============================================================
(println "\n== Listing 1.8: distribution catalog ==")

(let [key (rng/fresh-key)
      keys (rng/split-n key 6)]
  ;; Gaussian
  (let [d (dist/gaussian 0 1)
        v (dc/dist-sample d (nth keys 0))]
    (assert-true "gaussian samples" (mx/array? v))
    (assert-true "gaussian log-prob finite" (js/Number.isFinite (mx/item (dc/dist-log-prob d v)))))

  ;; Uniform
  (let [d (dist/uniform 0 1)
        v (dc/dist-sample d (nth keys 1))]
    (assert-true "uniform samples in [0,1]" (let [x (mx/item v)] (and (>= x 0) (<= x 1)))))

  ;; Bernoulli
  (let [d (dist/bernoulli 0.7)
        v (dc/dist-sample d (nth keys 2))]
    (assert-true "bernoulli samples 0 or 1" (let [x (mx/item v)] (or (= x 0.0) (= x 1.0)))))

  ;; Beta
  (let [d (dist/beta-dist 2 5)
        v (dc/dist-sample d (nth keys 3))]
    (assert-true "beta samples in [0,1]" (let [x (mx/item v)] (and (>= x 0) (<= x 1)))))

  ;; Gamma
  (let [d (dist/gamma-dist 2 1)
        v (dc/dist-sample d (nth keys 4))]
    (assert-true "gamma samples positive" (> (mx/item v) 0)))

  ;; Poisson
  (let [d (dist/poisson 5)
        v (dc/dist-sample d (nth keys 5))]
    (assert-true "poisson samples non-negative integer" (>= (mx/item v) 0))))

;; ============================================================
;; Summary
;; ============================================================
(println (str "\n== Chapter 1 tests: " @pass " PASS, " @fail " FAIL =="))
(when (pos? @fail) (js/process.exit 1))
