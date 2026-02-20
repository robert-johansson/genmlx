(ns genmlx.batch-sample-n-test
  "Smoke tests for newly added dist-sample-n implementations."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]))

(defn assert-true [msg pred]
  (if pred
    (println "  PASS:" msg)
    (println "  FAIL:" msg)))

(defn assert-close [msg expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (println "  PASS:" msg)
      (println (str "  FAIL: " msg " (expected " expected ", got " actual ", diff " diff ")")))))

(println "\n=== Batch dist-sample-n Tests ===")

;; ---------------------------------------------------------------------------
;; discrete-uniform
;; ---------------------------------------------------------------------------
(println "\n-- discrete-uniform dist-sample-n --")
(let [d (dist/discrete-uniform (mx/scalar 1 mx/int32) (mx/scalar 6 mx/int32))
      key (rng/fresh-key 42)
      samples (dc/dist-sample-n d key 100)]
  (mx/eval! samples)
  (assert-true "shape is [100]" (= [100] (mx/shape samples)))
  (let [vals (mx/->clj samples)
        mn (apply min vals)
        mx-val (apply max vals)]
    (assert-true "min >= 1" (>= mn 1))
    (assert-true "max <= 6" (<= mx-val 6))))

;; ---------------------------------------------------------------------------
;; geometric
;; ---------------------------------------------------------------------------
(println "\n-- geometric dist-sample-n --")
(let [d (dist/geometric (mx/scalar 0.3))
      key (rng/fresh-key 42)
      samples (dc/dist-sample-n d key 100)]
  (mx/eval! samples)
  (assert-true "shape is [100]" (= [100] (mx/shape samples)))
  (let [vals (mx/->clj samples)
        mn (apply min vals)]
    (assert-true "all >= 0" (>= mn 0))
    ;; Mean should be (1-p)/p = 0.7/0.3 ≈ 2.33
    (let [mean-val (/ (reduce + vals) (count vals))]
      (assert-close "mean ≈ 2.33" 2.33 mean-val 1.5))))

;; ---------------------------------------------------------------------------
;; categorical
;; ---------------------------------------------------------------------------
(println "\n-- categorical dist-sample-n --")
(let [logits (mx/array [0.0 0.0 0.0])  ;; uniform over 3 categories
      d (dist/categorical logits)
      key (rng/fresh-key 42)
      samples (dc/dist-sample-n d key 100)]
  (mx/eval! samples)
  (assert-true "shape is [100]" (= [100] (mx/shape samples)))
  (let [vals (mx/->clj samples)
        mn (apply min vals)
        mx-val (apply max vals)]
    (assert-true "min >= 0" (>= mn 0))
    (assert-true "max <= 2" (<= mx-val 2))))

;; ---------------------------------------------------------------------------
;; multivariate-normal
;; ---------------------------------------------------------------------------
(println "\n-- multivariate-normal dist-sample-n --")
(let [mu (mx/array [1.0 2.0 3.0])
      cov (mx/array [[1.0 0.0 0.0] [0.0 1.0 0.0] [0.0 0.0 1.0]])
      d (dist/multivariate-normal mu cov)
      key (rng/fresh-key 42)
      samples (dc/dist-sample-n d key 100)]
  (mx/eval! samples)
  (assert-true "shape is [100, 3]" (= [100 3] (mx/shape samples)))
  ;; Check mean is roughly [1, 2, 3]
  (let [mean-vec (mx/->clj (mx/mean samples [0]))]
    (assert-close "mean[0] ≈ 1" 1.0 (nth mean-vec 0) 0.5)
    (assert-close "mean[1] ≈ 2" 2.0 (nth mean-vec 1) 0.5)
    (assert-close "mean[2] ≈ 3" 3.0 (nth mean-vec 2) 0.5)))

;; ---------------------------------------------------------------------------
;; binomial
;; ---------------------------------------------------------------------------
(println "\n-- binomial dist-sample-n --")
(let [d (dist/binomial (mx/scalar 20) (mx/scalar 0.5))
      key (rng/fresh-key 42)
      samples (dc/dist-sample-n d key 100)]
  (mx/eval! samples)
  (assert-true "shape is [100]" (= [100] (mx/shape samples)))
  (let [vals (mx/->clj samples)
        mn (apply min vals)
        mx-val (apply max vals)
        mean-val (/ (reduce + vals) (count vals))]
    (assert-true "min >= 0" (>= mn 0))
    (assert-true "max <= 20" (<= mx-val 20))
    ;; Mean should be n*p = 10
    (assert-close "mean ≈ 10" 10.0 mean-val 2.0)))

;; ---------------------------------------------------------------------------
;; student-t
;; ---------------------------------------------------------------------------
(println "\n-- student-t dist-sample-n --")
(let [d (dist/student-t (mx/scalar 10) (mx/scalar 0.0) (mx/scalar 1.0))
      key (rng/fresh-key 42)
      samples (dc/dist-sample-n d key 200)]
  (mx/eval! samples)
  (assert-true "shape is [200]" (= [200] (mx/shape samples)))
  (let [vals (mx/->clj samples)
        mean-val (/ (reduce + vals) (count vals))]
    ;; Student-t with loc=0 should have mean ≈ 0 (for df > 1)
    (assert-close "mean ≈ 0" 0.0 mean-val 0.5)))

(println "\nAll batch dist-sample-n tests complete.")
