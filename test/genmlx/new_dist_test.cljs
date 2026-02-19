(ns genmlx.new-dist-test
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm])
  (:require-macros [genmlx.gen :refer [gen]]))

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

(println "\n=== New Distribution Tests ===")

;; =========================================================================
;; Broadcasted Normal
;; =========================================================================

(println "\n-- Broadcasted Normal --")

(let [mu (mx/array [1.0 2.0 3.0])
      sigma (mx/array [0.1 0.2 0.3])
      d (dist/broadcasted-normal mu sigma)
      v (dist/sample d)]
  (mx/eval! v)
  (assert-true "broadcasted-normal sample shape is [3]"
               (= [3] (mx/shape v)))
  (assert-true "broadcasted-normal sample values are finite"
               (every? js/isFinite (mx/->clj v))))

;; Log-prob: sum of independent Gaussian log-probs
(let [mu (mx/array [0.0 0.0])
      sigma (mx/array [1.0 1.0])
      d (dist/broadcasted-normal mu sigma)
      v (mx/array [0.0 0.0])
      lp (dist/log-prob d v)]
  (mx/eval! lp)
  ;; Each element contributes log(1/sqrt(2*pi)) ≈ -0.9189, sum ≈ -1.8379
  (assert-close "broadcasted-normal log-prob at zeros"
                -1.8379 (mx/item lp) 0.01))

;; dist-sample-n produces [N, ...shape]
(let [mu (mx/array [1.0 2.0])
      sigma (mx/array [0.5 0.5])
      d (dist/broadcasted-normal mu sigma)
      samples (dc/dist-sample-n d nil 50)]
  (mx/eval! samples)
  (assert-true "broadcasted-normal sample-n shape is [50,2]"
               (= [50 2] (mx/shape samples))))

;; Statistical validation: sample mean near mu
(let [mu (mx/array [5.0 -3.0])
      sigma (mx/array [0.5 0.5])
      d (dist/broadcasted-normal mu sigma)
      samples (dc/dist-sample-n d nil 1000)
      sample-mean (mx/mean samples [0])]
  (mx/eval! sample-mean)
  (let [means (mx/->clj sample-mean)]
    (assert-close "broadcasted-normal mean[0] ≈ 5" 5.0 (first means) 0.2)
    (assert-close "broadcasted-normal mean[1] ≈ -3" -3.0 (second means) 0.2)))

;; GFI: use inside gen body
(let [model (gen []
              (dyn/trace :x (dist/broadcasted-normal (mx/array [0.0 0.0])
                                                      (mx/array [1.0 1.0]))))
      trace (p/simulate model [])]
  (assert-true "broadcasted-normal works in gen body" (some? trace)))

;; =========================================================================
;; Beta-Uniform Mixture
;; =========================================================================

(println "\n-- Beta-Uniform Mixture --")

(let [d (dist/beta-uniform-mixture 0.5 2.0 5.0)
      v (dist/sample d)]
  (mx/eval! v)
  (let [val (mx/item v)]
    (assert-true "beta-uniform-mixture sample in [0,1]"
                 (and (>= val 0.0) (<= val 1.0)))))

;; Log-prob should be finite for values in (0,1)
(let [d (dist/beta-uniform-mixture 0.5 2.0 5.0)
      lp (dist/log-prob d (mx/scalar 0.3))]
  (mx/eval! lp)
  (assert-true "beta-uniform-mixture log-prob is finite"
               (js/isFinite (mx/item lp))))

;; Statistical: mean of Beta(2,5) is 2/7 ≈ 0.286, uniform mean is 0.5
;; Mixture mean = 0.5 * 0.286 + 0.5 * 0.5 = 0.393
(let [d (dist/beta-uniform-mixture 0.5 2.0 5.0)
      samples (mapv (fn [_]
                      (let [v (dist/sample d)]
                        (mx/eval! v) (mx/item v)))
                    (range 2000))
      mean (/ (reduce + samples) (count samples))]
  (assert-close "beta-uniform-mixture mean ≈ 0.39" 0.393 mean 0.05))

;; GFI
(let [model (gen []
              (dyn/trace :p (dist/beta-uniform-mixture 0.5 2.0 5.0)))
      trace (p/simulate model [])]
  (assert-true "beta-uniform-mixture works in gen body" (some? trace)))

;; =========================================================================
;; Piecewise Uniform
;; =========================================================================

(println "\n-- Piecewise Uniform --")

(let [bounds (mx/array [0.0 1.0 3.0 5.0])
      probs (mx/array [1.0 2.0 1.0])
      d (dist/piecewise-uniform bounds probs)
      v (dist/sample d)]
  (mx/eval! v)
  (let [val (mx/item v)]
    (assert-true "piecewise-uniform sample in [0,5)"
                 (and (>= val 0.0) (< val 5.0)))))

;; Log-prob: bin [0,1) has width 1, prob 1/4 → density 1/4 → log = log(0.25)
(let [bounds (mx/array [0.0 1.0 3.0 5.0])
      probs (mx/array [1.0 2.0 1.0])
      d (dist/piecewise-uniform bounds probs)
      lp (dist/log-prob d (mx/scalar 0.5))]
  (mx/eval! lp)
  ;; total = 4, bin 0: prob=1, width=1 → density = 1/(4*1) = 0.25
  (assert-close "piecewise-uniform log-prob in bin 0"
                (js/Math.log 0.25) (mx/item lp) 0.01))

;; Log-prob in bin 1: [1,3), width 2, prob 2/4 → density = 2/(4*2) = 0.25
(let [bounds (mx/array [0.0 1.0 3.0 5.0])
      probs (mx/array [1.0 2.0 1.0])
      d (dist/piecewise-uniform bounds probs)
      lp (dist/log-prob d (mx/scalar 2.0))]
  (mx/eval! lp)
  (assert-close "piecewise-uniform log-prob in bin 1"
                (js/Math.log 0.25) (mx/item lp) 0.01))

;; Out of bounds → -Inf
(let [bounds (mx/array [0.0 1.0 3.0 5.0])
      probs (mx/array [1.0 2.0 1.0])
      d (dist/piecewise-uniform bounds probs)
      lp (dist/log-prob d (mx/scalar 6.0))]
  (mx/eval! lp)
  (assert-true "piecewise-uniform log-prob out of bounds is -Inf"
               (= ##-Inf (mx/item lp))))

;; GFI
(let [model (gen []
              (dyn/trace :x (dist/piecewise-uniform (mx/array [0.0 1.0 2.0])
                                                     (mx/array [1.0 1.0]))))
      trace (p/simulate model [])]
  (assert-true "piecewise-uniform works in gen body" (some? trace)))

;; =========================================================================
;; Wishart
;; =========================================================================

(println "\n-- Wishart --")

;; 2x2 identity scale, df=5
(let [V (mx/eye 2)
      d (dist/wishart 5 V)
      W (dist/sample d)]
  (mx/eval! W)
  (assert-true "wishart sample shape is [2,2]"
               (= [2 2] (mx/shape W)))
  ;; Wishart sample should be symmetric positive definite
  (let [vals (mx/->clj W)]
    (assert-close "wishart sample is symmetric (W[0,1] ≈ W[1,0])"
                  (get-in vals [0 1]) (get-in vals [1 0]) 0.001)))

;; Log-prob should be finite for a valid PD matrix
(let [V (mx/eye 2)
      d (dist/wishart 5 V)
      W (dist/sample d)
      _ (mx/eval! W)
      lp (dist/log-prob d W)]
  (mx/eval! lp)
  (assert-true "wishart log-prob is finite"
               (js/isFinite (mx/item lp))))

;; Statistical: E[W] = df * V for Wishart(df, V)
;; With V=I, df=5: E[W] = 5*I, so diagonal ≈ 5
(let [V (mx/eye 2)
      d (dist/wishart 5 V)
      n 500
      samples (mapv (fn [_]
                      (let [W (dist/sample d)]
                        (mx/eval! W) (mx/->clj W)))
                    (range n))
      mean-00 (/ (reduce + (map #(get-in % [0 0]) samples)) n)
      mean-11 (/ (reduce + (map #(get-in % [1 1]) samples)) n)
      mean-01 (/ (reduce + (map #(get-in % [0 1]) samples)) n)]
  (assert-close "wishart E[W[0,0]] ≈ 5" 5.0 mean-00 1.0)
  (assert-close "wishart E[W[1,1]] ≈ 5" 5.0 mean-11 1.0)
  (assert-close "wishart E[W[0,1]] ≈ 0" 0.0 mean-01 0.5))

;; =========================================================================
;; Inverse Wishart
;; =========================================================================

(println "\n-- Inverse Wishart --")

;; 2x2 identity scale, df=5
(let [Psi (mx/eye 2)
      d (dist/inv-wishart 5 Psi)
      X (dist/sample d)]
  (mx/eval! X)
  (assert-true "inv-wishart sample shape is [2,2]"
               (= [2 2] (mx/shape X)))
  ;; Should be symmetric
  (let [vals (mx/->clj X)]
    (assert-close "inv-wishart sample is symmetric"
                  (get-in vals [0 1]) (get-in vals [1 0]) 0.001)))

;; Log-prob should be finite
(let [Psi (mx/eye 2)
      d (dist/inv-wishart 5 Psi)
      X (dist/sample d)
      _ (mx/eval! X)
      lp (dist/log-prob d X)]
  (mx/eval! lp)
  (assert-true "inv-wishart log-prob is finite"
               (js/isFinite (mx/item lp))))

;; Statistical: E[X] = Psi / (df - k - 1) for df > k + 1
;; With Psi=I, df=5, k=2: E[X] = I / 2, so diagonal ≈ 0.5
(let [Psi (mx/eye 2)
      d (dist/inv-wishart 5 Psi)
      n 500
      samples (mapv (fn [_]
                      (let [X (dist/sample d)]
                        (mx/eval! X) (mx/->clj X)))
                    (range n))
      mean-00 (/ (reduce + (map #(get-in % [0 0]) samples)) n)
      mean-11 (/ (reduce + (map #(get-in % [1 1]) samples)) n)
      mean-01 (/ (reduce + (map #(get-in % [0 1]) samples)) n)]
  (assert-close "inv-wishart E[X[0,0]] ≈ 0.5" 0.5 mean-00 0.15)
  (assert-close "inv-wishart E[X[1,1]] ≈ 0.5" 0.5 mean-11 0.15)
  (assert-close "inv-wishart E[X[0,1]] ≈ 0" 0.0 mean-01 0.1))

(println "\nAll new distribution tests complete.")
