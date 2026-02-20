(ns genmlx.gamma-batch-test
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
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

;; ---------------------------------------------------------------------------
;; Gamma batch sampling
;; ---------------------------------------------------------------------------

(println "\n=== Gamma Batch Sampling Tests ===")

(println "\n-- gamma alpha=2.0, rate=1.0 --")
(let [d (dist/gamma-dist (mx/scalar 2.0) (mx/scalar 1.0))
      n 2000
      samples (dc/dist-sample-n d (rng/fresh-key 42) n)
      _ (mx/eval! samples)
      sh (mx/shape samples)
      vals (mx/->clj samples)
      mean (/ (reduce + vals) (count vals))
      variance (/ (reduce + (map #(* (- % mean) (- % mean)) vals)) (count vals))]
  (assert-true "shape is [2000]" (= sh [n]))
  (assert-true "all samples > 0" (every? pos? vals))
  ;; Gamma(alpha, rate): mean = alpha/rate = 2.0, var = alpha/rate^2 = 2.0
  (assert-close "mean ≈ 2.0" 2.0 mean 0.15)
  (assert-close "variance ≈ 2.0" 2.0 variance 0.3))

(println "\n-- gamma alpha=10.0, rate=2.0 --")
(let [d (dist/gamma-dist (mx/scalar 10.0) (mx/scalar 2.0))
      n 2000
      samples (dc/dist-sample-n d (rng/fresh-key 123) n)
      _ (mx/eval! samples)
      vals (mx/->clj samples)
      mean (/ (reduce + vals) (count vals))
      variance (/ (reduce + (map #(* (- % mean) (- % mean)) vals)) (count vals))]
  (assert-true "all samples > 0" (every? pos? vals))
  ;; mean = 10/2 = 5.0, var = 10/4 = 2.5
  (assert-close "mean ≈ 5.0" 5.0 mean 0.2)
  (assert-close "variance ≈ 2.5" 2.5 variance 0.4))

(println "\n-- gamma alpha=1.0, rate=1.0 (exponential) --")
(let [d (dist/gamma-dist (mx/scalar 1.0) (mx/scalar 1.0))
      n 2000
      samples (dc/dist-sample-n d (rng/fresh-key 77) n)
      _ (mx/eval! samples)
      vals (mx/->clj samples)
      mean (/ (reduce + vals) (count vals))]
  (assert-true "all samples > 0" (every? pos? vals))
  ;; mean = 1/1 = 1.0
  (assert-close "mean ≈ 1.0" 1.0 mean 0.1))

(println "\n-- gamma alpha=0.5, rate=1.0 (Ahrens-Dieter path) --")
(let [d (dist/gamma-dist (mx/scalar 0.5) (mx/scalar 1.0))
      n 2000
      samples (dc/dist-sample-n d (rng/fresh-key 99) n)
      _ (mx/eval! samples)
      vals (mx/->clj samples)
      mean (/ (reduce + vals) (count vals))
      variance (/ (reduce + (map #(* (- % mean) (- % mean)) vals)) (count vals))]
  (assert-true "all samples > 0" (every? pos? vals))
  ;; mean = 0.5, var = 0.5
  (assert-close "mean ≈ 0.5" 0.5 mean 0.1)
  (assert-close "variance ≈ 0.5" 0.5 variance 0.15))

;; ---------------------------------------------------------------------------
;; Inv-gamma batch sampling
;; ---------------------------------------------------------------------------

(println "\n-- inv-gamma shape=3.0, scale=2.0 --")
(let [d (dist/inv-gamma (mx/scalar 3.0) (mx/scalar 2.0))
      n 2000
      samples (dc/dist-sample-n d (rng/fresh-key 55) n)
      _ (mx/eval! samples)
      sh (mx/shape samples)
      vals (mx/->clj samples)
      mean (/ (reduce + vals) (count vals))]
  (assert-true "shape is [2000]" (= sh [n]))
  (assert-true "all samples > 0" (every? pos? vals))
  ;; InvGamma(a,b): mean = b/(a-1) = 2/2 = 1.0 (for a > 1)
  (assert-close "mean ≈ 1.0" 1.0 mean 0.15))

;; ---------------------------------------------------------------------------
;; Beta batch sampling
;; ---------------------------------------------------------------------------

(println "\n-- beta alpha=2.0, beta=5.0 --")
(let [d (dist/beta-dist (mx/scalar 2.0) (mx/scalar 5.0))
      n 2000
      samples (dc/dist-sample-n d (rng/fresh-key 33) n)
      _ (mx/eval! samples)
      sh (mx/shape samples)
      vals (mx/->clj samples)
      mean (/ (reduce + vals) (count vals))]
  (assert-true "shape is [2000]" (= sh [n]))
  (assert-true "all samples in (0,1)" (every? #(and (pos? %) (< % 1.0)) vals))
  ;; Beta(a,b): mean = a/(a+b) = 2/7 ≈ 0.2857
  (assert-close "mean ≈ 0.2857" (/ 2.0 7.0) mean 0.03))

(println "\n-- beta alpha=0.5, beta=0.5 (U-shaped) --")
(let [d (dist/beta-dist (mx/scalar 0.5) (mx/scalar 0.5))
      n 2000
      samples (dc/dist-sample-n d (rng/fresh-key 44) n)
      _ (mx/eval! samples)
      vals (mx/->clj samples)
      mean (/ (reduce + vals) (count vals))]
  ;; With alpha,beta < 1 the density concentrates at 0 and 1;
  ;; float32 can round to exactly 0.0 or 1.0, so use [0,1] not (0,1)
  (assert-true "all samples in [0,1]" (every? #(and (>= % 0.0) (<= % 1.0)) vals))
  ;; Beta(0.5, 0.5): mean = 0.5
  (assert-close "mean ≈ 0.5" 0.5 mean 0.05))

;; ---------------------------------------------------------------------------
;; Dirichlet batch sampling
;; ---------------------------------------------------------------------------

(println "\n-- dirichlet [2.0, 3.0, 5.0] --")
(let [d (dist/dirichlet (mx/array [2.0 3.0 5.0]))
      n 2000
      samples (dc/dist-sample-n d (rng/fresh-key 66) n)
      _ (mx/eval! samples)
      sh (mx/shape samples)]
  (assert-true "shape is [2000 3]" (= sh [n 3]))
  ;; Check each row sums to ~1.0
  (let [row-sums (mx/sum samples [1])
        _ (mx/eval! row-sums)
        sums (mx/->clj row-sums)]
    (assert-true "all rows sum to ~1.0"
                 (every? #(< (js/Math.abs (- % 1.0)) 0.01) sums)))
  ;; Check means: Dirichlet mean_k = alpha_k / sum(alpha) = [0.2, 0.3, 0.5]
  (let [col-means (mx/divide (mx/sum samples [0]) (mx/scalar (float n)))
        _ (mx/eval! col-means)
        means (mx/->clj col-means)]
    (assert-close "mean[0] ≈ 0.2" 0.2 (nth means 0) 0.03)
    (assert-close "mean[1] ≈ 0.3" 0.3 (nth means 1) 0.03)
    (assert-close "mean[2] ≈ 0.5" 0.5 (nth means 2) 0.03)))

;; ---------------------------------------------------------------------------
;; vgenerate benchmark: Bayesian model with gamma/beta priors
;; ---------------------------------------------------------------------------
;; This is where batched gamma really shines — inside vectorized inference
;; the entire model graph (sampling + log-probs + scores + weights) is built
;; once for N particles and evaluated as a single fused computation.

(println "\n-- vgenerate: Bayesian model with gamma/beta priors (N=200) --")

(let [model (gen []
              (let [noise-prec (dyn/trace :noise-prec (dist/gamma-dist (mx/scalar 2.0) (mx/scalar 1.0)))
                    noise-std  (mx/divide (mx/scalar 1.0) (mx/sqrt noise-prec))
                    weight     (dyn/trace :weight (dist/gaussian 0 5))
                    bias       (dyn/trace :bias (dist/gaussian 0 5))]
                (dyn/trace :y0 (dist/gaussian (mx/add bias weight) noise-std))
                (dyn/trace :y1 (dist/gaussian (mx/add bias (mx/multiply weight (mx/scalar 2.0))) noise-std))
                (dyn/trace :y2 (dist/gaussian (mx/add bias (mx/multiply weight (mx/scalar 3.0))) noise-std))
                nil))
      obs (cm/choicemap :y0 (mx/scalar 2.1) :y1 (mx/scalar 3.9) :y2 (mx/scalar 6.2))
      n 200

      ;; Warm up both paths
      _ (do (let [{:keys [trace weight]} (p/generate model [] obs)]
              (mx/eval! (:score trace) weight))
            (let [vt (dyn/vgenerate model [] obs n nil)]
              (mx/eval! (:score vt) (:weight vt))))

      ;; Sequential: N separate generates
      t0 (js/Date.now)
      _ (doseq [_ (range n)]
          (let [{:keys [trace weight]} (p/generate model [] obs)]
            (mx/eval! (:score trace) weight)))
      t1 (js/Date.now)

      ;; Batched: single vgenerate
      t2 (js/Date.now)
      vt (dyn/vgenerate model [] obs n nil)
      _ (mx/eval! (:score vt) (:weight vt))
      t3 (js/Date.now)

      seq-ms (- t1 t0)
      batch-ms (- t3 t2)]
  (println (str "  Sequential (200 x generate): " seq-ms "ms"))
  (println (str "  Batched    (vgenerate 200):   " batch-ms "ms"))
  (when (pos? batch-ms)
    (println (str "  Speedup: " (.toFixed (/ seq-ms batch-ms) 1) "x"))))

(println "\n=== All gamma batch tests complete ===")
