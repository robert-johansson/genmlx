(ns genmlx.lanczos-test
  "Tests for the optimized Lanczos log-gamma (g=5, 6-term Numerical Recipes).
   Verifies accuracy of log-gamma and affected distribution log-probs."
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

;; Reference log-gamma values from known mathematical identities:
;; log-gamma(0.5) = log(sqrt(pi)) ≈ 0.5723649429247
;; log-gamma(1)   = 0
;; log-gamma(2)   = 0
;; log-gamma(5)   = log(4!) = log(24) ≈ 3.17805383035
;; log-gamma(10)  = log(9!) = log(362880) ≈ 12.80182748008
;; log-gamma(50)  = log(49!) ≈ 144.56574394634

(def reference-values
  [[0.5  0.5723649429247001]
   [1.0  0.0]
   [2.0  0.0]
   [5.0  3.1780538303479458]
   [10.0 12.801827480081469]
   [50.0 144.56574394634488]])

(println "\n=== Lanczos Log-Gamma Accuracy Tests ===")

;; ---------------------------------------------------------------------------
;; Test 1: mlx-log-gamma accuracy via Gamma distribution
;; ---------------------------------------------------------------------------
;; Test mlx-log-gamma indirectly: Gamma(a,1) log-prob at x=1 uses mlx-log-gamma(a).
;; log-prob of Gamma(a, b=1) at x=1:
;;   = (a-1)*log(1) - 1/1 - log-gamma(a) - a*log(1)
;;   = 0 - 1 - log-gamma(a) - 0
;;   = -1 - log-gamma(a)
;; So mlx-log-gamma(a) = -(log-prob + 1)

(println "\n-- log-gamma via Gamma(a,1) at x=1 --")
(doseq [[x-val expected-lg] reference-values]
  (let [d  (dist/gamma-dist x-val 1.0)
        lp (dist/log-prob d (mx/scalar 1.0))
        _  (mx/eval! lp)
        lp-val (mx/item lp)
        ;; log-prob = -1 - log-gamma(a) => log-gamma(a) = -(lp + 1)
        inferred-lg (- (- lp-val) 1.0)]
    (assert-close (str "log-gamma(" x-val ") ≈ " expected-lg)
                  expected-lg inferred-lg 1e-5)))

;; ---------------------------------------------------------------------------
;; Test 2: Affected distribution log-probs against analytical values
;; ---------------------------------------------------------------------------
(println "\n-- Distribution log-prob correctness --")

;; Beta(2,5) at x=0.3
(let [d  (dist/beta-dist 2 5)
      lp (dist/log-prob d (mx/scalar 0.3))
      _  (mx/eval! lp)
      lp-val (mx/item lp)]
  (assert-close "Beta(2,5) at 0.3" 0.7705 lp-val 0.01))

;; Gamma(3,1) at x=2.0
(let [d  (dist/gamma-dist 3 1)
      lp (dist/log-prob d (mx/scalar 2.0))
      _  (mx/eval! lp)
      lp-val (mx/item lp)]
  (assert-close "Gamma(3,1) at 2.0" -1.3069 lp-val 0.01))

;; Poisson(5) at x=3
(let [d  (dist/poisson 5)
      lp (dist/log-prob d (mx/scalar 3))
      _  (mx/eval! lp)
      lp-val (mx/item lp)]
  (assert-close "Poisson(5) at 3" -1.9636 lp-val 0.01))

;; Student-t(3, 0, 1) at x=1.0
;; Computed: log-gamma(2) - log-gamma(1.5) - 0.5*log(3*pi) - log(1) - 2*log(1+1/3)
;;         = 0 - (-0.1208) - 1.1216 - 0 - 0.5754 ≈ -1.5763
(let [d  (dist/student-t 3 0 1)
      lp (dist/log-prob d (mx/scalar 1.0))
      _  (mx/eval! lp)
      lp-val (mx/item lp)]
  (assert-close "Student-t(3,0,1) at 1.0" -1.5763 lp-val 0.01))

;; ---------------------------------------------------------------------------
;; Test 3: Vectorized correctness — mlx-log-gamma on [N]-shaped inputs
;; ---------------------------------------------------------------------------
(println "\n-- Vectorized log-gamma correctness --")

;; Gamma(3,1) log-prob on a batch of values should give correct per-element results
(let [d      (dist/gamma-dist 3 1)
      xs     (mx/array [1.0 2.0 3.0 4.0 5.0])
      lp-vec (dist/log-prob d xs)
      _      (mx/eval! lp-vec)
      ;; Also compute scalar log-probs for reference
      lp-refs (mapv (fn [x]
                      (let [lp (dist/log-prob d (mx/scalar x))]
                        (mx/eval! lp)
                        (mx/item lp)))
                    [1.0 2.0 3.0 4.0 5.0])]
  (assert-true "Vectorized Gamma log-prob shape is [5]"
               (= [5] (vec (mx/shape lp-vec))))
  (doseq [i (range 5)]
    (let [vec-val (mx/item (mx/index lp-vec i))
          ref-val (nth lp-refs i)]
      (assert-close (str "Gamma(3,1) vectorized[" i "] matches scalar")
                    ref-val vec-val 1e-5))))

;; Beta(2,5) vectorized
(let [d      (dist/beta-dist 2 5)
      xs     (mx/array [0.1 0.3 0.5 0.7 0.9])
      lp-vec (dist/log-prob d xs)
      _      (mx/eval! lp-vec)
      lp-refs (mapv (fn [x]
                      (let [lp (dist/log-prob d (mx/scalar x))]
                        (mx/eval! lp)
                        (mx/item lp)))
                    [0.1 0.3 0.5 0.7 0.9])]
  (assert-true "Vectorized Beta log-prob shape is [5]"
               (= [5] (vec (mx/shape lp-vec))))
  (doseq [i (range 5)]
    (let [vec-val (mx/item (mx/index lp-vec i))
          ref-val (nth lp-refs i)]
      (assert-close (str "Beta(2,5) vectorized[" i "] matches scalar")
                    ref-val vec-val 1e-5))))

(println "\n=== Lanczos Tests Complete ===")
