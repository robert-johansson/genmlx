(ns genmlx.lanczos-test
  "Tests for the optimized Lanczos log-gamma (g=5, 6-term Numerical Recipes)."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]))

;; Reference log-gamma values from known mathematical identities
(def reference-values
  [[0.5  0.5723649429247001]
   [1.0  0.0]
   [2.0  0.0]
   [5.0  3.1780538303479458]
   [10.0 12.801827480081469]
   [50.0 144.56574394634488]])

;; ---------------------------------------------------------------------------
;; Test 1: mlx-log-gamma accuracy via Gamma distribution
;; ---------------------------------------------------------------------------

(deftest log-gamma-via-gamma-dist
  (testing "log-gamma via Gamma(a,1) at x=1"
    (doseq [[x-val expected-lg] reference-values]
      (let [d  (dist/gamma-dist x-val 1.0)
            lp (dist/log-prob d (mx/scalar 1.0))
            _  (mx/eval! lp)
            lp-val (mx/item lp)
            inferred-lg (- (- lp-val) 1.0)]
        (is (h/close? expected-lg inferred-lg 1e-5)
            (str "log-gamma(" x-val ") ~ " expected-lg))))))

;; ---------------------------------------------------------------------------
;; Test 2: Affected distribution log-probs against analytical values
;; ---------------------------------------------------------------------------

(deftest distribution-log-prob-correctness
  (testing "Beta(2,5) at 0.3"
    (let [d  (dist/beta-dist 2 5)
          lp (dist/log-prob d (mx/scalar 0.3))
          _  (mx/eval! lp)]
      (is (h/close? 0.7705 (mx/item lp) 0.01) "Beta(2,5) at 0.3")))

  (testing "Gamma(3,1) at 2.0"
    (let [d  (dist/gamma-dist 3 1)
          lp (dist/log-prob d (mx/scalar 2.0))
          _  (mx/eval! lp)]
      (is (h/close? -1.3069 (mx/item lp) 0.01) "Gamma(3,1) at 2.0")))

  (testing "Poisson(5) at 3"
    (let [d  (dist/poisson 5)
          lp (dist/log-prob d (mx/scalar 3))
          _  (mx/eval! lp)]
      (is (h/close? -1.9636 (mx/item lp) 0.01) "Poisson(5) at 3")))

  (testing "Student-t(3,0,1) at 1.0"
    (let [d  (dist/student-t 3 0 1)
          lp (dist/log-prob d (mx/scalar 1.0))
          _  (mx/eval! lp)]
      (is (h/close? -1.5763 (mx/item lp) 0.01) "Student-t(3,0,1) at 1.0"))))

;; ---------------------------------------------------------------------------
;; Test 3: Vectorized correctness
;; ---------------------------------------------------------------------------

(deftest vectorized-log-gamma
  (testing "Gamma(3,1) vectorized matches scalar"
    (let [d      (dist/gamma-dist 3 1)
          xs     (mx/array [1.0 2.0 3.0 4.0 5.0])
          lp-vec (dist/log-prob d xs)
          _      (mx/eval! lp-vec)
          lp-refs (mapv (fn [x]
                          (let [lp (dist/log-prob d (mx/scalar x))]
                            (mx/eval! lp)
                            (mx/item lp)))
                        [1.0 2.0 3.0 4.0 5.0])]
      (is (= [5] (vec (mx/shape lp-vec))) "Vectorized Gamma log-prob shape is [5]")
      (doseq [i (range 5)]
        (let [vec-val (mx/item (mx/index lp-vec i))
              ref-val (nth lp-refs i)]
          (is (h/close? ref-val vec-val 1e-5)
              (str "Gamma(3,1) vectorized[" i "] matches scalar"))))))

  (testing "Beta(2,5) vectorized matches scalar"
    (let [d      (dist/beta-dist 2 5)
          xs     (mx/array [0.1 0.3 0.5 0.7 0.9])
          lp-vec (dist/log-prob d xs)
          _      (mx/eval! lp-vec)
          lp-refs (mapv (fn [x]
                          (let [lp (dist/log-prob d (mx/scalar x))]
                            (mx/eval! lp)
                            (mx/item lp)))
                        [0.1 0.3 0.5 0.7 0.9])]
      (is (= [5] (vec (mx/shape lp-vec))) "Vectorized Beta log-prob shape is [5]")
      (doseq [i (range 5)]
        (let [vec-val (mx/item (mx/index lp-vec i))
              ref-val (nth lp-refs i)]
          (is (h/close? ref-val vec-val 1e-5)
              (str "Beta(2,5) vectorized[" i "] matches scalar")))))))

(cljs.test/run-tests)
