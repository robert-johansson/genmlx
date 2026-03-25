(ns genmlx.searchsorted-test
  "Tests for mx/searchsorted and O(N) systematic resampling."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.vectorized :as vec]))

(defn- assert-array-equal [expected-vec actual-arr]
  (mx/eval! actual-arr)
  (= expected-vec (mx/->clj actual-arr)))

;; =========================================================================
;; Section 1: searchsorted basic correctness
;; =========================================================================

(deftest searchsorted-basic-left
  (testing "basic searchsorted (left side)"
    (let [sorted (mx/array [1 3 5 7])
          values (mx/array [2 4 6])
          result (mx/searchsorted sorted values)]
      (is (assert-array-equal [1 2 3] result) "basic: [1,3,5,7] search [2,4,6] -> [1,2,3]"))

    (let [sorted (mx/array [1 3 5 7])
          values (mx/array [0 1 3 5 7 8])
          result (mx/searchsorted sorted values)]
      (is (assert-array-equal [0 0 1 2 3 4] result) "boundary: values at and beyond edges"))

    (let [sorted (mx/array [1 3 3 5])
          values (mx/array [3])
          result (mx/searchsorted sorted values)]
      (is (assert-array-equal [1] result) "left side: duplicates -> first valid index"))))

;; =========================================================================
;; Section 2: searchsorted right side
;; =========================================================================

(deftest searchsorted-right
  (testing "right side"
    (let [sorted (mx/array [1 3 3 5])
          values (mx/array [3])
          result (mx/searchsorted sorted values :right)]
      (is (assert-array-equal [3] result) "right side: duplicates -> after last"))

    (let [sorted (mx/array [1 3 5 7])
          values (mx/array [0 1 7 8])
          result-l (mx/searchsorted sorted values :left)
          result-r (mx/searchsorted sorted values :right)]
      (is (assert-array-equal [0 0 3 4] result-l) "left: boundary values")
      (is (assert-array-equal [0 1 4 4] result-r) "right: boundary values"))))

;; =========================================================================
;; Section 3: Edge cases
;; =========================================================================

(deftest searchsorted-edge-cases
  (testing "edge cases"
    (let [sorted (mx/array [10 20 30])
          values (mx/array [5])
          result (mx/searchsorted sorted values)]
      (is (assert-array-equal [0] result) "value < all -> 0"))

    (let [sorted (mx/array [10 20 30])
          values (mx/array [35])
          result (mx/searchsorted sorted values)]
      (is (assert-array-equal [3] result) "value > all -> N"))

    (let [sorted (mx/array [5])
          values (mx/array [3 5 7])
          result (mx/searchsorted sorted values)]
      (is (assert-array-equal [0 0 1] result) "single element sorted"))

    (let [sorted (mx/array [1 3 5])
          values (mx/array (clj->js []))
          result (mx/searchsorted sorted values)]
      (mx/eval! result)
      (is (= [0] (mx/shape result)) "empty values -> empty result"))))

;; =========================================================================
;; Section 4: Float32 precision
;; =========================================================================

(deftest searchsorted-float32
  (testing "float32 precision"
    (let [sorted (mx/array [0.1 0.3 0.5 0.7 0.9])
          values (mx/array [0.2 0.4 0.6 0.8])
          result (mx/searchsorted sorted values)]
      (is (assert-array-equal [1 2 3 4] result) "float32 basic"))

    (let [sorted (mx/array [0.0 0.2 0.5 0.8 1.0])
          values (mx/array [0.1 0.3 0.6 0.9])
          result (mx/searchsorted sorted values)]
      (is (assert-array-equal [1 2 3 4] result) "CDF-like array"))))

;; =========================================================================
;; Section 5: Shape preservation
;; =========================================================================

(deftest searchsorted-shape
  (testing "shape preservation"
    (let [sorted (mx/array [1 3 5 7 9])
          values-1d (mx/array [2 4 6])
          result-1d (mx/searchsorted sorted values-1d)]
      (mx/eval! result-1d)
      (is (= [3] (vec (mx/shape result-1d))) "1D shape preserved"))

    (let [sorted (mx/array [1 3 5 7 9])
          values-2d (mx/reshape (mx/array [2 4 6 8]) [2 2])
          result-2d (mx/searchsorted sorted values-2d)]
      (mx/eval! result-2d)
      (is (= [2 2] (vec (mx/shape result-2d))) "2D shape preserved"))

    (let [sorted (mx/array [1.0 3.0 5.0])
          values (mx/array [2.0])
          result (mx/searchsorted sorted values)]
      (mx/eval! result)
      (is (= mx/int32 (mx/dtype result)) "output dtype is int32"))))

;; =========================================================================
;; Section 6: Large N correctness
;; =========================================================================

(deftest searchsorted-large-n
  (testing "large N (10000)"
    (let [n 10000
          sorted (mx/arange 0 n 1)
          values (mx/add (mx/arange 0 10 1) (mx/scalar 0.5))
          result (mx/searchsorted sorted values)]
      (mx/eval! result)
      (is (assert-array-equal [1 2 3 4 5 6 7 8 9 10] result) "large N: search in [0..9999] for [0.5..9.5]"))

    (let [n 10000
          raw (mx/divide (mx/arange 1 (inc n) 1) (mx/scalar (float n)))
          cdf raw
          thresholds (mx/divide (mx/arange 0 10 1) (mx/scalar 10.0))
          result (mx/searchsorted cdf thresholds)]
      (mx/eval! result)
      (let [result-vec (mx/->clj result)]
        (is (every? #(and (>= % 0) (<= % n)) result-vec) "large N CDF: all indices in range")
        (is (apply <= result-vec) "large N CDF: indices monotonically non-decreasing")))))

;; =========================================================================
;; Section 7: O(N) systematic resampling correctness
;; =========================================================================

(deftest systematic-resampling
  (testing "uniform weights"
    (let [n 100
          log-w (mx/zeros [n])
          key (rng/fresh-key 42)
          indices (vec/systematic-resample-indices log-w n key)]
      (mx/eval! indices)
      (is (= [n] (vec (mx/shape indices))) "uniform weights: shape")
      (is (= mx/int32 (mx/dtype indices)) "uniform weights: dtype")
      (let [idx-vec (mx/->clj indices)]
        (is (= (set (range n)) (set idx-vec)) "uniform weights: each particle resampled once")
        (is (every? #(and (>= % 0) (< % n)) idx-vec) "uniform weights: all indices in range"))))

  (testing "degenerate weights"
    (let [n 50
          log-w (mx/array (into [] (concat [-1000] (repeat (dec n) -1e6))))
          key (rng/fresh-key 99)
          indices (vec/systematic-resample-indices log-w n key)]
      (mx/eval! indices)
      (let [idx-vec (mx/->clj indices)]
        (is (every? zero? idx-vec) "degenerate weights: all resample to particle 0"))))

  (testing "skewed weights"
    (let [n 100
          log-w (mx/array (into [] (concat (repeat 5 0.0) (repeat 95 -100.0))))
          key (rng/fresh-key 77)
          indices (vec/systematic-resample-indices log-w n key)]
      (mx/eval! indices)
      (let [idx-vec (mx/->clj indices)]
        (is (every? #(< % 5) idx-vec) "skewed weights: all resampled from high-weight particles")))))

;; =========================================================================
;; Section 8: Resampling statistical properties
;; =========================================================================

(deftest resampling-statistics
  (testing "resampling preserves expected proportions"
    (let [n 1000
          log-w (mx/array [(js/Math.log 0.5) (js/Math.log 0.3) (js/Math.log 0.2)])
          key (rng/fresh-key 123)
          indices (vec/systematic-resample-indices log-w n key)]
      (mx/eval! indices)
      (let [idx-vec (mx/->clj indices)
            counts (frequencies idx-vec)
            prop-0 (/ (get counts 0 0) n)
            prop-1 (/ (get counts 1 0) n)
            prop-2 (/ (get counts 2 0) n)]
        (is (h/close? 0.5 prop-0 0.01) "resampling proportion for p=0.5")
        (is (h/close? 0.3 prop-1 0.01) "resampling proportion for p=0.3")
        (is (h/close? 0.2 prop-2 0.01) "resampling proportion for p=0.2")))))

;; =========================================================================
;; Section 9: Deterministic resampling variant
;; =========================================================================

(deftest deterministic-resampling
  (testing "same u0 -> same indices"
    (let [n 50
          log-w (mx/array (into [] (map #(js/Math.log (inc %)) (range n))))
          u0 (mx/array [0.3])
          idx1 (vec/systematic-resample-indices-deterministic log-w n u0)
          idx2 (vec/systematic-resample-indices-deterministic log-w n u0)]
      (mx/eval! idx1)
      (mx/eval! idx2)
      (is (= (mx/->clj idx1) (mx/->clj idx2)) "deterministic: same u0 -> same result")))

  (testing "deterministic shape and dtype"
    (let [n 100
          log-w (mx/zeros [n])
          u0 (mx/array [0.5])
          det-indices (vec/systematic-resample-indices-deterministic log-w n u0)]
      (mx/eval! det-indices)
      (is (= [n] (vec (mx/shape det-indices))) "deterministic: shape")
      (is (= mx/int32 (mx/dtype det-indices)) "deterministic: dtype"))))

;; =========================================================================
;; Section 10: Resampling O(N) reference comparison
;; =========================================================================

(defn reference-systematic-resample
  "O(N^2) reference implementation using broadcasting."
  [log-weights n u0-val]
  (let [log-probs (mx/subtract log-weights (mx/logsumexp log-weights))
        probs (mx/exp log-probs)
        cdf (mx/cumsum probs)
        thresholds (mx/add (mx/scalar (/ u0-val n))
                           (mx/divide (mx/arange 0 n 1) (mx/scalar (float n))))
        cdf-2d (mx/reshape cdf [1 -1])
        thr-2d (mx/reshape thresholds [-1 1])
        mask (mx/greater-equal cdf-2d thr-2d)
        indices (mx/argmax mask 1)]
    (.astype indices mx/int32)))

(deftest on-vs-reference
  (testing "O(N) matches O(N^2) reference"
    (let [n 50
          log-w (mx/array (into [] (map #(- (* 2.0 (js/Math.sin (* % 0.3))) 1.0) (range n))))
          u0-val 0.37
          u0 (mx/array [u0-val])
          fast-indices (vec/systematic-resample-indices-deterministic log-w n u0)
          ref-indices (reference-systematic-resample log-w n u0-val)]
      (mx/eval! fast-indices)
      (mx/eval! ref-indices)
      (is (= (mx/->clj ref-indices) (mx/->clj fast-indices)) "O(N) matches O(N^2) reference")))

  (testing "uniform weights"
    (let [n 100
          log-w (mx/zeros [n])
          u0-val 0.5
          u0 (mx/array [u0-val])
          fast-indices (vec/systematic-resample-indices-deterministic log-w n u0)
          ref-indices (reference-systematic-resample log-w n u0-val)]
      (mx/eval! fast-indices)
      (mx/eval! ref-indices)
      (is (= (mx/->clj ref-indices) (mx/->clj fast-indices)) "O(N) matches O(N^2) for uniform weights")))

  (testing "skewed weights"
    (let [n 20
          log-w (mx/array (into [] (concat [0.0 0.0 0.0] (repeat 17 -50.0))))
          u0-val 0.1
          u0 (mx/array [u0-val])
          fast-indices (vec/systematic-resample-indices-deterministic log-w n u0)
          ref-indices (reference-systematic-resample log-w n u0-val)]
      (mx/eval! fast-indices)
      (mx/eval! ref-indices)
      (is (= (mx/->clj ref-indices) (mx/->clj fast-indices)) "O(N) matches O(N^2) for skewed weights"))))

(cljs.test/run-tests)
