(ns genmlx.searchsorted-test
  "Tests for mx/searchsorted and O(N) systematic resampling.

   Validates:
   1. searchsorted basic correctness (left/right side)
   2. Edge cases (empty, boundary, duplicates)
   3. Shape preservation
   4. Float32 precision
   5. Large N correctness
   6. O(N) resampling matches O(N^2) reference
   7. Resampling statistical properties
   8. Deterministic resampling variant

   Run: bun run --bun nbb test/genmlx/searchsorted_test.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.vectorized :as vec]))

;; ---------------------------------------------------------------------------
;; Test helpers
;; ---------------------------------------------------------------------------

(def ^:private pass-count (atom 0))
(def ^:private fail-count (atom 0))

(defn- assert-true [desc pred]
  (if pred
    (do (swap! pass-count inc)
        (println (str "  PASS: " desc)))
    (do (swap! fail-count inc)
        (println (str "  FAIL: " desc)))))

(defn- assert-equal [desc expected actual]
  (if (= expected actual)
    (do (swap! pass-count inc)
        (println (str "  PASS: " desc)))
    (do (swap! fail-count inc)
        (println (str "  FAIL: " desc " (expected=" expected " actual=" actual ")")))))

(defn- assert-close [desc expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (do (swap! pass-count inc)
          (println (str "  PASS: " desc " (expected=" (.toFixed expected 6)
                       " actual=" (.toFixed actual 6) " diff=" (.toFixed diff 8) ")")))
      (do (swap! fail-count inc)
          (println (str "  FAIL: " desc " (expected=" (.toFixed expected 6)
                       " actual=" (.toFixed actual 6) " diff=" (.toFixed diff 8) ")"))))))

(defn- assert-array-equal [desc expected-vec actual-arr]
  (mx/eval! actual-arr)
  (let [actual-vec (mx/->clj actual-arr)]
    (if (= expected-vec actual-vec)
      (do (swap! pass-count inc)
          (println (str "  PASS: " desc)))
      (do (swap! fail-count inc)
          (println (str "  FAIL: " desc " (expected=" expected-vec " actual=" actual-vec ")"))))))

(println "\n===== searchsorted + O(N) Resampling Tests =====\n")

;; =========================================================================
;; Section 1: searchsorted basic correctness
;; =========================================================================

(println "\n-- 1. Basic searchsorted (left side) --")

(let [sorted (mx/array [1 3 5 7])
      values (mx/array [2 4 6])
      result (mx/searchsorted sorted values)]
  (assert-array-equal "basic: [1,3,5,7] search [2,4,6] -> [1,2,3]"
    [1 2 3] result))

(let [sorted (mx/array [1 3 5 7])
      values (mx/array [0 1 3 5 7 8])
      result (mx/searchsorted sorted values)]
  (assert-array-equal "boundary: values at and beyond edges"
    [0 0 1 2 3 4] result))

(let [sorted (mx/array [1 3 3 5])
      values (mx/array [3])
      result (mx/searchsorted sorted values)]
  (assert-array-equal "left side: duplicates -> first valid index"
    [1] result))

;; =========================================================================
;; Section 2: searchsorted right side
;; =========================================================================

(println "\n-- 2. Right side --")

(let [sorted (mx/array [1 3 3 5])
      values (mx/array [3])
      result (mx/searchsorted sorted values :right)]
  (assert-array-equal "right side: duplicates -> after last"
    [3] result))

(let [sorted (mx/array [1 3 5 7])
      values (mx/array [0 1 7 8])
      result-l (mx/searchsorted sorted values :left)
      result-r (mx/searchsorted sorted values :right)]
  (assert-array-equal "left: boundary values" [0 0 3 4] result-l)
  (assert-array-equal "right: boundary values" [0 1 4 4] result-r))

;; =========================================================================
;; Section 3: Edge cases
;; =========================================================================

(println "\n-- 3. Edge cases --")

;; Value less than all elements -> index 0
(let [sorted (mx/array [10 20 30])
      values (mx/array [5])
      result (mx/searchsorted sorted values)]
  (assert-array-equal "value < all -> 0" [0] result))

;; Value greater than all elements -> N
(let [sorted (mx/array [10 20 30])
      values (mx/array [35])
      result (mx/searchsorted sorted values)]
  (assert-array-equal "value > all -> N" [3] result))

;; Single element sorted array
(let [sorted (mx/array [5])
      values (mx/array [3 5 7])
      result (mx/searchsorted sorted values)]
  (assert-array-equal "single element sorted" [0 0 1] result))

;; Empty values array
(let [sorted (mx/array [1 3 5])
      values (mx/array (clj->js []))
      result (mx/searchsorted sorted values)]
  (mx/eval! result)
  (assert-equal "empty values -> empty result" [0] (mx/shape result)))

;; =========================================================================
;; Section 4: Float32 precision
;; =========================================================================

(println "\n-- 4. Float32 precision --")

(let [sorted (mx/array [0.1 0.3 0.5 0.7 0.9])
      values (mx/array [0.2 0.4 0.6 0.8])
      result (mx/searchsorted sorted values)]
  (assert-array-equal "float32 basic" [1 2 3 4] result))

;; CDF-like sorted array (monotonically increasing from 0 to 1)
(let [sorted (mx/array [0.0 0.2 0.5 0.8 1.0])
      values (mx/array [0.1 0.3 0.6 0.9])
      result (mx/searchsorted sorted values)]
  (assert-array-equal "CDF-like array" [1 2 3 4] result))

;; =========================================================================
;; Section 5: Shape preservation
;; =========================================================================

(println "\n-- 5. Shape preservation --")

(let [sorted (mx/array [1 3 5 7 9])
      values-1d (mx/array [2 4 6])
      result-1d (mx/searchsorted sorted values-1d)]
  (mx/eval! result-1d)
  (assert-equal "1D shape preserved" [3] (vec (mx/shape result-1d))))

(let [sorted (mx/array [1 3 5 7 9])
      values-2d (mx/reshape (mx/array [2 4 6 8]) [2 2])
      result-2d (mx/searchsorted sorted values-2d)]
  (mx/eval! result-2d)
  (assert-equal "2D shape preserved" [2 2] (vec (mx/shape result-2d))))

;; Output dtype is int32
(let [sorted (mx/array [1.0 3.0 5.0])
      values (mx/array [2.0])
      result (mx/searchsorted sorted values)]
  (mx/eval! result)
  (assert-equal "output dtype is int32" mx/int32 (mx/dtype result)))

;; =========================================================================
;; Section 6: Large N correctness
;; =========================================================================

(println "\n-- 6. Large N (10000) --")

(let [n 10000
      ;; Create sorted array: 0, 1, 2, ..., n-1
      sorted (mx/arange 0 n 1)
      ;; Search for values at 0.5, 1.5, ..., 9.5
      values (mx/add (mx/arange 0 10 1) (mx/scalar 0.5))
      result (mx/searchsorted sorted values)]
  (mx/eval! result)
  (assert-array-equal "large N: search in [0..9999] for [0.5..9.5]"
    [1 2 3 4 5 6 7 8 9 10] result))

;; Large N with CDF-like distribution (cumsum of uniform)
(let [n 10000
      raw (mx/divide (mx/arange 1 (inc n) 1) (mx/scalar (float n)))
      cdf raw ;; Already cumulative: [1/N, 2/N, ..., 1.0]
      ;; Search for evenly spaced thresholds
      thresholds (mx/divide (mx/arange 0 10 1) (mx/scalar 10.0))
      result (mx/searchsorted cdf thresholds)]
  (mx/eval! result)
  (let [result-vec (mx/->clj result)]
    (assert-true "large N CDF: all indices in range"
      (every? #(and (>= % 0) (<= % n)) result-vec))
    (assert-true "large N CDF: indices monotonically non-decreasing"
      (apply <= result-vec))))

;; =========================================================================
;; Section 7: O(N) systematic resampling correctness
;; =========================================================================

(println "\n-- 7. O(N) systematic resampling --")

;; Uniform weights -> all particles equally likely
(let [n 100
      log-w (mx/zeros [n])  ;; uniform weights
      key (rng/fresh-key 42)
      indices (vec/systematic-resample-indices log-w n key)]
  (mx/eval! indices)
  (assert-equal "uniform weights: shape" [n] (vec (mx/shape indices)))
  (assert-equal "uniform weights: dtype" mx/int32 (mx/dtype indices))
  (let [idx-vec (mx/->clj indices)]
    ;; With uniform weights, systematic resampling should produce each index once
    (assert-equal "uniform weights: each particle resampled once"
      (set (range n)) (set idx-vec))
    (assert-true "uniform weights: all indices in range"
      (every? #(and (>= % 0) (< % n)) idx-vec))))

;; Degenerate weights: one particle has all weight
(let [n 50
      log-w (mx/array (into [] (concat [-1000] (repeat (dec n) -1e6))))
      key (rng/fresh-key 99)
      indices (vec/systematic-resample-indices log-w n key)]
  (mx/eval! indices)
  (let [idx-vec (mx/->clj indices)]
    (assert-true "degenerate weights: all resample to particle 0"
      (every? zero? idx-vec))))

;; High-weight particles should be resampled more
(let [n 100
      ;; Particles 0-4 have high weight, rest have low weight
      log-w (mx/array (into [] (concat (repeat 5 0.0) (repeat 95 -100.0))))
      key (rng/fresh-key 77)
      indices (vec/systematic-resample-indices log-w n key)]
  (mx/eval! indices)
  (let [idx-vec (mx/->clj indices)]
    (assert-true "skewed weights: all resampled from high-weight particles"
      (every? #(< % 5) idx-vec))))

;; =========================================================================
;; Section 8: Resampling statistical properties
;; =========================================================================

(println "\n-- 8. Resampling statistics --")

;; Verify resampling preserves expected proportions
(let [n 1000
      ;; 3 particles with weights proportional to [0.5, 0.3, 0.2]
      log-w (mx/array [(js/Math.log 0.5) (js/Math.log 0.3) (js/Math.log 0.2)])
      key (rng/fresh-key 123)
      indices (vec/systematic-resample-indices log-w n key)]
  (mx/eval! indices)
  (let [idx-vec (mx/->clj indices)
        counts (frequencies idx-vec)
        prop-0 (/ (get counts 0 0) n)
        prop-1 (/ (get counts 1 0) n)
        prop-2 (/ (get counts 2 0) n)]
    (assert-close "resampling proportion for p=0.5" 0.5 prop-0 0.01)
    (assert-close "resampling proportion for p=0.3" 0.3 prop-1 0.01)
    (assert-close "resampling proportion for p=0.2" 0.2 prop-2 0.01)))

;; =========================================================================
;; Section 9: Deterministic resampling variant
;; =========================================================================

(println "\n-- 9. Deterministic resampling --")

;; Same u0 -> same indices
(let [n 50
      log-w (mx/array (into [] (map #(js/Math.log (inc %)) (range n))))
      u0 (mx/array [0.3])
      idx1 (vec/systematic-resample-indices-deterministic log-w n u0)
      idx2 (vec/systematic-resample-indices-deterministic log-w n u0)]
  (mx/eval! idx1)
  (mx/eval! idx2)
  (assert-equal "deterministic: same u0 -> same result"
    (mx/->clj idx1) (mx/->clj idx2)))

;; Deterministic matches stochastic for same u0
(let [n 100
      log-w (mx/zeros [n])
      u0 (mx/array [0.5])
      det-indices (vec/systematic-resample-indices-deterministic log-w n u0)]
  (mx/eval! det-indices)
  (assert-equal "deterministic: shape" [n] (vec (mx/shape det-indices)))
  (assert-equal "deterministic: dtype" mx/int32 (mx/dtype det-indices)))

;; =========================================================================
;; Section 10: Resampling O(N) reference comparison
;; =========================================================================

(println "\n-- 10. O(N) vs reference comparison --")

;; Build O(N^2) reference resampling for comparison
(defn reference-systematic-resample
  "O(N^2) reference implementation using broadcasting.
   Computes the same result as the O(N) searchsorted version."
  [log-weights n u0-val]
  (let [log-probs (mx/subtract log-weights (mx/logsumexp log-weights))
        probs (mx/exp log-probs)
        cdf (mx/cumsum probs)
        thresholds (mx/add (mx/scalar (/ u0-val n))
                           (mx/divide (mx/arange 0 n 1) (mx/scalar (float n))))
        ;; O(N^2) approach: broadcast compare cdf[j] >= thresholds[i]
        ;; Each threshold finds the first cdf index that exceeds it
        cdf-2d (mx/reshape cdf [1 -1])       ;; [1, N_weights]
        thr-2d (mx/reshape thresholds [-1 1]) ;; [n, 1]
        ;; mask[i,j] = 1 if cdf[j] >= thresholds[i]
        mask (mx/greater-equal cdf-2d thr-2d)
        ;; argmax finds the first 1 in each row
        indices (mx/argmax mask 1)]
    (.astype indices mx/int32)))

;; Compare for a non-trivial weight vector
(let [n 50
      ;; Random-ish weights (deterministic)
      log-w (mx/array (into [] (map #(- (* 2.0 (js/Math.sin (* % 0.3))) 1.0) (range n))))
      u0-val 0.37
      u0 (mx/array [u0-val])
      ;; O(N) version
      fast-indices (vec/systematic-resample-indices-deterministic log-w n u0)
      ;; O(N^2) reference
      ref-indices (reference-systematic-resample log-w n u0-val)]
  (mx/eval! fast-indices)
  (mx/eval! ref-indices)
  (assert-equal "O(N) matches O(N^2) reference"
    (mx/->clj ref-indices) (mx/->clj fast-indices)))

;; Compare for uniform weights
(let [n 100
      log-w (mx/zeros [n])
      u0-val 0.5
      u0 (mx/array [u0-val])
      fast-indices (vec/systematic-resample-indices-deterministic log-w n u0)
      ref-indices (reference-systematic-resample log-w n u0-val)]
  (mx/eval! fast-indices)
  (mx/eval! ref-indices)
  (assert-equal "O(N) matches O(N^2) for uniform weights"
    (mx/->clj ref-indices) (mx/->clj fast-indices)))

;; Compare for skewed weights
(let [n 20
      log-w (mx/array (into [] (concat [0.0 0.0 0.0] (repeat 17 -50.0))))
      u0-val 0.1
      u0 (mx/array [u0-val])
      fast-indices (vec/systematic-resample-indices-deterministic log-w n u0)
      ref-indices (reference-systematic-resample log-w n u0-val)]
  (mx/eval! fast-indices)
  (mx/eval! ref-indices)
  (assert-equal "O(N) matches O(N^2) for skewed weights"
    (mx/->clj ref-indices) (mx/->clj fast-indices)))

;; =========================================================================
;; Results
;; =========================================================================

(println "\n===== Results =====")
(println (str "PASS: " @pass-count " / " (+ @pass-count @fail-count)))
(when (pos? @fail-count)
  (println (str "FAIL: " @fail-count)))
(if (zero? @fail-count)
  (println "===================")
  (do (println "\n  *** TESTS FAILED ***")
      (println "=========================")))
