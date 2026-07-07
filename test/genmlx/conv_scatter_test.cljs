;; @tier medium
(ns genmlx.conv-scatter-test
  "Tests for the genmlx-lgbx membrane additions: conv2d, put-along-axis,
   scatter-add, and the bincount helper. Includes the motivating use case —
   a best-shift-shaped offset search as one batched cross-correlation."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]))

;; -------------------------------------------------------------------------
;; Scatter family
;; -------------------------------------------------------------------------

(deftest put-along-axis-test
  (testing "overwrite semantics at unique indices"
    (let [a   (mx/zeros [5] mx/int32)
          idx (mx/array [1 3] mx/int32)
          v   (mx/array [7 9] mx/int32)]
      (is (= [0 7 0 9 0] (mx/->clj (mx/put-along-axis a idx v 0)))
          "values placed, rest untouched")))
  (testing "2-D along axis 1"
    (let [a   (mx/zeros [2 3] mx/int32)
          idx (mx/array [[0] [2]] mx/int32)
          v   (mx/array [[5] [6]] mx/int32)]
      (is (= [[5 0 0] [0 0 6]] (mx/->clj (mx/put-along-axis a idx v 1)))
          "row-wise placement"))))

(deftest scatter-add-test
  (testing "duplicate indices ACCUMULATE (the put-along-axis contrast)"
    (let [a   (mx/zeros [5] mx/int32)
          idx (mx/array [1 1 3 1] mx/int32)
          v   (mx/ones [4] mx/int32)]
      (is (= [0 3 0 1 0] (mx/->clj (mx/scatter-add a idx v 0)))
          "three writes to slot 1 sum to 3")))
  (testing "accumulates onto existing content"
    (let [a   (mx/array [10 20 30] mx/int32)
          idx (mx/array [2 2] mx/int32)
          v   (mx/array [1 2] mx/int32)]
      (is (= [10 20 33] (mx/->clj (mx/scatter-add a idx v 0)))
          "adds into non-zero base"))))

(deftest bincount-test
  (testing "histogram of ids"
    (is (= [2 0 3 1] (mx/->clj (mx/bincount (mx/array [0 2 2 3 0 2] mx/int32) 4)))
        "counts per id, absent ids zero"))
  (testing "nested/2-D ids flatten first"
    (is (= [1 2 1] (mx/->clj (mx/bincount (mx/array [[0 1] [1 2]] mx/int32) 3)))
        "shape-agnostic")))

;; -------------------------------------------------------------------------
;; conv2d
;; -------------------------------------------------------------------------

(deftest conv2d-basic-test
  (testing "3x3 input, 2x2 ones kernel = sliding window sums"
    (let [inp (mx/reshape (mx/arange 9) [1 3 3 1])       ;; 0..8
          w   (mx/ones [1 2 2 1])
          out (mx/conv2d inp w)]
      (is (= [1 2 2 1] (h/realize-shape out)) "valid conv output shape")
      ;; windows: [0 1 3 4]=8 [1 2 4 5]=12 [3 4 6 7]=20 [4 5 7 8]=24
      (is (= [[[[8.0] [12.0]] [[20.0] [24.0]]]] (mx/->clj out))
          "window sums")))
  (testing "stride and padding options"
    (let [inp (mx/ones [1 4 4 1])
          w   (mx/ones [1 2 2 1])]
      (is (= [1 2 2 1] (h/realize-shape (mx/conv2d inp w {:stride 2})))
          "stride 2 halves the grid")
      (is (= [1 5 5 1] (h/realize-shape (mx/conv2d inp w {:padding 1})))
          "padding grows the output"))))

(deftest conv2d-correlation-smoke-test
  (testing "best-shift as one correlation: recover a planted offset"
    ;; A sparse 16x16 pattern, planted into a 33x33 field at offset (9, 4).
    ;; Cross-correlating field (input) with pattern (kernel) peaks exactly
    ;; at the planted offset — conv2d does NOT flip the kernel, so the
    ;; argmax of the correlation map IS the shift. This is the arc3
    ;; best-shift 33x33 offset search as a single batched op.
    (let [ph 16 pw 16 fh 33 fw 33
          off-r 9 off-c 4
          ;; deterministic sparse pattern: 1 where (3i+5j) mod 7 == 0
          pat  (mapv (fn [i] (mapv (fn [j] (if (zero? (mod (+ (* 3 i) (* 5 j)) 7)) 1 0))
                                   (range pw)))
                     (range ph))
          field (vec (for [r (range fh)]
                       (vec (for [c (range fw)]
                              (let [i (- r off-r) j (- c off-c)]
                                (if (and (<= 0 i) (< i ph) (<= 0 j) (< j pw))
                                  (nth (nth pat i) j)
                                  0))))))
          inp  (mx/reshape (mx/array field) [1 fh fw 1])
          k    (mx/reshape (mx/array pat) [1 ph pw 1])
          corr (mx/conv2d inp k)                          ;; [1 18 18 1]
          flat (mx/reshape corr [(* (- fh ph -1) (- fw pw -1))])
          best (h/realize (mx/argmax flat))
          w'   (- fw pw -1)]
      (is (= [off-r off-c] [(quot best w') (mod best w')])
          "correlation argmax recovers the planted shift"))))

(cljs.test/run-tests)
