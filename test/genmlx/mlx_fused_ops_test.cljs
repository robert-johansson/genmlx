;; @tier fast
(ns genmlx.mlx-fused-ops-test
  "Model-free unit tests for the f6ov P0 fused NN primitives (genmlx.rs fast::
   ops): rms-norm, silu, scaled-dot-product-attention, rope. Each is checked
   against a manual MLX reference computation (independent oracle). No model
   load — just the array ops."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.mlx :as mx]))

(defn- max-abs-diff [a b]
  (mx/eval! a) (mx/eval! b)
  (let [fa (.toFloat32 a) fb (.toFloat32 b) n (.-length fa)]
    (loop [i 0 m 0.0]
      (if (< i n) (recur (inc i) (max m (js/Math.abs (- (aget fa i) (aget fb i))))) m))))

(deftest silu-matches-x-sigmoid
  (let [x (mx/array [0.0 1.0 -1.0 2.5 -3.0 0.25])
        ref (mx/multiply x (mx/sigmoid x))]
    (is (< (max-abs-diff (mx/silu x) ref) 1e-5) "silu(x) == x*sigmoid(x)")))

(deftest rms-norm-matches-reference
  (testing "x * rsqrt(mean(x^2)+eps) * weight over the last axis"
    (let [x (mx/array [1.0 2.0 3.0 4.0])
          w (mx/array [1.0 0.5 2.0 1.0])
          eps 1e-6
          ms (mx/mean (mx/multiply x x))
          inv (mx/divide (mx/scalar 1.0) (mx/sqrt (mx/add ms (mx/scalar eps))))
          ref (mx/multiply (mx/multiply x inv) w)]
      (is (< (max-abs-diff (mx/rms-norm x w eps) ref) 1e-4)))))

(deftest sdpa-matches-manual-attention
  (testing "softmax(q k^T * scale) v with no mask, [b=1 h=1 s=2 d=2]"
    (let [q (mx/array [1.0 0.0 0.5 1.0] [1 1 2 2])
          k (mx/array [1.0 0.2 0.0 1.0] [1 1 2 2])
          v (mx/array [1.0 2.0 3.0 4.0] [1 1 2 2])
          scale (/ 1.0 (js/Math.sqrt 2.0))
          scores (mx/multiply (mx/matmul q (mx/transpose k [0 1 3 2])) (mx/scalar scale))
          w (mx/softmax scores -1)
          ref (mx/matmul w v)]
      (is (< (max-abs-diff (mx/scaled-dot-product-attention q k v scale) ref) 1e-3)
          "fused SDPA == manual softmax-attention"))))

(deftest rope-rotation-properties
  (testing "rope is a per-position rotation: shape + norm preserved, offset matters"
    (let [x  (mx/array (mapv double (range 1 13)) [1 1 3 4])
          y0 (mx/rope x 4 false 10000.0 1.0 0)
          y5 (mx/rope x 4 false 10000.0 1.0 5)]
      (is (= [1 1 3 4] (vec (mx/shape y0))) "shape preserved")
      (is (< (js/Math.abs (- (mx/item (mx/norm x)) (mx/item (mx/norm y0)))) 1e-3)
          "Frobenius norm preserved (rotation)")
      (is (> (max-abs-diff x y0) 1e-3) "rope changes the input")
      (is (> (max-abs-diff y0 y5) 1e-3) "offset changes the output"))))

(cljs.test/run-tests)
