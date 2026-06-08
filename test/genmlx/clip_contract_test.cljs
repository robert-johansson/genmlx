;; @tier fast core
(ns genmlx.clip-contract-test
  "Regression guard for the mx/clip bounds contract (bean genmlx-aonv).

   mx/clip accepts lo/hi as Option<Either<&MxArray,f64>> at the NAPI boundary:
   MLX float-scalar / float-array / int32-array bounds, plain JS numbers, mixed
   (MxArray + number), and nil (one-sided / unbounded). The mlx-node MLX v0.31.2
   bump briefly narrowed this to f64-only, which broke examples/memo/mdp.cljs
   (int32 array bounds) and differentiable_resample (scalar MxArray bounds).
   Restored in mlx-node d2aee0c. These tests lock the full contract so it can
   never silently regress again."
  (:require [cljs.test :refer [deftest is]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]))

(def xf (mx/array #js [-2.0 0.5 3.0]))
(def xi (mx/array #js [-5 5 9] mx/int32))

(deftest clip-float-scalar-bounds
  (is (h/all-close? [0.0 0.5 1.0]
                    (h/realize-vec (mx/clip xf (mx/scalar 0.0) (mx/scalar 1.0))))))

(deftest clip-float-array-bounds
  (is (h/all-close? [-1.0 0.5 2.0]
                    (h/realize-vec (mx/clip xf
                                            (mx/array #js [-1.0 0.0 0.0])
                                            (mx/array #js [0.0 1.0 2.0]))))))

(deftest clip-int32-array-bounds  ;; the original memo/mdp.cljs regression case
  (is (h/all-close? [0.0 3.0 3.0]
                    (h/realize-vec (mx/clip xi
                                            (mx/array #js [0 0 0] mx/int32)
                                            (mx/array #js [3 3 3] mx/int32))))))

(deftest clip-js-number-bounds
  (is (h/all-close? [0.0 0.5 1.0]
                    (h/realize-vec (mx/clip xf 0.0 1.0)))))

(deftest clip-mixed-bounds  ;; MxArray lo + JS-number hi
  (is (h/all-close? [0.0 0.5 1.0]
                    (h/realize-vec (mx/clip xf (mx/scalar 0.0) 1.0)))))

(deftest clip-one-sided-bounds  ;; nil = unbounded on that side
  (is (h/all-close? [0.0 0.5 3.0]
                    (h/realize-vec (mx/clip xf (mx/scalar 0.0) nil)))
      "lo only (hi unbounded)")
  (is (h/all-close? [-2.0 0.5 1.0]
                    (h/realize-vec (mx/clip xf nil (mx/scalar 1.0))))
      "hi only (lo unbounded)"))

(cljs.test/run-tests)
