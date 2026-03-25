(ns genmlx.vupdate-test
  "Tests for batched update (vupdate)."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.vectorized :as vec])
  (:require-macros [genmlx.gen :refer [gen]]))

;; Model: x ~ N(0, 1), y ~ N(x, 0.5)
(def model
  (gen []
    (let [x (trace :x (dist/gaussian 0 1))]
      (trace :y (dist/gaussian x 0.5))
      x)))

(def n 50)

(deftest vupdate-shape-correctness
  (testing "shape correctness"
    (let [key (rng/fresh-key)
          [k1 k2] (rng/split key)
          obs1 (cm/choicemap :y (mx/scalar 2.0))
          vtrace (dyn/vgenerate model [] obs1 n k1)
          obs2 (cm/choicemap :y (mx/scalar 3.0))
          {:keys [vtrace weight]} (dyn/vupdate model vtrace obs2 k2)]
      (let [x-val (cm/get-value (cm/get-submap (:choices vtrace) :x))]
        (mx/eval! x-val)
        (is (= [n] (mx/shape x-val)) "vupdate :x shape"))
      (let [score (:score vtrace)]
        (mx/eval! score)
        (is (= [n] (mx/shape score)) "vupdate score shape"))
      (mx/eval! weight)
      (is (= [n] (mx/shape weight)) "vupdate weight shape"))))

(deftest vupdate-observation-updated
  (testing "observation updated correctly"
    (let [key (rng/fresh-key)
          [k1 k2] (rng/split key)
          obs1 (cm/choicemap :y (mx/scalar 2.0))
          vtrace (dyn/vgenerate model [] obs1 n k1)
          obs2 (cm/choicemap :y (mx/scalar 3.0))
          {:keys [vtrace]} (dyn/vupdate model vtrace obs2 k2)
          y-val (cm/get-value (cm/get-submap (:choices vtrace) :y))]
      (mx/eval! y-val)
      (is (h/close? 3.0 (mx/realize y-val) 1e-6) "vupdate: y is now 3.0"))))

(deftest vupdate-unconstrained-preserved
  (testing ":x unchanged"
    (let [key (rng/fresh-key)
          [k1 k2] (rng/split key)
          obs1 (cm/choicemap :y (mx/scalar 2.0))
          vtrace-before (dyn/vgenerate model [] obs1 n k1)
          x-before (cm/get-value (cm/get-submap (:choices vtrace-before) :x))
          _ (mx/eval! x-before)
          x-before-mean (mx/realize (mx/mean x-before))
          obs2 (cm/choicemap :y (mx/scalar 3.0))
          {:keys [vtrace]} (dyn/vupdate model vtrace-before obs2 k2)
          x-after (cm/get-value (cm/get-submap (:choices vtrace) :x))
          _ (mx/eval! x-after)
          x-after-mean (mx/realize (mx/mean x-after))]
      (is (h/close? x-before-mean x-after-mean 1e-6) "vupdate: :x unchanged"))))

(deftest vupdate-weights-finite
  (testing "weights finite"
    (let [key (rng/fresh-key)
          [k1 k2] (rng/split key)
          obs1 (cm/choicemap :y (mx/scalar 2.0))
          vtrace (dyn/vgenerate model [] obs1 n k1)
          obs2 (cm/choicemap :y (mx/scalar 3.0))
          {:keys [weight]} (dyn/vupdate model vtrace obs2 k2)]
      (mx/eval! weight)
      (let [w-min (mx/realize (mx/amin weight))
            w-max (mx/realize (mx/amax weight))]
        (is (and (js/isFinite w-min) (js/isFinite w-max)) "vupdate: all weights finite")))))

(deftest vupdate-statistical-equivalence
  (testing "statistical equivalence with sequential update"
    (let [n-test 30
          obs1 (cm/choicemap :y (mx/scalar 2.0))
          obs2 (cm/choicemap :y (mx/scalar 3.0))
          seq-weights (mapv (fn [_]
                              (let [{:keys [trace]} (p/generate (dyn/auto-key model) [] obs1)
                                    {:keys [weight]} (p/update (dyn/auto-key model) trace obs2)]
                                (mx/realize weight)))
                            (range n-test))
          seq-mean-w (/ (reduce + seq-weights) n-test)
          key (rng/fresh-key)
          [k1 k2] (rng/split key)
          vtrace (dyn/vgenerate model [] obs1 n-test k1)
          {:keys [weight]} (dyn/vupdate model vtrace obs2 k2)
          _ (mx/eval! weight)
          batch-mean-w (mx/realize (mx/mean weight))]
      (is (js/isFinite seq-mean-w) "sequential mean weight finite")
      (is (js/isFinite batch-mean-w) "batched mean weight finite")
      (is (h/close? seq-mean-w batch-mean-w 10.0) "mean weight similar"))))

(cljs.test/run-tests)
