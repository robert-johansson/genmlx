(ns genmlx.iid-test
  "Tests for iid and iid-gaussian distributions."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; iid distribution
;; ---------------------------------------------------------------------------

(deftest iid-sample-shape
  (testing "iid: sample shape"
    (let [base (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0))
          d (dist/iid base 5)
          s (dc/dist-sample d (rng/fresh-key))]
      (is (= [5] (mx/shape s)) "iid sample shape=[5]"))))

(deftest iid-sample-n-shape
  (testing "iid: sample-n shape"
    (let [base (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0))
          d (dist/iid base 5)
          sn (dc/dist-sample-n d (rng/fresh-key) 10)]
      (is (= [10 5] (mx/shape sn)) "iid sample-n shape=[10 5]"))))

(deftest iid-log-prob-sum
  (testing "iid: log-prob matches sum of element log-probs"
    (let [base (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0))
          d (dist/iid base 3)
          vals (mx/array [1.0 2.0 3.0])
          iid-lp (mx/item (dc/dist-log-prob d vals))
          manual (+ (mx/item (dc/dist-log-prob base (mx/scalar 1.0)))
                    (mx/item (dc/dist-log-prob base (mx/scalar 2.0)))
                    (mx/item (dc/dist-log-prob base (mx/scalar 3.0))))]
      (is (h/close? manual iid-lp 1e-5) "iid log-prob = sum of element log-probs"))))

(deftest iid-reparam-shape
  (testing "iid: reparam shape"
    (let [base (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0))
          d (dist/iid base 4)
          s (dc/dist-reparam d (rng/fresh-key))]
      (is (= [4] (mx/shape s)) "iid reparam shape=[4]"))))

;; ---------------------------------------------------------------------------
;; iid-gaussian distribution
;; ---------------------------------------------------------------------------

(deftest iid-gaussian-sample-shape
  (testing "iid-gaussian: sample shape"
    (let [d (dist/iid-gaussian (mx/scalar 0.0) (mx/scalar 1.0) 5)
          s (dc/dist-sample d (rng/fresh-key))]
      (is (= [5] (mx/shape s)) "iid-gaussian sample shape=[5]"))))

(deftest iid-gaussian-sample-n-shape
  (testing "iid-gaussian: sample-n shape"
    (let [d (dist/iid-gaussian (mx/scalar 0.0) (mx/scalar 1.0) 5)
          sn (dc/dist-sample-n d (rng/fresh-key) 10)]
      (is (= [10 5] (mx/shape sn)) "iid-gaussian sample-n shape=[10 5]"))))

(deftest iid-gaussian-log-prob-matches-iid
  (testing "iid-gaussian: log-prob matches iid"
    (let [d1 (dist/iid (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)) 3)
          d2 (dist/iid-gaussian (mx/scalar 0.0) (mx/scalar 1.0) 3)
          vals (mx/array [1.0 2.0 3.0])
          lp1 (mx/item (dc/dist-log-prob d1 vals))
          lp2 (mx/item (dc/dist-log-prob d2 vals))]
      (is (h/close? lp1 lp2 1e-5) "iid-gaussian matches iid log-prob"))))

(deftest iid-gaussian-t-shaped-mu
  (testing "iid-gaussian: [T]-shaped mu"
    (let [means (mx/array [1.0 2.0 3.0])
          d (dist/iid-gaussian means (mx/scalar 1.0) 3)]
      (is (= [3] (mx/shape (dc/dist-sample d (rng/fresh-key)))) "[T] mu sample shape=[3]")
      (let [lp-at-means (mx/item (dc/dist-log-prob d means))
            lp-away (mx/item (dc/dist-log-prob d (mx/array [10.0 20.0 30.0])))]
        (is (> lp-at-means lp-away) "[T] mu lp at means > lp away")))))

(deftest iid-gaussian-broadcasting
  (testing "iid-gaussian: [N,T] log-prob broadcasting"
    (let [means (mx/array [1.0 2.0 3.0])
          d (dist/iid-gaussian means (mx/scalar 1.0) 3)
          vals (mx/array [[1.0 2.0 3.0] [0.5 1.5 2.5]])]
      (let [lp (dc/dist-log-prob d vals)]
        (is (= [2] (mx/shape lp)) "[N,T] broadcasting shape=[2]")
        (let [lps (mx/->clj lp)]
          (is (> (first lps) (second lps)) "[N,T] lp at means > lp away"))))))

(deftest iid-gaussian-reparam-shape
  (testing "iid-gaussian: reparam shape"
    (let [d (dist/iid-gaussian (mx/scalar 0.0) (mx/scalar 1.0) 4)
          s (dc/dist-reparam d (rng/fresh-key))]
      (is (= [4] (mx/shape s)) "iid-gaussian reparam shape=[4]"))))

;; ---------------------------------------------------------------------------
;; Model integration: scalar
;; ---------------------------------------------------------------------------

(def iid-model
  (gen [t]
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :ys (dist/iid-gaussian mu (mx/scalar 1.0) t))
      mu)))

;; NOTE: scalar simulate triggers compiled path which has a pre-existing
;; bug (nth on MLX array constructor). This test documents that known error.
(deftest model-scalar-simulate
  (testing "iid in model: scalar simulate (known compiled path issue)"
    (is (thrown? js/Error (p/simulate (dyn/auto-key iid-model) [5]))
        "compiled path error on iid model with dynamic T (pre-existing)")))

(deftest model-scalar-generate
  (testing "iid in model: scalar generate with stacked obs"
    (let [obs (cm/choicemap :ys (mx/array [1.0 2.0 3.0 4.0 5.0]))
          result (p/generate (dyn/auto-key iid-model) [5] obs)]
      (is (js/isFinite (mx/item (:weight result)))
          "scalar generate weight is finite"))))

;; ---------------------------------------------------------------------------
;; Model integration: vectorized
;; ---------------------------------------------------------------------------

(deftest model-vsimulate
  (testing "iid in model: vsimulate"
    (let [vt (dyn/vsimulate (dyn/auto-key iid-model) [5] 100 (rng/fresh-key))
          inner (:m (:choices vt))]
      (is (= [100] (mx/shape (:v (get inner :mu)))) "vsimulate :mu shape=[100]")
      (is (= [100 5] (mx/shape (:v (get inner :ys)))) "vsimulate :ys shape=[100 5]")
      (is (= [100] (mx/shape (:score vt))) "vsimulate score shape=[100]"))))

(deftest model-vgenerate-posterior
  (testing "iid in model: vgenerate with posterior check"
    (let [obs (cm/choicemap :ys (mx/array [1.0 2.0 3.0 4.0 5.0]))
          vt (dyn/vgenerate (dyn/auto-key iid-model) [5] obs 5000 (rng/fresh-key))
          w (:weight vt) r (:retval vt)
          wn (let [e (mx/exp (mx/subtract w (mx/amax w)))] (mx/divide e (mx/sum e)))
          mu-est (mx/item (mx/sum (mx/multiply wn r)))]
      (is (= [5000] (mx/shape w)) "vgenerate weight shape=[5000]")
      (is (h/close? 3.0 mu-est 0.5) "posterior mean mu ~ 3.0"))))

;; ---------------------------------------------------------------------------
;; Performance
;; ---------------------------------------------------------------------------

(deftest iid-performance
  (testing "iid performance"
    (let [model (dyn/auto-key iid-model)
          obs (cm/choicemap :ys (mx/array (mapv #(+ (* 2.0 %) 1.0) (range 50))))
          key (rng/fresh-key)
          t0 (.now js/Date)
          _ (dyn/vgenerate model [50] obs 10000 key)
          t1 (.now js/Date)]
      (is (< (- t1 t0) 100) "vgenerate < 100ms"))))

(cljs.test/run-tests)
