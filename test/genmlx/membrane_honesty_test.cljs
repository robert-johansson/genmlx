;; @tier fast
(ns genmlx.membrane-honesty-test
  "Regression tests for the genmlx-vd2j membrane honesty bundle:
   eq?/neq? float truncation, meshgrid broadcast-to correctness,
   rng split nil-key guard, hmm-obs zero-lp contract."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist.core :as dc]
            [genmlx.inference.hmm-forward :as hmm]))

(deftest eq-float-operand-test
  (testing "eq?/neq? must not truncate non-integer number operands (genmlx-vd2j)"
    ;; pre-fix: (eq? arr 2.5) promoted 2.5 via int32 scalar -> compared against 2
    (let [arr (mx/array [2.0 2.5 3.0])]
      (is (= [0.0 1.0 0.0] (vec (mx/->clj (mx/eq? arr 2.5))))
          "eq? against 2.5 matches only 2.5")
      (is (= [1.0 0.0 1.0] (vec (mx/->clj (mx/neq? arr 2.5))))
          "neq? against 2.5 is the complement")
      (is (= [1.0 0.0 0.0] (vec (mx/->clj (mx/eq? arr 2))))
          "integer operands still compare exactly"))))

(deftest meshgrid-broadcast-test
  (testing "meshgrid fills broadcast dims correctly (genmlx-vd2j)"
    ;; pre-fix: native broadcastTo mis-fills size-1 source dims -> [v 0 0 ...]
    (let [grids (mx/meshgrid (mx/array [1.0 2.0 3.0]) (mx/array [10.0 20.0]))
          a-col (aget grids 0)
          b-row (aget grids 1)]
      (is (= [3 2] (vec (mx/shape a-col))) "a-col shape")
      (is (= [[1.0 1.0] [2.0 2.0] [3.0 3.0]]
             (mapv vec (mx/->clj a-col)))
          "a-col rows are constant (no zero-fill)")
      (is (= [[10.0 20.0] [10.0 20.0] [10.0 20.0]]
             (mapv vec (mx/->clj b-row)))
          "b-row columns are constant (no zero-fill)"))))

(deftest rng-split-nil-guard-test
  (testing "split/split-n on nil key raise actionable errors (genmlx-vd2j)"
    ;; pre-fix: check-key passed nil through -> raw NAPI error from .randomSplit
    (is (thrown-with-msg? js/Error #"rng/split: key is nil"
                          (rng/split nil))
        "split nil throws with clear message")
    (is (thrown-with-msg? js/Error #"rng/split-n: key is nil"
                          (rng/split-n nil 3))
        "split-n nil throws with clear message")
    (is (= [nil nil] (rng/split-or-nils nil))
        "split-or-nils still supports nil-as-no-entropy")
    (is (= [nil nil nil] (rng/split-n-or-nils nil 3))
        "split-n-or-nils still supports nil-as-no-entropy")
    (let [[k1 k2] (rng/split (rng/fresh-key 42))]
      (is (and (rng/valid-key? k1) (rng/valid-key? k2))
          "split on a real key still works"))))

(deftest hmm-obs-zero-lp-contract-test
  (testing "hmm-obs contributes ZERO under plain handlers — documented contract (genmlx-vd2j)"
    ;; This pins the honest docstring: if hmm-obs ever gains a real plain-handler
    ;; log-prob, update the docstring and this test together.
    (let [d (hmm/hmm-obs (mx/array [-1.0 -2.0]) (mx/scalar 1.0))]
      (is (zero? (mx/item (dc/dist-log-prob d (mx/scalar 0.0))))
          "log-prob is 0.0 under plain handler")
      (is (zero? (mx/item (dc/dist-sample* d (rng/fresh-key 1))))
          "sample is dummy 0.0 under plain handler"))))

(cljs.test/run-tests)
