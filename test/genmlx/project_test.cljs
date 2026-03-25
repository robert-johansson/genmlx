(ns genmlx.project-test
  "Tests for project (IProject protocol)."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.dist.core :as dc]
            [genmlx.combinators :as comb])
  (:require-macros [genmlx.gen :refer [gen]]))

(deftest project-all-equals-score
  (testing "project with sel/all should equal trace score"
    (let [model (dyn/auto-key (gen []
                  (trace :x (dist/gaussian 0 1))
                  (trace :y (dist/gaussian 0 1))
                  nil))
          trace (p/simulate model [])
          weight (p/project model trace sel/all)]
      (mx/eval! weight (:score trace))
      (is (h/close? (mx/item (:score trace)) (mx/item weight) 0.001)
          "project all = trace score"))))

(deftest project-none-equals-zero
  (testing "project with sel/none should return 0"
    (let [model (dyn/auto-key (gen []
                  (trace :x (dist/gaussian 0 1))
                  (trace :y (dist/gaussian 0 1))
                  nil))
          trace (p/simulate model [])
          weight (p/project model trace sel/none)]
      (mx/eval! weight)
      (is (h/close? 0.0 (mx/item weight) 0.001) "project none = 0"))))

(deftest project-subset-x
  (testing "project with subset selection :x"
    (let [model (dyn/auto-key (gen []
                  (trace :x (dist/gaussian 0 1))
                  (trace :y (dist/gaussian 0 1))
                  nil))
          constraints (cm/choicemap :x (mx/scalar 2.0) :y (mx/scalar 3.0))
          {:keys [trace]} (p/generate model [] constraints)
          weight-x (p/project model trace (sel/select :x))
          expected-lp-x (dc/dist-log-prob (dist/gaussian 0 1) (mx/scalar 2.0))]
      (mx/eval! weight-x expected-lp-x)
      (is (h/close? (mx/item expected-lp-x) (mx/item weight-x) 0.001)
          "project :x = log-prob of x"))))

(deftest project-subset-y
  (testing "project :y only"
    (let [model (dyn/auto-key (gen []
                  (trace :x (dist/gaussian 0 1))
                  (trace :y (dist/gaussian 0 1))
                  nil))
          constraints (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar -1.0))
          {:keys [trace]} (p/generate model [] constraints)
          weight-y (p/project model trace (sel/select :y))
          expected-lp-y (dc/dist-log-prob (dist/gaussian 0 1) (mx/scalar -1.0))]
      (mx/eval! weight-y expected-lp-y)
      (is (h/close? (mx/item expected-lp-y) (mx/item weight-y) 0.001)
          "project :y = log-prob of y"))))

(deftest project-complement
  (testing "complement selection"
    (let [model (dyn/auto-key (gen []
                  (trace :x (dist/gaussian 0 1))
                  (trace :y (dist/gaussian 0 1))
                  nil))
          constraints (cm/choicemap :x (mx/scalar 1.5) :y (mx/scalar -0.5))
          {:keys [trace]} (p/generate model [] constraints)
          weight-not-x (p/project model trace (sel/complement-sel (sel/select :x)))
          expected-lp-y (dc/dist-log-prob (dist/gaussian 0 1) (mx/scalar -0.5))]
      (mx/eval! weight-not-x expected-lp-y)
      (is (h/close? (mx/item expected-lp-y) (mx/item weight-not-x) 0.001)
          "project complement(:x) = log-prob of y"))))

(deftest project-distribution
  (testing "project on distribution directly"
    (let [d (dist/gaussian 5 2)
          trace (p/simulate d [])
          weight (p/project d trace sel/all)]
      (mx/eval! weight (:score trace))
      (is (h/close? (mx/item (:score trace)) (mx/item weight) 0.001)
          "dist project = score"))))

(deftest project-with-splice
  (testing "project with splice"
    (let [sub-model (dyn/auto-key (gen [mu]
                      (trace :z (dist/gaussian mu 1))
                      nil))
          model (dyn/auto-key (gen []
                  (let [x (trace :x (dist/gaussian 0 1))]
                    (splice :sub sub-model (mx/item x))
                    nil)))
          trace (p/simulate model [])
          weight-all (p/project model trace sel/all)]
      (mx/eval! weight-all (:score trace))
      (is (h/close? (mx/item (:score trace)) (mx/item weight-all) 0.001)
          "project all with splice = trace score"))))

(deftest project-map-combinator
  (testing "map combinator project all = score"
    (let [kernel (dyn/auto-key (gen [x]
                   (trace :y (dist/gaussian x 1))
                   nil))
          mapped (comb/map-combinator kernel)
          constraints (cm/choicemap
                        0 (cm/choicemap :y (mx/scalar 1.0))
                        1 (cm/choicemap :y (mx/scalar 2.0))
                        2 (cm/choicemap :y (mx/scalar 3.0)))
          {:keys [trace]} (p/generate mapped [[0.0 0.0 0.0]] constraints)
          weight-all (p/project mapped trace sel/all)]
      (mx/eval! weight-all (:score trace))
      (is (h/close? (mx/item (:score trace)) (mx/item weight-all) 0.001)
          "map project all = score"))))

(deftest project-map-combinator-subset
  (testing "map combinator select single element"
    (let [kernel (dyn/auto-key (gen [x]
                   (trace :y (dist/gaussian x 1))
                   nil))
          mapped (comb/map-combinator kernel)
          constraints (cm/choicemap
                        0 (cm/choicemap :y (mx/scalar 1.0))
                        1 (cm/choicemap :y (mx/scalar 2.0)))
          {:keys [trace]} (p/generate mapped [[0.0 0.0]] constraints)
          weight-0 (p/project mapped trace (sel/hierarchical 0 sel/all))
          expected-lp (dc/dist-log-prob (dist/gaussian 0 1) (mx/scalar 1.0))]
      (mx/eval! weight-0 expected-lp)
      (is (h/close? (mx/item expected-lp) (mx/item weight-0) 0.001)
          "map project element 0 = log-prob of y=1"))))

(deftest project-switch-combinator
  (testing "switch combinator"
    (let [branch-a (dyn/auto-key (gen []
                     (trace :v (dist/gaussian 0 1))
                     nil))
          branch-b (dyn/auto-key (gen []
                     (trace :v (dist/gaussian 10 1))
                     nil))
          sw (comb/switch-combinator branch-a branch-b)
          constraints (cm/choicemap :v (mx/scalar 0.5))
          {:keys [trace]} (p/generate sw [0] constraints)
          weight (p/project sw trace sel/all)]
      (mx/eval! weight (:score trace))
      (is (h/close? (mx/item (:score trace)) (mx/item weight) 0.001)
          "switch project = score"))))

(deftest project-unfold-combinator
  (testing "unfold combinator project all = score"
    (let [kernel (dyn/auto-key (gen [t state]
                   (let [v (trace :v (dist/gaussian state 1))]
                     (mx/eval! v)
                     (mx/item v))))
          uf (comb/unfold-combinator kernel)
          constraints (cm/choicemap
                        0 (cm/choicemap :v (mx/scalar 1.0))
                        1 (cm/choicemap :v (mx/scalar 2.0))
                        2 (cm/choicemap :v (mx/scalar 3.0)))
          {:keys [trace]} (p/generate uf [3 0.0] constraints)
          weight-all (p/project uf trace sel/all)]
      (mx/eval! weight-all (:score trace))
      (is (h/close? (mx/item (:score trace)) (mx/item weight-all) 0.001)
          "unfold project all = score"))))

(deftest project-unfold-subset
  (testing "unfold combinator subset"
    (let [kernel (dyn/auto-key (gen [t state]
                   (let [v (trace :v (dist/gaussian state 1))]
                     (mx/eval! v)
                     (mx/item v))))
          uf (comb/unfold-combinator kernel)
          constraints (cm/choicemap
                        0 (cm/choicemap :v (mx/scalar 1.0))
                        1 (cm/choicemap :v (mx/scalar 2.0)))
          {:keys [trace]} (p/generate uf [2 0.0] constraints)
          weight-0 (p/project uf trace (sel/hierarchical 0 sel/all))
          expected-lp (dc/dist-log-prob (dist/gaussian 0 1) (mx/scalar 1.0))]
      (mx/eval! weight-0 expected-lp)
      (is (h/close? (mx/item expected-lp) (mx/item weight-0) 0.001)
          "unfold project timestep 0"))))

(deftest project-mask-combinator
  (testing "mask combinator active and inactive"
    (let [inner (dyn/auto-key (gen []
                  (trace :v (dist/gaussian 0 1))
                  nil))
          masked (comb/mask-combinator inner)]
      (let [constraints-active (cm/choicemap :v (mx/scalar 1.5))
            {:keys [trace]} (p/generate masked [true] constraints-active)
            weight-active (p/project masked trace sel/all)]
        (mx/eval! weight-active (:score trace))
        (is (h/close? (mx/item (:score trace)) (mx/item weight-active) 0.001)
            "mask active project = score"))
      (let [trace (p/simulate masked [false])
            weight (p/project masked trace sel/all)]
        (mx/eval! weight)
        (is (h/close? 0.0 (mx/item weight) 0.001) "mask inactive project = 0")))))

(deftest project-manual-verification
  (testing "known values: x=0, y=0 under N(0,1)"
    (let [model (dyn/auto-key (gen []
                  (trace :x (dist/gaussian 0 1))
                  (trace :y (dist/gaussian 0 1))
                  nil))
          constraints (cm/choicemap :x (mx/scalar 0.0) :y (mx/scalar 0.0))
          {:keys [trace]} (p/generate model [] constraints)
          weight-x (p/project model trace (sel/select :x))
          weight-y (p/project model trace (sel/select :y))
          weight-all (p/project model trace sel/all)]
      (mx/eval! weight-x weight-y weight-all)
      (is (h/close? -0.9189 (mx/item weight-x) 0.01) "project x=0 under N(0,1)")
      (is (h/close? -0.9189 (mx/item weight-y) 0.01) "project y=0 under N(0,1)")
      (is (h/close? (+ (mx/item weight-x) (mx/item weight-y))
                    (mx/item weight-all) 0.001)
          "project all = x + y"))))

(cljs.test/run-tests)
