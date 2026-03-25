(ns genmlx.product-dist-test
  "Tests for product distributions."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm])
  (:require-macros [genmlx.gen :refer [gen]]))

(deftest vector-form-sample
  (testing "vector form: sample shape"
    (let [d (dist/product [(dist/gaussian 0 1) (dist/bernoulli 0.5) (dist/uniform 0 1)])
          v (dc/dist-sample d (rng/fresh-key))]
      (is (vector? v) "vector form returns vector")
      (is (= 3 (count v)) "vector form has 3 elements")
      (is (every? mx/array? v) "each element is MLX array"))))

(deftest map-form-sample
  (testing "map form: sample shape"
    (let [d (dist/product {:x (dist/gaussian 0 1) :y (dist/uniform 0 1)})
          v (dc/dist-sample d (rng/fresh-key))]
      (is (map? v) "map form returns map")
      (is (= #{:x :y} (set (keys v))) "map form has correct keys")
      (is (every? mx/array? (vals v)) "each value is MLX array"))))

(deftest vector-log-prob-sum
  (testing "log-prob = sum of component log-probs"
    (let [d1 (dist/gaussian 0 1)
          d2 (dist/uniform 0 1)
          prod (dist/product [d1 d2])
          v1 (mx/scalar 0.5)
          v2 (mx/scalar 0.3)
          lp1 (dc/dist-log-prob d1 v1)
          lp2 (dc/dist-log-prob d2 v2)
          lp-sum (mx/add lp1 lp2)
          lp-prod (dc/dist-log-prob prod [v1 v2])]
      (mx/eval! lp-sum lp-prod)
      (is (h/close? (mx/item lp-sum) (mx/item lp-prod) 1e-6)
          "log-prob = sum of components"))))

(deftest map-log-prob-sum
  (testing "map form log-prob"
    (let [d1 (dist/gaussian 2 1)
          d2 (dist/gaussian -1 3)
          prod (dist/product {:a d1 :b d2})
          va (mx/scalar 1.0)
          vb (mx/scalar 0.0)
          lp1 (dc/dist-log-prob d1 va)
          lp2 (dc/dist-log-prob d2 vb)
          lp-sum (mx/add lp1 lp2)
          lp-prod (dc/dist-log-prob prod {:a va :b vb})]
      (mx/eval! lp-sum lp-prod)
      (is (h/close? (mx/item lp-sum) (mx/item lp-prod) 1e-6)
          "map log-prob = sum of components"))))

(deftest reparam-all-continuous
  (testing "reparam for all-continuous"
    (let [prod (dist/product [(dist/gaussian 0 1) (dist/gaussian 5 2)])
          v (dc/dist-reparam prod (rng/fresh-key))]
      (is (vector? v) "reparam returns vector")
      (is (= 2 (count v)) "reparam vector has 2 elements"))))

(deftest reparam-throws-for-discrete
  (testing "reparam throws for discrete component"
    (let [prod (dist/product [(dist/gaussian 0 1) (dist/bernoulli 0.5)])]
      (is (try (dc/dist-reparam prod (rng/fresh-key))
               false
               (catch :default _ true))
          "reparam throws for mixed"))))

(deftest support-cartesian-product
  (testing "Cartesian product for discrete"
    (let [prod (dist/product [(dist/bernoulli 0.5) (dist/bernoulli 0.3)])
          sup (dc/dist-support prod)]
      (is (= 4 (count sup)) "support has 4 elements (2x2)")
      (is (every? vector? sup) "support elements are vectors"))))

(deftest map-form-support
  (testing "map form support"
    (let [prod (dist/product {:a (dist/bernoulli 0.5) :b (dist/bernoulli 0.7)})
          sup (dc/dist-support prod)]
      (is (= 4 (count sup)) "map support has 4 elements")
      (is (every? map? sup) "map support elements are maps"))))

(deftest support-throws-for-continuous
  (testing "support throws for continuous"
    (let [prod (dist/product [(dist/gaussian 0 1)])]
      (is (try (dc/dist-support prod)
               false
               (catch :default _ true))
          "support throws for continuous"))))

(deftest gfi-integration
  (testing "trace a product inside a gen body"
    (let [model (dyn/auto-key
                  (gen []
                    (let [v (trace :joint (dist/product [(dist/gaussian 0 1)
                                                         (dist/gaussian 5 2)]))]
                      v)))
          tr (p/simulate model [])]
      (is (vector? (:retval tr)) "trace retval is vector")
      (is (cm/has-value? (cm/get-submap (:choices tr) :joint)) "trace choices has :joint"))))

(deftest gfi-generate-with-constraints
  (testing "GFI generate with constraints"
    (let [prod (dist/product [(dist/gaussian 0 1) (dist/gaussian 5 2)])
          constrained-val [(mx/scalar 1.0) (mx/scalar 3.0)]
          {:keys [trace weight]} (p/generate prod [] (cm/->Value constrained-val))]
      (is (js/isFinite (mx/item weight)) "generate weight is finite")
      (is (let [rv (:retval trace)]
            (and (= 2 (count rv))
                 (< (js/Math.abs (- 1.0 (mx/item (nth rv 0)))) 1e-6)
                 (< (js/Math.abs (- 3.0 (mx/item (nth rv 1)))) 1e-6)))
          "generate retval matches constraint"))))

(deftest sample-n-batch-shapes
  (testing "vector form sample-n"
    (let [prod (dist/product [(dist/gaussian 0 1) (dist/uniform 0 1)])
          samples (dc/dist-sample-n prod (rng/fresh-key) 10)]
      (is (vector? samples) "sample-n returns vector")
      (is (= 2 (count samples)) "sample-n vector has 2 components")
      (is (= [10] (mx/shape (nth samples 0))) "component 0 shape is [10]")
      (is (= [10] (mx/shape (nth samples 1))) "component 1 shape is [10]"))))

(deftest map-form-sample-n
  (testing "map form sample-n"
    (let [prod (dist/product {:x (dist/gaussian 0 1) :y (dist/uniform 0 1)})
          samples (dc/dist-sample-n prod (rng/fresh-key) 5)]
      (is (map? samples) "map sample-n returns map")
      (is (= #{:x :y} (set (keys samples))) "map sample-n has correct keys")
      (is (= [5] (mx/shape (:x samples))) ":x shape is [5]")
      (is (= [5] (mx/shape (:y samples))) ":y shape is [5]"))))

(deftest statistical-validation
  (testing "sample means"
    (let [prod (dist/product [(dist/gaussian 3.0 0.5) (dist/gaussian -2.0 1.0)])
          samples (dc/dist-sample-n prod (rng/fresh-key) 5000)
          mean0 (mx/item (mx/mean (nth samples 0)))
          mean1 (mx/item (mx/mean (nth samples 1)))]
      (is (h/close? 3.0 mean0 0.2) "component 0 mean ~ 3.0")
      (is (h/close? -2.0 mean1 0.2) "component 1 mean ~ -2.0"))))

(cljs.test/run-tests)
