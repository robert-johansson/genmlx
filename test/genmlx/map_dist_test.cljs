(ns genmlx.map-dist-test
  "Tests for map->dist: creating distributions from plain maps."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm])
  (:require-macros [genmlx.gen :refer [gen]]))

(deftest basic-map->dist-usage
  (testing "custom uniform-like distribution"
    (let [my-uniform (dc/map->dist
                       {:type :test-uniform
                        :sample (fn [key]
                                  (rng/uniform (rng/ensure-key key) []))
                        :log-prob (fn [value]
                                    (mx/scalar 0.0))})
          v (dc/dist-sample my-uniform nil)]
      (mx/eval! v)
      (let [val (mx/realize v)]
        (is (and (>= val 0) (<= val 1)) "sample in [0,1]"))
      (let [lp (dc/dist-log-prob my-uniform (mx/scalar 0.5))]
        (mx/eval! lp)
        (is (h/close? 0.0 (mx/realize lp) 1e-6) "log-prob is 0")))))

(deftest gfi-integration
  (testing "use inside gen body"
    (let [my-dist (dc/map->dist
                    {:type :test-gaussian-bridge
                     :sample (fn [key]
                               (let [key (rng/ensure-key key)
                                     z (rng/normal key [])]
                                 (mx/add (mx/scalar 5.0)
                                         (mx/multiply (mx/scalar 0.1) z))))
                     :log-prob (fn [value]
                                 (let [z (mx/divide (mx/subtract value (mx/scalar 5.0))
                                                    (mx/scalar 0.1))]
                                   (mx/subtract
                                     (mx/negative (mx/multiply (mx/scalar 0.5)
                                                               (mx/multiply z z)))
                                     (mx/scalar (+ (* 0.5 (js/Math.log (* 2 js/Math.PI)))
                                                   (js/Math.log 0.1))))))})
          model (gen []
                  (let [x (trace :x my-dist)]
                    (mx/eval! x) (mx/item x)))
          trace (p/simulate (dyn/auto-key model) [])
          retval (:retval trace)]
      (is (< (js/Math.abs (- retval 5)) 1.0) "simulate: retval near 5")
      (is (js/isFinite (mx/realize (:score trace))) "simulate: score finite")

      (let [{:keys [trace weight]} (p/generate (dyn/auto-key model) [] (cm/choicemap :x (mx/scalar 5.0)))]
        (mx/eval! weight)
        (is (js/isFinite (mx/realize weight)) "generate: weight finite")
        (is (h/close? 5.0 (mx/realize (cm/get-value (cm/get-submap (:choices trace) :x))) 1e-6)
            "generate: constrained value"))

      (let [{:keys [weight]} (p/assess (dyn/auto-key model) [] (cm/choicemap :x (mx/scalar 5.0)))]
        (mx/eval! weight)
        (is (js/isFinite (mx/realize weight)) "assess: weight finite")))))

(deftest optional-reparam
  (testing "reparam returns finite value"
    (let [my-dist (dc/map->dist
                    {:type :test-reparam
                     :sample (fn [key]
                               (rng/normal (rng/ensure-key key) []))
                     :log-prob (fn [value]
                                 (mx/negative (mx/multiply (mx/scalar 0.5)
                                                           (mx/multiply value value))))
                     :reparam (fn [key]
                                (rng/normal (rng/ensure-key key) []))})
          v (dc/dist-reparam my-dist nil)]
      (mx/eval! v)
      (is (js/isFinite (mx/realize v)) "reparam: returns finite value"))))

(deftest optional-sample-n
  (testing "sample-n with custom implementation"
    (let [my-dist (dc/map->dist
                    {:type :test-batch
                     :sample (fn [key]
                               (rng/normal (rng/ensure-key key) []))
                     :log-prob (fn [value]
                                 (mx/scalar 0.0))
                     :sample-n (fn [key n]
                                 (rng/normal (rng/ensure-key key) [n]))})
          samples (dc/dist-sample-n my-dist (rng/fresh-key) 50)]
      (mx/eval! samples)
      (is (= [50] (mx/shape samples)) "sample-n: shape [50]"))))

(deftest auto-generated-type
  (testing "different auto types and correct sampling"
    (let [d1 (dc/map->dist {:sample (fn [k] (mx/scalar 1.0))
                             :log-prob (fn [v] (mx/scalar 0.0))})
          d2 (dc/map->dist {:sample (fn [k] (mx/scalar 2.0))
                             :log-prob (fn [v] (mx/scalar 0.0))})]
      (is (not= (:type d1) (:type d2)) "different auto types")
      (let [v1 (dc/dist-sample d1 nil)
            v2 (dc/dist-sample d2 nil)]
        (mx/eval! v1 v2)
        (is (h/close? 1.0 (mx/realize v1) 1e-6) "d1 samples 1.0")
        (is (h/close? 2.0 (mx/realize v2) 1e-6) "d2 samples 2.0")))))

(cljs.test/run-tests)
