(ns genmlx.map-test
  "Tests for MAP (Maximum A Posteriori) optimization."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.mcmc :as mcmc])
  (:require-macros [genmlx.gen :refer [gen]]))

(deftest gaussian-posterior
  (testing "x ~ N(0,10), observe y ~ N(x,1) at y=5, MAP ~ 4.95"
    (let [model (gen []
                  (let [x (trace :x (dist/gaussian 0 10))]
                    (trace :y (dist/gaussian x 1))
                    x))
          obs (cm/choicemap :y (mx/scalar 5.0))
          result (mcmc/map-optimize
                   {:iterations 2000 :lr 0.05 :addresses [:x]}
                   model [] obs)]
      (is (h/close? 4.95 (first (:params result)) 0.2) "MAP x ~ 4.95")
      (is (js/isFinite (:score result)) "score is finite")
      (is (= 2000 (count (:score-history result))) "score-history has entries")
      (is (some? (:trace result)) "trace returned"))))

(deftest multi-parameter
  (testing "x ~ N(0,10), z ~ N(0,10), observe y1 ~ N(x,0.5) at 3.0, y2 ~ N(z,0.5) at -2.0"
    (let [model (gen []
                  (let [x (trace :x (dist/gaussian 0 10))
                        z (trace :z (dist/gaussian 0 10))]
                    (trace :y1 (dist/gaussian x 0.5))
                    (trace :y2 (dist/gaussian z 0.5))
                    [x z]))
          obs (cm/merge-cm (cm/choicemap :y1 (mx/scalar 3.0))
                           (cm/choicemap :y2 (mx/scalar -2.0)))
          result (mcmc/map-optimize
                   {:iterations 2000 :lr 0.05 :addresses [:x :z]}
                   model [] obs)
          [x-val z-val] (:params result)]
      (is (h/close? 3.0 x-val 0.3) "MAP x ~ 3.0")
      (is (h/close? -2.0 z-val 0.3) "MAP z ~ -2.0"))))

(deftest score-monotonicity
  (testing "score should generally increase"
    (let [model (gen []
                  (let [x (trace :x (dist/gaussian 0 10))]
                    (trace :y (dist/gaussian x 1))
                    x))
          obs (cm/choicemap :y (mx/scalar 5.0))
          result (mcmc/map-optimize
                   {:iterations 200 :lr 0.01 :addresses [:x]}
                   model [] obs)
          history (:score-history result)
          initial-score (first history)
          final-score (last history)]
      (is (>= final-score initial-score) "final score >= initial score"))))

(deftest sgd-optimizer
  (testing "SGD optimizer converges"
    (let [model (gen []
                  (let [x (trace :x (dist/gaussian 0 10))]
                    (trace :y (dist/gaussian x 1))
                    x))
          obs (cm/choicemap :y (mx/scalar 5.0))
          result (mcmc/map-optimize
                   {:iterations 1000 :optimizer :sgd :lr 0.05 :addresses [:x]}
                   model [] obs)]
      (is (h/close? 4.95 (first (:params result)) 0.3) "SGD MAP x ~ 4.95"))))

(cljs.test/run-tests)
