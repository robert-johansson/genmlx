(ns genmlx.adev-test
  "ADEV gradient estimation tests."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.handler :as handler]
            [genmlx.inference.adev :as adev])
  (:require-macros [genmlx.gen :refer [gen]]))

(deftest has-reparam-detection-test
  (testing "has-reparam? detection"
    (is (adev/has-reparam? (dist/gaussian 0 1)) "gaussian is reparameterizable")
    (is (adev/has-reparam? (dist/uniform 0 1)) "uniform is reparameterizable")
    (is (adev/has-reparam? (dist/exponential 1)) "exponential is reparameterizable")
    (is (adev/has-reparam? (dist/laplace 0 1)) "laplace is reparameterizable")
    (is (not (adev/has-reparam? (dist/bernoulli 0.5))) "bernoulli is NOT reparameterizable")
    (is (not (adev/has-reparam? (dist/categorical (mx/array [-1 -1])))) "categorical is NOT reparameterizable")
    (is (not (adev/has-reparam? (dist/beta-dist 2 2))) "beta is NOT reparameterizable")))

(deftest pure-reparam-model-test
  (testing "pure reparam model (ADEV execute)"
    (let [model (gen []
                  (let [x (trace :x (dist/gaussian 0 1))]
                    x))
          key (rng/fresh-key)
          {:keys [trace reinforce-lp]} (adev/adev-execute model [] key)]
      (is (cm/has-value? (cm/get-submap (:choices trace) :x)) "trace has choices")
      (is (number? (mx/item (:score trace))) "trace has score")
      (is (h/close? 0.0 (mx/item reinforce-lp) 1e-6) "reinforce-lp is 0 for pure reparam"))))

(deftest mixed-model-test
  (testing "mixed model (reparam + REINFORCE)"
    (let [model (gen []
                  (let [x (trace :x (dist/gaussian 0 1))
                        b (trace :b (dist/bernoulli 0.5))]
                    (mx/add x b)))
          key (rng/fresh-key)
          {:keys [trace reinforce-lp]} (adev/adev-execute model [] key)]
      (is (cm/has-value? (cm/get-submap (:choices trace) :x)) "trace has gaussian choice")
      (is (cm/has-value? (cm/get-submap (:choices trace) :b)) "trace has bernoulli choice")
      (is (js/isFinite (mx/item reinforce-lp)) "reinforce-lp is finite"))))

(deftest adev-surrogate-loss-test
  (testing "ADEV surrogate loss"
    (let [model (gen []
                  (let [x (trace :x (dist/gaussian 0 1))
                        b (trace :b (dist/bernoulli 0.5))]
                    (mx/add x b)))
          cost-fn (fn [trace] (mx/square (:retval trace)))
          key (rng/fresh-key)
          surrogate (adev/adev-surrogate model [] cost-fn key)]
      (mx/eval! surrogate)
      (is (js/isFinite (mx/item surrogate)) "surrogate is finite")
      (is (>= (mx/item surrogate) 0.0) "surrogate is non-negative (squared cost)"))))

(deftest adev-gradient-with-params-test
  (testing "ADEV gradient with params"
    (let [model (gen []
                  (let [mu (param :mu 0.0)
                        x (trace :x (dist/gaussian mu (mx/scalar 1.0)))]
                    x))
          cost-fn (fn [trace]
                    (mx/square (mx/subtract (:retval trace) (mx/scalar 3.0))))
          param-names [:mu]
          params (mx/array [0.0])
          {:keys [loss grad]} (adev/adev-gradient {:n-samples 10}
                                                   model [] cost-fn
                                                   param-names params)]
      (mx/eval! loss grad)
      (is (js/isFinite (mx/item loss)) "loss is finite")
      (is (js/isFinite (mx/item (mx/index grad 0))) "grad is finite")
      (is (< (mx/item (mx/index grad 0)) 0) "grad is negative (should increase mu)"))))

(deftest adev-optimization-convergence-test
  (testing "ADEV optimization convergence"
    (let [model (gen []
                  (let [mu (param :mu 0.0)
                        x (trace :x (dist/gaussian mu (mx/scalar 1.0)))]
                    x))
          cost-fn (fn [trace]
                    (mx/square (mx/subtract (:retval trace) (mx/scalar 5.0))))
          param-names [:mu]
          init-params (mx/array [0.0])
          {:keys [params loss-history]} (adev/adev-optimize
                                          {:iterations 200 :lr 0.1 :n-samples 10}
                                          model [] cost-fn param-names init-params)
          final-mu (mx/item (mx/index params 0))
          first-loss (first loss-history)
          last-loss (last loss-history)]
      (is (h/close? 5.0 final-mu 1.5) "mu converges near 5.0")
      (is (< last-loss first-loss) "loss decreases"))))

(deftest gradient-finite-difference-test
  (testing "gradient correctness (finite difference check)"
    (let [model (gen []
                  (let [mu (param :mu 0.0)
                        x (trace :x (dist/gaussian mu (mx/scalar 1.0)))]
                    x))
          cost-fn (fn [trace] (mx/square (:retval trace)))
          param-names [:mu]
          mu-val 2.0
          params (mx/array [mu-val])
          {:keys [grad]} (adev/adev-gradient {:n-samples 500}
                                              model [] cost-fn
                                              param-names params)
          adev-grad (mx/item (mx/index grad 0))
          analytical-grad (* 2.0 mu-val)]
      (is (h/close? analytical-grad adev-grad 0.5) "ADEV grad ~ analytical 2*mu"))))

(deftest vadev-gradient-test
  (testing "vectorized ADEV gradient (vadev-gradient)"
    (let [model (gen []
                  (let [mu (param :mu 0.0)
                        x (trace :x (dist/gaussian mu (mx/scalar 1.0)))]
                    x))
          cost-fn (fn [result]
                    (mx/square (mx/subtract (:retval result) (mx/scalar 3.0))))
          param-names [:mu]
          params (mx/array [0.0])
          {:keys [loss grad]} (adev/vadev-gradient {:n-samples 100}
                                                    model [] cost-fn
                                                    param-names params)]
      (mx/eval! loss grad)
      (is (js/isFinite (mx/item loss)) "vadev loss is finite")
      (is (js/isFinite (mx/item (mx/index grad 0))) "vadev grad is finite")
      (is (< (mx/item (mx/index grad 0)) 0) "vadev grad is negative (should increase mu)"))))

(deftest vadev-vs-adev-agreement-test
  (testing "vadev-gradient vs adev-gradient agreement"
    (let [model (gen []
                  (let [mu (param :mu 0.0)
                        x (trace :x (dist/gaussian mu (mx/scalar 1.0)))]
                    x))
          cost-fn-scalar (fn [trace]
                           (mx/square (mx/subtract (:retval trace) (mx/scalar 0.0))))
          cost-fn-batch (fn [result]
                          (mx/square (mx/subtract (:retval result) (mx/scalar 0.0))))
          param-names [:mu]
          mu-val 2.0
          params (mx/array [mu-val])
          analytical-grad (* 2.0 mu-val)
          {adev-grad-arr :grad} (adev/adev-gradient {:n-samples 500}
                                                     model [] cost-fn-scalar
                                                     param-names params)
          adev-g (mx/item (mx/index adev-grad-arr 0))
          {vadev-grad-arr :grad} (adev/vadev-gradient {:n-samples 500}
                                                       model [] cost-fn-batch
                                                       param-names params)
          vadev-g (mx/item (mx/index vadev-grad-arr 0))]
      (is (h/close? analytical-grad adev-g 0.5) "adev-gradient ~ analytical 2*mu")
      (is (h/close? analytical-grad vadev-g 0.5) "vadev-gradient ~ analytical 2*mu"))))

(deftest compiled-adev-optimize-test
  (testing "compiled-adev-optimize convergence"
    (let [model (gen []
                  (let [mu (param :mu 0.0)
                        x (trace :x (dist/gaussian mu (mx/scalar 1.0)))]
                    x))
          cost-fn (fn [result]
                    (mx/square (mx/subtract (:retval result) (mx/scalar 5.0))))
          param-names [:mu]
          init-params (mx/array [0.0])
          {:keys [params loss-history]} (adev/compiled-adev-optimize
                                          {:iterations 200 :lr 0.1 :n-samples 100}
                                          model [] cost-fn param-names init-params)
          final-mu (mx/item (mx/index params 0))
          first-loss (first loss-history)
          last-loss (last loss-history)]
      (is (h/close? 5.0 final-mu 1.5) "mu converges near 5.0")
      (is (< last-loss first-loss) "loss decreases"))))

(deftest baseline-variance-reduction-test
  (testing "baseline reduces variance (mixed model)"
    (let [model (gen []
                  (let [mu (param :mu 0.0)
                        x (trace :x (dist/gaussian mu (mx/scalar 1.0)))
                        b (trace :b (dist/bernoulli 0.5))]
                    (mx/add x b)))
          cost-fn (fn [result]
                    (mx/square (mx/subtract (:retval result) (mx/scalar 5.0))))
          param-names [:mu]
          init-params (mx/array [0.0])
          iters 200
          bl-result (adev/compiled-adev-optimize
                      {:iterations iters :lr 0.1 :n-samples 100
                       :baseline-decay 0.9}
                      model [] cost-fn param-names init-params)
          half (quot iters 2)
          tail-bl (drop half (:loss-history bl-result))
          mean-fn (fn [xs] (/ (reduce + xs) (count xs)))
          var-fn (fn [xs]
                   (let [m (mean-fn xs)]
                     (/ (reduce + (map #(* (- % m) (- % m)) xs)) (count xs))))
          var-bl (var-fn tail-bl)]
      (is (js/isFinite var-bl) "baseline loss variance is finite"))))

(deftest compiled-adev-baseline-convergence-test
  (testing "compiled-adev-optimize with baseline convergence"
    (let [model (gen []
                  (let [mu (param :mu 0.0)
                        x (trace :x (dist/gaussian mu (mx/scalar 1.0)))]
                    x))
          cost-fn (fn [result]
                    (mx/square (mx/subtract (:retval result) (mx/scalar 5.0))))
          param-names [:mu]
          init-params (mx/array [0.0])
          {:keys [params loss-history]} (adev/compiled-adev-optimize
                                          {:iterations 200 :lr 0.1 :n-samples 100
                                           :baseline-decay 0.9}
                                          model [] cost-fn param-names init-params)
          final-mu (mx/item (mx/index params 0))
          first-loss (first loss-history)
          last-loss (last loss-history)]
      (is (h/close? 5.0 final-mu 1.5) "mu converges near 5.0 with baseline")
      (is (< last-loss first-loss) "loss decreases with baseline"))))

(cljs.test/run-tests)
