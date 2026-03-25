(ns genmlx.fit-test
  "fit API tests.
   Tests the one-call entry point across method selection, dispatch,
   posterior extraction, parameter learning, and edge cases."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p]
            [genmlx.fit :as fit]
            [genmlx.method-selection :as ms])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Test models
;; ---------------------------------------------------------------------------

;; 1. Conjugate Normal-Normal -> :exact
(def m-conjugate
  (gen [x]
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :y (dist/gaussian mu 1)))))

;; 2. Non-conjugate static (2 latent, uniform prior) -> :hmc
(def m-nonconj-small
  (gen []
    (let [a (trace :a (dist/uniform -5 5))
          b (trace :b (dist/uniform -5 5))]
      (trace :y (dist/gaussian (mx/add a b) 0.5)))))

;; 3. Simple 1-latent gaussian
(def m-simple
  (gen []
    (let [x (trace :x (dist/gaussian 0 1))]
      x)))

;; 4. Multi-latent gaussian (for handler-is)
(def m-multi
  (gen []
    (let [a (trace :a (dist/gaussian 0 1))
          b (trace :b (dist/gaussian 0 1))
          c (trace :c (dist/gaussian 0 1))]
      (trace :y (dist/gaussian (mx/add a (mx/add b c)) 1)))))

;; ---------------------------------------------------------------------------
;; Tests
;; ---------------------------------------------------------------------------

(deftest return-structure
  (testing "Return structure"
    (let [obs (cm/choicemap :y 3.0)
          result (fit/fit m-conjugate [0] obs)]
      (is (map? result) "result is a map")
      (is (contains? result :method) "has :method")
      (is (contains? result :trace) "has :trace")
      (is (contains? result :posterior) "has :posterior")
      (is (contains? result :log-ml) "has :log-ml")
      (is (contains? result :diagnostics) "has :diagnostics")
      (is (contains? result :elapsed-ms) "has :elapsed-ms")
      (is (keyword? (:method result)) ":method is keyword")
      (is (pos? (:elapsed-ms result)) ":elapsed-ms is positive")
      (is (map? (:diagnostics result)) ":diagnostics is map")
      (is (string? (get-in result [:diagnostics :reason])) ":diagnostics has :reason"))))

(deftest conjugate-auto-exact
  (testing "Conjugate model -> :exact"
    (let [obs (cm/choicemap :y 3.0)
          result (fit/fit m-conjugate [0] obs)]
      (is (= :exact (:method result)) "conjugate auto -> :exact")
      (is (some? (:trace result)) "trace is not nil")
      (is (number? (:log-ml result)) "log-ml is number")
      (is (neg? (:log-ml result)) "log-ml is negative (log-prob)")
      (is (contains? (:posterior result) :mu) "posterior has :mu")
      (is (contains? (get-in result [:posterior :mu]) :value) "posterior :mu has :value")
      ;; Conjugate posterior mean for N(0,10) prior, N(mu,1) likelihood, y=3:
      ;; posterior mean = (0/100 + 3/1) / (1/100 + 1/1) = 3 / 1.01 ~ 2.97
      (is (h/close? 2.97 (get-in result [:posterior :mu :value]) 0.1) "posterior mean near 2.97"))))

(deftest nonconj-static-hmc
  (testing "Non-conjugate static -> :hmc"
    (let [obs (cm/choicemap :y 2.0)
          result (fit/fit m-nonconj-small [] obs {:samples 30 :burn 10})]
      (is (= :hmc (:method result)) "non-conj static -> :hmc")
      (is (contains? (:posterior result) :a) "posterior has :a")
      (is (contains? (:posterior result) :b) "posterior has :b")
      (is (contains? (get-in result [:posterior :a]) :mean) "posterior :a has :mean")
      (is (contains? (get-in result [:posterior :a]) :std) "posterior :a has :std")
      (is (contains? (get-in result [:posterior :a]) :samples) "posterior :a has :samples")
      ;; a + b should be near 2.0 on average (sum constrained by observation)
      (let [a-mean (get-in result [:posterior :a :mean])
            b-mean (get-in result [:posterior :b :mean])]
        (is (h/close? 2.0 (+ a-mean b-mean) 2.0) "a+b mean near 2.0")))))

(deftest method-override
  (testing "Override conjugate model to use handler-IS"
    (let [obs (cm/choicemap :y 3.0)
          result (fit/fit m-conjugate [0] obs {:method :handler-is :particles 100})]
      (is (= :handler-is (:method result)) "override -> :handler-is")
      (is (some? (:trace result)) "has trace")
      (is (number? (:log-ml result)) "has log-ml")
      (is (some? (:posterior result)) "has posterior")
      (is (= "User-specified" (get-in result [:diagnostics :reason])) "reason is User-specified")))

  (testing "Override to :mcmc"
    (let [obs (cm/choicemap :y 3.0)
          result (fit/fit m-conjugate [0] obs {:method :mcmc :samples 10 :burn 5})]
      (is (= :mcmc (:method result)) "override -> :mcmc")
      (is (some? (:trace result)) "mcmc has trace"))))

(deftest handler-is-path
  (testing "Handler-IS path"
    (let [obs (cm/choicemap :y 1.0)
          result (fit/fit m-multi [] obs {:method :handler-is :particles 200})]
      (is (= :handler-is (:method result)) "handler-is method")
      (is (some? (:trace result)) "has trace")
      (is (number? (:log-ml result)) "has log-ml")
      (is (some? (:posterior result)) "posterior has latents"))))

(deftest verbose-mode
  (testing "Verbose mode does not crash"
    (let [obs (cm/choicemap :y 3.0)
          result (fit/fit m-conjugate [0] obs {:verbose? true})]
      (is (= :exact (:method result)) "verbose fit still returns :exact"))))

(deftest callback-invocation
  (testing "Callback invocation"
    (let [obs (cm/choicemap :y 3.0)
          callback-calls (atom [])
          result (fit/fit m-conjugate [0] obs {:learn [:mu]
                                                :iterations 10
                                                :lr 0.01
                                                :log-every 5
                                                :callback (fn [info]
                                                            (swap! callback-calls conj info))})]
      (is (pos? (count @callback-calls)) "callback was invoked")
      (is (contains? (first @callback-calls) :iter) "callback info has :iter")
      (is (contains? (first @callback-calls) :loss) "callback info has :loss"))))

(deftest learning-loop
  (testing "Learning loop (:learn)"
    (let [obs (cm/choicemap :y 3.0)
          result (fit/fit m-conjugate [0] obs {:learn [:mu]
                                                :iterations 20
                                                :lr 0.01})]
      (is (some? (:params result)) "learn produces :params")
      (is (some? (:loss-history result)) "learn produces :loss-history")
      (is (vector? (:loss-history result)) "loss-history is vector")
      (is (pos? (count (:loss-history result))) "loss-history is non-empty"))))

(deftest edge-cases
  (testing "Empty observations (nil)"
    (let [result (fit/fit m-simple [] nil)]
      (is (some? result) "nil data works")
      (is (keyword? (:method result)) "nil data has method")
      (is (pos? (:elapsed-ms result)) "nil data has elapsed")))

  (testing "Empty ChoiceMap"
    (let [result (fit/fit m-simple [] cm/EMPTY)]
      (is (some? result) "empty cm works")
      (is (keyword? (:method result)) "empty cm has method")))

  (testing "Unknown method throws"
    (let [threw? (atom false)]
      (try
        (fit/fit m-simple [] nil {:method :bogus})
        (catch :default _e
          (reset! threw? true)))
      (is @threw? "unknown method throws")))

  (testing "All-observed model (trivial exact)"
    (let [obs (cm/choicemap :x 1.0)
          result (fit/fit m-simple [] obs)]
      (is (some? result) "all-observed works")
      (is (some? (:log-ml result)) "all-observed has log-ml"))))

(deftest reproducibility-with-key
  (testing "Reproducibility with :key"
    (let [obs (cm/choicemap :y 3.0)
          key1 (rng/fresh-key 42)
          key2 (rng/fresh-key 42)
          r1 (fit/fit m-conjugate [0] obs {:method :handler-is :particles 50 :key key1})
          r2 (fit/fit m-conjugate [0] obs {:method :handler-is :particles 50 :key key2})]
      ;; Both runs with same seed should produce same log-ml
      (is (h/close? (:log-ml r1) (:log-ml r2) 0.001) "same key -> same log-ml"))))

(deftest diagnostics-content
  (testing "Diagnostics content"
    (let [obs (cm/choicemap :y 3.0)
          result (fit/fit m-conjugate [0] obs)]
      (is (string? (get-in result [:diagnostics :reason])) "diagnostics has :reason")
      (is (number? (get-in result [:diagnostics :n-residual])) "diagnostics has :n-residual")
      (is (number? (get-in result [:diagnostics :n-latent])) "diagnostics has :n-latent"))))

(deftest end-to-end-linear-regression
  (testing "End-to-end linear regression"
    (let [m-linreg (gen [xs]
                     (let [slope (trace :slope (dist/gaussian 0 10))
                           intercept (trace :intercept (dist/gaussian 0 10))]
                       (doseq [[j x] (map-indexed vector xs)]
                         (trace (keyword (str "y" j))
                                (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                                       intercept) 1)))
                       slope))
          ;; Data: y = 2*x + 1 + noise
          xs [0 1 2 3 4]
          obs (cm/choicemap :y0 1.0 :y1 3.1 :y2 4.9 :y3 7.0 :y4 9.1)
          result (fit/fit m-linreg [xs] obs)]
      (is (keyword? (:method result)) "linreg has method")
      (is (pos? (:elapsed-ms result)) "linreg has elapsed")
      ;; The auto-selected method should produce some result
      (is (or (some? (:posterior result))
              (some? (:trace result))) "linreg has posterior or trace"))))

(cljs.test/run-tests)
