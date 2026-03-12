(ns genmlx.fit-test
  "WP-4 Gate 4: fit API tests.
   Tests the one-call entry point across method selection, dispatch,
   posterior extraction, parameter learning, and edge cases (~30 tests)."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p]
            [genmlx.fit :as fit]
            [genmlx.method-selection :as ms])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Test helpers
;; ---------------------------------------------------------------------------

(def pass-count (atom 0))
(def fail-count (atom 0))

(defn assert-true [msg v]
  (if v
    (do (swap! pass-count inc) (println (str "  PASS: " msg)))
    (do (swap! fail-count inc) (println (str "  FAIL: " msg)))))

(defn assert-equal [msg expected actual]
  (if (= expected actual)
    (do (swap! pass-count inc) (println (str "  PASS: " msg)))
    (do (swap! fail-count inc) (println (str "  FAIL: " msg " expected=" expected " actual=" actual)))))

(defn assert-close [msg expected actual tol]
  (if (<= (js/Math.abs (- expected actual)) tol)
    (do (swap! pass-count inc) (println (str "  PASS: " msg)))
    (do (swap! fail-count inc) (println (str "  FAIL: " msg " expected=" expected " actual=" actual " tol=" tol)))))

;; ---------------------------------------------------------------------------
;; Test models
;; ---------------------------------------------------------------------------

;; 1. Conjugate Normal-Normal → :exact
(def m-conjugate
  (gen [x]
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :y (dist/gaussian mu 1)))))

;; 2. Non-conjugate static (2 latent, uniform prior) → :hmc
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

;; ===========================================================================
;; Section 1: Return structure
;; ===========================================================================

(println "\n== Section 1: Return structure ==")

(let [obs (cm/choicemap :y 3.0)
      result (fit/fit m-conjugate [0] obs)]
  (assert-true "result is a map" (map? result))
  (assert-true "has :method" (contains? result :method))
  (assert-true "has :trace" (contains? result :trace))
  (assert-true "has :posterior" (contains? result :posterior))
  (assert-true "has :log-ml" (contains? result :log-ml))
  (assert-true "has :diagnostics" (contains? result :diagnostics))
  (assert-true "has :elapsed-ms" (contains? result :elapsed-ms))
  (assert-true ":method is keyword" (keyword? (:method result)))
  (assert-true ":elapsed-ms is positive" (pos? (:elapsed-ms result)))
  (assert-true ":diagnostics is map" (map? (:diagnostics result)))
  (assert-true ":diagnostics has :reason" (string? (get-in result [:diagnostics :reason]))))

;; ===========================================================================
;; Section 2: Auto-selection → :exact for conjugate
;; ===========================================================================

(println "\n== Section 2: Conjugate model → :exact ==")

(let [obs (cm/choicemap :y 3.0)
      result (fit/fit m-conjugate [0] obs)]
  (assert-equal "conjugate auto → :exact" :exact (:method result))
  (assert-true "trace is not nil" (some? (:trace result)))
  (assert-true "log-ml is number" (number? (:log-ml result)))
  (assert-true "log-ml is negative (log-prob)" (neg? (:log-ml result)))
  (assert-true "posterior has :mu" (contains? (:posterior result) :mu))
  (assert-true "posterior :mu has :value" (contains? (get-in result [:posterior :mu]) :value))
  ;; Conjugate posterior mean for N(0,10) prior, N(mu,1) likelihood, y=3:
  ;; posterior mean = (0/100 + 3/1) / (1/100 + 1/1) = 3 / 1.01 ≈ 2.97
  (assert-close "posterior mean near 2.97" 2.97
                (get-in result [:posterior :mu :value]) 0.1))

;; ===========================================================================
;; Section 3: Auto-selection → :hmc for non-conjugate static
;; ===========================================================================

(println "\n== Section 3: Non-conjugate static → :hmc ==")

(let [obs (cm/choicemap :y 2.0)
      result (fit/fit m-nonconj-small [] obs {:samples 30 :burn 10})]
  (assert-equal "non-conj static → :hmc" :hmc (:method result))
  (assert-true "posterior has :a" (contains? (:posterior result) :a))
  (assert-true "posterior has :b" (contains? (:posterior result) :b))
  (assert-true "posterior :a has :mean" (contains? (get-in result [:posterior :a]) :mean))
  (assert-true "posterior :a has :std" (contains? (get-in result [:posterior :a]) :std))
  (assert-true "posterior :a has :samples" (contains? (get-in result [:posterior :a]) :samples))
  ;; a + b should be near 2.0 on average (sum constrained by observation)
  (let [a-mean (get-in result [:posterior :a :mean])
        b-mean (get-in result [:posterior :b :mean])]
    (assert-close "a+b mean near 2.0" 2.0 (+ a-mean b-mean) 2.0)))

;; ===========================================================================
;; Section 4: Method override
;; ===========================================================================

(println "\n== Section 4: Method override ==")

;; Override conjugate model to use handler-IS instead of exact
(let [obs (cm/choicemap :y 3.0)
      result (fit/fit m-conjugate [0] obs {:method :handler-is :particles 100})]
  (assert-equal "override → :handler-is" :handler-is (:method result))
  (assert-true "has trace" (some? (:trace result)))
  (assert-true "has log-ml" (number? (:log-ml result)))
  (assert-true "has posterior" (some? (:posterior result)))
  (assert-true "reason is User-specified"
               (= "User-specified" (get-in result [:diagnostics :reason]))))

;; Override to :mcmc
(let [obs (cm/choicemap :y 3.0)
      result (fit/fit m-conjugate [0] obs {:method :mcmc :samples 10 :burn 5})]
  (assert-equal "override → :mcmc" :mcmc (:method result))
  (assert-true "mcmc has trace" (some? (:trace result))))

;; ===========================================================================
;; Section 5: handler-IS path
;; ===========================================================================

(println "\n== Section 5: Handler-IS path ==")

(let [obs (cm/choicemap :y 1.0)
      result (fit/fit m-multi [] obs {:method :handler-is :particles 200})]
  (assert-equal "handler-is method" :handler-is (:method result))
  (assert-true "has trace" (some? (:trace result)))
  (assert-true "has log-ml" (number? (:log-ml result)))
  (assert-true "posterior has latents" (some? (:posterior result))))

;; ===========================================================================
;; Section 6: Verbose mode
;; ===========================================================================

(println "\n== Section 6: Verbose mode ==")

;; Just verify it doesn't crash — verbose prints to stdout
(let [obs (cm/choicemap :y 3.0)
      result (fit/fit m-conjugate [0] obs {:verbose? true})]
  (assert-equal "verbose fit still returns :exact" :exact (:method result)))

;; ===========================================================================
;; Section 7: Callback invocation
;; ===========================================================================

(println "\n== Section 7: Callback invocation ==")

(let [obs (cm/choicemap :y 3.0)
      callback-calls (atom [])
      result (fit/fit m-conjugate [0] obs {:learn [:mu]
                                            :iterations 10
                                            :lr 0.01
                                            :log-every 5
                                            :callback (fn [info]
                                                        (swap! callback-calls conj info))})]
  (assert-true "callback was invoked" (pos? (count @callback-calls)))
  (assert-true "callback info has :iter" (contains? (first @callback-calls) :iter))
  (assert-true "callback info has :loss" (contains? (first @callback-calls) :loss)))

;; ===========================================================================
;; Section 8: Learning loop (:learn)
;; ===========================================================================

(println "\n== Section 8: Learning loop ==")

(let [obs (cm/choicemap :y 3.0)
      result (fit/fit m-conjugate [0] obs {:learn [:mu]
                                            :iterations 20
                                            :lr 0.01})]
  (assert-true "learn produces :params" (some? (:params result)))
  (assert-true "learn produces :loss-history" (some? (:loss-history result)))
  (assert-true "loss-history is vector" (vector? (:loss-history result)))
  (assert-true "loss-history is non-empty" (pos? (count (:loss-history result)))))

;; ===========================================================================
;; Section 9: Edge cases
;; ===========================================================================

(println "\n== Section 9: Edge cases ==")

;; Empty observations (nil)
(let [result (fit/fit m-simple [] nil)]
  (assert-true "nil data works" (some? result))
  (assert-true "nil data has method" (keyword? (:method result)))
  (assert-true "nil data has elapsed" (pos? (:elapsed-ms result))))

;; Empty ChoiceMap
(let [result (fit/fit m-simple [] cm/EMPTY)]
  (assert-true "empty cm works" (some? result))
  (assert-true "empty cm has method" (keyword? (:method result))))

;; Unknown method throws
(let [threw? (atom false)]
  (try
    (fit/fit m-simple [] nil {:method :bogus})
    (catch :default _e
      (reset! threw? true)))
  (assert-true "unknown method throws" @threw?))

;; All-observed model (trivial exact)
(let [obs (cm/choicemap :x 1.0)
      result (fit/fit m-simple [] obs)]
  (assert-true "all-observed works" (some? result))
  (assert-true "all-observed has log-ml" (some? (:log-ml result))))

;; ===========================================================================
;; Section 10: Reproducibility with :key
;; ===========================================================================

(println "\n== Section 10: Reproducibility with :key ==")

(let [obs (cm/choicemap :y 3.0)
      key1 (rng/fresh-key 42)
      key2 (rng/fresh-key 42)
      r1 (fit/fit m-conjugate [0] obs {:method :handler-is :particles 50 :key key1})
      r2 (fit/fit m-conjugate [0] obs {:method :handler-is :particles 50 :key key2})]
  ;; Both runs with same seed should produce same log-ml
  ;; (handler-IS is stochastic, but seeded identically)
  (assert-close "same key → same log-ml" (:log-ml r1) (:log-ml r2) 0.001))

;; ===========================================================================
;; Section 11: Diagnostics content
;; ===========================================================================

(println "\n== Section 11: Diagnostics ==")

(let [obs (cm/choicemap :y 3.0)
      result (fit/fit m-conjugate [0] obs)]
  (assert-true "diagnostics has :reason" (string? (get-in result [:diagnostics :reason])))
  (assert-true "diagnostics has :n-residual" (number? (get-in result [:diagnostics :n-residual])))
  (assert-true "diagnostics has :n-latent" (number? (get-in result [:diagnostics :n-latent]))))

;; ===========================================================================
;; Section 12: End-to-end linear regression
;; ===========================================================================

(println "\n== Section 12: End-to-end linear regression ==")

(def m-linreg
  (gen [xs]
    (let [slope (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j))
               (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                      intercept) 1)))
      slope)))

;; Data: y = 2*x + 1 + noise
(let [xs [0 1 2 3 4]
      obs (cm/choicemap :y0 1.0 :y1 3.1 :y2 4.9 :y3 7.0 :y4 9.1)
      result (fit/fit m-linreg [xs] obs)]
  (assert-true "linreg has method" (keyword? (:method result)))
  (assert-true "linreg has elapsed" (pos? (:elapsed-ms result)))
  ;; The auto-selected method should produce some result
  (assert-true "linreg has posterior or trace" (or (some? (:posterior result))
                                                    (some? (:trace result)))))

;; ===========================================================================
;; Summary
;; ===========================================================================

(println (str "\n== fit_test Summary: " @pass-count "/" (+ @pass-count @fail-count) " passed =="))
(when (pos? @fail-count)
  (println (str "  " @fail-count " FAILURES")))
