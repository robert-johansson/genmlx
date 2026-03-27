(ns genmlx.tutorial.ch02-test
  "Test file for Tutorial Chapter 2: Conditioning — The Core Trick.
   Every code listing in the chapter has a corresponding test here."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.importance :as importance])
  (:require-macros [genmlx.gen :refer [gen]]))

(def pass (atom 0))
(def fail (atom 0))

(defn assert-true [msg pred]
  (if pred
    (do (swap! pass inc) (println "  PASS:" msg))
    (do (swap! fail inc) (println "  FAIL:" msg))))

(defn assert-close [msg expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (do (swap! pass inc) (println "  PASS:" msg))
      (do (swap! fail inc) (println "  FAIL:" msg (str "expected=" expected " actual=" actual " diff=" diff))))))

;; Reuse the linear model from Chapter 1
(def linear-model
  (gen [xs]
    (let [slope     (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j))
               (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                      intercept) 1)))
      slope)))

(def xs [1.0 2.0 3.0])

;; ============================================================
;; Listing 2.1: Building choice maps
;; ============================================================
(println "\n== Listing 2.1: building choice maps ==")

(let [obs (cm/choicemap :y0 (mx/scalar 2.5) :y1 (mx/scalar 4.5) :y2 (mx/scalar 6.5))]
  (assert-true "choicemap creates a Node" (instance? cm/Node obs))
  (assert-true "has :y0" (cm/has-value? (cm/get-submap obs :y0)))
  (assert-true "has :y1" (cm/has-value? (cm/get-submap obs :y1)))
  (assert-true "has :y2" (cm/has-value? (cm/get-submap obs :y2)))
  (assert-close ":y0 value" 2.5 (mx/item (cm/get-choice obs [:y0])) 0.001))

;; ============================================================
;; Listing 2.2: generate with observations
;; ============================================================
(println "\n== Listing 2.2: generate with observations ==")

(let [model (dyn/auto-key linear-model)
      obs (cm/choicemap :y0 (mx/scalar 2.5) :y1 (mx/scalar 4.5) :y2 (mx/scalar 6.5))
      {:keys [trace weight]} (p/generate model [xs] obs)]
  (assert-true "generate returns a trace" (some? trace))
  (assert-true "generate returns a weight" (some? weight))
  (assert-true "weight is finite" (js/Number.isFinite (mx/item weight)))
  (assert-true "score is finite" (js/Number.isFinite (mx/item (:score trace))))
  ;; Observed values are constrained
  (assert-close "y0 is constrained to 2.5"
                2.5 (mx/item (cm/get-choice (:choices trace) [:y0])) 0.001)
  (assert-close "y1 is constrained to 4.5"
                4.5 (mx/item (cm/get-choice (:choices trace) [:y1])) 0.001)
  ;; Latent values are sampled (not constrained)
  (assert-true "slope was sampled (exists)" (cm/has-value? (cm/get-submap (:choices trace) :slope)))
  (assert-true "intercept was sampled" (cm/has-value? (cm/get-submap (:choices trace) :intercept))))

;; ============================================================
;; Listing 2.3: The weight is log p(observations | latents)
;; ============================================================
(println "\n== Listing 2.3: weight meaning ==")

(let [model (dyn/auto-key linear-model)
      obs (cm/choicemap :y0 (mx/scalar 2.5) :y1 (mx/scalar 4.5) :y2 (mx/scalar 6.5))
      {:keys [trace weight]} (p/generate model [xs] obs)
      w (mx/item weight)]
  (assert-true "weight is not zero (observations have probability)" (not= 0.0 w))
  (assert-true "weight is negative (log-space)" (< w 0))
  ;; Weight should be sum of log-probs of observed y values under their conditional distributions
  ;; We can't easily decompose it here, but we can check it's reasonable
  (assert-true "weight is not -Inf" (> w -1000)))

;; ============================================================
;; Listing 2.4: Same model, different interpretation
;; ============================================================
(println "\n== Listing 2.4: simulate vs generate — same model ==")

(let [model (dyn/auto-key linear-model)
      obs (cm/choicemap :y0 (mx/scalar 2.5) :y1 (mx/scalar 4.5) :y2 (mx/scalar 6.5))
      ;; Simulate: all addresses sampled
      sim-trace (p/simulate model [xs])
      ;; Generate: y0,y1,y2 constrained; slope,intercept sampled
      {:keys [trace weight]} (p/generate model [xs] obs)]
  ;; Both traces have the same addresses
  (assert-true "simulate has :slope" (cm/has-value? (cm/get-submap (:choices sim-trace) :slope)))
  (assert-true "generate has :slope" (cm/has-value? (cm/get-submap (:choices trace) :slope)))
  ;; But generate's y values are fixed
  (let [sim-y0 (mx/item (cm/get-choice (:choices sim-trace) [:y0]))
        gen-y0 (mx/item (cm/get-choice (:choices trace) [:y0]))]
    (assert-true "simulate y0 is random" (number? sim-y0))
    (assert-close "generate y0 is constrained" 2.5 gen-y0 0.001)))

;; ============================================================
;; Listing 2.5: Importance sampling by hand
;; ============================================================
(println "\n== Listing 2.5: importance sampling by hand ==")

(let [model (dyn/auto-key linear-model)
      obs (cm/choicemap :y0 (mx/scalar 2.5) :y1 (mx/scalar 4.5) :y2 (mx/scalar 6.5))
      n 50
      results (mapv (fn [_] (p/generate model [xs] obs)) (range n))
      log-weights (mapv #(mx/item (:weight %)) results)
      ;; Normalize weights
      max-w (apply max log-weights)
      unnorm (mapv #(js/Math.exp (- % max-w)) log-weights)
      total (reduce + unnorm)
      weights (mapv #(/ % total) unnorm)]
  (assert-true "got 50 results" (= n (count results)))
  (assert-true "all weights finite" (every? js/Number.isFinite log-weights))
  (assert-close "normalized weights sum to 1" 1.0 (reduce + weights) 0.001)
  ;; Weighted mean of slope
  (let [slopes (mapv #(mx/item (cm/get-choice (:choices (:trace %)) [:slope])) results)
        weighted-mean (reduce + (map * slopes weights))]
    (assert-true "weighted slope is finite" (js/Number.isFinite weighted-mean))
    (assert-true "weighted slope is reasonable (between -20 and 20)"
                 (and (> weighted-mean -20) (< weighted-mean 20)))))

;; ============================================================
;; Listing 2.6: Built-in importance sampling
;; ============================================================
(println "\n== Listing 2.6: importance/importance-sampling ==")

(let [model linear-model
      obs (cm/choicemap :y0 (mx/scalar 2.5) :y1 (mx/scalar 4.5) :y2 (mx/scalar 6.5))
      {:keys [traces log-weights log-ml-estimate]}
      (importance/importance-sampling {:samples 100} model [xs] obs)]
  (assert-true "returns traces" (= 100 (count traces)))
  (assert-true "returns log-weights" (= 100 (count log-weights)))
  (assert-true "log-ML is finite" (js/Number.isFinite (mx/item log-ml-estimate)))
  (assert-true "log-ML is negative" (< (mx/item log-ml-estimate) 0)))

;; ============================================================
;; Listing 2.7: Log marginal likelihood
;; ============================================================
(println "\n== Listing 2.7: log marginal likelihood ==")

(let [model linear-model
      obs (cm/choicemap :y0 (mx/scalar 2.5) :y1 (mx/scalar 4.5) :y2 (mx/scalar 6.5))
      r1 (importance/importance-sampling {:samples 200} model [xs] obs)
      r2 (importance/importance-sampling {:samples 200} model [xs] obs)
      lml1 (mx/item (:log-ml-estimate r1))
      lml2 (mx/item (:log-ml-estimate r2))]
  (assert-true "two estimates are similar (within 3.0)"
               (< (js/Math.abs (- lml1 lml2)) 3.0))
  (assert-true "log-ML is in reasonable range" (and (> lml1 -30) (< lml1 0))))

;; ============================================================
;; Listing 2.8: Prior vs posterior
;; ============================================================
(println "\n== Listing 2.8: prior vs posterior ==")

(let [model (dyn/auto-key linear-model)
      obs (cm/choicemap :y0 (mx/scalar 2.5) :y1 (mx/scalar 4.5) :y2 (mx/scalar 6.5))
      ;; Prior samples (simulate)
      prior-slopes (mapv (fn [_]
                           (mx/item (cm/get-choice (:choices (p/simulate model [xs])) [:slope])))
                         (range 50))
      ;; Posterior samples (generate + weight)
      posterior-results (mapv (fn [_] (p/generate model [xs] obs)) (range 100))
      post-log-weights (mapv #(mx/item (:weight %)) posterior-results)
      post-slopes (mapv #(mx/item (cm/get-choice (:choices (:trace %)) [:slope])) posterior-results)
      ;; Normalize posterior weights
      max-w (apply max post-log-weights)
      unnorm (mapv #(js/Math.exp (- % max-w)) post-log-weights)
      total (reduce + unnorm)
      post-weights (mapv #(/ % total) unnorm)
      ;; Weighted posterior mean
      prior-mean (/ (reduce + prior-slopes) (count prior-slopes))
      post-mean (reduce + (map * post-slopes post-weights))]
  (assert-true "prior mean is near 0 (broad prior)" (< (js/Math.abs prior-mean) 5))
  ;; With data y=[2.5, 4.5, 6.5] at x=[1,2,3], the true slope is ~2.0
  (assert-true "posterior mean is closer to 2 than prior mean"
               (< (js/Math.abs (- post-mean 2.0))
                  (js/Math.abs (- prior-mean 2.0)))))

;; ============================================================
;; Listing 2.9: Nested choice maps (for splice)
;; ============================================================
(println "\n== Listing 2.9: nested choice maps ==")

(let [nested (cm/choicemap :params (cm/choicemap :slope (mx/scalar 2.0)
                                                  :intercept (mx/scalar 1.0)))]
  (assert-true "nested choicemap is a Node" (instance? cm/Node nested))
  (let [params-sub (cm/get-submap nested :params)]
    (assert-true "params submap exists" (some? params-sub))
    (assert-true "params has :slope" (cm/has-value? (cm/get-submap params-sub :slope)))
    (assert-close "slope value" 2.0
                  (mx/item (cm/get-choice nested [:params :slope])) 0.001)))

;; Also test cm/from-map for convenience
(let [from-map (cm/from-map {:a {:b 3.0} :c 5.0})]
  (assert-true "from-map creates structure" (instance? cm/Node from-map))
  (assert-true "nested :a :b accessible"
               (= 3.0 (cm/get-choice from-map [:a :b]))))

;; ============================================================
;; Summary
;; ============================================================
(println (str "\n== Chapter 2 tests: " @pass " PASS, " @fail " FAIL =="))
(when (pos? @fail) (js/process.exit 1))
