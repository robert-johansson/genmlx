(ns genmlx.adaptive-nuts-test
  "Tests for NUTS dual-averaging step-size + mass matrix adaptation,
   and HMC adapt-metric."
  (:require [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.dynamic :as dyn]
            [genmlx.dist :as dist]
            [genmlx.gen :refer [gen]]
            [genmlx.inference.mcmc :as mcmc]))

;; ---------------------------------------------------------------------------
;; Test helpers
;; ---------------------------------------------------------------------------

(defn- assert-true [desc pred]
  (println (str "  " (if pred "PASS" "FAIL") ": " desc))
  (when-not pred (throw (js/Error. (str "FAIL: " desc)))))

(defn- mean [xs] (/ (reduce + xs) (count xs)))

;; ---------------------------------------------------------------------------
;; Model: Gaussian with known posterior
;; Prior: mu ~ N(0, 10), Likelihood: y_i ~ N(mu, 1)
;; With y = [5.0, 5.5, 4.8]: posterior-mean ≈ 5.097, posterior-sd ≈ 0.577
;; ---------------------------------------------------------------------------

(def model
  (gen [n]
    (let [mu (dyn/trace :mu (dist/gaussian 0 10))]
      (dotimes [i n]
        (dyn/trace (keyword (str "y" i)) (dist/gaussian mu 1)))
      mu)))

(def obs (cm/choicemap :y0 5.0 :y1 5.5 :y2 4.8))

(println "\n=== Adaptive NUTS Tests ===")

;; ---------------------------------------------------------------------------
;; 1. NUTS adapted step-size is valid
;; ---------------------------------------------------------------------------

(println "\n-- NUTS adapted step-size validity --")

(let [samples (mcmc/nuts
                {:samples 50 :burn 100 :step-size 0.01
                 :addresses [:mu] :adapt-step-size true :compile? false
                 :device :cpu}
                model [3] obs)]
  (assert-true "returns 50 samples" (= 50 (count samples)))
  (assert-true "samples are finite" (every? #(not (js/isNaN (first %))) samples)))

;; ---------------------------------------------------------------------------
;; 2. NUTS posterior accuracy
;; ---------------------------------------------------------------------------

(println "\n-- NUTS posterior accuracy --")

(let [samples (mcmc/nuts
                {:samples 300 :burn 200 :step-size 0.01
                 :addresses [:mu] :adapt-step-size true :compile? false
                 :device :cpu}
                model [3] obs)
      vals (mapv first samples)
      m (mean vals)
      expected 5.097]
  (assert-true (str "posterior mean ≈ 5.1 (got " (.toFixed m 3) ")")
               (< (js/Math.abs (- m expected)) 0.5)))

;; ---------------------------------------------------------------------------
;; 3. NUTS from bad initial step-size — adaptation recovers
;; ---------------------------------------------------------------------------

(println "\n-- NUTS from bad initial (eps=1.0) --")

(let [samples (mcmc/nuts
                {:samples 100 :burn 150 :step-size 1.0
                 :addresses [:mu] :adapt-step-size true :compile? false
                 :device :cpu}
                model [3] obs)
      vals (mapv first samples)
      n-unique (count (distinct (map #(.toFixed % 2) vals)))
      m (mean vals)]
  (assert-true (str "adapted from bad init has diversity (" n-unique " unique)")
               (> n-unique 10))
  (assert-true (str "posterior mean correct (" (.toFixed m 3) ")")
               (< (js/Math.abs (- m 5.097)) 0.5)))

;; ---------------------------------------------------------------------------
;; 4. NUTS adapt-metric — samples finite, posterior correct
;; ---------------------------------------------------------------------------

(println "\n-- NUTS adapt-metric --")

(let [samples (mcmc/nuts
                {:samples 200 :burn 200 :step-size 0.01
                 :addresses [:mu] :adapt-step-size true :adapt-metric true
                 :compile? false :device :cpu}
                model [3] obs)
      vals (mapv first samples)
      m (mean vals)]
  (assert-true "adapt-metric returns 200 samples" (= 200 (count samples)))
  (assert-true "adapt-metric samples finite" (every? #(not (js/isNaN (first %))) samples))
  (assert-true (str "adapt-metric posterior mean ≈ 5.1 (got " (.toFixed m 3) ")")
               (< (js/Math.abs (- m 5.097)) 0.5)))

;; ---------------------------------------------------------------------------
;; 5. HMC adapt-metric — verify it works
;; ---------------------------------------------------------------------------

(println "\n-- HMC adapt-metric --")

(let [samples (mcmc/hmc
                {:samples 400 :burn 300 :step-size 0.01 :leapfrog-steps 10
                 :addresses [:mu] :adapt-step-size true :adapt-metric true
                 :compile? false :device :cpu}
                model [3] obs)
      vals (mapv first samples)
      m (mean vals)]
  (assert-true "HMC adapt-metric returns 400 samples" (= 400 (count samples)))
  (assert-true "HMC adapt-metric samples finite" (every? #(not (js/isNaN (first %))) samples))
  (assert-true (str "HMC adapt-metric posterior mean ≈ 5.1 (got " (.toFixed m 3) ")")
               (< (js/Math.abs (- m 5.097)) 0.7)))

;; ---------------------------------------------------------------------------
;; 6. Default behavior unchanged (no adaptation flags)
;; ---------------------------------------------------------------------------

(println "\n-- default behavior unchanged --")

(let [samples (mcmc/nuts
                {:samples 30 :burn 10 :step-size 0.05
                 :addresses [:mu] :compile? true :device :cpu}
                model [3] obs)]
  (assert-true "default NUTS (no adaptation) returns 30 samples" (= 30 (count samples))))

(println "\nAll adaptive NUTS tests passed!")
