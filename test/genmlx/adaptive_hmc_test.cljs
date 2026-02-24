(ns genmlx.adaptive-hmc-test
  "Tests for HMC dual-averaging step-size adaptation."
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
;; Posterior: mu | y ~ N(posterior-mean, posterior-var)
;; With y = [5.0, 5.5, 4.8]: posterior-mean ≈ 5.097, posterior-sd ≈ 0.577
;; ---------------------------------------------------------------------------

(def model
  (gen [n]
    (let [mu (dyn/trace :mu (dist/gaussian 0 10))]
      (dotimes [i n]
        (dyn/trace (keyword (str "y" i)) (dist/gaussian mu 1)))
      mu)))

(def obs (cm/choicemap :y0 5.0 :y1 5.5 :y2 4.8))

(println "\n=== Adaptive HMC (Dual Averaging) Tests ===")

;; ---------------------------------------------------------------------------
;; 1. Adapted step-size is valid
;; ---------------------------------------------------------------------------

(println "\n-- adapted step-size validity --")

(let [samples (mcmc/hmc
                {:samples 50 :burn 100 :step-size 0.01 :leapfrog-steps 10
                 :addresses [:mu] :adapt-step-size true :compile? false
                 :device :cpu}
                model [3] obs)]
  (assert-true "returns 50 samples" (= 50 (count samples)))
  (assert-true "samples are finite" (every? #(not (js/isNaN (first %))) samples)))

;; ---------------------------------------------------------------------------
;; 2. Posterior mean is correct
;; ---------------------------------------------------------------------------

(println "\n-- posterior accuracy --")

(let [samples (mcmc/hmc
                {:samples 300 :burn 200 :step-size 0.01 :leapfrog-steps 10
                 :addresses [:mu] :adapt-step-size true :compile? false
                 :device :cpu}
                model [3] obs)
      vals (mapv first samples)
      m (mean vals)
      ;; Analytical posterior mean: (sum(y) / sigma^2 + mu0/sigma0^2) / (n/sigma^2 + 1/sigma0^2)
      ;; = (15.3/1 + 0/100) / (3/1 + 1/100) ≈ 5.097
      expected 5.097]
  (assert-true (str "posterior mean ≈ 5.1 (got " (.toFixed m 3) ")")
               (< (js/Math.abs (- m expected)) 0.5)))

;; ---------------------------------------------------------------------------
;; 3. Adapted vs poorly-tuned: adapted should produce finite, non-degenerate samples
;; ---------------------------------------------------------------------------

(println "\n-- adapted vs poorly-tuned --")

(let [;; Start from a bad step-size (too large) and let adaptation find a good one
      ;; We test that adapted samples are diverse and have correct posterior
      adapted-samples (mcmc/hmc
                        {:samples 200 :burn 200 :step-size 0.5 :leapfrog-steps 10
                         :addresses [:mu] :adapt-step-size true :compile? false
                         :device :cpu}
                        model [3] obs)
      adapted-vals (mapv first adapted-samples)
      adapted-unique (count (distinct (map #(.toFixed % 2) adapted-vals)))
      m (mean adapted-vals)]
  (assert-true (str "adapted from bad init has diversity (" adapted-unique " unique)")
               (> adapted-unique 20))
  (assert-true (str "adapted posterior mean correct (" (.toFixed m 3) ")")
               (< (js/Math.abs (- m 5.097)) 0.5)))

;; ---------------------------------------------------------------------------
;; 4. Works with loop compilation (compile? true)
;; ---------------------------------------------------------------------------

(println "\n-- adaptive + loop compilation --")

(let [samples (mcmc/hmc
                {:samples 50 :burn 100 :step-size 0.01 :leapfrog-steps 10
                 :addresses [:mu] :adapt-step-size true :compile? true
                 :device :cpu}
                model [3] obs)]
  (assert-true "compiled path returns 50 samples" (= 50 (count samples)))
  (assert-true "compiled samples are finite"
               (every? #(not (js/isNaN (first %))) samples)))

;; ---------------------------------------------------------------------------
;; 5. Default (adapt-step-size false) is unchanged
;; ---------------------------------------------------------------------------

(println "\n-- default behavior unchanged --")

(let [samples (mcmc/hmc
                {:samples 30 :burn 20 :step-size 0.05 :leapfrog-steps 5
                 :addresses [:mu] :compile? true :device :cpu}
                model [3] obs)]
  (assert-true "default (no adaptation) returns 30 samples" (= 30 (count samples))))

(println "\nAll adaptive HMC tests passed!")
