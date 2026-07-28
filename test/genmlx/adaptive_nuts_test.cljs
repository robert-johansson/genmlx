;; @tier medium
(ns genmlx.adaptive-nuts-test
  "Tests for NUTS dual-averaging step-size + mass matrix adaptation,
   and HMC adapt-metric."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.dynamic :as dyn]
            [genmlx.dist :as dist]
            [genmlx.gen :refer [gen]]
            [genmlx.mlx.random :as rng]
            [genmlx.inference.mcmc :as mcmc]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- mean [xs] (/ (reduce + xs) (count xs)))

;; ---------------------------------------------------------------------------
;; Model: Gaussian with known posterior
;; Prior: mu ~ N(0, 10), Likelihood: y_i ~ N(mu, 1)
;; Posterior: mu | y ~ N(~5.097, ~0.577)
;; ---------------------------------------------------------------------------

(def model
  (gen [n]
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (dotimes [i n]
        (trace (keyword (str "y" i)) (dist/gaussian mu 1)))
      mu)))

(def obs (cm/choicemap :y0 5.0 :y1 5.5 :y2 4.8))

;; ---------------------------------------------------------------------------
;; Tests
;; ---------------------------------------------------------------------------

(deftest nuts-adapted-step-size-test
  (testing "NUTS adapted step-size is valid"
    (let [samples (mcmc/nuts
                    {:samples 50 :burn 100 :step-size 0.01
                     :addresses [:mu] :adapt-step-size true :compile? false
                     :device :cpu}
                    model [3] obs)]
      (is (= 50 (count samples)) "returns 50 samples")
      (is (every? #(not (js/isNaN (first %))) samples) "samples are finite"))))

;; SEEDED from a MEASURED 15-seed sweep (2026-07-28, RTX sm_120), the same
;; discipline as hmc-adapt-metric below and the compiled_optimizer precedent.
;; These three bands were the last unseeded Monte Carlo assertions in this file:
;; every run was a different experiment, so a regression and a bad draw looked
;; alike. They became seedable only once nuts started honouring :key for the
;; INITIAL TRACE as well as the steps (genmlx-n7du) — before that a :key here
;; would have pinned nothing.
;;
;; Measured at the SHIPPED budgets, |mean - 5.097| against the +/-0.5 band:
;;   posterior-accuracy   (300/200, eps 0.01)  15/15, worst 0.128, min uniq 145
;;   bad-initial-recovery (100/150, eps 1.0 )  15/15, worst 0.178, min uniq  69
;;   adapt-metric         (200/200, eps 0.01)  15/15, worst 0.208, min uniq  22
;; i.e. 2.4x-3.9x margin everywhere and no failures, so BUDGETS AND BANDS ARE
;; UNCHANGED — this adds reproducibility, it does not paper over a weak test.
;; Each seed below is its config's MEDIAN-error seed, not a lucky one.
(deftest nuts-posterior-accuracy-test
  (testing "NUTS posterior accuracy"
    ;; seed 371 = median error 0.0397 of the 0.5 band
    (let [samples (mcmc/nuts
                    {:samples 300 :burn 200 :step-size 0.01
                     :addresses [:mu] :adapt-step-size true :compile? false
                     :device :cpu :key (rng/fresh-key 371)}
                    model [3] obs)
          vals (mapv first samples)
          m (mean vals)
          expected 5.097]
      (is (< (js/Math.abs (- m expected)) 0.5) "posterior mean ~ 5.1"))))

(deftest nuts-bad-initial-recovery-test
  (testing "NUTS from bad initial step-size adapts"
    (let [samples (mcmc/nuts
                    ;; seed 260 = median error 0.0665 of the 0.5 band
                    {:samples 100 :burn 150 :step-size 1.0
                     :addresses [:mu] :adapt-step-size true :compile? false
                     :device :cpu :key (rng/fresh-key 260)}
                    model [3] obs)
          vals (mapv first samples)
          n-unique (count (distinct (map #(.toFixed % 2) vals)))
          m (mean vals)]
      (is (> n-unique 10) "adapted from bad init has diversity")
      (is (< (js/Math.abs (- m 5.097)) 0.5) "posterior mean correct"))))

(deftest nuts-adapt-metric-test
  (testing "NUTS adapt-metric"
    (let [samples (mcmc/nuts
                    ;; seed 297 = median error 0.0323 of the 0.5 band
                    {:samples 200 :burn 200 :step-size 0.01
                     :addresses [:mu] :adapt-step-size true :adapt-metric true
                     :compile? false :device :cpu :key (rng/fresh-key 297)}
                    model [3] obs)
          vals (mapv first samples)
          m (mean vals)]
      (is (= 200 (count samples)) "adapt-metric returns 200 samples")
      (is (every? #(not (js/isNaN (first %))) samples) "adapt-metric samples finite")
      (is (< (js/Math.abs (- m 5.097)) 0.5) "adapt-metric posterior mean ~ 5.1"))))

(deftest hmc-adapt-metric-test
  (testing "HMC adapt-metric"
    ;; Seeded + budgeted from a MEASURED sweep (2026-07-27, RTX sm_120), per the
    ;; compiled_optimizer discipline: at burn 300, 2 of 15 seeds land OUTSIDE the
    ;; ±0.7 band (0.75, 0.88 — slow-mixing adaptation, ~13% flake rate; the
    ;; battery red of 2026-07-27). At burn 600 all 15 seeds are in band (worst
    ;; |err| 0.43, median 0.028). Seed 3 is the MEDIAN-error seed (0.028), not a
    ;; lucky one; the band is unchanged. hmc honors :key for init AND chain
    ;; since 2026-07-26 — reproducibility verified bit-identical in the sweep.
    (let [samples (mcmc/hmc
                    {:samples 400 :burn 600 :step-size 0.01 :leapfrog-steps 10
                     :addresses [:mu] :adapt-step-size true :adapt-metric true
                     :compile? false :device :cpu :key (rng/fresh-key 3)}
                    model [3] obs)
          vals (mapv first samples)
          m (mean vals)]
      (is (= 400 (count samples)) "HMC adapt-metric returns 400 samples")
      (is (every? #(not (js/isNaN (first %))) samples) "HMC adapt-metric samples finite")
      (is (< (js/Math.abs (- m 5.097)) 0.7) "HMC adapt-metric posterior mean ~ 5.1"))))

(deftest default-behavior-unchanged-test
  (testing "default NUTS behavior unchanged"
    (let [samples (mcmc/nuts
                    {:samples 30 :burn 10 :step-size 0.05
                     :addresses [:mu] :compile? true :device :cpu}
                    model [3] obs)]
      (is (= 30 (count samples)) "default NUTS (no adaptation) returns 30 samples"))))

(cljs.test/run-tests)
