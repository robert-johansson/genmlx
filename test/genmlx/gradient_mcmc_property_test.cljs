(ns genmlx.gradient-mcmc-property-test
  "Property-based tests for gradient MCMC (MALA, HMC).
   Every test verifies a genuine probabilistic programming law:
   - MALA/HMC converge to the correct posterior (detailed balance)
   - Both methods agree on the same posterior (targeting equivalence)"
  (:require [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [cljs.test :as t]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.mcmc :as mcmc])
  (:require-macros [genmlx.gen :refer [gen]]
                   [clojure.test.check.clojure-test :refer [defspec]]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- finite? [x]
  (and (number? x) (js/isFinite x)))

(defn- close? [a b tol]
  (and (finite? a) (finite? b) (<= (js/Math.abs (- a b)) tol)))

;; Extract mu value from a MALA/HMC sample (vector of param values)
(defn- extract-mu [sample]
  (first sample))

;; ---------------------------------------------------------------------------
;; Model and pools
;; ---------------------------------------------------------------------------

;; Simple conjugate gaussian: prior N(0,5), likelihood N(mu,1), observe y=3
;; Prior variance = 25, likelihood variance = 1
;; Posterior: N(75/26, 25/26) ~ N(2.88, 0.96)
(def gauss-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 5))]
      (trace :y (dist/gaussian mu 1)))))

(def obs (cm/choicemap :y (mx/scalar 3.0)))
(def posterior-mean (/ 75.0 26.0))  ;; ~ 2.88

;; Key pool
(def key-pool
  (let [root (rng/fresh-key)]
    (vec (take 5 (iterate #(second (rng/split %)) root)))))
(def gen-key (gen/elements key-pool))

;; ---------------------------------------------------------------------------
;; GM1. MALA: posterior mean near analytical
;; Law: MALA converges to the correct posterior via Langevin dynamics + MH.
;; MALA returns vectors of parameter arrays (not Traces).
;; ---------------------------------------------------------------------------

(defspec mala-posterior-mean-near-analytical 5
  (prop/for-all [key gen-key]
    ;; step-size 1.0 gives ESS ~195/200 (vs ESS ~2 at 0.1)
    (let [samples (mcmc/mala {:samples 200 :burn 50 :step-size 1.0
                              :addresses [:mu] :key key :compile? false}
                             gauss-model [] obs)
          mu-vals (mapv extract-mu samples)
          mean-mu (/ (reduce + mu-vals) (count mu-vals))]
      (close? mean-mu posterior-mean 0.5))))

;; ---------------------------------------------------------------------------
;; GM2. HMC: posterior mean near analytical
;; Law: HMC converges to the correct posterior via Hamiltonian dynamics.
;; ---------------------------------------------------------------------------

(defspec hmc-posterior-mean-near-analytical 5
  (prop/for-all [key gen-key]
    (let [samples (mcmc/hmc {:samples 200 :burn 50 :step-size 1.0
                             :leapfrog-steps 5
                             :addresses [:mu] :key key :compile? false}
                            gauss-model [] obs)
          mu-vals (mapv extract-mu samples)
          mean-mu (/ (reduce + mu-vals) (count mu-vals))]
      (close? mean-mu posterior-mean 0.5))))

;; ---------------------------------------------------------------------------
;; GM3. MALA and HMC agree on posterior
;; Law: both gradient MCMC methods target the same stationary distribution.
;; ---------------------------------------------------------------------------

(defspec mala-and-hmc-posterior-means-agree 5
  (prop/for-all [key gen-key]
    (let [mala-samples (mcmc/mala {:samples 200 :burn 50 :step-size 1.0
                                   :addresses [:mu] :key key :compile? false}
                                  gauss-model [] obs)
          hmc-samples (mcmc/hmc {:samples 200 :burn 50 :step-size 1.0
                                 :leapfrog-steps 5
                                 :addresses [:mu] :key key :compile? false}
                                gauss-model [] obs)
          mala-mean (/ (reduce + (mapv extract-mu mala-samples))
                       (count mala-samples))
          hmc-mean (/ (reduce + (mapv extract-mu hmc-samples))
                      (count hmc-samples))]
      ;; Both should be near the same posterior mean
      (close? mala-mean hmc-mean 0.5))))

;; ---------------------------------------------------------------------------
;; GM4. MALA samples are finite and in reasonable range
;; Law: Langevin dynamics preserves the target measure's support -- all
;;      samples should be finite numbers in a reasonable range for the posterior.
;; ---------------------------------------------------------------------------

(defspec mala-all-samples-finite-and-in-posterior-support 10
  (prop/for-all [key gen-key]
    (let [samples (mcmc/mala {:samples 50 :burn 10 :step-size 1.0
                              :addresses [:mu] :key key :compile? false}
                             gauss-model [] obs)
          mu-vals (mapv extract-mu samples)]
      ;; All values should be finite (not NaN/Inf) and within
      ;; a reasonable range of the posterior (mean ~ 2.88, sd ~ 0.98)
      (every? (fn [v] (and (finite? v) (< (js/Math.abs v) 20))) mu-vals))))

(t/run-tests)
