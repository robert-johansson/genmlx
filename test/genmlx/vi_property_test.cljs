(ns genmlx.vi-property-test
  "Property-based tests for variational inference using test.check.
   Every test verifies a genuine algebraic law:
   - sigma > 0 (positivity of scale parameters)
   - ELBO <= true log-marginal (variational lower bound)
   - IWELBO(K=1) = ELBO (degenerate case)
   - ELBO improves over training (gradient ascent)
   - VI posterior mean near analytical posterior"
  (:require [clojure.test.check :as tc]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.vi :as vi]
            [genmlx.inference.util :as u])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Test infrastructure
;; ---------------------------------------------------------------------------

(def ^:private pass-count (volatile! 0))
(def ^:private fail-count (volatile! 0))

(defn- report-result [name result]
  (if (:pass? result)
    (do (vswap! pass-count inc)
        (println "  PASS:" name (str "(" (:num-tests result) " trials)")))
    (do (vswap! fail-count inc)
        (println "  FAIL:" name)
        (println "    seed:" (:seed result))
        (when-let [s (get-in result [:shrunk :smallest])]
          (println "    shrunk:" s)))))

(defn- check [name prop & {:keys [num-tests] :or {num-tests 50}}]
  (let [result (tc/quick-check num-tests prop)]
    (report-result name result)))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- finite? [x]
  (and (number? x) (js/isFinite x)))

(defn- close? [a b tol]
  (and (finite? a) (finite? b) (<= (js/Math.abs (- a b)) tol)))

;; ---------------------------------------------------------------------------
;; Setup: Conjugate Gaussian model with known log-marginal
;;
;; Model: prior N(0, 10), likelihood N(mu, 1), observe y = 3.0
;; True log P(y=3) = log-prob(N(0, sqrt(101)), 3.0)
;; Analytical posterior: N(mu_post, sigma_post^2)
;;   mu_post = 3.0 * 100 / 101 = 300/101 ~ 2.97
;;   sigma_post^2 = 100/101 ~ 0.99
;; ---------------------------------------------------------------------------

;; Log-density for VI (takes [1]-shaped param array, returns scalar)
;; Must include normalization constants for ELBO to be comparable to true log-marginal
(def log-2pi (js/Math.log (* 2.0 js/Math.PI)))

(def vi-log-density
  "Log joint density: log p(mu) + log p(y=3 | mu) (fully normalized)"
  (fn [z]
    (let [mu (mx/index z 0)
          ;; log p(mu) = log N(mu; 0, 10) = -0.5*log(2*pi*100) - 0.5*mu^2/100
          log-prior (mx/subtract
                      (mx/multiply (mx/scalar -0.5)
                                   (mx/divide (mx/square mu) (mx/scalar 100.0)))
                      (mx/scalar (* 0.5 (js/Math.log (* 2.0 js/Math.PI 100.0)))))
          ;; log p(y=3|mu) = log N(3; mu, 1) = -0.5*log(2*pi) - 0.5*(3-mu)^2
          log-lik   (mx/subtract
                      (mx/multiply (mx/scalar -0.5)
                                   (mx/square (mx/subtract mu (mx/scalar 3.0))))
                      (mx/scalar (* 0.5 log-2pi)))]
      (mx/add log-prior log-lik))))

(def vi-init-params (mx/array [0.0]))

;; True log-marginal: log N(3; 0, sqrt(101))
;; = -0.5 * log(2*pi*101) - 0.5 * 9/101
(def true-log-ml
  (let [log-2pi-101 (js/Math.log (* 2.0 js/Math.PI 101.0))]
    (- (* -0.5 log-2pi-101) (* 0.5 (/ 9.0 101.0)))))

;; Model for vi-from-model
(def gauss-model
  (dyn/auto-key
    (gen []
      (let [mu (trace :mu (dist/gaussian 0 10))
            y  (trace :y (dist/gaussian mu 1))]
        (mx/eval! mu)
        (mx/item mu)))))

(def gauss-obs (cm/choicemap :y (mx/scalar 3.0)))

(def key-pool (mapv #(rng/fresh-key %) [42 99 123 7 255]))
(def gen-key (gen/elements key-pool))

;; Simple 2D Gaussian for sigma > 0 tests
(def simple-log-density
  (fn [z]
    (mx/negative (mx/multiply (mx/scalar 0.5) (mx/sum (mx/square z))))))

(def simple-init (mx/array [1.0 1.0]))

(println "\n=== VI Property-Based Tests ===\n")

;; ---------------------------------------------------------------------------
;; Law: sigma > 0 (positivity constraint on variational scale)
;; sigma = exp(log_sigma), which is always positive
;; ---------------------------------------------------------------------------

(println "-- sigma positivity --")

(check "vi: sigma > 0"
  (prop/for-all [k gen-key]
    (let [result (vi/vi {:iterations 30 :learning-rate 0.05 :elbo-samples 3 :key k}
                         simple-log-density simple-init)
          sigma (:sigma result)]
      (mx/eval! sigma)
      (every? #(> % 0) (mx/->clj sigma))))
  :num-tests 20)

(check "compiled-vi: sigma > 0"
  (prop/for-all [k gen-key]
    (let [result (vi/compiled-vi {:iterations 30 :learning-rate 0.05
                                   :elbo-samples 3 :key k :device :cpu}
                                  simple-log-density simple-init)
          sigma (:sigma result)]
      (mx/eval! sigma)
      (every? #(> % 0) (mx/->clj sigma))))
  :num-tests 15)

(check "vi-from-model: sigma > 0"
  (prop/for-all [k gen-key]
    (let [result (vi/vi-from-model
                   {:iterations 20 :learning-rate 0.05 :elbo-samples 3 :key k}
                   gauss-model [] gauss-obs [:mu])
          sigma (:sigma result)]
      (mx/eval! sigma)
      (every? #(> % 0) (mx/->clj sigma))))
  :num-tests 10)

;; ---------------------------------------------------------------------------
;; Law: ELBO <= log P(data) (variational lower bound)
;; ---------------------------------------------------------------------------

(println "\n-- ELBO bound --")

(check "ELBO <= true log-marginal"
  (prop/for-all [k gen-key]
    (let [result (vi/vi {:iterations 200 :learning-rate 0.05 :elbo-samples 10 :key k}
                         vi-log-density vi-init-params)
          history (:elbo-history result)
          ;; Use median of last few ELBO values for stability
          n-last (min 5 (count history))
          last-few (take-last n-last history)
          sorted (sort last-few)
          median-elbo (nth sorted (quot (count sorted) 2))]
      ;; ELBO is a lower bound; allow 1.5 slack for MC estimation noise
      (<= median-elbo (+ true-log-ml 1.5))))
  :num-tests 15)

;; ---------------------------------------------------------------------------
;; Law: IWELBO(K=1) = ELBO (degenerate importance weighting)
;; ---------------------------------------------------------------------------

(println "\n-- IWELBO degenerate case --")

(check "IWELBO(K=1) = ELBO"
  (prop/for-all [k gen-key]
    (let [;; Log-p and log-q at same point
          log-p vi-log-density
          log-q (fn [z] (mx/multiply (mx/scalar -0.5) (mx/sum (mx/square z))))
          elbo-fn (vi/elbo-objective log-p log-q)
          iwelbo-fn (vi/iwelbo-objective log-p log-q)
          ;; Generate K=1 sample
          sample (rng/normal k [1 1])
          elbo-val (mx/item (mx/tidy-materialize #(elbo-fn sample)))
          iwelbo-val (mx/item (mx/tidy-materialize #(iwelbo-fn sample)))]
      (close? elbo-val iwelbo-val 0.01)))
  :num-tests 30)

;; ---------------------------------------------------------------------------
;; Law: gradient ascent increases ELBO
;; ---------------------------------------------------------------------------

(println "\n-- ELBO improvement --")

(check "ELBO improves over training"
  (prop/for-all [k gen-key]
    (let [result (vi/vi {:iterations 200 :learning-rate 0.05 :elbo-samples 10 :key k}
                         vi-log-density vi-init-params)
          history (:elbo-history result)]
      (when (>= (count history) 4)
        ;; Compare average of first quarter vs last quarter for robustness
        (let [n (count history)
              quarter (max 1 (quot n 4))
              early-mean (/ (reduce + (take quarter history)) quarter)
              late-mean  (/ (reduce + (take-last quarter history)) quarter)]
          (> late-mean early-mean)))))
  :num-tests 15)

;; ---------------------------------------------------------------------------
;; Law: VI converges to (near) true posterior
;; Analytical posterior: N(300/101, 100/101), mu_post ~ 2.97
;; ---------------------------------------------------------------------------

(println "\n-- VI posterior convergence --")

(check "VI posterior mean near analytical"
  (prop/for-all [k gen-key]
    (let [result (vi/vi {:iterations 200 :learning-rate 0.05 :elbo-samples 10 :key k}
                         vi-log-density vi-init-params)
          mu (:mu result)]
      (mx/eval! mu)
      (let [mu-val (first (mx/->clj mu))]
        (close? mu-val 2.97 1.5))))
  :num-tests 15)

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n=== VI Property Tests Complete: "
              @pass-count " passed, " @fail-count " failed ==="))
