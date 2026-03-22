(ns genmlx.inference-vi-test
  "Phase 4.6: Variational inference tests.
   VI (ADVI) uses a mean-field Gaussian guide and optimizes ELBO via Adam.

   For the Normal-Normal model with 1D latent mu:
     Posterior: N(2.994, 0.447)
     The VI guide should converge to mu ≈ 2.994, sigma ≈ 0.447.
     ELBO should be finite and non-decreasing (on average).

   Tolerance: VI approximation is inherently biased (mean-field),
   but for 1D Normal-Normal (which IS mean-field), VI should recover
   the exact posterior. Allow 0.3 for optimization noise."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.protocols :as p]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.vi :as vi]
            [genmlx.test-helpers :as h])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Test setup
;; ---------------------------------------------------------------------------
;; VI works with a log-density function, not a model directly.
;; We build a log-density from the Normal-Normal model.

(def ^:private ys [2.8 3.1 2.9 3.2 3.0])

;; Analytical posterior
(def ^:private mu-post 2.9940119760479043)
(def ^:private sigma-post 0.4467670516087703)

(defn- normal-normal-log-density
  "log p(mu, y) = log N(mu; 0, 10) + sum log N(y_i; mu, 1).
   params: [mu] (1D array)."
  [params]
  (let [mu (mx/index params 0)
        ;; Prior: log N(mu; 0, 10)
        prior-lp (mx/multiply (mx/scalar -0.5)
                   (mx/add (mx/scalar h/LOG-2PI)
                           (mx/scalar (* 2 (js/Math.log 10)))
                           (mx/divide (mx/square mu) (mx/scalar 100.0))))
        ;; Likelihood: sum_i log N(y_i; mu, 1)
        obs-lp (reduce mx/add
                  (map (fn [y]
                         (mx/multiply (mx/scalar -0.5)
                           (mx/add (mx/scalar h/LOG-2PI)
                                   (mx/square (mx/subtract mu (mx/scalar y))))))
                       ys))]
    (mx/add prior-lp obs-lp)))

;; ==========================================================================
;; 1. VI ELBO is finite
;; ==========================================================================

(deftest vi-elbo-finite
  (testing "ELBO history contains finite values"
    (let [result (vi/vi {:iterations 200 :learning-rate 0.05
                          :elbo-samples 10
                          :key (rng/fresh-key 42)}
                         normal-normal-log-density
                         (mx/array [0.0]))]
      (is (pos? (count (:elbo-history result))) "ELBO history is non-empty")
      (is (every? js/isFinite (:elbo-history result))
          "all ELBO values are finite"))))

;; ==========================================================================
;; 2. VI converges to correct posterior mean
;; ==========================================================================

(deftest vi-mean-convergence
  (testing "VI guide mean converges to posterior mean"
    (let [result (vi/vi {:iterations 1000 :learning-rate 0.05
                          :elbo-samples 10
                          :key (rng/fresh-key 99)}
                         normal-normal-log-density
                         (mx/array [0.0]))
          vi-mu (first (mx/->clj (:mu result)))]
      ;; 1D Normal-Normal is exactly mean-field, so VI should converge
      ;; to the exact posterior mean. Allow 0.3 for optimization noise.
      (is (h/close? mu-post vi-mu 0.3)
          (str "VI mu " vi-mu " ≈ " mu-post)))))

;; ==========================================================================
;; 3. VI converges to correct posterior std
;; ==========================================================================

(deftest vi-sigma-convergence
  (testing "VI guide sigma converges to posterior sigma"
    (let [result (vi/vi {:iterations 1000 :learning-rate 0.05
                          :elbo-samples 10
                          :key (rng/fresh-key 77)}
                         normal-normal-log-density
                         (mx/array [0.0]))
          vi-sigma (first (mx/->clj (:sigma result)))]
      ;; Allow 0.3 — sigma converges slower but 1000 iters should suffice
      (is (h/close? sigma-post vi-sigma 0.3)
          (str "VI sigma " vi-sigma " ≈ " sigma-post)))))

;; ==========================================================================
;; 4. VI ELBO is non-decreasing on average
;; ==========================================================================

(deftest vi-elbo-improves
  (testing "later ELBO is higher than initial ELBO"
    (let [result (vi/vi {:iterations 300 :learning-rate 0.05
                          :elbo-samples 10
                          :key (rng/fresh-key 55)}
                         normal-normal-log-density
                         (mx/array [0.0]))
          history (:elbo-history result)]
      (when (>= (count history) 4)
        (let [early-avg (h/sample-mean (take 2 history))
              late-avg (h/sample-mean (take-last 2 history))]
          (is (> late-avg early-avg)
              (str "ELBO improved: " early-avg " → " late-avg)))))))

;; ==========================================================================
;; 5. VI sample-fn produces correct number of samples
;; ==========================================================================

(deftest vi-sample-fn
  (testing "sample-fn produces N samples"
    (let [result (vi/vi {:iterations 100 :learning-rate 0.05
                          :elbo-samples 5
                          :key (rng/fresh-key 42)}
                         normal-normal-log-density
                         (mx/array [0.0]))
          samples ((:sample-fn result) 50)]
      (is (= 50 (count samples)) "sample-fn returns 50 samples")
      (is (every? js/isFinite samples) "all samples are finite"))))

;; ==========================================================================
;; Run tests
;; ==========================================================================

(cljs.test/run-tests)
