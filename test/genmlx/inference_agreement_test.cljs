(ns genmlx.inference-agreement-test
  "Phase 4.5: Cross-algorithm agreement tests.
   Verifies that IS, MH, and HMC all converge to the same analytical
   posterior for the Normal-Normal conjugate model.

   Posterior: mu | y ~ N(2.9940, 0.4468)

   Each algorithm uses enough samples for reliable convergence.
   All posterior means should agree within their individual tolerances
   of the analytical value."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.importance :as is]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.util :as u]
            [genmlx.test-helpers :as h])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Shared model and observations
;; ---------------------------------------------------------------------------

(def normal-normal-model
  (dyn/auto-key
    (gen [ys]
      (let [mu (trace :mu (dist/gaussian 0 10))]
        (doseq [[i y] (map-indexed vector ys)]
          (trace (keyword (str "y" i))
                 (dist/gaussian mu 1)))
        mu))))

(def ys [2.8 3.1 2.9 3.2 3.0])

(def obs
  (reduce (fn [cm [i y]]
            (cm/set-choice cm [(keyword (str "y" i))] (mx/scalar y)))
          cm/EMPTY
          (map-indexed vector ys)))

;; Analytical posterior
(def ^:private mu-post 2.9940119760479043)

;; ==========================================================================
;; Cross-algorithm agreement
;; ==========================================================================

(deftest algorithms-agree-on-posterior
  (testing "IS, MH, and HMC all converge to the same posterior mean"
    (let [;; IS: weighted mean
          is-result (is/importance-sampling
                      {:samples 3000 :key (rng/fresh-key 42)}
                      normal-normal-model [ys] obs)
          {:keys [probs]} (u/normalize-log-weights (:log-weights is-result))
          is-vals (mapv (fn [tr]
                          (let [v (cm/get-choice (:choices tr) [:mu])]
                            (mx/eval! v) (mx/item v)))
                        (:traces is-result))
          is-mean (reduce + (map * probs is-vals))

          ;; MH: sample mean
          mh-traces (mcmc/mh {:samples 500 :burn 200 :thin 2
                               :selection (sel/select :mu)
                               :key (rng/fresh-key 99)}
                              normal-normal-model [ys] obs)
          mh-vals (mapv (fn [tr]
                          (let [v (cm/get-choice (:choices tr) [:mu])]
                            (mx/eval! v) (mx/item v)))
                        mh-traces)
          mh-mean (h/sample-mean mh-vals)

          ;; HMC: sample mean
          hmc-samples (mcmc/hmc {:samples 200 :burn 100 :thin 1
                                  :step-size 0.1 :leapfrog-steps 10
                                  :addresses [:mu]
                                  :key (rng/fresh-key 77)
                                  :device :cpu :compile? false}
                                 normal-normal-model [ys] obs)
          ;; HMC returns Clojure vectors of JS numbers
          hmc-vals (mapv first hmc-samples)
          hmc-mean (h/sample-mean hmc-vals)]

      ;; Each should be near the analytical posterior mean
      (is (h/close? mu-post is-mean 0.15)
          (str "IS mean " is-mean " ≈ " mu-post))
      (is (h/close? mu-post mh-mean 0.20)
          (str "MH mean " mh-mean " ≈ " mu-post))
      (is (h/close? mu-post hmc-mean 0.15)
          (str "HMC mean " hmc-mean " ≈ " mu-post))

      ;; Pairwise agreement (difference should be small)
      (is (h/close? is-mean mh-mean 0.30)
          (str "IS-MH agree: " is-mean " ≈ " mh-mean))
      (is (h/close? is-mean hmc-mean 0.30)
          (str "IS-HMC agree: " is-mean " ≈ " hmc-mean))
      (is (h/close? mh-mean hmc-mean 0.30)
          (str "MH-HMC agree: " mh-mean " ≈ " hmc-mean)))))

;; ==========================================================================
;; Run tests
;; ==========================================================================

(cljs.test/run-tests)
