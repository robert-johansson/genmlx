;; @tier fast core
(ns genmlx.mcmc-warmup-reproducibility-test
  "Regression test for genmlx-vv3t: adaptive-warmup step-size + warmup
   end-state must be reproducible under a fixed :key.

   Before the fix, dual-averaging-warmup, find-reasonable-epsilon /
   find-reasonable-mala-epsilon, and the per-algorithm warmup step-fns drew
   fresh entropy independent of the user :key, so two runs with the SAME seed
   diverged (the adapted step-size + warmup q were non-deterministic). The
   fix threads the user key through warmup (disjoint from the sampling stream),
   so a fixed seed now produces bit-identical runs, while a different seed
   produces a different run (the key actually matters)."
  (:require [cljs.test :refer [deftest is testing run-tests]]
            [genmlx.test-helpers :as h]
            [genmlx.choicemap :as cm]
            [genmlx.mlx.random :as rng]
            [genmlx.gen :refer [gen]]
            [genmlx.dist :as dist]
            [genmlx.inference.mcmc :as mcmc]))

;; ---------------------------------------------------------------------------
;; Trivial model: mu ~ N(0, 5), y ~ N(mu, 1), observe y = 2.0
;; Posterior on mu is a simple Gaussian — enough to exercise the gradient
;; warmup path without large graphs.
;; ---------------------------------------------------------------------------

(def model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 5))]
      (trace :y (dist/gaussian mu 1)))))

(def obs (cm/choicemap :y 2.0))

;; Small workload that exercises the eager (compile? false) dual-averaging
;; warmup path: ~30 burn-in steps, 3 collected samples.
(def base-opts
  {:samples 3 :burn 30 :step-size 0.05 :addresses [:mu]
   :adapt-step-size true :compile? false :device :cpu})

(defn- run-alg
  "Run an adaptive-warmup MCMC algorithm with the given key, return the
   collected sample values (first slot of each sample row) as a Clojure vector
   of JS numbers — a deterministic run summary."
  [alg-fn key]
  (let [samples (alg-fn (assoc base-opts :key key) model [] obs)]
    (mapv first samples)))

(defn- identical-runs?
  "Two run summaries are bit-identical (tolerance 0.0)."
  [a b]
  (and (= (count a) (count b))
       (every? true? (map #(h/close? %1 %2 0.0) a b))))

(defn- runs-differ?
  "Run summaries differ in at least one position."
  [a b]
  (not (identical-runs? a b)))

;; ---------------------------------------------------------------------------
;; Per-algorithm reproducibility: same seed → bit-identical, different seed →
;; different run.
;; ---------------------------------------------------------------------------

(deftest nuts-warmup-reproducible
  (testing "NUTS adaptive warmup is reproducible under a fixed key"
    (let [a (run-alg mcmc/nuts (rng/fresh-key 7))
          b (run-alg mcmc/nuts (rng/fresh-key 7))
          c (run-alg mcmc/nuts (rng/fresh-key 99))]
      (is (identical-runs? a b)
          (str "same seed → identical: " a " vs " b))
      (is (runs-differ? a c)
          (str "different seed → different: seed7=" a " seed99=" c)))))

(deftest hmc-warmup-reproducible
  (testing "HMC adaptive warmup is reproducible under a fixed key"
    (let [a (run-alg mcmc/hmc (rng/fresh-key 7))
          b (run-alg mcmc/hmc (rng/fresh-key 7))
          c (run-alg mcmc/hmc (rng/fresh-key 99))]
      (is (identical-runs? a b)
          (str "same seed → identical: " a " vs " b))
      (is (runs-differ? a c)
          (str "different seed → different: seed7=" a " seed99=" c)))))

(deftest mala-warmup-reproducible
  (testing "MALA adaptive warmup is reproducible under a fixed key"
    (let [a (run-alg mcmc/mala (rng/fresh-key 7))
          b (run-alg mcmc/mala (rng/fresh-key 7))
          c (run-alg mcmc/mala (rng/fresh-key 99))]
      (is (identical-runs? a b)
          (str "same seed → identical: " a " vs " b))
      (is (runs-differ? a c)
          (str "different seed → different: seed7=" a " seed99=" c)))))

;; ---------------------------------------------------------------------------
;; nil-key path still works (no determinism asserted — unseeded is fresh
;; entropy by design). Guards against the new key threading breaking the
;; nil-tolerant callers.
;; ---------------------------------------------------------------------------

(deftest nil-key-still-works
  (testing "adaptive warmup with :key nil runs without error for all algorithms"
    (doseq [[label alg-fn] [["nuts" mcmc/nuts] ["hmc" mcmc/hmc] ["mala" mcmc/mala]]]
      (let [samples (alg-fn (assoc base-opts :key nil) model [] obs)]
        (is (= 3 (count samples)) (str label " returns 3 samples with nil key"))
        (is (every? #(h/finite? (first %)) samples)
            (str label " produces finite samples with nil key"))))))

(run-tests)
