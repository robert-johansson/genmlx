;; @tier medium
(ns genmlx.kernel-dsl-test
  "Tests for kernel DSL: random-walk, prior, proposal, gibbs."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.kernel :as kern])
  (:require-macros [genmlx.gen :refer [gen]]))

;; Shared model: mu ~ N(0, 10), obs_i ~ N(mu, 1) for i=0..4, all obs=3.0
;; Posterior: mu ~ N(3, ~0.45)
(def model
  (dyn/auto-key
    (gen []
      (let [mu (trace :mu (dist/gaussian 0 10))]
        (mx/eval! mu)
        (let [mu-val (mx/item mu)]
          (doseq [i (range 5)]
            (trace (keyword (str "obs" i))
                       (dist/gaussian mu-val 1)))
          mu-val)))))

(def observations
  (reduce (fn [cm i]
            (cm/set-choice cm [(keyword (str "obs" i))]
                           (mx/scalar 3.0)))
          cm/EMPTY (range 5)))

(defn extract-mu-mean [traces]
  (let [mu-vals (mapv (fn [t]
                        (mx/realize (cm/get-value (cm/get-submap (:choices t) :mu))))
                      traces)]
    (/ (reduce + mu-vals) (count mu-vals))))

(def model2
  (dyn/auto-key
    (gen [xs]
      (let [slope     (trace :slope (dist/gaussian 0 10))
            intercept (trace :intercept (dist/gaussian 0 10))]
        (mx/eval! slope intercept)
        (let [s (mx/item slope) b (mx/item intercept)]
          (doseq [[j x] (map-indexed vector xs)]
            (trace (keyword (str "y" j))
                       (dist/gaussian (+ (* s x) b) 1)))
          s)))))

(deftest random-walk-single-test
  (testing "random-walk: single address"
    (let [{:keys [trace]} (p/generate model [] observations)
          k (kern/random-walk :mu 0.5)
          traces (kern/run-kernel {:samples 200 :burn 100} k trace)
          mu-mean (extract-mu-mean traces)
          ar (:acceptance-rate (meta traces))]
      (is (= 200 (count traces)) "random-walk: 200 samples")
      (is (h/close? 3.0 mu-mean 1.0) "random-walk: posterior mu near 3")
      (is (> ar 0) "random-walk: acceptance rate > 0"))))

(deftest random-walk-multi-test
  (testing "random-walk: multi-address (map form)"
    (let [xs [1.0 2.0 3.0 4.0 5.0]
          obs (reduce (fn [cm [j x]]
                        (cm/set-choice cm [(keyword (str "y" j))]
                                      (mx/scalar (+ (* 2.0 x) 1.0))))
                      cm/EMPTY (map-indexed vector xs))
          ;; Seed BOTH ends. model2 is auto-keyed at its definition, so the
          ;; starting trace was a fresh prior draw, and run-kernel got no :key
          ;; either — so both the start AND the chain varied per run and these
          ;; bands were a coin flip (genmlx-c2x9). run-kernel already accepts
          ;; :key; the same pattern is used by mh-kernel-key-reproducibility-test
          ;; below.
          ;;
          ;; The BUDGET is deliberately unchanged. Measured 20/20 seeds pass at
          ;; this 200/300 budget (RTX sm_120, 2026-07-27, genmlx-c2x9 close-out;
          ;; an earlier 10-seed sweep agreed). Worst errors across those seeds:
          ;; slope 0.354 of the +/-1.5 band, intercept 1.327 of the +/-2.0 band
          ;; — so the INTERCEPT is the binding one, at ~1.5x margin, and it is
          ;; the number to watch if this ever reddens again. 600/2000 gives 0
          ;; failures too, so the extra cost buys nothing. No seed is load-
          ;; bearing here — unlike the compiled_optimizer case, seeding alone is
          ;; sufficient and hides nothing.
          {:keys [trace]} (p/generate (dyn/with-key model2 (rng/fresh-key 20260726)) [xs] obs)
          k (kern/random-walk {:slope 0.3 :intercept 0.3})
          traces (kern/run-kernel {:samples 200 :burn 300 :key (rng/fresh-key 20260727)} k trace)
          slope-vals (mapv (fn [t] (mx/realize (cm/get-value (cm/get-submap (:choices t) :slope)))) traces)
          intercept-vals (mapv (fn [t] (mx/realize (cm/get-value (cm/get-submap (:choices t) :intercept)))) traces)
          slope-mean (/ (reduce + slope-vals) (count slope-vals))
          intercept-mean (/ (reduce + intercept-vals) (count intercept-vals))]
      (is (h/close? 2.0 slope-mean 1.5) "random-walk(map): slope near 2")
      (is (h/close? 1.0 intercept-mean 2.0) "random-walk(map): intercept near 1"))))

(deftest prior-test
  (testing "prior: resample from prior"
    (let [{:keys [trace]} (p/generate model [] observations)
          k (kern/prior :mu)
          traces (kern/run-kernel {:samples 200 :burn 100} k trace)
          mu-mean (extract-mu-mean traces)
          ar (:acceptance-rate (meta traces))]
      (is (= 200 (count traces)) "prior: 200 samples")
      (is (h/close? 3.0 mu-mean 1.0) "prior: posterior mu near 3")
      (is (> ar 0) "prior: acceptance rate > 0"))))

(def sym-proposal
  (gen [current-choices]
    (let [cur-mu (cm/get-value (cm/get-submap current-choices :mu))]
      (mx/eval! cur-mu)
      (trace :mu (dist/gaussian (mx/item cur-mu) 0.5)))))

(deftest proposal-symmetric-test
  (testing "proposal: symmetric custom proposal"
    (let [{:keys [trace]} (p/generate model [] observations)
          k (kern/proposal sym-proposal)
          traces (kern/run-kernel {:samples 200 :burn 100} k trace)
          mu-mean (extract-mu-mean traces)
          ar (:acceptance-rate (meta traces))]
      (is (= 200 (count traces)) "proposal(sym): 200 samples")
      (is (h/close? 3.0 mu-mean 1.0) "proposal(sym): posterior mu near 3")
      (is (> ar 0) "proposal(sym): acceptance rate > 0"))))

(def fwd-proposal
  (gen [current-choices]
    (let [cur-mu (cm/get-value (cm/get-submap current-choices :mu))]
      (mx/eval! cur-mu)
      (trace :mu (dist/gaussian (+ (mx/item cur-mu) 0.1) 0.5)))))

(def bwd-proposal
  (gen [current-choices]
    (let [cur-mu (cm/get-value (cm/get-submap current-choices :mu))]
      (mx/eval! cur-mu)
      (trace :mu (dist/gaussian (+ (mx/item cur-mu) 0.1) 0.5)))))

(deftest proposal-asymmetric-test
  (testing "proposal: asymmetric forward/backward"
    (let [{:keys [trace]} (p/generate model [] observations)
          k (kern/proposal fwd-proposal :backward bwd-proposal)
          traces (kern/run-kernel {:samples 200 :burn 100} k trace)
          mu-mean (extract-mu-mean traces)
          ar (:acceptance-rate (meta traces))]
      (is (= 200 (count traces)) "proposal(asym): 200 samples")
      (is (h/close? 3.0 mu-mean 1.0) "proposal(asym): posterior mu near 3")
      (is (> ar 0) "proposal(asym): acceptance rate > 0"))))

(deftest gibbs-keyword-test
  (testing "gibbs: keyword args (prior-based)"
    (let [{:keys [trace]} (p/generate model [] observations)
          k (kern/gibbs :mu)
          traces (kern/run-kernel {:samples 200 :burn 100} k trace)
          mu-mean (extract-mu-mean traces)
          ar (:acceptance-rate (meta traces))]
      (is (= 200 (count traces)) "gibbs(kw): 200 samples")
      (is (h/close? 3.0 mu-mean 1.0) "gibbs(kw): posterior mu near 3")
      (is (> ar 0) "gibbs(kw): acceptance rate > 0"))))

(deftest gibbs-std-map-test
  (testing "gibbs: std map (random-walk-based)"
    (let [xs [1.0 2.0 3.0 4.0 5.0]
          obs (reduce (fn [cm [j x]]
                        (cm/set-choice cm [(keyword (str "y" j))]
                                      (mx/scalar (+ (* 2.0 x) 1.0))))
                      cm/EMPTY (map-indexed vector xs))
          {:keys [trace]} (p/generate model2 [xs] obs)
          k (kern/gibbs {:slope 0.3 :intercept 0.3})
          ;; more samples/burn: 200/100 gave a posterior-slope estimate that
          ;; occasionally fell outside tol 1.0 (MCMC estimate noise).
          traces (kern/run-kernel {:samples 600 :burn 300} k trace)
          slope-vals (mapv (fn [t] (mx/realize (cm/get-value (cm/get-submap (:choices t) :slope)))) traces)
          slope-mean (/ (reduce + slope-vals) (count slope-vals))]
      (is (h/close? 2.0 slope-mean 1.0) "gibbs(map): slope near 2"))))

(deftest compatibility-test
  (testing "compose with chain, repeat-kernel, mix-kernels"
    (let [{:keys [trace]} (p/generate model [] observations)
          k (kern/chain (kern/random-walk :mu 0.5) (kern/prior :mu))
          traces (kern/run-kernel {:samples 50 :burn 50} k trace)
          mu-mean (extract-mu-mean traces)]
      (is (h/close? 3.0 mu-mean 1.5) "chain(rw+prior): posterior mu near 3"))

    (let [{:keys [trace]} (p/generate model [] observations)
          k (kern/repeat-kernel 3 (kern/random-walk :mu 0.5))
          traces (kern/run-kernel {:samples 50 :burn 50} k trace)
          mu-mean (extract-mu-mean traces)]
      (is (h/close? 3.0 mu-mean 1.5) "repeat(rw): posterior mu near 3"))

    (let [{:keys [trace]} (p/generate model [] observations)
          k (kern/mix-kernels [[(kern/random-walk :mu 0.5) 0.7]
                               [(kern/prior :mu) 0.3]])
          traces (kern/run-kernel {:samples 50 :burn 50} k trace)
          mu-mean (extract-mu-mean traces)]
      (is (h/close? 3.0 mu-mean 1.5) "mix(rw+prior): posterior mu near 3"))

    (let [{:keys [trace]} (p/generate model [] observations)
          k (kern/random-walk :mu 0.5)
          callback-count (atom 0)
          traces (kern/run-kernel {:samples 10 :burn 0
                                   :callback (fn [_] (swap! callback-count inc))}
                                  k trace)]
      (is (= 10 @callback-count) "run-kernel callback fires"))))

(deftest mh-kernel-key-reproducibility-test
  (testing "genmlx-vv3t: mh-kernel proposal is seeded by the THREADED key, so a chain is reproducible under a fixed :key (previously the proposal used auto-key fresh entropy)"
    (let [{:keys [trace]} (p/generate model [] observations)
          k    (kern/mh-kernel (sel/select :mu))
          mus  (fn [traces]
                 (mapv (fn [t] (mx/realize (cm/get-value (cm/get-submap (:choices t) :mu)))) traces))
          run  (fn [seed]
                 (mus (kern/run-kernel {:samples 50 :burn 20 :key (rng/fresh-key seed)} k trace)))
          c1   (run 777)
          c2   (run 777)
          c3   (run 31337)]
      (is (= c1 c2)
          "two mh-kernel chains with the same :key are bit-identical (reproducible proposal)")
      (is (not= c1 c3)
          "a different :key yields a different chain (the key actually drives the proposal)"))))

(cljs.test/run-tests)
