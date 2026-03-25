(ns genmlx.phase-a-test
  "Tests for Phase A features: propose, custom proposal MH, mask combinator,
   vectorized update/regenerate, new distributions, unfold/switch update/regenerate."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.trace :as tr]
            [genmlx.combinators :as comb]
            [genmlx.vectorized :as vec]
            [genmlx.inference.mcmc :as mcmc])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; IPropose
;; ---------------------------------------------------------------------------

(deftest propose-distribution
  (testing "IPropose on Distribution: weight = log-prob of sample"
    (let [d (dist/gaussian 0 1)
          {:keys [choices weight retval]} (p/propose d [])]
      (mx/eval! weight retval)
      (let [manual-lp (dist/log-prob d retval)]
        (mx/eval! manual-lp)
        (is (h/close? (mx/item manual-lp) (mx/item weight) 1e-5)
            "propose weight matches manual log-prob")))))

(deftest propose-dynamic-gf
  (testing "IPropose on DynamicGF: weight = sum of log-probs"
    (let [model (gen []
                  (let [x (trace :x (dist/gaussian 0 1))]
                    x))
          {:keys [choices weight retval]} (p/propose (dyn/auto-key model) [])
          x-val (cm/get-value (cm/get-submap choices :x))
          _ (mx/eval! x-val weight)
          verify-lp (dist/log-prob (dist/gaussian 0 1) x-val)]
      (mx/eval! verify-lp)
      (is (h/close? (mx/item verify-lp) (mx/item weight) 1e-5)
          "propose weight = log-prob of sampled choice"))))

;; ---------------------------------------------------------------------------
;; IAssess
;; ---------------------------------------------------------------------------

(deftest assess-distribution
  (testing "IAssess on Distribution: known value"
    (let [d (dist/gaussian 0 1)
          {:keys [weight]} (p/assess d [] (cm/->Value (mx/scalar 0.0)))]
      (mx/eval! weight)
      (is (h/close? -0.91894 (mx/item weight) 1e-4) "assess N(0,1) at 0"))))

(deftest assess-dynamic-gf
  (testing "IAssess on DynamicGF: all choices specified"
    (let [model (gen []
                  (let [x (trace :x (dist/gaussian 0 1))
                        y (trace :y (dist/gaussian 0 1))]
                    (mx/add x y)))
          choices (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0))
          {:keys [weight]} (p/assess (dyn/auto-key model) [] choices)]
      (mx/eval! weight)
      (let [lp-x (mx/item (do (let [v (dist/log-prob (dist/gaussian 0 1) (mx/scalar 1.0))] (mx/eval! v) v)))
            lp-y (mx/item (do (let [v (dist/log-prob (dist/gaussian 0 1) (mx/scalar 2.0))] (mx/eval! v) v)))
            expected (+ lp-x lp-y)]
        (is (h/close? expected (mx/item weight) 1e-4) "assess sum of log-probs")))))

;; ---------------------------------------------------------------------------
;; New distributions
;; ---------------------------------------------------------------------------

(deftest cauchy-log-prob
  (testing "Cauchy log-prob"
    (let [d (dist/cauchy 0 1)]
      (let [lp (dist/log-prob d (mx/scalar 0.0))]
        (mx/eval! lp)
        (is (h/close? (- (js/Math.log js/Math.PI)) (mx/item lp) 1e-5)
            "cauchy(0,1) lp at 0"))
      (let [lp (dist/log-prob d (mx/scalar 1.0))]
        (mx/eval! lp)
        (is (h/close? (- (js/Math.log (* 2 js/Math.PI))) (mx/item lp) 1e-5)
            "cauchy(0,1) lp at 1")))))

(deftest cauchy-batch
  (testing "Cauchy batch sampling shape and log-prob broadcasting"
    (let [d (dist/cauchy 2 3)
          samples (dc/dist-sample-n d nil 200)
          _ (mx/eval! samples)
          lps (dc/dist-log-prob d samples)
          _ (mx/eval! lps)]
      (is (= [200] (mx/shape samples)) "cauchy batch shape")
      (is (= [200] (mx/shape lps)) "cauchy batch lp shape"))))

(deftest geometric-log-prob
  (testing "Geometric log-prob"
    (let [d (dist/geometric 0.3)]
      (doseq [k [0 1 2 5]]
        (let [lp (dist/log-prob d (mx/scalar k))]
          (mx/eval! lp)
          (let [expected (+ (* k (js/Math.log 0.7)) (js/Math.log 0.3))]
            (is (h/close? expected (mx/item lp) 0.01)
                (str "geometric lp at k=" k))))))))

(deftest binomial-log-prob
  (testing "Binomial log-prob and support"
    (let [d (dist/binomial 10 0.5)]
      (let [lp (dist/log-prob d (mx/scalar 5))]
        (mx/eval! lp)
        (is (h/close? (js/Math.log (/ 252 1024)) (mx/item lp) 0.01)
            "binomial(10,0.5) lp at 5"))
      (is (= 11 (count (dist/support d))) "binomial support size"))))

(deftest discrete-uniform-log-prob
  (testing "Discrete Uniform log-prob"
    (let [d (dist/discrete-uniform 1 6)]
      (let [lp (dist/log-prob d (mx/scalar 3))]
        (mx/eval! lp)
        (is (h/close? (- (js/Math.log 6)) (mx/item lp) 1e-5) "uniform(1,6) lp"))
      (let [lp (dist/log-prob d (mx/scalar 0))]
        (mx/eval! lp)
        (is (= ##-Inf (mx/item lp)) "uniform(1,6) lp at 0 = -Inf")))))

(deftest truncated-normal-test
  (testing "samples in bounds, lp > standard normal lp"
    (let [d (dist/truncated-normal 0 1 -1 1)
          samples (mapv (fn [_] (let [v (dist/sample d)] (mx/eval! v) (mx/item v)))
                        (range 200))]
      (is (every? #(and (>= % -1.0) (<= % 1.0)) samples)
          "all truncated-normal in [-1,1]")
      (let [tn-lp (dist/log-prob d (mx/scalar 0.0))
            n-lp (dist/log-prob (dist/gaussian 0 1) (mx/scalar 0.0))]
        (mx/eval! tn-lp n-lp)
        (is (> (mx/item tn-lp) (mx/item n-lp)) "truncated lp > standard lp at 0")))))

(deftest truncated-normal-batch
  (testing "batch sampling respects bounds"
    (let [d (dist/truncated-normal 0 1 -2 2)
          batch (dc/dist-sample-n d nil 500)]
      (mx/eval! batch)
      (let [vals (mx/->clj batch)]
        (is (every? #(and (>= % -2.0) (<= % 2.0)) vals) "all batch in [-2,2]")))))

(deftest inv-gamma-test
  (testing "inv-gamma(3,2) mean ~ 1"
    (let [d (dist/inv-gamma 3 2)
          samples (mapv (fn [_] (let [v (dist/sample d)] (mx/eval! v) (mx/item v)))
                        (range 2000))
          mean (/ (reduce + samples) (count samples))]
      (is (every? pos? samples) "all inv-gamma > 0")
      (is (h/close? 1.0 mean 0.15) "inv-gamma mean near 1"))))

;; ---------------------------------------------------------------------------
;; Unfold update
;; ---------------------------------------------------------------------------

(deftest unfold-update-test
  (testing "score changes when constraint changes"
    (let [step (gen [t state]
                 (let [next (trace :x (dist/gaussian state 0.1))]
                   (mx/eval! next)
                   (mx/item next)))
          unfold (comb/unfold-combinator (dyn/auto-key step))
          trace (p/simulate unfold [3 0.0])
          new-constraints (cm/set-choice cm/EMPTY [1] (cm/choicemap :x (mx/scalar 5.0)))
          {:keys [trace weight discard]} (p/update unfold trace new-constraints)]
      (is (instance? tr/Trace trace) "unfold update trace exists")
      (mx/eval! weight)
      (is (js/isFinite (mx/item weight)) "unfold update weight is finite")
      (is (not= 0.0 (mx/item weight)) "unfold update weight != 0"))))

(deftest switch-update-regenerate
  (testing "switch update and regenerate"
    (let [b0 (gen [] (let [x (trace :x (dist/gaussian 0 1))] (mx/eval! x) (mx/item x)))
          b1 (gen [] (let [x (trace :x (dist/gaussian 10 1))] (mx/eval! x) (mx/item x)))
          sw (comb/switch-combinator (dyn/auto-key b0) (dyn/auto-key b1))
          trace (p/simulate sw [0])
          {:keys [trace weight]} (p/update sw trace (cm/choicemap :x (mx/scalar 2.0)))
          x-val (cm/get-value (cm/get-submap (:choices trace) :x))]
      (mx/eval! x-val weight)
      (is (h/close? 2.0 (mx/item x-val) 1e-5) "switch update sets x=2")
      (is (js/isFinite (mx/item weight)) "switch update weight finite"))))

;; ---------------------------------------------------------------------------
;; Vectorized regenerate
;; ---------------------------------------------------------------------------

(deftest vectorized-regenerate
  (testing "changes selected addresses"
    (let [model (gen []
                  (trace :x (dist/gaussian 0 1))
                  (trace :y (dist/gaussian 0 1)))
          n 50
          key1 (rng/fresh-key 42)
          key2 (rng/fresh-key 99)
          vtrace (dyn/vsimulate model [] n key1)
          old-x (cm/get-value (cm/get-submap (:choices vtrace) :x))
          old-y (cm/get-value (cm/get-submap (:choices vtrace) :y))
          _ (mx/eval! old-x old-y)
          {:keys [vtrace]} (dyn/vregenerate model vtrace (sel/select :x) key2)
          new-x (cm/get-value (cm/get-submap (:choices vtrace) :x))
          new-y (cm/get-value (cm/get-submap (:choices vtrace) :y))
          _ (mx/eval! new-x new-y)]
      (is (not= (mx/->clj old-x) (mx/->clj new-x)) "vregenerate changed :x")
      (is (= (mx/->clj old-y) (mx/->clj new-y)) "vregenerate kept :y"))))

;; ---------------------------------------------------------------------------
;; Mask combinator
;; ---------------------------------------------------------------------------

(deftest mask-combinator-test
  (testing "active vs inactive"
    (let [inner (gen [] (let [x (trace :x (dist/gaussian 5 0.1))] (mx/eval! x) (mx/item x)))
          masked (comb/mask-combinator (dyn/auto-key inner))
          active-trace (p/simulate masked [true])
          inactive-trace (p/simulate masked [false])]
      (is (< (js/Math.abs (- (:retval active-trace) 5)) 2) "active retval near 5")
      (is (cm/has-value? (cm/get-submap (:choices active-trace) :x)) "active has :x choice")
      (mx/eval! (:score active-trace))
      (is (js/isFinite (mx/item (:score active-trace))) "active score is finite")
      (is (nil? (:retval inactive-trace)) "inactive retval nil")
      (is (not (cm/has-value? (cm/get-submap (:choices inactive-trace) :x))) "inactive choices empty")
      (mx/eval! (:score inactive-trace))
      (is (h/close? 0.0 (mx/item (:score inactive-trace)) 1e-10) "inactive score = 0"))))

;; ---------------------------------------------------------------------------
;; Gibbs sampling
;; ---------------------------------------------------------------------------

(deftest gibbs-posterior
  (testing "posterior on binary latent"
    (let [model (gen []
                  (let [z (trace :z (dist/bernoulli 0.5))]
                    (mx/eval! z)
                    (let [z-val (mx/item z)
                          mu (if (> z-val 0.5) 5 -5)]
                      (trace :x (dist/gaussian mu 1))
                      z-val)))
          observations (cm/choicemap :x (mx/scalar 4.5))
          traces (mcmc/gibbs
                   {:samples 100 :burn 50}
                   model [] observations
                   [{:addr :z :support [(mx/scalar 0.0) (mx/scalar 1.0)]}])
          z-vals (mapv (fn [t]
                          (mx/realize (cm/get-value (cm/get-submap (:choices t) :z))))
                        traces)
          z-mean (/ (reduce + z-vals) (count z-vals))]
      (is (> z-mean 0.8) "gibbs: z concentrates near 1 (>0.8)"))))

;; ---------------------------------------------------------------------------
;; Mixture distribution
;; ---------------------------------------------------------------------------

(deftest mixture-log-prob
  (testing "log-prob is logsumexp"
    (let [c1 (dist/gaussian -5 1)
          c2 (dist/gaussian 5 1)
          mix (dc/mixture [c1 c2] (mx/array [(js/Math.log 0.3) (js/Math.log 0.7)]))
          v (mx/scalar 5.0)
          lp (dc/dist-log-prob mix v)
          _ (mx/eval! lp)
          expected (js/Math.log (* 0.7 (/ 1.0 (js/Math.sqrt (* 2 js/Math.PI)))))]
      (is (h/close? expected (mx/item lp) 0.01) "mixture lp at v=5"))))

(cljs.test/run-tests)
