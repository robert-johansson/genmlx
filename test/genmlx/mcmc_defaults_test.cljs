;; @tier fast core
(ns genmlx.mcmc-defaults-test
  "genmlx-7ca0: mcmc default selection, tensor-score layout, acceptance-rate
   denominator.

   Independent oracles only: closed-form Gaussian posteriors / joint
   log-densities computed with host Math, observation-invariance checks, and
   exact step counting — never the code path under test."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.dist :as dist]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.kernel :as kern]
            [genmlx.inference.util :as u])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn- ->num
  "Realize an MLX scalar or pass a plain number through (constraint values
   are stored as plain numbers in choicemaps)."
  [v]
  (if (number? v) v (mx/realize v)))

(defn- gaussian-lp
  "Closed-form N(v | mu, sigma) log-density (host Math — independent oracle)."
  [v mu sigma]
  (- (* -0.5 (Math/pow (/ (- v mu) sigma) 2))
     (Math/log sigma)
     (* 0.5 (Math/log (* 2 Math/PI)))))

;; ============================================================
;; 1. sel/from-choicemap + complement (pure selection algebra)
;; ============================================================
(deftest test-from-choicemap-selection
  (testing "from-choicemap selects observed leaves; complement is the latents"
    (let [obs (cm/choicemap :y 2.0 :sub {:z 1.0})
          obs-sel (sel/from-choicemap obs)
          latents (sel/complement-sel obs-sel)]
      (is (sel/selected? obs-sel :y) "observed leaf selected")
      (is (not (sel/selected? obs-sel :mu)) "unobserved addr not selected")
      (is (sel/selected? (sel/get-subselection obs-sel :sub) :z)
          "nested observed leaf selected")
      (is (not (sel/selected? latents :y)) "complement excludes observation")
      (is (sel/selected? latents :mu) "complement includes latent")
      (is (not (sel/selected? (sel/get-subselection latents :sub) :z))
          "complement excludes nested observation")
      (is (sel/selected? (sel/get-subselection latents :sub) :other)
          "complement includes nested latent"))))

;; ============================================================
;; 2. mh default selection: observations stay fixed, posterior targeted
;; ============================================================
(deftest test-mh-default-selection-targets-posterior
  (testing "mh without :selection must not resample observations"
    ;; mu ~ N(0,1), y ~ N(mu,1), observe y=2.0.
    ;; Closed form: mu | y=2 is N(1.0, 1/2) — posterior mean 1.0, prior mean 0.
    (let [model (gen []
                  (let [mu (trace :mu (dist/gaussian (mx/scalar 0) (mx/scalar 1)))]
                    (trace :y (dist/gaussian mu (mx/scalar 1)))
                    mu))
          obs (cm/choicemap :y 2.0)
          traces (mcmc/mh {:samples 300 :burn 200 :key (rng/fresh-key 11)}
                          model [] obs)
          y-vals (mapv (fn [tr]
                         (->num (cm/get-value (cm/get-submap (:choices tr) :y))))
                       traces)
          mu-vals (mapv (fn [tr]
                          (->num (cm/get-value (cm/get-submap (:choices tr) :mu))))
                        traces)
          mu-mean (/ (reduce + mu-vals) (count mu-vals))]
      (is (every? #(< (Math/abs (- % 2.0)) 1e-6) y-vals)
          "observation :y stays exactly 2.0 in every kept trace")
      (is (< (Math/abs (- mu-mean 1.0)) 0.35)
          (str "posterior mean near 1.0 (closed form), got " mu-mean)))))

;; ============================================================
;; 3. Acceptance-rate denominator counts executed steps
;; ============================================================
(deftest test-acceptance-rate-denominator
  (testing "always-accepting kernel reports acceptance-rate exactly 1"
    ;; burn 2, samples 3, thin 4 → executed steps = 2 + (3-1)*4 + 1 = 11,
    ;; NOT burn + samples*thin = 14. An always-accept step-fn must yield 1.
    (let [out (kern/collect-samples
               {:samples 3 :burn 2 :thin 4 :key (rng/fresh-key 1)}
               (fn [state _k] {:state (inc state) :accepted? true})
               identity
               0)]
      (is (= 3 (count out)) "3 samples kept")
      (is (= 1 (:acceptance-rate (meta out)))
          "acceptance-rate is exactly 1 for an always-accepting kernel"))))

;; ============================================================
;; 4. Array-valued latent: score-fn matches closed-form joint
;; ============================================================
(deftest test-array-latent-score-layout
  (testing "prepare-mcmc-score scores array-valued latents correctly"
    ;; v ~ iid-gaussian [2] (array latent), y ~ N(mean(v), 1) observed.
    ;; Fully constrained score = lp(v0;0,1) + lp(v1;0,1) + lp(y; mean(v), 1).
    (let [model (gen []
                  (let [v (trace :v (dist/iid-gaussian (mx/scalar 0) (mx/scalar 1) 2))]
                    (trace :y (dist/gaussian (mx/mean v) (mx/scalar 1)))
                    v))
          obs (cm/choicemap :y 0.5)
          {:keys [trace]} (p/generate (dyn/with-key model (rng/fresh-key 3)) [] obs)
          {:keys [score-fn init-params n-params tensor-native?]}
          (u/prepare-mcmc-score model [] obs [:v] trace)
          v-arr (cm/get-value (cm/get-submap (:choices trace) :v))
          [v0 v1] (mx/->clj v-arr)
          expected (+ (gaussian-lp v0 0 1)
                      (gaussian-lp v1 0 1)
                      (gaussian-lp 0.5 (/ (+ v0 v1) 2) 1))
          got (->num (score-fn init-params))]
      (is (= 2 n-params) "array latent occupies 2 flat slots")
      (is (= [2] (vec (mx/shape init-params))) "init-params is the flattened latent")
      (is (< (Math/abs (- got expected)) 1e-3)
          (str "score matches closed-form joint: got " got " expected " expected)))))

(cljs.test/run-tests)
