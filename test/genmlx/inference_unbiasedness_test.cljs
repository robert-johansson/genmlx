(ns genmlx.inference-unbiasedness-test
  "Section 8.3: IS unbiasedness and SMC log-ML unbiasedness.

   IS unbiasedness property:
     The self-normalized IS estimator μ̂ = Σ w_i f(x_i) / Σ w_i
     is consistent: as N → ∞, μ̂ → E_p[f(x)] almost surely.

     Test: M independent IS trials with N particles each. The mean
     of the M estimates must equal E_p[f(x)] within z-test tolerance.

   SMC log-ML unbiasedness:
     The SMC normalizing constant estimator Ẑ is unbiased on the
     linear scale: E[Ẑ] = p(y).
     On the log scale: E[log Ẑ] ≤ log p(y) (Jensen's inequality).

     Test: M independent SMC trials, verify:
     (a) mean(exp(log-ML)) ≈ p(y) via z-test (linear unbiasedness)
     (b) mean(log-ML) ≤ log p(y) + margin (Jensen bound)
     (c) convergence: increasing particles tightens the estimate

   Model: Normal-Normal conjugate
     Prior: μ ~ N(0, 10), Likelihood: y_i ~ N(μ, 1), i=1..5
     Data: y = [2.8, 3.1, 2.9, 3.2, 3.0]
     Posterior: N(2.9940, 0.4468)
     Log marginal likelihood: -7.7979

   Tolerance policy:
     All statistical tests use z-test at z=3.5 (false-positive < 0.05%)
     over M independent runs, so SE = empirical_std / √M."
  (:require [cljs.test :refer [deftest is testing run-tests]]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.importance :as is]
            [genmlx.inference.smc :as smc]
            [genmlx.inference.util :as u]
            [genmlx.test-helpers :as th])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Model: Normal-Normal conjugate
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

(def normal-obs
  (reduce (fn [cm [i y]]
            (cm/set-choice cm [(keyword (str "y" i))] (mx/scalar y)))
          cm/EMPTY
          (map-indexed vector ys)))

;; ---------------------------------------------------------------------------
;; Analytical values
;; ---------------------------------------------------------------------------
;; precision_prior = 1/100 = 0.01
;; precision_obs   = 5/1 = 5.0
;; precision_post  = 5.01
;; mu_post = (0.01*0 + 5*3.0) / 5.01 = 2.9940
;; sigma_post = 1/sqrt(5.01) = 0.4468
;;
;; log p(y) via matrix determinant lemma:
;;   C = I + 100*J (5x5), |C| = 501
;;   C^{-1} = I - (100/501)*J
;;   y^T C^{-1} y = 45.1 - (100/501)*225 = 0.1898
;;   log p(y) = -5/2*log(2π) - 0.5*log(501) - 0.5*0.1898 = -7.7979

(def ^:private mu-post 2.9940119760479043)
(def ^:private sigma-post 0.4467670516087703)
(def ^:private analytical-log-ml -7.797905896206512)
(def ^:private analytical-ml (js/Math.exp analytical-log-ml))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- is-weighted-mean
  "Weighted mean of :mu from IS result."
  [{:keys [traces log-weights]}]
  (let [{:keys [probs]} (u/normalize-log-weights log-weights)
        vals (mapv (fn [tr]
                     (let [v (cm/get-choice (:choices tr) [:mu])]
                       (mx/eval! v) (mx/item v)))
                   traces)]
    (reduce + (map * probs vals))))

(defn- run-is-trial
  "Run one IS trial with N particles, return weighted mean of mu."
  [n seed]
  (let [result (is/importance-sampling
                 {:samples n :key (rng/fresh-key seed)}
                 normal-normal-model [ys] normal-obs)]
    (is-weighted-mean result)))

(defn- run-is-log-ml-trial
  "Run one IS trial, return log-ML estimate as JS number."
  [n seed]
  (let [result (is/importance-sampling
                 {:samples n :key (rng/fresh-key seed)}
                 normal-normal-model [ys] normal-obs)]
    (th/realize (:log-ml-estimate result))))

(defn- smc-weighted-mean
  "Weighted mean of :mu from SMC result."
  [{:keys [traces log-weights]}]
  (let [{:keys [probs]} (u/normalize-log-weights log-weights)
        vals (mapv (fn [tr]
                     (let [v (cm/get-choice (:choices tr) [:mu])]
                       (mx/eval! v) (mx/item v)))
                   traces)]
    (reduce + (map * probs vals))))

(defn- rmse
  "Root mean squared error of estimates relative to truth."
  [truth estimates]
  (js/Math.sqrt
    (/ (reduce + (map #(let [d (- % truth)] (* d d)) estimates))
       (count estimates))))

;; ==========================================================================
;; Test 1: IS estimator unbiasedness
;; ==========================================================================
;; M=20 independent IS trials, N=500 particles each.
;; z-test on the M estimates: mean should equal mu_post.
;; Uses empirical SE from the M trials — no assumed ESS.

(deftest is-estimator-unbiasedness
  (testing "IS weighted mean is unbiased: mean over M runs equals posterior mean"
    (let [M 20
          N 500
          estimates (mapv #(run-is-trial N (+ 1000 %)) (range M))
          grand-mean (th/sample-mean estimates)]
      (println (str "  IS unbiasedness: grand mean = " (.toFixed grand-mean 4)
                    ", analytical = " (.toFixed mu-post 4)
                    ", M=" M ", N=" N))
      (is (th/z-test-passes? mu-post estimates 3.5)
          (str "IS unbiased: grand mean " (.toFixed grand-mean 4)
               " vs analytical " (.toFixed mu-post 4))))))

;; ==========================================================================
;; Test 2: IS convergence rate
;; ==========================================================================
;; RMSE should decrease with more particles (consistency).
;; Run M=15 trials at N=100 and N=1000.
;; Verify: RMSE(N=1000) < RMSE(N=100) and RMSE(N=1000) is small.

(deftest is-convergence-rate
  (testing "IS RMSE decreases with more particles"
    (let [M 15
          est-small  (mapv #(run-is-trial 100 (+ 2000 %)) (range M))
          est-large  (mapv #(run-is-trial 1000 (+ 3000 %)) (range M))
          rmse-small (rmse mu-post est-small)
          rmse-large (rmse mu-post est-large)]
      (println (str "  IS convergence: RMSE(N=100) = " (.toFixed rmse-small 4)
                    ", RMSE(N=1000) = " (.toFixed rmse-large 4)))
      (is (< rmse-large rmse-small)
          (str "RMSE decreases: N=1000 → " (.toFixed rmse-large 4)
               " < N=100 → " (.toFixed rmse-small 4)))
      ;; RMSE at N=1000 should be < 1/3 of posterior sigma
      (is (< rmse-large (* 0.33 sigma-post))
          (str "RMSE(N=1000) = " (.toFixed rmse-large 4)
               " < " (.toFixed (* 0.33 sigma-post) 4))))))

;; ==========================================================================
;; Test 3: IS log-ML unbiasedness (linear scale)
;; ==========================================================================
;; The IS estimator Ẑ = (1/N) Σ w_i satisfies E[Ẑ] = p(y).
;; On the log scale, E[log Ẑ] ≤ log p(y) (Jensen's inequality).
;;
;; M=20 runs, N=1000 particles each:
;; (a) z-test on exp(log-ML) estimates vs p(y)
;; (b) verify Jensen bound: mean(log-ML) ≤ analytical + margin

(deftest is-log-ml-unbiasedness
  (testing "IS log-ML: linear-scale unbiasedness and Jensen bound"
    (let [M 20
          N 1000
          log-mls (mapv #(run-is-log-ml-trial N (+ 4000 %)) (range M))
          ml-linear (mapv js/Math.exp log-mls)
          mean-log-ml (th/sample-mean log-mls)
          mean-ml-linear (th/sample-mean ml-linear)]
      (println (str "  IS log-ML: mean(log-ML) = " (.toFixed mean-log-ml 4)
                    ", analytical = " (.toFixed analytical-log-ml 4)))
      (println (str "  IS log-ML: mean(exp(log-ML)) = " (.toFixed mean-ml-linear 6)
                    ", p(y) = " (.toFixed analytical-ml 6)))
      ;; All estimates finite
      (is (every? js/isFinite log-mls)
          "all log-ML estimates are finite")
      ;; Linear-scale unbiasedness via z-test
      (is (th/z-test-passes? analytical-ml ml-linear 3.5)
          (str "linear-scale unbiased: mean(exp(log-ML)) = "
               (.toFixed mean-ml-linear 6)
               " vs p(y) = " (.toFixed analytical-ml 6)))
      ;; Jensen bound: E[log Ẑ] ≤ log p(y), with 0.5 margin for finite-sample noise
      (is (<= mean-log-ml (+ analytical-log-ml 0.5))
          (str "Jensen bound: mean(log-ML) = "
               (.toFixed mean-log-ml 4)
               " ≤ " (.toFixed (+ analytical-log-ml 0.5) 4))))))

;; ==========================================================================
;; Test 4: SMC log-ML unbiasedness
;; ==========================================================================
;; Single-step SMC with all observations at once (= IS through SMC code path).
;; Multi-step SMC requires a sequential model (e.g., Unfold combinator)
;; because p/update on a non-sequential model produces incremental weights
;; that include old/new sampled value ratios, not standard PF weights.
;;
;; M=15 runs, 500 particles each.
;; Tests that the SMC infrastructure computes log-ML correctly.

(deftest smc-log-ml-unbiasedness
  (testing "SMC log-ML converges to analytical value"
    (let [M 15
          N 500
          log-mls (mapv (fn [i]
                          (let [result (smc/smc
                                         {:particles N
                                          :key (rng/fresh-key (+ 5000 i))}
                                         normal-normal-model [ys] [normal-obs])]
                            (th/realize (:log-ml-estimate result))))
                        (range M))
          ml-linear (mapv js/Math.exp log-mls)
          mean-log-ml (th/sample-mean log-mls)]
      (println (str "  SMC log-ML: mean(log-ML) = " (.toFixed mean-log-ml 4)
                    ", analytical = " (.toFixed analytical-log-ml 4)))
      ;; All finite
      (is (every? js/isFinite log-mls)
          "all SMC log-ML estimates are finite")
      ;; Linear-scale unbiasedness
      (is (th/z-test-passes? analytical-ml ml-linear 3.5)
          (str "SMC linear-scale unbiased: mean(exp(log-ML)) = "
               (.toFixed (th/sample-mean ml-linear) 6)
               " vs p(y) = " (.toFixed analytical-ml 6)))
      ;; Log-scale convergence: within 1.0 of analytical
      (is (th/close? analytical-log-ml mean-log-ml 1.0)
          (str "SMC log-ML near analytical: "
               (.toFixed mean-log-ml 4) " vs "
               (.toFixed analytical-log-ml 4))))))

;; ==========================================================================
;; Test 5: SMC posterior mean convergence
;; ==========================================================================
;; Verify SMC weighted particles converge to analytical posterior mean.

(deftest smc-posterior-convergence
  (testing "SMC weighted mean converges to analytical posterior mean"
    (let [M 15
          N 200
          estimates (mapv (fn [i]
                            (let [result (smc/smc
                                           {:particles N
                                            :key (rng/fresh-key (+ 6000 i))}
                                           normal-normal-model [ys] [normal-obs])]
                              (smc-weighted-mean result)))
                          (range M))
          grand-mean (th/sample-mean estimates)]
      (println (str "  SMC posterior: grand mean = " (.toFixed grand-mean 4)
                    ", analytical = " (.toFixed mu-post 4)))
      (is (th/z-test-passes? mu-post estimates 3.5)
          (str "SMC posterior mean unbiased: "
               (.toFixed grand-mean 4) " vs " (.toFixed mu-post 4))))))

;; ===========================================================================
;; Sequential model: Linear-Gaussian state-space model
;; ===========================================================================
;; x_0 ~ N(0, σ_x),  x_t ~ N(x_{t-1}, σ_x),  y_t ~ N(x_t, σ_y)
;;
;; Analytical log p(y) via Kalman filter. Multi-step SMC via
;; batched-smc-unfold, which runs the kernel ONCE per timestep
;; for all N particles — the proper particle filter, not p/update hacks.

(def ^:private ssm-sigma-x 1.0)
(def ^:private ssm-sigma-y (js/Math.sqrt 0.5))
(def ^:private ssm-Q (* ssm-sigma-x ssm-sigma-x))
(def ^:private ssm-R (* ssm-sigma-y ssm-sigma-y))

(def ssm-kernel
  "SSM kernel: x_t ~ N(state, σ_x), y_t ~ N(x_t, σ_y), returns x_t."
  (gen [t state]
    (let [x (trace :x (dist/gaussian state ssm-sigma-x))]
      (trace :y (dist/gaussian x ssm-sigma-y))
      x)))

(def ^:private ssm-observations [1.0 1.5 0.8 1.2 0.5])

(def ^:private ssm-obs-seq
  (mapv (fn [y] (cm/set-choice cm/EMPTY [:y] (mx/scalar y)))
        ssm-observations))

(defn- kalman-log-ml
  "Analytical log p(y) via Kalman filter for the SSM above.
   At t=0: prior on x_0 is N(init-state, Q).
   At t>0: transition x_t ~ N(x_{t-1}, Q), predict P = P_f + Q.
   Observation: y_t ~ N(x_t, R).

   Returns log p(y_0, ..., y_{T-1})."
  [observations Q R init-state]
  (let [log-2pi (js/Math.log (* 2 js/Math.PI))]
    (loop [t 0
           mu (double init-state)
           P (double Q)
           ll 0.0]
      (if (>= t (count observations))
        ll
        (let [y (nth observations t)
              S (+ P R)
              v (- y mu)
              ll-inc (* -0.5 (+ log-2pi (js/Math.log S) (/ (* v v) S)))
              K (/ P S)
              mu-f (+ mu (* K v))
              P-f (* (- 1.0 K) P)]
          (recur (inc t) mu-f (+ P-f Q) (+ ll ll-inc)))))))

(defn- kalman-filtered-mean
  "Kalman-filtered posterior mean E[x_T | y_0:T] at the final step."
  [observations Q R init-state]
  (loop [t 0
         mu (double init-state)
         P (double Q)]
    (if (>= t (count observations))
      mu
      (let [y (nth observations t)
            S (+ P R)
            v (- y mu)
            K (/ P S)
            mu-f (+ mu (* K v))
            P-f (* (- 1.0 K) P)]
        (recur (inc t) mu-f (+ P-f Q))))))

(def ^:private ssm-analytical-log-ml
  (kalman-log-ml ssm-observations ssm-Q ssm-R 0.0))

(def ^:private ssm-analytical-ml
  (js/Math.exp ssm-analytical-log-ml))

(def ^:private ssm-filtered-mean
  (kalman-filtered-mean ssm-observations ssm-Q ssm-R 0.0))

;; ==========================================================================
;; Test 6: Multi-step SMC log-ML unbiasedness
;; ==========================================================================
;; batched-smc-unfold runs the kernel per timestep for all N particles
;; via vgenerate. This is the proper bootstrap particle filter.
;;
;; The PF normalizing constant Ẑ = Π_t (1/N Σ_i w_t^i) is unbiased
;; on the linear scale: E[Ẑ] = p(y).
;;
;; M=15 independent runs, N=500 particles each.

(deftest smc-multistep-log-ml-unbiasedness
  (testing "Multi-step SMC log-ML via batched-smc-unfold"
    (let [M 15
          N 500
          log-mls (mapv (fn [i]
                          (let [result (smc/batched-smc-unfold
                                         {:particles N
                                          :key (rng/fresh-key (+ 7000 i))}
                                         ssm-kernel 0.0 ssm-obs-seq)]
                            (th/realize (:log-ml result))))
                        (range M))
          ml-linear (mapv js/Math.exp log-mls)
          mean-log-ml (th/sample-mean log-mls)]
      (println (str "  Multi-step SMC log-ML: mean(log-ML) = "
                    (.toFixed mean-log-ml 4)
                    ", Kalman analytical = " (.toFixed ssm-analytical-log-ml 4)))
      ;; All finite
      (is (every? js/isFinite log-mls)
          "all multi-step SMC log-ML estimates are finite")
      ;; Linear-scale unbiasedness via z-test
      (is (th/z-test-passes? ssm-analytical-ml ml-linear 3.5)
          (str "multi-step SMC linear-scale unbiased: mean(exp(log-ML)) = "
               (.toFixed (th/sample-mean ml-linear) 6)
               " vs p(y) = " (.toFixed ssm-analytical-ml 6)))
      ;; Jensen bound
      (is (<= mean-log-ml (+ ssm-analytical-log-ml 0.5))
          (str "Jensen bound: " (.toFixed mean-log-ml 4)
               " ≤ " (.toFixed (+ ssm-analytical-log-ml 0.5) 4)))
      ;; Log-scale convergence
      (is (th/close? ssm-analytical-log-ml mean-log-ml 0.5)
          (str "log-ML near Kalman: " (.toFixed mean-log-ml 4)
               " vs " (.toFixed ssm-analytical-log-ml 4))))))

;; ==========================================================================
;; Test 7: Multi-step SMC filtered state convergence
;; ==========================================================================
;; After 5-step PF, the resampled final states should approximate
;; E[x_4 | y_0:4] from the Kalman filter.

(deftest smc-multistep-filtered-state
  (testing "Multi-step SMC filtered state converges to Kalman posterior"
    (let [M 15
          N 500
          estimates (mapv (fn [i]
                            (let [result (smc/batched-smc-unfold
                                           {:particles N
                                            :key (rng/fresh-key (+ 8000 i))}
                                           ssm-kernel 0.0 ssm-obs-seq)
                                  states (:final-states result)]
                              (mx/eval! states)
                              (th/sample-mean (mx/->clj states))))
                          (range M))
          grand-mean (th/sample-mean estimates)]
      (println (str "  Multi-step SMC state: grand mean = "
                    (.toFixed grand-mean 4)
                    ", Kalman filtered = " (.toFixed ssm-filtered-mean 4)))
      (is (th/z-test-passes? ssm-filtered-mean estimates 3.5)
          (str "filtered state unbiased: " (.toFixed grand-mean 4)
               " vs Kalman " (.toFixed ssm-filtered-mean 4))))))

;; ==========================================================================
;; Run
;; ==========================================================================

(run-tests)
