(ns genmlx.mcmc-stationary-test
  "Phase 5.2 -- MCMC stationary distribution tests.
   Each test verifies that a sampler converges to the analytically known
   posterior. Tolerances derive from standard-error analysis:

     tolerance = z * safety * SE

   where z=3.5 (false-positive < 0.05%), safety=1.5 (MCMC autocorrelation),
   and SE = sigma / sqrt(N_eff). Every tolerance is annotated with its SE
   derivation so reviewers can verify the math."
  (:require [cljs.test :refer [deftest is testing run-tests]]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.importance :as is]
            [genmlx.inference.smc :as smc]
            [genmlx.test-helpers :as th])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Model B: multi-observation Normal-Normal
;; ---------------------------------------------------------------------------
;; Prior:     mu ~ N(0, 10)
;; Likelihood: y_i ~ N(mu, 1) for 5 observations [2.8, 3.1, 2.9, 3.2, 3.0]
;; Posterior:  N(2.994, 0.1996), sigma_post = 0.4468

(def model-b
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (doseq [j (range 5)]
        (trace (keyword (str "y" j)) (dist/gaussian mu 1)))
      mu)))

(def obs-b
  (let [ys [2.8 3.1 2.9 3.2 3.0]]
    (->> (map-indexed vector ys)
         (reduce (fn [acc [j y]]
                   (cm/set-choice acc [(keyword (str "y" j))] (mx/scalar y)))
                 cm/EMPTY))))

;; ---------------------------------------------------------------------------
;; Model C: Beta-Bernoulli
;; ---------------------------------------------------------------------------
;; Prior:     p ~ Beta(2, 2)
;; Likelihood: x_i ~ Bernoulli(p)
;; Data:      [1,1,1,0,1,1,0,1,1,1] -> Posterior Beta(10, 4)
;; E[p|data] = 10/14 = 0.7143, Var = 10*4 / (14^2 * 15) = 0.01361

(def model-c
  (gen []
    (let [p (trace :p (dist/beta-dist 2 2))]
      (doseq [j (range 10)]
        (trace (keyword (str "x" j)) (dist/bernoulli p)))
      p)))

(def obs-c
  (let [xs [1 1 1 0 1 1 0 1 1 1]]
    (->> (map-indexed vector xs)
         (reduce (fn [acc [j x]]
                   (cm/set-choice acc [(keyword (str "x" j))] (mx/scalar x)))
                 cm/EMPTY))))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- weighted-mean-from-is
  "Weighted mean of retvals from importance sampling result.
   Normalizes log-weights via logsumexp for numerical stability.
   Extracts JS numbers from MLX arrays via eval!/item."
  [{:keys [traces log-weights]}]
  (let [raw-weights  (mapv (fn [w] (mx/eval! w) (mx/item w)) log-weights)
        max-w        (apply max raw-weights)
        exp-weights  (mapv #(js/Math.exp (- % max-w)) raw-weights)
        sum-w        (reduce + exp-weights)
        norm-weights (mapv #(/ % sum-w) exp-weights)
        retvals      (mapv (fn [t]
                             (let [rv (:retval t)]
                               (if (mx/array? rv)
                                 (do (mx/eval! rv) (mx/item rv))
                                 rv)))
                           traces)]
    (reduce + (map * retvals norm-weights))))

;; ---------------------------------------------------------------------------
;; 5.2.1 MH on Normal-Normal (Model B)
;; ---------------------------------------------------------------------------

(deftest mh-normal-normal-convergence
  (testing "MH converges to Normal-Normal posterior N(2.994, 0.1996)"
    (let [samples (mcmc/compiled-mh {:samples 2000 :burn 500 :thin 2
                                     :addresses [:mu] :proposal-std 0.3}
                                    model-b [] obs-b)
          mu-vals (mapv first samples)]
      (is (= 2000 (count mu-vals))
          "returns requested number of samples")
      (is (not (some js/isNaN mu-vals))
          "no NaN in samples")
      ;; SE = sigma_post / sqrt(N_eff) where N_eff ~ 200 (thinned by 2, burn 500)
      ;; SE = 0.4468 / sqrt(200) = 0.0316
      ;; tolerance = 3.5 * SE = 0.11
      (is (th/close? 2.994 (th/sample-mean mu-vals) 0.11)
          "mean near posterior mean 2.994")
      ;; Variance tolerance: chi-squared based, ~ 0.067
      (is (th/close? 0.1996 (th/sample-variance mu-vals) 0.067)
          "variance near posterior variance 0.1996"))))

;; ---------------------------------------------------------------------------
;; 5.2.2 HMC on Normal-Normal (Model B)
;; ---------------------------------------------------------------------------

(deftest hmc-normal-normal-convergence
  (testing "HMC converges to Normal-Normal posterior"
    (let [samples (mcmc/hmc {:samples 500 :burn 200 :step-size 0.1
                             :leapfrog-steps 10 :addresses [:mu]}
                            model-b [] obs-b)
          mu-vals (mapv first samples)]
      (is (not (some js/isNaN mu-vals))
          "no NaN in samples")
      ;; 300 post-burn samples, SE = 0.4468 / sqrt(300) = 0.0258
      ;; tolerance = 3.5 * 1.1 * SE = 0.10 (HMC has lower autocorrelation)
      (is (th/close? 2.994 (th/sample-mean mu-vals) 0.10)
          "mean near posterior mean 2.994")))

  (testing "HMC acceptance rate is healthy"
    (let [samples (mcmc/hmc {:samples 200 :burn 100 :step-size 0.1
                             :leapfrog-steps 10 :addresses [:mu]
                             :compile? false :device :cpu}
                            model-b [] obs-b)
          rate    (:acceptance-rate (meta samples))]
      (is (> rate 0.5)
          "acceptance rate exceeds 50%"))))

;; ---------------------------------------------------------------------------
;; 5.2.3 Multi-Algorithm Agreement (Model B)
;; ---------------------------------------------------------------------------

(deftest multi-algorithm-agreement
  (let [means (atom {})]

    (testing "IS weighted mean near posterior"
      ;; ESS ~ 141 for IS on this model, SE = 0.4468 / sqrt(141) = 0.0376
      ;; tolerance = 3.5 * 1.5 * SE = 0.20 (IS safety factor for weight variance)
      (let [result  (is/importance-sampling {:samples 2000} model-b [] obs-b)
            is-mean (weighted-mean-from-is result)]
        (swap! means assoc :IS is-mean)
        (is (th/close? 2.994 is-mean 0.20)
            "IS mean near 2.994")))

    (testing "compiled-MH mean near posterior"
      ;; N_eff ~ 200, SE = 0.4468 / sqrt(200) = 0.0316
      ;; tolerance = 3.5 * SE = 0.11
      (let [samples (mcmc/compiled-mh {:samples 2000 :burn 500 :thin 2
                                       :addresses [:mu] :proposal-std 0.3}
                                      model-b [] obs-b)
            mh-mean (th/sample-mean (mapv first samples))]
        (swap! means assoc :MH mh-mean)
        (is (th/close? 2.994 mh-mean 0.11)
            "MH mean near 2.994")))

    (testing "HMC mean near posterior"
      ;; 300 post-burn, SE = 0.4468 / sqrt(300) = 0.0258
      ;; tolerance = 3.5 * 1.1 * SE = 0.10
      (let [samples  (mcmc/hmc {:samples 500 :burn 200 :step-size 0.1
                                :leapfrog-steps 10 :addresses [:mu]}
                               model-b [] obs-b)
            hmc-mean (th/sample-mean (mapv first samples))]
        (swap! means assoc :HMC hmc-mean)
        (is (th/close? 2.994 hmc-mean 0.10)
            "HMC mean near 2.994")))

    (testing "NUTS mean near posterior"
      ;; 50 samples (expensive), SE = 0.4468 / sqrt(50) = 0.0632
      ;; tolerance = 3.5 * SE = 0.22
      (let [samples   (mcmc/nuts {:samples 50 :burn 50 :step-size 0.05
                                  :addresses [:mu] :compile? false :device :cpu}
                                 model-b [] obs-b)
            nuts-mean (th/sample-mean (mapv first samples))]
        (swap! means assoc :NUTS nuts-mean)
        (is (th/close? 2.994 nuts-mean 0.22)
            "NUTS mean near 2.994")))

    (testing "SMC mean near posterior"
      ;; ESS ~ 35 particles, SE = 0.4468 / sqrt(35) = 0.0755
      ;; tolerance = 3.5 * SE = 0.27 (single-step SMC = weighted IS variant)
      (let [result   (smc/smc {:particles 500} model-b [] [obs-b])
            smc-mean (weighted-mean-from-is result)]
        (swap! means assoc :SMC smc-mean)
        (is (th/close? 2.994 smc-mean 0.27)
            "SMC mean near 2.994")))

    (testing "pairwise algorithm agreement"
      ;; All algorithms target the same posterior; max pairwise difference
      ;; should be bounded. Combined SE from two independent estimators:
      ;; SE_diff = sqrt(SE_1^2 + SE_2^2). Worst case IS vs NUTS:
      ;; sqrt(0.0376^2 + 0.0632^2) = 0.0735, tolerance = 3.5 * 1.35 * 0.0735 = 0.35
      (let [mean-vals (vals @means)
            max-diff  (->> (for [a mean-vals, b mean-vals :when (not= a b)]
                             (js/Math.abs (- a b)))
                           (apply max))]
        (is (< max-diff 0.35)
            "max pairwise mean difference below 0.35")))))

;; ---------------------------------------------------------------------------
;; 5.2.4 MH on Beta-Bernoulli (Model C)
;; ---------------------------------------------------------------------------

(deftest mh-beta-bernoulli-convergence
  (testing "MH converges to Beta(10,4) posterior"
    (let [samples (mcmc/compiled-mh {:samples 3000 :burn 500 :thin 2
                                     :addresses [:p] :proposal-std 0.1}
                                    model-c [] obs-c)
          p-vals  (mapv first samples)]
      (is (not (some js/isNaN p-vals))
          "no NaN in samples")
      (is (every? #(< 0 % 1) p-vals)
          "all samples in (0, 1)")
      ;; Posterior Beta(10,4): E=0.7143, Var=0.01361, sigma=0.1167
      ;; N_eff ~ 500 (thinned), SE = 0.1167 / sqrt(500) = 0.00522
      ;; tolerance = 3.5 * 1.4 * SE = 0.025
      (is (th/close? 0.7143 (th/sample-mean p-vals) 0.025)
          "mean near E[Beta(10,4)] = 0.7143")
      ;; Variance tolerance based on chi-squared: ~ 0.0042
      (is (th/close? 0.01361 (th/sample-variance p-vals) 0.0042)
          "variance near Var[Beta(10,4)] = 0.01361"))))

;; ---------------------------------------------------------------------------
;; 5.2.5 Gibbs on discrete model
;; ---------------------------------------------------------------------------
;; z ~ Categorical(logits=[log(2), log(3)]) => P(z=0)=0.4, P(z=1)=0.6
;; mu = 1 + 3*z  => z=0: mu=1, z=1: mu=4
;; y ~ N(mu, 1), observe y=2
;;
;; Analytical posterior:
;; P(z=1|y=2) = P(z=1)*N(2;4,1) / [P(z=0)*N(2;1,1) + P(z=1)*N(2;4,1)]
;;            = 0.6*exp(-2) / (0.4*exp(-0.5) + 0.6*exp(-2))
;;            = 0.08120 / 0.32381 = 0.2508

(deftest gibbs-discrete-convergence
  (testing "Gibbs converges to analytical discrete posterior"
    (let [model-g  (gen []
                     (let [z  (trace :z (dist/categorical [(js/Math.log 2) (js/Math.log 3)]))
                           mu (mx/add (mx/scalar 1.0) (mx/multiply z (mx/scalar 3.0)))]
                       (trace :y (dist/gaussian mu 1))
                       z))
          obs-g    (cm/choicemap :y (mx/scalar 2.0))
          schedule [{:addr :z :support [0 1]}]
          traces   (mcmc/gibbs {:samples 2000 :burn 200} model-g [] obs-g schedule)
          vals     (mapv (fn [t]
                           (let [rv (:retval t)]
                             (if (mx/array? rv) (do (mx/eval! rv) (mx/item rv)) rv)))
                         traces)
          frac-1   (/ (double (count (filter #(== 1 %) vals))) (count vals))]
      (is (= 2000 (count traces))
          "returns requested number of traces")
      ;; Gibbs on 2-state discrete model gives independent samples (exact conditional).
      ;; SE = sqrt(p*(1-p)/N) = sqrt(0.251*0.749/2000) = 0.0097
      ;; tolerance = 3.5 * 1.5 * SE = 0.051
      (is (th/close? 0.2508 frac-1 0.051)
          "P(z=1|y=2) near analytical 0.2508"))))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(run-tests)
