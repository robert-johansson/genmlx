(ns genmlx.inference-convergence-test
  "Inference convergence tests: Gamma-Poisson conjugate, HMC/NUTS acceptance."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.importance :as is]
            [genmlx.inference.mcmc :as mcmc])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Gamma-Poisson conjugate
;; Prior:      lambda ~ Gamma(alpha=3, rate=1)
;; Likelihood: x_i ~ Poisson(lambda), i=1..5
;; Data:       [2, 4, 3, 5, 1] (sum=15)
;; Posterior:  Gamma(alpha + sum = 18, rate + n = 6)
;; E[lambda|data] = 18/6 = 3.0
;; ---------------------------------------------------------------------------

(def gamma-poisson-model
  (gen [data]
    (let [lam (trace :lambda (dist/gamma-dist 3 1))]
      (mx/eval! lam)
      (let [lam-val (mx/item lam)]
        (doseq [[i x] (map-indexed vector data)]
          (trace (keyword (str "x" i)) (dist/poisson lam-val)))
        lam-val))))

(def gp-data [2 4 3 5 1])
(def gp-observations
  (reduce (fn [cm [i x]]
            (cm/set-choice cm [(keyword (str "x" i))] (mx/scalar x)))
          cm/EMPTY (map-indexed vector gp-data)))

(deftest gamma-poisson-is-test
  (testing "IS (100 particles)"
    (let [{:keys [traces log-weights]} (is/importance-sampling
                                         {:samples 100} gamma-poisson-model [gp-data] gp-observations)
          raw-weights (mapv (fn [w] (mx/eval! w) (mx/item w)) log-weights)
          max-w (apply max raw-weights)
          exp-weights (mapv (fn [w] (js/Math.exp (- w max-w))) raw-weights)
          sum-w (reduce + exp-weights)
          norm-weights (mapv (fn [w] (/ w sum-w)) exp-weights)
          lambda-vals (mapv :retval traces)
          weighted-mean (reduce + (map * lambda-vals norm-weights))]
      (is (h/close? 3.0 weighted-mean 0.5) "IS posterior mean(lambda) ~ 3.0")
      (is (= 100 (count traces)) "IS returns 100 traces"))))

(deftest gamma-poisson-mh-test
  (testing "MH (500 samples)"
    (let [model-mh (gen [data]
                     (let [lam (trace :lambda (dist/gamma-dist 3 1))]
                       (mx/eval! lam)
                       (let [lam-val (mx/item lam)]
                         (doseq [[i x] (map-indexed vector data)]
                           (trace (keyword (str "x" i)) (dist/poisson lam-val)))
                         lam-val)))
          traces (mcmc/mh {:samples 500 :burn 200 :selection (sel/select :lambda)}
                           model-mh [gp-data] gp-observations)
          lambda-vals (mapv (fn [t]
                              (let [v (cm/get-value (cm/get-submap (:choices t) :lambda))]
                                (mx/eval! v) (mx/item v)))
                            traces)
          lambda-mean (/ (reduce + lambda-vals) (count lambda-vals))]
      (is (h/close? 3.0 lambda-mean 0.5) "MH posterior mean(lambda) ~ 3.0")
      (is (= 500 (count traces)) "MH returns 500 traces"))))

;; ---------------------------------------------------------------------------
;; HMC/NUTS acceptance rate
;; ---------------------------------------------------------------------------

(def normal-normal
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (doseq [i (range 5)]
        (trace (keyword (str "obs" i)) (dist/gaussian mu 1)))
      mu)))

(def nn-observations
  (reduce (fn [cm i]
            (cm/set-choice cm [(keyword (str "obs" i))]
                           (mx/scalar (+ 3.0 (* 0.1 (- i 2))))))
          cm/EMPTY (range 5)))

(deftest hmc-acceptance-test
  (testing "HMC acceptance rate"
    (let [samples (mcmc/hmc {:samples 100 :burn 50 :step-size 0.05 :leapfrog-steps 10
                              :addresses [:mu] :compile? false :device :cpu}
                             normal-normal [] nn-observations)
          mu-vals (mapv first samples)
          has-nan (some js/isNaN mu-vals)
          n-unique (count (set mu-vals))
          acceptance-rate (/ (double n-unique) (count mu-vals))]
      (is (not has-nan) "HMC: no NaN in samples")
      (is (> acceptance-rate 0.3) "HMC: acceptance rate > 0.3")
      (is (h/close? 3.0 (/ (reduce + mu-vals) (count mu-vals)) 1.0)
          "HMC: posterior mean ~ 3"))))

(deftest nuts-acceptance-test
  (testing "NUTS acceptance rate"
    (let [samples (mcmc/nuts {:samples 50 :burn 50 :step-size 0.05
                               :addresses [:mu] :compile? false :device :cpu}
                              normal-normal [] nn-observations)
          mu-vals (mapv first samples)
          has-nan (some js/isNaN mu-vals)
          n-unique (count (set mu-vals))
          acceptance-rate (/ (double n-unique) (count mu-vals))]
      (is (not has-nan) "NUTS: no NaN in samples")
      (is (> acceptance-rate 0.3) "NUTS: acceptance rate > 0.3")
      (is (h/close? 3.0 (/ (reduce + mu-vals) (count mu-vals)) 1.0)
          "NUTS: posterior mean ~ 3"))))

(cljs.test/run-tests)
