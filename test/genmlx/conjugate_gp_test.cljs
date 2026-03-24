(ns genmlx.conjugate-gp-test
  "Conjugate posterior correctness tests -- Gamma-Poisson pair.
   Verifies that inference algorithms converge to the analytically-known
   posterior: Gamma(18, 6), E=3.0, Var=0.5.

   Tests: IS, Compiled MH, MH.

   NOTE: Gamma distribution log-prob uses the Lanczos gamma approximation,
   which creates many MLX scalar allocations per call. This limits the total
   number of inference iterations possible within Metal's 499K resource pool."
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

;; =========================================================================
;; Helpers
;; =========================================================================

(defn weighted-stats [traces log-weights param-key]
  (let [raw-weights (mapv (fn [w] (mx/eval! w) (mx/item w)) log-weights)
        max-w (apply max raw-weights)
        exp-weights (mapv (fn [w] (js/Math.exp (- w max-w))) raw-weights)
        sum-w (reduce + exp-weights)
        norm-weights (mapv (fn [w] (/ w sum-w)) exp-weights)
        vals (mapv (fn [t]
                     (let [v (cm/get-value (cm/get-submap (:choices t) param-key))]
                       (mx/eval! v) (mx/item v)))
                   traces)
        mean (reduce + (map * vals norm-weights))
        variance (reduce + (map (fn [v w] (* w (js/Math.pow (- v mean) 2)))
                                vals norm-weights))]
    [mean variance]))

(defn trace-stats [traces param-key]
  (let [vals (mapv (fn [t]
                     (let [v (cm/get-value (cm/get-submap (:choices t) param-key))]
                       (mx/eval! v) (mx/item v)))
                   traces)
        n (count vals)
        mean (/ (reduce + vals) n)
        variance (/ (reduce + (map (fn [v] (js/Math.pow (- v mean) 2)) vals))
                    (dec n))]
    [mean variance]))

(defn sample-stats [samples]
  (let [n (count samples)
        mean (/ (reduce + samples) n)
        variance (/ (reduce + (map (fn [v] (js/Math.pow (- v mean) 2)) samples))
                    (dec n))]
    [mean variance]))

;; =========================================================================
;; Gamma-Poisson
;; =========================================================================
;;
;; Prior:      lambda ~ Gamma(shape=3, rate=1)
;; Likelihood: x_i ~ Poisson(lambda), i=1..5
;; Data:       [2, 4, 3, 5, 1] (sum=15, n=5)
;;
;; Posterior:  Gamma(alpha + sum, rate + n) = Gamma(18, 6)
;;   E[lambda|data]   = 18/6 = 3.0
;;   Var[lambda|data]  = 18/36 = 0.5

(def model
  (gen [data]
    (let [lam (trace :lambda (dist/gamma-dist 3 1))]
      (doseq [[i x] (map-indexed vector data)]
        (trace (keyword (str "x" i)) (dist/poisson lam)))
      lam)))

(def data [2 4 3 5 1])
(def observations
  (reduce (fn [cm [i x]]
            (cm/set-choice cm [(keyword (str "x" i))] (mx/scalar x)))
          cm/EMPTY (map-indexed vector data)))

(def post-mean 3.0)
(def post-var 0.5)

(deftest compiled-mh-gp
  (testing "Compiled MH"
    (try
      (let [samples (mcmc/compiled-mh {:samples 400 :burn 300 :addresses [:lambda]
                                        :proposal-std 1.0 :compile? false :device :cpu}
                                       model [data] observations)
            lam-vals (mapv first samples)
            [mean variance] (sample-stats lam-vals)]
        (is (h/close? post-mean mean 0.5) "Compiled MH posterior mean ~ 3.0")
        (is (h/close? post-var variance 0.3) "Compiled MH posterior var ~ 0.5"))
      (catch :default e
        (is false (str "Compiled MH error: " (.-message e)))))))

(deftest is-gp
  (testing "IS"
    (try
      (let [{:keys [traces log-weights]}
            (is/importance-sampling {:samples 200} model [data] observations)
            [mean variance] (weighted-stats traces log-weights :lambda)]
        (is (h/close? post-mean mean 0.5) "IS posterior mean ~ 3.0")
        (is (h/close? post-var variance 0.3) "IS posterior var ~ 0.5"))
      (catch :default e
        (is false (str "IS error: " (.-message e)))))))

(deftest mh-gp
  (testing "MH"
    (try
      (let [traces (mcmc/mh {:samples 300 :burn 100 :selection (sel/select :lambda)}
                             model [data] observations)
            [mean variance] (trace-stats traces :lambda)]
        (is (h/close? post-mean mean 0.3) "MH posterior mean ~ 3.0")
        (is (h/close? post-var variance 0.3) "MH posterior var ~ 0.5"))
      (catch :default e
        (is false (str "MH error: " (.-message e)))))))

(cljs.test/run-tests)
