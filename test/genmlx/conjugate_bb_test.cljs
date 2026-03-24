(ns genmlx.conjugate-bb-test
  "Conjugate posterior correctness tests -- Beta-Bernoulli pair.
   Verifies that inference algorithms converge to the analytically-known
   posterior: Beta(10, 4), E=0.714, Var=0.0136.

   Tests: IS, Compiled MH, MH.

   NOTE: Beta distribution log-prob uses the Lanczos gamma approximation,
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
;; Beta-Bernoulli
;; =========================================================================
;;
;; Prior:      p ~ Beta(2, 2)
;; Likelihood: x_i ~ Bernoulli(p), i=1..10
;; Data:       [1,1,1,0,1,1,0,1,1,1] (sum=8, n=10)
;;
;; Posterior:  Beta(10, 4)
;;   E[p|data]   = 10/14 = 0.714
;;   Var[p|data]  = 40/2940 = 0.0136

(def model
  (gen [data]
    (let [p (trace :p (dist/beta-dist 2 2))]
      (doseq [[i x] (map-indexed vector data)]
        (trace (keyword (str "x" i)) (dist/bernoulli p)))
      p)))

(def data [1 1 1 0 1 1 0 1 1 1])
(def observations
  (reduce (fn [cm [i x]]
            (cm/set-choice cm [(keyword (str "x" i))] (mx/scalar x)))
          cm/EMPTY (map-indexed vector data)))

(def post-mean 0.714)
(def post-var 0.0136)

(deftest compiled-mh-bb
  (testing "Compiled MH"
    (try
      (let [samples (mcmc/compiled-mh {:samples 400 :burn 300 :addresses [:p]
                                        :proposal-std 0.15 :compile? false :device :cpu}
                                       model [data] observations)
            p-vals (mapv first samples)
            [mean variance] (sample-stats p-vals)]
        (is (h/close? post-mean mean 0.3) "Compiled MH posterior mean ~ 0.714")
        (is (h/close? post-var variance 0.02) "Compiled MH posterior var ~ 0.014"))
      (catch :default e
        (is false (str "Compiled MH error: " (.-message e)))))))

(deftest is-bb
  (testing "IS"
    (try
      (let [{:keys [traces log-weights]}
            (is/importance-sampling {:samples 200} model [data] observations)
            [mean variance] (weighted-stats traces log-weights :p)]
        (is (h/close? post-mean mean 0.3) "IS posterior mean ~ 0.714")
        (is (h/close? post-var variance 0.02) "IS posterior var ~ 0.014"))
      (catch :default e
        (is false (str "IS error: " (.-message e)))))))

(deftest mh-bb
  (testing "MH"
    (try
      (let [traces (mcmc/mh {:samples 300 :burn 100 :selection (sel/select :p)}
                             model [data] observations)
            [mean variance] (trace-stats traces :p)]
        (is (h/close? post-mean mean 0.3) "MH posterior mean ~ 0.714")
        (is (h/close? post-var variance 0.02) "MH posterior var ~ 0.014"))
      (catch :default e
        (is false (str "MH error: " (.-message e)))))))

(cljs.test/run-tests)
