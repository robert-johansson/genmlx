(ns genmlx.stress-test
  "Stress tests for Metal GPU resource management.
   Verifies that long inference runs complete without resource exhaustion."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dynamic :as dyn]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.smc :as smc]
            [genmlx.inference.importance :as is]
            [genmlx.inference.util :as u]
            [genmlx.gen :refer [gen]]))

(deftest mh-beta-bernoulli-stress
  (testing "MH on Beta-Bernoulli (5000 samples, 2000 burn)"
    (let [model (gen [n-obs]
                  (let [p (trace :p (dist/beta-dist 2 2))]
                    (mx/eval! p)
                    (let [p-val (mx/item p)]
                      (doseq [i (range n-obs)]
                        (trace (keyword (str "obs" i))
                                   (dist/bernoulli p-val)))
                      p-val)))
          n-obs 10
          observations (reduce (fn [cm i]
                                 (cm/set-choice cm [(keyword (str "obs" i))] (mx/scalar 1.0)))
                               cm/EMPTY (range n-obs))]
      (mx/clear-cache!)
      (mx/reset-peak-memory!)
      (let [traces (mcmc/mh {:samples 5000 :burn 2000 :selection (sel/select :p)}
                             model [n-obs] observations)]
        (is (= 5000 (count traces)) "Got 5000 traces")
        (let [p-vals (mapv (fn [t]
                             (let [v (cm/get-value (cm/get-submap (:choices t) :p))]
                               (mx/eval! v) (mx/item v)))
                           (take-last 100 traces))
              p-mean (/ (reduce + p-vals) (count p-vals))]
          (is (< (js/Math.abs (- p-mean 0.857)) 0.15) "Posterior mean near 0.857"))
        (u/dispose-trace traces)))))

(deftest long-mh-chain-stress
  (testing "Long MH chain (10000 samples, 5000 burn, Beta)"
    (mx/clear-cache!)
    (mx/force-gc!)
    (let [model (gen []
                  (let [x (trace :x (dist/beta-dist 2 2))]
                    x))
          obs (cm/choicemap)]
      (mx/clear-cache!)
      (mx/reset-peak-memory!)
      (let [traces (mcmc/mh {:samples 10000 :burn 5000 :selection (sel/select :x)}
                             model [] obs)]
        (is (= 10000 (count traces)) "Completed 10000 iterations")
        (let [x-vals (mapv (fn [t]
                             (let [v (cm/get-value (cm/get-submap (:choices t) :x))]
                               (mx/eval! v) (mx/item v)))
                           (take-last 200 traces))
              x-mean (/ (reduce + x-vals) (count x-vals))]
          (is (< (js/Math.abs (- x-mean 0.5)) 0.15) "Posterior mean near 0.5"))
        (u/dispose-trace traces)))))

(deftest smc-stress
  (testing "SMC with 20 timesteps, 100 particles"
    (mx/clear-cache!)
    (mx/force-gc!)
    (let [model (gen [xs]
                  (let [mu (trace :mu (dist/gaussian 0 5))]
                    (mx/eval! mu)
                    (let [m (mx/item mu)]
                      (doseq [[i x] (map-indexed vector xs)]
                        (trace (keyword (str "y" i)) (dist/gaussian m 1)))
                      m)))
          n-timesteps 20
          data (vec (repeatedly n-timesteps #(+ 3.0 (* 0.5 (- (js/Math.random) 0.5)))))
          obs-seq (mapv (fn [t]
                          (reduce (fn [cm i]
                                    (cm/set-choice cm [(keyword (str "y" i))] (mx/scalar (nth data i))))
                                  cm/EMPTY (range (inc t))))
                        (range n-timesteps))]
      (mx/clear-cache!)
      (mx/reset-peak-memory!)
      (let [result (smc/smc {:particles 100} model [data] obs-seq)]
        (is (= 100 (count (:traces result))) "Got 100 particles")
        (let [log-ml (:log-ml-estimate result)]
          (mx/eval! log-ml)
          (is (js/isFinite (mx/item log-ml)) "Log-ML is finite"))))))

(cljs.test/run-tests)
