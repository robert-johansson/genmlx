(ns genmlx.inference-test
  "Tests for importance sampling and Metropolis-Hastings inference."
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

(deftest importance-sampling-coin-flip-test
  (testing "IS on coin flip model with all-heads observations"
    (let [model (gen []
                  (let [p (trace :p (dist/uniform 0.01 0.99))]
                    (mx/eval! p)
                    (let [p-val (mx/item p)]
                      (doseq [i (range 10)]
                        (trace (keyword (str "flip" i))
                               (dist/bernoulli p-val)))
                      p-val)))
          observations (reduce (fn [cm i]
                                 (cm/set-choice cm [(keyword (str "flip" i))] (mx/scalar 1.0)))
                               cm/EMPTY (range 10))
          {:keys [traces log-weights]} (is/importance-sampling
                                         {:samples 200} model [] observations)]
      (is (= 200 (count traces)) "IS returns traces")
      (is (= 200 (count log-weights)) "IS returns weights")
      (let [weights-arr (mx/array (mapv (fn [w] (mx/eval! w) (mx/item w)) log-weights))
            log-probs (mx/subtract weights-arr (mx/logsumexp weights-arr))
            _ (mx/eval! log-probs)
            probs (mx/->clj (mx/exp log-probs))
            p-vals (mapv :retval traces)
            weighted-mean (reduce + (map * p-vals probs))]
        (is (> weighted-mean 0.5) "IS posterior p > 0.5")))))

(deftest mh-gaussian-posterior-test
  (testing "MH on Gaussian model with observations near 3.0"
    (let [model (gen []
                  (let [mu (trace :mu (dist/gaussian 0 10))]
                    (mx/eval! mu)
                    (let [mu-val (mx/item mu)]
                      (doseq [i (range 5)]
                        (trace (keyword (str "obs" i))
                               (dist/gaussian mu-val 1)))
                      mu-val)))
          observations (reduce (fn [cm i]
                                 (cm/set-choice cm [(keyword (str "obs" i))]
                                                (mx/scalar (+ 3.0 (* 0.1 (- i 2))))))
                               cm/EMPTY (range 5))
          traces (mcmc/mh {:samples 200 :burn 100 :selection (sel/select :mu)}
                           model [] observations)]
      (is (= 200 (count traces)) "MH returns traces")
      (let [mu-vals (mapv (fn [t]
                            (let [v (cm/get-value (cm/get-submap (:choices t) :mu))]
                              (mx/eval! v) (mx/item v)))
                          traces)
            mu-mean (/ (reduce + mu-vals) (count mu-vals))]
        (is (h/close? 3.0 mu-mean 1.0) "MH posterior mu near 3")
        (is (> mu-mean 1.0) "MH posterior mu > 1")
        (is (< mu-mean 5.0) "MH posterior mu < 5")))))

(cljs.test/run-tests)
