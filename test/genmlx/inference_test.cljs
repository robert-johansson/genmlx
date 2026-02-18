(ns genmlx.inference-test
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.importance :as is]
            [genmlx.inference.mcmc :as mcmc])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn assert-true [msg actual]
  (if actual
    (println "  PASS:" msg)
    (println "  FAIL:" msg "- expected truthy")))

(defn assert-close [msg expected actual tolerance]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tolerance)
      (println "  PASS:" msg)
      (do (println "  FAIL:" msg)
          (println "    expected:" expected "+/-" tolerance)
          (println "    actual:  " actual)))))

(println "\n=== Inference Tests ===\n")

;; Simple model: coin flip
(println "-- Importance sampling (coin flip) --")
(let [model (gen []
              (let [p (dyn/trace :p (dist/uniform 0.01 0.99))]
                (mx/eval! p)
                (let [p-val (mx/item p)]
                  (doseq [i (range 10)]
                    (dyn/trace (keyword (str "flip" i))
                               (dist/bernoulli p-val)))
                  p-val)))
      ;; All heads (10/10) -> posterior should concentrate near p=1
      observations (reduce (fn [cm i]
                             (cm/set-choice cm [(keyword (str "flip" i))] (mx/scalar 1.0)))
                           cm/EMPTY (range 10))
      {:keys [traces log-weights]} (is/importance-sampling
                                     {:samples 200} model [] observations)]
  (assert-true "IS returns traces" (= 200 (count traces)))
  (assert-true "IS returns weights" (= 200 (count log-weights)))
  ;; Compute weighted mean of p
  (let [weights-arr (mx/array (mapv (fn [w] (mx/eval! w) (mx/item w)) log-weights))
        log-probs (mx/subtract weights-arr (mx/logsumexp weights-arr))
        _ (mx/eval! log-probs)
        probs (mx/->clj (mx/exp log-probs))
        p-vals (mapv (fn [t] (tr/get-retval t)) traces)
        weighted-mean (reduce + (map * p-vals probs))]
    (assert-true "IS posterior p > 0.5" (> weighted-mean 0.5))))

;; MH on simple model
(println "\n-- MH (Gaussian posterior) --")
(let [model (gen []
              (let [mu (dyn/trace :mu (dist/gaussian 0 10))]
                (mx/eval! mu)
                (let [mu-val (mx/item mu)]
                  (doseq [i (range 5)]
                    (dyn/trace (keyword (str "obs" i))
                               (dist/gaussian mu-val 1)))
                  mu-val)))
      ;; All observations near 3.0
      observations (reduce (fn [cm i]
                             (cm/set-choice cm [(keyword (str "obs" i))]
                                            (mx/scalar (+ 3.0 (* 0.1 (- i 2))))))
                           cm/EMPTY (range 5))
      traces (mcmc/mh {:samples 200 :burn 100 :selection (sel/select :mu)}
                       model [] observations)]
  (assert-true "MH returns traces" (= 200 (count traces)))
  ;; Check posterior concentrates near 3
  (let [mu-vals (mapv (fn [t]
                        (let [v (cm/get-value (cm/get-submap (tr/get-choices t) :mu))]
                          (mx/eval! v) (mx/item v)))
                      traces)
        mu-mean (/ (reduce + mu-vals) (count mu-vals))]
    (assert-close "MH posterior mu near 3" 3.0 mu-mean 1.0)))

(println "\nAll inference tests complete.")
