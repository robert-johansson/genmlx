(ns genmlx.vectorized-mcmc-fix-test
  "Regression test: vectorized MCMC on multi-parameter models.
   Previously crashed with unordered_map::at on 2+ parameters."
  (:require [genmlx.mlx :as mx]
            [genmlx.dynamic :as dyn]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p]
            [genmlx.inference.mcmc :as mcmc])
  (:require-macros [genmlx.gen :refer [gen]]))

;; Model B: 2 inferred parameters (slope + intercept)
(def model-b
  (gen [xs]
    (let [slope     (dyn/trace :slope (dist/gaussian 0 10))
          intercept (dyn/trace :intercept (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (dyn/trace (keyword (str "y" j))
                   (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                          intercept) 1)))
      slope)))

(defn assert-true [desc v]
  (if v
    (println "  PASS:" desc)
    (println "  FAIL:" desc)))

(println "\n=== Vectorized MCMC multi-parameter regression test ===\n")

(let [xs     [1.0 2.0 3.0]
      obs    (cm/choicemap [:y0 (mx/scalar 2.1)]
                           [:y1 (mx/scalar 4.0)]
                           [:y2 (mx/scalar 5.9)])
      n-chains 4
      n-samples 5]

  ;; Test vectorized-compiled-mh on 2-param model
  (println "-- vectorized-compiled-mh (2 params) --")
  (let [result (mcmc/vectorized-compiled-mh
                 {:samples n-samples :burn 0 :addresses [:slope :intercept]
                  :proposal-std 0.5 :n-chains n-chains :device :cpu}
                 model-b [xs] obs)]
    (assert-true "returns vector" (vector? result))
    (assert-true "correct sample count" (= (count result) n-samples))
    (println "  Got" (count result) "samples"))

  ;; Test vectorized-mala on 2-param model
  (println "\n-- vectorized-mala (2 params) --")
  (let [result (mcmc/vectorized-mala
                 {:samples n-samples :burn 0 :step-size 0.01
                  :addresses [:slope :intercept] :n-chains n-chains :device :cpu}
                 model-b [xs] obs)]
    (assert-true "returns vector" (vector? result))
    (assert-true "correct sample count" (= (count result) n-samples))
    (println "  Got" (count result) "samples"))

  ;; Test vectorized-hmc on 2-param model
  (println "\n-- vectorized-hmc (2 params) --")
  (let [result (mcmc/vectorized-hmc
                 {:samples n-samples :burn 0 :step-size 0.01
                  :leapfrog-steps 5 :addresses [:slope :intercept]
                  :n-chains n-chains :device :cpu}
                 model-b [xs] obs)]
    (assert-true "returns vector" (vector? result))
    (assert-true "correct sample count" (= (count result) n-samples))
    (println "  Got" (count result) "samples")))

(println "\n=== Done ===")
