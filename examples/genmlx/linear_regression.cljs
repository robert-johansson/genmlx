(ns linear-regression
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.mcmc :as mcmc])
  (:require-macros [genmlx.gen :refer [gen]]))

;; Bayesian linear regression — all values stay as MLX arrays
;; Inside gen bodies, trace/splice/param are local bindings injected by the macro
(def model
  (gen [xs]
    (let [slope     (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j))
               (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                      intercept) 1)))
      slope)))

;; Observations — data generated from y = 2x + 0.1 + noise
(def xs [1.0 2.0 3.0 4.0 5.0])
(def observations
  (cm/choicemap :y0 (mx/scalar 2.1) :y1 (mx/scalar 3.9) :y2 (mx/scalar 6.2)
                :y3 (mx/scalar 7.8) :y4 (mx/scalar 10.1)))

;; Metropolis-Hastings
(println "Running MH inference (500 samples, 100 burn-in)...")
(def traces
  (mcmc/mh {:samples 500 :burn 100
            :selection (sel/select :slope :intercept)}
           model [xs] observations))

;; Examine posterior
(let [slopes     (mapv #(mx/item (cm/get-choice (:choices %) [:slope])) traces)
      intercepts (mapv #(mx/item (cm/get-choice (:choices %) [:intercept])) traces)
      mean-slope     (/ (reduce + slopes) (count slopes))
      mean-intercept (/ (reduce + intercepts) (count intercepts))]
  (println "Posterior slope mean:    " mean-slope "(true: ~2.0)")
  (println "Posterior intercept mean:" mean-intercept "(true: ~0.1)"))
