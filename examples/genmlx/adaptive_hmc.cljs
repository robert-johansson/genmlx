;; Adaptive HMC with GenMLX
;; ========================
;;
;; Correlated posterior from tight Bayesian regression (sigma=0.3).
;; Random-walk MH struggles; adaptive HMC auto-tunes step-size
;; during warmup and explores efficiently.
;;
;; Demonstrates: HMC with dual averaging, MH comparison, gradient MCMC.
;;
;; Run: bun run --bun nbb examples/genmlx/adaptive_hmc.cljs

(ns adaptive-hmc
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.mcmc :as mcmc])
  (:require-macros [genmlx.gen :refer [gen]]))

;; --- Helpers ---

(defn mean [vs] (/ (reduce + vs) (count vs)))

(defn std [vs]
  (let [m (mean vs)
        v (/ (reduce + (map #(* (- % m) (- % m)) vs)) (max 1 (dec (count vs))))]
    (js/Math.sqrt v)))

;; --- Model ---

;; Tight noise (sigma=0.3) creates strong correlation between slope and intercept.
;; This is where random-walk MH wastes proposals — it can't follow the narrow ridge.
(def model
  (gen [xs]
    (let [slope     (trace :slope (dist/gaussian 0 5))
          intercept (trace :intercept (dist/gaussian 0 5))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j))
               (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                      intercept) 0.3)))
      slope)))

;; --- Data ---

;; True slope=2.0, intercept=0.5, tight noise
(def xs [0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5])
(def ys [0.62 1.48 2.41 3.55 4.37 5.61 6.49 7.53 8.42 9.57])
(def observations
  (reduce (fn [cm [j y]] (cm/set-choice cm [(keyword (str "y" j))] (mx/scalar y)))
          cm/EMPTY
          (map-indexed vector ys)))

;; --- MH (random-walk) ---

(println "\n-- Metropolis-Hastings (random-walk, 500 samples) --")
(println "Random-walk MH proposes isotropically — it struggles with correlated posteriors.")

(let [traces (mcmc/mh {:samples 500 :burn 200
                        :selection (sel/select :slope :intercept)}
                       model [xs] observations)
      slopes     (mapv #(mx/item (cm/get-choice (:choices %) [:slope])) traces)
      intercepts (mapv #(mx/item (cm/get-choice (:choices %) [:intercept])) traces)]
  (println (str "  E[slope]     = " (.toFixed (mean slopes) 3)
               " +/- " (.toFixed (std slopes) 3) "  (true: 2.0)"))
  (println (str "  E[intercept] = " (.toFixed (mean intercepts) 3)
               " +/- " (.toFixed (std intercepts) 3) "  (true: 0.5)")))

;; --- Adaptive HMC ---

(println "\n-- Adaptive HMC (500 samples, dual averaging) --")
(println "HMC uses gradients to follow the posterior ridge. Dual averaging tunes step-size.")

(let [samples (mcmc/hmc {:samples 500 :burn 200
                          :leapfrog-steps 10
                          :addresses [:slope :intercept]
                          :adapt-step-size true}
                         model [xs] observations)
      slopes     (mapv #(nth % 0) samples)
      intercepts (mapv #(nth % 1) samples)]
  (println (str "  E[slope]     = " (.toFixed (mean slopes) 3)
               " +/- " (.toFixed (std slopes) 3) "  (true: 2.0)"))
  (println (str "  E[intercept] = " (.toFixed (mean intercepts) 3)
               " +/- " (.toFixed (std intercepts) 3) "  (true: 0.5)")))

(println "\n-- Summary --")
(println "HMC with adaptive step-size explores correlated posteriors efficiently.")
(println "No manual tuning needed — dual averaging (Hoffman & Gelman 2014) handles it.")
