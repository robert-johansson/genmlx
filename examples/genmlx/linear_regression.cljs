;; Bayesian Linear Regression with GenMLX
;; ========================================
;;
;; Given noisy observations y = slope * x + intercept + noise,
;; infer the posterior distribution over slope and intercept.
;;
;; Demonstrates: gen macro, trace sites, IS + MALA inference, posterior analysis.
;;
;; Run: bun run --bun nbb examples/genmlx/linear_regression.cljs

(ns linear-regression
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.importance :as is])
  (:require-macros [genmlx.gen :refer [gen]]))

;; --- Helpers ---

(defn get-val
  "Extract a scalar JS number from a trace's choicemap at the given address."
  [trace addr]
  (mx/item (cm/get-value (cm/get-submap (:choices trace) addr))))

(defn mean [vs] (/ (reduce + vs) (count vs)))

(defn std [vs]
  (let [m (mean vs)]
    (js/Math.sqrt (/ (reduce + (map #(* (- % m) (- % m)) vs)) (count vs)))))

;; --- Model ---
;; A generative model is a regular ClojureScript function wrapped in `gen`.
;; `trace` names random choices — these are the values inference reasons about.
;; All arithmetic uses mx/ ops to stay in the MLX computation graph.

(def model
  (gen [xs]
    (let [slope     (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j))
               (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                      intercept)
                              1)))
      {:slope slope :intercept intercept})))

;; --- Data ---
;; True parameters: slope=2.0, intercept=0.5

(def xs [1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0])
(def observations
  (reduce (fn [cm [j y]]
            (cm/set-choice cm [(keyword (str "y" j))] (mx/scalar y)))
          cm/EMPTY
          (map-indexed vector [2.6 4.3 7.1 8.4 10.8 12.3 14.0 16.9])))

(println "=== Bayesian Linear Regression ===\n")

;; --- Importance Sampling ---

(println "-- Importance Sampling (1000 particles) --")
(let [{:keys [traces log-weights]}
      (is/importance-sampling {:samples 1000} model [xs] observations)
      log-w (mx/array (mapv #(mx/item %) log-weights))
      w (mx/softmax log-w)
      _ (mx/eval! w)
      slopes     (mx/array (mapv #(get-val % :slope) traces))
      intercepts (mx/array (mapv #(get-val % :intercept) traces))
      mean-slope (mx/item (mx/sum (mx/multiply w slopes)))
      mean-int   (mx/item (mx/sum (mx/multiply w intercepts)))]
  (println (str "  E[slope]     = " (.toFixed mean-slope 3) "  (true: 2.0)"))
  (println (str "  E[intercept] = " (.toFixed mean-int 3) "  (true: 0.5)")))

;; --- MALA (gradient-informed MCMC) ---
;; MALA uses the gradient of the log-posterior to make informed proposals,
;; mixing much better than random-walk MH on correlated parameters.

(println "\n-- MALA (500 samples, 100 burn-in, gradient-informed) --")
(let [samples (mcmc/mala {:samples 500 :burn 100 :step-size 0.1
                          :addresses [:slope :intercept]
                          :compile? false}
                         model [xs] observations)
      slopes     (mapv first samples)
      intercepts (mapv second samples)]
  (println (str "  E[slope]     = " (.toFixed (mean slopes) 3)
               " +/- " (.toFixed (std slopes) 3) "  (true: 2.0)"))
  (println (str "  E[intercept] = " (.toFixed (mean intercepts) 3)
               " +/- " (.toFixed (std intercepts) 3) "  (true: 0.5)")))

;; --- Posterior Predictive ---

(println "\n-- Posterior Predictive --")
(let [samples (mcmc/mala {:samples 200 :burn 100 :step-size 0.1
                          :addresses [:slope :intercept]
                          :compile? false}
                         model [xs] observations)
      x-new 10.0
      preds (mapv (fn [s] (+ (* (first s) x-new) (second s))) samples)]
  (println (str "  y(x=10) = " (.toFixed (mean preds) 2)
               " +/- " (.toFixed (std preds) 2)
               "  (true: ~20.5)")))
