;; One-Call Inference with GenMLX
;; ==============================
;;
;; The fit API inspects model structure and auto-selects the optimal
;; inference algorithm. Three models, one function call each.
;;
;; Demonstrates: fit API (L4), automatic method selection, schema-driven inference.
;;
;; Run: bun run --bun nbb examples/genmlx/fit_api.cljs

(ns fit-api
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.fit :as fit])
  (:require-macros [genmlx.gen :refer [gen]]))

;; =========================================================================
;; Model 1: Coin fairness (Beta-Bernoulli conjugate → :exact)
;; =========================================================================

;; Beta prior on bias, Bernoulli observations.
;; Fully conjugate — fit selects :exact (analytical, zero-variance).
(def coin-model
  (gen []
    (let [theta (trace :theta (dist/beta-dist 2 2))]
      (trace :y1 (dist/bernoulli theta))
      (trace :y2 (dist/bernoulli theta))
      (trace :y3 (dist/bernoulli theta))
      (trace :y4 (dist/bernoulli theta))
      (trace :y5 (dist/bernoulli theta))
      theta)))

;; 4 heads, 1 tail
(def coin-obs
  (cm/choicemap :y1 (mx/scalar 1) :y2 (mx/scalar 1) :y3 (mx/scalar 0)
                :y4 (mx/scalar 1) :y5 (mx/scalar 1)))

(println "\n-- Model 1: Coin fairness (Beta-Bernoulli) --")
(let [result (fit/fit coin-model [] coin-obs {:verbose? true})]
  (println (str "  Method: " (name (:method result))))
  (println (str "  Posterior theta: " (.toFixed (get-in result [:posterior :theta :value]) 3)))
  (println (str "  Log-ML: " (.toFixed (:log-ml result) 3)))
  (println (str "  Time: " (:elapsed-ms result) "ms")))

;; =========================================================================
;; Model 2: Robust mean estimation (non-conjugate → :hmc)
;; =========================================================================

;; Uniform prior on mu — not conjugate with Gaussian likelihood.
;; fit auto-selects :hmc for gradient-based sampling.
(def mean-model
  (gen []
    (let [mu (trace :mu (dist/uniform -10 10))]
      (trace :y1 (dist/gaussian mu 1))
      (trace :y2 (dist/gaussian mu 1))
      (trace :y3 (dist/gaussian mu 1))
      (trace :y4 (dist/gaussian mu 1))
      (trace :y5 (dist/gaussian mu 1))
      mu)))

;; Observations around true mu = 3.0
(def mean-obs
  (cm/choicemap :y1 (mx/scalar 2.8) :y2 (mx/scalar 3.2) :y3 (mx/scalar 2.9)
                :y4 (mx/scalar 3.5) :y5 (mx/scalar 3.1)))

(println "\n-- Model 2: Robust mean (non-conjugate, uniform prior) --")
(let [result (fit/fit mean-model [] mean-obs {:verbose? true})]
  (println (str "  Method: " (name (:method result))))
  (when-let [post (:posterior result)]
    (doseq [[addr stats] post]
      (if (:mean stats)
        (println (str "  " (name addr) ": " (.toFixed (:mean stats) 3)
                     " +/- " (.toFixed (:std stats) 3) "  (true: 3.0)"))
        (println (str "  " (name addr) ": " (.toFixed (:value stats) 3))))))
  (println (str "  Time: " (:elapsed-ms result) "ms")))

;; =========================================================================
;; Model 3: Coin model, method override → :handler-is
;; =========================================================================

;; Override auto-selection to use importance sampling instead of :exact.
;; Both methods answer the same question — fit respects the override.
(println "\n-- Model 3: Coin model, override → :handler-is (1000 particles) --")
(let [result (fit/fit coin-model [] coin-obs
                      {:verbose? true :method :handler-is :particles 1000})]
  (println (str "  Method: " (name (:method result))))
  (when-let [post (:posterior result)]
    (doseq [[addr stats] post]
      (if (:mean stats)
        (println (str "  " (name addr) ": " (.toFixed (:mean stats) 3)))
        (println (str "  " (name addr) ": " (.toFixed (:value stats) 3)
                     "  (exact was 0.667)")))))
  (println (str "  Log-ML: " (some-> (:log-ml result) (.toFixed 3))
               "  (exact was -3.332)"))
  (println (str "  Time: " (:elapsed-ms result) "ms")))

;; --- Summary ---

(println "\n-- Summary --")
(println "One function, three scenarios. fit inspects model structure —")
(println "conjugacy, dimensionality, differentiability — and selects the")
(println "optimal method. Override with :method when you know better.")
