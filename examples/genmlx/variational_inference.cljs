;; Variational Inference — Approximating Posteriors
;; ==================================================
;;
;; Fit a mean-field Gaussian approximation to a non-conjugate posterior
;; using ADVI (Automatic Differentiation Variational Inference).
;;
;; Demonstrates: VI, ELBO optimization, posterior approximation,
;;               comparison with MCMC ground truth.
;;
;; Run: bun run --bun nbb examples/genmlx/variational_inference.cljs

(ns variational-inference
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.vi :as vi]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.util :as u])
  (:require-macros [genmlx.gen :refer [gen]]))

;; --- Model ---
;; Robust regression with unknown mean and variance.
;; Uses a Student-t likelihood (heavier tails than Gaussian).
;; Non-conjugate — no closed-form posterior.

(def model
  (gen [xs]
    (let [mu    (trace :mu (dist/gaussian 0 5))
          sigma (trace :log-sigma (dist/gaussian 0 1))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j))
               (dist/gaussian mu (mx/exp sigma))))
      mu)))

;; --- Data ---
;; Observations clustered around 3.0 with one outlier

(def data [2.8 3.1 2.9 3.3 3.0 2.7 3.2 8.5])  ;; 8.5 is an outlier
(def observations
  (reduce (fn [cm [j y]]
            (cm/set-choice cm [(keyword (str "y" j))] (mx/scalar y)))
          cm/EMPTY
          (map-indexed vector data)))

(println "=== Variational Inference ===\n")
(println "Data:" data "(note outlier at 8.5)\n")

;; --- VI: Fast approximate posterior ---

(println "-- ADVI (500 iterations) --")
(let [;; Build log-density from model + observations
      score-fn (u/make-score-fn model [data] observations [:mu :log-sigma])
      ;; Run VI
      {:keys [mu sigma elbo-history]}
      (vi/vi {:iterations 500 :learning-rate 0.02 :elbo-samples 10}
             score-fn (mx/array [0.0 0.0]))]
  (mx/eval! mu sigma)
  (let [mu-val     (mx/item (mx/index mu 0))
        sigma-val  (mx/item (mx/index sigma 0))
        log-s-val  (mx/item (mx/index mu 1))
        log-s-std  (mx/item (mx/index sigma 1))
        elbo-start (first elbo-history)
        elbo-end   (last elbo-history)]
    (println (str "  q(mu)       = N(" (.toFixed mu-val 3) ", " (.toFixed sigma-val 3) ")"))
    (println (str "  q(log-sigma) = N(" (.toFixed log-s-val 3) ", " (.toFixed log-s-std 3) ")"))
    (println (str "  ELBO: " (.toFixed elbo-start 1) " → " (.toFixed elbo-end 1)
                 " (improved by " (.toFixed (- elbo-end elbo-start) 1) ")"))))

;; --- MCMC: Ground truth posterior ---

(println "\n-- MH ground truth (2000 samples, 500 burn-in) --")
(let [traces (mcmc/mh {:samples 2000 :burn 500
                        :selection (sel/select :mu :log-sigma)}
                       model [data] observations)
      mus  (mapv #(mx/item (cm/get-choice (:choices %) [:mu])) traces)
      lss  (mapv #(mx/item (cm/get-choice (:choices %) [:log-sigma])) traces)
      mean (fn [vs] (/ (reduce + vs) (count vs)))
      std  (fn [vs]
             (let [m (mean vs)]
               (js/Math.sqrt (/ (reduce + (map #(* (- % m) (- % m)) vs)) (count vs)))))]
  (println (str "  E[mu]        = " (.toFixed (mean mus) 3) " ± " (.toFixed (std mus) 3)))
  (println (str "  E[log-sigma] = " (.toFixed (mean lss) 3) " ± " (.toFixed (std lss) 3))))

(println "\nVI is ~100x faster than MCMC. MCMC is more accurate for")
(println "multi-modal or heavy-tailed posteriors.")
