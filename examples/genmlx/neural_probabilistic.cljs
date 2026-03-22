;; Neural Networks Meet Probabilistic Programming
;; ================================================
;;
;; Part 1: Train an MLP to learn sin(x) using MLX's native NN training.
;; Part 2: Wrap the trained network as a generative function and embed it
;; in a probabilistic model that adds calibrated uncertainty via MCMC.
;;
;; Demonstrates: nn layers, nn->gen-fn, training loop, MH for uncertainty.
;;
;; Run: bun run --bun nbb examples/genmlx/neural_probabilistic.cljs

(ns neural-probabilistic
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.dynamic :as dyn]
            [genmlx.mlx.random :as rng]
            [genmlx.nn :as nn]
            [genmlx.inference.mcmc :as mcmc])
  (:require-macros [genmlx.gen :refer [gen]]))

;; --- Helpers ---

(defn mean [vs] (/ (reduce + vs) (count vs)))
(defn std [vs]
  (let [m (mean vs)
        v (/ (reduce + (map #(* (- % m) (- % m)) vs)) (max 1 (dec (count vs))))]
    (js/Math.sqrt v)))

;; =========================================================================
;; Part 1: Train a neural network on y = sin(x)
;; =========================================================================

(println "\n-- Part 1: Training MLP on sin(x) --")

;; Create MLP: 1 → 32 → 32 → 1
(def net (nn/sequential [(nn/linear 1 32) (nn/relu)
                          (nn/linear 32 32) (nn/relu)
                          (nn/linear 32 1)]))
(def opt (nn/optimizer :adam 0.01))

;; Training data: 50 points on [-pi, pi]
(def n-train 50)
(def x-train (mx/array (vec (for [i (range n-train)]
                              [(- (* 2 js/Math.PI (/ i n-train)) js/Math.PI)]))))
(def y-train (mx/sin x-train))

;; value-and-grad + training loop
(def vg (nn/value-and-grad
         net
         (fn [x y-true]
           (mx/mean (mx/square (mx/subtract (.forward net x) y-true))))))

(dotimes [i 500]
  (let [loss (mx/training-step! net opt vg x-train y-train)]
    (when (zero? (mod (inc i) 100))
      (println (str "  Step " (inc i) ": MSE = " (.toFixed loss 6))))))

;; Test predictions
(println "\n-- Predictions vs. ground truth --")
(let [x-test (mx/array [[-2.0] [-1.0] [0.0] [1.0] [2.0]])
      y-pred (.forward net x-test)]
  (mx/eval! y-pred)
  (doseq [i (range 5)]
    (let [x (nth [-2.0 -1.0 0.0 1.0 2.0] i)
          pred (mx/item (mx/index y-pred i))
          true-val (js/Math.sin x)]
      (println (str "  x=" (.toFixed x 1)
                   "  pred=" (.toFixed pred 3)
                   "  true=" (.toFixed true-val 3))))))

;; =========================================================================
;; Part 2: Wrap NN as generative function, infer uncertainty
;; =========================================================================

(println "\n-- Part 2: Probabilistic model with trained NN --")

;; Wrap trained network as a deterministic generative function (score=0)
(def nn-gf (nn/nn->gen-fn net))

;; Probabilistic model: NN predicts mean, MH infers noise level
(def uncertain-model
  (dyn/auto-key
    (gen [xs ys]
      (let [log-sigma (trace :log-sigma (dist/gaussian -1 1))
            sigma     (mx/exp log-sigma)]
        (doseq [[j [x y]] (map-indexed vector (map vector xs ys))]
          (let [pred (mx/reshape (.forward net (mx/array [[x]])) [])]
            (trace (keyword (str "y" j))
                   (dist/gaussian pred sigma))))
        log-sigma))))

;; Noisy test data: sin(x) + Gaussian noise(sigma=0.2)
(def noise-sigma 0.2)
(def test-xs [0.0 0.5 1.0 1.5 2.0 2.5])
(def noise-key (genmlx.mlx.random/fresh-key))
(def noise-vals (mapv #(mx/item (genmlx.mlx.random/normal % []))
                      (genmlx.mlx.random/split-n noise-key (count test-xs))))
(def test-ys (mapv (fn [x n] (+ (js/Math.sin x) (* noise-sigma n))) test-xs noise-vals))

(def obs
  (reduce (fn [cm [j y]]
            (cm/set-choice cm [(keyword (str "y" j))] (mx/scalar y)))
          cm/EMPTY
          (map-indexed vector test-ys)))

;; MH to infer the noise level
(let [traces (mcmc/mh {:samples 500 :burn 200
                        :selection (sel/select :log-sigma)}
                       uncertain-model [test-xs test-ys] obs)
      log-sigmas (mapv #(mx/item (cm/get-choice (:choices %) [:log-sigma])) traces)
      sigmas (mapv js/Math.exp log-sigmas)]
  (println (str "  True noise sigma: " (.toFixed noise-sigma 3)))
  (println (str "  Inferred sigma:   " (.toFixed (mean sigmas) 3)
               " +/- " (.toFixed (std sigmas) 3))))

(println "\n-- Summary --")
(println "Train an MLP with MLX's native NN infrastructure, then wrap it")
(println "as a generative function. The probabilistic model adds calibrated")
(println "uncertainty — MCMC infers the noise level the NN can't capture.")
