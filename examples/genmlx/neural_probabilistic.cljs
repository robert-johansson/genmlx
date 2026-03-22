;; Neural Networks Meet Probabilistic Programming
;; ================================================
;;
;; Wrap a neural network as a generative function and embed it
;; in a probabilistic model. The NN provides deterministic structure,
;; the probabilistic model adds calibrated uncertainty via MCMC.
;;
;; Demonstrates: nn layers, nn->gen-fn, splice, MH for uncertainty quantification.
;;
;; Run: bun run --bun nbb examples/genmlx/neural_probabilistic.cljs

(ns neural-probabilistic
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.dynamic :as dyn]
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
;; Part 1: Neural network as a deterministic generative function
;; =========================================================================

(println "\n-- Part 1: NN as deterministic gen function --")

;; Create a simple linear layer: y = W*x + b
(def layer (nn/linear 1 1))
(def nn-gf (nn/nn->gen-fn layer))

;; Simulate: deterministic forward pass, score = 0
(let [input (mx/array [3.0] [1 1])
      trace (p/simulate nn-gf [input])]
  (println (str "  Input: 3.0"))
  (println (str "  Output: " (.toFixed (mx/item (mx/reshape (:retval trace) [])) 4)))
  (println (str "  Score: " (mx/item (:score trace)) " (deterministic — always 0)"))
  (println (str "  Choices: " (:choices trace) " (no random choices)")))

;; =========================================================================
;; Part 2: NN inside a probabilistic model
;; =========================================================================

(println "\n-- Part 2: Probabilistic model with NN component --")

;; The NN provides a learned prediction; the probabilistic model
;; adds a noise parameter that we infer from data.
;; This is the key bridge: deterministic structure + calibrated uncertainty.

(def uncertain-model
  (dyn/auto-key
    (gen [xs ys]
      (let [log-sigma (trace :log-sigma (dist/gaussian -1 1))
            sigma     (mx/exp log-sigma)]
        (doseq [[j [x y]] (map-indexed vector (map vector xs ys))]
          (let [x-arr (mx/array [x] [1 1])
                pred  (mx/reshape (.forward layer x-arr) [])]
            (trace (keyword (str "y" j))
                   (dist/gaussian pred sigma))))
        log-sigma))))

;; Synthetic data: observations with known noise level
(def noise-sigma 0.3)
(def test-xs [0.0 1.0 2.0 3.0 4.0 5.0])

;; Generate "observations" using the NN's own predictions + noise
(def nn-predictions
  (mapv (fn [x]
          (let [pred (mx/item (mx/reshape (.forward layer (mx/array [x] [1 1])) []))]
            (mx/eval! (mx/scalar pred))
            pred))
        test-xs))

(def test-ys (mapv #(+ % (* noise-sigma (- (* 2 (js/Math.random)) 1))) nn-predictions))

(def obs
  (reduce (fn [cm [j y]]
            (cm/set-choice cm [(keyword (str "y" j))] (mx/scalar y)))
          cm/EMPTY
          (map-indexed vector test-ys)))

;; --- MH to infer the noise level ---

(println "\n-- MH inference for noise level --")
(println (str "  True noise sigma: " (.toFixed noise-sigma 3)))

(let [traces (mcmc/mh {:samples 500 :burn 200
                        :selection (sel/select :log-sigma)}
                       uncertain-model [test-xs test-ys] obs)
      log-sigmas (mapv #(mx/item (cm/get-choice (:choices %) [:log-sigma])) traces)
      sigmas (mapv js/Math.exp log-sigmas)]
  (println (str "  Inferred sigma:   " (.toFixed (mean sigmas) 3)
               " +/- " (.toFixed (std sigmas) 3)))
  (println (str "  Inferred log(sigma): " (.toFixed (mean log-sigmas) 3)
               " +/- " (.toFixed (std log-sigmas) 3))))

(println "\n-- Summary --")
(println "The NN provides deterministic structure (score=0, no random choices).")
(println "The probabilistic model adds uncertainty. MCMC infers the noise level.")
(println "nn->gen-fn bridges deep learning and probabilistic programming.")
