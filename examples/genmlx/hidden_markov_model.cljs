;; Hidden Markov Model with Inference
;; ====================================
;;
;; A continuous-state HMM: hidden position evolves by random walk,
;; observed through noisy measurements.
;;
;; Demonstrates: sequential models, importance sampling,
;;               gradient-informed MCMC (HMC), posterior filtering.
;;
;; Run: bun run --bun nbb examples/genmlx/hidden_markov_model.cljs

(ns hidden-markov-model
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.importance :as is]
            [genmlx.inference.mcmc :as mcmc])
  (:require-macros [genmlx.gen :refer [gen]]))

;; --- Helpers ---

(defn mean [vs] (/ (reduce + vs) (count vs)))

;; --- Model ---
;; Hidden state: continuous position, random walk
;; Observation: noisy measurement at each timestep

(def hmm
  (gen [n-steps]
    (loop [t 0, state (mx/scalar 0.0)]
      (if (>= t n-steps)
        state
        (let [new-state (trace (keyword (str "z" t))
                               (dist/gaussian state 0.5))
              _obs      (trace (keyword (str "y" t))
                               (dist/gaussian new-state 1.0))]
          (recur (inc t) new-state))))))

;; --- Synthetic data ---

(def observations [0.5 1.2 1.8 2.5 1.5 0.9])
(def true-states  [0.3 0.8 1.5 2.1 1.8 1.2])
(def n-steps (count observations))
(def state-addrs (mapv #(keyword (str "z" %)) (range n-steps)))

(def obs-cm
  (reduce (fn [cm [t y]]
            (cm/set-choice cm [(keyword (str "y" t))] (mx/scalar y)))
          cm/EMPTY
          (map-indexed vector observations)))

(println "=== Hidden Markov Model ===\n")
(println "True states:  " true-states)
(println "Observations: " observations)

;; --- Importance Sampling ---

(println "\n-- Importance Sampling (1000 particles) --")
(let [{:keys [traces log-weights]}
      (is/importance-sampling {:samples 1000} hmm [n-steps] obs-cm)
      log-w (mx/array (mapv #(mx/item %) log-weights))
      w (mx/softmax log-w)
      _ (mx/eval! w)
      ess (/ 1.0 (mx/item (mx/sum (mx/square w))))]
  (println (str "  ESS = " (.toFixed ess 1) " / 1000"))
  (doseq [t (range n-steps)]
    (let [addr (keyword (str "z" t))
          vals (mapv #(mx/item (cm/get-value (cm/get-submap (:choices %) addr))) traces)
          weighted-mean (reduce + (map-indexed
                                    (fn [i v] (* v (mx/item (mx/index w i))))
                                    vals))]
      (println (str "  t=" t ": E[z] = " (.toFixed weighted-mean 2)
                   "  (true: " (nth true-states t) ")")))))

;; --- HMC (gradient-informed, handles correlated latents well) ---
;; HMC uses Hamiltonian dynamics to propose all 6 states jointly,
;; avoiding the poor mixing of per-site random-walk MH.

(println "\n-- HMC (300 samples, 100 burn-in) --")
(let [samples (mcmc/hmc {:samples 300 :burn 100
                         :step-size 0.1 :leapfrog-steps 10
                         :addresses state-addrs
                         :compile? false}
                        hmm [n-steps] obs-cm)]
  (doseq [t (range n-steps)]
    (let [vals (mapv #(nth % t) samples)]
      (println (str "  t=" t ": E[z] = " (.toFixed (mean vals) 2)
                   "  (true: " (nth true-states t) ")")))))
