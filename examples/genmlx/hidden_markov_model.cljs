;; Hidden Markov Model with Inference
;; ====================================
;;
;; A continuous-state HMM: hidden position evolves by random walk,
;; observed through noisy measurements. Compare IS vs MH inference.
;;
;; Demonstrates: sequential models, importance sampling, MH,
;;               posterior filtering, model structure.
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

;; --- Model ---
;; Hidden state: continuous position, random walk
;; Observation: noisy measurement at each timestep

(def hmm
  (gen [n-steps]
    (loop [t 0, state (mx/scalar 0.0)]
      (if (>= t n-steps)
        state
        (let [;; State transition: random walk
              new-state (trace (keyword (str "z" t))
                               (dist/gaussian state 0.5))
              ;; Observation: noisy measurement
              _obs      (trace (keyword (str "y" t))
                               (dist/gaussian new-state 1.0))]
          (recur (inc t) new-state))))))

;; --- Synthetic data ---
;; True states drift: 0 → 0.3 → 0.8 → 1.5 → 2.1 → 1.8

(def observations [0.5 1.2 1.8 2.5 1.5 0.9])
(def true-states  [0.3 0.8 1.5 2.1 1.8 1.2])
(def n-steps (count observations))

(def obs-cm
  (reduce (fn [cm [t y]]
            (cm/set-choice cm [(keyword (str "y" t))] (mx/scalar y)))
          cm/EMPTY
          (map-indexed vector observations)))

(println "=== Hidden Markov Model ===\n")
(println "True states:  " true-states)
(println "Observations: " observations)

;; --- Importance Sampling ---

(println "\n-- Importance Sampling (500 particles) --")
(let [{:keys [traces log-weights]}
      (is/importance-sampling {:samples 500} hmm [n-steps] obs-cm)
      ;; Normalize weights
      log-w (mx/array (mapv #(mx/item %) log-weights))
      w (mx/softmax log-w)
      _ (mx/eval! w)
      ess (/ 1.0 (mx/item (mx/sum (mx/square w))))]
  (println (str "  ESS = " (.toFixed ess 1) " / 500"))
  (doseq [t (range n-steps)]
    (let [addr (keyword (str "z" t))
          vals (mapv #(mx/item (cm/get-choice (:choices %) [addr])) traces)
          weighted-mean (reduce + (map-indexed
                                    (fn [i v] (* v (mx/item (mx/index w i))))
                                    vals))]
      (println (str "  t=" t ": E[z] = " (.toFixed weighted-mean 2)
                   "  (true: " (nth true-states t) ")")))))

;; --- MH (more accurate for longer sequences) ---

(println "\n-- MH (500 samples, 100 burn-in) --")
(let [state-sel (apply sel/select (mapv #(keyword (str "z" %)) (range n-steps)))
      traces (mcmc/mh {:samples 500 :burn 100 :selection state-sel}
                       hmm [n-steps] obs-cm)
      mean-fn (fn [vs] (/ (reduce + vs) (count vs)))]
  (doseq [t (range n-steps)]
    (let [addr (keyword (str "z" t))
          vals (mapv #(mx/item (cm/get-choice (:choices %) [addr])) traces)
          m (mean-fn vals)]
      (println (str "  t=" t ": E[z] = " (.toFixed m 2)
                   "  (true: " (nth true-states t) ")")))))
