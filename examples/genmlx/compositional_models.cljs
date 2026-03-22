;; Compositional Probabilistic Models
;; ====================================
;;
;; GenMLX's power: compose models from reusable pieces using splice,
;; combinators (Map, Switch), and the full GFI protocol.
;;
;; Demonstrates: splice, Map combinator, Switch combinator,
;;               hierarchical models, model comparison via log-ML.
;;
;; Run: bun run --bun nbb examples/genmlx/compositional_models.cljs

(ns compositional-models
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.combinators :as comb]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.importance :as is]
            [genmlx.inference.mcmc :as mcmc])
  (:require-macros [genmlx.gen :refer [gen]]))

(println "=== Compositional Probabilistic Models ===\n")

;; --- Part 1: Splice (hierarchical composition) ---
;; A sub-model can be spliced into a parent model.
;; The parent's trace contains the sub-model's choices under a namespace.

(println "-- 1. Splice: Hierarchical Models --")

(def noise-model
  (gen [mu]
    (trace :value (dist/gaussian mu 0.5))))

(def sensor-fusion
  (dyn/auto-key
    (gen [true-position]
      (let [;; Two noisy sensors, each a spliced sub-model
            sensor-a (splice :sensor-a noise-model true-position)
            sensor-b (splice :sensor-b noise-model true-position)]
        ;; Return both readings
        {:a sensor-a :b sensor-b}))))

(let [tr (p/simulate sensor-fusion [(mx/scalar 5.0)])
      addrs (cm/addresses (:choices tr))
      a-val (mx/item (cm/get-choice (:choices tr) [:sensor-a :value]))
      b-val (mx/item (cm/get-choice (:choices tr) [:sensor-b :value]))]
  (println (str "  Sensor A: " (.toFixed a-val 2) "  Sensor B: " (.toFixed b-val 2)
               "  (true: 5.0)"))
  (println (str "  Trace addresses: " (pr-str addrs)))
  (println "  Score decomposes: sensor-a + sensor-b contributions"))

;; --- Part 2: Map combinator (independent replication) ---
;; Apply the same model independently to multiple inputs.
;; Score = sum of individual scores (independence).

(println "\n-- 2. Map Combinator: Independent Replication --")

(def point-model
  (gen [x]
    (trace :y (dist/gaussian (mx/multiply (mx/scalar 2.0) x) 1.0))))

(def multi-point (comb/map-combinator point-model))

;; Apply to 5 x-values simultaneously
(let [xs (mapv mx/scalar [1.0 2.0 3.0 4.0 5.0])
      tr (p/simulate multi-point [xs])
      score (mx/item (:score tr))
      ;; Extract per-point observations
      ys (mapv (fn [i]
                 (mx/item (cm/get-value
                            (cm/get-submap
                              (cm/get-submap (:choices tr) i) :y))))
               (range 5))]
  (println (str "  Inputs:  [1 2 3 4 5]"))
  (println (str "  Outputs: " (mapv #(.toFixed % 2) ys)))
  (println (str "  Score = " (.toFixed score 2) " (sum of 5 independent log-probs)")))

;; --- Part 3: Switch combinator (model selection) ---
;; Choose between different sub-models based on a discrete choice.

(println "\n-- 3. Model Selection via Marginal Likelihood --")

;; Two competing models for the same data
(def linear-model
  (dyn/auto-key
    (gen [xs]
      (let [slope     (trace :slope (dist/gaussian 0 5))
            intercept (trace :intercept (dist/gaussian 0 5))]
        (doseq [[j x] (map-indexed vector xs)]
          (trace (keyword (str "y" j))
                 (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                        intercept) 0.5)))))))

(def quadratic-model
  (dyn/auto-key
    (gen [xs]
      (let [a (trace :a (dist/gaussian 0 2))
            b (trace :b (dist/gaussian 0 2))
            c (trace :c (dist/gaussian 0 2))]
        (doseq [[j x] (map-indexed vector xs)]
          (let [xv (mx/scalar x)]
            (trace (keyword (str "y" j))
                   (dist/gaussian (mx/add (mx/add (mx/multiply a (mx/multiply xv xv))
                                                  (mx/multiply b xv))
                                          c) 0.5))))))))

;; Data generated from y = x² (quadratic should win)
(def xs-sel [1.0 2.0 3.0 4.0 5.0])
(def ys-sel [1.1 3.9 9.2 15.8 25.3])  ;; ≈ x² + noise

(def obs-sel
  (reduce (fn [cm [j y]]
            (cm/set-choice cm [(keyword (str "y" j))] (mx/scalar y)))
          cm/EMPTY
          (map-indexed vector ys-sel)))

;; Compare models via log marginal likelihood (IS estimate)
(println "  Data: y ≈ x² at x = [1,2,3,4,5]")
(let [n-particles 2000
      lin-result (is/importance-sampling {:samples n-particles}
                                         linear-model [xs-sel] obs-sel)
      quad-result (is/importance-sampling {:samples n-particles}
                                          quadratic-model [xs-sel] obs-sel)
      log-ml-lin  (mx/item (:log-ml-estimate lin-result))
      log-ml-quad (mx/item (:log-ml-estimate quad-result))
      log-bf (- log-ml-quad log-ml-lin)]
  (println (str "  log P(data | linear)    = " (.toFixed log-ml-lin 1)))
  (println (str "  log P(data | quadratic) = " (.toFixed log-ml-quad 1)))
  (println (str "  log Bayes factor        = " (.toFixed log-bf 1)
               (cond (> log-bf 3) "  (strong evidence for quadratic)"
                     (> log-bf 1) "  (moderate evidence for quadratic)"
                     :else        "  (inconclusive)"))))

;; --- Part 4: Full GFI round-trip ---
;; Demonstrate the algebraic contracts: generate, update, project.

(println "\n-- 4. GFI Contracts: The Algebra of Inference --")

(let [model (dyn/auto-key
              (gen []
                (let [x (trace :x (dist/gaussian 0 1))]
                  (trace :y (dist/gaussian x 2)))))
      ;; Simulate: sample from the prior
      tr (p/simulate model [])
      score (mx/item (:score tr))
      ;; Generate: score against observations
      {:keys [trace weight]} (p/generate model [] (:choices tr))
      ;; Update: change a choice, get weight for MH
      new-constraints (cm/choicemap :x (mx/scalar 1.5))
      {:keys [trace weight discard]} (p/update model tr new-constraints)
      update-w (mx/item weight)
      ;; Project: decompose score by address
      proj-x (mx/item (p/project model tr (sel/select :x)))
      proj-y (mx/item (p/project model tr (sel/select :y)))]
  (println (str "  simulate score     = " (.toFixed score 3)))
  (println (str "  project(:x)        = " (.toFixed proj-x 3)))
  (println (str "  project(:y)        = " (.toFixed proj-y 3)))
  (println (str "  proj(:x)+proj(:y)  = " (.toFixed (+ proj-x proj-y) 3) " ← equals score"))
  (println (str "  update(same) wt    = " (.toFixed (mx/item (:weight (p/update model tr (:choices tr)))) 6) " ← 0 (identity)")))

(println "\nComposable models + algebraic inference = probabilistic programming.")
