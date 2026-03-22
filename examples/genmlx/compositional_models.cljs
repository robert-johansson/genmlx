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

(println "\n-- 3. Switch Combinator: Model Selection --")

(def linear-model
  (gen [x]
    (let [slope (trace :slope (dist/gaussian 2 1))]
      (trace :y (dist/gaussian (mx/multiply slope x) 0.5)))))

(def quadratic-model
  (gen [x]
    (let [a (trace :a (dist/gaussian 1 0.5))]
      (trace :y (dist/gaussian (mx/multiply a (mx/multiply x x)) 0.5)))))

(def model-selector
  (dyn/auto-key
    (gen [x]
      (let [;; Prior: 50/50 between linear and quadratic
            choice (trace :model-choice (dist/bernoulli 0.5))]
        ;; Branch on model choice
        (if (> (mx/item choice) 0.5)
          (splice :model linear-model x)
          (splice :model quadratic-model x))))))

;; Generate data from the quadratic model (true: y = x^2)
(def test-data
  (cm/from-map {:y (mx/scalar 8.5)}))  ;; x=3 → y ≈ 9

;; Compare models via importance sampling
(println "  Data: x=3, y=8.5 (consistent with y=x²=9, not y=2x=6)")
(let [n-samples 1000
      {:keys [traces log-weights]}
      (is/importance-sampling {:samples n-samples}
                              model-selector [(mx/scalar 3.0)]
                              (cm/from-map {:model (cm/from-map {:y (mx/scalar 8.5)})}))
      ;; Count how many particles chose each model
      choices (mapv #(mx/item (cm/get-choice (:choices %) [:model-choice])) traces)
      n-linear (count (filter #(> % 0.5) choices))
      n-quad   (- n-samples n-linear)]
  (println (str "  Prior:     linear 50% / quadratic 50%"))
  (println (str "  Posterior: linear " (.toFixed (* 100 (/ n-linear n-samples)) 0)
               "% / quadratic " (.toFixed (* 100 (/ n-quad n-samples)) 0) "%")))

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
