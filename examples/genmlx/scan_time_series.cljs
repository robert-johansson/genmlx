;; Scan Combinator for Time Series with GenMLX
;; =============================================
;;
;; A noisy random walk observed through sensors — a classic state-space model.
;; Define a single-step kernel, Scan handles temporal iteration.
;; Run importance sampling to recover hidden states from observations.
;;
;; Demonstrates: Scan combinator, temporal models, IS filtering, trace indexing.
;;
;; Run: bun run --bun nbb examples/genmlx/scan_time_series.cljs

(ns scan-time-series
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.combinators :as comb]
            [genmlx.inference.importance :as is])
  (:require-macros [genmlx.gen :refer [gen]]))

;; --- Helpers ---

(defn mean [vs] (/ (reduce + vs) (count vs)))

(defn weighted-mean [vals log-ws]
  (let [max-w (apply max log-ws)
        ws (mapv #(js/Math.exp (- % max-w)) log-ws)
        s (reduce + ws)]
    (/ (reduce + (map * vals ws)) s)))

;; --- Single-step kernel ---

;; State evolves as a random walk: z_t ~ N(z_{t-1}, 0.5)
;; Observations are noisy: y_t ~ N(z_t, 1.0)
;; The kernel takes [carry input] and returns [new-carry output].
(def step-kernel
  (dyn/auto-key
    (gen [state _input]
      (let [z (trace :z (dist/gaussian state 0.5))
            y (trace :y (dist/gaussian z 1.0))]
        [z y]))))

;; Scan wraps the kernel into a full temporal model
(def ssm (comb/scan-combinator step-kernel))

;; --- Forward simulation ---

(println "\n-- Forward simulation (8 timesteps) --")
(let [init (mx/scalar 0.0)
      inputs (vec (repeat 8 (mx/scalar 0.0)))
      trace (p/simulate (dyn/auto-key ssm) [init inputs])]
  (println "Trace structure: choices indexed [t :z] and [t :y]")
  (doseq [t (range 8)]
    (let [z (mx/item (cm/get-choice (:choices trace) [t :z]))
          y (mx/item (cm/get-choice (:choices trace) [t :y]))]
      (println (str "  t=" t "  z=" (.toFixed z 2) "  y=" (.toFixed y 2))))))

;; --- Observations (true hidden states drift upward) ---

(def true-states [0.3 0.8 1.5 2.0 1.7 1.2 0.6 0.1])
(def obs-values  [0.5 1.2 1.8 2.5 1.5 0.9 0.3 -0.2])

(def observations
  (reduce (fn [cm t]
            (cm/set-choice cm [t :y] (mx/scalar (nth obs-values t))))
          cm/EMPTY
          (range 8)))

;; --- Importance sampling to recover hidden states ---

(println "\n-- IS filtering (500 particles) --")
(let [init (mx/scalar 0.0)
      inputs (vec (repeat 8 (mx/scalar 0.0)))
      result (is/importance-sampling {:samples 500}
                                     (dyn/auto-key ssm)
                                     [init inputs]
                                     observations)
      traces (:traces result)
      log-ws (mapv #(mx/item %) (:log-weights result))]
  (println (str "  Log-ML estimate: " (.toFixed (mx/item (:log-ml-estimate result)) 3)))
  (println)
  (println "  t  true_z   obs_y   E[z|y]")
  (println "  ─  ──────   ─────   ──────")
  (doseq [t (range 8)]
    (let [z-vals (mapv #(mx/item (cm/get-choice (:choices %) [t :z])) traces)
          ez (weighted-mean z-vals log-ws)]
      (println (str "  " t "   "
                   (.toFixed (nth true-states t) 2) "    "
                   (.toFixed (nth obs-values t) 2) "    "
                   (.toFixed ez 2))))))

;; --- Summary ---

(println "\n-- Summary --")
(println "Define one step, Scan builds the full temporal model.")
(println "Correct trace indexing, inference decomposition, and")
(println "combinator optimizations — all for free.")
