;; Analytical Elimination with GenMLX
;; ===================================
;;
;; GenMLX auto-detects conjugate structure in your model and
;; analytically marginalizes latent variables — dramatically
;; reducing estimator variance with zero user effort.
;;
;; Demonstrates: auto-conjugacy detection (L3), variance reduction, schema introspection.
;;
;; Run: bun run --bun nbb examples/genmlx/analytical_elimination.cljs

(ns analytical-elimination
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn])
  (:require-macros [genmlx.gen :refer [gen]]))

;; --- Helpers ---

(defn mean [vs] (/ (reduce + vs) (count vs)))

(defn variance [vs]
  (let [m (mean vs)]
    (/ (reduce + (map #(* (- % m) (- % m)) vs)) (max 1 (dec (count vs))))))

(defn generate-weight [model args obs key]
  (let [{:keys [weight]} (p/generate (dyn/with-key model key) args obs)]
    (mx/eval! weight)
    (mx/item weight)))

(defn log-ml-trial [model args obs n-particles seed]
  (let [keys (rng/split-n (rng/fresh-key seed) n-particles)
        log-ws (mapv #(generate-weight model args obs %) keys)
        max-w (apply max log-ws)
        lse (+ max-w (js/Math.log (reduce + (map (fn [w] (js/Math.exp (- w max-w))) log-ws))))]
    (- lse (js/Math.log n-particles))))

;; --- Model ---

;; Gaussian mean estimation with 8 static observation sites.
;; Prior N(0, 10), likelihood N(mu, 1) — Normal-Normal conjugate.
;; Static addresses let GenMLX detect conjugacy at construction time.
(def model
  (gen [y1 y2 y3 y4 y5 y6 y7 y8]
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :y1 (dist/gaussian mu 1))
      (trace :y2 (dist/gaussian mu 1))
      (trace :y3 (dist/gaussian mu 1))
      (trace :y4 (dist/gaussian mu 1))
      (trace :y5 (dist/gaussian mu 1))
      (trace :y6 (dist/gaussian mu 1))
      (trace :y7 (dist/gaussian mu 1))
      (trace :y8 (dist/gaussian mu 1))
      mu)))

;; --- Data ---

(def ys [2.8 3.2 2.9 3.5 3.1 2.7 3.3 3.0])
(def args (mapv mx/scalar ys))
(def observations
  (cm/choicemap :y1 (mx/scalar 2.8) :y2 (mx/scalar 3.2) :y3 (mx/scalar 2.9)
                :y4 (mx/scalar 3.5) :y5 (mx/scalar 3.1) :y6 (mx/scalar 2.7)
                :y7 (mx/scalar 3.3) :y8 (mx/scalar 3.0)))

;; --- Analytic ground truth ---

(let [prior-var 100.0  n 8  sum-y (reduce + ys)
      post-var (/ 1.0 (+ (/ 1.0 prior-var) n))
      post-mean (* post-var (/ sum-y 1.0))]
  (println "\n-- Analytic posterior (computed by hand) --")
  (println (str "  mu ~ N(" (.toFixed post-mean 3) ", " (.toFixed (js/Math.sqrt post-var) 3) ")")))

;; --- What GenMLX detects ---

(println "\n-- Schema introspection --")
(let [pairs (get-in model [:schema :conjugate-pairs])]
  (if (seq pairs)
    (doseq [p pairs]
      (println (str "  Conjugate pair: " (:prior-addr p) " → " (:obs-addr p)
                   " (family: " (:family p) ")")))
    (println "  No conjugate pairs detected")))

;; --- Standard IS (analytical handlers disabled) ---

(println "\n-- Standard IS (analytical elimination disabled) --")
(let [model-plain (assoc model :schema
                         (dissoc (:schema model)
                                 :has-conjugate? :conjugate-pairs
                                 :analytical-plan :auto-handlers))
      log-mls (mapv (fn [i] (log-ml-trial model-plain args observations 200 i))
                    (range 10))]
  (println (str "  10 trials, 200 particles each"))
  (println (str "  Log-ML mean: " (.toFixed (mean log-mls) 3)))
  (println (str "  Log-ML variance: " (.toFixed (variance log-mls) 4))))

;; --- Analytical IS (auto-conjugacy active) ---

(println "\n-- Analytical IS (auto-conjugacy active) --")
(let [log-mls (mapv (fn [i] (log-ml-trial model args observations 200 i))
                    (range 10))]
  (println (str "  10 trials, 200 particles each"))
  (println (str "  Log-ML mean: " (.toFixed (mean log-mls) 3)))
  (println (str "  Log-ML variance: " (.toFixed (variance log-mls) 4))))

;; --- Summary ---

(println "\n-- Summary --")
(println "Same model, same inference call. GenMLX detects Normal-Normal conjugacy")
(println "and analytically marginalizes the prior — lower variance, same cost.")
(println "No annotations needed. Write natural models, get free performance.")
