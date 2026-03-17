(ns genmlx.batched-smc-map-test
  "Test batched-smc-unfold with map-valued state."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.inference.smc :as smc])
  (:require-macros [genmlx.gen :refer [gen]]))

(println "\n=== batched-smc-unfold map-state test ===")

;; Test 1: Map-valued state
(println "\nTest 1: map-valued state kernel")
(def map-kernel
  (gen [t state]
    (let [prev-a (if state (:a state) (mx/scalar 0.0))
          prev-b (if state (:b state) (mx/scalar 1.0))
          a (trace :a (dist/gaussian prev-a 0.5))
          b (trace :b (dist/gaussian prev-b 0.5))]
      (trace :y (dist/gaussian (mx/add a b) 1.0))
      {:a a :b b})))

(def obs-seq
  [(cm/set-value cm/EMPTY :y (mx/scalar 1.5))
   (cm/set-value cm/EMPTY :y (mx/scalar 2.0))
   (cm/set-value cm/EMPTY :y (mx/scalar 2.5))
   (cm/set-value cm/EMPTY :y (mx/scalar 3.0))
   (cm/set-value cm/EMPTY :y (mx/scalar 3.5))])

(let [result (smc/batched-smc-unfold
               {:particles 200 :key (rng/fresh-key 42)}
               map-kernel nil obs-seq)
      lml (mx/item (:log-ml result))
      states (:final-states result)]
  (println (str "  log-ML: " (.toFixed lml 2)))
  (println (str "  state type: " (if (map? states) "map" "other")))
  (println (str "  :a shape: " (mx/shape (:a states))))
  (println (str "  :b shape: " (mx/shape (:b states))))
  (println (str "  :a mean: " (.toFixed (mx/item (mx/mean (:a states))) 2)))
  (println (str "  :b mean: " (.toFixed (mx/item (mx/mean (:b states))) 2)))
  (if (and (js/isFinite lml) (map? states) (= [200] (mx/shape (:a states))))
    (println "  PASS")
    (println "  FAIL")))

;; Test 2: Scalar state (backward compat)
(println "\nTest 2: scalar state (backward compat)")
(def scalar-kernel
  (gen [t state]
    (let [prev (or state (mx/scalar 0.0))
          x (trace :x (dist/gaussian prev 1.0))]
      (trace :y (dist/gaussian x 0.5))
      x)))

(let [obs [(cm/set-value cm/EMPTY :y (mx/scalar 1.0))
           (cm/set-value cm/EMPTY :y (mx/scalar 2.0))
           (cm/set-value cm/EMPTY :y (mx/scalar 3.0))]
      result (smc/batched-smc-unfold
               {:particles 100 :key (rng/fresh-key 99)}
               scalar-kernel nil obs)
      lml (mx/item (:log-ml result))]
  (println (str "  log-ML: " (.toFixed lml 2)))
  (if (js/isFinite lml)
    (println "  PASS")
    (println "  FAIL")))

(println "\nAll tests complete.")
(.exit js/process 0)
