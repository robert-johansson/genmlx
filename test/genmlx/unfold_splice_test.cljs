(ns genmlx.unfold-splice-test
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.combinators :as comb])
  (:require-macros [genmlx.gen :refer [gen]]))

(def kernel
  (dyn/auto-key
    (gen [t state a b]
      (let [mean (if state
                   (mx/add (mx/multiply a (:x state)) b)
                   (mx/scalar 0.0))
            x (trace :x (dist/gaussian mean 1.0))]
        {:x x}))))

(def unfold (comb/unfold-combinator kernel))

(println "Test 1: Unfold simulate with extra args")
(let [trace (p/simulate unfold [3 nil (mx/scalar 0.5) (mx/scalar 1.0)])]
  (println (str "  retval: " (count (:retval trace)) " states"))
  (println "  PASS"))

(println "Test 2: splice with extra args")
(def outer
  (dyn/auto-key
    (gen []
      (let [a (trace :a (dist/gaussian (mx/scalar 0.5) 0.1))
            b (trace :b (dist/gaussian (mx/scalar 1.0) 0.1))]
        (splice :seq unfold 3 nil a b)))))

(let [trace (p/simulate outer [])]
  (println (str "  score: " (.toFixed (mx/item (:score trace)) 2)))
  (println "  PASS"))

(println "Test 3: generate with observations via splice")
(let [obs (-> cm/EMPTY
              (cm/set-choice [:seq 0 :x] (mx/scalar 1.0))
              (cm/set-choice [:seq 1 :x] (mx/scalar 2.0))
              (cm/set-choice [:seq 2 :x] (mx/scalar 3.0)))
      {:keys [weight]} (p/generate outer [] obs)]
  (println (str "  weight: " (.toFixed (mx/item weight) 2)))
  (println "  PASS"))

(println "\nAll Unfold splice tests passed!")
(.exit js/process 0)
