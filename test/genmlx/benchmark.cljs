(ns genmlx.benchmark
  "Performance benchmarks for GenMLX."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.mcmc :as mcmc])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn time-it [label f]
  (let [start (js/Date.now)
        result (f)
        elapsed (- (js/Date.now) start)]
    (println (str "  " label ": " elapsed "ms"))
    result))

(println "\n=== GenMLX Benchmarks ===\n")

;; Benchmark 1: Simulate throughput
(println "-- Simulate throughput --")
(let [model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 1))
                    y (dyn/trace :y (dist/gaussian 0 1))]
                (mx/eval! x y)
                (+ (mx/item x) (mx/item y))))]
  (time-it "1000 simulations"
    (fn [] (dotimes [_ 1000] (p/simulate model [])))))

;; Benchmark 2: Generate + score
(println "\n-- Generate throughput --")
(let [model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 1))
                    y (dyn/trace :y (dist/gaussian 0 1))]
                (mx/eval! x y)
                (+ (mx/item x) (mx/item y))))
      constraints (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0))]
  (time-it "1000 generates"
    (fn [] (dotimes [_ 1000] (p/generate model [] constraints)))))

;; Benchmark 3: MH on 5-param model
(println "\n-- MH inference (5-param model) --")
(let [model (gen [xs]
              (let [slope     (dyn/trace :slope (dist/gaussian 0 10))
                    intercept (dyn/trace :intercept (dist/gaussian 0 10))]
                (mx/eval! slope intercept)
                (let [s (mx/item slope) i (mx/item intercept)]
                  (doseq [[j x] (map-indexed vector xs)]
                    (dyn/trace (keyword (str "y" j))
                               (dist/gaussian (+ (* s x) i) 1)))
                  [s i])))
      xs [1.0 2.0 3.0 4.0 5.0]
      observations (reduce (fn [cm [j y]]
                             (cm/set-choice cm [(keyword (str "y" j))] (mx/scalar y)))
                           cm/EMPTY
                           (map-indexed vector [2.1 3.9 6.2 7.8 10.1]))]
  (time-it "500 MH samples (burn=100)"
    (fn [] (mcmc/mh {:samples 500 :burn 100
                     :selection (sel/select :slope :intercept)}
                    model [xs] observations))))

;; Benchmark 4: MLX array operations
(println "\n-- MLX array operations --")
(time-it "10000 scalar ops (add/multiply/exp)"
  (fn []
    (let [a (mx/scalar 1.0)
          b (mx/scalar 2.0)]
      (dotimes [_ 10000]
        (let [c (mx/add a b)
              d (mx/multiply c a)
              e (mx/exp d)]
          (mx/eval! e))))))

(println "\nBenchmarks complete.")
