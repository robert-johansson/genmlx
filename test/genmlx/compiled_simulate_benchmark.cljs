(ns genmlx.compiled-simulate-benchmark
  "Benchmark: compiled simulate (noise transforms + mx/compile-fn) vs handler."
  (:require [genmlx.gen :refer [gen]]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]))

(defn- force-handler
  "Strip compiled-simulate from schema, forcing handler path."
  [model]
  (let [schema (dissoc (:schema model) :compiled-simulate)]
    (assoc model :schema schema)))

(defn- benchmark-model [desc n model args]
  (let [handler-model (dyn/auto-key (force-handler model))
        compiled-model (dyn/auto-key model)
        compiled? (boolean (:compiled-simulate (:schema model)))
        k (rng/fresh-key 42)]
    (println (str "\n" desc " (compiled=" compiled? ", n=" n ")"))
    (when-not compiled?
      (println "  SKIPPED: model not compiled")
      (println))
    (when compiled?
      ;; Warm up handler
      (dotimes [_ 5]

        (mx/eval! (:score (p/simulate (dyn/with-key handler-model k) args))))
      ;; Benchmark handler
      (let [h-start (js/Date.now)]
        (dotimes [_ n]
  
          (mx/eval! (:score (p/simulate (dyn/with-key handler-model k) args))))
        (let [h-ms (- (js/Date.now) h-start)]
          ;; Warm up compiled
          (dotimes [_ 5]
    
            (mx/eval! (:score (p/simulate (dyn/with-key compiled-model k) args))))
          ;; Benchmark compiled
          (let [c-start (js/Date.now)]
            (dotimes [_ n]
      
              (mx/eval! (:score (p/simulate (dyn/with-key compiled-model k) args))))
            (let [c-ms (- (js/Date.now) c-start)
                  speedup (/ h-ms (max c-ms 1))]
              (println (str "  handler:  " h-ms "ms (" (.toFixed (/ h-ms n) 2) " ms/call)"))
              (println (str "  compiled: " c-ms "ms (" (.toFixed (/ c-ms n) 2) " ms/call)"))
              (println (str "  SPEEDUP:  " (.toFixed speedup 2) "x")))))))))

(println "=== L1-M2 Compiled Simulate Benchmark ===")
(println "Noise-transform inlining + mx/compile-fn fusion")

;; Model 1: Single gaussian
(benchmark-model "1-site gaussian" 500
  (gen [] (trace :x (dist/gaussian 0 1))) [])

;; Model 2: 3-site dependent
(benchmark-model "3-site dependent" 500
  (gen []
    (let [a (trace :a (dist/gaussian 0 10))
          b (trace :b (dist/gaussian 0 5))
          c (trace :c (dist/gaussian a 1))]
      c))
  [])

;; Model 3: Linear regression
(benchmark-model "linreg (3 sites, 1 param)" 500
  (gen [x]
    (let [slope (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 5))
          y (trace :y (dist/gaussian (mx/add (mx/multiply slope x) intercept) 1))]
      slope))
  [(mx/scalar 3.0)])

;; Model 4: Mixed distributions
(benchmark-model "5-site mixed (gauss+unif+bern)" 500
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))
          sigma (trace :sigma (dist/uniform 0.1 5))
          x (trace :x (dist/gaussian mu sigma))
          p (trace :p (dist/uniform 0.01 0.99))
          coin (trace :coin (dist/bernoulli p))]
      x))
  [])

;; Model 5: 10-site gaussian chain
(benchmark-model "10-site gaussian chain" 300
  (gen []
    (let [x0 (trace :x0 (dist/gaussian 0 10))
          x1 (trace :x1 (dist/gaussian x0 1))
          x2 (trace :x2 (dist/gaussian x1 1))
          x3 (trace :x3 (dist/gaussian x2 1))
          x4 (trace :x4 (dist/gaussian x3 1))
          x5 (trace :x5 (dist/gaussian x4 1))
          x6 (trace :x6 (dist/gaussian x5 1))
          x7 (trace :x7 (dist/gaussian x6 1))
          x8 (trace :x8 (dist/gaussian x7 1))
          x9 (trace :x9 (dist/gaussian x8 1))]
      x9))
  [])

(println "\n=== Done ===")
