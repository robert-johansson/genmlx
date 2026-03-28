;; Smoke test: jax-js on WebGPU via Dawn (Metal), called from nbb
;; Run: nbb test/webgpu/smoke_test.cljs

(ns webgpu.smoke-test
  (:require [promesa.core :as p]))

;; --- Bootstrap WebGPU via Dawn ---
(def dawn (js/require "webgpu"))
(let [gpu-globals (.-globals dawn)
      gpu         (.create dawn #js [])]
  (js/Object.assign js/globalThis gpu-globals)
  (js/Object.defineProperty js/globalThis "navigator"
    #js {:value #js {:gpu gpu} :writable true :configurable true}))

(p/let [jax     (js/import "@jax-js/jax")
        np      (.-numpy jax)
        grad-fn (.-grad jax)
        vag-fn  (.-valueAndGrad jax)
        jit-fn  (.-jit jax)
        rng     (.-random jax)
        _       (.init jax)
        _       (do (.defaultDevice jax "webgpu")
                    (println "\nUsing WebGPU (Metal) backend via jax-js\n"))]

  ;; Test 1: Basic arithmetic
  (println "-- Test 1: Basic arithmetic --")
  (p/let [a (.array np #js [1 2 3 4])
          b (.array np #js [10 20 30 40])
          c (.add (.-ref a) (.mul a b))
          d (.data c)]
    (println "a + a*b =" d))

  ;; Test 2: Reductions
  (println "\n-- Test 2: Reductions --")
  (p/let [x (.array np #js [1 2 3 4 5])
          d (.data (.sum np x))]
    (println "sum([1..5]) =" d))

  ;; Test 3: Math functions
  (println "\n-- Test 3: Math functions --")
  (p/let [d (.data (.exp np (.array np #js [0 1 2])))]
    (println "exp([0,1,2]) =" d))

  ;; Test 4: grad
  (println "\n-- Test 4: grad --")
  (p/let [loss   (fn [x] (.sum np (.square np x)))
          g-loss (grad-fn loss)
          result (g-loss (.array np #js [3 4]))
          d      (.data result)]
    (println "grad(sum(x^2)) at [3,4] =" d))

  ;; Test 5: value_and_grad
  (println "\n-- Test 5: value_and_grad --")
  (p/let [loss (fn [x] (.sum np (.square np x)))
          vg   (vag-fn loss)
          res  (vg (.array np #js [1 2 3]))
          vd   (.data (aget res 0))
          gd   (.data (aget res 1))]
    (println "value =" vd " grad =" gd))

  ;; Test 6: jit
  (println "\n-- Test 6: jit --")
  (p/let [f (jit-fn (fn [x y]
                      (.sqrt np (.add (.square np (.-ref x))
                                      (.square np y)))))
          r (f (.array np #js [3]) (.array np #js [4]))
          d (.data r)]
    (println "sqrt(3^2 + 4^2) =" d))

  ;; Test 7: matmul
  (println "\n-- Test 7: Matmul --")
  (p/let [a (.array np #js [#js [1 2] #js [3 4]])
          b (.array np #js [#js [5 6] #js [7 8]])
          d (.data (.dot np a b))]
    (println "matmul =" d))

  ;; Test 8: Random PRNG
  (println "\n-- Test 8: Random PRNG --")
  (p/let [k  (.key rng 42)
          ks (.split rng k)
          k1 (.take np (.-ref ks) (.array np 0) 0)
          d  (.data (.normal rng k1 #js [5]))]
    (println "normal samples =" d))

  (println "\n=== All nbb/jax-js smoke tests passed ==="))
