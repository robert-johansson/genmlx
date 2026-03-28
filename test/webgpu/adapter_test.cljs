;; Test the WebGPU mx/ adapter against the same API as MLX
;; Run: nbb -cp src-webgpu test/webgpu/adapter_test.cljs

(ns webgpu.adapter-test
  (:require [promesa.core :as p]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]))

(def pass-count (atom 0))
(def fail-count (atom 0))

(defn assert-close [desc expected actual tol]
  (let [ok? (< (js/Math.abs (- expected actual)) tol)]
    (if ok?
      (do (swap! pass-count inc)
          (println "  PASS:" desc))
      (do (swap! fail-count inc)
          (println "  FAIL:" desc "expected" expected "got" actual)))))

(defn assert-true [desc v]
  (if v
    (do (swap! pass-count inc)
        (println "  PASS:" desc))
    (do (swap! fail-count inc)
        (println "  FAIL:" desc))))

(defn assert-equal [desc expected actual]
  (if (= expected actual)
    (do (swap! pass-count inc)
        (println "  PASS:" desc))
    (do (swap! fail-count inc)
        (println "  FAIL:" desc "expected" expected "got" actual))))

(defn item-async
  "Async item: await .data() then extract first element."
  [a]
  (-> (.data a)
      (.then (fn [d] (if (= 1 (.-length d)) (aget d 0) (vec d))))))

(-> (mx/init!)
    (.then
     (fn [_]
       (println "\n=== WebGPU mx/ Adapter Test ===\n")

       (println "-- Array creation --")
       (let [s (mx/scalar 42.0)]
         (assert-equal "scalar shape" [] (mx/shape s)))
       (let [a (mx/array [1 2 3 4])]
         (assert-equal "array shape" [4] (mx/shape a)))
       (let [z (mx/zeros [2 3])]
         (assert-equal "zeros shape" [2 3] (mx/shape z)))
       (let [o (mx/ones [3])]
         (assert-equal "ones shape" [3] (mx/shape o)))
       (let [e (mx/eye 3)]
         (assert-equal "eye shape" [3 3] (mx/shape e)))
       (let [r (mx/arange 5)]
         (assert-equal "arange shape" [5] (mx/shape r)))

       (println "\n-- Arithmetic --")
       (p/let [a (mx/array [1 2 3])
               b (mx/array [4 5 6])
               c (mx/add a b)
               d (item-async c)]
         (assert-equal "add" [5 7 9] d))

       (p/let [a (mx/array [2 3 4])
               b (mx/array [10 20 30])
               c (mx/multiply a b)
               d (item-async c)]
         (assert-equal "multiply" [20 60 120] d))

       (p/let [a (mx/scalar 5.0)
               b (mx/scalar 3.0)
               c (mx/subtract a b)
               d (item-async c)]
         (assert-close "subtract" 2.0 d 0.001))

       (p/let [a (mx/scalar 10.0)
               b (mx/scalar 4.0)
               c (mx/divide a b)
               d (item-async c)]
         (assert-close "divide" 2.5 d 0.001))

       (p/let [a (mx/scalar 3.0)
               c (mx/square a)
               d (item-async c)]
         (assert-close "square" 9.0 d 0.001))

       (p/let [a (mx/scalar 9.0)
               c (mx/sqrt a)
               d (item-async c)]
         (assert-close "sqrt" 3.0 d 0.001))

       (println "\n-- Math functions --")
       (p/let [a (mx/scalar 1.0)
               d (item-async (mx/exp a))]
         (assert-close "exp(1)" js/Math.E d 0.001))

       (p/let [a (mx/scalar js/Math.E)
               d (item-async (mx/log a))]
         (assert-close "log(e)" 1.0 d 0.001))

       (p/let [a (mx/scalar 0.0)
               d (item-async (mx/sin a))]
         (assert-close "sin(0)" 0.0 d 0.001))

       (p/let [a (mx/scalar 0.0)
               d (item-async (mx/cos a))]
         (assert-close "cos(0)" 1.0 d 0.001))

       (p/let [a (mx/scalar 0.5)
               d (item-async (mx/erf a))]
         (assert-close "erf(0.5)" 0.5205 d 0.01))

       (println "\n-- Reductions --")
       (p/let [a (mx/array [1 2 3 4 5])
               d (item-async (mx/sum a))]
         (assert-close "sum" 15.0 d 0.001))

       (p/let [a (mx/array [2 4 6])
               d (item-async (mx/mean a))]
         (assert-close "mean" 4.0 d 0.001))

       (p/let [a (mx/array [3 1 4 1 5])
               d (item-async (mx/amax a))]
         (assert-close "amax" 5.0 d 0.001))

       (println "\n-- Comparison --")
       (p/let [a (mx/scalar 3.0)
               b (mx/scalar 5.0)
               c (mx/where (mx/greater a b) a b)
               d (item-async c)]
         (assert-close "where(3>5, 3, 5)" 5.0 d 0.001))

       (println "\n-- Shape manipulation --")
       (let [a (mx/array [1 2 3 4 5 6])
             b (mx/reshape a [2 3])]
         (assert-equal "reshape" [2 3] (mx/shape b)))

       (let [a (mx/array [[1 2] [3 4]])
             b (mx/transpose a)]
         (assert-equal "transpose shape" [2 2] (mx/shape b)))

       (println "\n-- Matrix operations --")
       (p/let [a (mx/array [[1 2] [3 4]])
               b (mx/array [[5 6] [7 8]])
               c (mx/matmul a b)
               d (item-async c)]
         (assert-equal "matmul" [19 22 43 50] d))

       (println "\n-- Linear algebra --")
       (p/let [a (mx/array [[4 2] [2 3]])
               L (mx/cholesky a)
               d (item-async L)]
         (assert-close "cholesky [0,0]" 2.0 (nth d 0) 0.01))

       (println "\n-- Autograd --")
       (p/let [loss (fn [x] (mx/sum (mx/square x)))
               grad-fn (mx/grad loss)
               x (mx/array [3.0 4.0])
               g (grad-fn x)
               d (item-async g)]
         (assert-equal "grad" [6 8] (mapv #(js/Math.round %) d)))

       (p/let [loss (fn [x] (mx/sum (mx/square x)))
               vg (mx/value-and-grad loss)
               [v g] (vg (mx/array [1.0 2.0 3.0]))
               vd (item-async v)
               gd (item-async g)]
         (assert-close "value-and-grad value" 14.0 vd 0.001)
         (assert-equal "value-and-grad grad" [2 4 6] (mapv #(js/Math.round %) gd)))

       (println "\n-- Transforms --")
       (p/let [f (mx/compile-fn (fn [x y] (mx/sqrt (mx/add (mx/square x) (mx/square y)))))
               r (f (mx/scalar 3.0) (mx/scalar 4.0))
               d (item-async r)]
         (assert-close "compile-fn (jit)" 5.0 d 0.001))

       (println "\n-- Random PRNG --")
       (let [k (rng/fresh-key 42)
             [k1 _k2] (rng/split k)]
         (assert-true "key shape" (= [2] (mx/shape k)))
         (assert-true "split key shape" (= [2] (mx/shape k1))))

       (p/let [k (rng/fresh-key 42)
               [k1 _k2] (rng/split k)
               samples (rng/normal k1 [1000])
               m (mx/mean samples)
               d (item-async m)]
         (assert-close "normal mean ≈ 0" 0.0 d 0.15))

       (p/let [k (rng/fresh-key 42)
               [k1 _k2] (rng/split k)
               samples (rng/uniform k1 [1000])
               m (mx/mean samples)
               d (item-async m)]
         (assert-close "uniform mean ≈ 0.5" 0.5 d 0.1))

       ;; Summary
       (.then (js/Promise.resolve nil)
              (fn [_]
                (println (str "\n=== Results: " @pass-count " passed, " @fail-count " failed ==="))
                (when (pos? @fail-count)
                  (js/process.exit 1)))))))
