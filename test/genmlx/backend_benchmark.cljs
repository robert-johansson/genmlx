(ns genmlx.backend-benchmark
  "Benchmark CLJS↔NAPI dispatch overhead for mlx-node (genmlx.rs).
   Measures the dispatch path, not MLX compute (which is identical)."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]))

(defn now-ms [] (.now js/Date))

(defn bench [label n f]
  ;; Warm up
  (dotimes [_ 5] (f))
  (mx/force-gc!)
  ;; Timed run
  (let [start (now-ms)]
    (dotimes [_ n] (f))
    (let [elapsed (- (now-ms) start)
          per-op (/ elapsed n)]
      (println (str "  " label ": " (.toFixed per-op 3) "ms/op"
                    " (" n " iters, " elapsed "ms total)")))))

(println "\n=== Backend Benchmark ===\n")

;; ── 1. Scalar creation ──────────────────────────────────────────────
(println "-- Scalar creation --")
(bench "scalar float" 10000 #(mx/scalar 3.14))
(bench "scalar int" 10000 #(mx/scalar 7 mx/int32))

;; ── 2. Unary ops (number input → Either path) ──────────────────────
(println "\n-- Unary ops (number input) --")
(bench "exp(number)" 10000 #(mx/exp 2.0))
(bench "log(number)" 10000 #(mx/log 2.0))
(bench "sqrt(number)" 10000 #(mx/sqrt 9.0))

;; ── 3. Unary ops (MxArray input) ────────────────────────────────────
(println "\n-- Unary ops (MxArray input) --")
(let [x (mx/scalar 3.0)]
  (bench "exp(array)" 10000 #(mx/exp x))
  (bench "log(array)" 10000 #(mx/log x))
  (bench "sqrt(array)" 10000 #(mx/sqrt x)))

;; ── 4. Binary ops ───────────────────────────────────────────────────
(println "\n-- Binary ops --")
(let [a (mx/scalar 3.0) b (mx/scalar 4.0)]
  (bench "add(arr,arr)" 10000 #(mx/add a b))
  (bench "add(num,arr)" 10000 #(mx/add 3.0 b))
  (bench "multiply(arr,arr)" 10000 #(mx/multiply a b)))

;; ── 5. Shape creation (the BigInt64Array elimination test) ──────────
(println "\n-- Shape creation --")
(bench "zeros [100 100]" 5000 #(mx/zeros [100 100]))
(bench "ones [10 10 10]" 5000 #(mx/ones [10 10 10]))
(bench "reshape" 5000
       (let [a (mx/zeros [100])] #(mx/reshape a [10 10])))

;; ── 6. Reductions ───────────────────────────────────────────────────
(println "\n-- Reductions --")
(let [a (mx/ones [100 100])]
  (bench "sum(all)" 5000 #(mx/sum a))
  (bench "sum(axis=0)" 5000 #(mx/sum a [0]))
  (bench "mean(all)" 5000 #(mx/mean a)))

;; ── 7. Matmul ───────────────────────────────────────────────────────
(println "\n-- Matmul --")
(let [a (mx/ones [64 64]) b (mx/ones [64 64])]
  (bench "matmul 64x64" 2000 #(mx/matmul a b)))

;; ── 8. Autograd ─────────────────────────────────────────────────────
(println "\n-- Autograd --")
(let [x (mx/scalar 3.0)
      gf (mx/grad (fn [x] (mx/multiply x x)))]
  (bench "grad(x^2)" 1000 #(gf x)))

(let [x (mx/scalar 3.0)
      vgf (mx/value-and-grad (fn [x] (mx/multiply x x)))]
  (bench "value-and-grad(x^2)" 1000 #(vgf x)))

;; ── 9. PRNG ─────────────────────────────────────────────────────────
(println "\n-- PRNG --")
(let [k (rng/fresh-key 42)]
  (bench "split" 5000 #(rng/split k))
  (bench "normal [100]" 2000 #(rng/normal k [100]))
  (bench "uniform [100]" 2000 #(rng/uniform k [100])))

;; ── 10. End-to-end: linalg ───────────────────────────────────────────
(println "\n-- Linalg --")
(let [a (mx/eye 32)]
  (bench "cholesky 32x32" 1000 #(mx/cholesky a))
  (bench "inv 32x32" 1000 #(mx/inv a)))

;; ── 11. Eval + item (extraction overhead) ───────────────────────────
(println "\n-- Extraction --")
(let [x (mx/scalar 42.0)]
  (bench "eval!+item" 5000 #(do (mx/eval! x) (mx/item x))))

(println "\n=== Done ===")
