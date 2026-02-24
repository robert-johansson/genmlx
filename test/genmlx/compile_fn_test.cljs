(ns compile-fn-test
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]))

(println "\n=== Test 1: Does compile-fn re-execute the JS body on each call? ===\n")

(let [call-count (atom 0)
      f (mx/compile-fn
          (fn [x]
            (swap! call-count inc)
            (println (str "  JS body executed! (call #" @call-count ")"))
            (mx/add x (mx/multiply x x))))
      _ (println "Calling f first time:")
      r1 (f (mx/array 1.0))
      _ (mx/eval! r1)
      _ (println (str "  Result: " (mx/item r1)))
      _ (println "Calling f second time (same shape):")
      r2 (f (mx/array 2.0))
      _ (mx/eval! r2)
      _ (println (str "  Result: " (mx/item r2)))
      _ (println "Calling f third time (same shape):")
      r3 (f (mx/array 3.0))
      _ (mx/eval! r3)
      _ (println (str "  Result: " (mx/item r3)))
      _ (println "Calling f fourth time (different shape — vector):")
      r4 (f (mx/array [1.0 2.0 3.0]))
      _ (mx/eval! r4)
      _ (println (str "  Result: " (vec (mx/->clj r4))))]
  (println (str "\nTotal JS body executions: " @call-count " (out of 4 calls)")))

(println "\n=== Test 2: compile-fn with grad — does body re-execute? ===\n")

(let [call-count (atom 0)
      score-fn (fn [x]
                 (swap! call-count inc)
                 ;; -0.5 * x^2  (unnormalized gaussian log-prob)
                 (mx/multiply (mx/array -0.5) (mx/multiply x x)))
      grad-fn (mx/grad score-fn)
      compiled-grad (mx/compile-fn grad-fn)
      _ (println "Calling compiled-grad first time:")
      g1 (compiled-grad (mx/array 1.0))
      _ (mx/eval! g1)
      _ (println (str "  Gradient at x=1: " (mx/item g1) " (expected: -1.0)"))
      _ (reset! call-count 0)
      _ (println "Calling compiled-grad second time:")
      g2 (compiled-grad (mx/array 2.0))
      _ (mx/eval! g2)
      _ (println (str "  Gradient at x=2: " (mx/item g2) " (expected: -2.0)"))
      _ (println "Calling compiled-grad third time:")
      g3 (compiled-grad (mx/array 3.0))
      _ (mx/eval! g3)
      _ (println (str "  Gradient at x=3: " (mx/item g3) " (expected: -3.0)"))]
  (println (str "\nBody executions after first call: " @call-count " (out of 2 subsequent calls)")))

(println "\n=== Test 3: compile-fn with vmap — does body re-execute? ===\n")

(let [call-count (atom 0)
      f (fn [x]
          (swap! call-count inc)
          (mx/add x (mx/multiply x x)))
      vmapped (mx/vmap f)
      compiled-vmapped (mx/compile-fn vmapped)
      _ (println "Calling compiled-vmap first time:")
      r1 (compiled-vmapped (mx/array [1.0 2.0 3.0]))
      _ (mx/eval! r1)
      _ (println (str "  Result: " (vec (mx/->clj r1)) " (expected: [2 6 12])"))
      _ (reset! call-count 0)
      _ (println "Calling compiled-vmap second time:")
      r2 (compiled-vmapped (mx/array [4.0 5.0 6.0]))
      _ (mx/eval! r2)
      _ (println (str "  Result: " (vec (mx/->clj r2)) " (expected: [20 30 42])"))]
  (println (str "\nBody executions after first call: " @call-count " (out of 1 subsequent call)")))

(println "\n=== Test 4: compile-fn with vmap + grad — the triple transform ===\n")

(let [call-count (atom 0)
      score-fn (fn [x]
                 (swap! call-count inc)
                 (mx/multiply (mx/array -0.5) (mx/multiply x x)))
      ;; Triple: compile(vmap(grad(f)))
      triple (mx/compile-fn (mx/vmap (mx/grad score-fn)))
      _ (println "Calling triple transform first time:")
      g1 (triple (mx/array [1.0 2.0 3.0]))
      _ (mx/eval! g1)
      _ (println (str "  Gradients: " (vec (mx/->clj g1)) " (expected: [-1 -2 -3])"))
      _ (reset! call-count 0)
      _ (println "Calling triple transform second time:")
      g2 (triple (mx/array [4.0 5.0 6.0]))
      _ (mx/eval! g2)
      _ (println (str "  Gradients: " (vec (mx/->clj g2)) " (expected: [-4 -5 -6])"))]
  (println (str "\nBody executions after first call: " @call-count " (out of 1 subsequent call)")))

(println "\n=== Test 5: Timing — compiled pure fn vs raw fn (1000 calls) ===\n")

(let [raw-fn (fn [x] (mx/add x (mx/multiply x x)))
      compiled-fn (mx/compile-fn raw-fn)
      input (mx/array 2.0)
      ;; Warmup
      _ (dotimes [_ 10] (mx/eval! (raw-fn input)))
      _ (dotimes [_ 10] (mx/eval! (compiled-fn input)))
      ;; Benchmark raw
      t0 (js/performance.now)
      _ (dotimes [_ 1000]
          (mx/eval! (raw-fn input)))
      t1 (js/performance.now)
      raw-time (- t1 t0)
      ;; Benchmark compiled
      t2 (js/performance.now)
      _ (dotimes [_ 1000]
          (mx/eval! (compiled-fn input)))
      t3 (js/performance.now)
      compiled-time (- t3 t2)]
  (println (str "  Raw function (1000 calls):      " (.toFixed raw-time 1) " ms (" (.toFixed (/ raw-time 1000) 3) " ms/call)"))
  (println (str "  Compiled function (1000 calls):  " (.toFixed compiled-time 1) " ms (" (.toFixed (/ compiled-time 1000) 3) " ms/call)"))
  (println (str "  Speedup: " (.toFixed (/ raw-time compiled-time) 2) "x")))

(println "\n=== Test 6: Timing — compiled vs raw with more complex function ===\n")

(let [;; Simulate a simple 5-site score function (pure MLX ops)
      raw-fn (fn [params]
               (let [x0 (mx/take-idx params 0)
                     x1 (mx/take-idx params 1)
                     x2 (mx/take-idx params 2)
                     x3 (mx/take-idx params 3)
                     x4 (mx/take-idx params 4)
                     lp0 (mx/multiply (mx/array -0.5) (mx/multiply x0 x0))
                     lp1 (mx/multiply (mx/array -0.5) (mx/multiply x1 x1))
                     lp2 (mx/multiply (mx/array -0.5) (mx/multiply x2 x2))
                     ;; Dependent: x3's "mu" depends on x0+x1
                     mu3 (mx/add x0 x1)
                     diff3 (mx/subtract x3 mu3)
                     lp3 (mx/multiply (mx/array -0.5) (mx/multiply diff3 diff3))
                     ;; Dependent: x4's "mu" depends on x2
                     mu4 (mx/multiply (mx/array 2.0) x2)
                     diff4 (mx/subtract x4 mu4)
                     lp4 (mx/multiply (mx/array -0.5) (mx/multiply diff4 diff4))]
                 (mx/add lp0 (mx/add lp1 (mx/add lp2 (mx/add lp3 lp4))))))
      compiled-fn (mx/compile-fn raw-fn)
      compiled-grad (mx/compile-fn (mx/grad raw-fn))
      input (mx/array [0.5 -0.3 1.2 0.8 -0.5])
      ;; Warmup
      _ (dotimes [_ 10] (mx/eval! (raw-fn input)))
      _ (dotimes [_ 10] (mx/eval! (compiled-fn input)))
      _ (dotimes [_ 10] (mx/eval! (compiled-grad input)))
      ;; Benchmark raw
      t0 (js/performance.now)
      _ (dotimes [_ 1000] (mx/eval! (raw-fn input)))
      t1 (js/performance.now)
      raw-time (- t1 t0)
      ;; Benchmark compiled score
      t2 (js/performance.now)
      _ (dotimes [_ 1000] (mx/eval! (compiled-fn input)))
      t3 (js/performance.now)
      compiled-time (- t3 t2)
      ;; Benchmark compiled grad
      t4 (js/performance.now)
      _ (dotimes [_ 1000] (mx/eval! (compiled-grad input)))
      t5 (js/performance.now)
      grad-time (- t5 t4)]
  (println (str "  5-site score function:"))
  (println (str "  Raw (1000 calls):            " (.toFixed raw-time 1) " ms (" (.toFixed (/ raw-time 1000) 3) " ms/call)"))
  (println (str "  Compiled (1000 calls):        " (.toFixed compiled-time 1) " ms (" (.toFixed (/ compiled-time 1000) 3) " ms/call)"))
  (println (str "  Compiled+grad (1000 calls):   " (.toFixed grad-time 1) " ms (" (.toFixed (/ grad-time 1000) 3) " ms/call)"))
  (println (str "  Score speedup: " (.toFixed (/ raw-time compiled-time) 2) "x"))
  (println (str "  vs current score-fn (1.41ms): " (.toFixed (/ 1.41 (/ compiled-time 1000)) 0) "x potential")))

(println "\nDone.")
