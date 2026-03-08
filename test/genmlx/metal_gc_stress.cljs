(ns genmlx.metal-gc-stress
  "Stress test to verify Metal buffer cleanup under sustained workload.
   Tests four strategies: sync, gc+sweep, async, and sweep-only."
  (:require [genmlx.mlx :as mx]))

(defn create-batch [batch-size]
  (dotimes [_ batch-size]
    (let [a (mx/ones [100 100])
          b (mx/ones [100 100])
          c (mx/add a b)]
      (mx/eval! c))))

(defn report [i max-wrappers]
  (let [wrappers (mx/get-wrappers-count)
        active-mb (/ (mx/get-active-memory) 1048576.0)]
    (println (str "  i=" i
                  " wrappers=" wrappers
                  " active=" (.toFixed active-mb 1) "MB"
                  " peak-wrappers=" (max max-wrappers wrappers)))
    (max max-wrappers wrappers)))

(defn stress-test-sync [n batch-size]
  (println "\n=== Test 1: Sync (no cleanup) ===")
  (println (str "Initial wrappers: " (mx/get-wrappers-count)))
  (loop [i 0 max-w 0]
    (when (< i n)
      (create-batch batch-size)
      (let [new-max (if (zero? (mod i (* batch-size 10)))
                      (report i max-w)
                      (max max-w (mx/get-wrappers-count)))]
        (recur (+ i batch-size) new-max))))
  (println (str "Final: wrappers=" (mx/get-wrappers-count)
               " memory=" (.toFixed (/ (mx/get-active-memory) 1048576.0) 1) "MB")))

(defn stress-test-gc-sweep [n batch-size]
  (println "\n=== Test 2: GC + sweep-dead-arrays! ===")
  (println (str "Initial wrappers: " (mx/get-wrappers-count)))
  (loop [i 0 max-w 0]
    (when (< i n)
      (create-batch batch-size)
      (when (zero? (mod i (* batch-size 5)))
        (mx/force-gc!)
        (mx/sweep-dead-arrays!))
      (let [new-max (if (zero? (mod i (* batch-size 10)))
                      (report i max-w)
                      (max max-w (mx/get-wrappers-count)))]
        (recur (+ i batch-size) new-max))))
  (println (str "Final: wrappers=" (mx/get-wrappers-count)
               " memory=" (.toFixed (/ (mx/get-active-memory) 1048576.0) 1) "MB")))

(defn stress-test-sweep-only [n batch-size]
  (println "\n=== Test 3: sweep-dead-arrays! only (no GC) ===")
  (println (str "Initial wrappers: " (mx/get-wrappers-count)))
  (loop [i 0 max-w 0]
    (when (< i n)
      (create-batch batch-size)
      (when (zero? (mod i (* batch-size 5)))
        (mx/sweep-dead-arrays!))
      (let [new-max (if (zero? (mod i (* batch-size 10)))
                      (report i max-w)
                      (max max-w (mx/get-wrappers-count)))]
        (recur (+ i batch-size) new-max))))
  (println (str "Final: wrappers=" (mx/get-wrappers-count)
               " memory=" (.toFixed (/ (mx/get-active-memory) 1048576.0) 1) "MB")))

(def N 30000)
(def BATCH 100)

(stress-test-sync N BATCH)
(stress-test-gc-sweep N BATCH)
(stress-test-sweep-only N BATCH)
(println "\nAll tests complete.")
