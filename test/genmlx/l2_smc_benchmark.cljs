(ns genmlx.l2-smc-benchmark
  "Level 2 Gate 1+2: compiled SMC benchmark.

   Gate 1: Compare compiled SMC per-step time against handler smc-unfold.
   Gate 2: N/A for now (chunk size sweep deferred — single-step is baseline).

   Benchmark: Random walk kernel, T=20 steps, N=100 particles."
  (:require [genmlx.gen :refer [gen]]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.combinators :as comb]
            [genmlx.inference.smc :as smc]
            [genmlx.inference.compiled-smc :as csmc]))

;; ---------------------------------------------------------------------------
;; Benchmark kernel
;; ---------------------------------------------------------------------------

(def rw-kernel
  (gen [t state]
    (let [new-state (trace :x (dist/gaussian state 1))]
      (trace :y (dist/gaussian new-state 0.5))
      new-state)))

;; ---------------------------------------------------------------------------
;; Timing helper
;; ---------------------------------------------------------------------------

(defn- time-ms [f]
  (let [start (js/Date.now)]
    (f)
    (- (js/Date.now) start)))

;; ---------------------------------------------------------------------------
;; Gate 1: Compiled SMC vs handler smc-unfold
;; ---------------------------------------------------------------------------

(println "\n== Gate 1: Compiled SMC vs Handler SMC ==")

(let [N 100
      T 20
      obs-seq (mapv (fn [_] (cm/choicemap :y (mx/scalar 1.5))) (range T))
      key (rng/fresh-key 42)
      n-warmup 2
      n-runs 5]

  ;; Warmup
  (println "  Warming up...")
  (dotimes [_ n-warmup]
    (csmc/compiled-smc {:particles N :key (rng/fresh-key 100)}
                        rw-kernel (mx/scalar 0.0) obs-seq)
    (mx/clear-cache!))
  (dotimes [_ n-warmup]
    (smc/smc-unfold {:particles N :key (rng/fresh-key 200)}
                     (dyn/auto-key rw-kernel) (mx/scalar 0.0) obs-seq)
    (mx/clear-cache!))

  ;; Compiled SMC timing
  (println "  Running compiled SMC...")
  (let [compiled-times
        (mapv (fn [i]
                (mx/clear-cache!)
                (let [t (time-ms
                          #(let [r (csmc/compiled-smc
                                     {:particles N :key (rng/fresh-key (+ 300 i))}
                                     rw-kernel (mx/scalar 0.0) obs-seq)]
                             (mx/eval! (:log-ml r) (:particles r))))]
                  t))
              (range n-runs))
        compiled-avg (/ (reduce + compiled-times) n-runs)]

    ;; Handler SMC timing (smc-unfold)
    (println "  Running handler smc-unfold...")
    (let [handler-times
          (mapv (fn [i]
                  (mx/clear-cache!)
                  (let [t (time-ms
                            #(let [r (smc/smc-unfold
                                       {:particles N :key (rng/fresh-key (+ 400 i))}
                                       (dyn/auto-key rw-kernel) (mx/scalar 0.0) obs-seq)]
                               (mx/eval! (:log-ml r))))]
                    t))
                (range n-runs))
          handler-avg (/ (reduce + handler-times) n-runs)
          speedup (/ handler-avg compiled-avg)]

      (println (str "\n  Results (T=" T ", N=" N "):"))
      (println (str "    Compiled SMC: " (.toFixed compiled-avg 1) " ms avg"
                    " (" (clojure.string/join ", " (mapv str compiled-times)) ")"))
      (println (str "    Handler SMC:  " (.toFixed handler-avg 1) " ms avg"
                    " (" (clojure.string/join ", " (mapv str handler-times)) ")"))
      (println (str "    Speedup:      " (.toFixed speedup 2) "x"))
      (println (str "    Gate 1 criterion: >2x → " (if (> speedup 2) "PASS" "FAIL"))))))

;; ---------------------------------------------------------------------------
;; Log-ML comparison
;; ---------------------------------------------------------------------------

(println "\n== Log-ML comparison ==")

(let [N 100
      T 10
      obs-seq (mapv (fn [_] (cm/choicemap :y (mx/scalar 1.5))) (range T))
      n-runs 20
      compiled-lmls
      (mapv (fn [s]
              (let [r (csmc/compiled-smc {:particles N :key (rng/fresh-key (+ s 500))}
                                          rw-kernel (mx/scalar 0.0) obs-seq)]
                (mx/eval! (:log-ml r))
                (mx/item (:log-ml r))))
            (range n-runs))
      handler-lmls
      (mapv (fn [s]
              (let [r (smc/smc-unfold {:particles N :key (rng/fresh-key (+ s 600))}
                                       (dyn/auto-key rw-kernel) (mx/scalar 0.0) obs-seq)]
                (mx/eval! (:log-ml r))
                (mx/item (:log-ml r))))
            (range n-runs))
      c-mean (/ (reduce + compiled-lmls) n-runs)
      h-mean (/ (reduce + handler-lmls) n-runs)]
  (println (str "  Compiled log-ML mean: " (.toFixed c-mean 3)))
  (println (str "  Handler  log-ML mean: " (.toFixed h-mean 3)))
  (println (str "  Difference: " (.toFixed (js/Math.abs (- c-mean h-mean)) 3))))
