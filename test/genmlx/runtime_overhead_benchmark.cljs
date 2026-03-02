(ns genmlx.runtime-overhead-benchmark
  "Micro-benchmarks: runtime-as-parameter overhead and Metal memory profile."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.mcmc :as mcmc])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn time-it [label n f]
  (let [start (js/Date.now)
        _ (dotimes [_ n] (f))
        elapsed (- (js/Date.now) start)
        per-call (/ elapsed n)]
    (println (str "  " label ": " elapsed "ms total, "
                  (.toFixed per-call 3) "ms/call"))
    elapsed))

(println "\n=== Runtime Overhead Micro-Benchmarks ===\n")

;; Benchmark 1: Minimal gen fn — single trace site
(println "-- Single trace site: p/simulate overhead --")
(let [trivial (gen [] (trace :x (dist/gaussian 0 1)))]
  (time-it "1000 simulations (1 trace site)" 1000
    #(p/simulate trivial [])))
(mx/clear-cache!)

;; Benchmark 2: Raw function call baseline
(println "\n-- Baseline: raw function call --")
(let [raw-fn (fn [] (mx/add (mx/scalar 1.0) (mx/scalar 2.0)))]
  (time-it "1000 raw fn calls" 1000 raw-fn))
(mx/clear-cache!)

;; Benchmark 3: Two trace sites
(println "\n-- Two trace sites --")
(let [two-site (gen []
                 (let [x (trace :x (dist/gaussian 0 1))
                       y (trace :y (dist/gaussian 0 1))]
                   (mx/add x y)))]
  (time-it "1000 simulations (2 trace sites)" 1000
    #(p/simulate two-site [])))
(mx/clear-cache!)

;; Benchmark 4: Five trace sites
(println "\n-- Five trace sites --")
(let [five-site (gen []
                  (let [a (trace :a (dist/gaussian 0 1))
                        b (trace :b (dist/gaussian 0 1))
                        c (trace :c (dist/gaussian 0 1))
                        d (trace :d (dist/gaussian 0 1))
                        e (trace :e (dist/gaussian 0 1))]
                    (mx/add a (mx/add b (mx/add c (mx/add d e))))))]
  (time-it "1000 simulations (5 trace sites)" 1000
    #(p/simulate five-site [])))
(mx/clear-cache!)

;; Benchmark 5: Metal buffer memory profile during sustained inference
(println "\n-- Metal memory profile: sustained MH inference --")
(let [model (gen [xs]
              (let [slope     (trace :slope (dist/gaussian 0 10))
                    intercept (trace :intercept (dist/gaussian 0 10))]
                (mx/eval! slope intercept)
                (let [s (mx/item slope) i (mx/item intercept)]
                  (doseq [[j x] (map-indexed vector xs)]
                    (trace (keyword (str "y" j))
                           (dist/gaussian (+ (* s x) i) 1)))
                  [s i])))
      xs [1.0 2.0 3.0 4.0 5.0]
      obs (reduce (fn [cm [j y]]
                    (cm/set-choice cm [(keyword (str "y" j))] (mx/scalar y)))
                  cm/EMPTY
                  (map-indexed vector [2.1 3.9 6.2 7.8 10.1]))]
  (mx/clear-cache!)
  (mx/reset-peak-memory!)
  (let [mem-before (mx/get-active-memory)
        _ (println (str "  Before: active=" mem-before " bytes"))
        _ (mcmc/mh {:samples 1000 :burn 200 :selection (sel/select :slope :intercept)}
                   model [xs] obs)
        mem-after (mx/get-active-memory)
        mem-peak (mx/get-peak-memory)]
    (println (str "  After:  active=" mem-after " bytes, peak=" mem-peak " bytes"))
    (println (str "  Delta:  " (- mem-after mem-before) " bytes"))))

(println "\nMicro-benchmarks complete.")
