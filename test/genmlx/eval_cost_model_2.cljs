(ns eval-cost-model-2
  "Part 2: Tests that need a fresh process (Metal resource limit).
   Tests 4-7 from the eval cost model."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.util :as u]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.gen :refer [gen]]))

(defn bench [f n warmup]
  (dotimes [_ warmup] (f))
  (let [t0 (js/performance.now)
        _ (dotimes [_ n] (f))
        t1 (js/performance.now)]
    (/ (- t1 t0) n)))

;; ---------------------------------------------------------------------------
;; Model setup
;; ---------------------------------------------------------------------------

(def xs [1.0 2.0 3.0 4.0 5.0])

(def linreg
  (gen [xs]
    (let [slope     (dyn/trace :slope (dist/gaussian (mx/scalar 0) (mx/scalar 10)))
          intercept (dyn/trace :intercept (dist/gaussian (mx/scalar 0) (mx/scalar 10)))]
      (doseq [[j x] (map-indexed vector xs)]
        (dyn/trace (keyword (str "y" j))
                   (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                          intercept)
                                  (mx/scalar 1))))
      slope)))

(def init-trace (p/simulate linreg [xs]))
(def observations
  (reduce (fn [cm k]
            (cm/set-choice cm [k] (cm/get-choice (:choices init-trace) [k])))
          cm/EMPTY
          [:y0 :y1 :y2 :y3 :y4]))

(def param-addrs [:slope :intercept])

;; ---------------------------------------------------------------------------
;; Test 4: Micro-batched lazy MH
;; ---------------------------------------------------------------------------

(println "\n=== TEST 4: Micro-batched lazy MH ===\n")

(let [score-fn (mx/compile-fn (u/make-score-fn linreg [xs] observations param-addrs))
      std (mx/scalar 0.1)
      init-params (mx/array [1.0 0.5])
      _ (mx/eval! (score-fn init-params))

      ;; Eager MH step (eval every step — current approach)
      eager-step (fn [params]
                   (let [noise (rng/normal (rng/fresh-key) [2])
                         proposal (mx/add params (mx/multiply std noise))
                         score-cur (score-fn params)
                         score-prop (score-fn proposal)]
                     (mx/eval! score-cur score-prop)
                     (let [log-alpha (- (mx/item score-prop) (mx/item score-cur))
                           accept? (< (js/Math.log (js/Math.random)) log-alpha)]
                       (if accept? proposal params))))

      ;; Lazy MH step (NO eval — uses mx/where for accept/reject)
      lazy-step (fn [params]
                  (let [noise (rng/normal (rng/fresh-key) [2])
                        proposal (mx/add params (mx/multiply std noise))
                        score-cur (score-fn params)
                        score-prop (score-fn proposal)
                        log-alpha (mx/subtract score-prop score-cur)
                        u (mx/log (rng/uniform (rng/fresh-key) []))
                        accept-mask (mx/greater log-alpha u)]
                    (mx/where accept-mask proposal params)))

      n-trials 10
      warmup 3
      total-steps 200]

  ;; Eager: 200 steps, eval every step
  (let [cost (bench
               (fn []
                 (mx/tidy
                   (fn []
                     (loop [params init-params, i 0]
                       (if (>= i total-steps)
                         (do (mx/eval! params) params)
                         (recur (eager-step params) (inc i)))))))
               n-trials warmup)]
    (println (str "  Eager MH (200 steps, eval/step):    "
                  (.toFixed cost 1) " ms  (" (.toFixed (/ cost total-steps) 3) " ms/step)")))

  ;; Lazy: eval at end only (will this even work?)
  (let [cost (bench
               (fn []
                 (mx/tidy
                   (fn []
                     (loop [params init-params, i 0]
                       (if (>= i total-steps)
                         (do (mx/eval! params) params)
                         (recur (lazy-step params) (inc i)))))))
               n-trials warmup)]
    (println (str "  Lazy MH (200 steps, eval at end):   "
                  (.toFixed cost 1) " ms  (" (.toFixed (/ cost total-steps) 3) " ms/step)")))

  ;; Lazy batched: eval every K steps with tidy
  (doseq [k [5 10 20 50]]
    (let [cost (bench
                 (fn []
                   (loop [params init-params, i 0]
                     (if (>= i total-steps)
                       (do (mx/eval! params) params)
                       (let [params (lazy-step params)
                             params (if (zero? (mod (inc i) k))
                                      (do (mx/eval! params) params)
                                      params)]
                         (recur params (inc i))))))
                 n-trials warmup)]
      (println (str "  Lazy MH (200 steps, eval every " k "): "
                    (.toFixed cost 1) " ms  (" (.toFixed (/ cost total-steps) 3) " ms/step)")))))

;; ---------------------------------------------------------------------------
;; Test 5: Does vmap compose with make-score-fn?
;; ---------------------------------------------------------------------------

(println "\n=== TEST 5: Does vmap compose with make-score-fn? ===\n")

(let [score-fn (u/make-score-fn linreg [xs] observations param-addrs)
      test-params (mx/array [1.0 0.5])
      scalar-result (score-fn test-params)
      _ (mx/eval! scalar-result)
      _ (println (str "  Scalar score: " (mx/item scalar-result)))]

  (try
    (let [vmapped (mx/vmap score-fn)
          batched-params (mx/stack [(mx/array [1.0 0.5])
                                    (mx/array [2.0 -0.5])
                                    (mx/array [0.0 0.0])])
          result (vmapped batched-params)]
      (mx/eval! result)
      (println (str "  vmap(score-fn) shape: " (mx/shape result)))
      (println (str "  vmap(score-fn) values: " (vec (mx/->clj result))))
      (println "  VERDICT: vmap composes with make-score-fn!"))
    (catch :default e
      (println (str "  vmap(score-fn) FAILED: " (.-message e)))
      (println "  VERDICT: vmap does NOT compose with make-score-fn"))))

;; ---------------------------------------------------------------------------
;; Test 6: Triple transform compile(vmap(grad(score-fn)))
;; ---------------------------------------------------------------------------

(println "\n=== TEST 6: Triple transform ===\n")

(let [score-fn (u/make-score-fn linreg [xs] observations param-addrs)]

  ;; grad(score-fn)
  (try
    (let [grad-fn (mx/grad score-fn)
          g (grad-fn (mx/array [1.0 0.5]))]
      (mx/eval! g)
      (println (str "  grad(score-fn): " (vec (mx/->clj g)))))
    (catch :default e
      (println (str "  grad(score-fn) FAILED: " (.-message e)))))

  ;; compile(grad(score-fn))
  (try
    (let [cg (mx/compile-fn (mx/grad score-fn))
          g (cg (mx/array [1.0 0.5]))]
      (mx/eval! g)
      (println (str "  compile(grad(score-fn)): " (vec (mx/->clj g)))))
    (catch :default e
      (println (str "  compile(grad(score-fn)) FAILED: " (.-message e)))))

  ;; vmap(grad(score-fn))
  (try
    (let [vg (mx/vmap (mx/grad score-fn))
          batched (mx/stack [(mx/array [1.0 0.5])
                             (mx/array [2.0 -0.5])
                             (mx/array [0.0 0.0])])
          result (vg batched)]
      (mx/eval! result)
      (println (str "  vmap(grad(score-fn)) shape: " (mx/shape result)))
      (println (str "  vmap(grad(score-fn)): " (vec (map vec (mx/->clj result))))))
    (catch :default e
      (println (str "  vmap(grad(score-fn)) FAILED: " (.-message e)))))

  ;; THE FULL TRIPLE TRANSFORM
  (try
    (let [triple (mx/compile-fn (mx/vmap (mx/grad score-fn)))
          batched (mx/stack [(mx/array [1.0 0.5])
                             (mx/array [2.0 -0.5])
                             (mx/array [0.0 0.0])])
          result (triple batched)]
      (mx/eval! result)
      (println (str "  compile(vmap(grad(score-fn))) shape: " (mx/shape result)))
      (println "  THE TRIPLE TRANSFORM WORKS!")

      ;; Benchmark: triple vs sequential
      (let [_ (dotimes [_ 20] (mx/eval! (triple batched)))
            n 300
            t0 (js/performance.now)
            _ (dotimes [_ n] (mx/eval! (triple batched)))
            t1 (js/performance.now)
            triple-time (/ (- t1 t0) n)

            compiled-grad (mx/compile-fn (mx/grad score-fn))
            _ (dotimes [_ 20] (mx/eval! (compiled-grad (mx/array [1.0 0.5]))))
            t2 (js/performance.now)
            _ (dotimes [_ n]
                (let [g1 (compiled-grad (mx/array [1.0 0.5]))
                      g2 (compiled-grad (mx/array [2.0 -0.5]))
                      g3 (compiled-grad (mx/array [0.0 0.0]))]
                  (mx/eval! g1 g2 g3)))
            t3 (js/performance.now)
            seq-time (/ (- t3 t2) n)]
        (println (str "\n  Triple (N=3): " (.toFixed triple-time 3) " ms"))
        (println (str "  3× sequential: " (.toFixed seq-time 3) " ms"))
        (println (str "  Speedup: " (.toFixed (/ seq-time triple-time) 2) "x"))))
    (catch :default e
      (println (str "  compile(vmap(grad(score-fn))) FAILED: " (.-message e))))))

;; ---------------------------------------------------------------------------
;; Test 7: Bare eval! dispatch cost
;; ---------------------------------------------------------------------------

(println "\n=== TEST 7: Bare eval! dispatch cost ===\n")

(let [pre-evaled (mx/scalar 42.0)
      _ (mx/eval! pre-evaled)
      n 5000
      warmup 100

      cost-noop (bench #(mx/eval! pre-evaled) n warmup)
      cost-trivial (bench #(mx/eval! (mx/add pre-evaled pre-evaled)) n warmup)
      cost-10 (bench
                #(let [results (mapv (fn [_] (mx/add pre-evaled pre-evaled)) (range 10))]
                   (apply mx/eval! results))
                n warmup)
      cost-10-sep (bench
                    #(dotimes [_ 10]
                       (mx/eval! (mx/add pre-evaled pre-evaled)))
                    n warmup)]

  (println (str "  eval!(already-evaluated):          " (.toFixed cost-noop 4) " ms"))
  (println (str "  eval!(1 trivial op):               " (.toFixed cost-trivial 4) " ms"))
  (println (str "  eval!(10 trivial ops, 1 call):     " (.toFixed cost-10 4) " ms"))
  (println (str "  10 × eval!(1 trivial op):          " (.toFixed cost-10-sep 4) " ms"))
  (println (str "\n  Fixed dispatch per eval!: ~" (.toFixed (- cost-trivial cost-noop) 4) " ms"))
  (println (str "  Batching 10→1 eval! savings: " (.toFixed (/ cost-10-sep cost-10) 1) "x")))

(println "\nDone.")
