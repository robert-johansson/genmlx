(ns eval-cost-model
  "Ground truth test: what is the eval! cost model?

   KEY QUESTION: Is eval! cost per-call (synchronization) or per-kernel (dispatch)?

   If eval!(K ops) ≈ K * eval!(1 op):  per-kernel → micro-batching won't help
   If eval!(K ops) ≈ eval!(1 op):      per-call  → micro-batching gives Kx

   Also tests:
   - Does compile-fn handle lazy (unevaluated) inputs?
   - Does vmap compose with the existing SCI-based score-fn?
   - What is the cost of mx/where (lazy accept/reject)?"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.util :as u]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn bench
  "Run f n times after warmup, return ms/call."
  [f n warmup]
  (dotimes [_ warmup] (f))
  (let [t0 (js/performance.now)
        _ (dotimes [_ n] (f))
        t1 (js/performance.now)]
    (/ (- t1 t0) n)))

;; ---------------------------------------------------------------------------
;; Test 1: eval! cost vs lazy graph depth (simple ops)
;; ---------------------------------------------------------------------------

(println "\n=== TEST 1: eval! cost vs lazy graph depth (simple add chain) ===\n")

(let [x (mx/scalar 1.0)
      ;; Build chains of K lazy adds: x + x + x + ... (K times)
      make-chain (fn [k]
                   (loop [acc x, i 0]
                     (if (>= i k) acc
                       (recur (mx/add acc x) (inc i)))))
      n 500
      warmup 50]

  (doseq [k [1 5 10 20 50 100 200 500]]
    (let [cost (bench
                 (fn []
                   (let [chain (make-chain k)]
                     (mx/eval! chain)))
                 n warmup)]
      (println (str "  K=" k " ops:  "
                    (.toFixed cost 3) " ms/eval  "
                    "(" (.toFixed (/ cost k) 4) " ms/op)")))))

;; ---------------------------------------------------------------------------
;; Test 2: eval! cost vs lazy graph depth (compiled score-fn chains)
;; ---------------------------------------------------------------------------

(println "\n=== TEST 2: eval! cost vs chained compiled-fn calls ===\n")

(let [;; A simple compiled function: f(x) = x + x*x
      f (mx/compile-fn (fn [x] (mx/add x (mx/multiply x x))))
      x (mx/scalar 1.0)
      ;; Warmup compile cache
      _ (mx/eval! (f x))
      n 500
      warmup 50]

  (doseq [k [1 5 10 20 50 100]]
    (let [cost (bench
                 (fn []
                   (let [result (loop [acc x, i 0]
                                  (if (>= i k) acc
                                    (recur (f acc) (inc i))))]
                     (mx/eval! result)))
                 n warmup)]
      (println (str "  K=" k " compiled calls:  "
                    (.toFixed cost 3) " ms/eval  "
                    "(" (.toFixed (/ cost k) 4) " ms/call)")))))

;; ---------------------------------------------------------------------------
;; Test 3: eval! cost for K independent compiled-fn calls (not chained)
;; ---------------------------------------------------------------------------

(println "\n=== TEST 3: eval! cost for K INDEPENDENT compiled-fn calls ===\n")

(let [f (mx/compile-fn (fn [x] (mx/add x (mx/multiply x x))))
      make-inputs (fn [k] (mapv #(mx/scalar (double %)) (range k)))
      _ (mx/eval! (f (mx/scalar 1.0)))
      n 500
      warmup 50]

  (doseq [k [1 5 10 20 50 100]]
    (let [inputs (make-inputs k)
          cost (bench
                 (fn []
                   (let [results (mapv f inputs)]
                     (apply mx/eval! results)))
                 n warmup)]
      (println (str "  K=" k " independent calls:  "
                    (.toFixed cost 3) " ms/eval  "
                    "(" (.toFixed (/ cost k) 4) " ms/call)")))))

;; ---------------------------------------------------------------------------
;; Test 4: Simulated micro-batched MH (lazy accept/reject via mx/where)
;; ---------------------------------------------------------------------------

(println "\n=== TEST 4: Simulated micro-batched lazy MH ===\n")

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

(let [score-fn (mx/compile-fn (u/make-score-fn linreg [xs] observations param-addrs))
      std (mx/scalar 0.1)
      init-params (mx/array [1.0 0.5])
      _ (mx/eval! (score-fn init-params))

      ;; Eager MH step (eval every step — current approach)
      eager-step (fn [params]
                   (let [noise (mx/random-normal [2])
                         proposal (mx/add params (mx/multiply std noise))
                         score-cur (score-fn params)
                         score-prop (score-fn proposal)]
                     (mx/eval! score-cur score-prop)
                     (let [log-alpha (- (mx/item score-prop) (mx/item score-cur))
                           accept? (< (js/Math.log (js/Math.random)) log-alpha)]
                       (if accept? proposal params))))

      ;; Lazy MH step (NO eval — uses mx/where for accept/reject)
      lazy-step (fn [params]
                  (let [noise (mx/random-normal [2])
                        proposal (mx/add params (mx/multiply std noise))
                        score-cur (score-fn params)
                        score-prop (score-fn proposal)
                        log-alpha (mx/subtract score-prop score-cur)
                        ;; Lazy accept/reject: no eval needed
                        u (mx/log (mx/random-uniform []))
                        accept-mask (mx/greater log-alpha u)]
                    ;; mx/where selects lazily — no materialization
                    (mx/where accept-mask proposal params)))

      n-trials 20
      warmup 5]

  ;; Eager: 200 steps, eval every step
  (let [cost (bench
               (fn []
                 (loop [params init-params, i 0]
                   (if (>= i 200)
                     (do (mx/eval! params) params)
                     (recur (eager-step params) (inc i)))))
               n-trials warmup)]
    (println (str "  Eager MH (200 steps, eval/step):    " (.toFixed cost 1) " ms  (" (.toFixed (/ cost 200) 3) " ms/step)")))

  ;; Lazy: 200 steps, eval at end only
  (let [cost (bench
               (fn []
                 (loop [params init-params, i 0]
                   (if (>= i 200)
                     (do (mx/eval! params) params)
                     (recur (lazy-step params) (inc i)))))
               n-trials warmup)]
    (println (str "  Lazy MH (200 steps, eval at end):   " (.toFixed cost 1) " ms  (" (.toFixed (/ cost 200) 3) " ms/step)")))

  ;; Lazy batched: 200 steps, eval every K steps
  (doseq [k [5 10 20 50]]
    (let [cost (bench
                 (fn []
                   (loop [params init-params, i 0]
                     (if (>= i 200)
                       (do (mx/eval! params) params)
                       (let [params (lazy-step params)
                             params (if (zero? (mod (inc i) k))
                                      (do (mx/eval! params) params)
                                      params)]
                         (recur params (inc i))))))
                 n-trials warmup)]
      (println (str "  Lazy MH (200 steps, eval every " k "): "
                    (.toFixed cost 1) " ms  (" (.toFixed (/ cost 200) 3) " ms/step)")))))

;; ---------------------------------------------------------------------------
;; Test 5: Does vmap compose with make-score-fn?
;; ---------------------------------------------------------------------------

(println "\n=== TEST 5: Does vmap compose with make-score-fn? ===\n")

(let [score-fn (u/make-score-fn linreg [xs] observations param-addrs)
      test-params (mx/array [1.0 0.5])
      ;; Scalar score (baseline)
      scalar-result (score-fn test-params)
      _ (mx/eval! scalar-result)
      _ (println (str "  Scalar score: " (mx/item scalar-result)))]

  ;; Try vmap on the score function
  (try
    (let [vmapped (mx/vmap score-fn)
          ;; Stack 5 different parameter sets: [5, 2]
          batched-params (mx/stack [(mx/array [1.0 0.5])
                                    (mx/array [2.0 -0.5])
                                    (mx/array [0.0 0.0])
                                    (mx/array [-1.0 1.0])
                                    (mx/array [0.5 0.5])])
          result (vmapped batched-params)]
      (mx/eval! result)
      (println (str "  vmap(score-fn) result shape: " (mx/shape result)))
      (println (str "  vmap(score-fn) values: " (vec (mx/->clj result))))
      ;; Verify: first element should match scalar
      (println (str "  First matches scalar: "
                    (< (js/Math.abs (- (mx/item scalar-result)
                                      (first (mx/->clj result)))) 1e-3)))
      (println "  VERDICT: vmap composes with make-score-fn!"))
    (catch :default e
      (println (str "  vmap(score-fn) FAILED: " (.-message e)))
      (println "  VERDICT: vmap does NOT compose with make-score-fn"))))

;; ---------------------------------------------------------------------------
;; Test 6: Does compile(vmap(grad(score-fn))) work? — the triple transform
;; ---------------------------------------------------------------------------

(println "\n=== TEST 6: Triple transform compile(vmap(grad(score-fn))) ===\n")

(let [score-fn (u/make-score-fn linreg [xs] observations param-addrs)]

  ;; First: does grad(score-fn) work?
  (try
    (let [grad-fn (mx/grad score-fn)
          g (grad-fn (mx/array [1.0 0.5]))]
      (mx/eval! g)
      (println (str "  grad(score-fn) works: " (vec (mx/->clj g)))))
    (catch :default e
      (println (str "  grad(score-fn) FAILED: " (.-message e)))))

  ;; Second: does compile(grad(score-fn)) work?
  (try
    (let [compiled-grad (mx/compile-fn (mx/grad score-fn))
          g (compiled-grad (mx/array [1.0 0.5]))]
      (mx/eval! g)
      (println (str "  compile(grad(score-fn)) works: " (vec (mx/->clj g)))))
    (catch :default e
      (println (str "  compile(grad(score-fn)) FAILED: " (.-message e)))))

  ;; Third: does vmap(grad(score-fn)) work?
  (try
    (let [vg (mx/vmap (mx/grad score-fn))
          batched (mx/stack [(mx/array [1.0 0.5])
                             (mx/array [2.0 -0.5])
                             (mx/array [0.0 0.0])])
          result (vg batched)]
      (mx/eval! result)
      (println (str "  vmap(grad(score-fn)) shape: " (mx/shape result)))
      (println (str "  vmap(grad(score-fn)) works: " (vec (map vec (mx/->clj result))))))
    (catch :default e
      (println (str "  vmap(grad(score-fn)) FAILED: " (.-message e)))))

  ;; Fourth: the full triple transform
  (try
    (let [triple (mx/compile-fn (mx/vmap (mx/grad score-fn)))
          batched (mx/stack [(mx/array [1.0 0.5])
                             (mx/array [2.0 -0.5])
                             (mx/array [0.0 0.0])])
          result (triple batched)]
      (mx/eval! result)
      (println (str "  compile(vmap(grad(score-fn))) shape: " (mx/shape result)))
      (println (str "  compile(vmap(grad(score-fn))) works!"))

      ;; Benchmark
      (let [_ (dotimes [_ 20] (mx/eval! (triple batched)))
            n 500
            t0 (js/performance.now)
            _ (dotimes [_ n] (mx/eval! (triple batched)))
            t1 (js/performance.now)
            triple-time (/ (- t1 t0) n)
            ;; Compare to N sequential grad calls
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
        (println (str "\n  Triple transform (N=3): " (.toFixed triple-time 3) " ms/call"))
        (println (str "  3x sequential grad:    " (.toFixed seq-time 3) " ms/call"))
        (println (str "  Speedup: " (.toFixed (/ seq-time triple-time) 2) "x"))))
    (catch :default e
      (println (str "  compile(vmap(grad(score-fn))) FAILED: " (.-message e))))))

;; ---------------------------------------------------------------------------
;; Test 7: Bare eval! dispatch cost (no computation)
;; ---------------------------------------------------------------------------

(println "\n=== TEST 7: Bare eval! dispatch cost ===\n")

(let [pre-evaled (mx/scalar 42.0)
      _ (mx/eval! pre-evaled)
      n 5000
      warmup 100

      ;; eval! on already-evaluated scalar (should be nearly free)
      cost-noop (bench #(mx/eval! pre-evaled) n warmup)

      ;; eval! on a single trivial op
      cost-trivial (bench
                     #(let [r (mx/add pre-evaled pre-evaled)]
                        (mx/eval! r))
                     n warmup)

      ;; eval! on 10 independent trivial ops
      cost-10 (bench
                #(let [results (mapv (fn [_] (mx/add pre-evaled pre-evaled)) (range 10))]
                   (apply mx/eval! results))
                n warmup)

      ;; 10 separate eval! calls on trivial ops
      cost-10-sep (bench
                    #(dotimes [_ 10]
                       (mx/eval! (mx/add pre-evaled pre-evaled)))
                    n warmup)]

  (println (str "  eval!(already-evaluated):          " (.toFixed cost-noop 4) " ms"))
  (println (str "  eval!(1 trivial op):               " (.toFixed cost-trivial 4) " ms"))
  (println (str "  eval!(10 independent trivial ops): " (.toFixed cost-10 4) " ms"))
  (println (str "  10 × eval!(1 trivial op):          " (.toFixed cost-10-sep 4) " ms"))
  (println (str "\n  Dispatch overhead per eval! call: ~" (.toFixed (- cost-trivial cost-noop) 4) " ms"))
  (println (str "  Cost of batching 10 into 1 eval!: " (.toFixed (/ cost-10-sep cost-10) 1) "x savings")))

(println "\nDone.")
