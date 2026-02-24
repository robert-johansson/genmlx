(ns lowering-ground-truth
  "Ground truth test: what is the absolute floor cost of a score function
   on bare Metal, and does mx/compile-fn already skip the SCI body?

   This test answers the question BEFORE we write any compiler code."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.util :as u]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Test model: 7-site linear regression (matches benchmark)
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

;; Generate some observations
(def init-trace (p/simulate linreg [xs]))
(def observations
  (reduce (fn [cm k]
            (cm/set-choice cm [k] (cm/get-choice (:choices init-trace) [k])))
          cm/EMPTY
          [:y0 :y1 :y2 :y3 :y4]))

(def param-addrs [:slope :intercept])

;; ---------------------------------------------------------------------------
;; Test 1: Does compile-fn re-execute the body for make-score-fn?
;; ---------------------------------------------------------------------------

(println "\n=== TEST 1: Does compile-fn re-execute body for make-score-fn? ===\n")

(let [call-count (atom 0)
      ;; Wrap make-score-fn to count body executions
      score-fn (let [indexed-addrs (mapv vector (range) param-addrs)]
                 (fn [params]
                   (swap! call-count inc)
                   (let [cm (reduce
                              (fn [cm [i addr]]
                                (cm/set-choice cm [addr] (mx/index params i)))
                              observations
                              indexed-addrs)]
                     (:weight (p/generate linreg [xs] cm)))))
      compiled-score (mx/compile-fn score-fn)
      test-params (mx/array [1.0 0.5])]

  ;; First call (trace)
  (println "Call 1 (trace):")
  (let [r (compiled-score test-params)]
    (mx/eval! r)
    (println (str "  Score: " (mx/item r)))
    (println (str "  Body executions: " @call-count)))

  (reset! call-count 0)

  ;; Second call (same shape — should cache if pure)
  (println "Call 2 (same shape):")
  (let [r (compiled-score (mx/array [2.0 -0.5]))]
    (mx/eval! r)
    (println (str "  Score: " (mx/item r)))
    (println (str "  Body executions after reset: " @call-count)))

  ;; Third call
  (println "Call 3 (same shape):")
  (let [r (compiled-score (mx/array [0.0 0.0]))]
    (mx/eval! r)
    (println (str "  Score: " (mx/item r)))
    (println (str "  Body executions after reset: " @call-count)))

  (println (str "\n  VERDICT: Body re-executed " @call-count " times for 2 subsequent calls"))
  (println (str "  If 0: compile-fn caches → lowering mainly helps first-call cost"))
  (println (str "  If 2: compile-fn re-executes → lowering gives ~8x on every call")))

;; ---------------------------------------------------------------------------
;; Test 2: Pure MLX 7-site score (hand-written, no SCI)
;; ---------------------------------------------------------------------------

(println "\n=== TEST 2: Hand-written pure MLX 7-site score function ===\n")

(let [;; Pre-compute constants
      LOG-2PI-HALF (mx/scalar (* 0.5 (js/Math.log (* 2.0 js/Math.PI))))
      HALF (mx/scalar 0.5)
      ;; Inline gaussian log-prob (pure MLX, no multimethod dispatch)
      gaussian-lp (fn [v mu sigma]
                    (let [z (mx/divide (mx/subtract v mu) sigma)]
                      (mx/negative
                        (mx/add LOG-2PI-HALF
                                (mx/log sigma)
                                (mx/multiply HALF (mx/square z))))))
      ;; Pre-resolve observation values
      y-obs (mapv #(cm/get-choice observations [%]) [:y0 :y1 :y2 :y3 :y4])
      x-vals (mapv mx/scalar xs)
      prior-mu (mx/scalar 0)
      prior-sigma (mx/scalar 10)
      obs-sigma (mx/scalar 1)

      ;; The lowered score function — pure MLX, no SCI, no handler
      raw-score-fn
      (fn [params]
        (let [slope (mx/take-idx params (mx/scalar 0 mx/int32))
              intercept (mx/take-idx params (mx/scalar 1 mx/int32))]
          ;; Prior terms
          (let [lp-slope (gaussian-lp slope prior-mu prior-sigma)
                lp-intercept (gaussian-lp intercept prior-mu prior-sigma)
                ;; Likelihood terms (dependent: mu = slope * x + intercept)
                lp-obs (reduce
                         (fn [acc [x-val y-val]]
                           (let [mu (mx/add (mx/multiply slope x-val) intercept)]
                             (mx/add acc (gaussian-lp y-val mu obs-sigma))))
                         (mx/scalar 0.0)
                         (map vector x-vals y-obs))]
            (mx/add lp-slope (mx/add lp-intercept lp-obs)))))

      compiled-raw (mx/compile-fn raw-score-fn)
      compiled-grad-raw (mx/compile-fn (mx/grad raw-score-fn))

      ;; Verify correctness: compare to make-score-fn
      test-params (mx/array [1.0 0.5])
      orig-score-fn (u/make-score-fn linreg [xs] observations param-addrs)
      orig-result (orig-score-fn test-params)
      _ (mx/eval! orig-result)
      raw-result (raw-score-fn test-params)
      _ (mx/eval! raw-result)]

  (println (str "  Original score-fn result:  " (mx/item orig-result)))
  (println (str "  Hand-written MLX result:   " (mx/item raw-result)))
  (println (str "  Match: " (< (js/Math.abs (- (mx/item orig-result) (mx/item raw-result))) 1e-4)))

  ;; Warmup
  (dotimes [_ 20] (mx/eval! (raw-score-fn test-params)))
  (dotimes [_ 20] (mx/eval! (compiled-raw test-params)))
  (dotimes [_ 20] (mx/eval! (compiled-grad-raw test-params)))
  (dotimes [_ 20] (mx/eval! (orig-score-fn test-params)))

  (let [n 500
        ;; Benchmark: original make-score-fn (SCI + generate)
        t0 (js/performance.now)
        _ (dotimes [_ n] (mx/eval! (orig-score-fn test-params)))
        t1 (js/performance.now)
        orig-time (/ (- t1 t0) n)

        ;; Benchmark: compiled original (compile-fn wrapping make-score-fn)
        compiled-orig (mx/compile-fn (u/make-score-fn linreg [xs] observations param-addrs))
        _ (dotimes [_ 20] (mx/eval! (compiled-orig test-params)))
        t2 (js/performance.now)
        _ (dotimes [_ n] (mx/eval! (compiled-orig test-params)))
        t3 (js/performance.now)
        compiled-orig-time (/ (- t3 t2) n)

        ;; Benchmark: hand-written raw (no compile)
        t4 (js/performance.now)
        _ (dotimes [_ n] (mx/eval! (raw-score-fn test-params)))
        t5 (js/performance.now)
        raw-time (/ (- t5 t4) n)

        ;; Benchmark: hand-written compiled
        t6 (js/performance.now)
        _ (dotimes [_ n] (mx/eval! (compiled-raw test-params)))
        t7 (js/performance.now)
        compiled-raw-time (/ (- t7 t6) n)

        ;; Benchmark: hand-written compiled grad
        t8 (js/performance.now)
        _ (dotimes [_ n] (mx/eval! (compiled-grad-raw test-params)))
        t9 (js/performance.now)
        compiled-grad-time (/ (- t9 t8) n)]

    (println "\n  Timing (7-site linreg, median of" n "calls):")
    (println (str "  ┌──────────────────────────────┬──────────┬─────────┐"))
    (println (str "  │ Variant                      │ ms/call  │ vs orig │"))
    (println (str "  ├──────────────────────────────┼──────────┼─────────┤"))
    (println (str "  │ Original (SCI + p/generate)  │ " (.toFixed orig-time 3) "   │ 1.0x    │"))
    (println (str "  │ Compiled original             │ " (.toFixed compiled-orig-time 3) "   │ " (.toFixed (/ orig-time compiled-orig-time) 1) "x    │"))
    (println (str "  │ Hand-written raw (no compile) │ " (.toFixed raw-time 3) "   │ " (.toFixed (/ orig-time raw-time) 1) "x    │"))
    (println (str "  │ Hand-written compiled         │ " (.toFixed compiled-raw-time 3) "   │ " (.toFixed (/ orig-time compiled-raw-time) 1) "x    │"))
    (println (str "  │ Hand-written compiled+grad    │ " (.toFixed compiled-grad-time 3) "   │ " (.toFixed (/ orig-time compiled-grad-time) 1) "x    │"))
    (println (str "  └──────────────────────────────┴──────────┴─────────┘"))
    (println (str "\n  Gap: compiled-original (" (.toFixed compiled-orig-time 3) "ms) vs compiled-handwritten (" (.toFixed compiled-raw-time 3) "ms)"))
    (println (str "  This gap is what model lowering can capture: " (.toFixed (/ compiled-orig-time compiled-raw-time) 1) "x"))
    (println (str "  Metal floor (compiled handwritten): " (.toFixed compiled-raw-time 3) "ms"))))

;; ---------------------------------------------------------------------------
;; Test 3: Verify compile-fn caching for pure functions (control)
;; ---------------------------------------------------------------------------

(println "\n=== TEST 3: compile-fn caching — pure function control ===\n")

(let [call-count (atom 0)
      pure-fn (fn [params]
                (swap! call-count inc)
                (let [a (mx/take-idx params (mx/scalar 0 mx/int32))
                      b (mx/take-idx params (mx/scalar 1 mx/int32))]
                  (mx/add (mx/multiply a a) (mx/multiply b b))))
      compiled (mx/compile-fn pure-fn)
      _ (let [r (compiled (mx/array [1.0 2.0]))] (mx/eval! r))
      _ (reset! call-count 0)
      _ (let [r (compiled (mx/array [3.0 4.0]))] (mx/eval! r))
      _ (let [r (compiled (mx/array [5.0 6.0]))] (mx/eval! r))]
  (println (str "  Pure function: body re-executed " @call-count " times for 2 subsequent calls"))
  (println (str "  (Expected: 0 — confirms compile-fn caching works for pure fns)")))

;; ---------------------------------------------------------------------------
;; Test 4: What about SCI closures inside compile-fn?
;; ---------------------------------------------------------------------------

(println "\n=== TEST 4: SCI closure (with reduce loop) inside compile-fn ===\n")

(let [call-count (atom 0)
      ;; A function with SCI constructs (reduce, map, etc.) but no side effects
      ;; that affect the MLX graph
      x-consts (mapv mx/scalar [1.0 2.0 3.0 4.0 5.0])
      y-consts (mapv mx/scalar [2.5 4.1 6.0 8.2 9.8])
      sigma-const (mx/scalar 1.0)
      LOG-2PI-HALF (mx/scalar (* 0.5 (js/Math.log (* 2.0 js/Math.PI))))
      HALF (mx/scalar 0.5)
      gaussian-lp (fn [v mu sigma]
                    (let [z (mx/divide (mx/subtract v mu) sigma)]
                      (mx/negative
                        (mx/add LOG-2PI-HALF
                                (mx/log sigma)
                                (mx/multiply HALF (mx/square z))))))
      score-fn (fn [params]
                 (swap! call-count inc)
                 (let [slope (mx/take-idx params (mx/scalar 0 mx/int32))
                       intercept (mx/take-idx params (mx/scalar 1 mx/int32))]
                   (reduce
                     (fn [acc [x-val y-val]]
                       (let [mu (mx/add (mx/multiply slope x-val) intercept)]
                         (mx/add acc (gaussian-lp y-val mu sigma-const))))
                     (mx/scalar 0.0)
                     (map vector x-consts y-consts))))
      compiled (mx/compile-fn score-fn)
      _ (let [r (compiled (mx/array [1.0 0.5]))] (mx/eval! r))
      _ (reset! call-count 0)
      _ (let [r (compiled (mx/array [2.0 -0.5]))] (mx/eval! r))
      _ (let [r (compiled (mx/array [0.0 0.0]))] (mx/eval! r))]
  (println (str "  SCI closure with reduce: body re-executed " @call-count " times for 2 subsequent calls"))
  (println (str "  If 0: compile-fn caches SCI closures too (great!)")))

;; ---------------------------------------------------------------------------
;; Test 5: What about functions with volatile! (like the handler)?
;; ---------------------------------------------------------------------------

(println "\n=== TEST 5: Function with volatile! inside compile-fn ===\n")

(let [call-count (atom 0)
      ;; Simulates what the handler does: volatile! for state
      score-fn (fn [params]
                 (swap! call-count inc)
                 (let [state (volatile! (mx/scalar 0.0))
                       a (mx/take-idx params (mx/scalar 0 mx/int32))
                       b (mx/take-idx params (mx/scalar 1 mx/int32))]
                   (vswap! state #(mx/add % (mx/multiply a a)))
                   (vswap! state #(mx/add % (mx/multiply b b)))
                   @state))
      compiled (mx/compile-fn score-fn)
      _ (let [r (compiled (mx/array [1.0 2.0]))] (mx/eval! r))
      _ (reset! call-count 0)
      _ (let [r (compiled (mx/array [3.0 4.0]))] (mx/eval! r))
      _ (let [r (compiled (mx/array [5.0 6.0]))] (mx/eval! r))]
  (println (str "  volatile! function: body re-executed " @call-count " times for 2 subsequent calls"))
  (println (str "  If 0: volatile! doesn't prevent caching (compile-fn ignores JS side effects)"))
  (println (str "  If 2: volatile! prevents caching (compile-fn detects side effects)")))

(println "\nDone.")
