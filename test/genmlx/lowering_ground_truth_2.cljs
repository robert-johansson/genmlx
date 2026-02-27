(ns lowering-ground-truth-2
  "Part 2: Where does time ACTUALLY go in compiled inference?
   Now that we know compile-fn caches score functions, what's the REAL bottleneck?"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.util :as u]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.gen :refer [gen]]
            [genmlx.inference.mcmc :as mcmc]))

;; ---------------------------------------------------------------------------
;; Same model as ground truth test 1
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
;; Test 1: Decompose compiled-mh step cost
;; ---------------------------------------------------------------------------

(println "\n=== TEST 1: Decompose compiled-mh step cost ===\n")

(let [;; Set up what compiled-mh does internally
      score-fn (u/make-compiled-score-fn linreg [xs] observations param-addrs)
      params (mx/array [1.0 0.5])
      step-size (mx/scalar 0.1)

      ;; Warmup score-fn
      _ (dotimes [_ 20]
          (mx/eval! (score-fn params)))

      n 1000

      ;; A: Just the compiled score-fn call + eval
      t0 (js/performance.now)
      _ (dotimes [_ n]
          (mx/eval! (score-fn params)))
      t1 (js/performance.now)
      score-time (/ (- t1 t0) n)

      ;; B: Score-fn call WITHOUT eval (just graph construction)
      t2 (js/performance.now)
      _ (dotimes [_ n]
          (score-fn params))
      t3 (js/performance.now)
      score-no-eval-time (/ (- t3 t2) n)

      ;; C: Just eval! on a pre-built result
      pre-built (score-fn params)
      t4 (js/performance.now)
      _ (dotimes [_ n]
          (mx/eval! pre-built))
      t5 (js/performance.now)
      eval-time (/ (- t5 t4) n)

      ;; D: Proposal: generate random noise + add to params
      t6 (js/performance.now)
      _ (dotimes [_ n]
          (let [noise (rng/normal (rng/fresh-key) [2])
                proposed (mx/add params (mx/multiply step-size noise))]
            (mx/eval! proposed)))
      t7 (js/performance.now)
      proposal-time (/ (- t7 t6) n)

      ;; E: MH accept/reject computation
      t8 (js/performance.now)
      _ (dotimes [_ n]
          (let [log-alpha (mx/scalar -0.5)]
            (mx/eval! log-alpha)
            (let [u (js/Math.random)]
              (< (js/Math.log u) (mx/item log-alpha)))))
      t9 (js/performance.now)
      accept-time (/ (- t9 t8) n)

      ;; F: Full MH step (proposal + 2x score + accept)
      t10 (js/performance.now)
      _ (let [current-score (score-fn params)]
          (mx/eval! current-score)
          (dotimes [_ n]
            (let [noise (rng/normal (rng/fresh-key) [2])
                  proposed (mx/add params (mx/multiply step-size noise))
                  proposed-score (score-fn proposed)]
              (mx/eval! proposed-score)
              (let [log-alpha (- (mx/item proposed-score) (mx/item current-score))]
                (< (js/Math.log (js/Math.random)) log-alpha)))))
      t11 (js/performance.now)
      full-step-time (/ (- t11 t10) n)]

  (println "  Component breakdown (1000 iterations):")
  (println (str "  ┌──────────────────────────────┬──────────┐"))
  (println (str "  │ Component                    │ ms/call  │"))
  (println (str "  ├──────────────────────────────┼──────────┤"))
  (println (str "  │ score-fn + eval              │ " (.toFixed score-time 4) "  │"))
  (println (str "  │ score-fn (no eval)           │ " (.toFixed score-no-eval-time 4) "  │"))
  (println (str "  │ eval! only (pre-built)       │ " (.toFixed eval-time 4) "  │"))
  (println (str "  │ proposal (noise + add)       │ " (.toFixed proposal-time 4) "  │"))
  (println (str "  │ accept/reject logic          │ " (.toFixed accept-time 4) "  │"))
  (println (str "  │ Full MH step (manual)        │ " (.toFixed full-step-time 4) "  │"))
  (println (str "  └──────────────────────────────┴──────────┘"))
  (println (str "\n  Sum of parts: " (.toFixed (+ score-time proposal-time accept-time) 4) "ms"))
  (println (str "  eval! overhead: " (.toFixed (- score-time score-no-eval-time) 4) "ms (graph dispatch to Metal)")))

;; ---------------------------------------------------------------------------
;; Test 2: Actual compiled-mh from mcmc.cljs — profile total
;; ---------------------------------------------------------------------------

(println "\n=== TEST 2: Actual compiled-mh timing (from mcmc.cljs) ===\n")

(let [;; Initialize
      {:keys [trace]} (p/generate linreg [xs] observations)
      key (rng/fresh-key)

      opts-base {:addresses param-addrs :proposal-std 0.1}

      ;; Warmup
      _ (dotimes [_ 3]
          (mcmc/compiled-mh (assoc opts-base :samples 50) linreg [xs] observations))

      ;; Benchmark: 200 steps
      t0 (js/performance.now)
      result (mcmc/compiled-mh (assoc opts-base :samples 200) linreg [xs] observations)
      t1 (js/performance.now)
      total-time (- t1 t0)

      ;; Benchmark: 1000 steps
      t2 (js/performance.now)
      _ (mcmc/compiled-mh (assoc opts-base :samples 1000) linreg [xs] observations)
      t3 (js/performance.now)
      total-time-1k (- t3 t2)]

  (println (str "  compiled-mh 200 steps:  " (.toFixed total-time 1) " ms (" (.toFixed (/ total-time 200) 3) " ms/step)"))
  (println (str "  compiled-mh 1000 steps: " (.toFixed total-time-1k 1) " ms (" (.toFixed (/ total-time-1k 1000) 3) " ms/step)"))
  (println (str "  Scaling: " (.toFixed (/ (/ total-time-1k 1000) (/ total-time 200)) 2) "x (expect ~1.0 for linear scaling)")))

;; ---------------------------------------------------------------------------
;; Test 3: Gradient functions — same caching question
;; ---------------------------------------------------------------------------

(println "\n=== TEST 3: Compiled gradient — is it already at the floor? ===\n")

(let [;; Original compiled grad (uses make-score-fn internally)
      compiled-grad-orig (u/make-compiled-grad-score linreg [xs] observations param-addrs)
      compiled-val-grad-orig (u/make-compiled-val-grad linreg [xs] observations param-addrs)

      ;; Hand-written grad for comparison
      LOG-2PI-HALF (mx/scalar (* 0.5 (js/Math.log (* 2.0 js/Math.PI))))
      HALF (mx/scalar 0.5)
      gaussian-lp (fn [v mu sigma]
                    (let [z (mx/divide (mx/subtract v mu) sigma)]
                      (mx/negative
                        (mx/add LOG-2PI-HALF
                                (mx/log sigma)
                                (mx/multiply HALF (mx/square z))))))
      y-obs (mapv #(cm/get-choice observations [%]) [:y0 :y1 :y2 :y3 :y4])
      x-vals (mapv mx/scalar xs)
      prior-mu (mx/scalar 0)
      prior-sigma (mx/scalar 10)
      obs-sigma (mx/scalar 1)
      handwritten-score (fn [params]
                          (let [slope (mx/take-idx params (mx/scalar 0 mx/int32))
                                intercept (mx/take-idx params (mx/scalar 1 mx/int32))]
                            (let [lp-slope (gaussian-lp slope prior-mu prior-sigma)
                                  lp-intercept (gaussian-lp intercept prior-mu prior-sigma)
                                  lp-obs (reduce
                                           (fn [acc [x-val y-val]]
                                             (let [mu (mx/add (mx/multiply slope x-val) intercept)]
                                               (mx/add acc (gaussian-lp y-val mu obs-sigma))))
                                           (mx/scalar 0.0)
                                           (map vector x-vals y-obs))]
                              (mx/add lp-slope (mx/add lp-intercept lp-obs)))))
      compiled-grad-hand (mx/compile-fn (mx/grad handwritten-score))
      compiled-vg-hand (mx/compile-fn (mx/value-and-grad handwritten-score))

      test-params (mx/array [1.0 0.5])
      n 500

      ;; Warmup
      _ (dotimes [_ 20] (mx/eval! (compiled-grad-orig test-params)))
      _ (dotimes [_ 20] (mx/eval! (compiled-grad-hand test-params)))
      _ (dotimes [_ 20] (let [[v g] (compiled-val-grad-orig test-params)] (mx/eval! v g)))
      _ (dotimes [_ 20] (let [[v g] (compiled-vg-hand test-params)] (mx/eval! v g)))

      ;; Benchmark: compiled grad original
      t0 (js/performance.now)
      _ (dotimes [_ n] (mx/eval! (compiled-grad-orig test-params)))
      t1 (js/performance.now)
      grad-orig-time (/ (- t1 t0) n)

      ;; Benchmark: compiled grad handwritten
      t2 (js/performance.now)
      _ (dotimes [_ n] (mx/eval! (compiled-grad-hand test-params)))
      t3 (js/performance.now)
      grad-hand-time (/ (- t3 t2) n)

      ;; Benchmark: compiled value-and-grad original
      t4 (js/performance.now)
      _ (dotimes [_ n] (let [[v g] (compiled-val-grad-orig test-params)] (mx/eval! v g)))
      t5 (js/performance.now)
      vg-orig-time (/ (- t5 t4) n)

      ;; Benchmark: compiled value-and-grad handwritten
      t6 (js/performance.now)
      _ (dotimes [_ n] (let [[v g] (compiled-vg-hand test-params)] (mx/eval! v g)))
      t7 (js/performance.now)
      vg-hand-time (/ (- t7 t6) n)]

  (println "  Gradient timing (7-site linreg):")
  (println (str "  ┌──────────────────────────────┬──────────┬─────────┐"))
  (println (str "  │ Variant                      │ ms/call  │ gap     │"))
  (println (str "  ├──────────────────────────────┼──────────┼─────────┤"))
  (println (str "  │ grad (orig + compile)        │ " (.toFixed grad-orig-time 3) "   │         │"))
  (println (str "  │ grad (hand + compile)        │ " (.toFixed grad-hand-time 3) "   │ " (.toFixed (/ grad-orig-time grad-hand-time) 2) "x   │"))
  (println (str "  │ val+grad (orig + compile)    │ " (.toFixed vg-orig-time 3) "   │         │"))
  (println (str "  │ val+grad (hand + compile)    │ " (.toFixed vg-hand-time 3) "   │ " (.toFixed (/ vg-orig-time vg-hand-time) 2) "x   │"))
  (println (str "  └──────────────────────────────┴──────────┴─────────┘"))
  (println (str "\n  If gap ≈ 1.0x: compile-fn already at floor for gradients too")))

;; ---------------------------------------------------------------------------
;; Test 4: simulate/generate — the REAL gap
;; ---------------------------------------------------------------------------

(println "\n=== TEST 4: simulate/generate — is THIS the real target? ===\n")

(let [n 200

      ;; Benchmark simulate
      _ (dotimes [_ 5] (p/simulate linreg [xs]))
      t0 (js/performance.now)
      _ (dotimes [_ n] (p/simulate linreg [xs]))
      t1 (js/performance.now)
      sim-time (/ (- t1 t0) n)

      ;; Benchmark generate
      _ (dotimes [_ 5] (p/generate linreg [xs] observations))
      t2 (js/performance.now)
      _ (dotimes [_ n] (p/generate linreg [xs] observations))
      t3 (js/performance.now)
      gen-time (/ (- t3 t2) n)

      ;; Can we compile-fn wrap simulate?
      ;; simulate has random output → graph topology might change
      ;; Let's test: wrap the body-fn + handler in a function
      sim-fn (fn [key-arr]
               (let [key (rng/ensure-key key-arr)
                     result (genmlx.handler/run-handler
                              genmlx.handler/simulate-handler
                              {:choices cm/EMPTY :score (mx/scalar 0.0) :key key
                               :executor nil}
                              #(apply (:body-fn linreg) [xs]))]
                 (:score result)))
      ;; Test if this is compilable
      test-key (rng/fresh-key)
      r (sim-fn test-key)
      _ (mx/eval! r)
      _ (println (str "  simulate-as-fn test: score = " (mx/item r)))

      ;; Can compile-fn cache it?
      call-count (atom 0)
      sim-fn-counted (fn [key-arr]
                       (swap! call-count inc)
                       (sim-fn key-arr))
      compiled-sim (mx/compile-fn sim-fn-counted)
      _ (let [r (compiled-sim test-key)] (mx/eval! r))
      _ (reset! call-count 0)
      _ (let [r (compiled-sim (rng/fresh-key))] (mx/eval! r))
      _ (let [r (compiled-sim (rng/fresh-key))] (mx/eval! r))
      sim-cached? (zero? @call-count)

      ;; Time compiled simulate
      _ (dotimes [_ 20] (mx/eval! (compiled-sim (rng/fresh-key))))
      t4 (js/performance.now)
      _ (dotimes [_ n] (mx/eval! (compiled-sim (rng/fresh-key))))
      t5 (js/performance.now)
      compiled-sim-time (/ (- t5 t4) n)]

  (println "\n  simulate/generate timing:")
  (println (str "  ┌──────────────────────────────┬──────────┬──────────────┐"))
  (println (str "  │ Operation                    │ ms/call  │ vs Gen.jl    │"))
  (println (str "  ├──────────────────────────────┼──────────┼──────────────┤"))
  (println (str "  │ simulate (full GFI)          │ " (.toFixed sim-time 3) "   │ " (.toFixed (/ sim-time 0.021) 0) "x          │"))
  (println (str "  │ generate (full GFI)          │ " (.toFixed gen-time 3) "   │ " (.toFixed (/ gen-time 0.024) 0) "x          │"))
  (println (str "  │ compiled simulate (score)    │ " (.toFixed compiled-sim-time 3) "   │ " (.toFixed (/ compiled-sim-time 0.021) 0) "x          │"))
  (println (str "  └──────────────────────────────┴──────────┴──────────────┘"))
  (println (str "\n  compile-fn cached simulate body: " sim-cached? " (body re-executed " @call-count " times)"))
  (println (str "  Compiled simulate speedup: " (.toFixed (/ sim-time compiled-sim-time) 1) "x over raw")))

(println "\nDone.")
