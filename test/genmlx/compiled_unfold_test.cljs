(ns genmlx.compiled-unfold-test
  "Tests for compiled unfold (Tier 2a).
   Validates correctness against the standard UnfoldCombinator,
   then benchmarks speedup."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.compiled :as compiled]
            [genmlx.combinators :as comb])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn assert-true [label v]
  (if v
    (println (str "  PASS: " label))
    (println (str "  FAIL: " label))))

(defn assert-close [label expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (println (str "  PASS: " label " (expected=" (.toFixed expected 4)
                    " actual=" (.toFixed actual 4) ")"))
      (println (str "  FAIL: " label " (expected=" (.toFixed expected 4)
                    " actual=" (.toFixed actual 4)
                    " diff=" (.toFixed diff 4) " tol=" tol ")")))))

(defn bench [label f n-runs]
  (f) ;; warmup
  (let [start (js/Date.now)]
    (dotimes [_ n-runs] (f))
    (let [elapsed (- (js/Date.now) start)
          per-run (/ elapsed n-runs)]
      (println (str "  " label ": " (.toFixed per-run 1) "ms"))
      per-run)))

;; ---------------------------------------------------------------------------
;; Test 1: Basic compiled unfold simulate
;; ---------------------------------------------------------------------------

(println "\n=== Test 1: Compiled unfold simulate (random walk) ===")

;; Random walk: state_{t+1} = state_t + noise
(let [state-dim 2
      noise-dim 2
      n-steps 10
      step-fn (fn [state noise]
                (let [new-state (mx/add state noise)
                      ;; log-prob of standard normal noise
                      lp (mx/multiply (mx/scalar -0.5)
                                      (mx/add (mx/sum (mx/multiply noise noise))
                                              (mx/scalar (* state-dim (js/Math.log (* 2 js/Math.PI))))))]
                  [new-state lp]))
      compiled-fn (compiled/make-compiled-unfold step-fn n-steps state-dim noise-dim)
      key (rng/fresh-key)
      noise (rng/normal key [n-steps noise-dim])
      init-state (mx/zeros [state-dim])
      [final-state states total-score] (compiled-fn init-state noise)]
  (mx/materialize! final-state states total-score)
  (assert-true "final-state is [2]-shaped" (= [state-dim] (vec (mx/shape final-state))))
  (assert-true "states is [10,2]-shaped" (= [n-steps state-dim] (vec (mx/shape states))))
  (assert-true "total-score is scalar" (= [] (vec (mx/shape total-score))))
  (assert-true "total-score is finite" (js/isFinite (mx/item total-score)))
  ;; Verify: final state should equal sum of all noise rows
  (let [expected-final (mx/sum noise 0)
        diff (mx/sum (mx/abs (mx/subtract final-state expected-final)))]
    (mx/materialize! diff)
    (assert-close "final state = cumsum(noise)" 0.0 (mx/item diff) 1e-4))
  (println "  Score:" (.toFixed (mx/item total-score) 4)))

;; ---------------------------------------------------------------------------
;; Test 2: Compiled unfold generate (with observations)
;; ---------------------------------------------------------------------------

(println "\n=== Test 2: Compiled unfold generate (linear dynamics + obs) ===")

(let [state-dim 1
      noise-dim 1
      obs-dim 1
      n-steps 5
      ;; Simple: state_{t+1} = 0.9 * state_t + noise, obs = state + obs_noise
      transition-fn (fn [state]
                      [(mx/multiply (mx/scalar 0.9) state)   ;; mean
                       (mx/array [0.1])])                     ;; std
      observation-fn (fn [state]
                       [state                                  ;; obs mean
                        (mx/array [0.5])])                     ;; obs std
      step-fn (compiled/make-gaussian-step-with-obs
                transition-fn observation-fn state-dim obs-dim)
      compiled-fn (compiled/make-compiled-unfold-generate
                    step-fn n-steps state-dim noise-dim obs-dim)
      key (rng/fresh-key)
      noise (rng/normal key [n-steps noise-dim])
      ;; Observations: pretend we saw [1.0, 0.9, 0.8, 0.7, 0.6]
      obs (mx/reshape (mx/array [1.0 0.9 0.8 0.7 0.6]) [n-steps obs-dim])
      init-state (mx/zeros [state-dim])
      [final-state states total-score total-weight] (compiled-fn init-state noise obs)]
  (mx/materialize! final-state states total-score total-weight)
  (assert-true "final-state is [1]-shaped" (= [state-dim] (vec (mx/shape final-state))))
  (assert-true "states is [5,1]-shaped" (= [n-steps state-dim] (vec (mx/shape states))))
  (assert-true "total-score is scalar" (= [] (vec (mx/shape total-score))))
  (assert-true "total-weight is scalar" (= [] (vec (mx/shape total-weight))))
  (assert-true "total-score is finite" (js/isFinite (mx/item total-score)))
  (assert-true "total-weight is finite" (js/isFinite (mx/item total-weight)))
  (println "  Score:" (.toFixed (mx/item total-score) 4))
  (println "  Weight:" (.toFixed (mx/item total-weight) 4)))

;; ---------------------------------------------------------------------------
;; Test 3: High-level API (compiled-unfold-simulate returns Trace)
;; ---------------------------------------------------------------------------

(println "\n=== Test 3: High-level API (Trace output) ===")

(let [state-dim 2
      noise-dim 2
      n-steps 5
      step-fn (fn [state noise]
                (let [new-state (mx/add state noise)
                      lp (mx/multiply (mx/scalar -0.5)
                                      (mx/add (mx/sum (mx/multiply noise noise))
                                              (mx/scalar (* state-dim (js/Math.log (* 2 js/Math.PI))))))]
                  [new-state lp]))
      trace (compiled/compiled-unfold-simulate
              {:step-fn step-fn :n-steps n-steps
               :state-dim state-dim :noise-dim noise-dim}
              (mx/zeros [state-dim])
              (rng/fresh-key))]
  (assert-true "returns a Trace" (instance? genmlx.trace/Trace trace))
  (assert-true "trace has choices" (not= (:choices trace) cm/EMPTY))
  (assert-true "trace has score" (js/isFinite (mx/item (:score trace))))
  (assert-true "retval has :states" (some? (get-in (:retval trace) [:states])))
  (println "  Score:" (.toFixed (mx/item (:score trace)) 4)))

;; ---------------------------------------------------------------------------
;; Test 4: Statistical validation — compare against standard unfold
;; ---------------------------------------------------------------------------

(println "\n=== Test 4: Statistical agreement with standard UnfoldCombinator ===")

;; We can't compare individual traces (different randomness),
;; but we can compare mean scores across many runs.
;; A random walk with known transition should give similar average scores.

(let [n-runs 200
      n-steps 5
      state-dim 1
      noise-dim 1
      ;; Standard unfold kernel
      kernel (dyn/auto-key
               (gen [t state]
                 (let [new-state (trace :x (dist/gaussian (mx/multiply (mx/scalar 0.9) state) 0.1))]
                   new-state)))
      unfold-gf (comb/unfold-combinator kernel)
      ;; Compiled version with same dynamics
      step-fn (fn [state noise]
                (let [mean (mx/multiply (mx/scalar 0.9) state)
                      std (mx/array [0.1])
                      new-state (mx/add mean (mx/multiply std noise))
                      ;; Gaussian log-prob
                      diff (mx/subtract new-state mean)
                      lp (mx/subtract
                           (mx/subtract
                             (mx/multiply (mx/scalar -0.5)
                                          (mx/sum (mx/divide (mx/multiply diff diff)
                                                             (mx/multiply std std))))
                             (mx/sum (mx/log std)))
                           (mx/scalar (* 0.5 state-dim (js/Math.log (* 2 js/Math.PI)))))]
                  [new-state lp]))
      ;; Run standard unfold
      std-scores (mapv (fn [_]
                         (let [tr (p/simulate unfold-gf [n-steps (mx/zeros [state-dim])])]
                           (mx/item (:score tr))))
                       (range n-runs))
      std-mean (/ (reduce + std-scores) n-runs)
      ;; Run compiled unfold
      compiled-scores (mapv (fn [_]
                              (let [tr (compiled/compiled-unfold-simulate
                                         {:step-fn step-fn :n-steps n-steps
                                          :state-dim state-dim :noise-dim noise-dim}
                                         (mx/zeros [state-dim])
                                         (rng/fresh-key))]
                                (mx/item (:score tr))))
                            (range n-runs))
      compiled-mean (/ (reduce + compiled-scores) n-runs)]
  (println (str "  Standard unfold mean score: " (.toFixed std-mean 3)))
  (println (str "  Compiled unfold mean score: " (.toFixed compiled-mean 3)))
  (assert-close "mean scores agree" std-mean compiled-mean 1.0))

;; ---------------------------------------------------------------------------
;; Test 5: Benchmark — compiled vs standard unfold
;; ---------------------------------------------------------------------------

(println "\n=== Test 5: Benchmark — compiled vs standard unfold ===")

(doseq [[label n-steps state-dim] [["small (T=10, D=2)" 10 2]
                                    ["medium (T=50, D=5)" 50 5]
                                    ["large (T=100, D=10)" 100 10]]]
  (println (str "\n  " label ":"))
  (let [noise-dim state-dim
        ;; Standard unfold
        kernel (dyn/auto-key
                 (gen [t state]
                   (let [new-state (trace :x (dist/gaussian
                                               (mx/multiply (mx/scalar 0.9) state)
                                               0.1))]
                     new-state)))
        unfold-gf (comb/unfold-combinator kernel)
        ;; Compiled step-fn
        step-fn (fn [state noise]
                  (let [mean (mx/multiply (mx/scalar 0.9) state)
                        std (mx/broadcast-to (mx/array [0.1]) [state-dim])
                        new-state (mx/add mean (mx/multiply std noise))
                        diff (mx/subtract new-state mean)
                        lp (mx/subtract
                             (mx/subtract
                               (mx/multiply (mx/scalar -0.5)
                                            (mx/sum (mx/divide (mx/multiply diff diff)
                                                               (mx/multiply std std))))
                               (mx/sum (mx/log std)))
                             (mx/scalar (* 0.5 state-dim (js/Math.log (* 2 js/Math.PI)))))]
                    [new-state lp]))
        n-bench 20
        t-std (bench "standard unfold"
                #(let [tr (p/simulate unfold-gf [n-steps (mx/zeros [state-dim])])]
                   (mx/materialize! (:score tr)))
                n-bench)
        _ (mx/clear-cache!)
        t-compiled (bench "compiled unfold"
                     #(let [compiled-fn (compiled/make-compiled-unfold
                                          step-fn n-steps state-dim noise-dim)
                            noise (rng/normal (rng/fresh-key) [n-steps noise-dim])
                            [s states sc] (compiled-fn (mx/zeros [state-dim]) noise)]
                        (mx/materialize! sc))
                     n-bench)
        _ (mx/clear-cache!)
        ;; Also bench with pre-compiled fn (amortize compilation cost)
        pre-compiled (compiled/make-compiled-unfold step-fn n-steps state-dim noise-dim)
        t-precompiled (bench "compiled (pre-built)"
                        #(let [noise (rng/normal (rng/fresh-key) [n-steps noise-dim])
                               [s states sc] (pre-compiled (mx/zeros [state-dim]) noise)]
                           (mx/materialize! sc))
                        n-bench)]
    (println (str "  speedup (incl. compile): " (.toFixed (/ t-std t-compiled) 1) "x"))
    (println (str "  speedup (pre-compiled):  " (.toFixed (/ t-std t-precompiled) 1) "x"))
    (mx/clear-cache!)))

(println "\n=== All compiled unfold tests complete ===")
