(ns genmlx.compiled-unfold-test
  "Tests for compiled unfold (Tier 2a).
   Validates correctness against the standard UnfoldCombinator,
   then benchmarks speedup."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
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

(defn bench [label f n-runs]
  (f) ;; warmup
  (let [start (js/Date.now)]
    (dotimes [_ n-runs] (f))
    (let [elapsed (- (js/Date.now) start)
          per-run (/ elapsed n-runs)]
      (println (str "  " label ": " (.toFixed per-run 1) "ms"))
      per-run)))

;; ---------------------------------------------------------------------------
;; Tests
;; ---------------------------------------------------------------------------

(deftest compiled-unfold-simulate-test
  (testing "compiled unfold simulate (random walk)"
    (let [state-dim 2
          noise-dim 2
          n-steps 10
          step-fn (fn [state noise]
                    (let [new-state (mx/add state noise)
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
      (is (= [state-dim] (vec (mx/shape final-state))) "final-state is [2]-shaped")
      (is (= [n-steps state-dim] (vec (mx/shape states))) "states is [10,2]-shaped")
      (is (= [] (vec (mx/shape total-score))) "total-score is scalar")
      (is (js/isFinite (mx/item total-score)) "total-score is finite")
      (let [expected-final (mx/sum noise 0)
            diff (mx/sum (mx/abs (mx/subtract final-state expected-final)))]
        (mx/materialize! diff)
        (is (h/close? 0.0 (mx/item diff) 1e-4) "final state = cumsum(noise)")))))

(deftest compiled-unfold-generate-test
  (testing "compiled unfold generate (linear dynamics + obs)"
    (let [state-dim 1
          noise-dim 1
          obs-dim 1
          n-steps 5
          transition-fn (fn [state]
                          [(mx/multiply (mx/scalar 0.9) state)
                           (mx/array [0.1])])
          observation-fn (fn [state]
                           [state
                            (mx/array [0.5])])
          step-fn (compiled/make-gaussian-step-with-obs
                    transition-fn observation-fn state-dim obs-dim)
          compiled-fn (compiled/make-compiled-unfold-generate
                        step-fn n-steps state-dim noise-dim obs-dim)
          key (rng/fresh-key)
          noise (rng/normal key [n-steps noise-dim])
          obs (mx/reshape (mx/array [1.0 0.9 0.8 0.7 0.6]) [n-steps obs-dim])
          init-state (mx/zeros [state-dim])
          [final-state states total-score total-weight] (compiled-fn init-state noise obs)]
      (mx/materialize! final-state states total-score total-weight)
      (is (= [state-dim] (vec (mx/shape final-state))) "final-state is [1]-shaped")
      (is (= [n-steps state-dim] (vec (mx/shape states))) "states is [5,1]-shaped")
      (is (= [] (vec (mx/shape total-score))) "total-score is scalar")
      (is (= [] (vec (mx/shape total-weight))) "total-weight is scalar")
      (is (js/isFinite (mx/item total-score)) "total-score is finite")
      (is (js/isFinite (mx/item total-weight)) "total-weight is finite"))))

(deftest high-level-api-test
  (testing "high-level API (Trace output)"
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
      (is (instance? genmlx.trace/Trace trace) "returns a Trace")
      (is (not= (:choices trace) cm/EMPTY) "trace has choices")
      (is (js/isFinite (mx/item (:score trace))) "trace has score")
      (is (some? (get-in (:retval trace) [:states])) "retval has :states"))))

(deftest statistical-agreement-test
  (testing "statistical agreement with standard UnfoldCombinator"
    (let [n-runs 200
          n-steps 5
          state-dim 1
          noise-dim 1
          kernel (dyn/auto-key
                   (gen [t state]
                     (let [new-state (trace :x (dist/gaussian (mx/multiply (mx/scalar 0.9) state) 0.1))]
                       new-state)))
          unfold-gf (comb/unfold-combinator kernel)
          step-fn (fn [state noise]
                    (let [mean (mx/multiply (mx/scalar 0.9) state)
                          std (mx/array [0.1])
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
          std-scores (mapv (fn [_]
                             (let [tr (p/simulate unfold-gf [n-steps (mx/zeros [state-dim])])]
                               (mx/item (:score tr))))
                           (range n-runs))
          std-mean (/ (reduce + std-scores) n-runs)
          compiled-scores (mapv (fn [_]
                                  (let [tr (compiled/compiled-unfold-simulate
                                             {:step-fn step-fn :n-steps n-steps
                                              :state-dim state-dim :noise-dim noise-dim}
                                             (mx/zeros [state-dim])
                                             (rng/fresh-key))]
                                    (mx/item (:score tr))))
                                (range n-runs))
          compiled-mean (/ (reduce + compiled-scores) n-runs)]
      (is (h/close? std-mean compiled-mean 1.0) "mean scores agree"))))

(deftest benchmark-test
  (testing "benchmark compiled vs standard unfold"
    (doseq [[label n-steps-b state-dim-b] [["small (T=10, D=2)" 10 2]
                                            ["medium (T=50, D=5)" 50 5]
                                            ["large (T=100, D=10)" 100 10]]]
      (let [noise-dim state-dim-b
            kernel (dyn/auto-key
                     (gen [t state]
                       (let [new-state (trace :x (dist/gaussian
                                                   (mx/multiply (mx/scalar 0.9) state)
                                                   0.1))]
                         new-state)))
            unfold-gf (comb/unfold-combinator kernel)
            step-fn (fn [state noise]
                      (let [mean (mx/multiply (mx/scalar 0.9) state)
                            std (mx/broadcast-to (mx/array [0.1]) [state-dim-b])
                            new-state (mx/add mean (mx/multiply std noise))
                            diff (mx/subtract new-state mean)
                            lp (mx/subtract
                                 (mx/subtract
                                   (mx/multiply (mx/scalar -0.5)
                                                (mx/sum (mx/divide (mx/multiply diff diff)
                                                                   (mx/multiply std std))))
                                   (mx/sum (mx/log std)))
                                 (mx/scalar (* 0.5 state-dim-b (js/Math.log (* 2 js/Math.PI)))))]
                        [new-state lp]))
            n-bench 20
            t-std (bench (str label " standard unfold")
                    #(let [tr (p/simulate unfold-gf [n-steps-b (mx/zeros [state-dim-b])])]
                       (mx/materialize! (:score tr)))
                    n-bench)
            _ (mx/clear-cache!)
            t-compiled (bench (str label " compiled unfold")
                         #(let [compiled-fn (compiled/make-compiled-unfold
                                              step-fn n-steps-b state-dim-b noise-dim)
                                noise (rng/normal (rng/fresh-key) [n-steps-b noise-dim])
                                [s states sc] (compiled-fn (mx/zeros [state-dim-b]) noise)]
                            (mx/materialize! sc))
                         n-bench)]
        (mx/clear-cache!)
        (is (pos? t-std) (str label ": standard ran"))
        (is (pos? t-compiled) (str label ": compiled ran"))))))

(cljs.test/run-tests)
