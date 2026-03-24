(ns genmlx.compiled-pf-test
  "Tests for compiled particle filter (Tier 2c).
   Validates correctness and benchmarks against batched-smc-unfold."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.compiled :as compiled]
            [genmlx.inference.smc :as smc]
            [genmlx.vectorized :as vec])
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
;; Test model: 1D random walk with Gaussian observations
;; ---------------------------------------------------------------------------

(def state-dim 1)
(def noise-dim 1)
(def obs-dim 1)
(def n-steps 10)

(def true-states
  (loop [t 0, state 0.0, states []]
    (if (>= t n-steps)
      states
      (let [new-state (+ (* 0.9 state) (* 0.1 (- (* 2 (js/Math.random)) 1)))]
        (recur (inc t) new-state (conj states new-state))))))

(def observations
  (mapv (fn [x] (+ x (* 0.5 (- (* 2 (js/Math.random)) 1)))) true-states))

(def obs-tensor (mx/reshape (mx/array (vec observations)) [n-steps obs-dim]))

;; ---------------------------------------------------------------------------
;; Tests
;; ---------------------------------------------------------------------------

(deftest compiled-particle-filter-test
  (testing "basic compiled particle filter (1D random walk)"
    (let [n-particles 500
          transition-fn (fn [states]
                          [(mx/multiply (mx/scalar 0.9) states)
                           (mx/array [0.1])])
          observation-fn (fn [states]
                           [states
                            (mx/array [0.5])])
          step-fn (compiled/make-gaussian-particle-step
                    transition-fn observation-fn state-dim obs-dim)
          init-states (mx/zeros [n-particles state-dim])
          result (compiled/compiled-particle-filter
                   {:particle-step-fn step-fn
                    :n-steps n-steps :n-particles n-particles
                    :state-dim state-dim :noise-dim noise-dim :obs-dim obs-dim}
                   init-states obs-tensor (rng/fresh-key))]
      (is (some? (:final-states result)) "returns final-states")
      (is (= [n-particles state-dim] (vec (mx/shape (:final-states result)))) "final-states shape [N,D]")
      (is (js/isFinite (:log-ml result)) "log-ml is finite")
      (is (neg? (:log-ml result)) "log-ml is negative")
      (let [mean-state (mx/item (mx/mean (:final-states result)))]
        (is (h/close? (last true-states) mean-state 0.5) "particle mean near truth")))))

(deftest deterministic-resampling-test
  (testing "deterministic resampling consistency"
    (let [n-particles 100
          n-steps-test 5
          transition-fn (fn [states]
                          [(mx/multiply (mx/scalar 0.9) states) (mx/array [0.1])])
          observation-fn (fn [states] [states (mx/array [0.5])])
          step-fn (compiled/make-gaussian-particle-step
                    transition-fn observation-fn state-dim obs-dim)
          compiled-fn (compiled/make-compiled-particle-filter
                        step-fn n-steps-test n-particles state-dim noise-dim obs-dim)
          init (mx/zeros [n-particles state-dim])
          obs-t (mx/reshape (mx/array [1.0 0.9 0.8 0.7 0.6]) [n-steps-test obs-dim])
          key (rng/fresh-key 42)
          [nk uk] (rng/split key)
          noise (rng/normal nk [n-steps-test n-particles noise-dim])
          uniforms (rng/uniform uk [n-steps-test 1])
          [s1 ml1] (compiled-fn init noise obs-t uniforms)
          _ (mx/materialize! ml1)
          [s2 ml2] (compiled-fn init noise obs-t uniforms)
          _ (mx/materialize! ml2)]
      (is (h/close? (mx/item ml1) (mx/item ml2) 1e-4) "same inputs -> same log-ML"))))

(deftest statistical-agreement-test
  (testing "statistical agreement with batched-smc-unfold"
    (let [n-particles 500
          n-runs 20
          transition-fn (fn [states]
                          [(mx/multiply (mx/scalar 0.9) states) (mx/array [0.1])])
          observation-fn (fn [states] [states (mx/array [0.5])])
          step-fn (compiled/make-gaussian-particle-step
                    transition-fn observation-fn state-dim obs-dim)
          compiled-mls (mapv
                         (fn [_]
                           (:log-ml
                             (compiled/compiled-particle-filter
                               {:particle-step-fn step-fn
                                :n-steps n-steps :n-particles n-particles
                                :state-dim state-dim :noise-dim noise-dim :obs-dim obs-dim}
                               (mx/zeros [n-particles state-dim])
                               obs-tensor (rng/fresh-key))))
                         (range n-runs))
          compiled-mean (/ (reduce + compiled-mls) n-runs)
          kernel (dyn/auto-key
                   (gen [t state]
                     (let [prior-mean (if (some? state)
                                        (mx/multiply (mx/scalar 0.9) state)
                                        (mx/scalar 0.0))
                           new-state (trace :x (dist/gaussian prior-mean 0.1))
                           _ (trace :y (dist/gaussian new-state 0.5))]
                       new-state)))
          obs-seq (mapv (fn [y]
                          (genmlx.choicemap/choicemap :y (mx/scalar y)))
                        observations)
          batched-mls (mapv
                        (fn [_]
                          (let [r (smc/batched-smc-unfold
                                    {:particles n-particles :key (rng/fresh-key)}
                                    kernel nil obs-seq)]
                            (mx/item (:log-ml r))))
                        (range n-runs))
          batched-mean (/ (reduce + batched-mls) n-runs)]
      (is (h/close? batched-mean compiled-mean 2.0) "log-ML estimates agree"))))

(deftest benchmark-test
  (testing "benchmark compiled PF vs batched-smc-unfold"
    (doseq [[label n-steps-b n-particles-b]
            [["small (T=5, N=100)" 5 100]
             ["medium (T=10, N=500)" 10 500]
             ["large (T=20, N=1000)" 20 1000]]]
      (let [obs-t (mx/reshape (mx/array (vec (repeatedly n-steps-b #(* 0.5 (js/Math.random)))))
                              [n-steps-b obs-dim])
            obs-seq (mapv (fn [i]
                            (genmlx.choicemap/choicemap
                              :y (mx/scalar (nth (mx/->clj (mx/reshape obs-t [n-steps-b])) i))))
                          (range n-steps-b))
            transition-fn (fn [states]
                            [(mx/multiply (mx/scalar 0.9) states) (mx/array [0.1])])
            observation-fn (fn [states] [states (mx/array [0.5])])
            step-fn (compiled/make-gaussian-particle-step
                      transition-fn observation-fn state-dim obs-dim)
            kernel (dyn/auto-key
                     (gen [t state]
                       (let [prior-mean (if (some? state)
                                          (mx/multiply (mx/scalar 0.9) state)
                                          (mx/scalar 0.0))
                             new-state (trace :x (dist/gaussian prior-mean 0.1))
                             _ (trace :y (dist/gaussian new-state 0.5))]
                         new-state)))
            n-bench 10
            t-batched (bench (str label " batched-smc-unfold")
                        #(smc/batched-smc-unfold
                           {:particles n-particles-b :key (rng/fresh-key)}
                           kernel nil obs-seq)
                        n-bench)
            _ (mx/clear-cache!)
            t-compiled (bench (str label " compiled PF (incl. compile)")
                         #(compiled/compiled-particle-filter
                            {:particle-step-fn step-fn
                             :n-steps n-steps-b :n-particles n-particles-b
                             :state-dim state-dim :noise-dim noise-dim :obs-dim obs-dim}
                            (mx/zeros [n-particles-b state-dim])
                            obs-t (rng/fresh-key))
                         n-bench)]
        (mx/clear-cache!)
        ;; Just ensure both run without error -- speedup is informational
        (is (pos? t-batched) (str label ": batched ran"))
        (is (pos? t-compiled) (str label ": compiled ran"))))))

(cljs.test/run-tests)
