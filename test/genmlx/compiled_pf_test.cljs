(ns genmlx.compiled-pf-test
  "Tests for compiled particle filter (Tier 2c).
   Validates correctness and benchmarks against batched-smc-unfold."
  (:require [genmlx.mlx :as mx]
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
;; Test model: 1D random walk with Gaussian observations
;; State: x_t = 0.9 * x_{t-1} + noise(0, 0.1)
;; Obs:   y_t = x_t + noise(0, 0.5)
;; ---------------------------------------------------------------------------

(def state-dim 1)
(def noise-dim 1)
(def obs-dim 1)
(def n-steps 10)

;; Generate synthetic data
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
;; Test 1: Basic compiled particle filter
;; ---------------------------------------------------------------------------

(println "\n=== Test 1: Compiled particle filter (1D random walk) ===")

(let [n-particles 500
      transition-fn (fn [states]
                      [(mx/multiply (mx/scalar 0.9) states)  ;; mean [N,1]
                       (mx/array [0.1])])                     ;; std [1]
      observation-fn (fn [states]
                       [states                                 ;; obs-mean [N,1]
                        (mx/array [0.5])])                     ;; obs-std [1]
      step-fn (compiled/make-gaussian-particle-step
                transition-fn observation-fn state-dim obs-dim)
      init-states (mx/zeros [n-particles state-dim])
      result (compiled/compiled-particle-filter
               {:particle-step-fn step-fn
                :n-steps n-steps :n-particles n-particles
                :state-dim state-dim :noise-dim noise-dim :obs-dim obs-dim}
               init-states obs-tensor (rng/fresh-key))]
  (assert-true "returns final-states" (some? (:final-states result)))
  (assert-true "final-states shape [N,D]"
    (= [n-particles state-dim] (vec (mx/shape (:final-states result)))))
  (assert-true "log-ml is finite" (js/isFinite (:log-ml result)))
  (assert-true "log-ml is negative" (neg? (:log-ml result)))
  (println "  log-ML:" (.toFixed (:log-ml result) 4))
  ;; Check that particle mean is roughly near the last true state
  (let [mean-state (mx/item (mx/mean (:final-states result)))]
    (println "  particle mean:" (.toFixed mean-state 4)
             "  true:" (.toFixed (last true-states) 4))
    (assert-close "particle mean near truth" (last true-states) mean-state 0.5)))

;; ---------------------------------------------------------------------------
;; Test 2: Deterministic resampling consistency
;; ---------------------------------------------------------------------------

(println "\n=== Test 2: Deterministic resampling (low-level API) ===")

;; Same noise + uniform → same result via low-level make-compiled-particle-filter
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
      ;; Fixed noise and uniforms
      key (rng/fresh-key 42)
      [nk uk] (rng/split key)
      noise (rng/normal nk [n-steps-test n-particles noise-dim])
      uniforms (rng/uniform uk [n-steps-test 1])
      [s1 ml1] (compiled-fn init noise obs-t uniforms)
      _ (mx/materialize! ml1)
      [s2 ml2] (compiled-fn init noise obs-t uniforms)
      _ (mx/materialize! ml2)]
  (assert-close "same inputs → same log-ML" (mx/item ml1) (mx/item ml2) 1e-4))

;; ---------------------------------------------------------------------------
;; Test 3: Statistical validation — log-ML against batched-smc-unfold
;; ---------------------------------------------------------------------------

(println "\n=== Test 3: Statistical agreement with batched-smc-unfold ===")

(let [n-particles 500
      n-runs 20
      ;; Compiled particle filter
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
      ;; batched-smc-unfold with equivalent model
      ;; init-state is nil at t=0 for batched-smc-unfold, so use 0.0 as default
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
  (println "  Compiled PF mean log-ML:" (.toFixed compiled-mean 3))
  (println "  Batched SMC mean log-ML:" (.toFixed batched-mean 3))
  ;; They should be close (both are unbiased estimators of the same quantity)
  (assert-close "log-ML estimates agree" batched-mean compiled-mean 2.0))

;; ---------------------------------------------------------------------------
;; Test 4: Benchmark — compiled PF vs batched-smc-unfold
;; ---------------------------------------------------------------------------

(println "\n=== Test 4: Benchmark — compiled PF vs batched-smc-unfold ===")

(doseq [[label n-steps n-particles]
        [["small (T=5, N=100)" 5 100]
         ["medium (T=10, N=500)" 10 500]
         ["large (T=20, N=1000)" 20 1000]]]
  (println (str "\n  " label ":"))
  (let [;; Generate observations for this size
        obs-t (mx/reshape (mx/array (vec (repeatedly n-steps #(* 0.5 (js/Math.random)))))
                          [n-steps obs-dim])
        obs-seq (mapv (fn [i]
                        (genmlx.choicemap/choicemap
                          :y (mx/scalar (nth (mx/->clj (mx/reshape obs-t [n-steps])) i))))
                      (range n-steps))
        ;; Compiled PF setup
        transition-fn (fn [states]
                        [(mx/multiply (mx/scalar 0.9) states) (mx/array [0.1])])
        observation-fn (fn [states] [states (mx/array [0.5])])
        step-fn (compiled/make-gaussian-particle-step
                  transition-fn observation-fn state-dim obs-dim)
        ;; batched-smc-unfold kernel
        kernel (dyn/auto-key
                 (gen [t state]
                   (let [prior-mean (if (some? state)
                                      (mx/multiply (mx/scalar 0.9) state)
                                      (mx/scalar 0.0))
                         new-state (trace :x (dist/gaussian prior-mean 0.1))
                         _ (trace :y (dist/gaussian new-state 0.5))]
                     new-state)))
        n-bench 10
        ;; Benchmark batched-smc-unfold
        t-batched (bench "batched-smc-unfold"
                    #(smc/batched-smc-unfold
                       {:particles n-particles :key (rng/fresh-key)}
                       kernel nil obs-seq)
                    n-bench)
        _ (mx/clear-cache!)
        ;; Benchmark compiled PF (including compile time)
        t-compiled (bench "compiled PF (incl. compile)"
                     #(compiled/compiled-particle-filter
                        {:particle-step-fn step-fn
                         :n-steps n-steps :n-particles n-particles
                         :state-dim state-dim :noise-dim noise-dim :obs-dim obs-dim}
                        (mx/zeros [n-particles state-dim])
                        obs-t (rng/fresh-key))
                     n-bench)
        _ (mx/clear-cache!)
        ;; Benchmark pre-compiled PF
        pre-compiled (compiled/make-compiled-particle-filter
                       step-fn n-steps n-particles state-dim noise-dim obs-dim)
        t-precompiled (bench "compiled PF (pre-built)"
                        #(let [key (rng/fresh-key)
                               [nk uk] (rng/split key)
                               noise (rng/normal nk [n-steps n-particles noise-dim])
                               uniforms (rng/uniform uk [n-steps 1])
                               [states ml] (pre-compiled
                                             (mx/zeros [n-particles state-dim])
                                             noise obs-t uniforms)]
                           (mx/materialize! states ml))
                        n-bench)]
    (println (str "  speedup (incl. compile): " (.toFixed (/ t-batched t-compiled) 1) "x"))
    (println (str "  speedup (pre-compiled):  " (.toFixed (/ t-batched t-precompiled) 1) "x"))
    (mx/clear-cache!)))

(println "\n=== All compiled particle filter tests complete ===")
