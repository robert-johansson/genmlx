(ns genmlx.compiled-smc-test
  "Level 2 WP-2 tests: compiled SMC bootstrap particle filter.

   Tests cover:
   1. generate-smc-noise produces correct shapes
   2. systematic-resample-tensor produces valid ancestors
   3. make-smc-extend-step builds extend for static kernels
   4. compiled-smc runs and produces valid results
   5. log-ML estimate is reasonable
   6. smc-result->traces produces TensorTraces"
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.gen :refer [gen]]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.compiled-ops :as compiled]
            [genmlx.tensor-trace :as tt]
            [genmlx.inference.compiled-smc :as csmc]))

;; ---------------------------------------------------------------------------
;; Test kernel: simple Gaussian random walk
;; ---------------------------------------------------------------------------

(def rw-kernel
  "Random walk kernel: state evolves as N(state, 1), observation is N(state, 0.5)."
  (gen [t state]
    (let [new-state (trace :x (dist/gaussian state 1))]
      (trace :y (dist/gaussian new-state 0.5))
      new-state)))

;; ---------------------------------------------------------------------------
;; Tests
;; ---------------------------------------------------------------------------

(deftest generate-smc-noise-test
  (testing "generate-smc-noise shapes"
    (let [key (rng/fresh-key 42)
          T 10 N 50 K 3
          noise (csmc/generate-smc-noise key T N K)]
      (is (some? (:extend-noise noise)) "has :extend-noise")
      (is (some? (:resample-uniforms noise)) "has :resample-uniforms")
      (mx/eval! (:extend-noise noise) (:resample-uniforms noise))
      (is (= [T N K] (mx/shape (:extend-noise noise))) "extend-noise shape")
      (is (= [T] (mx/shape (:resample-uniforms noise))) "resample-uniforms shape"))))

(deftest systematic-resample-tensor-test
  (testing "uniform weights"
    (let [N 10
          K 3
          particles (mx/ones [N K])
          log-weights (mx/zeros [N])
          uniform (mx/scalar 0.3)
          _ (mx/eval! particles log-weights uniform)
          {:keys [particles ancestors]} (csmc/systematic-resample-tensor
                                          particles log-weights uniform N)]
      (mx/eval! particles ancestors)
      (is (= [N K] (mx/shape particles)) "resampled shape")
      (is (= [N] (mx/shape ancestors)) "ancestors shape")
      (let [anc-vals (mx/->clj ancestors)]
        (is (every? #(and (>= % 0) (< % N)) anc-vals) "all ancestors in [0,N)"))))

  (testing "skewed weights"
    (let [N 8
          K 2
          particles (mx/reshape (mx/astype (mx/arange 0 (* N K) 1) mx/float32) [N K])
          log-weights (mx/array [-0.1 -10 -10 -10 -10 -10 -10 -10])
          uniform (mx/scalar 0.5)
          _ (mx/eval! particles log-weights uniform)
          {:keys [particles ancestors]} (csmc/systematic-resample-tensor
                                          particles log-weights uniform N)]
      (mx/eval! particles ancestors)
      (let [anc-vals (mx/->clj ancestors)]
        (is (> (count (filter zero? anc-vals)) (/ N 2)) "skewed: most ancestors are 0")))))

(deftest make-smc-extend-step-test
  (testing "extend step"
    (let [schema (:schema rw-kernel)
          source (:source rw-kernel)]
      (is (some? schema) "kernel has schema")
      (is (:static? schema) "kernel is static")

      (let [extend-fn (compiled/make-smc-extend-step schema source)]
        (is (some? extend-fn) "extend-fn not nil")

        (let [N 5
              K 2
              noise (rng/normal (rng/fresh-key 123) [N K])
              _ (mx/eval! noise)
              obs (cm/choicemap :y (mx/scalar 1.0))
              result (extend-fn noise [0 (mx/scalar 0.0)] obs)]
          (is (some? (:values-map result)) "has :values-map")
          (is (some? (:log-prob result)) "has :log-prob")
          (is (some? (:addr-index result)) "has :addr-index")
          (is (some? (:all-addrs result)) "has :all-addrs")

          (mx/eval! (:log-prob result))
          (is (= [N] (mx/shape (:log-prob result))) "log-prob shape is [N]")
          (is (some? (get (:values-map result) :x)) ":x in values-map")
          (is (some? (get (:values-map result) :y)) ":y in values-map")

          (let [x-vals (get (:values-map result) :x)]
            (mx/eval! x-vals)
            (is (= [N] (mx/shape x-vals)) ":x shape is [N]"))

          (let [lp-vals (mx/->clj (:log-prob result))]
            (is (every? js/isFinite lp-vals) "all log-probs finite")))))))

(deftest compiled-smc-test
  (testing "compiled-smc runs correctly"
    (let [key (rng/fresh-key 999)
          N 20
          T 5
          obs-seq (mapv (fn [t] (cm/choicemap :y (mx/scalar 2.0))) (range T))
          result (csmc/compiled-smc
                   {:particles N :key key}
                   rw-kernel (mx/scalar 0.0) obs-seq)]
      (is (some? (:log-ml result)) "result has :log-ml")
      (is (some? (:particles result)) "result has :particles")
      (is (some? (:addr-index result)) "result has :addr-index")

      (mx/eval! (:log-ml result) (:particles result))
      (is (= [N 2] (mx/shape (:particles result))) "particles shape")
      (is (js/isFinite (mx/item (:log-ml result))) "log-ml is finite")
      (is (< (mx/item (:log-ml result)) 0) "log-ml is negative")

      (is (= 2 (count (:addr-index result))) "addr-index has 2 entries")
      (is (some? (get (:addr-index result) :x)) ":x in addr-index")
      (is (some? (get (:addr-index result) :y)) ":y in addr-index"))))

(deftest smc-result-to-traces-test
  (testing "smc-result->traces"
    (let [key (rng/fresh-key 777)
          N 10
          T 3
          obs-seq (mapv (fn [_] (cm/choicemap :y (mx/scalar 1.5))) (range T))
          result (csmc/compiled-smc {:particles N :key key}
                                     rw-kernel (mx/scalar 0.0) obs-seq)
          traces (csmc/smc-result->traces result rw-kernel)]
      (is (= N (count traces)) "correct number of traces")
      (is (instance? tt/TensorTrace (first traces)) "first trace is TensorTrace")
      (let [t0 (first traces)
            choices (:choices t0)]
        (is (not= cm/EMPTY (cm/get-submap choices :x)) "choices has :x")
        (let [x-val (cm/get-value (cm/get-submap choices :x))]
          (mx/eval! x-val)
          (is (js/isFinite (mx/item x-val)) ":x value is finite"))))))

(deftest randomness-test
  (testing "different keys give different results"
    (let [N 10
          T 3
          obs-seq (mapv (fn [_] (cm/choicemap :y (mx/scalar 1.0))) (range T))
          r1 (csmc/compiled-smc {:particles N :key (rng/fresh-key 1)}
                                  rw-kernel (mx/scalar 0.0) obs-seq)
          r2 (csmc/compiled-smc {:particles N :key (rng/fresh-key 2)}
                                  rw-kernel (mx/scalar 0.0) obs-seq)]
      (mx/eval! (:log-ml r1) (:log-ml r2))
      (let [lml1 (mx/item (:log-ml r1))
            lml2 (mx/item (:log-ml r2))]
        (is (> (js/Math.abs (- lml1 lml2)) 1e-6) "different keys -> different log-ml")))))

(deftest callback-test
  (testing "callback works"
    (let [steps (atom [])
          result (csmc/compiled-smc
                   {:particles 10 :key (rng/fresh-key 42)
                    :callback (fn [info] (swap! steps conj (:step info)))}
                   rw-kernel (mx/scalar 0.0)
                   [(cm/choicemap :y (mx/scalar 1.0))
                    (cm/choicemap :y (mx/scalar 1.5))])]
      (is (= 2 (count @steps)) "callback called 2 times")
      (is (= [0 1] @steps) "steps are 0,1"))))

(deftest log-ml-consistency-test
  (testing "log-ML consistency"
    (let [N 50
          T 5
          n-runs 10
          obs-seq (mapv (fn [_] (cm/choicemap :y (mx/scalar 1.5))) (range T))
          compiled-lmls
          (mapv (fn [seed]
                  (let [r (csmc/compiled-smc {:particles N :key (rng/fresh-key seed)}
                                              rw-kernel (mx/scalar 0.0) obs-seq)]
                    (mx/eval! (:log-ml r))
                    (mx/item (:log-ml r))))
                (range n-runs))
          avg-compiled (/ (reduce + compiled-lmls) n-runs)]
      (is (js/isFinite avg-compiled) "avg log-ML is finite")
      (is (< avg-compiled 0) "avg log-ML is negative")
      (is (> avg-compiled -100) "avg log-ML > -100"))))

(deftest particle-scaling-test
  (testing "larger particle count gives tighter estimate"
    (let [T 3
          obs-seq (mapv (fn [_] (cm/choicemap :y (mx/scalar 1.0))) (range T))
          run-avg (fn [N n-runs]
                    (let [lmls (mapv (fn [s]
                                       (let [r (csmc/compiled-smc
                                                 {:particles N :key (rng/fresh-key (+ s 1000))}
                                                 rw-kernel (mx/scalar 0.0) obs-seq)]
                                         (mx/eval! (:log-ml r))
                                         (mx/item (:log-ml r))))
                                     (range n-runs))]
                      {:mean (/ (reduce + lmls) n-runs)
                       :var (let [m (/ (reduce + lmls) n-runs)]
                              (/ (reduce + (map #(* (- % m) (- % m)) lmls)) n-runs))}))
          small (run-avg 10 20)
          large (run-avg 100 20)]
      (is (< (:var large) (:var small)) "N=100 variance < N=10 variance"))))

(cljs.test/run-tests)
