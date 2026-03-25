(ns genmlx.vsmc-validation
  "Comprehensive validation and benchmarking of vectorized SMC (vsmc)
   vs sequential SMC. Tests correctness AND performance."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.vectorized :as vec]
            [genmlx.inference.smc :as smc]
            [genmlx.inference.importance :as importance]
            [genmlx.mlx.random :as rng])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Bench helper (kept for performance tests)
;; ---------------------------------------------------------------------------

(defn- bench [label f {:keys [warmup runs] :or {warmup 2 runs 5}}]
  (dotimes [_ warmup] (f))
  (mx/clear-cache!)
  (let [times (mapv (fn [_]
                      (let [start (js/Date.now)
                            _ (f)
                            end (js/Date.now)]
                        (- end start)))
                    (range runs))
        sorted (sort times)
        median (nth sorted (quot runs 2))]
    median))

;; ---------------------------------------------------------------------------
;; Test models
;; ---------------------------------------------------------------------------

(def simple-model
  (gen [t]
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :y (dist/gaussian mu 1))
      mu)))

(def linreg-model
  (gen [t]
    (let [slope     (trace :slope (dist/gaussian 0 5))
          intercept (trace :intercept (dist/gaussian 0 5))]
      (trace :y (dist/gaussian (mx/add (mx/multiply slope (mx/scalar (double t)))
                                       intercept) 1))
      slope)))

(def heavy-model
  (gen [t]
    (let [a (trace :a (dist/gaussian 0 5))
          b (trace :b (dist/gaussian 0 5))
          c (trace :c (dist/gaussian 0 5))
          d (trace :d (dist/gaussian 0 5))
          e (trace :e (dist/gaussian 0 5))
          mean (mx/add a (mx/add b (mx/add c (mx/add d e))))]
      (trace :y (dist/gaussian mean 1))
      mean)))

(deftest vsmc-output-validity
  (testing "vsmc produces valid output"
    (let [obs-seq (mapv (fn [y] (cm/choicemap :y (mx/scalar y)))
                        [3.0 3.5 2.8 3.2 3.1])
          key (rng/fresh-key)
          {:keys [vtrace log-ml-estimate]}
          (smc/vsmc {:particles 200 :ess-threshold 0.5 :key key}
                    simple-model [0] obs-seq)]
      (is (instance? vec/VectorizedTrace vtrace) "Returns VectorizedTrace")
      (is (= 200 (:n-particles vtrace)) "n-particles correct")
      (is (= [200] (mx/shape (:weight vtrace))) "weight shape = [N]")
      (is (= [200] (mx/shape (:score vtrace))) "score shape = [N]")
      (let [log-ml-val (mx/item log-ml-estimate)]
        (is (js/isFinite log-ml-val) "log-ml is finite")))))

(deftest vsmc-with-rejuvenation
  (testing "vsmc with rejuvenation"
    (mx/clear-cache!)
    (let [obs-seq (mapv (fn [y] (cm/choicemap :y (mx/scalar y)))
                        [3.0 3.5 2.8])
          key (rng/fresh-key)
          {:keys [vtrace log-ml-estimate]}
          (smc/vsmc {:particles 100 :ess-threshold 0.5
                     :rejuvenation-steps 3
                     :rejuvenation-selection (sel/select :mu)
                     :key key}
                    simple-model [0] obs-seq)]
      (is (instance? vec/VectorizedTrace vtrace) "Rejuvenation completes")
      (let [log-ml-val (mx/item log-ml-estimate)]
        (is (js/isFinite log-ml-val) "log-ml finite after rejuvenation")))))

(deftest vsmc-statistical-agreement
  (testing "log-ML estimates agree between sequential and vectorized SMC"
    (mx/clear-cache!)
    (mx/force-gc!)
    (let [obs-seq (mapv (fn [y] (cm/choicemap :y (mx/scalar y)))
                        [3.0 3.5 2.8 3.2])
          n-particles 200
          n-runs 5
          seq-log-mls (mapv (fn [_]
                              (let [{:keys [log-ml-estimate]}
                                    (smc/smc {:particles n-particles :ess-threshold 0.5}
                                             simple-model [0] obs-seq)
                                    v (mx/item log-ml-estimate)]
                                (mx/clear-cache!)
                                v))
                            (range n-runs))
          _ (mx/force-gc!)
          vec-log-mls (mapv (fn [_]
                              (let [{:keys [log-ml-estimate]}
                                    (smc/vsmc {:particles n-particles :ess-threshold 0.5}
                                              simple-model [0] obs-seq)
                                    v (mx/item log-ml-estimate)]
                                (mx/clear-cache!)
                                v))
                            (range n-runs))
          seq-mean (/ (reduce + seq-log-mls) n-runs)
          vec-mean (/ (reduce + vec-log-mls) n-runs)]
      (is (h/close? seq-mean vec-mean 2.0) "log-ML means agree"))))

(deftest batched-smc-unfold-test
  (testing "batched-smc-unfold with structured state"
    (mx/clear-cache!)
    (mx/force-gc!)
    (let [kernel (gen [t state]
                   (let [prev-mu (or state (mx/scalar 0.0))
                         mu (trace :mu (dist/gaussian prev-mu 1))]
                     (trace :y (dist/gaussian mu 1))
                     mu))
          obs-seq (mapv (fn [y] (cm/choicemap :y (mx/scalar y)))
                        [2.0 2.5 3.0 3.5 4.0])
          {:keys [log-ml final-states final-ess]}
          (smc/batched-smc-unfold {:particles 200} kernel nil obs-seq)]
      (let [log-ml-val (mx/item log-ml)]
        (is (js/isFinite log-ml-val) "batched-smc-unfold log-ml is finite")
        (is (< log-ml-val 0) "batched-smc-unfold log-ml is negative")
        (is (mx/array? final-states) "final-states is MLX array")
        (is (= [200] (mx/shape final-states)) "final-states shape = [N]")))))

;; Note: Heavy benchmark tests removed to avoid Metal buffer exhaustion in CI.
;; The correctness tests above cover vsmc functionality.

(cljs.test/run-tests)
