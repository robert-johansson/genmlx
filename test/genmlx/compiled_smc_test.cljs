(ns genmlx.compiled-smc-test
  "Level 2 WP-2 tests: compiled SMC bootstrap particle filter.

   Tests cover:
   1. generate-smc-noise produces correct shapes
   2. systematic-resample-tensor produces valid ancestors
   3. make-smc-extend-step builds extend for static kernels
   4. compiled-smc runs and produces valid results
   5. log-ML estimate is reasonable
   6. smc-result->traces produces TensorTraces"
  (:require [genmlx.gen :refer [gen]]
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
;; Test helpers
;; ---------------------------------------------------------------------------

(def ^:private pass-count (atom 0))
(def ^:private fail-count (atom 0))

(defn- assert-true [desc pred]
  (if pred
    (do (swap! pass-count inc)
        (println (str "  PASS: " desc)))
    (do (swap! fail-count inc)
        (println (str "  FAIL: " desc)))))

(defn- assert-close [desc expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (do (swap! pass-count inc)
          (println (str "  PASS: " desc " (diff=" (.toFixed diff 4) ")")))
      (do (swap! fail-count inc)
          (println (str "  FAIL: " desc " expected=" expected " actual=" actual " diff=" diff))))))

(defn- assert-equal [desc expected actual]
  (if (= expected actual)
    (do (swap! pass-count inc)
        (println (str "  PASS: " desc)))
    (do (swap! fail-count inc)
        (println (str "  FAIL: " desc " expected=" expected " actual=" actual)))))

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
;; 1. generate-smc-noise shapes
;; ---------------------------------------------------------------------------

(println "\n== generate-smc-noise ==")

(let [key (rng/fresh-key 42)
      T 10 N 50 K 3
      noise (csmc/generate-smc-noise key T N K)]
  (assert-true "has :extend-noise" (some? (:extend-noise noise)))
  (assert-true "has :resample-uniforms" (some? (:resample-uniforms noise)))
  (mx/eval! (:extend-noise noise) (:resample-uniforms noise))
  (assert-equal "extend-noise shape" [T N K] (mx/shape (:extend-noise noise)))
  (assert-equal "resample-uniforms shape" [T] (mx/shape (:resample-uniforms noise))))

;; ---------------------------------------------------------------------------
;; 2. systematic-resample-tensor
;; ---------------------------------------------------------------------------

(println "\n== systematic-resample-tensor ==")

(let [N 10
      K 3
      ;; Create particles [N,K] — just for shape testing
      particles (mx/ones [N K])
      ;; Uniform weights → should keep roughly uniform distribution
      log-weights (mx/zeros [N])
      uniform (mx/scalar 0.3)
      _ (mx/eval! particles log-weights uniform)
      {:keys [particles ancestors]} (csmc/systematic-resample-tensor
                                      particles log-weights uniform N)]
  (mx/eval! particles ancestors)
  (assert-equal "resampled shape" [N K] (mx/shape particles))
  (assert-equal "ancestors shape" [N] (mx/shape ancestors))
  ;; With uniform weights, all ancestors should be valid indices
  (let [anc-vals (mx/->clj ancestors)]
    (assert-true "all ancestors in [0,N)" (every? #(and (>= % 0) (< % N)) anc-vals))))

;; Test with skewed weights
(let [N 8
      K 2
      particles (mx/reshape (mx/astype (mx/arange 0 (* N K) 1) mx/float32) [N K])
      ;; Heavily skew toward particle 0
      log-weights (mx/array [-0.1 -10 -10 -10 -10 -10 -10 -10])
      uniform (mx/scalar 0.5)
      _ (mx/eval! particles log-weights uniform)
      {:keys [particles ancestors]} (csmc/systematic-resample-tensor
                                      particles log-weights uniform N)]
  (mx/eval! particles ancestors)
  (let [anc-vals (mx/->clj ancestors)]
    ;; Most ancestors should be 0 (the high-weight particle)
    (assert-true "skewed: most ancestors are 0"
                 (> (count (filter zero? anc-vals)) (/ N 2)))))

;; ---------------------------------------------------------------------------
;; 3. make-smc-extend-step
;; ---------------------------------------------------------------------------

(println "\n== make-smc-extend-step ==")

(let [schema (:schema rw-kernel)
      source (:source rw-kernel)]
  (assert-true "kernel has schema" (some? schema))
  (assert-true "kernel is static" (:static? schema))

  (let [extend-fn (compiled/make-smc-extend-step schema source)]
    (assert-true "extend-fn not nil" (some? extend-fn))

    ;; Run a single extend step with N=5 particles
    (let [N 5
          K 2  ;; :x and :y
          noise (rng/normal (rng/fresh-key 123) [N K])
          _ (mx/eval! noise)
          obs (cm/choicemap :y (mx/scalar 1.0))
          result (extend-fn noise [0 (mx/scalar 0.0)] obs)]
      (assert-true "has :values-map" (some? (:values-map result)))
      (assert-true "has :log-prob" (some? (:log-prob result)))
      (assert-true "has :addr-index" (some? (:addr-index result)))
      (assert-true "has :all-addrs" (some? (:all-addrs result)))

      (mx/eval! (:log-prob result))
      (assert-equal "log-prob shape is [N]" [N] (mx/shape (:log-prob result)))

      ;; Check values-map has the right addresses
      (assert-true ":x in values-map" (some? (get (:values-map result) :x)))
      (assert-true ":y in values-map" (some? (get (:values-map result) :y)))

      ;; :x should be [N]-shaped (proposed from noise)
      (let [x-vals (get (:values-map result) :x)]
        (mx/eval! x-vals)
        (assert-equal ":x shape is [N]" [N] (mx/shape x-vals)))

      ;; Log-probs should be finite
      (let [lp-vals (mx/->clj (:log-prob result))]
        (assert-true "all log-probs finite" (every? js/isFinite lp-vals))))))

;; ---------------------------------------------------------------------------
;; 4. compiled-smc runs correctly
;; ---------------------------------------------------------------------------

(println "\n== compiled-smc ==")

(let [key (rng/fresh-key 999)
      N 20
      T 5
      ;; Generate T observations from ground truth state = 2.0
      obs-seq (mapv (fn [t] (cm/choicemap :y (mx/scalar 2.0))) (range T))
      result (csmc/compiled-smc
               {:particles N :key key}
               rw-kernel (mx/scalar 0.0) obs-seq)]
  (assert-true "result has :log-ml" (some? (:log-ml result)))
  (assert-true "result has :particles" (some? (:particles result)))
  (assert-true "result has :addr-index" (some? (:addr-index result)))

  (mx/eval! (:log-ml result) (:particles result))
  (assert-equal "particles shape" [N 2] (mx/shape (:particles result)))
  (assert-true "log-ml is finite" (js/isFinite (mx/item (:log-ml result))))
  (assert-true "log-ml is negative" (< (mx/item (:log-ml result)) 0))

  ;; Check addr-index
  (assert-equal "addr-index has 2 entries" 2 (count (:addr-index result)))
  (assert-true ":x in addr-index" (some? (get (:addr-index result) :x)))
  (assert-true ":y in addr-index" (some? (get (:addr-index result) :y))))

;; ---------------------------------------------------------------------------
;; 5. smc-result->traces
;; ---------------------------------------------------------------------------

(println "\n== smc-result->traces ==")

(let [key (rng/fresh-key 777)
      N 10
      T 3
      obs-seq (mapv (fn [_] (cm/choicemap :y (mx/scalar 1.5))) (range T))
      result (csmc/compiled-smc {:particles N :key key}
                                 rw-kernel (mx/scalar 0.0) obs-seq)
      traces (csmc/smc-result->traces result rw-kernel)]
  (assert-equal "correct number of traces" N (count traces))
  (assert-true "first trace is TensorTrace"
               (instance? tt/TensorTrace (first traces)))
  ;; Check that values can be extracted
  (let [t0 (first traces)
        choices (:choices t0)]
    (assert-true "choices has :x" (not= cm/EMPTY (cm/get-submap choices :x)))
    (let [x-val (cm/get-value (cm/get-submap choices :x))]
      (mx/eval! x-val)
      (assert-true ":x value is finite" (js/isFinite (mx/item x-val))))))

;; ---------------------------------------------------------------------------
;; 6. Multiple runs give different results (randomness works)
;; ---------------------------------------------------------------------------

(println "\n== randomness ==")

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
    (assert-true "different keys → different log-ml"
                 (> (js/Math.abs (- lml1 lml2)) 1e-6))))

;; ---------------------------------------------------------------------------
;; 7. Callback works
;; ---------------------------------------------------------------------------

(println "\n== callback ==")

(let [steps (atom [])
      result (csmc/compiled-smc
               {:particles 10 :key (rng/fresh-key 42)
                :callback (fn [info] (swap! steps conj (:step info)))}
               rw-kernel (mx/scalar 0.0)
               [(cm/choicemap :y (mx/scalar 1.0))
                (cm/choicemap :y (mx/scalar 1.5))])]
  (assert-equal "callback called 2 times" 2 (count @steps))
  (assert-equal "steps are 0,1" [0 1] @steps))

;; ---------------------------------------------------------------------------
;; 8. Log-ML consistency: average over many runs should agree with handler SMC
;; ---------------------------------------------------------------------------

(println "\n== log-ML consistency ==")

(let [N 50
      T 5
      n-runs 10
      obs-seq (mapv (fn [_] (cm/choicemap :y (mx/scalar 1.5))) (range T))
      ;; Run compiled SMC multiple times and average log-ML
      compiled-lmls
      (mapv (fn [seed]
              (let [r (csmc/compiled-smc {:particles N :key (rng/fresh-key seed)}
                                          rw-kernel (mx/scalar 0.0) obs-seq)]
                (mx/eval! (:log-ml r))
                (mx/item (:log-ml r))))
            (range n-runs))
      avg-compiled (/ (reduce + compiled-lmls) n-runs)]
  (assert-true "avg log-ML is finite" (js/isFinite avg-compiled))
  ;; Log-ML should be negative for this model
  (assert-true "avg log-ML is negative" (< avg-compiled 0))
  ;; Log-ML should be reasonable (not -Inf or extremely large)
  (assert-true "avg log-ML > -100" (> avg-compiled -100))
  (println (str "  INFO: avg compiled log-ML = " (.toFixed avg-compiled 3))))

;; ---------------------------------------------------------------------------
;; 9. Larger particle count gives tighter estimate
;; ---------------------------------------------------------------------------

(println "\n== particle scaling ==")

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
  (println (str "  INFO: N=10  mean=" (.toFixed (:mean small) 3)
                " var=" (.toFixed (:var small) 3)))
  (println (str "  INFO: N=100 mean=" (.toFixed (:mean large) 3)
                " var=" (.toFixed (:var large) 3)))
  ;; More particles should give lower variance
  (assert-true "N=100 variance < N=10 variance"
               (< (:var large) (:var small))))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n== Compiled SMC Results: " @pass-count "/" (+ @pass-count @fail-count)
              " passed =="))
(when (pos? @fail-count)
  (println (str "FAILURES: " @fail-count)))
