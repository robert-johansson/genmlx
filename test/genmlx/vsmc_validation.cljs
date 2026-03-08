(ns genmlx.vsmc-validation
  "Comprehensive validation and benchmarking of vectorized SMC (vsmc)
   vs sequential SMC. Tests correctness AND performance."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.vectorized :as vec]
            [genmlx.inference.smc :as smc]
            [genmlx.inference.importance :as importance]
            [genmlx.mlx.random :as rng])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn assert-true [msg pred]
  (if pred
    (println (str "  PASS: " msg))
    (println (str "  FAIL: " msg))))

(defn assert-close [msg expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (println (str "  PASS: " msg " (expected=" (.toFixed expected 3)
                    " actual=" (.toFixed actual 3) ")"))
      (println (str "  FAIL: " msg " (expected=" (.toFixed expected 3)
                    " actual=" (.toFixed actual 3) " diff=" (.toFixed diff 3) ")")))))

(defn bench [label f {:keys [warmup runs] :or {warmup 2 runs 5}}]
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
    (println (str "  " label ": " median "ms (median of " runs ")"))
    median))

;; ---------------------------------------------------------------------------
;; Test models
;; ---------------------------------------------------------------------------

;; Model 1: Simple — mu ~ N(0,10), y_t ~ N(mu, 1)
(def simple-model
  (gen [t]
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :y (dist/gaussian mu 1))
      mu)))

;; Model 2: Two latent variables — slope + intercept
(def linreg-model
  (gen [t]
    (let [slope     (trace :slope (dist/gaussian 0 5))
          intercept (trace :intercept (dist/gaussian 0 5))]
      (trace :y (dist/gaussian (mx/add (mx/multiply slope (mx/scalar (double t)))
                                       intercept) 1))
      slope)))

;; Model 3: Heavier — 5 latent variables
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

;; ---------------------------------------------------------------------------
;; Test 1: vsmc produces valid output
;; ---------------------------------------------------------------------------

(println "\n=== Test 1: vsmc output validity ===")

(let [obs-seq (mapv (fn [y] (cm/choicemap :y (mx/scalar y)))
                    [3.0 3.5 2.8 3.2 3.1])
      key (rng/fresh-key)
      {:keys [vtrace log-ml-estimate]}
      (smc/vsmc {:particles 200 :ess-threshold 0.5 :key key}
                simple-model [0] obs-seq)]
  (assert-true "Returns VectorizedTrace"
               (instance? vec/VectorizedTrace vtrace))
  (assert-true "n-particles correct"
               (= 200 (:n-particles vtrace)))
  (assert-true "weight shape = [N]"
               (= [200] (mx/shape (:weight vtrace))))
  (assert-true "score shape = [N]"
               (= [200] (mx/shape (:score vtrace))))
  (let [log-ml-val (mx/item log-ml-estimate)]
    (assert-true "log-ml is finite" (js/isFinite log-ml-val))
    (println (str "  log-ml = " (.toFixed log-ml-val 3)))))

;; ---------------------------------------------------------------------------
;; Test 2: vsmc with rejuvenation
;; ---------------------------------------------------------------------------

(println "\n=== Test 2: vsmc with rejuvenation ===")
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
  (assert-true "Rejuvenation completes"
               (instance? vec/VectorizedTrace vtrace))
  (let [log-ml-val (mx/item log-ml-estimate)]
    (assert-true "log-ml finite after rejuvenation" (js/isFinite log-ml-val))
    (println (str "  log-ml = " (.toFixed log-ml-val 3)))))

;; ---------------------------------------------------------------------------
;; Test 3: vsmc vs sequential smc — statistical agreement
;; ---------------------------------------------------------------------------

(println "\n=== Test 3: Statistical agreement (log-ML estimates) ===")
(mx/clear-cache!)
(mx/force-gc!)

(let [obs-seq (mapv (fn [y] (cm/choicemap :y (mx/scalar y)))
                    [3.0 3.5 2.8 3.2])
      n-particles 200
      n-runs 5
      ;; Run sequential SMC multiple times (with GC between runs)
      seq-log-mls (mapv (fn [_]
                          (let [{:keys [log-ml-estimate]}
                                (smc/smc {:particles n-particles :ess-threshold 0.5}
                                         simple-model [0] obs-seq)
                                v (mx/item log-ml-estimate)]
                            (mx/clear-cache!)
                            v))
                        (range n-runs))
      _ (mx/force-gc!)
      ;; Run vectorized SMC multiple times
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
  (println (str "  Sequential SMC mean log-ML: " (.toFixed seq-mean 3)
               " (runs: " (mapv #(.toFixed % 2) seq-log-mls) ")"))
  (println (str "  Vectorized SMC mean log-ML: " (.toFixed vec-mean 3)
               " (runs: " (mapv #(.toFixed % 2) vec-log-mls) ")"))
  ;; They should agree within Monte Carlo noise
  (assert-close "log-ML means agree" seq-mean vec-mean 2.0))

;; ---------------------------------------------------------------------------
;; Test 4: batched-smc-unfold works with structured state
;; ---------------------------------------------------------------------------

(println "\n=== Test 4: batched-smc-unfold with structured state ===")
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
    (assert-true "batched-smc-unfold log-ml is finite" (js/isFinite log-ml-val))
    (assert-true "batched-smc-unfold log-ml is negative" (< log-ml-val 0))
    (println (str "  log-ml = " (.toFixed log-ml-val 3)))
    (assert-true "final-states is MLX array" (mx/array? final-states))
    (assert-true "final-states shape = [N]" (= [200] (mx/shape final-states)))))

;; ---------------------------------------------------------------------------
;; Benchmark 1: Sequential vs Vectorized SMC — simple model
;; ---------------------------------------------------------------------------

(println "\n=== Benchmark 1: Simple model (1 latent, 5 steps) ===")
(mx/clear-cache!)
(mx/force-gc!)

(doseq [n [50 200 500]]
  (let [obs-seq (mapv (fn [y] (cm/choicemap :y (mx/scalar y)))
                      [3.0 3.5 2.8 3.2 3.1])
        key (rng/fresh-key)]
    (println (str "\nN=" n " particles:"))
    (let [t-seq (bench "Sequential SMC"
                  (fn [] (smc/smc {:particles n :ess-threshold 0.5}
                                  simple-model [0] obs-seq))
                  {:warmup 1 :runs 3})
          _ (do (mx/clear-cache!) (mx/force-gc!))
          t-vec (bench "Vectorized SMC"
                  (fn [] (smc/vsmc {:particles n :ess-threshold 0.5 :key key}
                                   simple-model [0] obs-seq))
                  {:warmup 1 :runs 3})
          speedup (if (pos? t-vec) (/ t-seq t-vec) ##Inf)]
      (println (str "  Speedup: " (.toFixed speedup 1) "x")))
    (mx/clear-cache!)))

;; ---------------------------------------------------------------------------
;; Benchmark 2: Two-latent model, more timesteps
;; ---------------------------------------------------------------------------

(println "\n=== Benchmark 2: Linreg model (2 latents, 10 steps) ===")
(mx/clear-cache!)

(doseq [n [50 200]]
  (let [obs-seq (mapv (fn [t] (cm/choicemap :y (mx/scalar (+ (* 2.0 t) 1.0 (* 0.5 (js/Math.random))))))
                      (range 10))
        key (rng/fresh-key)]
    (println (str "\nN=" n " particles:"))
    (let [t-seq (bench "Sequential SMC"
                  (fn [] (smc/smc {:particles n :ess-threshold 0.5}
                                  linreg-model [0] obs-seq))
                  {:warmup 1 :runs 3})
          _ (do (mx/clear-cache!) (mx/force-gc!))
          t-vec (bench "Vectorized SMC"
                  (fn [] (smc/vsmc {:particles n :ess-threshold 0.5 :key key}
                                   linreg-model [0] obs-seq))
                  {:warmup 1 :runs 3})
          speedup (if (pos? t-vec) (/ t-seq t-vec) ##Inf)]
      (println (str "  Speedup: " (.toFixed speedup 1) "x")))
    (mx/clear-cache!)))

;; ---------------------------------------------------------------------------
;; Benchmark 3: Heavy model (5 latents)
;; ---------------------------------------------------------------------------

(println "\n=== Benchmark 3: Heavy model (5 latents, 5 steps) ===")
(mx/clear-cache!)

(doseq [n [50 200]]
  (let [obs-seq (mapv (fn [_] (cm/choicemap :y (mx/scalar 5.0)))
                      (range 5))
        key (rng/fresh-key)]
    (println (str "\nN=" n " particles:"))
    (let [t-seq (bench "Sequential SMC"
                  (fn [] (smc/smc {:particles n :ess-threshold 0.5}
                                  heavy-model [0] obs-seq))
                  {:warmup 1 :runs 3})
          _ (do (mx/clear-cache!) (mx/force-gc!))
          t-vec (bench "Vectorized SMC"
                  (fn [] (smc/vsmc {:particles n :ess-threshold 0.5 :key key}
                                   heavy-model [0] obs-seq))
                  {:warmup 1 :runs 3})
          speedup (if (pos? t-vec) (/ t-seq t-vec) ##Inf)]
      (println (str "  Speedup: " (.toFixed speedup 1) "x")))
    (mx/clear-cache!)))

;; ---------------------------------------------------------------------------
;; Benchmark 4: Rejuvenation overhead
;; ---------------------------------------------------------------------------

(println "\n=== Benchmark 4: Rejuvenation overhead (200 particles, 5 steps) ===")
(mx/clear-cache!)

(doseq [mh-steps [0 1 3 5]]
  (let [obs-seq (mapv (fn [y] (cm/choicemap :y (mx/scalar y)))
                      [3.0 3.5 2.8 3.2 3.1])
        key (rng/fresh-key)]
    (println (str "\n" mh-steps " MH rejuvenation steps:"))
    (bench "Vectorized SMC"
      (fn [] (smc/vsmc {:particles 200 :ess-threshold 0.5
                        :rejuvenation-steps mh-steps
                        :rejuvenation-selection (sel/select :mu)
                        :key key}
                       simple-model [0] obs-seq))
      {:warmup 1 :runs 3})
    (mx/clear-cache!)))

;; ---------------------------------------------------------------------------
;; Benchmark 5: batched-smc-unfold vs smc-unfold
;; ---------------------------------------------------------------------------

(println "\n=== Benchmark 5: smc-unfold vs batched-smc-unfold ===")
(mx/clear-cache!)

(let [kernel (gen [t state]
               (let [prev-mu (or state (mx/scalar 0.0))
                     mu (trace :mu (dist/gaussian prev-mu 1))]
                 (trace :y (dist/gaussian mu 1))
                 mu))
      obs-seq (mapv (fn [y] (cm/choicemap :y (mx/scalar y)))
                    [2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5])]
  (doseq [n [50 200 500]]
    (println (str "\nN=" n " particles, 10 steps:"))
    (let [t-seq (bench "smc-unfold (sequential)"
                  (fn [] (smc/smc-unfold {:particles n} kernel nil obs-seq))
                  {:warmup 1 :runs 3})
          _ (do (mx/clear-cache!) (mx/force-gc!))
          t-bat (bench "batched-smc-unfold"
                  (fn [] (smc/batched-smc-unfold {:particles n} kernel nil obs-seq))
                  {:warmup 1 :runs 3})
          speedup (if (pos? t-bat) (/ t-seq t-bat) ##Inf)]
      (println (str "  Speedup: " (.toFixed speedup 1) "x")))
    (mx/clear-cache!)))

(println "\n=== All vsmc validation and benchmarks complete ===")
