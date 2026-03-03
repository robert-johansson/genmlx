(ns genmlx.perfbench-large
  "Large-model IS benchmark: linear regression with D features.

   Matches GenJAX large model benchmark:
   - Design matrix: X[j,i] = sin(π(i+1) * xj), xj = j/(M-1), M=50
   - D latent weights ~ Normal(0,1), σ=0.05
   - Observations: y = X @ w_true (noise-free, deterministic)

   Usage:
     bun run --bun nbb test/genmlx/perfbench_large.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.choicemap :as cm]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.inference.importance :as is])
  (:require-macros [genmlx.gen :refer [gen]]))

(def M 50)
(def noise-std 0.05)

;; ---------------------------------------------------------------------------
;; Design matrix & data (deterministic, matches JAX benchmark)
;; ---------------------------------------------------------------------------

(defn make-design-matrix [D]
  (mapv (fn [j]
          (let [xj (/ (double j) (dec M))]
            (mapv (fn [i]
                    (js/Math.sin (* js/Math.PI (inc i) xj)))
                  (range D))))
        (range M)))

(defn make-true-weights [D]
  (mapv (fn [i] (/ (if (even? i) 1.0 -1.0) (inc i)))
        (range D)))

(defn make-observations-data [X-rows true-weights]
  (mapv (fn [row]
          (reduce + (map * row true-weights)))
        X-rows))

;; ---------------------------------------------------------------------------
;; Model factory — per-site (original)
;; ---------------------------------------------------------------------------

(defn make-large-model [D]
  (dyn/auto-key
    (gen [X-rows]
      (let [weights (mapv (fn [i] (trace (keyword (str "w" i)) (dist/gaussian 0 1)))
                          (range D))]
        (doseq [[j row] (map-indexed vector X-rows)]
          (let [dot-terms (map (fn [w x-val] (mx/multiply w (mx/scalar x-val)))
                               weights row)
                y-pred (reduce mx/add (mx/scalar 0.0) dot-terms)]
            (trace (keyword (str "y" j))
                   (dist/gaussian y-pred noise-std))))))))

(defn make-observations-cm [X-rows true-weights]
  (let [ys (make-observations-data X-rows true-weights)]
    (apply cm/choicemap
      (mapcat (fn [j y]
                [(keyword (str "y" j)) (mx/scalar y)])
              (range M) ys))))

;; ---------------------------------------------------------------------------
;; Model factory — vectorized (gaussian-vec + matmul)
;; ---------------------------------------------------------------------------

(defn make-design-matrix-mx [D]
  (mx/reshape (mx/array (apply concat (make-design-matrix D))) [M D]))

(defn make-large-model-fast [D]
  (let [mu-zeros (mx/zeros [D])]
    (dyn/auto-key
      (gen [X-mx]
        (let [weights (trace :weights (dist/gaussian-vec mu-zeros 1.0))
              y-pred  (mx/matmul weights (mx/transpose X-mx))]
          (trace :ys (dist/gaussian-vec y-pred noise-std)))))))

(defn make-observations-cm-fast [X-rows true-weights]
  (let [ys (make-observations-data X-rows true-weights)]
    (cm/choicemap :ys (mx/array ys))))

;; ---------------------------------------------------------------------------
;; Timing (matching GenJAX methodology)
;; ---------------------------------------------------------------------------

(defn perf-now []
  (/ (js/performance.now) 1000.0))

(defn timing [f repeats inner-repeats]
  (let [times (loop [i 0 acc (transient [])]
                (if (>= i repeats)
                  (persistent! acc)
                  (let [inner-min
                        (loop [j 0 best js/Infinity]
                          (if (>= j inner-repeats)
                            best
                            (let [start (perf-now)
                                  _ (f)
                                  elapsed (- (perf-now) start)]
                              (recur (inc j) (min best elapsed)))))]
                    (recur (inc i) (conj! acc inner-min)))))
        n (count times)
        mean (/ (reduce + times) n)
        variance (/ (reduce + (map #(let [d (- % mean)] (* d d)) times)) n)
        std (js/Math.sqrt variance)]
    {:times times :mean mean :std std}))

(defn benchmark-with-warmup [f opts]
  (let [{:keys [warmup-runs repeats inner-repeats]
         :or {warmup-runs 5 repeats 20 inner-repeats 20}} opts]
    (dotimes [_ warmup-runs] (f))
    (timing f repeats inner-repeats)))

;; ---------------------------------------------------------------------------
;; JSON output
;; ---------------------------------------------------------------------------

(def fs (js/require "fs"))
(def path-mod (js/require "path"))

(def output-dir
  (.resolve path-mod (js/process.cwd)
            "genjax/examples/perfbench/data_cpu/curvefit_large/genmlx"))

(defn ensure-dir [dir]
  (when-not (.existsSync fs dir)
    (.mkdirSync fs dir #js {:recursive true})))

(defn write-json [filename data]
  (ensure-dir output-dir)
  (let [filepath (str output-dir "/" filename)]
    (.writeFileSync fs filepath (js/JSON.stringify (clj->js data) nil 2))
    (println (str "  Wrote: " filepath))))

;; ---------------------------------------------------------------------------
;; Benchmark runners
;; ---------------------------------------------------------------------------

(defn run-is-benchmark [D n-particles opts]
  (println (str "\n=== IS (per-site) D=" D " n=" n-particles " ==="))
  (let [X-rows (make-design-matrix D)
        true-w (make-true-weights D)
        obs (make-observations-cm X-rows true-w)
        model (make-large-model D)
        task-fn (fn []
                  (mx/tidy
                    #(let [r (is/vectorized-importance-sampling
                               {:samples n-particles}
                               model [X-rows] obs)]
                       (mx/materialize! (:log-ml-estimate r)
                                        (:weight (:vtrace r)))
                       nil)))
        {:keys [times mean std]} (benchmark-with-warmup task-fn opts)
        result {:framework "genmlx"
                :method "importance_sampling"
                :variant "per_site"
                :n_particles n-particles
                :n_features D
                :n_points M
                :times (vec times)
                :mean_time mean
                :std_time std}]
    (println (str "  Mean: " (.toFixed (* mean 1000) 3) " ms  "
                  "Std: " (.toFixed (* std 1000) 3) " ms"))
    (write-json (str "is_D" D "_n" n-particles ".json") result)
    result))

(defn run-is-benchmark-fast [D n-particles opts]
  (println (str "\n=== IS (gaussian-vec) D=" D " n=" n-particles " ==="))
  (let [X-rows (make-design-matrix D)
        X-mx (make-design-matrix-mx D)
        true-w (make-true-weights D)
        obs (make-observations-cm-fast X-rows true-w)
        model (make-large-model-fast D)
        task-fn (fn []
                  (mx/tidy
                    #(let [r (is/vectorized-importance-sampling
                               {:samples n-particles}
                               model [X-mx] obs)]
                       (mx/materialize! (:log-ml-estimate r)
                                        (:weight (:vtrace r)))
                       nil)))
        {:keys [times mean std]} (benchmark-with-warmup task-fn opts)
        result {:framework "genmlx"
                :method "importance_sampling"
                :variant "gaussian_vec"
                :n_particles n-particles
                :n_features D
                :n_points M
                :times (vec times)
                :mean_time mean
                :std_time std}]
    (println (str "  Mean: " (.toFixed (* mean 1000) 3) " ms  "
                  "Std: " (.toFixed (* std 1000) 3) " ms"))
    (write-json (str "is_fast_D" D "_n" n-particles ".json") result)
    result))

;; ---------------------------------------------------------------------------
;; Main
;; ---------------------------------------------------------------------------

(defn main []
  (println "GenMLX Large Model Perfbench")
  (println (str "Design: X[j,i] = sin(π(i+1)·xj), M=" M " points, σ=" noise-std))
  (println (str "Output: " output-dir))

  (let [n-particles 10000
        D-values [10 25 50 100 200]
        bench-opts {:warmup-runs 5 :repeats 20 :inner-repeats 20}]

    ;; Verify first model (both variants)
    (let [D (first D-values)
          X-rows (make-design-matrix D)
          X-mx (make-design-matrix-mx D)
          true-w (make-true-weights D)
          ys (make-observations-data X-rows true-w)]
      (println (str "\nVerification (D=" D "): first 5 ys = "
                    (mapv #(.toFixed % 4) (take 5 ys))))

      ;; Quick sanity: fast model simulate
      (let [model (make-large-model-fast D)
            tr (p/simulate model [X-mx])]
        (println (str "  Fast model simulate OK: weights shape="
                      (mx/shape (cm/get-value (cm/get-submap (:choices tr) :weights)))
                      " ys shape="
                      (mx/shape (cm/get-value (cm/get-submap (:choices tr) :ys))))))

      ;; Quick sanity: fast model vectorized IS
      (let [model (make-large-model-fast D)
            obs (make-observations-cm-fast X-rows true-w)
            r (is/vectorized-importance-sampling {:samples 100} model [X-mx] obs)]
        (println (str "  Fast model IS OK: log-ML="
                      (.toFixed (mx/realize (:log-ml-estimate r)) 2)))))

    ;; Run benchmarks — fast variant
    (println "\n--- Gaussian-vec (fast) benchmarks ---")
    (let [fast-results
          (doall (for [D D-values]
                   (run-is-benchmark-fast D n-particles bench-opts)))]

      ;; Summary table
      (println "\n=== Summary ===")
      (println "D\tgaussian-vec (ms)")
      (doseq [r fast-results]
        (println (str (:n_features r) "\t"
                      (.toFixed (* (:mean_time r) 1000) 1))))))

  (println "\n=== Done ===")
  (.exit js/process 0))

(main)
