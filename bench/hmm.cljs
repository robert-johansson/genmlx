(ns bench.hmm
  "HMM inference comparison: IS fails on sequential models, SMC converges.

   2-state Gaussian-emission HMM, T=50 timesteps.
   Transition: A = [[0.9, 0.1], [0.1, 0.9]] (sticky)
   Emission: y_t | z_t=k ~ N(mu_k, 1.0), mu = [-2, 2]

   Algorithms:
   1. Vectorized IS (N=1000) — expected to FAIL (low ESS)
   2. Sequential IS (N=100) — loop over p/generate
   3. Batched SMC (N=100) — may crash due to Metal pipeline leak (P0-4)
   4. Batched SMC (N=250) — same, more particles

   Ground truth: forward algorithm (exact log-ML).

   Usage: bun run --bun nbb bench/hmm.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.importance :as is]
            [genmlx.inference.smc :as smc])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Infrastructure
;; ---------------------------------------------------------------------------

(defn perf-now [] (js/performance.now))

(def fs (js/require "fs"))
(def path-mod (js/require "path"))
(def results-dir
  (or (aget (.-env js/process) "GENMLX_RESULTS_DIR")
      (.resolve path-mod (js/process.cwd) "results/hmm")))

(defn ensure-dir [dir]
  (when-not (.existsSync fs dir)
    (.mkdirSync fs dir #js {:recursive true})))

(defn write-json [filename data]
  (ensure-dir results-dir)
  (let [filepath (str results-dir "/" filename)]
    (.writeFileSync fs filepath (js/JSON.stringify (clj->js data) nil 2))
    (println (str "  wrote: " filepath))))

(defn benchmark [label f & {:keys [warmup-n outer-n inner-n]
                             :or {warmup-n 1 outer-n 3 inner-n 1}}]
  (println (str "\n  [" label "] warming up..."))
  (dotimes [_ warmup-n] (f) (mx/materialize!))
  (mx/clear-cache!)
  (let [outer-times
        (vec (for [_ (range outer-n)]
               (let [inner-times
                     (vec (for [_ (range inner-n)]
                            (let [t0 (perf-now)]
                              (f)
                              (mx/materialize!)
                              (- (perf-now) t0))))]
                 (mx/clear-cache!)
                 (apply min inner-times))))
        mean-ms (/ (reduce + outer-times) (count outer-times))
        std-ms  (js/Math.sqrt (/ (reduce + (map #(* (- % mean-ms) (- % mean-ms))
                                                 outer-times))
                                  (max 1 (dec (count outer-times)))))]
    (println (str "  [" label "] " (.toFixed mean-ms 3) " +/- " (.toFixed std-ms 3) " ms"))
    {:label label :mean-ms mean-ms :std-ms std-ms
     :min-ms (apply min outer-times) :raw outer-times}))

(defn safe-benchmark
  "Like benchmark but catches errors and returns a skip marker."
  [label f & opts]
  (try
    (apply benchmark label f opts)
    (catch :default e
      (println (str "  [" label "] SKIPPED — " (.-message e)))
      {:label label :mean-ms nil :std-ms nil :min-ms nil :raw []
       :skipped true :error (.-message e)})))


;; ---------------------------------------------------------------------------
;; HMM parameters
;; ---------------------------------------------------------------------------

(def K 2)        ;; number of hidden states
(def T 50)       ;; number of timesteps

;; Transition matrix A[i,j] = P(z_t=j | z_{t-1}=i)
(def transition-probs [[0.9 0.1] [0.1 0.9]])

;; Log-transition matrix
(def transition-log-probs
  (mapv (fn [row] (mapv #(js/Math.log %) row)) transition-probs))
(def transition-logits
  (mx/reshape (mx/array (clj->js (apply concat transition-log-probs))) [K K]))

;; Initial distribution
(def init-probs [0.5 0.5])
(def init-log-probs (mapv #(js/Math.log %) init-probs))
(def init-logits (mx/array (clj->js init-log-probs)))

;; Emission parameters
(def emission-means-vec [-2.0 2.0])
(def emission-means (mx/array (clj->js emission-means-vec)))
(def sigma-obs 1.0)
(def sigma-obs-arr (mx/scalar sigma-obs))


;; ---------------------------------------------------------------------------
;; Data (generated with seed 42, hardcoded to avoid lazy-graph crash in loader)
;; ---------------------------------------------------------------------------

(def zs-true [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 1 1 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0])

(def ys-data [3.070 -2.705 -1.695 -3.586 -3.487 -2.084 -1.983 -3.072
              -0.495 -2.301 0.161 -1.769 -1.536 -0.805 0.851 2.223
              1.253 2.740 2.009 -0.089 1.709 2.528 -1.942 -2.348
              -2.661 -2.928 -2.599 -1.194 -3.515 -1.982 -2.571 -2.338
              -2.913 -1.772 0.147 -1.991 -1.350 2.660 1.810 2.121
              2.972 5.104 1.414 3.214 2.140 1.725 2.608 1.461
              2.415 -1.998])

;; ---------------------------------------------------------------------------
;; Forward algorithm — exact log-ML ground truth (pure JS)
;; ---------------------------------------------------------------------------

(defn gaussian-log-pdf [x mu sigma]
  (let [z (/ (- x mu) sigma)]
    (- (- (* 0.5 z z))
       (+ (* 0.5 (js/Math.log (* 2.0 js/Math.PI)))
          (js/Math.log sigma)))))

(defn logsumexp-vec [xs]
  (let [m (apply max xs)]
    (+ m (js/Math.log (reduce + (map #(js/Math.exp (- % m)) xs))))))

(defn forward-algorithm
  "Exact forward algorithm. Returns {:log-ml log P(y_{1:T})}."
  [ys]
  (let [log-alpha-0
        (mapv (fn [k]
                (+ (js/Math.log (nth init-probs k))
                   (gaussian-log-pdf (nth ys 0) (nth emission-means-vec k) sigma-obs)))
              (range K))
        result
        (loop [t 1, log-alpha log-alpha-0]
          (if (>= t (count ys))
            log-alpha
            (let [new-log-alpha
                  (mapv (fn [j]
                          (let [terms (mapv (fn [i]
                                              (+ (nth log-alpha i)
                                                 (js/Math.log (nth (nth transition-probs i) j))))
                                            (range K))]
                            (+ (logsumexp-vec terms)
                               (gaussian-log-pdf (nth ys t) (nth emission-means-vec j) sigma-obs))))
                        (range K))]
              (recur (inc t) new-log-alpha))))]
    {:log-ml (logsumexp-vec result)}))

(def forward-result (forward-algorithm ys-data))
(def exact-log-ml (:log-ml forward-result))


;; ---------------------------------------------------------------------------
;; Models
;; ---------------------------------------------------------------------------

;; Vectorized kernel for batched SMC: no mx/eval!/mx/item, shapes flow through
(def hmm-kernel-vec
  (gen [t z-prev]
    (let [logits (if (nil? z-prev)
                   init-logits
                   (mx/take-idx transition-logits z-prev 0))
          z (trace :z (dist/categorical logits))
          mu (mx/take-idx emission-means z)
          _ (trace :y (dist/gaussian mu (mx/scalar sigma-obs)))]
      z)))

;; Flat HMM model: unrolled loop, flat address keys (:z0 :y0 :z1 :y1 ...)
;; Used by both vectorized IS and sequential IS
(def hmm-vec-model
  (gen [T-steps]
    (loop [t 0, z nil]
      (if (>= t T-steps)
        z
        (let [logits (if (nil? z)
                       init-logits
                       (mx/take-idx transition-logits z 0))
              z-new (trace (keyword (str "z" t)) (dist/categorical logits))
              mu (mx/take-idx emission-means z-new)
              _ (trace (keyword (str "y" t)) (dist/gaussian mu sigma-obs-arr))]
          (recur (inc t) z-new))))))


;; ---------------------------------------------------------------------------
;; Observations
;; ---------------------------------------------------------------------------

;; Flat observations for vectorized IS (flat keys :y0, :y1, ...)
(def vec-observations
  (reduce (fn [cm t]
            (cm/set-choice cm [(keyword (str "y" t))] (mx/scalar (nth ys-data t))))
          cm/EMPTY (range T)))

;; Per-step observations for SMC (kernel-level, no integer prefix)
(def smc-obs-seq
  (mapv (fn [t] (cm/choicemap :y (mx/scalar (nth ys-data t)))) (range T)))

;; ---------------------------------------------------------------------------
;; Print header
;; ---------------------------------------------------------------------------

(println "\n=== HMM Inference Comparison: IS vs SMC ===")
(println (str "HMM: K=" K " states, T=" T " timesteps"))
(println (str "Transition: A=[[0.9,0.1],[0.1,0.9]] (sticky)"))
(println (str "Emission: mu=[-2,2], sigma=" sigma-obs))
(println (str "Exact log P(y_{1:T}) = " (.toFixed exact-log-ml 4)))

;; ---------------------------------------------------------------------------
;; Experiment helpers
;; ---------------------------------------------------------------------------

(defn compute-ess-from-js-weights
  "ESS from a vector of JS-number log-weights."
  [log-ws]
  (let [max-w (apply max log-ws)
        ws (map #(js/Math.exp (- % max-w)) log-ws)
        s (reduce + ws)
        nw (map #(/ % s) ws)]
    (/ 1.0 (reduce + (map #(* % %) nw)))))

(defn compute-log-ml-from-js-weights
  "Log-ML from a vector of JS-number log-weights."
  [log-ws]
  (let [n (count log-ws)
        max-w (apply max log-ws)]
    (+ max-w (- (js/Math.log (reduce + (map #(js/Math.exp (- % max-w)) log-ws)))
                (js/Math.log n)))))

(defn run-summarize
  "Run experiment-fn n-runs times, summarize log-ml, error, ESS, and time."
  [experiment-fn n-runs seeds]
  (let [runs (vec (for [seed seeds]
                    (do (experiment-fn seed))))
        log-mls (mapv :log-ml runs)
        errors (mapv #(js/Math.abs (- (:log-ml %) exact-log-ml)) runs)
        ess-vals (mapv :ess runs)
        times (mapv :time-ms runs)
        mean-fn (fn [xs] (/ (reduce + xs) (count xs)))
        std-fn (fn [xs]
                 (let [m (mean-fn xs)]
                   (js/Math.sqrt (/ (reduce + (map #(let [d (- % m)] (* d d)) xs))
                                    (max 1 (count xs))))))]
    {:log-ml-mean (mean-fn log-mls) :log-ml-std (std-fn log-mls)
     :error-mean (mean-fn errors) :error-std (std-fn errors)
     :ess-mean (mean-fn ess-vals) :ess-std (std-fn ess-vals)
     :time-mean (mean-fn times) :time-std (std-fn times)
     :raw-log-mls (vec log-mls) :raw-errors (vec errors) :raw-ess (vec ess-vals)}))

;; ---------------------------------------------------------------------------
;; 1. Vectorized IS (N=1000) — expected to FAIL (low ESS)
;; ---------------------------------------------------------------------------

(println "\n-- 1. Vectorized IS (N=1000) --")

(defn run-vec-is [seed]
  (let [start (perf-now)
        result (is/vectorized-importance-sampling
                 {:samples 1000 :key (rng/fresh-key seed)}
                 hmm-vec-model [T] vec-observations)
        log-ml (mx/realize (:log-ml-estimate result))
        ;; ESS from the vtrace weights
        w (:weight (:vtrace result))
        _ (mx/materialize! w)
        log-ws-arr w
        ;; Compute ESS: exp(2*logsumexp(lw) - logsumexp(2*lw))
        lse1 (mx/item (mx/logsumexp log-ws-arr))
        lse2 (mx/item (mx/logsumexp (mx/multiply log-ws-arr (mx/scalar 2.0))))
        ess (js/Math.exp (- (* 2.0 lse1) lse2))
        elapsed (- (perf-now) start)]
    (mx/clear-cache!)
    {:log-ml log-ml :ess ess :time-ms elapsed}))

(def vis-1000-summary
  (run-summarize run-vec-is 3 [100 101 102]))

(println (str "  log-ML: " (.toFixed (:log-ml-mean vis-1000-summary) 2)
              " +/- " (.toFixed (:log-ml-std vis-1000-summary) 2)))
(println (str "  |error|: " (.toFixed (:error-mean vis-1000-summary) 2)
              " +/- " (.toFixed (:error-std vis-1000-summary) 2)))
(println (str "  ESS: " (.toFixed (:ess-mean vis-1000-summary) 1)))

(mx/clear-cache!)

;; ---------------------------------------------------------------------------
;; 2. Sequential IS (N=100) — loop calling p/generate
;; ---------------------------------------------------------------------------

(println "\n-- 2. Sequential IS (N=100) --")

(defn run-seq-is [seed]
  (let [start (perf-now)
        rk (rng/ensure-key (rng/fresh-key seed))
        n-particles 100
        log-weights
        (loop [i 0, rk rk, acc (transient [])]
          (if (>= i n-particles)
            (persistent! acc)
            (let [[ki next-rk] (rng/split rk)
                  r (p/generate (dyn/with-key hmm-vec-model ki)
                                [T] vec-observations)
                  w (mx/realize (:weight r))]
              (when (zero? (mod i 25))
                (mx/sweep-dead-arrays!)
                (mx/clear-cache!))
              (recur (inc i) next-rk (conj! acc w)))))
        elapsed (- (perf-now) start)
        log-ml (compute-log-ml-from-js-weights log-weights)
        ess (compute-ess-from-js-weights log-weights)]
    (mx/clear-cache!)
    {:log-ml log-ml :ess ess :time-ms elapsed}))

(def seq-is-100-summary
  (run-summarize run-seq-is 3 [200 201 202]))

(println (str "  log-ML: " (.toFixed (:log-ml-mean seq-is-100-summary) 2)
              " +/- " (.toFixed (:log-ml-std seq-is-100-summary) 2)))
(println (str "  |error|: " (.toFixed (:error-mean seq-is-100-summary) 2)
              " +/- " (.toFixed (:error-std seq-is-100-summary) 2)))
(println (str "  ESS: " (.toFixed (:ess-mean seq-is-100-summary) 1)))

(mx/clear-cache!)

;; ---------------------------------------------------------------------------
;; 3. Batched SMC (N=100) — may crash due to Metal pipeline leak
;; ---------------------------------------------------------------------------

(println "\n-- 3. Batched SMC (N=100) --")

(defn run-batched-smc [n-particles seed]
  (let [start (perf-now)
        result (smc/batched-smc-unfold {:particles n-particles
                                         :key (rng/fresh-key seed)}
                                        (dyn/auto-key hmm-kernel-vec) nil smc-obs-seq)
        elapsed (- (perf-now) start)]
    (mx/clear-cache!)
    {:log-ml (mx/item (:log-ml result))
     :ess (:final-ess result)
     :time-ms elapsed}))

(def batched-smc-100-summary
  (try
    (let [s (run-summarize #(run-batched-smc 100 %) 3 [300 301 302])]
      (println (str "  log-ML: " (.toFixed (:log-ml-mean s) 2)
                    " +/- " (.toFixed (:log-ml-std s) 2)))
      (println (str "  |error|: " (.toFixed (:error-mean s) 2)
                    " +/- " (.toFixed (:error-std s) 2)))
      (println (str "  ESS: " (.toFixed (:ess-mean s) 1)))
      s)
    (catch :default e
      (println (str "  SKIPPED — " (.-message e)))
      {:skipped true :error (.-message e)})))

(mx/clear-cache!)

;; ---------------------------------------------------------------------------
;; 4. Batched SMC (N=200) — may crash due to Metal pipeline leak
;; ---------------------------------------------------------------------------

(println "\n-- 4. Batched SMC (N=200) --")

(def batched-smc-200-summary
  (try
    (let [s (run-summarize #(run-batched-smc 200 %) 3 [400 401 402])]
      (println (str "  log-ML: " (.toFixed (:log-ml-mean s) 2)
                    " +/- " (.toFixed (:log-ml-std s) 2)))
      (println (str "  |error|: " (.toFixed (:error-mean s) 2)
                    " +/- " (.toFixed (:error-std s) 2)))
      (println (str "  ESS: " (.toFixed (:ess-mean s) 1)))
      s)
    (catch :default e
      (println (str "  SKIPPED — " (.-message e)))
      {:skipped true :error (.-message e)})))

(mx/clear-cache!)

;; ---------------------------------------------------------------------------
;; Summary table
;; ---------------------------------------------------------------------------

(println "\n\n========================================")
(println "   HMM Inference: IS vs SMC")
(println "========================================")
(println (str "\nExact log P(y_{1:T}) = " (.toFixed exact-log-ml 4)))

(defn fmt-summary-row [name summary particles]
  (if (:skipped summary)
    (str "| " name " | " particles " | SKIPPED | — | — | — |")
    (str "| " name " | " particles
         " | " (.toFixed (:log-ml-mean summary) 2)
         " +/- " (.toFixed (:log-ml-std summary) 2)
         " | " (.toFixed (:error-mean summary) 2)
         " | " (.toFixed (:ess-mean summary) 1)
         " | " (.toFixed (:time-mean summary) 0) " ms |")))

(println "\n| Method | N | log-ML | |Error| | ESS | Time |")
(println "|--------|---|--------|---------|-----|------|")
(println (fmt-summary-row "Vectorized IS" vis-1000-summary 1000))
(println (fmt-summary-row "Sequential IS" seq-is-100-summary 100))
(println (fmt-summary-row "Batched SMC" batched-smc-100-summary 100))
(println (fmt-summary-row "Batched SMC" batched-smc-200-summary 200))

;; ---------------------------------------------------------------------------
;; Write results JSON
;; ---------------------------------------------------------------------------

(println "\n-- Writing results --")

(defn summary->json [name method particles summary]
  (if (:skipped summary)
    {:algorithm name :method method :particles particles
     :skipped true :error (:error summary)}
    {:algorithm name :method method :particles particles
     :log_ml_mean (:log-ml-mean summary)
     :log_ml_std (:log-ml-std summary)
     :error_mean (:error-mean summary)
     :error_std (:error-std summary)
     :ess_mean (:ess-mean summary)
     :ess_std (:ess-std summary)
     :time_ms_mean (:time-mean summary)
     :time_ms_std (:time-std summary)
     :raw_log_mls (:raw-log-mls summary)
     :raw_errors (:raw-errors summary)
     :raw_ess (:raw-ess summary)}))

(write-json "data.json"
  {:experiment "hmm_is_vs_smc"
   :timestamp (.toISOString (js/Date.))
   :model {:K K :T T
           :transition_probs transition-probs
           :init_probs init-probs
           :emission_means emission-means-vec
           :sigma_obs sigma-obs}
   :exact_log_ml exact-log-ml
   :data {:zs_true zs-true :ys ys-data}
   :algorithms
   [(summary->json "VIS_1000" "vectorized-is" 1000 vis-1000-summary)
    (summary->json "Seq_IS_100" "sequential-is" 100 seq-is-100-summary)
    (summary->json "Batched_SMC_100" "batched-smc" 100 batched-smc-100-summary)
    (summary->json "Batched_SMC_200" "batched-smc" 200 batched-smc-200-summary)]})

(println "\nDone.")
