(ns genmlx.paper-bench-changepoint
  "Paper Experiment 8: Changepoint Detection — Expressiveness Showcase.

   Data-dependent random structure: number of changepoints varies per execution.
   Directly tests expressiveness: `if` on sampled Bernoulli — natural in GenMLX.

   Model:
     T=100 timesteps, p_change=0.05 (~5 segments expected)
     At each t:
       cp_t ~ Bernoulli(p_change)  [t > 0]
       If cp_t = 1: mu_new ~ N(0, 5) (new segment mean)
       y_t ~ N(mu_current, 1.0)

   Ground truth: DP forward algorithm (run-length formulation).
   Inference: SMC via smc-unfold, IS via p/generate.

   Usage: bun run --bun nbb test/genmlx/paper_bench_changepoint.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.smc :as smc]
            [genmlx.combinators :as comb])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Timing & I/O
;; ---------------------------------------------------------------------------

(defn perf-now [] (js/performance.now))

(def fs (js/require "fs"))
(def path-mod (js/require "path"))
(def results-dir
  (.resolve path-mod (js/process.cwd) "results/exp8_changepoint"))

(defn ensure-dir [dir]
  (when-not (.existsSync fs dir)
    (.mkdirSync fs dir #js {:recursive true})))

(defn write-json [filename data]
  (ensure-dir results-dir)
  (let [filepath (str results-dir "/" filename)]
    (.writeFileSync fs filepath (js/JSON.stringify (clj->js data) nil 2))
    (println (str "  Wrote: " filepath))))

;; ---------------------------------------------------------------------------
;; Statistics helpers
;; ---------------------------------------------------------------------------

(defn mean-fn [xs] (/ (reduce + xs) (count xs)))
(defn std-fn [xs]
  (let [m (mean-fn xs) n (count xs)]
    (if (<= n 1) 0.0
      (js/Math.sqrt (/ (reduce + (map #(let [d (- % m)] (* d d)) xs)) n)))))

(defn logsumexp-vec [xs]
  (let [m (apply max xs)]
    (+ m (js/Math.log (reduce + (map #(js/Math.exp (- % m)) xs))))))

(defn gaussian-log-pdf [x mu sigma]
  (let [z (/ (- x mu) sigma)]
    (- (- (* 0.5 z z))
       (+ (* 0.5 (js/Math.log (* 2.0 js/Math.PI))) (js/Math.log sigma)))))

;; ---------------------------------------------------------------------------
;; Model parameters & data generation
;; ---------------------------------------------------------------------------

(def T-default 100)
(def p-change 0.05)
(def mu-prior-std 5.0)
(def sigma-obs 1.0)

(defn generate-changepoint-data [T seed]
  (let [keys (rng/split-n (rng/fresh-key seed) (* T 3))]
    (loop [t 0, mu 0.0, ys [], cps [], mus [0.0]]
      (if (>= t T)
        {:ys ys :changepoints cps :segment-means mus :T T}
        (let [is-cp (and (> t 0) (< (mx/item (rng/uniform (nth keys (* t 3)) [])) p-change))
              new-mu (if is-cp
                       (* mu-prior-std (mx/item (rng/normal (nth keys (+ (* t 3) 1)) [])))
                       mu)
              y (+ new-mu (* sigma-obs (mx/item (rng/normal (nth keys (+ (* t 3) 2)) []))))]
          (recur (inc t) new-mu (conj ys y)
                 (if is-cp (conj cps t) cps)
                 (if is-cp (conj mus new-mu) mus)))))))

(def data-42 (generate-changepoint-data T-default 42))
(def ys-data (:ys data-42))
(def true-cps (:changepoints data-42))
(def true-means (:segment-means data-42))

(println "\n=== Paper Experiment 8: Changepoint Detection ===")
(println (str "T=" T-default ", p_change=" p-change))
(println (str "True changepoints: " true-cps " (" (count true-cps) " total)"))
(println (str "Segment means: " (mapv #(.toFixed % 2) true-means)))

;; ---------------------------------------------------------------------------
;; Ground truth: Run-length DP forward algorithm
;; ---------------------------------------------------------------------------

(defn dp-forward-algorithm [ys]
  (let [T (count ys)
        log-p (js/Math.log p-change)
        log-1-p (js/Math.log (- 1.0 p-change))
        sigma-prior-sq (* mu-prior-std mu-prior-std)
        sigma-obs-sq (* sigma-obs sigma-obs)
        sigma-marginal (js/Math.sqrt (+ sigma-obs-sq sigma-prior-sq))
        init-alpha [(gaussian-log-pdf (nth ys 0) 0.0 sigma-marginal)]
        init-stats [{:sum (nth ys 0) :n 1}]
        result
        (loop [t 1, alpha init-alpha, stats init-stats]
          (if (>= t T)
            {:alpha alpha}
            (let [y-t (nth ys t)
                  n-runs (count alpha)
                  cp-term (+ (logsumexp-vec alpha) log-p
                             (gaussian-log-pdf y-t 0.0 sigma-marginal))
                  continue-terms
                  (mapv (fn [r]
                          (let [{:keys [sum n]} (nth stats r)
                                sigma-n-sq (/ 1.0 (+ (/ n sigma-obs-sq) (/ 1.0 sigma-prior-sq)))
                                mu-n (* sigma-n-sq (/ sum sigma-obs-sq))
                                pred-sigma (js/Math.sqrt (+ sigma-obs-sq sigma-n-sq))]
                            (+ (nth alpha r) log-1-p (gaussian-log-pdf y-t mu-n pred-sigma))))
                        (range n-runs))
                  new-alpha (vec (cons cp-term continue-terms))
                  new-stats (vec (cons {:sum y-t :n 1}
                                       (mapv (fn [r]
                                               (let [{:keys [sum n]} (nth stats r)]
                                                 {:sum (+ sum y-t) :n (inc n)}))
                                             (range n-runs))))]
              (recur (inc t) new-alpha new-stats))))]
    {:log-ml (logsumexp-vec (:alpha result))}))

(def dp-result (dp-forward-algorithm ys-data))
(def exact-log-ml (:log-ml dp-result))
(println (str "\nExact log P(y) = " (.toFixed exact-log-ml 4)))

;; ---------------------------------------------------------------------------
;; Changepoint kernel for SMC (state = JS number = current mu)
;; ---------------------------------------------------------------------------

(def cp-kernel
  (gen [t state]
    (let [prev-mu (if (nil? state) 0.0 state)
          cp (if (zero? t)
               (mx/scalar 0.0)
               (trace :cp (dist/bernoulli (mx/scalar p-change))))
          _ (mx/eval! cp)
          is-cp (> (mx/item cp) 0.5)
          new-mu (if is-cp
                   (let [m (trace :mu_new (dist/gaussian 0 mu-prior-std))]
                     (mx/eval! m) (mx/item m))
                   prev-mu)
          _ (trace :y (dist/gaussian (mx/scalar new-mu) sigma-obs))]
      new-mu)))

(def smc-obs-seq
  (mapv (fn [t] (cm/choicemap :y (mx/scalar (nth ys-data t)))) (range T-default)))

;; ---------------------------------------------------------------------------
;; IS flat model
;; ---------------------------------------------------------------------------

(def cp-flat-model
  (dyn/auto-key
    (gen [T-steps]
      (loop [t 0, mu 0.0]
        (if (>= t T-steps)
          mu
          (let [cp (if (zero? t)
                     (mx/scalar 0.0)
                     (trace (keyword (str "cp" t)) (dist/bernoulli (mx/scalar p-change))))
                _ (mx/eval! cp)
                is-cp (> (mx/item cp) 0.5)
                new-mu (if is-cp
                         (let [m (trace (keyword (str "mu" t)) (dist/gaussian 0 mu-prior-std))]
                           (mx/eval! m) (mx/item m))
                         mu)
                _ (trace (keyword (str "y" t)) (dist/gaussian (mx/scalar new-mu) sigma-obs))]
            (recur (inc t) new-mu)))))))

(def flat-observations
  (reduce (fn [cm t]
            (cm/set-choice cm [(keyword (str "y" t))] (mx/scalar (nth ys-data t))))
          cm/EMPTY (range T-default)))

;; ---------------------------------------------------------------------------
;; Experiment runners
;; ---------------------------------------------------------------------------

(defn run-smc [n-particles seed]
  (let [start (perf-now)
        result (smc/smc-unfold {:particles n-particles :key (rng/fresh-key seed)}
                               (dyn/auto-key cp-kernel) nil smc-obs-seq)
        elapsed (- (perf-now) start)]
    (mx/clear-cache!) (mx/force-gc!) (mx/clear-cache!)
    {:log-ml (mx/item (:log-ml result)) :ess (:final-ess result) :time-ms elapsed}))

(defn run-is [n-particles seed]
  (let [start (perf-now)
        log-weights
        (loop [i 0, rk (rng/ensure-key (rng/fresh-key seed)), acc (transient [])]
          (if (>= i n-particles)
            (persistent! acc)
            (let [[ki next-rk] (rng/split rk)
                  w (mx/tidy
                      #(let [r (p/generate (dyn/with-key cp-flat-model ki)
                                           [T-default] flat-observations)]
                         (mx/realize (:weight r))))]
              (when (zero? (mod i 50)) (mx/clear-cache!) (mx/force-gc!))
              (recur (inc i) next-rk (conj! acc w)))))
        elapsed (- (perf-now) start)
        lw-arr (mx/array (clj->js log-weights))
        log-ml (mx/item (mx/subtract (mx/logsumexp lw-arr) (mx/scalar (js/Math.log n-particles))))
        lp (mx/subtract lw-arr (mx/logsumexp lw-arr))
        probs (mx/exp lp)
        _ (mx/materialize! probs)
        ess (mx/item (mx/divide (mx/square (mx/sum probs)) (mx/sum (mx/square probs))))]
    (mx/clear-cache!) (mx/force-gc!) (mx/clear-cache!)
    {:log-ml log-ml :ess ess :time-ms elapsed}))

(defn summarize [runs]
  (let [log-mls (mapv :log-ml runs)
        errors (mapv #(js/Math.abs (- (:log-ml %) exact-log-ml)) runs)
        times (mapv :time-ms runs)
        ess-vals (mapv :ess runs)]
    {:log-ml {:mean (mean-fn log-mls) :std (std-fn log-mls)}
     :error {:mean (mean-fn errors) :std (std-fn errors)}
     :ess {:mean (mean-fn ess-vals) :std (std-fn ess-vals)}
     :time-ms {:mean (mean-fn times) :std (std-fn times)}
     :raw-log-mls log-mls :raw-errors errors}))

;; Incremental results
(def results-acc (atom []))

(defn run-experiment [name method n-particles n-runs runner-fn]
  (println (str "\n-- " name " (" n-runs " runs) --"))
  (let [runs (vec (for [i (range n-runs)]
                    (do (println (str "  run " i "..."))
                        (runner-fn n-particles (+ 200 (* n-particles i) i)))))
        summary (summarize runs)]
    (println (str "  log-ML: " (.toFixed (get-in summary [:log-ml :mean]) 2)
                  " +/- " (.toFixed (get-in summary [:log-ml :std]) 2)))
    (println (str "  |error|: " (.toFixed (get-in summary [:error :mean]) 2)
                  " +/- " (.toFixed (get-in summary [:error :std]) 2)))
    (println (str "  ESS: " (.toFixed (get-in summary [:ess :mean]) 1)
                  "  time: " (.toFixed (get-in summary [:time-ms :mean]) 0) "ms"))

    (swap! results-acc conj
           {:algorithm name :method method :particles n-particles
            :log_ml (get-in summary [:log-ml :mean])
            :log_ml_std (get-in summary [:log-ml :std])
            :error (get-in summary [:error :mean])
            :error_std (get-in summary [:error :std])
            :ess (get-in summary [:ess :mean])
            :ess_std (get-in summary [:ess :std])
            :time_ms (get-in summary [:time-ms :mean])
            :time_ms_std (get-in summary [:time-ms :std])
            :raw_log_mls (:raw-log-mls summary)
            :raw_errors (:raw-errors summary)})

    ;; Save incrementally
    (write-json "changepoint_results.json"
                {:model {:T T-default :p_change p-change
                         :mu_prior_std mu-prior-std :sigma_obs sigma-obs}
                 :data {:ys ys-data :true_changepoints true-cps
                        :true_segment_means true-means
                        :n_true_changepoints (count true-cps)}
                 :exact_log_ml exact-log-ml
                 :algorithms @results-acc})
    summary))

;; ---------------------------------------------------------------------------
;; Run experiments
;; ---------------------------------------------------------------------------

(def smc-100 (run-experiment "SMC_100" "smc" 100 3 run-smc))
(def smc-250 (run-experiment "SMC_250" "smc" 250 3 run-smc))
(def smc-500 (run-experiment "SMC_500" "smc" 500 3 run-smc))
(def is-1000 (run-experiment "IS_1000" "is" 1000 3 run-is))

;; ---------------------------------------------------------------------------
;; Write SUMMARY.md
;; ---------------------------------------------------------------------------

(let [summary
      (str "# Experiment 8: Changepoint Detection\n\n"
           "**Date:** 2026-03-05\n"
           "**Model:** T=" T-default ", p_change=" p-change
           ", mu ~ N(0," mu-prior-std "), sigma=" sigma-obs "\n"
           "**True changepoints:** " true-cps " (" (count true-cps) " total)\n"
           "**Exact log P(y):** " (.toFixed exact-log-ml 4) "\n\n"
           "## Key Expressiveness Feature\n\n"
           "```clojure\n"
           "(let [cp (trace :cp (dist/bernoulli p-change))\n"
           "      _ (mx/eval! cp)\n"
           "      is-cp (> (mx/item cp) 0.5)\n"
           "      new-mu (if is-cp\n"
           "               (trace :mu_new (dist/gaussian 0 5))\n"
           "               prev-mu)]\n"
           "  (trace :y (dist/gaussian new-mu 1.0)))\n"
           "```\n\n"
           "The `if` on a sampled Bernoulli creates **data-dependent random structure**.\n\n"
           "## Results\n\n"
           "| Method | Particles | log-ML error | ESS | Time (ms) |\n"
           "|--------|-----------|-------------|-----|----------|\n"
           (apply str
             (for [r @results-acc]
               (str "| " (:method r) " | " (:particles r)
                    " | " (.toFixed (:error r) 2)
                    " +/- " (.toFixed (:error_std r) 2)
                    " | " (.toFixed (:ess r) 1)
                    " | " (.toFixed (:time_ms r) 0) " |\n")))
           "\n## Interpretation\n\n"
           "SMC via Unfold combinator exploits sequential structure. IS with prior "
           "proposals suffers weight degeneracy. The model demonstrates GenMLX's "
           "expressiveness: data-dependent structure is natural.\n")]
  (.writeFileSync fs (str results-dir "/SUMMARY.md") summary)
  (println (str "\n  Wrote: " results-dir "/SUMMARY.md")))

(println "\nAll changepoint benchmarks complete.")
(.exit js/process 0)
