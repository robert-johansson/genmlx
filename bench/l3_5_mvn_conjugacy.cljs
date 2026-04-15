(ns bench.l3-5-mvn-conjugacy
  "L3.5 MVN Conjugacy Benchmark.

   Demonstrates multivariate normal-normal conjugacy — the L3.5 extension
   that uses mx/solve (Kalman gain form) instead of matrix inverse.

   Measures variance reduction and posterior accuracy for a D-dimensional
   MVN prior + MVN observation model, comparing:
   - L2 standard IS (analytical handlers stripped, prior proposal)
   - L3.5 analytical (auto-detected MVN conjugacy, exact posterior)

   The MVN-MVN conjugate update in auto_analytical.cljs computes:
     M = S0 + R (marginal covariance)
     K = S0 * M^{-1} via mx/solve (Kalman gain form)
     m1 = m0 + K * (y - m0)
     S1 = S0 - K * S0
   This avoids 3x matrix inverse calls for numerical stability.

   Output: results/l3.5-mvn-conjugacy/data.json

   Usage: bun run --bun nbb bench/l3_5_mvn_conjugacy.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.importance :as is]
            [genmlx.inference.util :as u])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Infrastructure
;; ---------------------------------------------------------------------------

(def fs (js/require "fs"))
(def path-mod (js/require "path"))

(def results-dir
  (or (aget (.-env js/process) "GENMLX_RESULTS_DIR")
      (.resolve path-mod (js/process.cwd) "results/l3.5-mvn-conjugacy")))

(defn ensure-dir [dir]
  (when-not (.existsSync fs dir)
    (.mkdirSync fs dir #js {:recursive true})))

(defn write-json [filename data]
  (ensure-dir results-dir)
  (let [filepath (str results-dir "/" filename)]
    (.writeFileSync fs filepath (js/JSON.stringify (clj->js data) nil 2))
    (println (str "  wrote: " filepath))))

(defn perf-now [] (js/performance.now))

(defn benchmark [label f & {:keys [warmup-n outer-n inner-n]
                             :or {warmup-n 10 outer-n 5 inner-n 10}}]
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
    (println (str "  [" label "] " (.toFixed mean-ms 3) " +/- "
                  (.toFixed std-ms 3) " ms"))
    {:label label :mean-ms mean-ms :std-ms std-ms
     :min-ms (apply min outer-times) :max-ms (apply max outer-times)
     :raw outer-times}))

;; ---------------------------------------------------------------------------
;; Statistics helpers
;; ---------------------------------------------------------------------------

(defn mean [xs] (/ (reduce + xs) (count xs)))

(defn variance [xs]
  (let [m (mean xs)
        n (count xs)]
    (/ (reduce + (map #(let [d (- % m)] (* d d)) xs)) n)))

(defn std [xs] (js/Math.sqrt (variance xs)))

(defn ess-from-log-weights
  "Effective sample size from log-weights."
  [log-ws]
  (let [max-w (apply max log-ws)
        ws (map #(js/Math.exp (- % max-w)) log-ws)
        s (reduce + ws)
        nw (map #(/ % s) ws)]
    (/ 1.0 (reduce + (map #(* % %) nw)))))

(defn log-ml-from-log-weights
  "Log marginal likelihood estimate via log-sum-exp."
  [log-ws]
  (let [n (count log-ws)
        max-w (apply max log-ws)
        lse (+ max-w (js/Math.log (reduce + (map #(js/Math.exp (- % max-w)) log-ws))))]
    (- lse (js/Math.log n))))

;; ---------------------------------------------------------------------------
;; Strip analytical plan (forces L2 / prior-proposal IS)
;; ---------------------------------------------------------------------------

(defn strip-analytical
  "Remove auto-handlers from schema, forcing standard generate (prior proposal).
   This gives the L2 baseline for comparison."
  [gf]
  (assoc gf :schema (dissoc (:schema gf) :auto-handlers :conjugate-pairs
                            :has-conjugate? :analytical-plan)))

;; ---------------------------------------------------------------------------
;; Model helpers
;; ---------------------------------------------------------------------------

(defn eye
  "D x D identity matrix."
  [d]
  (let [vals (for [i (range d) j (range d)] (if (= i j) 1.0 0.0))]
    (mx/reshape (mx/array (vec vals)) [d d])))

(defn scale-eye
  "D x D diagonal matrix with scalar s on the diagonal."
  [d s]
  (let [vals (for [i (range d) j (range d)] (if (= i j) s 0.0))]
    (mx/reshape (mx/array (vec vals)) [d d])))

;; ---------------------------------------------------------------------------
;; Configuration
;; ---------------------------------------------------------------------------

(def D 5)
(def prior-var 100.0)   ;; prior covariance = 100 * I
(def obs-var 1.0)        ;; observation covariance = I
(def n-particles 1000)
(def n-trials 10)

;; ---------------------------------------------------------------------------
;; Model: D-dimensional MVN prior + MVN observation
;; ---------------------------------------------------------------------------

;; The model uses dist/multivariate-normal for both prior and observation.
;; L3.5 auto-detects the MVN-MVN conjugate pair and applies the Kalman gain
;; form update using mx/solve.

(def mvn-model
  (dyn/auto-key
    (gen []
      (let [mu (trace :mu (dist/multivariate-normal
                            (mx/zeros [D])
                            (scale-eye D prior-var)))]
        (trace :y (dist/multivariate-normal mu (eye D)))
        mu))))

;; Observation: a fixed D-dimensional vector
(def obs-data (mx/array [3.2 -1.5 0.8 2.1 -0.4]))

(def observations
  (cm/set-value cm/EMPTY :y obs-data))

;; ---------------------------------------------------------------------------
;; Analytic ground truth
;; ---------------------------------------------------------------------------
;; For MVN-MVN conjugacy with diagonal covariances:
;;   Prior:  mu ~ N(0, sigma_prior^2 * I)
;;   Obs:    y | mu ~ N(mu, sigma_obs^2 * I)
;;   Posterior: mu | y ~ N(m1, S1) where
;;     S1 = (S0^{-1} + R^{-1})^{-1}
;;     m1 = S1 * R^{-1} * y    (since prior mean is 0)
;;
;; With diagonal covariances this decomposes into D independent scalar updates:
;;   posterior_var_i = 1 / (1/prior_var + 1/obs_var)
;;   posterior_mean_i = posterior_var_i * (0/prior_var + obs_i/obs_var)
;;                    = posterior_var_i * obs_i / obs_var

(def analytic-posterior-var (/ 1.0 (+ (/ 1.0 prior-var) (/ 1.0 obs-var))))
(def obs-raw [3.2 -1.5 0.8 2.1 -0.4])
(def analytic-posterior-mean
  (mapv #(* analytic-posterior-var (/ % obs-var)) obs-raw))

;; Analytic marginal LL: y ~ N(0, S0 + R) = N(0, (prior_var + obs_var) * I)
;; For diagonal case: sum_i of -0.5 * (log(2pi) + log(prior_var + obs_var) + y_i^2/(prior_var + obs_var))
(def ^:private LOG-2PI 1.8378770664093453)
(def marginal-var (+ prior-var obs-var))
(def analytic-log-ml
  (reduce + (map (fn [yi]
                   (* -0.5 (+ LOG-2PI
                              (js/Math.log marginal-var)
                              (/ (* yi yi) marginal-var))))
                 obs-raw)))

;; ---------------------------------------------------------------------------
;; IS trial runner (memory-safe)
;; ---------------------------------------------------------------------------

(defn generate-weight
  "Run p/generate and extract the log-weight as a JS number."
  [model args obs]
  (let [{:keys [weight]} (p/generate model args obs)]
    (mx/eval! weight)
    (mx/item weight)))

(defn run-is-trial
  "Run one IS trial: generate n-particles weights, return log-ML and ESS.
   Uses tidy-run per particle and periodic cleanup to avoid OOM."
  [model args obs n seed]
  (let [keys (rng/split-n (rng/fresh-key seed) n)
        log-ws (into []
                 (map-indexed
                   (fn [i ki]
                     (let [w (mx/tidy-run
                               (fn []
                                 (let [{:keys [weight]} (p/generate (dyn/with-key model ki) args obs)]
                                   (mx/materialize! weight)
                                   (mx/item weight)))
                               (fn [_] []))]
                       (when (zero? (mod (inc i) 100)) (mx/sweep-dead-arrays!))
                       w)))
                 keys)]
    {:log-ml (log-ml-from-log-weights log-ws)
     :ess (ess-from-log-weights log-ws)}))

(defn run-trials
  "Run n-trials IS trials and collect statistics."
  [model args obs n-per-trial n-trials seed-base]
  (let [trials (mapv (fn [t]
                       (let [r (run-is-trial model args obs n-per-trial (+ seed-base t))]
                         (mx/sweep-dead-arrays!)
                         r))
                     (range n-trials))
        log-mls (mapv :log-ml trials)
        esses (mapv :ess trials)]
    {:log-mls log-mls :esses esses
     :log-ml-mean (mean log-mls) :log-ml-std (std log-mls)
     :log-ml-var (variance log-mls)
     :ess-mean (mean esses)}))

;; ---------------------------------------------------------------------------
;; Main execution
;; ---------------------------------------------------------------------------

(println "\n" (apply str (repeat 70 "=")))
(println "  L3.5 MVN Conjugacy Benchmark")
(println (apply str (repeat 70 "=")))
(println (str "\n  D=" D " dimensions, prior_var=" prior-var ", obs_var=" obs-var))
(println (str "  " n-particles " IS particles x " n-trials " trials"))

;; Verify conjugacy detection
(let [s (:schema mvn-model)]
  (println (str "\n  Schema: static=" (:static? s)
                ", conjugate-pairs=" (count (:conjugate-pairs s))
                ", has-auto-handlers=" (some? (:auto-handlers s))))
  (when-let [pairs (:conjugate-pairs s)]
    (doseq [pair pairs]
      (println (str "    Pair: " (:prior-addr pair) " -> " (:obs-addr pair)
                    " [" (:family pair) "]")))))

;; Analytic ground truth
(println (str "\n  Analytic posterior mean: " (mapv #(.toFixed % 4) analytic-posterior-mean)))
(println (str "  Analytic posterior var (each dim): " (.toFixed analytic-posterior-var 6)))
(println (str "  Analytic log-ML: " (.toFixed analytic-log-ml 6)))

;; =========================================================================
;; Condition 1: L3.5 analytical (auto-detected MVN conjugacy)
;; =========================================================================

(println (str "\n" (apply str (repeat 60 "-"))))
(println "  Condition 1: L3.5 Analytical (MVN conjugacy via mx/solve)")
(println (apply str (repeat 60 "-")))

;; Single-generate check: L3.5 should give deterministic weight = marginal LL
(let [result (p/generate mvn-model [] observations)
      w (mx/item (:weight result))
      mu-val (cm/get-value (cm/get-submap (:choices (:trace result)) :mu))]
  (println (str "  Single generate weight: " (.toFixed w 6)
                " (analytic: " (.toFixed analytic-log-ml 6) ")"))
  (println (str "  Posterior mean from trace:"))
  (dotimes [i D]
    (println (str "    dim " i ": " (.toFixed (mx/item (mx/index mu-val i)) 4)
                  " (analytic: " (.toFixed (nth analytic-posterior-mean i) 4) ")"))))

;; Timing
(def l35-timing
  (benchmark "L3.5-analytical"
             (fn [] (generate-weight mvn-model [] observations))
             :warmup-n 10 :outer-n 5 :inner-n 15))

;; IS trials (L3.5 gives exact weights, so all particles have identical weight)
(println (str "\n  Running L3.5 IS, " n-particles " particles x " n-trials " trials..."))
(def l35-results
  (run-trials mvn-model [] observations n-particles n-trials 1000))

(println (str "  L3.5 log-ML: " (.toFixed (:log-ml-mean l35-results) 6)
              " +/- " (.toFixed (:log-ml-std l35-results) 6)))
(println (str "  L3.5 ESS:    " (.toFixed (:ess-mean l35-results) 1)
              " / " n-particles
              " (" (.toFixed (* 100 (/ (:ess-mean l35-results) n-particles)) 1) "%)"))

;; Posterior mean error (from a single generate)
(def l35-posterior-error
  (let [result (p/generate mvn-model [] observations)
        mu-val (cm/get-value (cm/get-submap (:choices (:trace result)) :mu))
        errors (mapv (fn [i]
                       (js/Math.abs (- (mx/item (mx/index mu-val i))
                                       (nth analytic-posterior-mean i))))
                     (range D))]
    (mean errors)))

(println (str "  L3.5 mean posterior error: " (.toFixed l35-posterior-error 6)))

;; =========================================================================
;; Condition 2: L2 standard IS (prior proposal, no analytical handlers)
;; =========================================================================

(println (str "\n" (apply str (repeat 60 "-"))))
(println "  Condition 2: L2 Standard IS (prior proposal)")
(println (apply str (repeat 60 "-")))

(def l2-model (dyn/auto-key (strip-analytical mvn-model)))

;; Timing
(def l2-timing
  (benchmark "L2-standard-IS"
             (fn [] (generate-weight l2-model [] observations))
             :warmup-n 10 :outer-n 5 :inner-n 15))

;; IS trials
(println (str "\n  Running L2 IS, " n-particles " particles x " n-trials " trials..."))
(def l2-results
  (run-trials l2-model [] observations n-particles n-trials 2000))

(println (str "  L2 log-ML: " (.toFixed (:log-ml-mean l2-results) 6)
              " +/- " (.toFixed (:log-ml-std l2-results) 6)))
(println (str "  L2 ESS:    " (.toFixed (:ess-mean l2-results) 1)
              " / " n-particles
              " (" (.toFixed (* 100 (/ (:ess-mean l2-results) n-particles)) 1) "%)"))

;; Posterior mean error (weighted average from IS, memory-safe)
(def l2-posterior-error
  (let [n-post 100
        keys (rng/split-n (rng/fresh-key 9999) n-post)
        results (into []
                  (map-indexed
                    (fn [idx ki]
                      (let [r (p/generate (dyn/with-key l2-model ki) [] observations)
                            w (mx/item (:weight r))
                            mu-val (cm/get-value (cm/get-submap (:choices (:trace r)) :mu))
                            mu-js (mapv #(mx/item (mx/index mu-val %)) (range D))]
                        (when (zero? (mod (inc idx) 25)) (mx/sweep-dead-arrays!))
                        {:weight w :mu-js mu-js})))
                  keys)
        log-ws (mapv :weight results)
        max-w (apply max log-ws)
        ws (mapv #(js/Math.exp (- % max-w)) log-ws)
        sum-w (reduce + ws)
        nw (mapv #(/ % sum-w) ws)
        weighted-means (mapv (fn [i]
                               (reduce + (map-indexed
                                           (fn [j r]
                                             (* (nth nw j) (nth (:mu-js r) i)))
                                           results)))
                             (range D))
        errors (mapv (fn [i]
                       (js/Math.abs (- (nth weighted-means i)
                                       (nth analytic-posterior-mean i))))
                     (range D))]
    (mean errors)))

(println (str "  L2 mean posterior error: " (.toFixed l2-posterior-error 6)))

;; =========================================================================
;; Dimension scaling
;; =========================================================================

(println (str "\n" (apply str (repeat 60 "-"))))
(println "  Dimension Scaling")
(println (apply str (repeat 60 "-")))

(def dimension-scaling
  (vec (for [d [2 5 10 20 50]]
         (let [model (dyn/auto-key
                       (gen []
                         (let [mu (trace :mu (dist/multivariate-normal
                                               (mx/zeros [d])
                                               (scale-eye d prior-var)))]
                           (trace :y (dist/multivariate-normal mu (eye d)))
                           mu)))
               obs-val (mx/array (vec (take d (cycle [3.2 -1.5 0.8 2.1 -0.4]))))
               obs (cm/set-value cm/EMPTY :y obs-val)
               has-auto (some? (:auto-handlers (:schema model)))
               n-reps 20
               ;; Time analytical generate
               _ (dotimes [_ 3] (generate-weight model [] obs)) ;; warmup
               t0 (perf-now)
               _ (dotimes [_ n-reps] (generate-weight model [] obs))
               _ (mx/materialize!)
               t-analytical (/ (- (perf-now) t0) n-reps)
               _ (mx/sweep-dead-arrays!)
               ;; Stripped model
               model-l2 (dyn/auto-key (strip-analytical model))
               _ (dotimes [_ 3] (generate-weight model-l2 [] obs)) ;; warmup
               t1 (perf-now)
               _ (dotimes [_ n-reps] (generate-weight model-l2 [] obs))
               _ (mx/materialize!)
               t-standard (/ (- (perf-now) t1) n-reps)
               _ (mx/sweep-dead-arrays!)]
           (println (str "  d=" d
                         ": analytical=" (.toFixed t-analytical 3) "ms"
                         ", standard=" (.toFixed t-standard 3) "ms"
                         ", has-auto=" has-auto))
           {:d d
            :has-auto-handlers has-auto
            :analytical-ms t-analytical
            :standard-ms t-standard}))))

;; =========================================================================
;; Summary
;; =========================================================================

(def var-ratio
  (if (> (:log-ml-var l35-results) 1e-20)
    (/ (:log-ml-var l2-results) (:log-ml-var l35-results))
    js/Infinity))

(def ess-ratio
  (/ (:ess-mean l35-results) (max (:ess-mean l2-results) 0.01)))

(println (str "\n" (apply str (repeat 70 "="))))
(println "         L3.5 MVN CONJUGACY RESULTS")
(println (apply str (repeat 70 "=")))

(println (str "\n  Dimensions:            " D))
(println (str "  Prior covariance:      " prior-var " * I"))
(println (str "  Observation covariance: " obs-var " * I"))
(println (str "  Analytic log-ML:       " (.toFixed analytic-log-ml 6)))
(println)

(println "| Condition | log-ML mean | log-ML std | ESS | Time (ms) | Post. Error |")
(println "|-----------|-------------|------------|-----|-----------|-------------|")
(println (str "| L3.5-analytical | "
              (.toFixed (:log-ml-mean l35-results) 6) " | "
              (.toFixed (:log-ml-std l35-results) 6) " | "
              (.toFixed (:ess-mean l35-results) 1) " | "
              (.toFixed (:mean-ms l35-timing) 3) " | "
              (.toFixed l35-posterior-error 6) " |"))
(println (str "| L2-standard-IS  | "
              (.toFixed (:log-ml-mean l2-results) 6) " | "
              (.toFixed (:log-ml-std l2-results) 6) " | "
              (.toFixed (:ess-mean l2-results) 1) " | "
              (.toFixed (:mean-ms l2-timing) 3) " | "
              (.toFixed l2-posterior-error 6) " |"))

(println (str "\n  Variance reduction:  " (if (= var-ratio js/Infinity)
                                              "Inf (L3.5 exact)"
                                              (str (.toFixed var-ratio 1) "x"))))
(println (str "  ESS improvement:     " (.toFixed ess-ratio 1) "x"))

(println "\n  Dimension scaling:")
(println "  | D  | Analytical (ms) | Standard (ms) | Auto-detected |")
(println "  |----|-----------------|---------------|---------------|")
(doseq [{:keys [d analytical-ms standard-ms has-auto-handlers]} dimension-scaling]
  (println (str "  | " (.padStart (str d) 2)
                " | " (.padStart (.toFixed analytical-ms 3) 15)
                " | " (.padStart (.toFixed standard-ms 3) 13)
                " | " (.padStart (str has-auto-handlers) 13) " |")))

;; =========================================================================
;; Write JSON results
;; =========================================================================

(write-json "data.json"
  {:experiment "l3.5-mvn-conjugacy"
   :description "MVN-MVN conjugacy via Kalman gain form (mx/solve)"
   :timestamp (.toISOString (js/Date.))
   :hardware {:platform "macOS" :chip "Apple Silicon" :gpu "Metal"}
   :dimensions D
   :config {:prior-var prior-var
            :obs-var obs-var
            :n-particles n-particles
            :n-trials n-trials}
   :conditions
   {:L2-standard-IS
    {:log-ml-mean (:log-ml-mean l2-results)
     :log-ml-std (:log-ml-std l2-results)
     :log-ml-var (:log-ml-var l2-results)
     :ess-mean (:ess-mean l2-results)
     :posterior-mean-error l2-posterior-error
     :timing-ms (:mean-ms l2-timing)
     :timing-std-ms (:std-ms l2-timing)}
    :L3.5-analytical
    {:log-ml-mean (:log-ml-mean l35-results)
     :log-ml-std (:log-ml-std l35-results)
     :log-ml-var (:log-ml-var l35-results)
     :ess-mean (:ess-mean l35-results)
     :posterior-mean-error l35-posterior-error
     :timing-ms (:mean-ms l35-timing)
     :timing-std-ms (:std-ms l35-timing)}}
   :variance-reduction (if (= var-ratio js/Infinity) "Infinity" var-ratio)
   :ess-improvement ess-ratio
   :analytic-posterior
   {:mean analytic-posterior-mean
    :var analytic-posterior-var
    :log-ml analytic-log-ml}
   :dimension-scaling
   (mapv (fn [{:keys [d analytical-ms standard-ms has-auto-handlers]}]
           {:d d :analytical-ms analytical-ms :standard-ms standard-ms
            :auto-detected has-auto-handlers})
         dimension-scaling)})

(println "\nL3.5 MVN conjugacy benchmark complete.")
