(ns bench.rao-blackwell
  "Rao-Blackwellization benchmark.

   Measures the benefit of partial conjugacy (L3 analytical elimination)
   versus full importance sampling (L2) on a model with BOTH conjugate
   and non-conjugate parameters.

   Model: 2 Normal-Normal conjugate groups (mu1, mu2) + 1 non-conjugate
   parameter (sigma with gamma prior, not conjugate to gaussian obs).

   L3 analytically eliminates mu1 and mu2, reducing the effective
   dimensionality of the sampling problem. L2 samples all three
   parameters from the prior.

   Metrics: log-ML estimate variance, ESS, timing.

   Output: results/rao-blackwell/data.json

   Usage: bun run --bun nbb bench/rao_blackwell.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.importance :as is]
            [genmlx.inference.util :as u]
            [genmlx.compiled :as compiled])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Infrastructure
;; ---------------------------------------------------------------------------

(def fs (js/require "fs"))
(def path-mod (js/require "path"))

(def out-dir
  (or (aget (.-env js/process) "GENMLX_RESULTS_DIR")
      (.resolve path-mod (js/process.cwd) "results/rao-blackwell")))

(defn ensure-dir [dir]
  (when-not (.existsSync fs dir)
    (.mkdirSync fs dir #js {:recursive true})))

(defn write-json [filename data]
  (ensure-dir out-dir)
  (let [filepath (str out-dir "/" filename)]
    (.writeFileSync fs filepath (js/JSON.stringify (clj->js data) nil 2))
    (println (str "  Wrote: " filepath))))

(defn perf-now [] (js/performance.now))

(defn benchmark
  "Run f repeatedly, return timing statistics."
  [label f & {:keys [warmup-n outer-n inner-n]
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

;; Independent closed-form marginal log-ML (exactness ground truth).
(def ^:private LOG-2PI 1.8378770664093453)

(defn nn-shared-marginal
  "Joint marginal log p(y) for shared-mean Normal-Normal:
   mu ~ N(m0, prior-var); y_i ~ N(mu, obs-var)."
  [ys m0 prior-var obs-var]
  (let [n (count ys)
        ds (map #(- % m0) ys)
        sum-d (reduce + ds)
        sum-d2 (reduce + (map #(* % %) ds))
        denom (+ obs-var (* n prior-var))
        logdet (+ (* (dec n) (js/Math.log obs-var)) (js/Math.log denom))
        quad (/ (- sum-d2 (* (/ prior-var denom) sum-d sum-d)) obs-var)]
    (* -0.5 (+ (* n LOG-2PI) logdet quad))))

;; sigma ~ Gamma(2,1) is sampled but never enters the likelihood, so the marginal
;; is purely the two independent Normal-Normal groups (mu1: y1,y2; mu2: y3,y4).
(def closed-form-log-ml
  (+ (nn-shared-marginal [1.5 2.0] 0.0 100.0 1.0)
     (nn-shared-marginal [-0.5 -1.0] 0.0 100.0 1.0)))

;; ---------------------------------------------------------------------------
;; Strip analytical plan (forces L2 / prior-proposal IS)
;; ---------------------------------------------------------------------------

(defn strip-analytical
  "Remove auto-handlers from schema, forcing standard generate (prior proposal).
   This gives us the L2 baseline for comparison."
  [gf]
  (assoc gf :schema (dissoc (:schema gf) :auto-handlers :conjugate-pairs
                            :has-conjugate? :analytical-plan)))

;; ---------------------------------------------------------------------------
;; IS trial runner
;; ---------------------------------------------------------------------------

(defn generate-weight
  "Run p/generate and extract the log-weight as a JS number."
  [model args obs]
  (let [{:keys [weight]} (p/generate model args obs)]
    (mx/eval! weight)
    (mx/item weight)))

(defn run-is-trial
  "Run one IS trial: generate n-particles weights, return log-ML and ESS."
  [model args obs n-particles seed]
  (let [keys (rng/split-n (rng/fresh-key seed) n-particles)
        log-ws (mapv (fn [ki]
                       (let [w (generate-weight (dyn/with-key model ki) args obs)]
                         (mx/clear-cache!)
                         w))
                     keys)]
    {:log-ml (log-ml-from-log-weights log-ws)
     :ess (ess-from-log-weights log-ws)}))

(defn run-trials
  "Run n-trials IS trials and collect log-ML and ESS vectors."
  [model args obs n-particles n-trials seed-base]
  (let [trials (mapv (fn [t]
                       (let [result (run-is-trial model args obs n-particles (+ seed-base t))]
                         (mx/clear-cache!)
                         result))
                     (range n-trials))
        log-mls (mapv :log-ml trials)
        esses (mapv :ess trials)]
    {:log-mls log-mls :esses esses
     :log-ml-mean (mean log-mls) :log-ml-std (std log-mls)
     :log-ml-var (variance log-mls)
     :ess-mean (mean esses)}))

;; ---------------------------------------------------------------------------
;; Model: 2 NN-conjugate groups + 1 non-conjugate
;; ---------------------------------------------------------------------------

;; mu1, mu2 have gaussian priors and gaussian observations => Normal-Normal conjugate
;; sigma has gamma prior but appears as std-dev in gaussian obs => NOT conjugate
(def mixed-model
  (dyn/auto-key
    (gen []
      (let [mu1   (trace :mu1 (dist/gaussian 0 10))
            mu2   (trace :mu2 (dist/gaussian 0 10))
            sigma (trace :sigma (dist/gamma-dist 2 1))]
        (trace :y1 (dist/gaussian mu1 1))
        (trace :y2 (dist/gaussian mu1 1))
        (trace :y3 (dist/gaussian mu2 1))
        (trace :y4 (dist/gaussian mu2 1))
        ;; sigma is sampled but not analytically eliminable
        {:mu1 mu1 :mu2 mu2 :sigma sigma}))))

;; Observations
(def obs
  (-> cm/EMPTY
      (cm/set-value :y1 (mx/scalar 1.5))
      (cm/set-value :y2 (mx/scalar 2.0))
      (cm/set-value :y3 (mx/scalar -0.5))
      (cm/set-value :y4 (mx/scalar -1.0))))

;; ---------------------------------------------------------------------------
;; Config
;; ---------------------------------------------------------------------------

(def n-particles 50)
(def n-trials 3)

;; ---------------------------------------------------------------------------
;; Main execution
;; ---------------------------------------------------------------------------

(println "\n" (apply str (repeat 70 "=")))
(println "  Rao-Blackwellization Benchmark")
(println (apply str (repeat 70 "=")))
(println "\n  Model: 2 NN-conjugate groups (mu1, mu2) + 1 non-conjugate (sigma)")
(println (str "  Config: " n-particles " IS particles, " n-trials " trials"))

;; Show conjugacy info from schema
(let [conj-pairs (get-in mixed-model [:schema :conjugate-pairs])]
  (println (str "\n  Conjugate pairs detected: " (count (or conj-pairs []))))
  (when (seq conj-pairs)
    (doseq [pair conj-pairs]
      (println (str "    " (:prior-addr pair) " -> " (:obs-addr pair)
                    " [" (:family pair) "]")))))

;; ---------------------------------------------------------------------------
;; L3: With analytical elimination (Rao-Blackwellized)
;; ---------------------------------------------------------------------------

(println "\n" (apply str (repeat 60 "-")))
(println "  L3: With analytical elimination (Rao-Blackwellized)")
(println (apply str (repeat 60 "-")))

(def l3-timing
  (benchmark "L3-analytical"
             (fn [] (generate-weight mixed-model [] obs))
             :warmup-n 2 :outer-n 3 :inner-n 3))

(println (str "\n  Running L3 trials: " n-particles " particles x " n-trials " trials..."))
(def l3-results (run-trials mixed-model [] obs n-particles n-trials 1000))

(println (str "  L3 log-ML: " (.toFixed (:log-ml-mean l3-results) 6)
              " +/- " (.toFixed (:log-ml-std l3-results) 6)))
(println (str "  L3 ESS:    " (.toFixed (:ess-mean l3-results) 1)
              " / " n-particles
              " (" (.toFixed (* 100 (/ (:ess-mean l3-results) n-particles)) 1) "%)"))

;; ---------------------------------------------------------------------------
;; L2: No analytical elimination (full prior-proposal IS)
;; ---------------------------------------------------------------------------

(println "\n" (apply str (repeat 60 "-")))
(println "  L2: No analytical elimination (prior-proposal IS)")
(println (apply str (repeat 60 "-")))

(def l2-model (dyn/auto-key (strip-analytical mixed-model)))

(def l2-timing
  (benchmark "L2-no-analytical"
             (fn [] (generate-weight l2-model [] obs))
             :warmup-n 2 :outer-n 3 :inner-n 3))

(println (str "\n  Running L2 trials: " n-particles " particles x " n-trials " trials..."))
(def l2-results (run-trials l2-model [] obs n-particles n-trials 2000))

(println (str "  L2 log-ML: " (.toFixed (:log-ml-mean l2-results) 6)
              " +/- " (.toFixed (:log-ml-std l2-results) 6)))
(println (str "  L2 ESS:    " (.toFixed (:ess-mean l2-results) 1)
              " / " n-particles
              " (" (.toFixed (* 100 (/ (:ess-mean l2-results) n-particles)) 1) "%)"))

;; ---------------------------------------------------------------------------
;; Comparison
;; ---------------------------------------------------------------------------

(println "\n" (apply str (repeat 60 "-")))
(println "  Comparison")
(println (apply str (repeat 60 "-")))

;; Exactness (headline): |L3 log-ML − closed form|. Variance reduction is a
;; corollary, omitted when L3 is exact (var=0 → undefined ratio, the old Inf artifact).
(def l3-abs-error (js/Math.abs (- (:log-ml-mean l3-results) closed-form-log-ml)))
(def l2-abs-error (js/Math.abs (- (:log-ml-mean l2-results) closed-form-log-ml)))
(def l3-exact? (< (:log-ml-var l3-results) 1e-20))
(def var-ratio
  (when-not l3-exact?
    (/ (:log-ml-var l2-results) (:log-ml-var l3-results))))

(def ess-ratio
  (/ (:ess-mean l3-results) (max (:ess-mean l2-results) 0.01)))

(println (str "  Closed-form marginal log-ML: " (.toFixed closed-form-log-ml 6)))
(println (str "  |L3 log-ML − closed form|: " (.toExponential l3-abs-error 3)
              " nats  (exactness — the headline)"))
(println (str "  |L2 log-ML − closed form|: " (.toFixed l2-abs-error 4) " nats"))
(println (str "  Variance reduction (L2/L3): "
              (if l3-exact? "n/a (L3 exact, var=0)"
                  (str (.toFixed var-ratio 1) "x"))))
(println (str "  ESS improvement (L3/L2):    " (.toFixed ess-ratio 1) "x"))
(println (str "  L2 time: " (.toFixed (:mean-ms l2-timing) 3) " ms"))
(println (str "  L3 time: " (.toFixed (:mean-ms l3-timing) 3) " ms"))

;; ---------------------------------------------------------------------------
;; Summary table
;; ---------------------------------------------------------------------------

(println "\n\n" (apply str (repeat 70 "=")))
(println "         RAO-BLACKWELLIZATION RESULTS")
(println (apply str (repeat 70 "=")))

(println "\n| Condition            | log-ML mean | log-ML std | ESS mean  | Time (ms) |")
(println "|----------------------|-------------|------------|-----------|-----------|")
(println (str "| L2 (no analytical)   | "
              (.padStart (.toFixed (:log-ml-mean l2-results) 4) 11)
              " | " (.padStart (.toFixed (:log-ml-std l2-results) 6) 10)
              " | " (.padStart (.toFixed (:ess-mean l2-results) 1) 9)
              " | " (.padStart (.toFixed (:mean-ms l2-timing) 3) 9)
              " |"))
(println (str "| L3 (with analytical) | "
              (.padStart (.toFixed (:log-ml-mean l3-results) 4) 11)
              " | " (.padStart (.toFixed (:log-ml-std l3-results) 6) 10)
              " | " (.padStart (.toFixed (:ess-mean l3-results) 1) 9)
              " | " (.padStart (.toFixed (:mean-ms l3-timing) 3) 9)
              " |"))
(println (str "\n  |L3 log-ML − closed form|: " (.toExponential l3-abs-error 3) " nats"))
(println (str "  Variance reduction ratio: "
              (if l3-exact? "n/a (L3 exact, var=0)"
                  (str (.toFixed var-ratio 1) "x"))))
(println (str "  Conjugate sites eliminated: mu1, mu2"))
(println (str "  Non-conjugate sites sampled: sigma"))

;; ---------------------------------------------------------------------------
;; Write JSON results
;; ---------------------------------------------------------------------------

(write-json "data.json"
  (cond->
   {:experiment "rao-blackwell"
    :description (str "Analytical elimination exactness: mu1, mu2 marginalized (sigma "
                      "sampled but irrelevant to the marginal). |L3 log-ML − closed form|.")
    :timestamp (.toISOString (js/Date.))
    :model "mixed: 2 NN-conjugate + 1 non-conjugate"
    :config {:n_particles n-particles :n_trials n-trials}
    :closed-form-log-ml closed-form-log-ml
    :conditions
    {:L2-no-analytical
     {:log-ml-mean (:log-ml-mean l2-results)
      :log-ml-std (:log-ml-std l2-results)
      :log-ml-var (:log-ml-var l2-results)
      :log-ml-abs-error-nats l2-abs-error
      :ess-mean (:ess-mean l2-results)
      :timing-ms (:mean-ms l2-timing)
      :timing-std-ms (:std-ms l2-timing)}
     :L3-with-analytical
     {:log-ml-mean (:log-ml-mean l3-results)
      :log-ml-std (:log-ml-std l3-results)
      :log-ml-var (:log-ml-var l3-results)
      :log-ml-abs-error-nats l3-abs-error
      :ess-mean (:ess-mean l3-results)
      :timing-ms (:mean-ms l3-timing)
      :timing-std-ms (:std-ms l3-timing)}}
    :ess-improvement ess-ratio
    :conjugate-sites ["mu1" "mu2"]
    :non-conjugate-sites ["sigma"]}
    (some? var-ratio) (assoc :variance-reduction var-ratio)))

(println "\nRao-Blackwellization benchmark complete.")
