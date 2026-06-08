(ns bench.conjugacy
  "L3 Auto-Conjugacy Showcase.

   Demonstrates automatic analytical elimination across 5 conjugate families.
   For each model, compares standard importance sampling (prior proposal)
   against L3 auto-conjugacy, measuring variance reduction in log-ML estimates.

   Sub-experiments:
     5A: Normal-Normal (mean estimation, 5 obs)
     5B: Beta-Bernoulli (coin flip, 5 obs)
     5C: Gamma-Poisson (count data, 4 obs)
     5D: Mixed model (partial conjugacy: 2 NN groups + 1 non-conjugate)
     5E: Static LinReg (full elimination, slope + intercept)

   Output: results/conjugacy/data.json

   Usage: bun run --bun nbb bench/conjugacy.cljs"
  (:require [clojure.string]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.importance :as is]
            [genmlx.inference.util :as u]
            [genmlx.method-selection :as ms]
            [genmlx.fit :as fit])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Infrastructure
;; ---------------------------------------------------------------------------

(defn perf-now [] (js/performance.now))

(def fs (js/require "fs"))
(def path-mod (js/require "path"))

;; Output dir: from env (orchestrator) or default
(def results-dir
  (or (aget (.-env js/process) "GENMLX_RESULTS_DIR")
      (.resolve path-mod (js/process.cwd) "results/conjugacy")))

(defn ensure-dir [dir]
  (when-not (.existsSync fs dir)
    (.mkdirSync fs dir #js {:recursive true})))

(defn write-json [filename data]
  (ensure-dir results-dir)
  (let [filepath (str results-dir "/" filename)]
    (.writeFileSync fs filepath (js/JSON.stringify (clj->js data) nil 2))
    (println (str "  Wrote: " filepath))))

(defn benchmark [label f & {:keys [warmup-n outer-n inner-n]
                             :or {warmup-n 10 outer-n 5 inner-n 10}}]
  (println (str "\n  [" label "] warming up..."))
  (dotimes [_ warmup-n] (f) (mx/materialize!))
  (mx/clear-cache!)
  (let [outer-times
        (vec (for [rep (range outer-n)]
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
;; Independent closed-form marginal log-ML (the exactness ground truth)
;; ---------------------------------------------------------------------------
;; These are derived INDEPENDENTLY of conjugate.cljs / the L3 path, so |L3 - cf|
;; is a genuine cross-check of analytical elimination, not a circular comparison.

(def ^:private LOG-2PI 1.8378770664093453)

(def ^:private lanczos-c
  [0.99999999999980993 676.5203681218851 -1259.1392167224028
   771.32342877765313 -176.61502916214059 12.507343278686905
   -0.13857109526572012 9.9843695780195716e-6 1.5056327351493116e-7])

(defn lgamma
  "Log-gamma via the Lanczos approximation (g=7)."
  [x]
  (if (< x 0.5)
    (- (js/Math.log (/ js/Math.PI (js/Math.sin (* js/Math.PI x)))) (lgamma (- 1.0 x)))
    (let [x (- x 1.0)
          t (+ x 7.5)
          a (reduce (fn [a i] (+ a (/ (nth lanczos-c i) (+ x i))))
                    (nth lanczos-c 0) (range 1 (count lanczos-c)))]
      (+ (* 0.5 (js/Math.log (* 2.0 js/Math.PI)))
         (* (+ x 0.5) (js/Math.log t)) (- t) (js/Math.log a)))))

(defn log-beta [a b] (- (+ (lgamma a) (lgamma b)) (lgamma (+ a b))))

(defn nn-shared-marginal
  "Joint marginal log p(y) for shared-mean Normal-Normal:
   mu ~ N(m0, prior-var); y_i ~ N(mu, obs-var). y ~ N(m0·1, obs-var·I + prior-var·11ᵀ)."
  [ys m0 prior-var obs-var]
  (let [n (count ys)
        ds (map #(- % m0) ys)
        sum-d (reduce + ds)
        sum-d2 (reduce + (map #(* % %) ds))
        denom (+ obs-var (* n prior-var))
        logdet (+ (* (dec n) (js/Math.log obs-var)) (js/Math.log denom))
        quad (/ (- sum-d2 (* (/ prior-var denom) sum-d sum-d)) obs-var)]
    (* -0.5 (+ (* n LOG-2PI) logdet quad))))

(defn bb-marginal
  "Beta-Bernoulli marginal: logB(a+s, b+n-s) − logB(a,b). s = #successes."
  [s n a b]
  (- (log-beta (+ a s) (+ b (- n s))) (log-beta a b)))

(defn gp-marginal
  "Gamma-Poisson marginal (Gamma shape al, rate be): rate ~ Gamma(al,be); y_i ~ Poisson(rate)."
  [ys al be]
  (let [n (count ys) S (reduce + ys)]
    (+ (* al (js/Math.log be))
       (- (lgamma al))
       (- (reduce + (map #(lgamma (+ % 1.0)) ys)))
       (lgamma (+ al S))
       (- (* (+ al S) (js/Math.log (+ be n)))))))

(defn linreg-marginal
  "Joint marginal for 2-parameter Bayesian linear regression:
   slope,intercept ~ N(0, prior-var); y_j ~ N(slope·x_j + intercept, obs-var).
   y ~ N(0, obs-var·I + prior-var·X Xᵀ), X row = [x_j, 1]. Computed via the rank-2
   Woodbury / matrix-determinant lemma (independent of the eliminator)."
  [xs ys prior-var obs-var]
  (let [n (count xs)
        sx (reduce + xs)
        sx2 (reduce + (map #(* % %) xs))
        sxy (reduce + (map * xs ys))
        sy (reduce + ys)
        syy (reduce + (map #(* % %) ys))
        ratio (/ prior-var obs-var)
        ;; det(I2 + ratio·XᵀX), XᵀX = [[sx2 sx][sx n]]
        b00 (+ 1.0 (* ratio sx2)) b01 (* ratio sx)
        b10 (* ratio sx)         b11 (+ 1.0 (* ratio n))
        det-b (- (* b00 b11) (* b01 b10))
        logdet (+ (* n (js/Math.log obs-var)) (js/Math.log det-b))
        ;; quad = (1/obs-var)[yᵀy − vᵀ A⁻¹ v], A = (obs/prior)I2 + XᵀX, v = Xᵀy
        k (/ obs-var prior-var)
        a00 (+ k sx2) a01 sx a11 (+ k n)
        det-a (- (* a00 a11) (* a01 a01))
        v0 sxy v1 sy
        ;; A⁻¹ v
        ai0 (/ (- (* a11 v0) (* a01 v1)) det-a)
        ai1 (/ (- (* a00 v1) (* a01 v0)) det-a)
        quad (/ (- syy (+ (* v0 ai0) (* v1 ai1))) obs-var)]
    (* -0.5 (+ (* n LOG-2PI) logdet quad))))

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
;; Sub-experiment runner
;; ---------------------------------------------------------------------------

(defn run-sub-experiment
  "Run a full sub-experiment: method selection, L2 IS, L3 conjugacy.
   Returns a result map for JSON serialization."
  [label model args obs n-particles n-trials & {:keys [ground-truth]}]
  (println (str "\n" (apply str (repeat 60 "-"))))
  (println (str "  " label))
  (println (apply str (repeat 60 "-")))

  ;; 1. Method selection
  (let [model-ak (dyn/auto-key model)
        sel (ms/select-method model-ak obs)
        _ (println (str "  Method: " (:method sel) " (" (:reason sel) ")"))
        _ (println (str "  Eliminated: " (vec (:eliminated sel))
                        " (" (count (:eliminated sel)) " addrs)"))
        _ (println (str "  Residual: " (vec (:residual-addrs sel))
                        " (" (:n-residual sel) " addrs)"))

        ;; Conjugacy info from schema
        conj-pairs (get-in model-ak [:schema :conjugate-pairs])
        _ (when (seq conj-pairs)
            (doseq [pair conj-pairs]
              (println (str "    Pair: " (:prior-addr pair) " -> " (:obs-addr pair)
                            " [" (:family pair) "]"))))

        ;; 2. L3 (auto-conjugacy) trials
        _ (println (str "\n  Running L3 (auto-conjugacy), " n-particles " particles x " n-trials " trials..."))
        l3-timing (benchmark (str label "-L3")
                             (fn [] (generate-weight model-ak [] obs))
                             :warmup-n 3 :outer-n 3 :inner-n 5)
        l3-results (run-trials model-ak [] obs n-particles n-trials 1000)
        _ (println (str "  L3 log-ML: " (.toFixed (:log-ml-mean l3-results) 6)
                        " +/- " (.toFixed (:log-ml-std l3-results) 6)))
        _ (println (str "  L3 ESS:    " (.toFixed (:ess-mean l3-results) 1)
                        " / " n-particles
                        " (" (.toFixed (* 100 (/ (:ess-mean l3-results) n-particles)) 1) "%)"))

        ;; 3. L2 (standard IS, no analytical handlers) trials
        _ (mx/clear-cache!)
        l2-model (strip-analytical model-ak)
        _ (println (str "\n  Running L2 (prior-proposal IS), " n-particles " particles x " n-trials " trials..."))
        l2-timing (benchmark (str label "-L2")
                             (fn [] (generate-weight (dyn/auto-key l2-model) [] obs))
                             :warmup-n 3 :outer-n 3 :inner-n 5)
        l2-results (run-trials (dyn/auto-key l2-model) [] obs n-particles n-trials 2000)
        _ (println (str "  L2 log-ML: " (.toFixed (:log-ml-mean l2-results) 6)
                        " +/- " (.toFixed (:log-ml-std l2-results) 6)))
        _ (println (str "  L2 ESS:    " (.toFixed (:ess-mean l2-results) 1)
                        " / " n-particles
                        " (" (.toFixed (* 100 (/ (:ess-mean l2-results) n-particles)) 1) "%)"))

        ;; 4. Exactness (the headline) + variance reduction (corollary).
        ;; The honest headline for analytical elimination is |L3 log-ML − closed
        ;; form| in nats: ~0 when L3 is exact. Variance reduction is a corollary
        ;; and is OMITTED when L3 is exact (its var is 0, so the ratio is a
        ;; divide-by-~0 against a different model — the old >1000x/Inf artifact).
        cf (:log-ml ground-truth)
        l3-err (when cf (js/Math.abs (- (:log-ml-mean l3-results) cf)))
        l2-err (when cf (js/Math.abs (- (:log-ml-mean l2-results) cf)))
        l3-exact? (< (:log-ml-var l3-results) 1e-20)
        var-ratio (when-not l3-exact?
                    (/ (:log-ml-var l2-results) (:log-ml-var l3-results)))
        ess-ratio (/ (:ess-mean l3-results) (max (:ess-mean l2-results) 0.01))
        _ (println (str "\n  --- Summary ---"))
        _ (when cf
            (println (str "  |L3 log-ML − closed form|: " (.toExponential l3-err 3)
                          " nats  (exactness — the headline)"))
            (println (str "  |L2 log-ML − closed form|: " (.toFixed l2-err 4) " nats")))
        _ (println (str "  Variance reduction:  "
                        (if l3-exact? "n/a (L3 exact, var=0)"
                            (str (.toFixed var-ratio 1) "x"))))
        _ (println (str "  ESS improvement:     " (.toFixed ess-ratio 1) "x"))
        _ (println (str "  L3 time: " (.toFixed (:mean-ms l3-timing) 3) " ms"
                        ", L2 time: " (.toFixed (:mean-ms l2-timing) 3) " ms"))]

    ;; Return result map. :variance_reduction is OMITTED when L3 is exact.
    (cond-> {:label label
             :method_selection {:method (name (:method sel))
                                :reason (:reason sel)
                                :eliminated (vec (map name (:eliminated sel)))
                                :residual (vec (map name (:residual-addrs sel)))
                                :n_eliminated (count (:eliminated sel))
                                :n_residual (:n-residual sel)}
             :conjugate_pairs (mapv (fn [p] {:prior (name (:prior-addr p))
                                              :obs (name (:obs-addr p))
                                              :family (name (:family p))})
                                    conj-pairs)
             :closed_form_log_ml cf
             :l3 {:log_ml_mean (:log-ml-mean l3-results)
                  :log_ml_std (:log-ml-std l3-results)
                  :log_ml_var (:log-ml-var l3-results)
                  :log_ml_abs_error_nats l3-err
                  :ess_mean (:ess-mean l3-results)
                  :time_ms (:mean-ms l3-timing)
                  :time_std_ms (:std-ms l3-timing)}
             :l2 {:log_ml_mean (:log-ml-mean l2-results)
                  :log_ml_std (:log-ml-std l2-results)
                  :log_ml_var (:log-ml-var l2-results)
                  :log_ml_abs_error_nats l2-err
                  :ess_mean (:ess-mean l2-results)
                  :time_ms (:mean-ms l2-timing)
                  :time_std_ms (:std-ms l2-timing)}
             :ess_improvement ess-ratio
             :ground_truth ground-truth
             :n_particles n-particles
             :n_trials n-trials}
      (some? var-ratio) (assoc :variance_reduction var-ratio))))

;; =========================================================================
;; 5A: Normal-Normal (mean estimation)
;; =========================================================================

(def mean-model
  (dyn/auto-key
    (gen []
      (let [mu (trace :mu (dist/gaussian 0 10))]
        (trace :y1 (dist/gaussian mu 1))
        (trace :y2 (dist/gaussian mu 1))
        (trace :y3 (dist/gaussian mu 1))
        (trace :y4 (dist/gaussian mu 1))
        (trace :y5 (dist/gaussian mu 1))
        mu))))

;; Observations: y1=1.0, y2=1.5, y3=0.8, y4=1.2, y5=1.1
(def mean-obs
  (-> cm/EMPTY
      (cm/set-value :y1 (mx/scalar 1.0))
      (cm/set-value :y2 (mx/scalar 1.5))
      (cm/set-value :y3 (mx/scalar 0.8))
      (cm/set-value :y4 (mx/scalar 1.2))
      (cm/set-value :y5 (mx/scalar 1.1))))

;; Ground truth: Normal-Normal conjugate posterior
;; Prior: N(0, 10^2=100), Likelihood: y_i ~ N(mu, 1)
;; Posterior: N(mu_post, sigma_post^2)
;;   sigma_post^2 = 1 / (1/100 + 5/1) = 1/5.01 = 0.1996
;;   mu_post = sigma_post^2 * (0/100 + (1.0+1.5+0.8+1.2+1.1)/1) = 0.1996 * 5.6 = 1.1178
(def nn-ground-truth
  (let [sigma-prior-sq 100.0
        sigma-obs-sq 1.0
        n 5
        sum-y (+ 1.0 1.5 0.8 1.2 1.1)
        posterior-var (/ 1.0 (+ (/ 1.0 sigma-prior-sq) (/ n sigma-obs-sq)))
        posterior-mean (* posterior-var (+ (/ 0.0 sigma-prior-sq) (/ sum-y sigma-obs-sq)))]
    {:posterior-mean posterior-mean
     :posterior-var posterior-var
     :posterior-std (js/Math.sqrt posterior-var)}))

;; =========================================================================
;; 5B: Beta-Bernoulli (coin flip)
;; =========================================================================

(def coin-model
  (dyn/auto-key
    (gen []
      (let [theta (trace :theta (dist/beta-dist 2 2))]
        (trace :y1 (dist/bernoulli theta))
        (trace :y2 (dist/bernoulli theta))
        (trace :y3 (dist/bernoulli theta))
        (trace :y4 (dist/bernoulli theta))
        (trace :y5 (dist/bernoulli theta))
        theta))))

;; Observations: y1=1, y2=1, y3=0, y4=1, y5=0 (3 heads, 2 tails)
(def coin-obs
  (-> cm/EMPTY
      (cm/set-value :y1 (mx/scalar 1))
      (cm/set-value :y2 (mx/scalar 1))
      (cm/set-value :y3 (mx/scalar 0))
      (cm/set-value :y4 (mx/scalar 1))
      (cm/set-value :y5 (mx/scalar 0))))

;; Ground truth: Beta(2+3, 2+2) = Beta(5, 4)
;; E[theta] = 5/9 = 0.5556
(def bb-ground-truth
  {:posterior-alpha 5.0
   :posterior-beta 4.0
   :posterior-mean (/ 5.0 9.0)})

;; =========================================================================
;; 5C: Gamma-Poisson (count data)
;; =========================================================================

(def count-model
  (dyn/auto-key
    (gen []
      (let [rate (trace :rate (dist/gamma-dist 2 1))]
        (trace :y1 (dist/poisson rate))
        (trace :y2 (dist/poisson rate))
        (trace :y3 (dist/poisson rate))
        (trace :y4 (dist/poisson rate))
        rate))))

;; Observations: y1=3, y2=5, y3=2, y4=4 (sum=14)
(def count-obs
  (-> cm/EMPTY
      (cm/set-value :y1 (mx/array 3))
      (cm/set-value :y2 (mx/array 5))
      (cm/set-value :y3 (mx/array 2))
      (cm/set-value :y4 (mx/array 4))))

;; Ground truth: Gamma(2+14, 1+4) = Gamma(16, 5)
;; E[rate] = 16/5 = 3.2
(def gp-ground-truth
  {:posterior-shape 16.0
   :posterior-rate 5.0
   :posterior-mean (/ 16.0 5.0)})

;; =========================================================================
;; 5D: Mixed model (partial conjugacy)
;; =========================================================================

(def mixed-model
  (dyn/auto-key
    (gen []
      (let [mu-a  (trace :mu-a (dist/gaussian 0 10))
            mu-b  (trace :mu-b (dist/gaussian 0 10))
            sigma (trace :sigma (dist/gamma-dist 2 1))]
        (trace :y1 (dist/gaussian mu-a sigma))
        (trace :y2 (dist/gaussian mu-a sigma))
        (trace :y3 (dist/gaussian mu-b sigma))
        (trace :y4 (dist/gaussian mu-b sigma))
        sigma))))

;; Observations: y1=1.5, y2=2.0, y3=-0.5, y4=-1.0
(def mixed-obs
  (-> cm/EMPTY
      (cm/set-value :y1 (mx/scalar 1.5))
      (cm/set-value :y2 (mx/scalar 2.0))
      (cm/set-value :y3 (mx/scalar -0.5))
      (cm/set-value :y4 (mx/scalar -1.0))))

;; Ground truth: mu-a, mu-b are Normal-Normal conjugate (given sigma).
;; sigma is NOT conjugate (Gamma prior with Gaussian likelihood is not conjugate).
;; L3 should eliminate mu-a, mu-b and sample only sigma.
(def mixed-ground-truth
  {:conjugate-latents [:mu-a :mu-b]
   :non-conjugate-latents [:sigma]
   :note "mu-a, mu-b eliminated analytically; sigma must be sampled"})

;; =========================================================================
;; 5E: Static LinReg (full elimination)
;; =========================================================================

(def sigma-obs 1.0)
(def sigma-prior 10.0)

(def linreg-model
  (dyn/auto-key
    (gen [x1 x2 x3 x4 x5]
      (let [slope     (trace :slope (dist/gaussian 0 sigma-prior))
            intercept (trace :intercept (dist/gaussian 0 sigma-prior))]
        (trace :y1 (dist/gaussian (mx/add (mx/multiply slope x1) intercept) sigma-obs))
        (trace :y2 (dist/gaussian (mx/add (mx/multiply slope x2) intercept) sigma-obs))
        (trace :y3 (dist/gaussian (mx/add (mx/multiply slope x3) intercept) sigma-obs))
        (trace :y4 (dist/gaussian (mx/add (mx/multiply slope x4) intercept) sigma-obs))
        (trace :y5 (dist/gaussian (mx/add (mx/multiply slope x5) intercept) sigma-obs))
        slope))))

(def linreg-xs [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)
                (mx/scalar 4.0) (mx/scalar 5.0)])
(def linreg-ys [2.3 4.7 6.1 8.9 10.2])

(def linreg-obs
  (-> cm/EMPTY
      (cm/set-choice [:y1] (mx/scalar 2.3))
      (cm/set-choice [:y2] (mx/scalar 4.7))
      (cm/set-choice [:y3] (mx/scalar 6.1))
      (cm/set-choice [:y4] (mx/scalar 8.9))
      (cm/set-choice [:y5] (mx/scalar 10.2))))

;; Analytic posterior for Normal-Normal conjugate linear regression
(defn compute-analytic-posterior [xs ys]
  (let [sx  (reduce + xs)
        sx2 (reduce + (map #(* % %) xs))
        sxy (reduce + (map * xs ys))
        sy  (reduce + ys)
        n   (double (count xs))
        inv-prior (/ 1.0 (* sigma-prior sigma-prior))
        inv-obs   (/ 1.0 (* sigma-obs sigma-obs))
        p00 (+ (* sx2 inv-obs) inv-prior)
        p01 (* sx inv-obs)
        p11 (+ (* n inv-obs) inv-prior)
        det (- (* p00 p11) (* p01 p01))
        s00 (/ p11 det)
        s01 (/ (- p01) det)
        s11 (/ p00 det)]
    {:slope-mean (+ (* s00 (* sxy inv-obs)) (* s01 (* sy inv-obs)))
     :slope-std (js/Math.sqrt s00)
     :intercept-mean (+ (* s01 (* sxy inv-obs)) (* s11 (* sy inv-obs)))
     :intercept-std (js/Math.sqrt s11)}))

(def linreg-ground-truth
  (compute-analytic-posterior [1.0 2.0 3.0 4.0 5.0] linreg-ys))

;; =========================================================================
;; Main execution
;; =========================================================================

(println "\n" (apply str (repeat 70 "=")))
(println "  EXPERIMENT 5: L3 Auto-Conjugacy Showcase")
(println (apply str (repeat 70 "=")))
(println "\n  Measures variance reduction from automatic analytical elimination.")
(println "  5 conjugate families tested, 1000 IS particles per trial, 10 trials.\n")

(def n-particles 50)
(def n-trials 3)

;; ---------------------------------------------------------------------------
;; 5A: Normal-Normal
;; ---------------------------------------------------------------------------

(println "\n" (apply str (repeat 70 "=")))
(println "  5A: Normal-Normal (mean estimation)")
(println (apply str (repeat 70 "=")))
(println (str "  Ground truth posterior: N("
              (.toFixed (:posterior-mean nn-ground-truth) 4) ", "
              (.toFixed (:posterior-std nn-ground-truth) 4) "^2)"))

(def result-5a
  (run-sub-experiment "5A-Normal-Normal" mean-model [] mean-obs
                      n-particles n-trials
                      :ground-truth {:distribution "Normal"
                                     :mean (:posterior-mean nn-ground-truth)
                                     :std (:posterior-std nn-ground-truth)
                                     :var (:posterior-var nn-ground-truth)
                                     :log-ml (nn-shared-marginal [1.0 1.5 0.8 1.2 1.1] 0.0 100.0 1.0)}))

;; ---------------------------------------------------------------------------
;; 5B: Beta-Bernoulli
;; ---------------------------------------------------------------------------

(println "\n" (apply str (repeat 70 "=")))
(println "  5B: Beta-Bernoulli (coin flip)")
(println (apply str (repeat 70 "=")))
(println (str "  Ground truth posterior: Beta("
              (:posterior-alpha bb-ground-truth) ", "
              (:posterior-beta bb-ground-truth) ")"
              ", E[theta]=" (.toFixed (:posterior-mean bb-ground-truth) 4)))

(def result-5b
  (run-sub-experiment "5B-Beta-Bernoulli" coin-model [] coin-obs
                      n-particles n-trials
                      :ground-truth {:distribution "Beta"
                                     :alpha (:posterior-alpha bb-ground-truth)
                                     :beta (:posterior-beta bb-ground-truth)
                                     :mean (:posterior-mean bb-ground-truth)
                                     :log-ml (bb-marginal 3 5 2.0 2.0)}))

;; ---------------------------------------------------------------------------
;; 5C: Gamma-Poisson
;; ---------------------------------------------------------------------------

(println "\n" (apply str (repeat 70 "=")))
(println "  5C: Gamma-Poisson (count data)")
(println (apply str (repeat 70 "=")))
(println (str "  Ground truth posterior: Gamma("
              (:posterior-shape gp-ground-truth) ", "
              (:posterior-rate gp-ground-truth) ")"
              ", E[rate]=" (.toFixed (:posterior-mean gp-ground-truth) 4)))

(def result-5c
  (run-sub-experiment "5C-Gamma-Poisson" count-model [] count-obs
                      n-particles n-trials
                      :ground-truth {:distribution "Gamma"
                                     :shape (:posterior-shape gp-ground-truth)
                                     :rate (:posterior-rate gp-ground-truth)
                                     :mean (:posterior-mean gp-ground-truth)
                                     :log-ml (gp-marginal [3.0 5.0 2.0 4.0] 2.0 1.0)}))

;; ---------------------------------------------------------------------------
;; 5D: Mixed model (partial conjugacy)
;; ---------------------------------------------------------------------------

(println "\n" (apply str (repeat 70 "=")))
(println "  5D: Mixed Model (partial conjugacy)")
(println (apply str (repeat 70 "=")))
(println "  mu-a, mu-b: Normal-Normal conjugate (should be eliminated)")
(println "  sigma: Gamma prior with Gaussian likelihood (NOT conjugate)")

(def result-5d
  (run-sub-experiment "5D-Mixed" mixed-model [] mixed-obs
                      n-particles n-trials
                      :ground-truth mixed-ground-truth))

;; ---------------------------------------------------------------------------
;; 5E: Static LinReg (full elimination)
;; ---------------------------------------------------------------------------

(println "\n" (apply str (repeat 70 "=")))
(println "  5E: Static Linear Regression (full elimination)")
(println (apply str (repeat 70 "=")))
(println (str "  Ground truth: slope=" (.toFixed (:slope-mean linreg-ground-truth) 4)
              " +/- " (.toFixed (:slope-std linreg-ground-truth) 4)
              ", intercept=" (.toFixed (:intercept-mean linreg-ground-truth) 4)
              " +/- " (.toFixed (:intercept-std linreg-ground-truth) 4)))

;; 5E uses args, not []
(println (str "\n" (apply str (repeat 60 "-"))))
(println "  5E-LinReg")
(println (apply str (repeat 60 "-")))

(let [model-ak linreg-model
      sel (ms/select-method model-ak linreg-obs)]
  (println (str "  Method: " (:method sel) " (" (:reason sel) ")"))
  (println (str "  Eliminated: " (vec (:eliminated sel))
                " (" (count (:eliminated sel)) " addrs)"))
  (println (str "  Residual: " (vec (:residual-addrs sel))
                " (" (:n-residual sel) " addrs)"))
  (let [conj-pairs (get-in model-ak [:schema :conjugate-pairs])]
    (when (seq conj-pairs)
      (doseq [pair conj-pairs]
        (println (str "    Pair: " (:prior-addr pair) " -> " (:obs-addr pair)
                      " [" (:family pair) "]"))))))

;; L3 timing
(def l3-linreg-timing
  (benchmark "5E-LinReg-L3"
             (fn [] (generate-weight linreg-model linreg-xs linreg-obs))
             :warmup-n 3 :outer-n 3 :inner-n 5))

;; L3 trials
(println (str "\n  Running L3, " n-particles " particles x " n-trials " trials..."))
(def l3-linreg-results
  (let [trials (mapv (fn [t]
                       (run-is-trial linreg-model linreg-xs linreg-obs n-particles (+ 3000 t)))
                     (range n-trials))
        log-mls (mapv :log-ml trials)
        esses (mapv :ess trials)]
    {:log-mls log-mls :esses esses
     :log-ml-mean (mean log-mls) :log-ml-std (std log-mls)
     :log-ml-var (variance log-mls)
     :ess-mean (mean esses)}))
(println (str "  L3 log-ML: " (.toFixed (:log-ml-mean l3-linreg-results) 6)
              " +/- " (.toFixed (:log-ml-std l3-linreg-results) 6)))
(println (str "  L3 ESS:    " (.toFixed (:ess-mean l3-linreg-results) 1)
              " / " n-particles))

;; L2 timing
(def l2-linreg-model (strip-analytical linreg-model))
(def l2-linreg-timing
  (benchmark "5E-LinReg-L2"
             (fn [] (generate-weight (dyn/auto-key l2-linreg-model) linreg-xs linreg-obs))
             :warmup-n 3 :outer-n 3 :inner-n 5))

;; L2 trials
(println (str "\n  Running L2, " n-particles " particles x " n-trials " trials..."))
(def l2-linreg-results
  (let [trials (mapv (fn [t]
                       (run-is-trial (dyn/auto-key l2-linreg-model) linreg-xs linreg-obs
                                     n-particles (+ 4000 t)))
                     (range n-trials))
        log-mls (mapv :log-ml trials)
        esses (mapv :ess trials)]
    {:log-mls log-mls :esses esses
     :log-ml-mean (mean log-mls) :log-ml-std (std log-mls)
     :log-ml-var (variance log-mls)
     :ess-mean (mean esses)}))
(println (str "  L2 log-ML: " (.toFixed (:log-ml-mean l2-linreg-results) 6)
              " +/- " (.toFixed (:log-ml-std l2-linreg-results) 6)))
(println (str "  L2 ESS:    " (.toFixed (:ess-mean l2-linreg-results) 1)
              " / " n-particles))

;; Exactness (headline) + variance reduction (corollary, omitted when exact)
(def linreg-cf (linreg-marginal [1.0 2.0 3.0 4.0 5.0] linreg-ys
                                (* sigma-prior sigma-prior) (* sigma-obs sigma-obs)))
(def linreg-l3-err (js/Math.abs (- (:log-ml-mean l3-linreg-results) linreg-cf)))
(def linreg-l2-err (js/Math.abs (- (:log-ml-mean l2-linreg-results) linreg-cf)))
(def linreg-l3-exact? (< (:log-ml-var l3-linreg-results) 1e-20))
(def linreg-var-ratio
  (when-not linreg-l3-exact?
    (/ (:log-ml-var l2-linreg-results) (:log-ml-var l3-linreg-results))))
(def linreg-ess-ratio
  (/ (:ess-mean l3-linreg-results) (max (:ess-mean l2-linreg-results) 0.01)))

(println (str "\n  --- Summary ---"))
(println (str "  |L3 log-ML − closed form|: " (.toExponential linreg-l3-err 3)
              " nats  (exactness — the headline; closed form = "
              (.toFixed linreg-cf 4) ")"))
(println (str "  |L2 log-ML − closed form|: " (.toFixed linreg-l2-err 4) " nats"))
(println (str "  Variance reduction:  " (if linreg-l3-exact? "n/a (L3 exact, var=0)"
                                            (str (.toFixed linreg-var-ratio 1) "x"))))
(println (str "  ESS improvement:     " (.toFixed linreg-ess-ratio 1) "x"))

(def result-5e
  (cond-> {:label "5E-LinReg"
           :method_selection (let [sel (ms/select-method linreg-model linreg-obs)]
                               {:method (name (:method sel))
                                :reason (:reason sel)
                                :eliminated (vec (map name (:eliminated sel)))
                                :residual (vec (map name (:residual-addrs sel)))
                                :n_eliminated (count (:eliminated sel))
                                :n_residual (:n-residual sel)})
           :conjugate_pairs (mapv (fn [p] {:prior (name (:prior-addr p))
                                            :obs (name (:obs-addr p))
                                            :family (name (:family p))})
                                  (get-in linreg-model [:schema :conjugate-pairs]))
           :closed_form_log_ml linreg-cf
           :l3 {:log_ml_mean (:log-ml-mean l3-linreg-results)
                :log_ml_std (:log-ml-std l3-linreg-results)
                :log_ml_var (:log-ml-var l3-linreg-results)
                :log_ml_abs_error_nats linreg-l3-err
                :ess_mean (:ess-mean l3-linreg-results)
                :time_ms (:mean-ms l3-linreg-timing)
                :time_std_ms (:std-ms l3-linreg-timing)}
           :l2 {:log_ml_mean (:log-ml-mean l2-linreg-results)
                :log_ml_std (:log-ml-std l2-linreg-results)
                :log_ml_var (:log-ml-var l2-linreg-results)
                :log_ml_abs_error_nats linreg-l2-err
                :ess_mean (:ess-mean l2-linreg-results)
                :time_ms (:mean-ms l2-linreg-timing)
                :time_std_ms (:std-ms l2-linreg-timing)}
           :ess_improvement linreg-ess-ratio
           :ground_truth linreg-ground-truth
           :n_particles n-particles
           :n_trials n-trials}
    (some? linreg-var-ratio) (assoc :variance_reduction linreg-var-ratio)))

;; =========================================================================
;; Summary table
;; =========================================================================

(println "\n\n" (apply str (repeat 70 "=")))
(println "         CONJUGACY RESULTS SUMMARY")
(println (apply str (repeat 70 "=")))

(def all-results [result-5a result-5b result-5c result-5d result-5e])

(println "\n| Experiment | Family | Elim | |L3−cf| (nats) | |L2−cf| (nats) | ESS Ratio |")
(println "|------------|--------|------|----------------|----------------|-----------|")
(doseq [r all-results]
  (let [families (if (seq (:conjugate_pairs r))
                   (clojure.string/join "," (distinct (map :family (:conjugate_pairs r))))
                   "none")
        n-elim (get-in r [:method_selection :n_eliminated])
        l3e (get-in r [:l3 :log_ml_abs_error_nats])
        l2e (get-in r [:l2 :log_ml_abs_error_nats])]
    (println (str "| " (.padEnd (:label r) 10)
                  " | " (.padEnd families 6)
                  " | " (.padStart (str n-elim) 4)
                  " | " (.padStart (if l3e (.toExponential l3e 2) "n/a (partial)") 14)
                  " | " (.padStart (if l2e (.toFixed l2e 4) "n/a") 14)
                  " | " (.padStart (.toFixed (:ess_improvement r) 1) 9)
                  " |"))))

(println)

;; Timing summary
(println "| Experiment | L3 time (ms) | L2 time (ms) |")
(println "|------------|--------------|--------------|")
(doseq [r all-results]
  (println (str "| " (.padEnd (:label r) 10)
                " | " (.padStart (.toFixed (get-in r [:l3 :time_ms]) 3) 12)
                " | " (.padStart (.toFixed (get-in r [:l2 :time_ms]) 3) 12)
                " |")))

;; =========================================================================
;; Write JSON results
;; =========================================================================

(write-json "data.json"
  {:experiment "conjugacy"
   :description "L3 Auto-Conjugacy Showcase: variance reduction from analytical elimination"
   :timestamp (.toISOString (js/Date.))
   :hardware {:platform "macOS" :chip "Apple Silicon" :gpu "Metal"}
   :config {:n_particles n-particles :n_trials n-trials}
   :sub_experiments
   {:exp5a result-5a
    :exp5b result-5b
    :exp5c result-5c
    :exp5d result-5d
    :exp5e result-5e}
   :summary
   {:families_tested ["normal-normal" "beta-bernoulli" "gamma-poisson" "mixed" "linreg"]
    ;; Exactness is the headline: |L3 log-ML − closed form| in nats. Every family
    ;; with a closed form (all but the partial-conjugacy 5D) eliminates to the
    ;; float32 floor. This replaces the old >1000x / Infinity / variance=0 artifact
    ;; (a divide-by-~0 against a different model).
    :log_ml_abs_error_nats
    (mapv (fn [r] {:label (:label r)
                   :l3 (get-in r [:l3 :log_ml_abs_error_nats])
                   :l2 (get-in r [:l2 :log_ml_abs_error_nats])})
          all-results)
    :max_l3_abs_error_nats
    (apply max (keep #(get-in % [:l3 :log_ml_abs_error_nats]) all-results))
    ;; Variance reduction is reported ONLY where finite (partial conjugacy / 5D).
    ;; Where L3 is exact its log-ML variance is 0, so the ratio is undefined and
    ;; is omitted rather than reported as Infinity.
    :variance_reductions
    (vec (keep (fn [r] (when-let [vr (:variance_reduction r)]
                         {:label (:label r) :ratio vr}))
               all-results))
    :ess_improvements (mapv (fn [r] {:label (:label r) :ratio (:ess_improvement r)})
                            all-results)
    :mean_ess_improvement
    (/ (reduce + (map :ess_improvement all-results)) (count all-results))}})

(println "\nExperiment 5 complete.")
