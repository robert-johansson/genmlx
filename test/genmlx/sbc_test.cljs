(ns genmlx.sbc-test
  "Simulation-Based Calibration (SBC) test suite.
   Talts et al. 2018: for each sim, sample theta* from prior, generate data,
   run inference, compute rank of theta* among posterior samples.
   If inference is correct, ranks ~ Uniform(0, L).

   Configuration via environment variables:
     SBC_N=500   Number of repetitions (default 500)
     SBC_L=200   Posterior samples per repetition (default 200)

   NUTS excluded: triggers Bun segfault (Bun bug, not inference bug).
   HMC validates the same gradient-based math."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.importance :as is]
            ["fs" :as fs])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ── Configuration (from env vars, with defaults) ─────────────────────────

(def N (js/parseInt (or (aget js/process.env "SBC_N") "500")))
(def L (js/parseInt (or (aget js/process.env "SBC_L") "200")))
(def N-BINS (if (< N 100) 5 10))
(def ALPHA 0.01)
(def MAX-FAIL-RATE 0.05)

;; ── Statistical utilities ────────────────────────────────────────────────

(defn compute-rank
  "Count how many posterior samples are strictly less than the true value."
  [true-val samples]
  (count (filter #(< % true-val) samples)))

(def chi-sq-critical
  "Chi-squared critical values at alpha=0.01."
  {4 13.28, 9 21.67, 14 29.14, 19 36.19})

(defn chi-squared-uniformity
  "Chi-squared goodness-of-fit test for uniformity of ranks.
   Returns {:statistic :pass? :df :critical}."
  [ranks n-sims]
  (let [bins (mapv #(js/Math.floor (/ (* % N-BINS) (inc L))) ranks)
        counts (reduce (fn [acc b] (update acc b (fnil inc 0))) {} bins)
        expected (/ n-sims N-BINS)
        df (dec N-BINS)
        critical (get chi-sq-critical df 21.67)
        all-bins (reduce (fn [m b] (if (contains? m b) m (assoc m b 0)))
                         counts (range N-BINS))
        statistic (reduce-kv
                   (fn [sum _bin observed]
                     (+ sum (/ (* (- observed expected) (- observed expected)) expected)))
                   0.0 all-bins)]
    {:statistic statistic :pass? (< statistic critical) :df df :critical critical}))

(defn ecdf-ks-uniformity
  "Kolmogorov-Smirnov test for uniformity of ranks.
   Ranks should be Uniform(0, L). Returns {:statistic :pass? :critical}.
   Alpha parameter allows Bonferroni correction."
  [ranks n-sims alpha]
  (let [normalized (sort (mapv #(/ % (inc L)) ranks))
        n (count normalized)
        statistic (reduce-kv
                   (fn [max-d i u]
                     (let [f-above (/ (inc i) n)
                           f-below (/ i n)]
                       (max max-d
                            (js/Math.abs (- f-above u))
                            (js/Math.abs (- f-below u)))))
                   0.0 (vec normalized))
        ;; KS critical: c(alpha)/sqrt(N), c(alpha) = sqrt(-0.5*ln(alpha/2))
        c-alpha (js/Math.sqrt (* -0.5 (js/Math.log (/ alpha 2))))
        critical (/ c-alpha (js/Math.sqrt n-sims))]
    {:statistic statistic :pass? (< statistic critical) :critical critical}))

;; ── Parameter extraction ─────────────────────────────────────────────────

(defn extract-observations
  "Extract observation values from a simulated trace into a choicemap."
  [trace obs-addrs]
  (reduce (fn [obs addr]
            (let [sub (cm/get-submap (:choices trace) addr)
                  v (cm/get-value sub)]
              (mx/eval! v)
              (cm/set-choice obs [addr] v)))
          cm/EMPTY obs-addrs))

(defn extract-true-params
  "Extract true parameter values from a simulated trace as JS numbers."
  [trace param-addrs]
  (mapv (fn [addr]
          (let [sub (cm/get-submap (:choices trace) addr)
                v (cm/get-value sub)]
            (mx/eval! v)
            (mx/item v)))
        param-addrs))

;; ── Inference dispatch ───────────────────────────────────────────────────

(defn make-extractor
  "Build extractor: (sample, param-idx) -> JS number."
  [param-addrs all-addrs]
  (let [index-map (into {} (map-indexed (fn [i a] [a i]) all-addrs))]
    (fn [sample param-idx]
      (let [addr (nth param-addrs param-idx)
            arr-idx (get index-map addr)]
        (if (sequential? sample)
          (nth sample arr-idx)
          sample)))))

(defn resample-from-is
  "Resample L unweighted samples from IS weighted traces.
   Returns vector of trace-like maps for extractor compatibility."
  [is-result n-samples]
  (let [log-weights (mapv (fn [w] (mx/eval! w) (mx/item w)) (:log-weights is-result))
        max-w (apply max log-weights)
        weights (mapv #(js/Math.exp (- % max-w)) log-weights)
        sum-w (reduce + weights)
        norm-w (mapv #(/ % sum-w) weights)
        ;; Build CDF for inverse-transform sampling
        cdf (reductions + norm-w)
        traces (:traces is-result)]
    (vec (repeatedly n-samples
                     (fn []
                       (let [u (js/Math.random)
                             idx (or (first (keep-indexed (fn [i c] (when (>= c u) i)) cdf))
                                     (dec (count traces)))]
                         (nth traces idx)))))))

(defn run-inference
  "Run inference and return {:samples vector :extractor fn}."
  [algo-key {:keys [model args param-addrs cmh-opts hmc-opts mh-opts is-opts]} observations]
  (case algo-key
    :cmh
    (let [opts (merge {:samples L :burn (max 100 (quot L 2)) :compile? true :device :cpu} cmh-opts)
          samples (mcmc/compiled-mh opts model args observations)
          addrs (:addresses cmh-opts)]
      {:samples samples :extractor (make-extractor param-addrs addrs)})

    :mh
    (let [opts (merge {:samples L :burn (max 200 L)} mh-opts)
          traces (mcmc/mh opts model args observations)]
      {:samples traces
       :extractor (fn [sample param-idx]
                    (let [addr (nth param-addrs param-idx)
                          sub (cm/get-submap (:choices sample) addr)
                          v (cm/get-value sub)]
                      (mx/eval! v)
                      (mx/item v)))})

    :hmc
    (let [opts (merge {:samples L :burn (max 100 (quot L 2)) :compile? true :device :cpu} hmc-opts)
          samples (mcmc/hmc opts model args observations)
          addrs (:addresses hmc-opts)]
      {:samples samples :extractor (make-extractor param-addrs addrs)})

    :is
    (let [n-particles (or (:particles is-opts) (* L 20))
          result (is/importance-sampling {:samples n-particles} model args observations)
          resampled (resample-from-is result L)]
      {:samples resampled
       :extractor (fn [sample param-idx]
                    (let [addr (nth param-addrs param-idx)
                          sub (cm/get-submap (:choices sample) addr)
                          v (cm/get-value sub)]
                      (mx/eval! v)
                      (mx/item v)))})))

;; ── Model definitions ────────────────────────────────────────────────────

(def single-gaussian
  {:name "single-gaussian"
   :model (dyn/auto-key (gen []
                             (let [mu (trace :mu (dist/gaussian 0 2))]
                               (trace :obs (dist/gaussian mu 1))
                               mu)))
   :args []
   :param-addrs [:mu], :obs-addrs [:obs]
   :algorithms [:cmh :hmc :is]
   :cmh-opts {:addresses [:mu] :proposal-std 1.0}
   :hmc-opts {:addresses [:mu] :step-size 0.1 :leapfrog-steps 10}
   :is-opts {}})

(def two-gaussians
  {:name "two-gaussians"
   :model (dyn/auto-key (gen []
                             (let [a (trace :a (dist/gaussian 0 2))
                                   b (trace :b (dist/gaussian 0 2))]
                               (trace :obs-a (dist/gaussian a 1))
                               (trace :obs-b (dist/gaussian b 1))
                               [a b])))
   :args []
   :param-addrs [:a :b], :obs-addrs [:obs-a :obs-b]
   :algorithms [:cmh :hmc :is]
   :cmh-opts {:addresses [:a :b] :proposal-std 0.7}
   :hmc-opts {:addresses [:a :b] :step-size 0.1 :leapfrog-steps 10}
   :is-opts {}})

(def gaussian-multi-obs
  {:name "gaussian-multi-obs"
   :model (dyn/auto-key (gen []
                             (let [mu (trace :mu (dist/gaussian 0 2))]
                               (trace :y0 (dist/gaussian mu 1))
                               (trace :y1 (dist/gaussian mu 1))
                               (trace :y2 (dist/gaussian mu 1))
                               mu)))
   :args []
   :param-addrs [:mu], :obs-addrs [:y0 :y1 :y2]
   :algorithms [:cmh :hmc :is]
   :cmh-opts {:addresses [:mu] :proposal-std 0.5}
   :hmc-opts {:addresses [:mu] :step-size 0.1 :leapfrog-steps 10}
   :is-opts {}})

;; Exponential prior (mx/eval! in body — standard MH for positivity)
(def exponential-spec
  {:name "exponential"
   :model (dyn/auto-key (gen []
                             (let [rate (trace :rate (dist/gamma-dist (mx/scalar 2) (mx/scalar 1)))]
                               (mx/eval! rate)
                               (let [rate-val (mx/item rate)]
                                 (trace :obs (dist/exponential (mx/scalar rate-val)))
                                 rate-val))))
   :args []
   :param-addrs [:rate], :obs-addrs [:obs]
   :algorithms [:mh]
   :mh-opts {:selection (sel/select :rate)}})

;; Coin flip (mx/item in body — standard MH for bounded support)
(def coin-flip
  {:name "coin-flip"
   :model (dyn/auto-key (gen []
                             (let [p-val (trace :p (dist/uniform 0.01 0.99))]
                               (mx/eval! p-val)
                               (let [p-num (mx/item p-val)]
                                 (doseq [i (range 5)]
                                   (trace (keyword (str "flip" i)) (dist/bernoulli p-num)))
                                 p-num))))
   :args []
   :param-addrs [:p], :obs-addrs [:flip0 :flip1 :flip2 :flip3 :flip4]
   :algorithms [:mh]
   :mh-opts {:selection (sel/select :p)}})

;; Linear regression (correlated params)
(def linear-regression
  {:name "linear-regression"
   :model (dyn/auto-key (gen [xs]
                             (let [slope (trace :slope (dist/gaussian 0 2))
                                   intercept (trace :intercept (dist/gaussian 0 2))]
                               (doseq [[j x] (map-indexed vector xs)]
                                 (trace (keyword (str "y" j))
                                        (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                                               intercept) 1)))
                               [slope intercept])))
   :args [[(mx/scalar 0.0) (mx/scalar 1.0) (mx/scalar 2.0)]]
   :param-addrs [:slope :intercept], :obs-addrs [:y0 :y1 :y2]
   :algorithms [:hmc :is]
   :hmc-opts {:addresses [:slope :intercept] :step-size 0.05 :leapfrog-steps 15}
   :is-opts {}})

;; Hierarchical model (correlated params)
(def hierarchical
  {:name "hierarchical"
   :model (dyn/auto-key (gen []
                             (let [mu (trace :mu (dist/gaussian 0 2))
                                   x (trace :x (dist/gaussian mu 1))]
                               (trace :obs (dist/gaussian x 1))
                               x)))
   :args []
   :param-addrs [:x], :obs-addrs [:obs]
   :algorithms [:hmc]
   :hmc-opts {:addresses [:mu :x] :step-size 0.1 :leapfrog-steps 10}})

;; Beta-Bernoulli (conjugate — analytical posterior known)
(def beta-bernoulli
  {:name "beta-bernoulli"
   :model (dyn/auto-key (gen []
                             (let [p (trace :p (dist/beta-dist (mx/scalar 2) (mx/scalar 2)))]
                               (mx/eval! p)
                               (let [p-num (mx/item p)]
                                 (doseq [i (range 10)]
                                   (trace (keyword (str "x" i)) (dist/bernoulli p-num)))
                                 p-num))))
   :args []
   :param-addrs [:p]
   :obs-addrs [:x0 :x1 :x2 :x3 :x4 :x5 :x6 :x7 :x8 :x9]
   :algorithms [:mh]
   :mh-opts {:selection (sel/select :p)}})

;; Gamma-Poisson (conjugate — analytical posterior known)
(def gamma-poisson
  {:name "gamma-poisson"
   :model (dyn/auto-key (gen []
                             (let [lam (trace :lambda (dist/gamma-dist (mx/scalar 3) (mx/scalar 2)))]
                               (mx/eval! lam)
                               (let [lam-num (mx/item lam)]
                                 (doseq [i (range 5)]
                                   (trace (keyword (str "y" i)) (dist/poisson (mx/scalar lam-num))))
                                 lam-num))))
   :args []
   :param-addrs [:lambda]
   :obs-addrs [:y0 :y1 :y2 :y3 :y4]
   :algorithms [:mh]
   :mh-opts {:selection (sel/select :lambda)}})

;; ── SBC harness ──────────────────────────────────────────────────────────

(defn run-sbc-single
  "Run one SBC simulation. Returns vector of ranks (one per param)."
  [model-spec algo-key]
  (mx/tidy
   (fn []
     (let [{:keys [model args param-addrs obs-addrs]} model-spec
           trace (p/simulate model args)
           true-params (extract-true-params trace param-addrs)
           observations (extract-observations trace obs-addrs)
           {:keys [samples extractor]} (run-inference algo-key model-spec observations)
           ranks (mapv
                  (fn [param-idx]
                    (let [true-val (nth true-params param-idx)
                          posterior (mapv #(extractor % param-idx) samples)]
                      (compute-rank true-val posterior)))
                  (range (count param-addrs)))]
       ranks))))

(defn run-sbc
  "Run N SBC simulations. Returns per-param results with ranks and tests.
   alpha: significance level (use Bonferroni-corrected value)."
  [model-spec algo-key n-sims alpha]
  (let [{:keys [param-addrs]} model-spec
        n-params (count param-addrs)
        ;; Worst-case rank for failed reps: L (maximum) — biases toward
        ;; detecting broken inference rather than hiding it.
        worst-rank L
        all-ranks (loop [i 0, ranks-acc (vec (repeat n-params [])), failures 0]
                    (if (>= i n-sims)
                      (if (> (/ failures (max 1 n-sims)) MAX-FAIL-RATE)
                        (do (println (str "  FAIL: >" (* 100 MAX-FAIL-RATE)
                                          "% simulations failed ("
                                          failures "/" n-sims ")"))
                            nil)
                        (do (when (pos? failures)
                              (println (str "  " failures "/" n-sims " failures ("
                                            (.toFixed (* 100 (/ failures n-sims)) 1)
                                            "%) — worst-case rank assigned")))
                            ranks-acc))
                      (let [_ (do
                                (when (zero? (mod i 50))
                                  (let [rss-mb (/ (.-rss (js/process.memoryUsage)) 1048576)]
                                    (println (str "  rep " i "/" n-sims
                                                  " (" failures " failures)"
                                                  " RSS=" (.toFixed rss-mb 0) "MB"))))
                                (mx/clear-cache!)
                                (mx/sweep-dead-arrays!)
                                (when (.-gc js/globalThis) (.gc js/globalThis)))
                            result (try
                                     (run-sbc-single model-spec algo-key)
                                     (catch :default e
                                       ;; Log every failure (not just every 10th)
                                       (println (str "    [sim " i " failed: " (.-message e) "]"))
                                       nil))]
                        (if result
                          (recur (inc i)
                                 (mapv (fn [j] (conj (nth ranks-acc j) (nth result j)))
                                       (range n-params))
                                 failures)
                          ;; Failed rep: assign worst-case rank to all params
                          (recur (inc i)
                                 (mapv (fn [j] (conj (nth ranks-acc j) worst-rank))
                                       (range n-params))
                                 (inc failures))))))]
    (when all-ranks
      (mapv (fn [j]
              (let [param (nth param-addrs j)
                    ranks (nth all-ranks j)
                    chi2 (chi-squared-uniformity ranks (count ranks))
                    ecdf (ecdf-ks-uniformity ranks (count ranks) alpha)]
                {:param param :ranks ranks :chi2 chi2 :ecdf ecdf}))
            (range n-params)))))

;; ── All models ───────────────────────────────────────────────────────────

(def all-models
  [single-gaussian
   two-gaussians
   gaussian-multi-obs
   exponential-spec
   coin-flip
   linear-regression
   hierarchical
   beta-bernoulli
   gamma-poisson])

;; ── Runner ───────────────────────────────────────────────────────────────

;; Cap Metal cache to 2GB to prevent RSS blowup (machine has 36GB)
(mx/set-cache-limit! (* 2 1024 1024 1024))

;; Count total parameter tests for Bonferroni correction
(def n-total-tests
  (reduce + (for [m all-models a (:algorithms m)]
              (count (:param-addrs m)))))

;; Bonferroni-corrected alpha: alpha / n_tests
(def ALPHA-CORRECTED (/ ALPHA n-total-tests))

(println (str "=== Simulation-Based Calibration (SBC) Tests ==="))
(println (str "N=" N ", L=" L ", " N-BINS " bins, alpha=" ALPHA
              " (Bonferroni: " (.toFixed ALPHA-CORRECTED 5)
              " for " n-total-tests " tests)"))
(println (str "Max failure rate: " (* 100 MAX-FAIL-RATE) "%"))
(println (str "Metal cache limit: 2GB"))

(def all-results (atom []))
(def summary (atom {:pass 0 :fail 0}))
(def results-path "dev/sbc_results.json")

(defn write-results!
  "Write current results to JSON incrementally."
  []
  (let [{:keys [pass fail]} @summary
        output (clj->js {:config {:N N :L L :N_BINS N-BINS :ALPHA ALPHA}
                         :results @all-results
                         :summary {:pass pass :fail fail :total (+ pass fail)
                                   :complete? false}})
        json (js/JSON.stringify output nil 2)]
    (fs/writeFileSync results-path json)))

(def all-combos (vec (for [m all-models a (:algorithms m)] [m a])))
(def total-combos (count all-combos))
(def combo-idx (atom 0))
(def run-start (js/Date.now))

(doseq [[model-spec algo-key] all-combos]
  (swap! combo-idx inc)
  (let [start (js/Date.now)
        elapsed-total (/ (- start run-start) 1000)
        eta (if (> @combo-idx 1)
              (let [avg (/ elapsed-total (dec @combo-idx))]
                (str " ~" (.toFixed (/ (* avg (- total-combos (dec @combo-idx))) 3600) 1) "h left"))
              "")
        _ (println (str "\n-- [" @combo-idx "/" total-combos "] SBC: "
                        (:name model-spec) " x " (name algo-key)
                        " (" (.toFixed (/ elapsed-total 60) 0) "min elapsed" eta ")" " --"))
        param-results (run-sbc model-spec algo-key N ALPHA-CORRECTED)
        elapsed (/ (- (js/Date.now) start) 1000)]
    (if param-results
      (do
        (doseq [{:keys [param chi2 ecdf]} param-results]
          (let [pass? (and (:pass? chi2) (:pass? ecdf))]
            (println (str "  " (if pass? "PASS" "FAIL") ": " (name param)
                          "  chi2=" (.toFixed (:statistic chi2) 2)
                          " (crit=" (:critical chi2) ")"
                          "  ks=" (.toFixed (:statistic ecdf) 3)
                          " (crit=" (.toFixed (:critical ecdf) 3) ")"))
            (swap! summary update (if pass? :pass :fail) inc)))
        (swap! all-results conj
               {:model (:name model-spec)
                :algorithm (name algo-key)
                :params (mapv (fn [{:keys [param ranks chi2 ecdf]}]
                                {:name (name param)
                                 :ranks ranks
                                 :chi2 (dissoc chi2 :pass?)
                                 :ecdf (dissoc ecdf :pass?)
                                 :pass? (and (:pass? chi2) (:pass? ecdf))})
                              param-results)
                :elapsed_s elapsed}))
      (swap! summary update :fail inc))
    (println (str "  [" (.toFixed elapsed 1) "s]"))
    ;; Write after each combo so crashes don't lose everything
    (write-results!)
    ;; Aggressive cleanup between combos — multiple GC passes to reclaim
    ;; JS heap accumulated during 500 reps of MCMC inference
    (mx/clear-cache!)
    (mx/sweep-dead-arrays!)
    (dotimes [_ 3]
      (when (.-gc js/globalThis) (.gc js/globalThis))
      (mx/sweep-dead-arrays!))
    (mx/compile-clear-cache!)
    (let [rss-mb (/ (.-rss (js/process.memoryUsage)) 1048576)]
      (println (str "  [RSS: " (.toFixed rss-mb 0) "MB, Metal active: "
                    (.toFixed (/ (mx/get-active-memory) 1048576) 0) "MB, cache: "
                    (.toFixed (/ (mx/get-cache-memory) 1048576) 0) "MB]")))))

;; ── Final write ──────────────────────────────────────────────────────────

(let [{:keys [pass fail]} @summary
      total (+ pass fail)
      output (clj->js {:config {:N N :L L :N_BINS N-BINS :ALPHA ALPHA}
                       :results @all-results
                       :summary {:pass pass :fail fail :total total
                                 :complete? true}})
      json (js/JSON.stringify output nil 2)]
  (fs/writeFileSync results-path json)
  (println (str "\n=== SBC Summary: " pass "/" total " passed ==="))
  (println (str "Results written to " results-path))
  (js/process.exit (if (zero? fail) 0 1)))
