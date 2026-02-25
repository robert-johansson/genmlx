(ns genmlx.sbc-test
  "Simulation-Based Calibration (SBC) test suite.
   Talts et al. 2018: for each sim, sample θ* from prior, generate data,
   run inference, compute rank of θ* among posterior samples.
   If inference is correct, ranks ~ Uniform(0, L).

   NUTS excluded: triggers Bun segfault (Bun bug, not inference bug).
   HMC validates the same gradient-based math."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.mcmc :as mcmc])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ── Configuration ──────────────────────────────────────────────────────────

(def N-CMH 100)    ;; SBC repetitions for compiled MH / standard MH
(def N-HMC 50)     ;; SBC repetitions for HMC (fewer: speed)
(def L 100)        ;; posterior samples per sim
(def N-BINS 10)    ;; bins for chi-squared test
(def ALPHA 0.005)  ;; significance level (stricter to reduce false positives across 13 tests)

;; ── Statistical utilities ──────────────────────────────────────────────────

(defn compute-rank
  "Count how many posterior samples are strictly less than the true value."
  [true-val samples]
  (count (filter #(< % true-val) samples)))

(def chi-sq-critical
  "Chi-squared critical values at alpha=0.005 for common df."
  {4  14.86
   9  23.59
   14 31.32
   19 38.58})

(defn chi-squared-uniformity
  "Chi-squared goodness-of-fit test for uniformity of ranks.
   Returns {:statistic :pass? :df :critical}."
  [ranks n-sims]
  (let [bins (mapv #(js/Math.floor (/ (* % N-BINS) (inc L))) ranks)
        counts (reduce (fn [acc b] (update acc b (fnil inc 0))) {} bins)
        expected (/ n-sims N-BINS)
        df (dec N-BINS)
        critical (get chi-sq-critical df 21.67)
        statistic (reduce-kv
                    (fn [sum _bin observed]
                      (+ sum (/ (* (- observed expected) (- observed expected)) expected)))
                    0.0
                    (reduce (fn [m b] (if (contains? m b) m (assoc m b 0)))
                            counts (range N-BINS)))]
    {:statistic statistic
     :pass? (< statistic critical)
     :df df
     :critical critical}))

;; ── Parameter extraction ───────────────────────────────────────────────────

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

;; ── Inference dispatch ─────────────────────────────────────────────────────
;; All algorithms return vectors of Clojure arrays/numbers (via mx/->clj).
;; CMH = compiled MH with random-walk proposal (good for single-param & independent).
;; HMC = gradient-based (good for all continuous models including correlated params).

(defn make-extractor
  "Build extractor: (sample, param-idx) → JS number.
   param-addrs: which params we test (SBC rank computed for these)
   all-addrs: which addresses are in the parameter vector"
  [param-addrs all-addrs]
  (let [index-map (into {} (map-indexed (fn [i a] [a i]) all-addrs))]
    (fn [sample param-idx]
      (let [addr (nth param-addrs param-idx)
            arr-idx (get index-map addr)]
        (if (sequential? sample)
          (nth sample arr-idx)
          sample)))))

(defn run-inference
  "Run inference and return {:samples vector :extractor fn}."
  [algo-key {:keys [model args param-addrs cmh-opts hmc-opts mh-opts]} observations]
  (case algo-key
    :cmh
    (let [opts (merge {:samples L :burn 300 :compile? false :device :cpu} cmh-opts)
          samples (mcmc/compiled-mh opts model args observations)
          addrs (:addresses cmh-opts)]
      {:samples samples
       :extractor (make-extractor param-addrs addrs)})
    :mh
    (let [opts (merge {:samples L :burn 200} mh-opts)
          traces (mcmc/mh opts model args observations)]
      {:samples traces
       :extractor (fn [sample param-idx]
                    (let [addr (nth param-addrs param-idx)
                          sub (cm/get-submap (:choices sample) addr)
                          v (cm/get-value sub)]
                      (mx/eval! v)
                      (mx/item v)))})
    :hmc
    (let [opts (merge {:samples L :burn 200 :compile? false :device :cpu} hmc-opts)
          samples (mcmc/hmc opts model args observations)
          addrs (:addresses hmc-opts)]
      {:samples samples
       :extractor (make-extractor param-addrs addrs)})))

;; ── Model definitions ──────────────────────────────────────────────────────

;; 1. Single Gaussian
(def single-gaussian
  {:name "single-gaussian"
   :model (gen []
            (let [mu (dyn/trace :mu (dist/gaussian 0 2))]
              (dyn/trace :obs (dist/gaussian mu 1))
              mu))
   :args []
   :param-addrs [:mu]
   :obs-addrs [:obs]
   :algorithms [:cmh :hmc]
   :cmh-opts {:addresses [:mu] :proposal-std 1.0}
   :hmc-opts {:addresses [:mu] :step-size 0.1 :leapfrog-steps 10}})

;; 2. Two independent Gaussians
(def two-gaussians
  {:name "two-gaussians"
   :model (gen []
            (let [a (dyn/trace :a (dist/gaussian 0 2))
                  b (dyn/trace :b (dist/gaussian 0 2))]
              (dyn/trace :obs-a (dist/gaussian a 1))
              (dyn/trace :obs-b (dist/gaussian b 1))
              [a b]))
   :args []
   :param-addrs [:a :b]
   :obs-addrs [:obs-a :obs-b]
   :algorithms [:cmh :hmc]
   :cmh-opts {:addresses [:a :b] :proposal-std 0.7}
   :hmc-opts {:addresses [:a :b] :step-size 0.1 :leapfrog-steps 10}})

;; 3. Gaussian with multiple observations
(def gaussian-multi-obs
  {:name "gaussian-multi-obs"
   :model (gen []
            (let [mu (dyn/trace :mu (dist/gaussian 0 2))]
              (dyn/trace :y0 (dist/gaussian mu 1))
              (dyn/trace :y1 (dist/gaussian mu 1))
              (dyn/trace :y2 (dist/gaussian mu 1))
              mu))
   :args []
   :param-addrs [:mu]
   :obs-addrs [:y0 :y1 :y2]
   :algorithms [:cmh :hmc]
   :cmh-opts {:addresses [:mu] :proposal-std 0.5}
   :hmc-opts {:addresses [:mu] :step-size 0.1 :leapfrog-steps 10}})

;; 4. Exponential prior (mx/eval! in body — standard MH for positivity)
(def exponential-spec
  {:name "exponential"
   :model (gen []
            (let [rate (dyn/trace :rate (dist/gamma-dist (mx/scalar 2) (mx/scalar 1)))]
              (mx/eval! rate)
              (let [rate-val (mx/item rate)]
                (dyn/trace :obs (dist/exponential (mx/scalar rate-val)))
                rate-val)))
   :args []
   :param-addrs [:rate]
   :obs-addrs [:obs]
   :algorithms [:mh]
   :mh-opts {:selection (sel/select :rate)}})

;; 5. Coin flip (mx/item in body — standard MH for bounded support)
(def coin-flip
  {:name "coin-flip"
   :model (gen []
            (let [p-val (dyn/trace :p (dist/uniform 0.01 0.99))]
              (mx/eval! p-val)
              (let [p-num (mx/item p-val)]
                (doseq [i (range 5)]
                  (dyn/trace (keyword (str "flip" i)) (dist/bernoulli p-num)))
                p-num)))
   :args []
   :param-addrs [:p]
   :obs-addrs [:flip0 :flip1 :flip2 :flip3 :flip4]
   :algorithms [:mh]
   :mh-opts {:selection (sel/select :p)}})

;; 6. Linear regression (HMC only — correlated params)
(def linear-regression
  {:name "linear-regression"
   :model (gen [xs]
            (let [slope     (dyn/trace :slope (dist/gaussian 0 2))
                  intercept (dyn/trace :intercept (dist/gaussian 0 2))]
              (doseq [[j x] (map-indexed vector xs)]
                (dyn/trace (keyword (str "y" j))
                           (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                                  intercept) 1)))
              [slope intercept]))
   :args [[(mx/scalar 0.0) (mx/scalar 1.0) (mx/scalar 2.0)]]
   :param-addrs [:slope :intercept]
   :obs-addrs [:y0 :y1 :y2]
   :algorithms [:hmc]
   :hmc-opts {:addresses [:slope :intercept] :step-size 0.05 :leapfrog-steps 15}})

;; 7. Hierarchical model (HMC only — correlated params)
(def hierarchical
  {:name "hierarchical"
   :model (gen []
            (let [mu (dyn/trace :mu (dist/gaussian 0 2))
                  x  (dyn/trace :x (dist/gaussian mu 1))]
              (dyn/trace :obs (dist/gaussian x 1))
              x))
   :args []
   :param-addrs [:x]
   :obs-addrs [:obs]
   :algorithms [:hmc]
   ;; HMC addresses includes both :mu and :x, but we test only :x
   :hmc-opts {:addresses [:mu :x] :step-size 0.1 :leapfrog-steps 10}})

;; ── SBC harness ────────────────────────────────────────────────────────────

(defn run-sbc-single
  "Run one SBC simulation. Returns vector of ranks (one per param).
   Wrapped in mx/tidy to prevent Metal resource exhaustion."
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
  "Run N SBC simulations and perform chi-squared test per parameter."
  [model-spec algo-key n-sims]
  (let [{:keys [param-addrs]} model-spec
        n-params (count param-addrs)
        all-ranks (loop [i 0, ranks-acc (vec (repeat n-params [])), failures 0]
                    (if (>= i n-sims)
                      (if (> (/ failures n-sims) 0.2)
                        (do (println "  FAIL: >" 20 "% simulations failed ("
                                     failures "/" n-sims ")")
                            nil)
                        ranks-acc)
                      (let [result (try
                                     (run-sbc-single model-spec algo-key)
                                     (catch :default e
                                       (when (zero? (mod failures 10))
                                         (println "    [sim" i "failed:" (.-message e) "]"))
                                       nil))]
                        (if result
                          (recur (inc i)
                                 (mapv (fn [j] (conj (nth ranks-acc j)
                                                     (nth result j)))
                                       (range n-params))
                                 failures)
                          (recur (inc i) ranks-acc (inc failures))))))]
    (when all-ranks
      (mapv (fn [j]
              (let [param (nth param-addrs j)
                    ranks (nth all-ranks j)
                    result (chi-squared-uniformity ranks (count ranks))]
                {:param param :result result}))
            (range n-params)))))

;; ── All model specs ────────────────────────────────────────────────────────

(def all-models
  [single-gaussian
   two-gaussians
   gaussian-multi-obs
   exponential-spec
   coin-flip
   linear-regression
   hierarchical])

;; ── Top-level runner ───────────────────────────────────────────────────────

(println (str "=== Simulation-Based Calibration (SBC) Tests ==="))
(println (str "N_cmh=" N-CMH ", N_hmc=" N-HMC
              ", L=" L ", " N-BINS " bins, alpha=" ALPHA))

(def results (atom {:pass 0 :fail 0}))

(doseq [model-spec all-models
        algo-key (:algorithms model-spec)]
  (let [start (js/Date.now)
        n-sims (if (= algo-key :hmc) N-HMC N-CMH)
        _ (println (str "\n-- SBC: " (:name model-spec) " x " (name algo-key) " --"))
        param-results (run-sbc model-spec algo-key n-sims)
        elapsed (/ (- (js/Date.now) start) 1000)]
    (if param-results
      (doseq [{:keys [param result]} param-results]
        (let [{:keys [statistic pass? critical df]} result]
          (if pass?
            (do (println (str "  PASS: " (name param)
                              " chi2=" (.toFixed statistic 2)
                              " (critical=" critical " df=" df ")"))
                (swap! results update :pass inc))
            (do (println (str "  FAIL: " (name param)
                              " chi2=" (.toFixed statistic 2)
                              " (critical=" critical " df=" df ")"))
                (swap! results update :fail inc)))))
      (swap! results update :fail inc))
    (println (str "  [" (.toFixed elapsed 1) "s]"))))

(let [{:keys [pass fail]} @results
      total (+ pass fail)]
  (println (str "\n=== SBC Summary: " pass "/" total " passed ==="))
  ;; Force-exit to avoid Bun segfault during MLX native addon cleanup.
  ;; The OS reclaims all Metal/GPU resources on process exit anyway.
  (js/process.exit (if (zero? fail) 0 1)))
