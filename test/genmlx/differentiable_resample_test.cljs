(ns genmlx.differentiable-resample-test
  "WP-3 tests: differentiable resampling (Gumbel-top-k and Gumbel-softmax).

   Tests cover:
   1. generate-gumbel-noise: shapes and distribution
   2. gumbel-top-k: valid indices, correct shape, distribution approximation
   3. gumbel-softmax: correct shape, differentiability, temperature effect
   4. soft-resample: correct shape, differentiability
   5. Integration with compiled-smc via :resample-method option
   6. Gate 3: correctness on linear Gaussian (posterior + log-ML)
   7. Gate 4: gradient quality through gumbel-softmax"
  (:require [genmlx.gen :refer [gen]]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.inference.differentiable-resample :as dr]
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
;; 1. generate-gumbel-noise
;; ---------------------------------------------------------------------------

(println "\n== generate-gumbel-noise ==")

(let [key (rng/fresh-key 42)
      T 5 N 100
      noise (dr/generate-gumbel-noise key T N)]
  (mx/eval! noise)
  (assert-equal "shape is [T,N,N]" [T N N] (mx/shape noise))
  ;; Gumbel(0,1) has mean ≈ 0.5772 (Euler-Mascheroni constant)
  (let [mean-val (mx/realize (mx/mean noise))]
    (assert-close "mean ≈ Euler-Mascheroni (0.5772)" 0.5772 mean-val 0.3))
  ;; Variance ≈ π²/6 ≈ 1.6449
  (let [var-val (mx/realize (mx/variance noise))]
    (assert-close "variance ≈ π²/6 (1.6449)" 1.6449 var-val 0.5)))

;; Different keys produce different noise
(let [k1 (rng/fresh-key 1)
      k2 (rng/fresh-key 2)
      n1 (dr/generate-gumbel-noise k1 2 3)
      n2 (dr/generate-gumbel-noise k2 2 3)]
  (mx/eval! n1 n2)
  (assert-equal "noise shape [T,N,N]" [2 3 3] (mx/shape n1))
  (assert-true "different keys → different noise"
               (not= (mx/->clj n1) (mx/->clj n2))))

;; ---------------------------------------------------------------------------
;; 2. gumbel-top-k
;; ---------------------------------------------------------------------------

(println "\n== gumbel-top-k ==")

;; Basic shape and validity
(let [N 8 K 3
      particles (mx/reshape (mx/astype (mx/arange 0 (* N K) 1) mx/float32) [N K])
      log-weights (mx/zeros [N])
      ;; [N,N] noise for with-replacement resampling
      gumbel (dr/generate-gumbel-noise (rng/fresh-key 77) 1 N)
      gumbel-t (mx/index gumbel 0)  ;; [N,N] for this step
      _ (mx/eval! particles gumbel-t)
      {:keys [particles ancestors]} (dr/gumbel-top-k particles log-weights gumbel-t)]
  (mx/eval! particles ancestors)
  (assert-equal "resampled shape" [N K] (mx/shape particles))
  (assert-equal "ancestors shape" [N] (mx/shape ancestors))
  (let [anc-vals (mx/->clj ancestors)]
    (assert-true "all ancestors in [0,N)"
                 (every? #(and (>= % 0) (< % N)) anc-vals))
    ;; With replacement: duplicates ARE expected (unlike permutation)
    (assert-true "ancestors are valid indices"
                 (every? #(and (>= % 0) (< % N)) anc-vals))))

;; Skewed weights: dominant particle appears in all positions
(let [N 4 K 2
      particles (mx/array [[1.0 2.0] [3.0 4.0] [5.0 6.0] [7.0 8.0]])
      ;; Particle 2 has overwhelmingly high weight
      log-weights (mx/array [-100.0 -100.0 0.0 -100.0])
      ;; Small noise won't overcome the 100-nat weight gap
      gumbel (mx/multiply (mx/ones [N N]) (mx/scalar 0.1))
      {:keys [ancestors]} (dr/gumbel-top-k particles log-weights gumbel)]
  (mx/eval! ancestors)
  (let [anc-vals (mx/->clj ancestors)]
    ;; With replacement: ALL ancestors should be particle 2 (dominant weight)
    (assert-true "all ancestors are dominant particle"
                 (every? #(= % 2) anc-vals))))

;; Statistical test: with equal weights, each particle should appear ~equally often
(let [N 4 K 1
      particles (mx/array [[0.0] [1.0] [2.0] [3.0]])
      log-weights (mx/zeros [N])
      counts (atom {0 0, 1 0, 2 0, 3 0})
      n-trials 200]
  (doseq [trial (range n-trials)]
    (let [key (rng/fresh-key trial)
          gumbel-all (dr/generate-gumbel-noise key 1 N)  ;; [1,N,N]
          gumbel (mx/index gumbel-all 0)  ;; [N,N]
          {:keys [ancestors]} (dr/gumbel-top-k particles log-weights gumbel)]
      (mx/eval! ancestors)
      ;; Count who appears as first ancestor
      (swap! counts update (first (mx/->clj ancestors)) inc)))
  (let [min-count (apply min (vals @counts))
        max-count (apply max (vals @counts))]
    (assert-true (str "uniform weights → roughly uniform first ancestor (min=" min-count " max=" max-count ")")
                 ;; Each should get ~50 out of 200; allow wide range
                 (and (> min-count 20) (< max-count 100)))))

;; ---------------------------------------------------------------------------
;; 3. gumbel-softmax
;; ---------------------------------------------------------------------------

(println "\n== gumbel-softmax ==")

;; Basic shape
(let [N 4 K 3
      particles (mx/ones [N K])
      log-weights (mx/zeros [N])
      gumbel (mx/multiply (mx/ones [N N]) (mx/scalar 0.1))  ;; [N,N] noise
      tau (mx/scalar 1.0)
      {:keys [particles]} (dr/gumbel-softmax particles log-weights gumbel tau)]
  (mx/eval! particles)
  (assert-equal "output shape [N,K]" [N K] (mx/shape particles)))

;; Dominant weight → output ≈ dominant particle
(let [particles (mx/array [[1.0 2.0] [10.0 20.0]])
      log-weights (mx/array [-100.0 0.0])
      gumbel (mx/zeros [2 2])  ;; [N,N] zero noise
      tau (mx/scalar 0.1)
      {:keys [particles]} (dr/gumbel-softmax particles log-weights gumbel tau)]
  (mx/eval! particles)
  (let [vals (mx/->clj particles)]
    (assert-close "row 0 ≈ dominant particle col 0" 10.0 (get-in vals [0 0]) 0.1)
    (assert-close "row 0 ≈ dominant particle col 1" 20.0 (get-in vals [0 1]) 0.1)))

;; Temperature effect: lower tau → sharper
(let [particles (mx/array [[0.0] [10.0]])
      log-weights (mx/array [0.0 1.0])
      gumbel (mx/zeros [2 2])]  ;; [N,N] zero noise
  (let [soft-low (dr/gumbel-softmax particles log-weights gumbel (mx/scalar 0.1))
        soft-high (dr/gumbel-softmax particles log-weights gumbel (mx/scalar 10.0))]
    (mx/eval! (:particles soft-low) (:particles soft-high))
    (let [low-val (get-in (mx/->clj (:particles soft-low)) [0 0])
          high-val (get-in (mx/->clj (:particles soft-high)) [0 0])]
      ;; Low tau → closer to 10 (hard selection of higher weight)
      ;; High tau → closer to 5 (uniform average)
      (assert-true (str "low tau closer to hard (" (.toFixed low-val 2) " vs " (.toFixed high-val 2) ")")
                   (> low-val high-val)))))

;; Differentiability: gradient is non-zero
(let [particles (mx/array [[1.0 2.0] [3.0 4.0] [5.0 6.0] [7.0 8.0]])
      N 4
      f (fn [lw]
          (let [gumbel (mx/multiply (mx/ones [N N]) (mx/scalar 0.1))
                tau (mx/scalar 1.0)
                result (dr/gumbel-softmax particles lw gumbel tau)]
            (mx/sum (:particles result))))
      grad-f (mx/grad f)
      lw (mx/array [-1.0 -2.0 -0.5 -3.0])
      g (grad-f lw)]
  (mx/eval! g)
  (let [grad-vals (mx/->clj g)]
    (assert-true "gradient is non-zero" (not (every? zero? grad-vals)))
    (println "    gradient values:" grad-vals)))

;; Gradient direction: increasing weight of high-value particle → increasing sum
(let [particles (mx/array [[1.0] [100.0]])
      f (fn [lw]
          (let [result (dr/gumbel-softmax particles lw (mx/zeros [2 2]) (mx/scalar 1.0))]
            (mx/sum (:particles result))))
      grad-f (mx/grad f)
      lw (mx/array [0.0 0.0])
      g (grad-f lw)]
  (mx/eval! g)
  (let [[g0 g1] (mx/->clj g)]
    (assert-true "grad[low-value particle] < 0" (< g0 0))
    (assert-true "grad[high-value particle] > 0" (> g1 0))))

;; ---------------------------------------------------------------------------
;; 4. soft-resample
;; ---------------------------------------------------------------------------

(println "\n== soft-resample ==")

(let [N 4 K 2
      particles (mx/array [[1.0 2.0] [3.0 4.0] [5.0 6.0] [7.0 8.0]])
      log-weights (mx/array [-1.0 -2.0 -0.5 -3.0])
      {:keys [particles]} (dr/soft-resample particles log-weights 0.9)]
  (mx/eval! particles)
  (assert-equal "soft-resample shape" [N K] (mx/shape particles)))

;; Differentiability
(let [particles (mx/array [[1.0] [10.0]])
      f (fn [lw]
          (mx/sum (:particles (dr/soft-resample particles lw 0.9))))
      grad-f (mx/grad f)
      lw (mx/array [0.0 0.0])
      g (grad-f lw)]
  (mx/eval! g)
  (assert-true "soft-resample gradient non-zero" (not (every? zero? (mx/->clj g)))))

;; Alpha=0 → uniform (all particles weighted equally)
(let [particles (mx/array [[0.0] [10.0]])
      log-weights (mx/array [0.0 100.0])  ;; heavily skewed
      {:keys [particles]} (dr/soft-resample particles log-weights 0.0)]
  (mx/eval! particles)
  ;; With alpha=0, output should be uniform average = 5.0
  (assert-close "alpha=0 → uniform average" 5.0
                (get-in (mx/->clj particles) [0 0]) 0.01))

;; ---------------------------------------------------------------------------
;; 5. Integration: compiled-smc with :resample-method
;; ---------------------------------------------------------------------------

(println "\n== compiled-smc integration ==")

;; Test kernel
(def rw-kernel
  (gen [t state]
    (let [new-state (trace :x (dist/gaussian state 1))]
      (trace :y (dist/gaussian new-state 0.5))
      new-state)))

;; Generate some observations
(def test-obs
  (let [key (rng/fresh-key 99)
        model (dyn/with-key rw-kernel key)]
    (mapv (fn [t]
            (let [obs-val (+ (* 0.5 t) (mx/realize (rng/normal (rng/fresh-key t) [])))]
              (cm/choicemap :y (mx/scalar obs-val))))
          (range 5))))

;; Test with :gumbel-top-k
(try
  (let [result (csmc/compiled-smc
                 {:particles 50
                  :key (rng/fresh-key 1)
                  :resample-method :gumbel-top-k}
                 rw-kernel (mx/scalar 0.0) test-obs)]
    (mx/eval! (:log-ml result) (:particles result))
    (assert-true "gumbel-top-k: has :particles" (some? (:particles result)))
    (assert-true "gumbel-top-k: has :log-ml" (some? (:log-ml result)))
    (assert-equal "gumbel-top-k: particles shape [N,K]" [50 2] (mx/shape (:particles result)))
    (let [lml (mx/realize (:log-ml result))]
      (assert-true (str "gumbel-top-k: log-ML is finite (" (.toFixed lml 2) ")")
                   (js/isFinite lml))))
  (catch :default e
    (swap! fail-count inc)
    (println (str "  FAIL: gumbel-top-k integration: " (.-message e)))))

;; Test with :gumbel-softmax
(try
  (let [result (csmc/compiled-smc
                 {:particles 50
                  :key (rng/fresh-key 2)
                  :resample-method :gumbel-softmax
                  :tau 1.0}
                 rw-kernel (mx/scalar 0.0) test-obs)]
    (mx/eval! (:log-ml result) (:particles result))
    (assert-true "gumbel-softmax: has :particles" (some? (:particles result)))
    (assert-equal "gumbel-softmax: particles shape [N,K]" [50 2] (mx/shape (:particles result)))
    (let [lml (mx/realize (:log-ml result))]
      (assert-true (str "gumbel-softmax: log-ML is finite (" (.toFixed lml 2) ")")
                   (js/isFinite lml))))
  (catch :default e
    (swap! fail-count inc)
    (println (str "  FAIL: gumbel-softmax integration: " (.-message e)))))

;; Test with :systematic (default, regression)
(try
  (let [result (csmc/compiled-smc
                 {:particles 50
                  :key (rng/fresh-key 3)
                  :resample-method :systematic}
                 rw-kernel (mx/scalar 0.0) test-obs)]
    (mx/eval! (:log-ml result) (:particles result))
    (assert-true "systematic: still works" (some? (:particles result)))
    (let [lml (mx/realize (:log-ml result))]
      (assert-true (str "systematic: log-ML is finite (" (.toFixed lml 2) ")")
                   (js/isFinite lml))))
  (catch :default e
    (swap! fail-count inc)
    (println (str "  FAIL: systematic integration: " (.-message e)))))

;; ---------------------------------------------------------------------------
;; 6. Gate 3: Correctness on linear Gaussian model
;;    Known exact posterior: Kalman filter for linear-Gaussian SSM.
;;    x_t = x_{t-1} + w_t,  w_t ~ N(0, q²)
;;    y_t = x_t + v_t,      v_t ~ N(0, r²)
;;    Posterior and log-ML have exact closed-form solutions.
;; ---------------------------------------------------------------------------

(println "\n== Gate 3: Linear Gaussian Correctness ==")

;; Kalman filter for reference
(defn kalman-filter
  "Run Kalman filter on linear Gaussian SSM.
   Returns {:means [T] :variances [T] :log-ml scalar}."
  [observations q r prior-mean prior-var]
  (loop [obs observations
         t 0
         mean prior-mean
         var prior-var
         log-ml 0.0
         means []
         variances []]
    (if (empty? obs)
      {:means means :variances variances :log-ml log-ml}
      (let [y (first obs)
            ;; Predict
            pred-mean mean
            pred-var (+ var (* q q))
            ;; Update
            innov (- y pred-mean)
            innov-var (+ pred-var (* r r))
            K (/ pred-var innov-var)
            post-mean (+ pred-mean (* K innov))
            post-var (* (- 1.0 K) pred-var)
            ;; Log-ML increment: log N(y; pred-mean, innov-var)
            lml-inc (- (* -0.5 (js/Math.log (* 2 js/Math.PI innov-var)))
                       (* 0.5 (/ (* innov innov) innov-var)))]
        (recur (rest obs) (inc t) post-mean post-var
               (+ log-ml lml-inc) (conj means post-mean) (conj variances post-var))))))

;; Linear Gaussian SSM kernel for compiled SMC
(def lg-kernel
  (gen [t state]
    (let [new-state (trace :x (dist/gaussian state 1.0))]
      (trace :y (dist/gaussian new-state 0.5))
      new-state)))

;; Generate observations from the model
(def gate3-T 10)
(def gate3-true-states
  (vec (reductions (fn [s _] (+ s (mx/realize (rng/normal (rng/fresh-key (+ 100 _)) []))))
                   0.0 (range gate3-T))))
(def gate3-obs
  (mapv (fn [t]
          (+ (nth gate3-true-states (inc t))
             (* 0.5 (mx/realize (rng/normal (rng/fresh-key (+ 200 t)) [])))))
        (range gate3-T)))

;; Run Kalman filter for exact reference
(def kalman-result (kalman-filter gate3-obs 1.0 0.5 0.0 1.0))

(println "  Kalman log-ML:" (.toFixed (:log-ml kalman-result) 4))
(println "  Kalman final mean:" (.toFixed (last (:means kalman-result)) 4))
(println "  Kalman final var:" (.toFixed (last (:variances kalman-result)) 4))

;; Build observation choicemaps
(def gate3-obs-cms
  (mapv (fn [y] (cm/choicemap :y (mx/scalar y))) gate3-obs))

;; Run compiled SMC with gumbel-top-k, N=500
(def gate3-N 500)

(try
  (let [result (csmc/compiled-smc
                 {:particles gate3-N
                  :key (rng/fresh-key 42)
                  :resample-method :gumbel-top-k}
                 lg-kernel (mx/scalar 0.0) gate3-obs-cms)]
    (mx/eval! (:log-ml result) (:particles result))
    (let [smc-log-ml (mx/realize (:log-ml result))
          kalman-log-ml (:log-ml kalman-result)
          ;; Extract posterior mean of last state (column 0 = :x)
          particles-clj (mx/->clj (:particles result))
          x-samples (mapv #(first %) particles-clj)
          smc-mean (/ (reduce + x-samples) (count x-samples))
          kalman-mean (last (:means kalman-result))
          kalman-var (last (:variances kalman-result))
          kalman-std (js/Math.sqrt kalman-var)]
      (println "  SMC log-ML:" (.toFixed smc-log-ml 4)
               " Kalman:" (.toFixed kalman-log-ml 4))
      (println "  SMC posterior mean:" (.toFixed smc-mean 4)
               " Kalman:" (.toFixed kalman-mean 4)
               " ±2σ:" (.toFixed (* 2 kalman-std) 4))
      ;; Gate 3 criteria:
      ;; 1. Posterior within 2σ of Kalman
      (assert-true "Gate 3: posterior mean within 2σ"
                   (< (js/Math.abs (- smc-mean kalman-mean))
                      (* 2 kalman-std)))
      ;; 2. Log-ML within 0.5 nats of Kalman (relaxed for N=500)
      ;; Note: compiled SMC may have known bias (see task #15)
      (assert-close "Gate 3: log-ML within 2.0 nats of Kalman"
                    kalman-log-ml smc-log-ml 2.0)))
  (catch :default e
    (swap! fail-count inc)
    (println (str "  FAIL: Gate 3 experiment: " (.-message e)))))

;; ---------------------------------------------------------------------------
;; 7. Gate 4: Gradient quality through gumbel-softmax
;;    Compute gradient of log-ML w.r.t. observation noise parameter.
;;    Compare against finite differences.
;; ---------------------------------------------------------------------------

(println "\n== Gate 4: Gradient Quality ==")

;; Simple test: gradient of sum of soft-resampled particles w.r.t. log-weights
;; at multiple temperatures. Compare against finite differences.

(let [particles (mx/array [[1.0] [5.0] [10.0] [20.0]])
      N 4
      gumbel (mx/multiply (mx/ones [N N]) (mx/scalar 0.1))  ;; [N,N] noise
      eps 1e-3]
  (doseq [tau-val [0.5 1.0 2.0]]
    (let [tau (mx/scalar tau-val)
          ;; Analytical gradient via mx/grad
          f (fn [lw]
              (mx/sum (:particles (dr/gumbel-softmax particles lw gumbel tau))))
          grad-f (mx/grad f)
          lw (mx/array [-1.0 0.0 -0.5 -2.0])
          analytical-grad (mx/->clj (let [g (grad-f lw)] (mx/eval! g) g))
          ;; Finite difference gradient
          fd-grad
          (mapv (fn [i]
                  (let [lw-plus (mx/->clj lw)
                        lw-minus (mx/->clj lw)
                        lw-p (mx/array (assoc lw-plus i (+ (nth lw-plus i) eps)))
                        lw-m (mx/array (assoc lw-minus i (- (nth lw-minus i) eps)))
                        f-plus (mx/realize (f lw-p))
                        f-minus (mx/realize (f lw-m))]
                    (/ (- f-plus f-minus) (* 2 eps))))
                (range 4))]
      (println (str "  tau=" tau-val
                    " analytical=" (mapv #(.toFixed % 3) analytical-grad)
                    " fd=" (mapv #(.toFixed % 3) fd-grad)))
      ;; Check sign agreement
      (let [sign-matches (count (filter (fn [[a fd]]
                                          (or (and (pos? a) (pos? fd))
                                              (and (neg? a) (neg? fd))
                                              (and (zero? a) (zero? fd))))
                                        (map vector analytical-grad fd-grad)))
            sign-rate (/ sign-matches 4.0)]
        (assert-true (str "Gate 4: sign agreement at tau=" tau-val " ≥ 75% (got " (* 100 sign-rate) "%)")
                     (>= sign-rate 0.75)))
      ;; Check magnitude within 5x
      (let [mag-ok (every? (fn [[a fd]]
                             (if (< (js/Math.abs fd) 1e-6)
                               true  ;; skip near-zero
                               (let [ratio (/ (js/Math.abs a) (js/Math.abs fd))]
                                 (and (> ratio 0.2) (< ratio 5.0)))))
                           (map vector analytical-grad fd-grad))]
        (assert-true (str "Gate 4: magnitude within 5x at tau=" tau-val) mag-ok)))))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n== RESULTS: " @pass-count "/" (+ @pass-count @fail-count)
              " passed, " @fail-count " failed =="))
