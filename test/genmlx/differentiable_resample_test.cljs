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
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.gen :refer [gen]]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.inference.differentiable-resample :as dr]
            [genmlx.inference.compiled-smc :as csmc]))

;; ---------------------------------------------------------------------------
;; 1. generate-gumbel-noise
;; ---------------------------------------------------------------------------

(deftest generate-gumbel-noise-test
  (testing "shape and distribution"
    (let [key (rng/fresh-key 42)
          T 5 N 100
          noise (dr/generate-gumbel-noise key T N)]
      (mx/eval! noise)
      (is (= [T N N] (mx/shape noise)) "shape is [T,N,N]")
      (let [mean-val (mx/realize (mx/mean noise))]
        (is (h/close? 0.5772 mean-val 0.3) "mean ~ Euler-Mascheroni (0.5772)"))
      (let [var-val (mx/realize (mx/variance noise))]
        (is (h/close? 1.6449 var-val 0.5) "variance ~ pi^2/6 (1.6449)"))))

  (testing "different keys produce different noise"
    (let [k1 (rng/fresh-key 1)
          k2 (rng/fresh-key 2)
          n1 (dr/generate-gumbel-noise k1 2 3)
          n2 (dr/generate-gumbel-noise k2 2 3)]
      (mx/eval! n1 n2)
      (is (= [2 3 3] (mx/shape n1)) "noise shape [T,N,N]")
      (is (not= (mx/->clj n1) (mx/->clj n2)) "different keys -> different noise"))))

;; ---------------------------------------------------------------------------
;; 2. gumbel-top-k
;; ---------------------------------------------------------------------------

(deftest gumbel-top-k-test
  (testing "basic shape and validity"
    (let [N 8 K 3
          particles (mx/reshape (mx/astype (mx/arange 0 (* N K) 1) mx/float32) [N K])
          log-weights (mx/zeros [N])
          gumbel (dr/generate-gumbel-noise (rng/fresh-key 77) 1 N)
          gumbel-t (mx/index gumbel 0)
          _ (mx/eval! particles gumbel-t)
          {:keys [particles ancestors]} (dr/gumbel-top-k particles log-weights gumbel-t)]
      (mx/eval! particles ancestors)
      (is (= [N K] (mx/shape particles)) "resampled shape")
      (is (= [N] (mx/shape ancestors)) "ancestors shape")
      (let [anc-vals (mx/->clj ancestors)]
        (is (every? #(and (>= % 0) (< % N)) anc-vals) "all ancestors in [0,N)")
        (is (every? #(and (>= % 0) (< % N)) anc-vals) "ancestors are valid indices"))))

  (testing "skewed weights: dominant particle appears in all positions"
    (let [N 4 K 2
          particles (mx/array [[1.0 2.0] [3.0 4.0] [5.0 6.0] [7.0 8.0]])
          log-weights (mx/array [-100.0 -100.0 0.0 -100.0])
          gumbel (mx/multiply (mx/ones [N N]) (mx/scalar 0.1))
          {:keys [ancestors]} (dr/gumbel-top-k particles log-weights gumbel)]
      (mx/eval! ancestors)
      (let [anc-vals (mx/->clj ancestors)]
        (is (every? #(= % 2) anc-vals) "all ancestors are dominant particle"))))

  (testing "uniform weights produce roughly uniform first ancestor"
    (let [N 4 K 1
          particles (mx/array [[0.0] [1.0] [2.0] [3.0]])
          log-weights (mx/zeros [N])
          counts (atom {0 0, 1 0, 2 0, 3 0})
          n-trials 200]
      (doseq [trial (range n-trials)]
        (let [key (rng/fresh-key trial)
              gumbel-all (dr/generate-gumbel-noise key 1 N)
              gumbel (mx/index gumbel-all 0)
              {:keys [ancestors]} (dr/gumbel-top-k particles log-weights gumbel)]
          (mx/eval! ancestors)
          (swap! counts update (first (mx/->clj ancestors)) inc)))
      (let [min-count (apply min (vals @counts))
            max-count (apply max (vals @counts))]
        (is (and (> min-count 20) (< max-count 100))
            (str "uniform weights -> roughly uniform first ancestor (min=" min-count " max=" max-count ")"))))))

;; ---------------------------------------------------------------------------
;; 3. gumbel-softmax
;; ---------------------------------------------------------------------------

(deftest gumbel-softmax-test
  (testing "basic shape"
    (let [N 4 K 3
          particles (mx/ones [N K])
          log-weights (mx/zeros [N])
          gumbel (mx/multiply (mx/ones [N N]) (mx/scalar 0.1))
          tau (mx/scalar 1.0)
          {:keys [particles]} (dr/gumbel-softmax particles log-weights gumbel tau)]
      (mx/eval! particles)
      (is (= [N K] (mx/shape particles)) "output shape [N,K]")))

  (testing "dominant weight -> output ~ dominant particle"
    (let [particles (mx/array [[1.0 2.0] [10.0 20.0]])
          log-weights (mx/array [-100.0 0.0])
          gumbel (mx/zeros [2 2])
          tau (mx/scalar 0.1)
          {:keys [particles]} (dr/gumbel-softmax particles log-weights gumbel tau)]
      (mx/eval! particles)
      (let [vals (mx/->clj particles)]
        (is (h/close? 10.0 (get-in vals [0 0]) 0.1) "row 0 ~ dominant particle col 0")
        (is (h/close? 20.0 (get-in vals [0 1]) 0.1) "row 0 ~ dominant particle col 1"))))

  (testing "temperature effect: lower tau -> sharper"
    (let [particles (mx/array [[0.0] [10.0]])
          log-weights (mx/array [0.0 1.0])
          gumbel (mx/zeros [2 2])]
      (let [soft-low (dr/gumbel-softmax particles log-weights gumbel (mx/scalar 0.1))
            soft-high (dr/gumbel-softmax particles log-weights gumbel (mx/scalar 10.0))]
        (mx/eval! (:particles soft-low) (:particles soft-high))
        (let [low-val (get-in (mx/->clj (:particles soft-low)) [0 0])
              high-val (get-in (mx/->clj (:particles soft-high)) [0 0])]
          (is (> low-val high-val)
              (str "low tau closer to hard (" (.toFixed low-val 2) " vs " (.toFixed high-val 2) ")"))))))

  (testing "differentiability: gradient is non-zero"
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
        (is (not (every? zero? grad-vals)) "gradient is non-zero"))))

  (testing "gradient direction: increasing weight of high-value particle -> increasing sum"
    (let [particles (mx/array [[1.0] [100.0]])
          f (fn [lw]
              (let [result (dr/gumbel-softmax particles lw (mx/zeros [2 2]) (mx/scalar 1.0))]
                (mx/sum (:particles result))))
          grad-f (mx/grad f)
          lw (mx/array [0.0 0.0])
          g (grad-f lw)]
      (mx/eval! g)
      (let [[g0 g1] (mx/->clj g)]
        (is (< g0 0) "grad[low-value particle] < 0")
        (is (> g1 0) "grad[high-value particle] > 0")))))

;; ---------------------------------------------------------------------------
;; 4. soft-resample
;; ---------------------------------------------------------------------------

(deftest soft-resample-test
  (testing "correct shape"
    (let [N 4 K 2
          particles (mx/array [[1.0 2.0] [3.0 4.0] [5.0 6.0] [7.0 8.0]])
          log-weights (mx/array [-1.0 -2.0 -0.5 -3.0])
          {:keys [particles]} (dr/soft-resample particles log-weights 0.9)]
      (mx/eval! particles)
      (is (= [N K] (mx/shape particles)) "soft-resample shape")))

  (testing "differentiability"
    (let [particles (mx/array [[1.0] [10.0]])
          f (fn [lw]
              (mx/sum (:particles (dr/soft-resample particles lw 0.9))))
          grad-f (mx/grad f)
          lw (mx/array [0.0 0.0])
          g (grad-f lw)]
      (mx/eval! g)
      (is (not (every? zero? (mx/->clj g))) "soft-resample gradient non-zero")))

  (testing "alpha=0 -> uniform average"
    (let [particles (mx/array [[0.0] [10.0]])
          log-weights (mx/array [0.0 100.0])
          {:keys [particles]} (dr/soft-resample particles log-weights 0.0)]
      (mx/eval! particles)
      (is (h/close? 5.0 (get-in (mx/->clj particles) [0 0]) 0.01) "alpha=0 -> uniform average"))))

;; ---------------------------------------------------------------------------
;; 5. Integration: compiled-smc with :resample-method
;; ---------------------------------------------------------------------------

(def rw-kernel
  (gen [t state]
    (let [new-state (trace :x (dist/gaussian state 1))]
      (trace :y (dist/gaussian new-state 0.5))
      new-state)))

(def test-obs-cms
  (let [key (rng/fresh-key 99)
        model (dyn/with-key rw-kernel key)]
    (mapv (fn [t]
            (let [obs-val (+ (* 0.5 t) (mx/realize (rng/normal (rng/fresh-key t) [])))]
              (cm/choicemap :y (mx/scalar obs-val))))
          (range 5))))

(deftest compiled-smc-integration
  (testing "gumbel-top-k"
    (let [result (csmc/compiled-smc
                   {:particles 50
                    :key (rng/fresh-key 1)
                    :resample-method :gumbel-top-k}
                   rw-kernel (mx/scalar 0.0) test-obs-cms)]
      (mx/eval! (:log-ml result) (:particles result))
      (is (some? (:particles result)) "gumbel-top-k: has :particles")
      (is (some? (:log-ml result)) "gumbel-top-k: has :log-ml")
      (is (= [50 2] (mx/shape (:particles result))) "gumbel-top-k: particles shape [N,K]")
      (let [lml (mx/realize (:log-ml result))]
        (is (js/isFinite lml) (str "gumbel-top-k: log-ML is finite (" (.toFixed lml 2) ")")))))

  (testing "gumbel-softmax"
    (let [result (csmc/compiled-smc
                   {:particles 50
                    :key (rng/fresh-key 2)
                    :resample-method :gumbel-softmax
                    :tau 1.0}
                   rw-kernel (mx/scalar 0.0) test-obs-cms)]
      (mx/eval! (:log-ml result) (:particles result))
      (is (some? (:particles result)) "gumbel-softmax: has :particles")
      (is (= [50 2] (mx/shape (:particles result))) "gumbel-softmax: particles shape [N,K]")
      (let [lml (mx/realize (:log-ml result))]
        (is (js/isFinite lml) (str "gumbel-softmax: log-ML is finite (" (.toFixed lml 2) ")")))))

  (testing "systematic (default, regression)"
    (let [result (csmc/compiled-smc
                   {:particles 50
                    :key (rng/fresh-key 3)
                    :resample-method :systematic}
                   rw-kernel (mx/scalar 0.0) test-obs-cms)]
      (mx/eval! (:log-ml result) (:particles result))
      (is (some? (:particles result)) "systematic: still works")
      (let [lml (mx/realize (:log-ml result))]
        (is (js/isFinite lml) (str "systematic: log-ML is finite (" (.toFixed lml 2) ")"))))))

;; ---------------------------------------------------------------------------
;; 6. Gate 3: Correctness on linear Gaussian model
;; ---------------------------------------------------------------------------

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
            pred-mean mean
            pred-var (+ var (* q q))
            innov (- y pred-mean)
            innov-var (+ pred-var (* r r))
            K (/ pred-var innov-var)
            post-mean (+ pred-mean (* K innov))
            post-var (* (- 1.0 K) pred-var)
            lml-inc (- (* -0.5 (js/Math.log (* 2 js/Math.PI innov-var)))
                       (* 0.5 (/ (* innov innov) innov-var)))]
        (recur (rest obs) (inc t) post-mean post-var
               (+ log-ml lml-inc) (conj means post-mean) (conj variances post-var))))))

(def lg-kernel
  (gen [t state]
    (let [new-state (trace :x (dist/gaussian state 1.0))]
      (trace :y (dist/gaussian new-state 0.5))
      new-state)))

(def gate3-T 10)
(def gate3-true-states
  (vec (reductions (fn [s _] (+ s (mx/realize (rng/normal (rng/fresh-key (+ 100 _)) []))))
                   0.0 (range gate3-T))))
(def gate3-obs
  (mapv (fn [t]
          (+ (nth gate3-true-states (inc t))
             (* 0.5 (mx/realize (rng/normal (rng/fresh-key (+ 200 t)) [])))))
        (range gate3-T)))
(def kalman-result (kalman-filter gate3-obs 1.0 0.5 0.0 1.0))
(def gate3-obs-cms
  (mapv (fn [y] (cm/choicemap :y (mx/scalar y))) gate3-obs))
(def gate3-N 500)

(deftest gate3-linear-gaussian-correctness
  (testing "Gate 3: compiled SMC on linear Gaussian"
    (let [result (csmc/compiled-smc
                   {:particles gate3-N
                    :key (rng/fresh-key 42)
                    :resample-method :gumbel-top-k}
                   lg-kernel (mx/scalar 0.0) gate3-obs-cms)]
      (mx/eval! (:log-ml result) (:particles result))
      (let [smc-log-ml (mx/realize (:log-ml result))
            kalman-log-ml (:log-ml kalman-result)
            particles-clj (mx/->clj (:particles result))
            x-samples (mapv #(first %) particles-clj)
            smc-mean (/ (reduce + x-samples) (count x-samples))
            kalman-mean (last (:means kalman-result))
            kalman-var (last (:variances kalman-result))
            kalman-std (js/Math.sqrt kalman-var)]
        (is (< (js/Math.abs (- smc-mean kalman-mean)) (* 2 kalman-std))
            "Gate 3: posterior mean within 2 sigma")
        (is (h/close? kalman-log-ml smc-log-ml 2.0)
            "Gate 3: log-ML within 2.0 nats of Kalman")))))

;; ---------------------------------------------------------------------------
;; 7. Gate 4: Gradient quality through gumbel-softmax
;; ---------------------------------------------------------------------------

(deftest gate4-gradient-quality
  (testing "Gate 4: gradient sign and magnitude agreement"
    (let [particles (mx/array [[1.0] [5.0] [10.0] [20.0]])
          N 4
          gumbel (mx/multiply (mx/ones [N N]) (mx/scalar 0.1))
          eps 1e-3]
      (doseq [tau-val [0.5 1.0 2.0]]
        (let [tau (mx/scalar tau-val)
              f (fn [lw]
                  (mx/sum (:particles (dr/gumbel-softmax particles lw gumbel tau))))
              grad-f (mx/grad f)
              lw (mx/array [-1.0 0.0 -0.5 -2.0])
              analytical-grad (mx/->clj (let [g (grad-f lw)] (mx/eval! g) g))
              fd-grad
              (mapv (fn [i]
                      (let [lw-plus (mx/->clj lw)
                            lw-minus (mx/->clj lw)
                            lw-p (mx/array (assoc lw-plus i (+ (nth lw-plus i) eps)))
                            lw-m (mx/array (assoc lw-minus i (- (nth lw-minus i) eps)))
                            f-plus (mx/realize (f lw-p))
                            f-minus (mx/realize (f lw-m))]
                        (/ (- f-plus f-minus) (* 2 eps))))
                    (range 4))
              sign-matches (count (filter (fn [[a fd]]
                                            (or (and (pos? a) (pos? fd))
                                                (and (neg? a) (neg? fd))
                                                (and (zero? a) (zero? fd))))
                                          (map vector analytical-grad fd-grad)))
              sign-rate (/ sign-matches 4.0)
              mag-ok (every? (fn [[a fd]]
                               (if (< (js/Math.abs fd) 1e-6)
                                 true
                                 (let [ratio (/ (js/Math.abs a) (js/Math.abs fd))]
                                   (and (> ratio 0.2) (< ratio 5.0)))))
                             (map vector analytical-grad fd-grad))]
          (is (>= sign-rate 0.75)
              (str "Gate 4: sign agreement at tau=" tau-val " >= 75% (got " (* 100 sign-rate) "%)"))
          (is mag-ok
              (str "Gate 4: magnitude within 5x at tau=" tau-val)))))))

(cljs.test/run-tests)
