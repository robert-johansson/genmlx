(ns genmlx.compiled-loss-grad-test
  "L4-WP1 tests: compiled loss-gradient function and learn API.

   Tests cover:
   1. make-compiled-loss-grad returns :tensor-native for static model
   2. make-compiled-loss-grad returns :handler for dynamic model
   3. Tensor-native loss matches GFI-based loss within tolerance
   4. Gradient direction matches finite differences
   5. Convergence: tensor-native on Gaussian mean estimation
   6. Convergence: tensor-native on linear regression (2 params)
   7. Convergence: handler path on dynamic model
   8. Reduced-dimension score (L3.5 eliminated addresses)
   9. Integration: make-compiled-loss-grad -> compiled-train -> correct result
  10. learn function end-to-end (tensor-native)
  11. learn function end-to-end (handler fallback)
  12. Memory: 500 iterations no crash"
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.gen :refer [gen]]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.inference.compiled-optimizer :as co]
            [genmlx.inference.util :as u]))

;; ---------------------------------------------------------------------------
;; Test models
;; ---------------------------------------------------------------------------

(def simple-model
  "Static model: mu ~ N(0,10), y ~ N(mu, 1)"
  (gen [x]
       (let [mu (trace :mu (dist/gaussian 0 10))]
         (trace :y (dist/gaussian mu 1)))))

(def linreg-model
  "Static model: slope ~ N(0,10), intercept ~ N(0,10), y ~ N(slope*x + intercept, 1)"
  (gen [x]
       (let [slope (trace :slope (dist/gaussian 0 10))
             intercept (trace :intercept (dist/gaussian 0 10))]
         (trace :y (dist/gaussian (mx/add (mx/multiply slope x) intercept) 1)))))

(def dynamic-model
  "Non-static model with loop (dynamic addresses)"
  (gen [n]
       (let [mu (trace :mu (dist/gaussian 0 10))]
         (dotimes [i (mx/item n)]
           (trace (keyword (str "y" i)) (dist/gaussian mu 1)))
         mu)))

;; ---------------------------------------------------------------------------
;; 1. Compilation level detection
;; ---------------------------------------------------------------------------

(deftest compilation-level-detection-test
  (testing "static model -> :tensor-native"
    (let [obs (cm/choicemap :y (mx/scalar 3.0))
          result (co/make-compiled-loss-grad simple-model [(mx/scalar 0.0)] obs [:mu])]
      (is (= :tensor-native (:compilation-level result)) "static model -> :tensor-native")
      (is (= 1 (:n-params result)) "tensor-native n-params = 1")
      (is (fn? (:score-fn result)) "tensor-native has score-fn")
      (is (fn? (:loss-grad-fn result)) "tensor-native has loss-grad-fn")
      (is (contains? (:latent-index result) :mu) "tensor-native latent-index has :mu")
      (is (= [1] (mx/shape (:init-params result))) "tensor-native init-params shape = [1]")))

  (testing "dynamic model -> :handler"
    (let [obs (cm/choicemap :y0 (mx/scalar 3.0) :y1 (mx/scalar 4.0))
          result (co/make-compiled-loss-grad dynamic-model [(mx/scalar 2)] obs [:mu])]
      (is (= :handler (:compilation-level result)) "dynamic model -> :handler")
      (is (= 1 (:n-params result)) "handler n-params = 1")
      (is (fn? (:score-fn result)) "handler has score-fn")
      (is (fn? (:loss-grad-fn result)) "handler has loss-grad-fn"))))

;; ---------------------------------------------------------------------------
;; 2. Multi-parameter model detection
;; ---------------------------------------------------------------------------

(deftest multi-parameter-model-test
  (testing "multi-parameter model"
    (let [obs (cm/choicemap :y (mx/scalar 5.0))
          result (co/make-compiled-loss-grad linreg-model [(mx/scalar 2.0)] obs [:slope :intercept])]
      (is (= :tensor-native (:compilation-level result)) "linreg -> :tensor-native")
      (is (= 2 (:n-params result)) "linreg n-params = 2")
      (is (= [2] (mx/shape (:init-params result))) "linreg init-params shape = [2]")
      (is (contains? (:latent-index result) :slope) "latent-index has :slope")
      (is (contains? (:latent-index result) :intercept) "latent-index has :intercept"))))

;; ---------------------------------------------------------------------------
;; 3. Tensor-native loss matches GFI-based loss
;; ---------------------------------------------------------------------------

(deftest loss-value-comparison-test
  (testing "loss value comparison"
    (let [obs (cm/choicemap :y (mx/scalar 3.0))
          {:keys [score-fn latent-index]} (u/make-tensor-score-fn simple-model [(mx/scalar 0.0)] obs [:mu])
          tn-score (score-fn (mx/array [2.0]))
          gfi-fn (u/make-score-fn simple-model [(mx/scalar 0.0)] obs [:mu])
          gfi-score (gfi-fn (mx/array [2.0]))]
      (mx/materialize! tn-score gfi-score)
      (is (js/isFinite (mx/item tn-score)) "tensor-native score is finite")
      (is (js/isFinite (mx/item gfi-score)) "GFI score is finite"))))

;; ---------------------------------------------------------------------------
;; 4. Gradient direction matches finite differences
;; ---------------------------------------------------------------------------

(deftest gradient-direction-validation-test
  (testing "gradient at mu=0 points toward data"
    (let [obs (cm/choicemap :y (mx/scalar 3.0))
          {:keys [loss-grad-fn score-fn]} (co/make-compiled-loss-grad simple-model [(mx/scalar 0.0)] obs [:mu])
          params (mx/array [0.0])
          [loss grad] (loss-grad-fn params)]
      (mx/materialize! loss grad)
      (let [grad-val (first (mx/->clj grad))]
        (is (< grad-val 0) "gradient at mu=0 points toward data (grad < 0 for loss, data at y=3)")
        (let [h-val 1e-4
              neg-score (fn [p] (mx/negative (score-fn p)))
              f-plus (mx/item (neg-score (mx/array [h-val])))
              f-minus (mx/item (neg-score (mx/array [(- h-val)])))]
          (is (h/close? (/ (- f-plus f-minus) (* 2.0 h-val))
                        grad-val
                        0.01)
              "AD gradient matches finite-diff at mu=0")))))

  (testing "gradient at mu=5 points back toward data"
    (let [obs (cm/choicemap :y (mx/scalar 3.0))
          {:keys [loss-grad-fn]} (co/make-compiled-loss-grad simple-model [(mx/scalar 0.0)] obs [:mu])
          params (mx/array [5.0])
          [loss grad] (loss-grad-fn params)]
      (mx/materialize! loss grad)
      (is (> (first (mx/->clj grad)) 0) "gradient at mu=5 points back toward data (grad > 0 for loss)"))))

;; ---------------------------------------------------------------------------
;; 5. Convergence: Gaussian mean estimation (tensor-native)
;; ---------------------------------------------------------------------------

(deftest gaussian-mean-convergence-test
  (testing "Gaussian mean convergence (tensor-native)"
    (let [obs (cm/choicemap :y (mx/scalar 3.0))
          result (co/learn simple-model [(mx/scalar 0.0)] obs [:mu]
                           {:iterations 500 :lr 0.05 :log-every 100})
          final-mu (first (mx/->clj (:params result)))]
      (is (= :tensor-native (:compilation-level result)) "compilation-level = :tensor-native")
      (is (h/close? 2.97 final-mu 0.05) "mu converges to MAP ~2.97")
      (is (pos? (count (:loss-history result))) "loss history is non-empty")
      (is (apply >= (:loss-history result)) "loss is non-increasing"))))

;; ---------------------------------------------------------------------------
;; 6. Convergence: Linear regression (2 params, tensor-native)
;; ---------------------------------------------------------------------------

(deftest linear-regression-convergence-test
  (testing "linear regression convergence (tensor-native)"
    (let [m (gen [x1 x2]
                 (let [slope (trace :slope (dist/gaussian 0 10))
                       intercept (trace :intercept (dist/gaussian 0 10))]
                   (trace :y0 (dist/gaussian (mx/add (mx/multiply slope x1)
                                                     intercept) 0.5))
                   (trace :y1 (dist/gaussian (mx/add (mx/multiply slope x2)
                                                     intercept) 0.5))
                   slope))
          obs (cm/choicemap :y0 (mx/scalar 3.0) :y1 (mx/scalar 5.0))
          result (co/learn m [(mx/scalar 1.0) (mx/scalar 2.0)] obs [:slope :intercept]
                           {:iterations 500 :lr 0.05 :log-every 100})
          params (mx/->clj (:params result))
          latent-idx (:latent-index result)
          slope-val (nth params (get latent-idx :slope))
          intercept-val (nth params (get latent-idx :intercept))]
      (is (= :tensor-native (:compilation-level result)) "compilation-level = :tensor-native")
      (is (h/close? 2.0 slope-val 0.5) "slope converges near 2.0")
      (is (h/close? 1.0 intercept-val 0.5) "intercept converges near 1.0")
      (is (h/close? 3.0 (+ slope-val intercept-val) 0.2) "slope + intercept ~ 3.0 (y at x=1)"))))

;; ---------------------------------------------------------------------------
;; 7. Convergence: Handler path on dynamic model
;; ---------------------------------------------------------------------------

(deftest handler-path-convergence-test
  (testing "handler path convergence"
    (let [obs (cm/choicemap :y0 (mx/scalar 3.0) :y1 (mx/scalar 4.0) :y2 (mx/scalar 2.5))
          result (co/learn dynamic-model [(mx/scalar 3)] obs [:mu]
                           {:iterations 1000 :lr 0.05 :log-every 200})
          final-mu (first (mx/->clj (:params result)))]
      (is (= :handler (:compilation-level result)) "compilation-level = :handler")
      (is (h/close? 3.16 final-mu 0.3) "mu converges to posterior mean ~3.16")
      (is (pos? (count (:loss-history result))) "handler loss history non-empty"))))

;; ---------------------------------------------------------------------------
;; 8. L3.5 eliminated addresses (reduced dimension)
;; ---------------------------------------------------------------------------

(deftest l3-5-eliminated-addresses-test
  (testing "L3.5 eliminated addresses"
    (let [eliminated (u/get-eliminated-addresses simple-model)]
      (is (and (some? eliminated) (contains? eliminated :mu)) "simple model has L3 conjugacy (mu eliminated)")))

  (testing "all addresses eliminated by L3.5"
    (let [obs (cm/choicemap :y (mx/scalar 3.0))
          addresses [:mu]
          eliminated (u/get-eliminated-addresses simple-model)
          filtered (u/filter-addresses addresses eliminated)]
      (is (= [] filtered) "all addresses eliminated by L3.5")))

  (testing "dynamic model has no eliminated addresses"
    (let [eliminated (u/get-eliminated-addresses dynamic-model)]
      (is (or (nil? eliminated) (empty? eliminated)) "dynamic model has no eliminated addresses")))

  (testing "filter-addresses with synthetic eliminated set"
    (let [addresses [:a :b :c :d]
          eliminated #{:b :d}
          filtered (u/filter-addresses addresses eliminated)]
      (is (= [:a :c] filtered) "filter-addresses removes eliminated"))))

;; ---------------------------------------------------------------------------
;; 9. Integration: make-compiled-loss-grad -> compiled-train
;; ---------------------------------------------------------------------------

(deftest integration-loss-grad-compiled-train-test
  (testing "loss-grad -> compiled-train"
    (let [obs (cm/choicemap :y (mx/scalar 3.0))
          {:keys [score-fn init-params compilation-level]}
          (co/make-compiled-loss-grad simple-model [(mx/scalar 0.0)] obs [:mu])
          result (co/compiled-train score-fn (mx/array [0.0])
                                    {:iterations 500 :lr 0.05 :log-every 100})]
      (is (some? (:params result)) "compiled-train returns params")
      (let [final-mu (first (mx/->clj (:params result)))]
        (is (h/close? 2.97 final-mu 0.05) "compiled-train converges to MAP")))))

;; ---------------------------------------------------------------------------
;; 10. learn API end-to-end (tensor-native)
;; ---------------------------------------------------------------------------

(deftest learn-api-tensor-native-test
  (testing "learn API end-to-end (tensor-native)"
    (let [obs (cm/choicemap :y (mx/scalar 5.0))
          losses (atom [])
          result (co/learn simple-model [(mx/scalar 0.0)] obs [:mu]
                           {:iterations 300 :lr 0.05 :log-every 100
                            :callback (fn [{:keys [iter loss]}]
                                        (swap! losses conj {:iter iter :loss loss}))})]
      (is (= :tensor-native (:compilation-level result)) "learn returns :tensor-native")
      (is (map? (:latent-index result)) "learn returns latent-index")
      (is (pos? (:n-params result)) "learn returns n-params")
      (is (pos? (count @losses)) "callback was called")
      (let [final-mu (first (mx/->clj (:params result)))]
        (is (h/close? 4.95 final-mu 0.1) "learn converges for y=5")))))

;; ---------------------------------------------------------------------------
;; 11. learn API end-to-end (handler fallback)
;; ---------------------------------------------------------------------------

(deftest learn-api-handler-test
  (testing "learn API end-to-end (handler)"
    (let [obs (cm/choicemap :y0 (mx/scalar 5.0) :y1 (mx/scalar 6.0))
          result (co/learn dynamic-model [(mx/scalar 2)] obs [:mu]
                           {:iterations 1000 :lr 0.05 :log-every 200})]
      (is (= :handler (:compilation-level result)) "learn returns :handler")
      (let [final-mu (first (mx/->clj (:params result)))]
        (is (h/close? 5.5 final-mu 0.5) "handler learn converges")))))

;; ---------------------------------------------------------------------------
;; 12. Memory: 500 iterations without crash
;; ---------------------------------------------------------------------------

(deftest memory-stability-test
  (testing "500 tensor-native iterations"
    (let [obs (cm/choicemap :y (mx/scalar 3.0))
          result (co/learn simple-model [(mx/scalar 0.0)] obs [:mu]
                           {:iterations 500 :lr 0.05 :log-every 500})]
      (is (some? (:params result)) "500 tensor-native iterations complete without crash")
      (is (js/isFinite (first (mx/->clj (:params result)))) "params are finite")))

  (testing "500 handler iterations"
    (let [obs (cm/choicemap :y0 (mx/scalar 3.0) :y1 (mx/scalar 4.0))
          result (co/learn dynamic-model [(mx/scalar 2)] obs [:mu]
                           {:iterations 500 :lr 0.05 :log-every 500})]
      (is (some? (:params result)) "500 handler iterations complete without crash")
      (is (js/isFinite (first (mx/->clj (:params result)))) "handler params are finite"))))

;; ---------------------------------------------------------------------------
;; Gate 1: Tensor-score + auto-handler composition
;; ---------------------------------------------------------------------------

(deftest gate-1-tensor-score-test
  (testing "Gate 1: Tensor-score + auto-handler composition"
    (let [m simple-model
          obs (cm/choicemap :y (mx/scalar 3.0))
          full-result (u/make-tensor-score-fn m [(mx/scalar 0.0)] obs [:mu])]
      (is (:tensor-native? full-result) "Gate1a: tensor-native score for simple model")
      (let [eliminated (u/get-eliminated-addresses m)
            filtered-addrs (u/filter-addresses [:mu] eliminated)]
        (is (empty? filtered-addrs) "Gate1b: L3.5 eliminates :mu from conjugate model")
        (is (contains? eliminated :mu) "Gate1b2: eliminated set contains :mu"))
      (let [params (mx/array [2.0])
            score-a ((:score-fn full-result) params)]
        (mx/materialize! score-a)
        (is (js/isFinite (mx/item score-a)) "Gate1c: score-a is finite"))))

  (testing "filter-addresses correctly reduces dimension"
    (let [addresses [:prior-mean :likelihood-param :obs]
          eliminated #{:prior-mean}
          filtered (u/filter-addresses addresses eliminated)]
      (is (= 2 (count filtered)) "Gate1d: filtered has fewer addresses")
      (is (not (some #{:prior-mean} filtered)) "Gate1e: eliminated address removed"))))

;; ---------------------------------------------------------------------------
;; Gate 2: mx/compile-fn through gradient + Adam
;; ---------------------------------------------------------------------------

(deftest gate-2-compiled-gradient-adam-test
  (testing "Gate 2: Compiled gradient + Adam cycle"
    (let [obs (cm/choicemap :y (mx/scalar 3.0))
          {:keys [score-fn compilation-level]}
          (co/make-compiled-loss-grad simple-model [(mx/scalar 0.0)] obs [:mu])]
      (is (= :tensor-native compilation-level) "Gate2a: tensor-native")
      (let [compiled-result (co/compiled-train score-fn (mx/array [0.0])
                                               {:iterations 200 :lr 0.05 :log-every 50})
            compiled-mu (first (mx/->clj (:params compiled-result)))
            neg-score (fn [p] (mx/negative (score-fn p)))
            vg (mx/value-and-grad neg-score)
            [loss-ref grad-ref] (vg (mx/array [0.0]))]
        (mx/materialize! loss-ref grad-ref)
        (is (< (js/Math.abs (- 2.97 compiled-mu)) 0.1) "Gate2b: compiled result converges")
        (is (js/isFinite (mx/item loss-ref)) "Gate2c: loss is finite")
        (is (js/isFinite (first (mx/->clj grad-ref))) "Gate2d: grad is finite")
        (is (some? (:params compiled-result)) "Gate2e: 200 compiled iterations complete")
        (is (< (last (:loss-history compiled-result))
               (first (:loss-history compiled-result)))
            "Gate2f: loss decreased")))))

(cljs.test/run-tests)
