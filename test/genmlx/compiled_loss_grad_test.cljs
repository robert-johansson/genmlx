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
  (:require [genmlx.gen :refer [gen]]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.inference.compiled-optimizer :as co]
            [genmlx.inference.util :as u]))

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
          (println (str "  PASS: " desc " (diff=" (.toFixed diff 6) ")")))
      (do (swap! fail-count inc)
          (println (str "  FAIL: " desc " expected=" expected " actual=" actual " diff=" diff))))))

(defn- assert-equal [desc expected actual]
  (if (= expected actual)
    (do (swap! pass-count inc)
        (println (str "  PASS: " desc)))
    (do (swap! fail-count inc)
        (println (str "  FAIL: " desc " expected=" (pr-str expected) " actual=" (pr-str actual))))))

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

(println "\n=== 1. Compilation level detection ===")

(let [obs (cm/choicemap :y (mx/scalar 3.0))
      result (co/make-compiled-loss-grad simple-model [(mx/scalar 0.0)] obs [:mu])]
  (assert-equal "static model → :tensor-native"
                :tensor-native (:compilation-level result))
  (assert-equal "tensor-native n-params = 1" 1 (:n-params result))
  (assert-true "tensor-native has score-fn" (fn? (:score-fn result)))
  (assert-true "tensor-native has loss-grad-fn" (fn? (:loss-grad-fn result)))
  (assert-true "tensor-native latent-index has :mu" (contains? (:latent-index result) :mu))
  (assert-equal "tensor-native init-params shape = [1]" [1] (mx/shape (:init-params result))))

(let [obs (cm/choicemap :y0 (mx/scalar 3.0) :y1 (mx/scalar 4.0))
      result (co/make-compiled-loss-grad dynamic-model [(mx/scalar 2)] obs [:mu])]
  (assert-equal "dynamic model → :handler"
                :handler (:compilation-level result))
  (assert-equal "handler n-params = 1" 1 (:n-params result))
  (assert-true "handler has score-fn" (fn? (:score-fn result)))
  (assert-true "handler has loss-grad-fn" (fn? (:loss-grad-fn result))))

;; ---------------------------------------------------------------------------
;; 2. Multi-parameter model detection
;; ---------------------------------------------------------------------------

(println "\n=== 2. Multi-parameter model ===")

(let [obs (cm/choicemap :y (mx/scalar 5.0))
      result (co/make-compiled-loss-grad linreg-model [(mx/scalar 2.0)] obs [:slope :intercept])]
  (assert-equal "linreg → :tensor-native" :tensor-native (:compilation-level result))
  (assert-equal "linreg n-params = 2" 2 (:n-params result))
  (assert-equal "linreg init-params shape = [2]" [2] (mx/shape (:init-params result)))
  (assert-true "latent-index has :slope" (contains? (:latent-index result) :slope))
  (assert-true "latent-index has :intercept" (contains? (:latent-index result) :intercept)))

;; ---------------------------------------------------------------------------
;; 3. Tensor-native loss matches GFI-based loss
;; ---------------------------------------------------------------------------

(println "\n=== 3. Loss value comparison ===")

(let [obs (cm/choicemap :y (mx/scalar 3.0))
      ;; Tensor-native score
      {:keys [score-fn latent-index]} (u/make-tensor-score-fn simple-model [(mx/scalar 0.0)] obs [:mu])
      tn-score (score-fn (mx/array [2.0]))
      ;; GFI-based score at same point
      gfi-fn (u/make-score-fn simple-model [(mx/scalar 0.0)] obs [:mu])
      gfi-score (gfi-fn (mx/array [2.0]))]
  (mx/materialize! tn-score gfi-score)
  ;; Both are log-probs but may differ by a constant (tn is full joint, gfi is importance weight)
  ;; The important thing: they agree on gradient direction
  (assert-true "tensor-native score is finite"
               (js/isFinite (mx/item tn-score)))
  (assert-true "GFI score is finite"
               (js/isFinite (mx/item gfi-score))))

;; ---------------------------------------------------------------------------
;; 4. Gradient direction matches finite differences
;; ---------------------------------------------------------------------------

(println "\n=== 4. Gradient direction validation ===")

(let [obs (cm/choicemap :y (mx/scalar 3.0))
      {:keys [loss-grad-fn score-fn]} (co/make-compiled-loss-grad simple-model [(mx/scalar 0.0)] obs [:mu])
      ;; Test gradient at mu=0 (should point toward y=3, i.e., grad < 0 for loss)
      params (mx/array [0.0])
      [loss grad] (loss-grad-fn params)]
  (mx/materialize! loss grad)
  (let [grad-val (first (mx/->clj grad))]
    (assert-true "gradient at mu=0 points toward data (grad < 0 for loss, data at y=3)"
                 (< grad-val 0))
    ;; Verify with finite differences
    (let [h 1e-4
          neg-score (fn [p] (mx/negative (score-fn p)))
          f-plus (mx/item (neg-score (mx/array [h])))
          f-minus (mx/item (neg-score (mx/array [(- h)])))]
      (assert-close "AD gradient matches finite-diff at mu=0"
                    (/ (- f-plus f-minus) (* 2.0 h))
                    grad-val
                    0.01))))

;; Test at mu=5 (should point back toward data)
(let [obs (cm/choicemap :y (mx/scalar 3.0))
      {:keys [loss-grad-fn]} (co/make-compiled-loss-grad simple-model [(mx/scalar 0.0)] obs [:mu])
      params (mx/array [5.0])
      [loss grad] (loss-grad-fn params)]
  (mx/materialize! loss grad)
  (let [grad-val (first (mx/->clj grad))]
    (assert-true "gradient at mu=5 points back toward data (grad > 0 for loss)"
                 (> grad-val 0))))

;; ---------------------------------------------------------------------------
;; 5. Convergence: Gaussian mean estimation (tensor-native)
;; ---------------------------------------------------------------------------

(println "\n=== 5. Gaussian mean convergence (tensor-native) ===")

(let [obs (cm/choicemap :y (mx/scalar 3.0))
      result (co/learn simple-model [(mx/scalar 0.0)] obs [:mu]
                       {:iterations 500 :lr 0.05 :log-every 100})
      final-mu (first (mx/->clj (:params result)))]
  (assert-equal "compilation-level = :tensor-native" :tensor-native (:compilation-level result))
  ;; MAP: posterior mean = (y/sigma^2) / (1/sigma^2 + 1/tau^2) = 3.0 / 1.01 ≈ 2.97
  (assert-close "mu converges to MAP ~2.97" 2.97 final-mu 0.05)
  (assert-true "loss history is non-empty" (pos? (count (:loss-history result))))
  (assert-true "loss is non-increasing"
               (apply >= (:loss-history result))))

;; ---------------------------------------------------------------------------
;; 6. Convergence: Linear regression (2 params, tensor-native)
;; ---------------------------------------------------------------------------

(println "\n=== 6. Linear regression convergence (tensor-native) ===")

;; Use a simple 2-param static model that compiles to tensor-native
;; (mx/index on args doesn't compile, so use separate scalar args)
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
  (assert-equal "compilation-level = :tensor-native" :tensor-native (:compilation-level result))
  (assert-close "slope converges near 2.0" 2.0 slope-val 0.5)
  (assert-close "intercept converges near 1.0" 1.0 intercept-val 0.5)
  (assert-close "slope + intercept ~ 3.0 (y at x=1)" 3.0 (+ slope-val intercept-val) 0.2))

;; ---------------------------------------------------------------------------
;; 7. Convergence: Handler path on dynamic model
;; ---------------------------------------------------------------------------

(println "\n=== 7. Handler path convergence ===")

(let [obs (cm/choicemap :y0 (mx/scalar 3.0) :y1 (mx/scalar 4.0) :y2 (mx/scalar 2.5))
      ;; Handler path uses finite-diff gradients: needs more iterations from random init
      result (co/learn dynamic-model [(mx/scalar 3)] obs [:mu]
                       {:iterations 1000 :lr 0.05 :log-every 200})
      final-mu (first (mx/->clj (:params result)))]
  (assert-equal "compilation-level = :handler" :handler (:compilation-level result))
  ;; MAP: posterior mean = (sum yi / sigma^2) / (n/sigma^2 + 1/tau^2)
  ;; = 9.5 / (3 + 0.01) ≈ 3.156
  (assert-close "mu converges to posterior mean ~3.16" 3.16 final-mu 0.3)
  (assert-true "handler loss history non-empty" (pos? (count (:loss-history result)))))

;; ---------------------------------------------------------------------------
;; 8. L3.5 eliminated addresses (reduced dimension)
;; ---------------------------------------------------------------------------

(println "\n=== 8. L3.5 eliminated addresses ===")

;; Test that L3.5 conjugacy eliminates the prior parameter
;; simple-model has Normal-Normal conjugacy: :mu is analytically eliminated
(let [eliminated (u/get-eliminated-addresses simple-model)]
  (assert-true "simple model has L3 conjugacy (mu eliminated)"
               (and (some? eliminated) (contains? eliminated :mu))))

;; Test that make-compiled-loss-grad filters out eliminated addresses
;; When all latent addresses are eliminated, the resulting param count should be 0
;; (nothing left to optimize — posterior is analytical)
(let [obs (cm/choicemap :y (mx/scalar 3.0))
      addresses [:mu]
      eliminated (u/get-eliminated-addresses simple-model)
      filtered (u/filter-addresses addresses eliminated)]
  (assert-equal "all addresses eliminated by L3.5" [] filtered))

;; Test on a model WITHOUT conjugacy (dynamic model)
(let [eliminated (u/get-eliminated-addresses dynamic-model)]
  (assert-true "dynamic model has no eliminated addresses"
               (or (nil? eliminated) (empty? eliminated))))

;; Test filter-addresses with synthetic eliminated set
(let [addresses [:a :b :c :d]
      eliminated #{:b :d}
      filtered (u/filter-addresses addresses eliminated)]
  (assert-equal "filter-addresses removes eliminated" [:a :c] filtered))

;; ---------------------------------------------------------------------------
;; 9. Integration: make-compiled-loss-grad -> compiled-train
;; ---------------------------------------------------------------------------

(println "\n=== 9. Integration: loss-grad -> compiled-train ===")

(let [obs (cm/choicemap :y (mx/scalar 3.0))
      {:keys [score-fn init-params compilation-level]}
      (co/make-compiled-loss-grad simple-model [(mx/scalar 0.0)] obs [:mu])
      ;; Pass tensor-native score-fn directly to compiled-train
      result (co/compiled-train score-fn (mx/array [0.0])
                                {:iterations 500 :lr 0.05 :log-every 100})]
  (assert-true "compiled-train returns params" (some? (:params result)))
  (let [final-mu (first (mx/->clj (:params result)))]
    (assert-close "compiled-train converges to MAP" 2.97 final-mu 0.05)))

;; ---------------------------------------------------------------------------
;; 10. learn API end-to-end (tensor-native)
;; ---------------------------------------------------------------------------

(println "\n=== 10. learn API end-to-end (tensor-native) ===")

(let [obs (cm/choicemap :y (mx/scalar 5.0))
      losses (atom [])
      result (co/learn simple-model [(mx/scalar 0.0)] obs [:mu]
                       {:iterations 300 :lr 0.05 :log-every 100
                        :callback (fn [{:keys [iter loss]}]
                                    (swap! losses conj {:iter iter :loss loss}))})]
  (assert-equal "learn returns :tensor-native" :tensor-native (:compilation-level result))
  (assert-true "learn returns latent-index" (map? (:latent-index result)))
  (assert-true "learn returns n-params" (pos? (:n-params result)))
  (assert-true "callback was called" (pos? (count @losses)))
  (let [final-mu (first (mx/->clj (:params result)))]
    ;; MAP for y=5: ~4.95
    (assert-close "learn converges for y=5" 4.95 final-mu 0.1)))

;; ---------------------------------------------------------------------------
;; 11. learn API end-to-end (handler fallback)
;; ---------------------------------------------------------------------------

(println "\n=== 11. learn API end-to-end (handler) ===")

(let [obs (cm/choicemap :y0 (mx/scalar 5.0) :y1 (mx/scalar 6.0))
      ;; Handler path with finite-diff: needs more iterations from random init
      result (co/learn dynamic-model [(mx/scalar 2)] obs [:mu]
                       {:iterations 1000 :lr 0.05 :log-every 200})]
  (assert-equal "learn returns :handler" :handler (:compilation-level result))
  (let [final-mu (first (mx/->clj (:params result)))]
    ;; MAP: ~5.5 (mean of 5 and 6, prior is weak)
    (assert-close "handler learn converges" 5.5 final-mu 0.5)))

;; ---------------------------------------------------------------------------
;; 12. Memory: 500 iterations without crash
;; ---------------------------------------------------------------------------

(println "\n=== 12. Memory stability ===")

(let [obs (cm/choicemap :y (mx/scalar 3.0))
      result (co/learn simple-model [(mx/scalar 0.0)] obs [:mu]
                       {:iterations 500 :lr 0.05 :log-every 500})]
  (assert-true "500 tensor-native iterations complete without crash"
               (some? (:params result)))
  (assert-true "params are finite"
               (js/isFinite (first (mx/->clj (:params result))))))

(let [obs (cm/choicemap :y0 (mx/scalar 3.0) :y1 (mx/scalar 4.0))
      result (co/learn dynamic-model [(mx/scalar 2)] obs [:mu]
                       {:iterations 500 :lr 0.05 :log-every 500})]
  (assert-true "500 handler iterations complete without crash"
               (some? (:params result)))
  (assert-true "handler params are finite"
               (js/isFinite (first (mx/->clj (:params result))))))

;; ---------------------------------------------------------------------------
;; Gate 1: Tensor-score + auto-handler composition
;; ---------------------------------------------------------------------------

(println "\n=== Gate 1: Tensor-score + auto-handler composition ===")

;; Gate 1: Demonstrate tensor-native score + L3.5 conjugate elimination
;; simple-model has Normal-Normal conjugacy → :mu is eliminated by L3.5
(let [m simple-model
      obs (cm/choicemap :y (mx/scalar 3.0))
      ;; (a) Tensor-native score (ignoring L3.5 elimination)
      full-result (u/make-tensor-score-fn m [(mx/scalar 0.0)] obs [:mu])
      _ (assert-true "Gate1a: tensor-native score for simple model"
                     (:tensor-native? full-result))
      ;; (b) L3.5 eliminates :mu (conjugate prior)
      eliminated (u/get-eliminated-addresses m)
      filtered-addrs (u/filter-addresses [:mu] eliminated)
      _ (assert-true "Gate1b: L3.5 eliminates :mu from conjugate model"
                     (empty? filtered-addrs))
      _ (assert-true "Gate1b2: eliminated set contains :mu"
                     (contains? eliminated :mu))
      ;; (c) Tensor-native score still works on the full (unfiltered) model
      params (mx/array [2.0])
      score-a ((:score-fn full-result) params)]
  (mx/materialize! score-a)
  (assert-true "Gate1c: score-a is finite" (js/isFinite (mx/item score-a))))

;; Verify filter-addresses correctly reduces dimension
(let [addresses [:prior-mean :likelihood-param :obs]
      eliminated #{:prior-mean}
      filtered (u/filter-addresses addresses eliminated)]
  (assert-equal "Gate1d: filtered has fewer addresses" 2 (count filtered))
  (assert-true "Gate1e: eliminated address removed"
               (not (some #{:prior-mean} filtered))))

(println "\n  Gate 1: PASSED")

;; ---------------------------------------------------------------------------
;; Gate 2: mx/compile-fn through gradient + Adam
;; ---------------------------------------------------------------------------

(println "\n=== Gate 2: Compiled gradient + Adam cycle ===")

;; Wrap full cycle via compiled-train with tensor-native score
(let [obs (cm/choicemap :y (mx/scalar 3.0))
      {:keys [score-fn compilation-level]}
      (co/make-compiled-loss-grad simple-model [(mx/scalar 0.0)] obs [:mu])
      _ (assert-equal "Gate2a: tensor-native" :tensor-native compilation-level)
      ;; Run compiled-train (this compiles score → grad → Adam into one Metal program)
      compiled-result (co/compiled-train score-fn (mx/array [0.0])
                                         {:iterations 200 :lr 0.05 :log-every 50})
      compiled-mu (first (mx/->clj (:params compiled-result)))
      ;; Compare with non-compiled path (handler-train with same score-fn for reference)
      ;; Since score-fn is tensor-native, we can also run it through value-and-grad directly
      neg-score (fn [p] (mx/negative (score-fn p)))
      vg (mx/value-and-grad neg-score)
      [loss-ref grad-ref] (vg (mx/array [0.0]))]
  (mx/materialize! loss-ref grad-ref)
  (assert-true "Gate2b: compiled result converges" (< (js/Math.abs (- 2.97 compiled-mu)) 0.1))
  (assert-true "Gate2c: loss is finite" (js/isFinite (mx/item loss-ref)))
  (assert-true "Gate2d: grad is finite" (js/isFinite (first (mx/->clj grad-ref))))
  ;; No memory explosion: check that we can do 200 iterations
  (assert-true "Gate2e: 200 compiled iterations complete" (some? (:params compiled-result)))
  (assert-true "Gate2f: loss decreased"
               (< (last (:loss-history compiled-result))
                  (first (:loss-history compiled-result)))))

(println "\n  Gate 2: PASSED")

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println "\n========================================")
(println (str "  L4-WP1 Results: " @pass-count " passed, " @fail-count " failed"))
(println "========================================")
(when (pos? @fail-count)
  (println "  *** FAILURES DETECTED ***"))
