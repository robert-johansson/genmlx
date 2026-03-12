(ns genmlx.compiled-optimizer-test
  "Tests for WP-0: Compiled optimization step.
   Covers make-compiled-opt-step, compiled-train, and compiled-generate score path."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.learning :as learn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.gen :refer [gen]]
            [genmlx.dist :as dist]
            [genmlx.inference.util :as u]
            [genmlx.inference.compiled-optimizer :as co]))

;; ---------------------------------------------------------------------------
;; Test helpers
;; ---------------------------------------------------------------------------

(def pass-count (volatile! 0))
(def fail-count (volatile! 0))

(defn assert-true [msg v]
  (if v
    (do (vswap! pass-count inc) (println (str "  PASS: " msg)))
    (do (vswap! fail-count inc) (println (str "  FAIL: " msg)))))

(defn assert-close [msg expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (do (vswap! pass-count inc) (println (str "  PASS: " msg)))
      (do (vswap! fail-count inc)
          (println (str "  FAIL: " msg
                        " expected=" expected
                        " actual=" actual
                        " diff=" diff))))))

;; ---------------------------------------------------------------------------
;; Score functions used across tests
;; ---------------------------------------------------------------------------

(defn quadratic-score
  "f(x) = -sum(x^2). Maximum at x=0."
  [params]
  (mx/negative (mx/sum (mx/multiply params params))))

(defn gaussian-score
  "Log-likelihood of data under N(mu, exp(log-sigma)).
   params = [mu, log-sigma]."
  [data]
  (fn [params]
    (let [mu (mx/index params 0)
          log-sig (mx/index params 1)
          sig (mx/exp log-sig)
          residuals (mx/subtract data mu)
          z (mx/divide residuals sig)
          log-probs (mx/subtract
                     (mx/subtract
                      (mx/multiply (mx/scalar -0.5) (mx/multiply z z))
                      log-sig)
                     (mx/scalar (* 0.5 (js/Math.log (* 2.0 js/Math.PI)))))]
      (mx/sum log-probs))))

;; ---------------------------------------------------------------------------
;; 1. make-compiled-opt-step tests
;; ---------------------------------------------------------------------------

(println "\n=== 1. make-compiled-opt-step basics ===")

(let [step (co/make-compiled-opt-step quadratic-score {})]
  (assert-true "returns a function" (fn? step)))

(let [step (co/make-compiled-opt-step quadratic-score {:lr 0.01})
      params (mx/array [3.0 -2.0])
      m (mx/zeros [2])
      v (mx/zeros [2])
      result (step params m v (mx/scalar 1.0))
      np (aget result 0)
      loss (aget result 3)]
  (mx/materialize! np loss)
  (let [np-clj (mx/->clj np)]
    (assert-true "single step: params move toward 0 (dim 0)"
                 (< (js/Math.abs (nth np-clj 0)) 3.0))
    (assert-true "single step: params move toward 0 (dim 1)"
                 (< (js/Math.abs (nth np-clj 1)) 2.0))
    (assert-close "single step: loss = sum(x^2) = 13" 13.0 (mx/item loss) 0.01)))

;; ---------------------------------------------------------------------------
;; 2. Adam bias correction tests
;; ---------------------------------------------------------------------------

(println "\n=== 2. Adam bias correction ===")

;; At t=1 with x=[1], grad of x^2 = 2, so:
;; m_1 = (1-beta1)*grad = 0.1*2 = 0.2
;; v_1 = (1-beta2)*grad^2 = 0.001*4 = 0.004
(let [step (co/make-compiled-opt-step quadratic-score {:lr 0.001 :beta1 0.9 :beta2 0.999})
      params (mx/array [1.0])
      m (mx/zeros [1])
      v (mx/zeros [1])
      result (step params m v (mx/scalar 1.0))
      new-m (aget result 1)
      new-v (aget result 2)]
  (mx/materialize! new-m new-v)
  (assert-close "t=1: m_1 = 0.2" 0.2 (mx/item new-m) 0.001)
  (assert-close "t=1: v_1 = 0.004" 0.004 (mx/item new-v) 0.0001))

;; Run 10 steps and verify moments are reasonable
(let [step (co/make-compiled-opt-step quadratic-score {:lr 0.001 :beta1 0.9 :beta2 0.999})
      result (loop [i 0 p (mx/array [1.0]) m (mx/zeros [1]) v (mx/zeros [1])]
               (if (>= i 10)
                 {:p p :m m :v v}
                 (let [r (step p m v (mx/scalar (double (inc i))))]
                   (recur (inc i) (aget r 0) (aget r 1) (aget r 2)))))]
  (mx/materialize! (:m result) (:v result) (:p result))
  (assert-true "t=10: m is positive (gradient pushes toward 0)"
               (pos? (mx/item (:m result))))
  (assert-true "t=10: v is positive" (pos? (mx/item (:v result))))
  (assert-true "t=10: params decreased" (< (mx/item (:p result)) 1.0)))

;; Run 100 steps — bias correction keeps working at high t
(let [step (co/make-compiled-opt-step quadratic-score {:lr 0.001 :beta1 0.9 :beta2 0.999})
      result (loop [i 0 p (mx/array [1.0]) m (mx/zeros [1]) v (mx/zeros [1])]
               (if (>= i 100)
                 {:p p :m m :v v}
                 (let [r (step p m v (mx/scalar (double (inc i))))]
                   (recur (inc i) (aget r 0) (aget r 1) (aget r 2)))))]
  (mx/materialize! (:p result))
  (assert-true "t=100: params closer to 0 than at t=10"
               (< (js/Math.abs (mx/item (:p result))) 1.0)))

;; ---------------------------------------------------------------------------
;; 3. Convergence on quadratic objective
;; ---------------------------------------------------------------------------

(println "\n=== 3. Quadratic convergence ===")

(let [step (co/make-compiled-opt-step quadratic-score {:lr 0.01})
      result (loop [i 0 p (mx/array [3.0 -2.0]) m (mx/zeros [2]) v (mx/zeros [2]) losses []]
               (if (>= i 100)
                 {:p p :losses losses}
                 (let [r (step p m v (mx/scalar (double (inc i))))
                       loss (aget r 3)]
                   (mx/materialize! loss)
                   (recur (inc i) (aget r 0) (aget r 1) (aget r 2)
                          (conj losses (mx/item loss))))))]
  (assert-true "100 steps: loss decreased"
               (< (last (:losses result)) (first (:losses result))))
  (assert-true "100 steps: loss monotonically non-increasing"
               (every? true? (map <= (rest (:losses result)) (:losses result))))
  (mx/materialize! (:p result))
  (let [final (mx/->clj (:p result))]
    (assert-true "100 steps: all params < 2.5 (tighter than initial [3,-2])"
                 (every? #(< (js/Math.abs %) 2.5) final))))

;; ---------------------------------------------------------------------------
;; 4. Convergence on Gaussian score function
;; ---------------------------------------------------------------------------

(println "\n=== 4. Gaussian score convergence ===")

(let [data (mx/array [1.5 2.3 1.8 2.1 2.5 1.9 2.0 2.2 1.7 2.4])
      ;; Sample mean = 2.04
      score-fn (gaussian-score data)
      step (co/make-compiled-opt-step score-fn {:lr 0.01})
      result (loop [i 0 p (mx/array [0.0 1.0]) m (mx/zeros [2]) v (mx/zeros [2])]
               (if (>= i 1000)
                 p
                 (let [r (step p m v (mx/scalar (double (inc i))))]
                   (recur (inc i) (aget r 0) (aget r 1) (aget r 2)))))]
  (mx/materialize! result)
  (let [final (mx/->clj result)
        mu (nth final 0)
        sigma (js/Math.exp (nth final 1))]
    (assert-close "Gaussian: mu converges to sample mean" 2.04 mu 0.05)
    (assert-true "Gaussian: sigma > 0" (pos? sigma))
    (assert-true "Gaussian: sigma < 1.0 (MLE for small sample)" (< sigma 1.0))))

;; ---------------------------------------------------------------------------
;; 5. Parameter dimensionality: K=1, K=5, K=20
;; ---------------------------------------------------------------------------

(println "\n=== 5. Parameter dimensionality ===")

(let [step (co/make-compiled-opt-step quadratic-score {:lr 0.01})
      ;; K=1
      r1 (step (mx/array [5.0]) (mx/zeros [1]) (mx/zeros [1]) (mx/scalar 1.0))
      np1 (aget r1 0)]
  (mx/materialize! np1)
  (assert-true "K=1: output shape [1]" (= [1] (mx/shape np1)))
  (assert-true "K=1: param moved toward 0" (< (js/Math.abs (mx/item np1)) 5.0)))

(let [step (co/make-compiled-opt-step quadratic-score {:lr 0.01})
      r5 (step (mx/array [1.0 2.0 3.0 4.0 5.0])
               (mx/zeros [5]) (mx/zeros [5]) (mx/scalar 1.0))
      np5 (aget r5 0)]
  (mx/materialize! np5)
  (assert-true "K=5: output shape [5]" (= [5] (mx/shape np5))))

(let [step (co/make-compiled-opt-step quadratic-score {:lr 0.01})
      r20 (step (mx/array (vec (range 1.0 21.0)))
                (mx/zeros [20]) (mx/zeros [20]) (mx/scalar 1.0))
      np20 (aget r20 0)]
  (mx/materialize! np20)
  (assert-true "K=20: output shape [20]" (= [20] (mx/shape np20))))

;; ---------------------------------------------------------------------------
;; 6. Results match learning/train within tolerance
;; ---------------------------------------------------------------------------

(println "\n=== 6. Match learning/train ===")

(let [init-params (mx/array [3.0 -2.0 1.5])
      iters 500

      ;; learning/train: loss-grad-fn
      loss-grad-fn (fn [params _key]
                     (let [loss-fn (fn [p] (mx/sum (mx/multiply p p)))
                           grad-fn (mx/grad loss-fn)
                           loss (loss-fn params)
                           grad (grad-fn params)]
                       {:loss loss :grad grad}))

      result-old (learn/train {:iterations iters :lr 0.01 :optimizer :adam}
                              loss-grad-fn init-params)

      result-new (co/compiled-train quadratic-score init-params
                                    {:iterations iters :lr 0.01 :log-every 1})]
  (mx/materialize! (:params result-old) (:params result-new))
  (let [old-final (mx/->clj (:params result-old))
        new-final (mx/->clj (:params result-new))]
    (assert-close "match learning/train: param 0" (nth old-final 0) (nth new-final 0) 0.01)
    (assert-close "match learning/train: param 1" (nth old-final 1) (nth new-final 1) 0.01)
    (assert-close "match learning/train: param 2" (nth old-final 2) (nth new-final 2) 0.01)
    (assert-close "match learning/train: final loss"
                  (last (:loss-history result-old))
                  (last (:loss-history result-new)) 0.01)))

;; ---------------------------------------------------------------------------
;; 7. compiled-train tests
;; ---------------------------------------------------------------------------

(println "\n=== 7. compiled-train ===")

(let [result (co/compiled-train quadratic-score (mx/array [3.0 -2.0])
                                {:iterations 200 :lr 0.01 :log-every 50})]
  (assert-true "compiled-train: returns :params" (some? (:params result)))
  (assert-true "compiled-train: returns :loss-history" (vector? (:loss-history result)))
  (mx/materialize! (:params result))
  (assert-true "compiled-train: params shape [2]" (= [2] (mx/shape (:params result))))
  (let [final (mx/->clj (:params result))]
    (assert-true "compiled-train: converged toward 0"
                 (and (< (js/Math.abs (nth final 0)) 3.0)
                      (< (js/Math.abs (nth final 1)) 2.0))))
  (assert-true "compiled-train: losses decrease"
               (< (last (:loss-history result)) (first (:loss-history result)))))

;; log-every controls number of recorded losses
(let [result (co/compiled-train quadratic-score (mx/array [3.0 -2.0])
                                {:iterations 100 :lr 0.01 :log-every 10})]
  ;; log at i=0, i=9(mod10=0), i=19, ... i=99 => 11 entries
  (assert-true "log-every=10, 100 iters: 11 loss entries"
               (= 11 (count (:loss-history result)))))

(let [result (co/compiled-train quadratic-score (mx/array [3.0 -2.0])
                                {:iterations 100 :lr 0.01 :log-every 25})]
  ;; log at i=0, i=24, i=49, i=74, i=99 => 5 entries
  (assert-true "log-every=25, 100 iters: 5 loss entries"
               (= 5 (count (:loss-history result)))))

;; callback is invoked
(let [cb-log (volatile! [])
      result (co/compiled-train quadratic-score (mx/array [3.0 -2.0])
                                {:iterations 50 :lr 0.01 :log-every 10
                                 :callback (fn [info] (vswap! cb-log conj (:iter info)))})]
  (assert-true "callback invoked at log points"
               (= 6 (count @cb-log)))
  (assert-true "callback iter values correct"
               (= [0 9 19 29 39 49] @cb-log)))

;; ---------------------------------------------------------------------------
;; 7b. Default opts (empty map uses defaults)
;; ---------------------------------------------------------------------------

(println "\n=== 7b. Default opts ===")

(let [result (co/compiled-train quadratic-score (mx/array [3.0 -2.0]) {})]
  (assert-true "default opts: returns :params" (some? (:params result)))
  (assert-true "default opts: params shape [2]" (= [2] (mx/shape (:params result))))
  ;; 1000 iters with log-every=50: log at i=0, then i=49,99,...,999 => 21 entries
  (assert-true "default opts: 21 loss entries (1000 iters, log-every=50)"
               (= 21 (count (:loss-history result)))))

;; ---------------------------------------------------------------------------
;; 8. Periodic cleanup runs without error
;; ---------------------------------------------------------------------------

(println "\n=== 8. Periodic cleanup ===")

(let [result (co/compiled-train quadratic-score (mx/array [1.0 2.0 3.0])
                                {:iterations 200 :lr 0.01 :log-every 50})]
  (assert-true "200 iterations with cleanup: no error"
               (some? (:params result)))
  (assert-true "200 iterations: losses recorded"
               (pos? (count (:loss-history result)))))

;; ---------------------------------------------------------------------------
;; 9. Memory stays bounded over 500 iterations
;; ---------------------------------------------------------------------------

(println "\n=== 9. Memory bounded ===")

(mx/reset-peak-memory!)
(let [mem-before (mx/get-active-memory)
      result (co/compiled-train quadratic-score
                                (mx/array (vec (range 1.0 11.0)))
                                {:iterations 500 :lr 0.01 :log-every 100})
      mem-after (mx/get-active-memory)
      mem-peak (mx/get-peak-memory)]
  (assert-true "500 iters, D=10: peak memory < 1MB"
               (< mem-peak (* 1024 1024)))
  (assert-true "500 iters: converging"
               (< (last (:loss-history result)) (first (:loss-history result)))))

;; ---------------------------------------------------------------------------
;; 10. Compiled-generate score function (middle tier)
;; ---------------------------------------------------------------------------

(println "\n=== 10. Compiled-generate score function ===")

;; Non-conjugate branch model: uniform prior + gaussian likelihood.
;; Branch → static?=false → no tensor-native score.
;; Branch-rewriting → compiled-generate works.
;; Uniform prior → no L3/3.5 conjugacy → no elimination.
(def cg-model
  (gen [flag]
    (let [mu (trace :mu (dist/uniform -10 10))]
      (if flag
        (trace :obs (dist/gaussian mu 1))
        (trace :obs (dist/gaussian mu 5)))
      mu)))

(def cg-obs (cm/choicemap :obs (mx/scalar 5.0)))

;; 10a. Model has compiled-generate but NOT tensor-native, not eliminated
(let [schema (:schema cg-model)]
  (assert-true "cg-model has :compiled-generate"
    (some? (:compiled-generate schema)))
  (assert-true "cg-model has-branches? = true"
    (:has-branches? schema))
  (assert-true "no eliminated addresses"
    (nil? (u/get-eliminated-addresses cg-model))))

;; 10b. make-compiled-generate-score-fn returns non-nil
(let [result (u/make-compiled-generate-score-fn cg-model [true] cg-obs [:mu])]
  (assert-true "make-compiled-generate-score-fn returns result"
    (some? result))
  (assert-true "result has :score-fn" (fn? (:score-fn result)))
  (assert-true "result has :latent-index" (map? (:latent-index result))))

;; 10c. Score values match handler (both go through compiled-gen under the hood)
(let [{cg-score-fn :score-fn} (u/make-compiled-generate-score-fn cg-model [true] cg-obs [:mu])
      handler-score-fn (u/make-score-fn (dyn/auto-key cg-model) [true] cg-obs [:mu])
      params (mx/array [3.0])
      cg-val (mx/item (cg-score-fn params))
      handler-val (mx/item (handler-score-fn params))]
  (assert-close "cg score matches handler" handler-val cg-val 0.01))

;; 10d. make-tensor-score-fn dispatches to compiled-generate (not tensor-native)
(let [result (u/make-tensor-score-fn cg-model [true] cg-obs [:mu])]
  (assert-true "tensor-native? is false" (not (:tensor-native? result)))
  (assert-true "compiled-generate? is true" (:compiled-generate? result)))

;; 10e. mx/grad produces correct gradients through compiled-generate score
;; Analytical: d/dmu [log U(mu|-10,10) + log N(5|mu,1)]
;; = 0 + (5-mu) = 2.0 at mu=3
(let [{:keys [score-fn]} (u/make-compiled-generate-score-fn cg-model [true] cg-obs [:mu])
      grad-fn (mx/grad score-fn)
      params (mx/array [3.0])
      grad-val (mx/item (grad-fn params))]
  (assert-close "mx/grad through cg score: gradient = 2.0" 2.0 grad-val 0.1))

;; 10f. mx/value-and-grad works
(let [{:keys [score-fn]} (u/make-compiled-generate-score-fn cg-model [true] cg-obs [:mu])
      neg-score (fn [p] (mx/negative (score-fn p)))
      vg (mx/value-and-grad neg-score)
      params (mx/array [3.0])
      [loss grad] (vg params)]
  (mx/materialize! loss grad)
  (assert-true "value-and-grad: loss is positive (neg log-prob)" (pos? (mx/item loss)))
  (assert-close "value-and-grad: grad = -2.0" -2.0 (mx/item grad) 0.1))

;; ---------------------------------------------------------------------------
;; 11. make-compiled-loss-grad with compiled-generate
;; ---------------------------------------------------------------------------

(println "\n=== 11. make-compiled-loss-grad compiled-generate path ===")

(let [result (co/make-compiled-loss-grad cg-model [true] cg-obs [:mu])]
  (assert-true "compilation-level is :compiled-generate"
    (= :compiled-generate (:compilation-level result)))
  (assert-true "has loss-grad-fn" (fn? (:loss-grad-fn result)))
  (assert-true "has score-fn" (fn? (:score-fn result)))
  (assert-true "has init-params" (some? (:init-params result)))
  (assert-true "n-params = 1" (= 1 (:n-params result))))

;; loss-grad-fn returns [loss, grad]
(let [{:keys [loss-grad-fn]} (co/make-compiled-loss-grad cg-model [true] cg-obs [:mu])
      params (mx/array [3.0])
      [loss grad] (loss-grad-fn params)]
  (mx/materialize! loss grad)
  (assert-true "loss-grad-fn: loss > 0" (pos? (mx/item loss)))
  (assert-true "loss-grad-fn: grad shape [1]" (= [1] (mx/shape grad))))

;; ---------------------------------------------------------------------------
;; 12. learn converges via compiled-generate path
;; ---------------------------------------------------------------------------

(println "\n=== 12. learn via compiled-generate ===")

;; MAP for uniform(-10,10) + N(obs=5|mu,1): mu=5 (MLE, uniform is flat)
(let [result (co/learn cg-model [true] cg-obs [:mu]
               {:iterations 200 :lr 0.05 :log-every 50})]
  (mx/materialize! (:params result))
  (assert-true "learn: compilation-level is :compiled-generate"
    (= :compiled-generate (:compilation-level result)))
  (assert-true "learn: has loss-history" (vector? (:loss-history result)))
  (assert-true "learn: loss decreased"
    (< (last (:loss-history result)) (first (:loss-history result))))
  (let [mu-final (mx/item (:params result))]
    (assert-close "learn: mu converges to MAP ≈ 5.0" 5.0 mu-final 0.3)))

;; ---------------------------------------------------------------------------
;; 13. Multi-site branch model
;; ---------------------------------------------------------------------------

(println "\n=== 13. Multi-site compiled-generate ===")

;; Uniform priors (non-conjugate) + branch around trace
(def cg-multi
  (gen [flag]
    (let [slope (trace :slope (dist/uniform -10 10))
          intercept (trace :intercept (dist/uniform -10 10))
          total (mx/add slope intercept)]
      (if flag
        (trace :y (dist/gaussian total 1))
        (trace :y (dist/gaussian total 5)))
      total)))

(def cg-multi-obs (cm/choicemap :y (mx/scalar 4.0)))

(let [result (co/make-compiled-loss-grad cg-multi [true] cg-multi-obs [:slope :intercept])]
  (assert-true "multi-site: compilation-level is :compiled-generate"
    (= :compiled-generate (:compilation-level result)))
  (assert-true "multi-site: n-params = 2" (= 2 (:n-params result))))

(let [result (co/learn cg-multi [true] cg-multi-obs [:slope :intercept]
               {:iterations 300 :lr 0.05 :log-every 100})]
  (mx/materialize! (:params result))
  (let [final (mx/->clj (:params result))
        total (+ (nth final 0) (nth final 1))]
    (assert-true "multi-site: loss decreased"
      (< (last (:loss-history result)) (first (:loss-history result))))
    (assert-close "multi-site: slope + intercept ≈ 4.0" 4.0 total 0.5)))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n=== SUMMARY: " @pass-count "/" (+ @pass-count @fail-count)
              " passed ==="))
(when (pos? @fail-count)
  (println (str "  " @fail-count " FAILURES")))
