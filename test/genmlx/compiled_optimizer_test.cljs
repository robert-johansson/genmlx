(ns genmlx.compiled-optimizer-test
  "Tests for WP-0: Compiled optimization step.
   Covers make-compiled-opt-step, compiled-train, and compiled-generate score path."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
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

(deftest make-compiled-opt-step-basics-test
  (testing "returns a function"
    (let [step (co/make-compiled-opt-step quadratic-score {})]
      (is (fn? step) "returns a function")))

  (testing "single step moves params toward optimum"
    (let [step (co/make-compiled-opt-step quadratic-score {:lr 0.01})
          params (mx/array [3.0 -2.0])
          m (mx/zeros [2])
          v (mx/zeros [2])
          result (step params m v (mx/scalar 1.0))
          np (aget result 0)
          loss (aget result 3)]
      (mx/materialize! np loss)
      (let [np-clj (mx/->clj np)]
        (is (< (js/Math.abs (nth np-clj 0)) 3.0) "single step: params move toward 0 (dim 0)")
        (is (< (js/Math.abs (nth np-clj 1)) 2.0) "single step: params move toward 0 (dim 1)")
        (is (h/close? 13.0 (mx/item loss) 0.01) "single step: loss = sum(x^2) = 13")))))

;; ---------------------------------------------------------------------------
;; 2. Adam bias correction tests
;; ---------------------------------------------------------------------------

(deftest adam-bias-correction-test
  (testing "t=1 moments"
    (let [step (co/make-compiled-opt-step quadratic-score {:lr 0.001 :beta1 0.9 :beta2 0.999})
          params (mx/array [1.0])
          m (mx/zeros [1])
          v (mx/zeros [1])
          result (step params m v (mx/scalar 1.0))
          new-m (aget result 1)
          new-v (aget result 2)]
      (mx/materialize! new-m new-v)
      (is (h/close? 0.2 (mx/item new-m) 0.001) "t=1: m_1 = 0.2")
      (is (h/close? 0.004 (mx/item new-v) 0.0001) "t=1: v_1 = 0.004")))

  (testing "t=10 moments"
    (let [step (co/make-compiled-opt-step quadratic-score {:lr 0.001 :beta1 0.9 :beta2 0.999})
          result (loop [i 0 p (mx/array [1.0]) m (mx/zeros [1]) v (mx/zeros [1])]
                   (if (>= i 10)
                     {:p p :m m :v v}
                     (let [r (step p m v (mx/scalar (double (inc i))))]
                       (recur (inc i) (aget r 0) (aget r 1) (aget r 2)))))]
      (mx/materialize! (:m result) (:v result) (:p result))
      (is (pos? (mx/item (:m result))) "t=10: m is positive (gradient pushes toward 0)")
      (is (pos? (mx/item (:v result))) "t=10: v is positive")
      (is (< (mx/item (:p result)) 1.0) "t=10: params decreased")))

  (testing "t=100 bias correction"
    (let [step (co/make-compiled-opt-step quadratic-score {:lr 0.001 :beta1 0.9 :beta2 0.999})
          result (loop [i 0 p (mx/array [1.0]) m (mx/zeros [1]) v (mx/zeros [1])]
                   (if (>= i 100)
                     {:p p :m m :v v}
                     (let [r (step p m v (mx/scalar (double (inc i))))]
                       (recur (inc i) (aget r 0) (aget r 1) (aget r 2)))))]
      (mx/materialize! (:p result))
      (is (< (js/Math.abs (mx/item (:p result))) 1.0) "t=100: params closer to 0 than at t=10"))))

;; ---------------------------------------------------------------------------
;; 3. Convergence on quadratic objective
;; ---------------------------------------------------------------------------

(deftest quadratic-convergence-test
  (testing "quadratic convergence"
    (let [step (co/make-compiled-opt-step quadratic-score {:lr 0.01})
          result (loop [i 0 p (mx/array [3.0 -2.0]) m (mx/zeros [2]) v (mx/zeros [2]) losses []]
                   (if (>= i 100)
                     {:p p :losses losses}
                     (let [r (step p m v (mx/scalar (double (inc i))))
                           loss (aget r 3)]
                       (mx/materialize! loss)
                       (recur (inc i) (aget r 0) (aget r 1) (aget r 2)
                              (conj losses (mx/item loss))))))]
      (is (< (last (:losses result)) (first (:losses result))) "100 steps: loss decreased")
      (is (every? true? (map <= (rest (:losses result)) (:losses result))) "100 steps: loss monotonically non-increasing")
      (mx/materialize! (:p result))
      (let [final (mx/->clj (:p result))]
        (is (every? #(< (js/Math.abs %) 2.5) final) "100 steps: all params < 2.5 (tighter than initial [3,-2])")))))

;; ---------------------------------------------------------------------------
;; 4. Convergence on Gaussian score function
;; ---------------------------------------------------------------------------

(deftest gaussian-score-convergence-test
  (testing "Gaussian score convergence"
    (let [data (mx/array [1.5 2.3 1.8 2.1 2.5 1.9 2.0 2.2 1.7 2.4])
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
        (is (h/close? 2.04 mu 0.05) "Gaussian: mu converges to sample mean")
        (is (pos? sigma) "Gaussian: sigma > 0")
        (is (< sigma 1.0) "Gaussian: sigma < 1.0 (MLE for small sample)")))))

;; ---------------------------------------------------------------------------
;; 5. Parameter dimensionality: K=1, K=5, K=20
;; ---------------------------------------------------------------------------

(deftest parameter-dimensionality-test
  (testing "K=1"
    (let [step (co/make-compiled-opt-step quadratic-score {:lr 0.01})
          r1 (step (mx/array [5.0]) (mx/zeros [1]) (mx/zeros [1]) (mx/scalar 1.0))
          np1 (aget r1 0)]
      (mx/materialize! np1)
      (is (= [1] (mx/shape np1)) "K=1: output shape [1]")
      (is (< (js/Math.abs (mx/item np1)) 5.0) "K=1: param moved toward 0")))

  (testing "K=5"
    (let [step (co/make-compiled-opt-step quadratic-score {:lr 0.01})
          r5 (step (mx/array [1.0 2.0 3.0 4.0 5.0])
                   (mx/zeros [5]) (mx/zeros [5]) (mx/scalar 1.0))
          np5 (aget r5 0)]
      (mx/materialize! np5)
      (is (= [5] (mx/shape np5)) "K=5: output shape [5]")))

  (testing "K=20"
    (let [step (co/make-compiled-opt-step quadratic-score {:lr 0.01})
          r20 (step (mx/array (vec (range 1.0 21.0)))
                    (mx/zeros [20]) (mx/zeros [20]) (mx/scalar 1.0))
          np20 (aget r20 0)]
      (mx/materialize! np20)
      (is (= [20] (mx/shape np20)) "K=20: output shape [20]"))))

;; ---------------------------------------------------------------------------
;; 6. Results match learning/train within tolerance
;; ---------------------------------------------------------------------------

(deftest match-learning-train-test
  (testing "match learning/train"
    (let [init-params (mx/array [3.0 -2.0 1.5])
          iters 500
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
        (is (h/close? (nth old-final 0) (nth new-final 0) 0.01) "match learning/train: param 0")
        (is (h/close? (nth old-final 1) (nth new-final 1) 0.01) "match learning/train: param 1")
        (is (h/close? (nth old-final 2) (nth new-final 2) 0.01) "match learning/train: param 2")
        (is (h/close? (last (:loss-history result-old))
                      (last (:loss-history result-new)) 0.01)
            "match learning/train: final loss")))))

;; ---------------------------------------------------------------------------
;; 7. compiled-train tests
;; ---------------------------------------------------------------------------

(deftest compiled-train-test
  (testing "basic compiled-train"
    (let [result (co/compiled-train quadratic-score (mx/array [3.0 -2.0])
                                    {:iterations 200 :lr 0.01 :log-every 50})]
      (is (some? (:params result)) "compiled-train: returns :params")
      (is (vector? (:loss-history result)) "compiled-train: returns :loss-history")
      (mx/materialize! (:params result))
      (is (= [2] (mx/shape (:params result))) "compiled-train: params shape [2]")
      (let [final (mx/->clj (:params result))]
        (is (and (< (js/Math.abs (nth final 0)) 3.0)
                 (< (js/Math.abs (nth final 1)) 2.0))
            "compiled-train: converged toward 0"))
      (is (< (last (:loss-history result)) (first (:loss-history result))) "compiled-train: losses decrease")))

  (testing "log-every controls recorded losses"
    (let [result (co/compiled-train quadratic-score (mx/array [3.0 -2.0])
                                    {:iterations 100 :lr 0.01 :log-every 10})]
      (is (= 11 (count (:loss-history result))) "log-every=10, 100 iters: 11 loss entries"))

    (let [result (co/compiled-train quadratic-score (mx/array [3.0 -2.0])
                                    {:iterations 100 :lr 0.01 :log-every 25})]
      (is (= 5 (count (:loss-history result))) "log-every=25, 100 iters: 5 loss entries")))

  (testing "callback is invoked"
    (let [cb-log (volatile! [])
          result (co/compiled-train quadratic-score (mx/array [3.0 -2.0])
                                    {:iterations 50 :lr 0.01 :log-every 10
                                     :callback (fn [info] (vswap! cb-log conj (:iter info)))})]
      (is (= 6 (count @cb-log)) "callback invoked at log points")
      (is (= [0 9 19 29 39 49] @cb-log) "callback iter values correct"))))

;; ---------------------------------------------------------------------------
;; 7b. Default opts (empty map uses defaults)
;; ---------------------------------------------------------------------------

(deftest default-opts-test
  (testing "default opts"
    (let [result (co/compiled-train quadratic-score (mx/array [3.0 -2.0]) {})]
      (is (some? (:params result)) "default opts: returns :params")
      (is (= [2] (mx/shape (:params result))) "default opts: params shape [2]")
      (is (= 21 (count (:loss-history result))) "default opts: 21 loss entries (1000 iters, log-every=50)"))))

;; ---------------------------------------------------------------------------
;; 8. Periodic cleanup runs without error
;; ---------------------------------------------------------------------------

(deftest periodic-cleanup-test
  (testing "periodic cleanup"
    (let [result (co/compiled-train quadratic-score (mx/array [1.0 2.0 3.0])
                                    {:iterations 200 :lr 0.01 :log-every 50})]
      (is (some? (:params result)) "200 iterations with cleanup: no error")
      (is (pos? (count (:loss-history result))) "200 iterations: losses recorded"))))

;; ---------------------------------------------------------------------------
;; 9. Memory stays bounded over 500 iterations
;; ---------------------------------------------------------------------------

(deftest memory-bounded-test
  (testing "memory bounded"
    (mx/reset-peak-memory!)
    (let [mem-before (mx/get-active-memory)
          result (co/compiled-train quadratic-score
                                    (mx/array (vec (range 1.0 11.0)))
                                    {:iterations 500 :lr 0.01 :log-every 100})
          mem-after (mx/get-active-memory)
          mem-peak (mx/get-peak-memory)]
      (is (< mem-peak (* 1024 1024)) "500 iters, D=10: peak memory < 1MB")
      (is (< (last (:loss-history result)) (first (:loss-history result))) "500 iters: converging"))))

;; ---------------------------------------------------------------------------
;; 10. Compiled-generate score function (middle tier)
;; ---------------------------------------------------------------------------

(def cg-model
  (gen [flag]
    (let [mu (trace :mu (dist/uniform -10 10))]
      (if flag
        (trace :obs (dist/gaussian mu 1))
        (trace :obs (dist/gaussian mu 5)))
      mu)))

(def cg-obs (cm/choicemap :obs (mx/scalar 5.0)))

(deftest compiled-generate-score-fn-test
  (testing "model has compiled-generate but NOT tensor-native"
    (let [schema (:schema cg-model)]
      (is (some? (:compiled-generate schema)) "cg-model has :compiled-generate")
      (is (:has-branches? schema) "cg-model has-branches? = true")
      (is (nil? (u/get-eliminated-addresses cg-model)) "no eliminated addresses")))

  (testing "make-compiled-generate-score-fn returns non-nil"
    (let [result (u/make-compiled-generate-score-fn cg-model [true] cg-obs [:mu])]
      (is (some? result) "make-compiled-generate-score-fn returns result")
      (is (fn? (:score-fn result)) "result has :score-fn")
      (is (map? (:latent-index result)) "result has :latent-index")))

  (testing "score values match handler"
    (let [{cg-score-fn :score-fn} (u/make-compiled-generate-score-fn cg-model [true] cg-obs [:mu])
          handler-score-fn (u/make-score-fn (dyn/auto-key cg-model) [true] cg-obs [:mu])
          params (mx/array [3.0])
          cg-val (mx/item (cg-score-fn params))
          handler-val (mx/item (handler-score-fn params))]
      (is (h/close? handler-val cg-val 0.01) "cg score matches handler")))

  (testing "make-tensor-score-fn dispatches to compiled-generate"
    (let [result (u/make-tensor-score-fn cg-model [true] cg-obs [:mu])]
      (is (not (:tensor-native? result)) "tensor-native? is false")
      (is (:compiled-generate? result) "compiled-generate? is true")))

  (testing "mx/grad through compiled-generate score"
    (let [{:keys [score-fn]} (u/make-compiled-generate-score-fn cg-model [true] cg-obs [:mu])
          grad-fn (mx/grad score-fn)
          params (mx/array [3.0])
          grad-val (mx/item (grad-fn params))]
      (is (h/close? 2.0 grad-val 0.1) "mx/grad through cg score: gradient = 2.0")))

  (testing "mx/value-and-grad works"
    (let [{:keys [score-fn]} (u/make-compiled-generate-score-fn cg-model [true] cg-obs [:mu])
          neg-score (fn [p] (mx/negative (score-fn p)))
          vg (mx/value-and-grad neg-score)
          params (mx/array [3.0])
          [loss grad] (vg params)]
      (mx/materialize! loss grad)
      (is (pos? (mx/item loss)) "value-and-grad: loss is positive (neg log-prob)")
      (is (h/close? -2.0 (mx/item grad) 0.1) "value-and-grad: grad = -2.0"))))

;; ---------------------------------------------------------------------------
;; 11. make-compiled-loss-grad with compiled-generate
;; ---------------------------------------------------------------------------

(deftest compiled-loss-grad-compiled-generate-test
  (testing "make-compiled-loss-grad compiled-generate path"
    (let [result (co/make-compiled-loss-grad cg-model [true] cg-obs [:mu])]
      (is (= :compiled-generate (:compilation-level result)) "compilation-level is :compiled-generate")
      (is (fn? (:loss-grad-fn result)) "has loss-grad-fn")
      (is (fn? (:score-fn result)) "has score-fn")
      (is (some? (:init-params result)) "has init-params")
      (is (= 1 (:n-params result)) "n-params = 1")))

  (testing "loss-grad-fn returns [loss, grad]"
    (let [{:keys [loss-grad-fn]} (co/make-compiled-loss-grad cg-model [true] cg-obs [:mu])
          params (mx/array [3.0])
          [loss grad] (loss-grad-fn params)]
      (mx/materialize! loss grad)
      (is (pos? (mx/item loss)) "loss-grad-fn: loss > 0")
      (is (= [1] (mx/shape grad)) "loss-grad-fn: grad shape [1]"))))

;; ---------------------------------------------------------------------------
;; 12. learn converges via compiled-generate path
;; ---------------------------------------------------------------------------

(deftest learn-compiled-generate-test
  (testing "learn via compiled-generate"
    (let [result (co/learn cg-model [true] cg-obs [:mu]
                   {:iterations 500 :lr 0.05 :log-every 100})]
      (mx/materialize! (:params result))
      (is (= :compiled-generate (:compilation-level result)) "learn: compilation-level is :compiled-generate")
      (is (vector? (:loss-history result)) "learn: has loss-history")
      (is (< (last (:loss-history result)) (first (:loss-history result))) "learn: loss decreased")
      (let [mu-final (mx/item (:params result))]
        (is (h/close? 5.0 mu-final 0.3) "learn: mu converges to MAP ~ 5.0")))))

;; ---------------------------------------------------------------------------
;; 13. Multi-site branch model
;; ---------------------------------------------------------------------------

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

(deftest multi-site-compiled-generate-test
  (testing "multi-site compiled-generate detection"
    (let [result (co/make-compiled-loss-grad cg-multi [true] cg-multi-obs [:slope :intercept])]
      (is (= :compiled-generate (:compilation-level result)) "multi-site: compilation-level is :compiled-generate")
      (is (= 2 (:n-params result)) "multi-site: n-params = 2")))

  (testing "multi-site learn convergence"
    (let [result (co/learn cg-multi [true] cg-multi-obs [:slope :intercept]
                   {:iterations 1000 :lr 0.05 :log-every 200})]
      (mx/materialize! (:params result))
      (let [final (mx/->clj (:params result))
            total (+ (nth final 0) (nth final 1))]
        (is (< (last (:loss-history result)) (first (:loss-history result))) "multi-site: loss decreased")
        (is (h/close? 4.0 total 1.0) "multi-site: slope + intercept ~ 4.0")))))

(cljs.test/run-tests)
