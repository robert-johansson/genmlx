(ns genmlx.kalman-test
  "Tests for Kalman filter middleware."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.kalman :as kalman])
  (:require-macros [genmlx.gen :refer [gen]]
                   [genmlx.dist.macros :refer [defdist]]))

(deftest kalman-latent-distribution-test
  (testing "kalman-latent log-prob matches gaussian"
    (let [d (kalman/kalman-latent (mx/scalar 0.9) (mx/scalar 1.0) (mx/scalar 0.5))
          g (dist/gaussian (mx/scalar 0.9) (mx/scalar 0.5))
          v (mx/scalar 1.2)
          lp-kalman (mx/item (dc/dist-log-prob d v))
          lp-gauss  (mx/item (dc/dist-log-prob g v))]
      (is (h/close? lp-gauss lp-kalman 1e-5)
          "kalman-latent log-prob matches gaussian"))))

(deftest kalman-obs-distribution-test
  (testing "kalman-obs log-prob matches gaussian (mask=1)"
    (let [d (kalman/kalman-obs (mx/scalar 5.0) (mx/scalar -2.0) (mx/scalar 1.0)
                                (mx/scalar 0.3) (mx/scalar 1.0))
          g (dist/gaussian (mx/scalar 3.0) (mx/scalar 0.3))
          v (mx/scalar 3.5)
          lp-kalman (mx/item (dc/dist-log-prob d v))
          lp-gauss  (mx/item (dc/dist-log-prob g v))]
      (is (h/close? lp-gauss lp-kalman 1e-5)
          "kalman-obs log-prob matches gaussian (mask=1)")))

  (testing "kalman-obs log-prob with mask=0 is 0"
    (let [d (kalman/kalman-obs (mx/scalar 5.0) (mx/scalar -2.0) (mx/scalar 1.0)
                                (mx/scalar 0.3) (mx/scalar 0.0))
          v (mx/scalar 3.5)
          lp (mx/item (dc/dist-log-prob d v))]
      (is (h/close? 0.0 lp 1e-7)
          "kalman-obs log-prob with mask=0 is 0"))))

(deftest kalman-predict-test
  (testing "kalman-predict"
    (let [belief {:mean (mx/array [1.0 2.0]) :var (mx/array [0.5 0.3])}
          rho (mx/scalar 0.9)
          q (mx/scalar 1.0)
          pred (kalman/kalman-predict belief rho q)]
      (is (h/close? 0.9 (mx/item (mx/index (:mean pred) 0)) 1e-5) "predict mean[0]")
      (is (h/close? 1.8 (mx/item (mx/index (:mean pred) 1)) 1e-5) "predict mean[1]")
      (is (h/close? 1.405 (mx/item (mx/index (:var pred) 0)) 1e-5) "predict var[0]")
      (is (h/close? 1.243 (mx/item (mx/index (:var pred) 1)) 1e-5) "predict var[1]"))))

(deftest kalman-update-test
  (testing "kalman-update"
    (let [belief {:mean (mx/array [1.0 2.0]) :var (mx/array [1.0 1.0])}
          obs (mx/array [3.0 5.0])
          base-mean (mx/array [0.5 1.0])
          loading (mx/scalar 2.0)
          noise-std (mx/scalar 0.5)
          mask (mx/array [1.0 1.0])
          {:keys [belief ll]} (kalman/kalman-update belief obs base-mean loading noise-std mask)
          S 4.25
          K (/ 2.0 S)]
      (is (h/close? (+ 1.0 (* K 0.5)) (mx/item (mx/index (:mean belief) 0)) 1e-4)
          "update mean[0]")
      (is (h/close? 2.0 (mx/item (mx/index (:mean belief) 1)) 1e-4)
          "update mean[1]")
      (is (h/close? (- 1.0 (* K 2.0)) (mx/item (mx/index (:var belief) 0)) 1e-4)
          "update var[0]")
      (is (js/isFinite (mx/item (mx/index ll 0))) "LL is finite"))))

(deftest kalman-update-mask-test
  (testing "kalman-update with mask"
    (let [belief {:mean (mx/array [1.0 2.0]) :var (mx/array [1.0 1.0])}
          obs (mx/array [3.0 5.0])
          base-mean (mx/array [0.5 1.0])
          loading (mx/scalar 2.0)
          noise-std (mx/scalar 0.5)
          mask (mx/array [1.0 0.0])
          {:keys [belief ll]} (kalman/kalman-update belief obs base-mean loading noise-std mask)]
      (is (h/close? 2.0 (mx/item (mx/index (:mean belief) 1)) 1e-5)
          "masked: mean[1] unchanged")
      (is (h/close? 1.0 (mx/item (mx/index (:var belief) 1)) 1e-5)
          "masked: var[1] unchanged")
      (is (h/close? 0.0 (mx/item (mx/index ll 1)) 1e-5)
          "masked: ll[1] = 0"))))

(deftest kalman-sequential-update-test
  (testing "kalman-sequential-update"
    (let [belief {:mean (mx/array [0.0]) :var (mx/array [1.0])}
          obs1 {:obs (mx/array [1.5]) :base-mean (mx/array [0.0])
                :loading (mx/scalar 1.0) :noise-std (mx/scalar 1.0)
                :mask (mx/array [1.0])}
          obs2 {:obs (mx/array [2.0]) :base-mean (mx/array [0.0])
                :loading (mx/scalar 1.0) :noise-std (mx/scalar 1.0)
                :mask (mx/array [1.0])}
          {:keys [belief ll]} (kalman/kalman-sequential-update belief [obs1 obs2])]
      (is (> (mx/item (mx/index (:mean belief) 0)) 0.0) "mean moved toward obs")
      (is (< (mx/item (mx/index (:var belief) 0)) 1.0) "var decreased")
      (is (js/isFinite (mx/item (mx/index ll 0))) "total LL is finite"))))

(deftest kalman-step-test
  (testing "kalman-step"
    (let [belief {:mean (mx/array [1.0]) :var (mx/array [0.5])}
          latent {:transition-coeff (mx/scalar 0.9) :process-noise (mx/scalar 1.0)}
          observations [{:obs (mx/array [2.0]) :base-mean (mx/array [0.0])
                         :loading (mx/scalar 1.5) :noise-std (mx/scalar 0.5)
                         :mask (mx/array [1.0])}]
          {:keys [belief ll]} (kalman/kalman-step belief latent observations)]
      (is (js/isFinite (mx/item (mx/index (:mean belief) 0))) "step: mean finite")
      (is (> (mx/item (mx/index (:var belief) 0)) 0.0) "step: var positive")
      (is (js/isFinite (mx/item (mx/index ll 0))) "step: ll finite"))))

(deftest kalman-generate-test
  (testing "kalman-generate handler middleware"
    (let [step-model (dyn/auto-key
                       (gen [_data]
                         (let [z (trace :z (kalman/kalman-latent (mx/scalar 0.5) (mx/scalar 0.0) (mx/scalar 1.0)))]
                           (trace :x (kalman/kalman-obs (mx/scalar 0.0) (mx/scalar 1.0) z
                                                        (mx/scalar 0.3) (mx/array [1.0 1.0 1.0])))
                           {:z z})))
          constraints (cm/choicemap :x (mx/array [1.5 2.0 0.5]))
          result (kalman/kalman-generate
                   step-model [nil] constraints :z 3 (rng/fresh-key 42))]
      (is (some? (:kalman-ll result)) "handler: has kalman-ll")
      (is (= [3] (vec (mx/shape (:kalman-ll result))))
          "handler: kalman-ll is [3]-shaped")
      (is (< (mx/item (mx/sum (:kalman-ll result))) 0.0)
          "handler: kalman-ll < 0")
      (is (some? (:kalman-belief result)) "handler: has kalman-belief"))))

(deftest kalman-level-consistency-test
  (testing "Level 1 vs Level 2 consistency"
    (let [n 5
          rho (mx/scalar 0.8)
          q (mx/scalar 1.0)
          obs-vals (mx/array [1.5 2.0 0.5 1.0 -0.5])
          base-mean (mx/zeros [n])
          loading (mx/scalar 1.0)
          noise-std (mx/scalar 0.5)
          mask (mx/ones [n])
          belief0 {:mean (mx/zeros [n]) :var (mx/zeros [n])}
          belief-pred (kalman/kalman-predict belief0 rho q)
          r0 (kalman/kalman-update belief-pred obs-vals base-mean loading noise-std mask)
          ll-level1 (mx/item (mx/sum (:ll r0)))
          step-model (dyn/auto-key
                       (gen [_data]
                         (let [z (trace :z (kalman/kalman-latent rho (mx/scalar 0.0) q))]
                           (trace :x (kalman/kalman-obs (mx/scalar 0.0) loading z noise-std mask))
                           {:z z})))
          result (kalman/kalman-generate
                   step-model [nil]
                   (cm/choicemap :x obs-vals)
                   :z n (rng/fresh-key 42))
          ll-level2 (mx/item (mx/sum (:kalman-ll result)))]
      (is (h/close? ll-level1 ll-level2 1e-4) "Level 1 matches Level 2"))))

(deftest kalman-gradient-test
  (testing "gradient through Kalman"
    (let [n 3
          f (fn [rho-arr]
              (let [rho (mx/index rho-arr 0)
                    belief0 {:mean (mx/array [0.5 -0.3 0.1]) :var (mx/ones [n])}
                    pred (kalman/kalman-predict belief0 rho (mx/scalar 1.0))
                    {:keys [ll]} (kalman/kalman-update
                                   pred
                                   (mx/array [1.0 2.0 -1.0])
                                   (mx/zeros [n])
                                   (mx/scalar 1.0)
                                   (mx/scalar 0.5)
                                   (mx/ones [n]))]
                (mx/sum ll)))
          grad-fn (mx/grad f)
          rho-arr (mx/array [0.8])
          grad (grad-fn rho-arr)]
      (is (js/isFinite (mx/item (mx/index grad 0))) "gradient is finite")
      (is (not= 0.0 (mx/item (mx/index grad 0))) "gradient is nonzero"))))

(deftest kalman-fold-test
  (testing "kalman-fold"
    (let [n 4
          rho (mx/scalar 0.7)
          q (mx/scalar 1.0)
          loading (mx/scalar 1.5)
          noise-std (mx/scalar 0.5)
          T 3
          data (mapv (fn [t] (mx/array (mapv #(+ (* 0.5 t) (* 0.1 %)) (range n)))) (range T))
          masks (vec (repeat T (mx/ones [n])))
          step-fn (dyn/auto-key
                    (gen [context]
                      (let [z (trace :z (kalman/kalman-latent rho (:prev-z context) q))]
                        (trace :x (kalman/kalman-obs (:base-mean context) loading z
                                                     noise-std (:mask context)))
                        {:z z})))
          context-fn (fn [t]
                       {:args [{:prev-z (mx/zeros [n])
                                :base-mean (mx/zeros [n])
                                :mask (nth masks t)}]
                        :constraints (cm/choicemap :x (nth data t))})
          result (kalman/kalman-fold step-fn :z n T context-fn)]
      (is (= [n] (vec (mx/shape result))) "fold: result is [4]-shaped")
      (is (js/isFinite (mx/item (mx/sum result))) "fold: total LL is finite")
      (is (< (mx/item (mx/sum result)) 0.0) "fold: total LL < 0"))))

(deftest kalman-fold-gradient-test
  (testing "kalman-fold gradient"
    (let [n 3
          T 2
          data [(mx/array [1.0 2.0 -1.0]) (mx/array [1.5 2.5 -0.5])]
          masks [(mx/ones [n]) (mx/ones [n])]
          f (fn [params-arr]
              (let [rho (mx/index params-arr 0)
                    loading (mx/index params-arr 1)
                    step-fn (dyn/auto-key
                              (gen [context]
                                (let [z (trace :z (kalman/kalman-latent rho (:prev-z context) (mx/scalar 1.0)))]
                                  (trace :x (kalman/kalman-obs (mx/zeros [n]) loading z
                                                               (mx/scalar 0.5) (:mask context)))
                                  {:z z})))
                    context-fn (fn [t]
                                 {:args [{:prev-z (mx/zeros [n])
                                          :base-mean (mx/zeros [n])
                                          :mask (nth masks t)}]
                                  :constraints (cm/choicemap :x (nth data t))})
                    ll (kalman/kalman-fold step-fn :z n T context-fn)]
                (mx/sum ll)))
          grad-fn (mx/grad f)
          params (mx/array [0.7 1.0])
          grad (grad-fn params)]
      (is (js/isFinite (mx/item (mx/index grad 0))) "fold grad: d/d(rho) finite")
      (is (js/isFinite (mx/item (mx/index grad 1))) "fold grad: d/d(loading) finite")
      (is (not= 0.0 (mx/item (mx/index grad 1))) "fold grad: d/d(loading) nonzero"))))

(deftest kalman-sequential-belief-test
  (testing "sequential belief update in handler"
    (let [n 2
          step-fn (dyn/auto-key
                    (gen [_ctx]
                      (let [z (trace :z (kalman/kalman-latent (mx/scalar 0.5) (mx/scalar 0.0) (mx/scalar 1.0)))]
                        (trace :x1 (kalman/kalman-obs (mx/zeros [n]) (mx/scalar 1.0) z
                                                       (mx/scalar 0.5) (mx/ones [n])))
                        (trace :x2 (kalman/kalman-obs (mx/zeros [n]) (mx/scalar 1.0) z
                                                       (mx/scalar 0.5) (mx/ones [n])))
                        {:z z})))
          result (kalman/kalman-generate
                   step-fn [nil]
                   (cm/choicemap :x1 (mx/array [2.0 3.0])
                                 :x2 (mx/array [2.5 3.5]))
                   :z n (rng/fresh-key 0))
          var-final (mx/item (mx/index (:var (:kalman-belief result)) 0))]
      (is (< var-final 1.0) "var after 2 obs < 1.0 (prior)")
      (is (< var-final 0.3) "var after 2 obs < 0.3"))))

(cljs.test/run-tests)
