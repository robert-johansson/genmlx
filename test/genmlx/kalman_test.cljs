(ns genmlx.kalman-test
  "Tests for Kalman filter middleware."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.kalman :as kalman])
  (:require-macros [genmlx.gen :refer [gen]]
                   [genmlx.dist.macros :refer [defdist]]))

(defn assert-close [msg expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (println (str "  PASS: " msg " (" actual ")"))
      (println (str "  FAIL: " msg " expected " expected " got " actual " diff " diff)))))

(defn assert-true [msg pred]
  (if pred
    (println (str "  PASS: " msg))
    (println (str "  FAIL: " msg))))

;; ---------------------------------------------------------------------------
;; Test 1: kalman-latent distribution
;; ---------------------------------------------------------------------------

(println "\n-- kalman-latent distribution --")
(let [d (kalman/kalman-latent (mx/scalar 0.9) (mx/scalar 1.0) (mx/scalar 0.5))
      g (dist/gaussian (mx/scalar 0.9) (mx/scalar 0.5))
      v (mx/scalar 1.2)
      lp-kalman (mx/item (dc/dist-log-prob d v))
      lp-gauss  (mx/item (dc/dist-log-prob g v))]
  (assert-close "kalman-latent log-prob matches gaussian" lp-gauss lp-kalman 1e-5))

;; ---------------------------------------------------------------------------
;; Test 2: kalman-obs distribution
;; ---------------------------------------------------------------------------

(println "\n-- kalman-obs distribution --")
(let [d (kalman/kalman-obs (mx/scalar 5.0) (mx/scalar -2.0) (mx/scalar 1.0)
                            (mx/scalar 0.3) (mx/scalar 1.0))
      g (dist/gaussian (mx/scalar 3.0) (mx/scalar 0.3))
      v (mx/scalar 3.5)
      lp-kalman (mx/item (dc/dist-log-prob d v))
      lp-gauss  (mx/item (dc/dist-log-prob g v))]
  (assert-close "kalman-obs log-prob matches gaussian (mask=1)" lp-gauss lp-kalman 1e-5))

;; Test mask=0 gives log-prob=0
(let [d (kalman/kalman-obs (mx/scalar 5.0) (mx/scalar -2.0) (mx/scalar 1.0)
                            (mx/scalar 0.3) (mx/scalar 0.0))
      v (mx/scalar 3.5)
      lp (mx/item (dc/dist-log-prob d v))]
  (assert-close "kalman-obs log-prob with mask=0 is 0" 0.0 lp 1e-7))

;; ---------------------------------------------------------------------------
;; Test 3: kalman-predict
;; ---------------------------------------------------------------------------

(println "\n-- kalman-predict --")
(let [belief {:mean (mx/array [1.0 2.0]) :var (mx/array [0.5 0.3])}
      rho (mx/scalar 0.9)
      q (mx/scalar 1.0)
      pred (kalman/kalman-predict belief rho q)]
  (assert-close "predict mean[0]" 0.9 (mx/item (mx/index (:mean pred) 0)) 1e-5)
  (assert-close "predict mean[1]" 1.8 (mx/item (mx/index (:mean pred) 1)) 1e-5)
  (assert-close "predict var[0]" 1.405 (mx/item (mx/index (:var pred) 0)) 1e-5)
  (assert-close "predict var[1]" 1.243 (mx/item (mx/index (:var pred) 1)) 1e-5))

;; ---------------------------------------------------------------------------
;; Test 4: kalman-update
;; ---------------------------------------------------------------------------

(println "\n-- kalman-update --")
(let [belief {:mean (mx/array [1.0 2.0]) :var (mx/array [1.0 1.0])}
      obs (mx/array [3.0 5.0])
      base-mean (mx/array [0.5 1.0])
      loading (mx/scalar 2.0)
      noise-std (mx/scalar 0.5)
      mask (mx/array [1.0 1.0])
      {:keys [belief ll]} (kalman/kalman-update belief obs base-mean loading noise-std mask)]
  (let [S 4.25
        K (/ 2.0 S)]
    (assert-close "update mean[0]" (+ 1.0 (* K 0.5)) (mx/item (mx/index (:mean belief) 0)) 1e-4)
    (assert-close "update mean[1]" 2.0 (mx/item (mx/index (:mean belief) 1)) 1e-4)
    (assert-close "update var[0]" (- 1.0 (* K 2.0)) (mx/item (mx/index (:var belief) 0)) 1e-4)
    (assert-true "LL is finite" (js/isFinite (mx/item (mx/index ll 0))))))

;; ---------------------------------------------------------------------------
;; Test 5: kalman-update with mask=0 (missing data)
;; ---------------------------------------------------------------------------

(println "\n-- kalman-update with mask --")
(let [belief {:mean (mx/array [1.0 2.0]) :var (mx/array [1.0 1.0])}
      obs (mx/array [3.0 5.0])
      base-mean (mx/array [0.5 1.0])
      loading (mx/scalar 2.0)
      noise-std (mx/scalar 0.5)
      mask (mx/array [1.0 0.0])
      {:keys [belief ll]} (kalman/kalman-update belief obs base-mean loading noise-std mask)]
  (assert-close "masked: mean[1] unchanged" 2.0 (mx/item (mx/index (:mean belief) 1)) 1e-5)
  (assert-close "masked: var[1] unchanged" 1.0 (mx/item (mx/index (:var belief) 1)) 1e-5)
  (assert-close "masked: ll[1] = 0" 0.0 (mx/item (mx/index ll 1)) 1e-5))

;; ---------------------------------------------------------------------------
;; Test 6: kalman-sequential-update
;; ---------------------------------------------------------------------------

(println "\n-- kalman-sequential-update --")
(let [belief {:mean (mx/array [0.0]) :var (mx/array [1.0])}
      obs1 {:obs (mx/array [1.5]) :base-mean (mx/array [0.0])
            :loading (mx/scalar 1.0) :noise-std (mx/scalar 1.0)
            :mask (mx/array [1.0])}
      obs2 {:obs (mx/array [2.0]) :base-mean (mx/array [0.0])
            :loading (mx/scalar 1.0) :noise-std (mx/scalar 1.0)
            :mask (mx/array [1.0])}
      {:keys [belief ll]} (kalman/kalman-sequential-update belief [obs1 obs2])]
  (assert-true "mean moved toward obs" (> (mx/item (mx/index (:mean belief) 0)) 0.0))
  (assert-true "var decreased" (< (mx/item (mx/index (:var belief) 0)) 1.0))
  (assert-true "total LL is finite" (js/isFinite (mx/item (mx/index ll 0)))))

;; ---------------------------------------------------------------------------
;; Test 7: kalman-step (predict + update)
;; ---------------------------------------------------------------------------

(println "\n-- kalman-step --")
(let [belief {:mean (mx/array [1.0]) :var (mx/array [0.5])}
      latent {:transition-coeff (mx/scalar 0.9) :process-noise (mx/scalar 1.0)}
      observations [{:obs (mx/array [2.0]) :base-mean (mx/array [0.0])
                     :loading (mx/scalar 1.5) :noise-std (mx/scalar 0.5)
                     :mask (mx/array [1.0])}]
      {:keys [belief ll]} (kalman/kalman-step belief latent observations)]
  (assert-true "step: mean finite" (js/isFinite (mx/item (mx/index (:mean belief) 0))))
  (assert-true "step: var positive" (> (mx/item (mx/index (:var belief) 0)) 0.0))
  (assert-true "step: ll finite" (js/isFinite (mx/item (mx/index ll 0)))))

;; ---------------------------------------------------------------------------
;; Test 8: Handler middleware (kalman-generate)
;; ---------------------------------------------------------------------------

(println "\n-- kalman-generate handler middleware --")
(let [step-model (dyn/auto-key
                   (gen [_data]
                     (let [z (trace :z (kalman/kalman-latent (mx/scalar 0.5) (mx/scalar 0.0) (mx/scalar 1.0)))]
                       (trace :x (kalman/kalman-obs (mx/scalar 0.0) (mx/scalar 1.0) z
                                                    (mx/scalar 0.3) (mx/array [1.0 1.0 1.0])))
                       {:z z})))
      constraints (cm/choicemap :x (mx/array [1.5 2.0 0.5]))
      result (kalman/kalman-generate
               step-model [nil] constraints :z 3 (rng/fresh-key 42))]
  (assert-true "handler: has kalman-ll" (some? (:kalman-ll result)))
  (assert-true "handler: kalman-ll is [3]-shaped"
    (= [3] (vec (mx/shape (:kalman-ll result)))))
  (assert-true "handler: kalman-ll < 0" (< (mx/item (mx/sum (:kalman-ll result))) 0.0))
  (assert-true "handler: has kalman-belief" (some? (:kalman-belief result)))
  (println (str "  total LL = " (.toFixed (mx/item (mx/sum (:kalman-ll result))) 4))))

;; ---------------------------------------------------------------------------
;; Test 9: Level 1 and Level 2 produce same LL
;; ---------------------------------------------------------------------------

(println "\n-- Level 1 vs Level 2 consistency --")
(let [n 5
      rho (mx/scalar 0.8)
      q (mx/scalar 1.0)
      obs-vals (mx/array [1.5 2.0 0.5 1.0 -0.5])
      base-mean (mx/zeros [n])
      loading (mx/scalar 1.0)
      noise-std (mx/scalar 0.5)
      mask (mx/ones [n])

      ;; Level 1: predict from {0, 0} then update (matching handler behavior)
      belief0 {:mean (mx/zeros [n]) :var (mx/zeros [n])}
      belief-pred (kalman/kalman-predict belief0 rho q)
      r0 (kalman/kalman-update belief-pred obs-vals base-mean loading noise-std mask)
      ll-level1 (mx/item (mx/sum (:ll r0)))

      ;; Level 2: handler middleware
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
  (assert-close "Level 1 matches Level 2" ll-level1 ll-level2 1e-4))

;; ---------------------------------------------------------------------------
;; Test 10: Gradient flows through Kalman operations
;; ---------------------------------------------------------------------------

(println "\n-- Gradient through Kalman --")
;; Use non-zero initial variance so rho affects the result
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
  (assert-true "gradient is finite" (js/isFinite (mx/item (mx/index grad 0))))
  (assert-true "gradient is nonzero" (not= 0.0 (mx/item (mx/index grad 0))))
  (println (str "  d(LL)/d(rho) = " (.toFixed (mx/item (mx/index grad 0)) 6))))

;; ---------------------------------------------------------------------------
;; Test 11: kalman-fold — multi-timestep temporal folding
;; ---------------------------------------------------------------------------

(println "\n-- kalman-fold --")
(let [n 4
      rho (mx/scalar 0.7)
      q (mx/scalar 1.0)
      loading (mx/scalar 1.5)
      noise-std (mx/scalar 0.5)
      T 3
      ;; Synthetic data: T timesteps × n elements
      data (mapv (fn [t] (mx/array (mapv #(+ (* 0.5 t) (* 0.1 %)) (range n)))) (range T))
      masks (vec (repeat T (mx/ones [n])))

      ;; Cognitive step: latent -> observation
      step-fn (dyn/auto-key
                (gen [context]
                  (let [z (trace :z (kalman/kalman-latent rho (:prev-z context) q))]
                    (trace :x (kalman/kalman-obs (:base-mean context) loading z
                                                 noise-std (:mask context)))
                    {:z z})))

      ;; Context function builds per-timestep args + constraints
      context-fn (fn [t]
                   {:args [{:prev-z (mx/zeros [n])
                            :base-mean (mx/zeros [n])
                            :mask (nth masks t)}]
                    :constraints (cm/choicemap :x (nth data t))})

      result (kalman/kalman-fold step-fn :z n T context-fn)]
  (assert-true "fold: result is [4]-shaped" (= [n] (vec (mx/shape result))))
  (assert-true "fold: total LL is finite" (js/isFinite (mx/item (mx/sum result))))
  (assert-true "fold: total LL < 0" (< (mx/item (mx/sum result)) 0.0))
  (println (str "  total LL = " (.toFixed (mx/item (mx/sum result)) 4))))

;; ---------------------------------------------------------------------------
;; Test 12: kalman-fold gradient
;; ---------------------------------------------------------------------------

(println "\n-- kalman-fold gradient --")
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
  (assert-true "fold grad: d/d(rho) finite" (js/isFinite (mx/item (mx/index grad 0))))
  (assert-true "fold grad: d/d(loading) finite" (js/isFinite (mx/item (mx/index grad 1))))
  (assert-true "fold grad: d/d(loading) nonzero" (not= 0.0 (mx/item (mx/index grad 1))))
  (println (str "  grad = [" (.toFixed (mx/item (mx/index grad 0)) 4) ", "
                (.toFixed (mx/item (mx/index grad 1)) 4) "]")))

;; ---------------------------------------------------------------------------
;; Test 13: Sequential update — handler uses latest belief per observation
;; ---------------------------------------------------------------------------

(println "\n-- Sequential belief update in handler --")
(let [n 2
      ;; Two observations that update the same latent
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
               :z n (rng/fresh-key 0))]
  ;; After two observations, belief should be tighter than after one
  (let [var-final (mx/item (mx/index (:var (:kalman-belief result)) 0))]
    (assert-true "var after 2 obs < 1.0 (prior)" (< var-final 1.0))
    ;; With loading=1, noise=0.5: after predict var=1, after 1 obs var~0.2,
    ;; after 2 obs var should be even smaller
    (assert-true "var after 2 obs < 0.3" (< var-final 0.3))
    (println (str "  final var = " (.toFixed var-final 4)))))

(println "\n-- All Kalman tests complete --")
