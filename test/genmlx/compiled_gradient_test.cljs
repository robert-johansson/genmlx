(ns genmlx.compiled-gradient-test
  "WP-4 tests: gradient through compiled inference.

   Tests cover:
   1. Differentiable MH chain: gradient matches finite differences
   2. Multi-step chains: gradient through 1, 5, 10, 50 steps
   3. Gradient direction: correct sign for simple cases
   4. Score gradient through chain: end-to-end API
   5. Gradient through SMC: log-ML gradient via gumbel-softmax
   6. Gate 5: memory scaling for backward pass (T=10..1000)"
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.gen :refer [gen]]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.inference.compiled-gradient :as cg]
            [genmlx.inference.util :as u]
            [genmlx.inference.compiled-smc :as csmc]
            [genmlx.inference.differentiable-resample :as dr]
            [genmlx.compiled-ops :as compiled]))

;; ---------------------------------------------------------------------------
;; 1. Differentiable MH chain: basic gradient
;; ---------------------------------------------------------------------------

(deftest differentiable-mh-chain-test
  (testing "chain runs and produces correct output"
    (let [score-fn (fn [params] (mx/negative (mx/sum (mx/multiply params params))))
          K 2
          T 5
          chain-fn (cg/make-differentiable-chain score-fn (mx/scalar 0.1) T K)
          init (mx/array [1.0 2.0])
          noise (rng/normal (rng/fresh-key 1) [T K])
          uniforms (rng/uniform (rng/fresh-key 2) [T])
          _ (mx/materialize! noise uniforms)
          result (chain-fn init noise uniforms)]
      (mx/eval! result)
      (is (= [K] (mx/shape result)) "chain output shape")
      (is (every? js/isFinite (mx/->clj result)) "chain output is finite")))

  (testing "gradient through single MH step"
    (let [score-fn (fn [params] (mx/negative (mx/sum (mx/multiply params params))))
          K 2
          noise (mx/zeros [1 K])
          uniforms (mx/zeros [1])
          chain-fn (cg/make-differentiable-chain score-fn (mx/scalar 0.1) 1 K)
          objective (fn [p0]
                      (let [final (chain-fn p0 noise uniforms)]
                        (score-fn final)))
          grad-fn (mx/grad objective)
          init (mx/array [1.0 2.0])
          g (grad-fn init)]
      (mx/eval! g)
      (is (not (every? zero? (mx/->clj g))) "gradient is non-zero"))))

;; ---------------------------------------------------------------------------
;; 2. Gradient matches finite differences
;; ---------------------------------------------------------------------------

(deftest gradient-vs-finite-differences-test
  (testing "gradient vs finite differences for various chain lengths"
    (let [score-fn (fn [params] (mx/negative (mx/sum (mx/multiply params params))))
          K 3
          eps 1e-4]
      (doseq [T [1 5 10]]
        (let [noise (rng/normal (rng/fresh-key 42) [T K])
              uniforms (rng/uniform (rng/fresh-key 43) [T])
              _ (mx/materialize! noise uniforms)
              chain-fn (cg/make-differentiable-chain score-fn (mx/scalar 0.1) T K)
              objective (fn [p0]
                          (score-fn (chain-fn p0 noise uniforms)))
              grad-fn (mx/grad objective)
              init (mx/array [0.5 -0.3 0.8])
              analytical (mx/->clj (let [g (grad-fn init)] (mx/eval! g) g))
              fd (mapv (fn [i]
                         (let [init-v (mx/->clj init)
                               p+ (mx/array (update init-v i + eps))
                               p- (mx/array (update init-v i - eps))
                               f+ (mx/realize (objective p+))
                               f- (mx/realize (objective p-))]
                           (/ (- f+ f-) (* 2 eps))))
                       (range K))
              signs (count (filter (fn [[a f]]
                                     (or (and (pos? a) (pos? f))
                                         (and (neg? a) (neg? f))
                                         (and (< (js/Math.abs a) 1e-6)
                                              (< (js/Math.abs f) 1e-6))))
                                   (map vector analytical fd)))
              sign-rate (/ signs K)]
          (is (>= sign-rate 0.67) (str "T=" T " sign agreement >= 67% (got " (* 100 sign-rate) "%)"))
          (let [mag-ok (every? (fn [[a f]]
                                 (if (< (js/Math.abs f) 1e-6)
                                   true
                                   (let [ratio (/ (js/Math.abs a) (js/Math.abs f))]
                                     (and (> ratio 0.1) (< ratio 10.0)))))
                               (map vector analytical fd))]
            (is mag-ok (str "T=" T " magnitude within 10x"))))))))

;; ---------------------------------------------------------------------------
;; 3. Gradient direction: correct sign
;; ---------------------------------------------------------------------------

(deftest gradient-direction-test
  (testing "gradient direction toward target"
    (let [target 3.0
          score-fn (fn [params]
                     (let [diff (mx/subtract params (mx/scalar target))]
                       (mx/negative (mx/sum (mx/multiply diff diff)))))
          K 1
          T 3
          noise (mx/zeros [T K])
          uniforms (mx/zeros [T])
          chain-fn (cg/make-differentiable-chain score-fn (mx/scalar 0.1) T K)
          objective (fn [p0] (score-fn (chain-fn p0 noise uniforms)))
          grad-fn (mx/grad objective)]
      (let [g (grad-fn (mx/array [1.0]))]
        (mx/eval! g)
        (is (> (first (mx/->clj g)) 0) "below target -> positive gradient"))
      (let [g (grad-fn (mx/array [5.0]))]
        (mx/eval! g)
        (is (< (first (mx/->clj g)) 0) "above target -> negative gradient")))))

;; ---------------------------------------------------------------------------
;; 4. mcmc-score-gradient API
;; ---------------------------------------------------------------------------

(deftest mcmc-score-gradient-api-test
  (testing "mcmc-score-gradient API"
    (let [make-score (fn [model-params]
                       (fn [latent]
                         (let [diff (mx/subtract latent model-params)]
                           (mx/negative (mx/divide
                                          (mx/sum (mx/multiply diff diff))
                                          (mx/scalar 2.0))))))
          init-latent (mx/array [0.5 -0.3])
          model-params (mx/array [1.0 2.0])
          result (cg/mcmc-score-gradient
                   make-score init-latent model-params
                   {:steps 5 :proposal-std 0.1 :key (rng/fresh-key 10)})]
      (mx/eval! (:value result) (:grad result))
      (is (js/isFinite (mx/realize (:value result))) "mcmc-score-gradient returns :value")
      (is (= [2] (mx/shape (:grad result))) "mcmc-score-gradient returns :grad")
      (is (not (every? zero? (mx/->clj (:grad result)))) "gradient is non-zero"))))

;; ---------------------------------------------------------------------------
;; 5. Gradient through SMC (gumbel-softmax)
;; ---------------------------------------------------------------------------

(def lg-kernel
  (gen [t state]
    (let [new-state (trace :x (dist/gaussian state 1.0))]
      (trace :y (dist/gaussian new-state 0.5))
      new-state)))

(def smc-obs
  (mapv (fn [t]
          (cm/choicemap :y (mx/scalar (* 0.5 t))))
        (range 5)))

(deftest gradient-through-smc-test
  (testing "gradient of log-ML w.r.t. init-state"
    (try
      (let [result (cg/smc-log-ml-gradient
                     lg-kernel (mx/scalar 0.0)
                     (mx/array [0.0])
                     smc-obs
                     {:particles 30 :tau 1.0 :key (rng/fresh-key 20)})]
        (mx/eval! (:value result) (:grad result))
        (let [v (mx/realize (:value result))
              g (mx/->clj (:grad result))]
          (is (js/isFinite v) (str "SMC log-ML is finite (" (.toFixed v 2) ")"))
          (is (not (every? #(< (js/Math.abs %) 1e-10) g)) "SMC gradient is non-zero")))
      (catch :default e
        (is false (str "SMC gradient: " (.-message e))))))

  (testing "FD comparison for SMC gradient"
    (try
      (let [eps 1e-3
            N-particles 30
            tau-val 1.0
            T (count smc-obs)
            schema (:schema lg-kernel)
            source (:source lg-kernel)
            static-sites (filterv :static? (:trace-sites schema))
            all-addrs (mapv :addr static-sites)
            K (count all-addrs)
            rk (rng/ensure-key (rng/fresh-key 20))
            [nk gk] (rng/split rk)
            extend-noise (rng/normal nk [T N-particles K])
            gumbel-noise (dr/generate-gumbel-noise gk T N-particles)
            _ (mx/materialize! extend-noise gumbel-noise)
            tau-arr (mx/scalar tau-val)
            extend-fn (compiled/make-smc-extend-step schema source)
            objective
            (fn [params]
              (let [init-n (mx/broadcast-to params [N-particles])]
                (loop [t 0
                       current-particles nil
                       current-state init-n
                       log-ml (mx/scalar 0.0)]
                  (if (>= t T)
                    log-ml
                    (let [obs-t (nth smc-obs t)
                          noise-t (mx/index extend-noise t)
                          kernel-args [(mx/ensure-array t) current-state]
                          {:keys [obs-log-prob values-map retval]}
                          (extend-fn noise-t kernel-args obs-t)
                          new-particles (mx/stack (mapv #(get values-map %) all-addrs) 1)
                          new-state (or retval (get values-map (first all-addrs)))
                          ml-inc (mx/subtract (mx/logsumexp obs-log-prob)
                                              (mx/scalar (js/Math.log N-particles)))
                          gumbel-t (mx/index gumbel-noise t)
                          {:keys [particles]} (dr/gumbel-softmax
                                                new-particles obs-log-prob
                                                gumbel-t tau-arr)
                          resampled-state (dr/gumbel-softmax-1d
                                            new-state obs-log-prob
                                            gumbel-t tau-arr)]
                      (recur (inc t) particles resampled-state
                             (mx/add log-ml ml-inc)))))))
            grad-fn (mx/grad objective)
            p0 (mx/array [0.0])
            analytical (first (mx/->clj (let [g (grad-fn p0)] (mx/eval! g) g)))
            f+ (mx/realize (objective (mx/array [eps])))
            f- (mx/realize (objective (mx/array [(- eps)])))
            fd (/ (- f+ f-) (* 2 eps))]
        (if (< (js/Math.abs fd) 1e-6)
          (is true "SMC FD near zero -- skip sign check")
          (is (or (and (pos? analytical) (pos? fd))
                  (and (neg? analytical) (neg? fd)))
              "SMC gradient sign matches FD")))
      (catch :default e
        (is false (str "SMC FD comparison: " (.-message e)))))))

;; ---------------------------------------------------------------------------
;; 6. Gate 5: Memory scaling for backward pass
;; ---------------------------------------------------------------------------

(deftest gate-5-memory-scaling-test
  (testing "memory scaling for backward pass"
    (let [K 5
          score-fn (fn [params]
                     (let [diff (mx/subtract params (mx/scalar 2.0))]
                       (mx/negative (mx/sum (mx/multiply diff diff)))))
          init (mx/array [0.0 0.0 0.0 0.0 0.0])
          std (mx/scalar 0.1)
          results (atom [])]
      (doseq [T [10 50 100 500 1000]]
        (let [noise (rng/normal (rng/fresh-key T) [T K])
              uniforms (rng/uniform (rng/fresh-key (+ T 1000)) [T])
              _ (mx/materialize! noise uniforms)
              _ (mx/clear-cache!)
              mem-before (.-heapUsed (js/process.memoryUsage))
              chain-fn (cg/make-differentiable-chain score-fn std T K)
              objective (fn [p0] (score-fn (chain-fn p0 noise uniforms)))
              grad-fn (mx/grad objective)
              g (grad-fn init)
              _ (mx/eval! g)
              mem-after (.-heapUsed (js/process.memoryUsage))
              mem-kb (/ (- mem-after mem-before) 1024.0)
              grad-norm (mx/realize (mx/sum (mx/abs g)))]
          (swap! results conj {:T T :mem-kb mem-kb :grad-norm grad-norm})))

      (let [results @results
            mem-100 (js/Math.abs (:mem-kb (first (filter #(= 100 (:T %)) results))))
            mem-1000 (js/Math.abs (:mem-kb (first (filter #(= 1000 (:T %)) results))))]
        (when (and mem-100 mem-1000 (> mem-100 1))
          (let [ratio (/ mem-1000 mem-100)]
            (is (< ratio 20.0) (str "Gate 5: memory scales <= 20x for 10x chain length (got " (.toFixed ratio 1) "x)")))))

      (let [last-result (last @results)]
        (is (js/isFinite (:grad-norm last-result)) "Gate 5: T=1000,K=5 completes without OOM")))))

;; ---------------------------------------------------------------------------
;; 7. Longer chain gradient quality
;; ---------------------------------------------------------------------------

(deftest longer-chain-gradient-quality-test
  (testing "gradient quality doesn't degrade with chain length"
    (let [score-fn (fn [params]
                     (let [diff (mx/subtract params (mx/scalar 1.5))]
                       (mx/negative (mx/sum (mx/multiply diff diff)))))
          K 2
          eps 1e-4
          init (mx/array [0.5 -0.3])]
      (doseq [T [1 10 50]]
        (let [noise (rng/normal (rng/fresh-key 99) [T K])
              uniforms (rng/uniform (rng/fresh-key 100) [T])
              _ (mx/materialize! noise uniforms)
              chain-fn (cg/make-differentiable-chain score-fn (mx/scalar 0.1) T K)
              objective (fn [p0] (score-fn (chain-fn p0 noise uniforms)))
              grad-fn (mx/grad objective)
              analytical (mx/->clj (let [g (grad-fn init)] (mx/eval! g) g))
              fd (mapv (fn [i]
                         (let [v (mx/->clj init)
                               p+ (mx/array (update v i + eps))
                               p- (mx/array (update v i - eps))]
                           (/ (- (mx/realize (objective p+))
                                 (mx/realize (objective p-)))
                              (* 2 eps))))
                       (range K))
              max-rel-err (apply max
                            (map (fn [[a f]]
                                   (if (< (js/Math.abs f) 1e-6) 0
                                     (/ (js/Math.abs (- a f))
                                        (max (js/Math.abs f) 1e-10))))
                                 (map vector analytical fd)))]
          (is (< max-rel-err 0.01) (str "T=" T " gradient relative error < 1%")))))))

(cljs.test/run-tests)
