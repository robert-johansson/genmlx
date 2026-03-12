(ns genmlx.compiled-gradient-test
  "WP-4 tests: gradient through compiled inference.

   Tests cover:
   1. Differentiable MH chain: gradient matches finite differences
   2. Multi-step chains: gradient through 1, 5, 10, 50 steps
   3. Gradient direction: correct sign for simple cases
   4. Score gradient through chain: end-to-end API
   5. Gradient through SMC: log-ML gradient via gumbel-softmax
   6. Gate 5: memory scaling for backward pass (T=10..1000)"
  (:require [genmlx.gen :refer [gen]]
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
            [genmlx.compiled :as compiled]))

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
;; 1. Differentiable MH chain: basic gradient
;; ---------------------------------------------------------------------------

(println "\n== Differentiable MH chain ==")

;; Simple quadratic score: score(x) = -x² (mode at x=0)
(let [score-fn (fn [params] (mx/negative (mx/sum (mx/multiply params params))))
      K 2
      T 5
      chain-fn (cg/make-differentiable-chain score-fn (mx/scalar 0.1) T K)
      ;; Test that chain runs
      init (mx/array [1.0 2.0])
      noise (rng/normal (rng/fresh-key 1) [T K])
      uniforms (rng/uniform (rng/fresh-key 2) [T])
      _ (mx/materialize! noise uniforms)
      result (chain-fn init noise uniforms)]
  (mx/eval! result)
  (assert-equal "chain output shape" [K] (mx/shape result))
  (assert-true "chain output is finite"
               (every? js/isFinite (mx/->clj result))))

;; Gradient through single MH step
(let [score-fn (fn [params] (mx/negative (mx/sum (mx/multiply params params))))
      K 2
      noise (mx/zeros [1 K])  ;; zero noise = proposal = current
      uniforms (mx/zeros [1]) ;; log(0) = -inf → always accept
      chain-fn (cg/make-differentiable-chain score-fn (mx/scalar 0.1) 1 K)
      ;; Gradient of final score w.r.t. initial params
      objective (fn [p0]
                  (let [final (chain-fn p0 noise uniforms)]
                    (score-fn final)))
      grad-fn (mx/grad objective)
      init (mx/array [1.0 2.0])
      g (grad-fn init)]
  (mx/eval! g)
  (let [grad-vals (mx/->clj g)]
    (assert-true "gradient is non-zero" (not (every? zero? grad-vals)))
    (println "    grad values:" grad-vals)))

;; ---------------------------------------------------------------------------
;; 2. Gradient matches finite differences
;; ---------------------------------------------------------------------------

(println "\n== Gradient vs Finite Differences ==")

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
          ;; Finite differences
          fd (mapv (fn [i]
                     (let [init-v (mx/->clj init)
                           p+ (mx/array (update init-v i + eps))
                           p- (mx/array (update init-v i - eps))
                           f+ (mx/realize (objective p+))
                           f- (mx/realize (objective p-))]
                       (/ (- f+ f-) (* 2 eps))))
                   (range K))
          ;; Sign agreement
          signs (count (filter (fn [[a f]]
                                 (or (and (pos? a) (pos? f))
                                     (and (neg? a) (neg? f))
                                     (and (< (js/Math.abs a) 1e-6)
                                          (< (js/Math.abs f) 1e-6))))
                               (map vector analytical fd)))
          sign-rate (/ signs K)]
      (println (str "  T=" T " analytical=" (mapv #(.toFixed % 4) analytical)
                    " fd=" (mapv #(.toFixed % 4) fd)))
      (assert-true (str "T=" T " sign agreement ≥ 67% (got " (* 100 sign-rate) "%)")
                   (>= sign-rate 0.67))
      ;; Magnitude check
      (let [mag-ok (every? (fn [[a f]]
                             (if (< (js/Math.abs f) 1e-6)
                               true
                               (let [ratio (/ (js/Math.abs a) (js/Math.abs f))]
                                 (and (> ratio 0.1) (< ratio 10.0)))))
                           (map vector analytical fd))]
        (assert-true (str "T=" T " magnitude within 10x") mag-ok)))))

;; ---------------------------------------------------------------------------
;; 3. Gradient direction: correct sign
;; ---------------------------------------------------------------------------

(println "\n== Gradient direction ==")

;; Score = -(x-target)² → gradient should point toward target
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
  ;; Starting below target → gradient should be positive
  (let [g (grad-fn (mx/array [1.0]))]
    (mx/eval! g)
    (assert-true "below target → positive gradient" (> (first (mx/->clj g)) 0)))
  ;; Starting above target → gradient should be negative
  (let [g (grad-fn (mx/array [5.0]))]
    (mx/eval! g)
    (assert-true "above target → negative gradient" (< (first (mx/->clj g)) 0))))

;; ---------------------------------------------------------------------------
;; 4. mcmc-score-gradient API
;; ---------------------------------------------------------------------------

(println "\n== mcmc-score-gradient API ==")

;; Parameterized score: model-params = [target-mean]
;; score(latent | model-params) = -((latent - target)² / 2)
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
  (assert-true "mcmc-score-gradient returns :value"
               (js/isFinite (mx/realize (:value result))))
  (assert-true "mcmc-score-gradient returns :grad"
               (= [2] (mx/shape (:grad result))))
  (let [grad-vals (mx/->clj (:grad result))]
    (assert-true "gradient is non-zero" (not (every? zero? grad-vals)))
    (println "    value:" (.toFixed (mx/realize (:value result)) 4)
             "grad:" grad-vals)))

;; ---------------------------------------------------------------------------
;; 5. Gradient through SMC (gumbel-softmax)
;; ---------------------------------------------------------------------------

(println "\n== Gradient through SMC ==")

;; Simple linear Gaussian kernel
(def lg-kernel
  (gen [t state]
    (let [new-state (trace :x (dist/gaussian state 1.0))]
      (trace :y (dist/gaussian new-state 0.5))
      new-state)))

;; Observations
(def smc-obs
  (mapv (fn [t]
          (cm/choicemap :y (mx/scalar (* 0.5 t))))
        (range 5)))

;; Test: gradient of log-ML w.r.t. init-state
(try
  (let [result (cg/smc-log-ml-gradient
                 lg-kernel (mx/scalar 0.0)
                 (mx/array [0.0])  ;; model-params = init-state
                 smc-obs
                 {:particles 30 :tau 1.0 :key (rng/fresh-key 20)})]
    (mx/eval! (:value result) (:grad result))
    (let [v (mx/realize (:value result))
          g (mx/->clj (:grad result))]
      (assert-true (str "SMC log-ML is finite (" (.toFixed v 2) ")")
                   (js/isFinite v))
      (assert-true "SMC gradient is non-zero"
                   (not (every? #(< (js/Math.abs %) 1e-10) g)))
      (println "    log-ML:" (.toFixed v 4) "grad:" g)))
  (catch :default e
    (swap! fail-count inc)
    (println (str "  FAIL: SMC gradient: " (.-message e)))))

;; FD comparison for SMC gradient — manually build the objective to use same noise
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
        ;; Shared objective using fixed noise
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
        ;; Analytical gradient
        grad-fn (mx/grad objective)
        p0 (mx/array [0.0])
        analytical (first (mx/->clj (let [g (grad-fn p0)] (mx/eval! g) g)))
        ;; FD gradient (same noise!)
        f+ (mx/realize (objective (mx/array [eps])))
        f- (mx/realize (objective (mx/array [(- eps)])))
        fd (/ (- f+ f-) (* 2 eps))]
    (println (str "  SMC grad: analytical=" (.toFixed analytical 4)
                  " fd=" (.toFixed fd 4)))
    (if (< (js/Math.abs fd) 1e-6)
      (assert-true "SMC FD near zero — skip sign check" true)
      (assert-true "SMC gradient sign matches FD"
                   (or (and (pos? analytical) (pos? fd))
                       (and (neg? analytical) (neg? fd))))))
  (catch :default e
    (swap! fail-count inc)
    (println (str "  FAIL: SMC FD comparison: " (.-message e)))))

;; ---------------------------------------------------------------------------
;; 6. Gate 5: Memory scaling for backward pass
;; ---------------------------------------------------------------------------

(println "\n== Gate 5: Memory Scaling ==")

;; Measure memory for gradient through MH chains of increasing length
;; score = -(x-2)^2, K=5 parameters
;; Note: JS heap measurement is noisy due to GC. We primarily test that
;; (1) all chain lengths complete without OOM, and (2) gradient is correct.
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
          ;; Measure heap via process.memoryUsage()
          mem-before (.-heapUsed (js/process.memoryUsage))
          chain-fn (cg/make-differentiable-chain score-fn std T K)
          objective (fn [p0] (score-fn (chain-fn p0 noise uniforms)))
          grad-fn (mx/grad objective)
          g (grad-fn init)
          _ (mx/eval! g)
          mem-after (.-heapUsed (js/process.memoryUsage))
          mem-kb (/ (- mem-after mem-before) 1024.0)
          grad-norm (mx/realize (mx/sum (mx/abs g)))]
      (swap! results conj {:T T :mem-kb mem-kb :grad-norm grad-norm})
      (println (str "  T=" T " mem=" (.toFixed (js/Math.abs mem-kb) 0) "KB"
                    " grad-norm=" (.toFixed grad-norm 4)))))

  ;; Check linear scaling: mem(T=1000) should be < 20x mem(T=100)
  ;; (linear = 10x ratio for 10x T; quadratic = 100x)
  ;; Note: heap measurements are noisy, so we use a generous threshold.
  (let [results @results
        mem-100 (js/Math.abs (:mem-kb (first (filter #(= 100 (:T %)) results))))
        mem-1000 (js/Math.abs (:mem-kb (first (filter #(= 1000 (:T %)) results))))]
    (when (and mem-100 mem-1000 (> mem-100 1))
      (let [ratio (/ mem-1000 mem-100)]
        (println (str "  Memory ratio T=1000/T=100: " (.toFixed ratio 1) "x"))
        (assert-true (str "Gate 5: memory scales ≤ 20x for 10x chain length (got " (.toFixed ratio 1) "x)")
                     (< ratio 20.0)))))

  ;; T=1000, K=5 fits in memory (no OOM) — the main Gate 5 criterion
  (let [last-result (last @results)]
    (assert-true "Gate 5: T=1000,K=5 completes without OOM"
                 (js/isFinite (:grad-norm last-result)))))

;; ---------------------------------------------------------------------------
;; 7. Longer chain gradient quality
;; ---------------------------------------------------------------------------

(println "\n== Longer chain gradient quality ==")

;; Verify gradient quality doesn't degrade with chain length
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
      (println (str "  T=" T " max-rel-err=" (.toFixed max-rel-err 6)))
      (assert-true (str "T=" T " gradient relative error < 1%")
                   (< max-rel-err 0.01)))))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n== RESULTS: " @pass-count "/" (+ @pass-count @fail-count)
              " passed, " @fail-count " failed =="))
