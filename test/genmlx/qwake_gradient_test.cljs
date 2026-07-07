;; @tier fast
(ns genmlx.qwake-gradient-test
  "genmlx-vsmi: the :qwake gradient must match the wake-phase gradient with
   FROZEN samples — only the SNIS weights were stop-gradiented, so with the
   default :reparam sampler the pathwise term sum_k w_k (dlogq/dz)(dz/dtheta)
   leaked into grad-theta (nonzero expectation => biased guide updates).

   Conjugate oracle: guide q = N(theta, 1) sampled by reparameterization
   z_k = theta + eps_k. The wake objective at frozen z is
   f(theta) = sum_k w_k log N(z_k; theta, 1), whose exact gradient is
   sum_k w_k (z_k - theta) (weights frozen at the sampling theta). The
   autodiff gradient of the FULL pipeline (theta -> samples -> qwake) must
   equal it; the pre-fix leak adds sum_k w_k d(logq)/dz * 1 = -sum_k w_k
   (z_k - theta) * 1 ... = cancels the true term entirely (dz/dtheta = 1,
   dlogq/dz = -(z-theta)), driving the gradient toward ZERO — maximal bias.

   Run: bunx --bun nbb@1.4.208 test/genmlx/qwake_gradient_test.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.inference.vi :as vi]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))

(defn assert-true [label v]
  (if v
    (do (swap! pass inc) (println "  PASS" label))
    (do (swap! fail inc) (println "  FAIL" label))))

;; log p(z) ∝ log N(z;0,1) + log N(1; z,1)  (unnormalized posterior, y*=1)
(defn- log-p [z]
  (let [z (mx/index z 0)]
    (mx/add (mx/multiply (mx/scalar -0.5) (mx/square z))
            (mx/multiply (mx/scalar -0.5) (mx/square (mx/subtract (mx/scalar 1.0) z))))))

;; guide q = N(theta, 1)
(defn- log-q [z theta]
  (let [z (mx/index z 0)]
    (mx/multiply (mx/scalar -0.5) (mx/square (mx/subtract z theta)))))

(def n 64)
(def key0 (rng/fresh-key 42))
(def eps (rng/normal key0 [n 1]))          ;; fixed noise
(def theta0 (mx/scalar 0.3))

(defn- sample-at [theta] (mx/add eps theta)) ;; reparameterized z = theta + eps

;; pipeline loss: theta -> samples(theta) -> qwake objective
(def pipeline
  (fn [theta]
    (let [obj (vi/qwake-objective log-p (fn [z] (log-q z theta)))]
      (obj (sample-at theta)))))

(def auto-grad (mx/realize ((mx/grad pipeline) theta0)))

;; exact frozen-sample gradient: sum_k w_k (z_k - theta)
(def exact
  (let [zs (vec (mx/->clj (mx/reshape (sample-at theta0) [n])))
        t (mx/realize theta0)
        lps (mapv (fn [z] (+ (* -0.5 z z) (* -0.5 (- 1.0 z) (- 1.0 z)))) zs)
        lqs (mapv (fn [z] (* -0.5 (- z t) (- z t))) zs)
        lws (mapv - lps lqs)
        m (apply max lws)
        ws (mapv #(js/Math.exp (- % m)) lws)
        s (reduce + ws)
        wn (mapv #(/ % s) ws)]
    (reduce + (map (fn [w z] (* w (- z t))) wn zs))))

(println "  autodiff grad =" auto-grad "| exact frozen-sample grad =" exact)
(assert-true "qwake pipeline gradient == exact wake gradient with frozen samples"
             (< (js/Math.abs (- auto-grad exact)) 1e-4))
(assert-true "gradient is not the leak-cancelled ~0"
             (> (js/Math.abs auto-grad) 1e-3))

(println (str "\n== qwake-gradient: " @pass " passed, " @fail " failed =="))
(when (pos? @fail) (set! (.-exitCode js/process) 1))
