(ns genmlx.gpu-benchmark
  "GPU-intensive benchmarks for GenMLX.
   Tests that actually exercise Apple Silicon GPU via MLX."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.vi :as vi])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn time-it [label f]
  (let [start (js/Date.now)
        result (f)
        elapsed (- (js/Date.now) start)]
    (println (str "  " label ": " elapsed "ms"))
    result))

(defn section [title]
  (println (str "\n── " title " ──")))

(println "\n=== GenMLX GPU Benchmarks ===")
(println (str "  Device: " (str (mx/default-device))))

;; ============================================================================
;; 1. Large Matrix Operations
;; ============================================================================

(section "Large Matrix Operations")

;; Matmul
(doseq [n [256 512 1024]]
  (let [a (mx/random-normal [n n])
        b (mx/random-normal [n n])]
    (mx/eval! a b)
    (time-it (str n "x" n " matmul")
      (fn []
        (let [c (mx/matmul a b)]
          (mx/eval! c)
          c)))))

;; Cholesky decomposition
(doseq [n [128 256 512]]
  (let [;; Build positive definite matrix: A^T A + nI
        a (mx/random-normal [n n])
        _ (mx/eval! a)
        ata (mx/matmul (mx/transpose a) a)
        spd (mx/add ata (mx/multiply (mx/scalar (float n)) (mx/eye n)))]
    (mx/eval! spd)
    (time-it (str n "x" n " Cholesky")
      (fn []
        (let [L (mx/cholesky spd)]
          (mx/eval! L)
          L)))))

;; Linear solve
(doseq [n [128 256 512]]
  (let [a (mx/random-normal [n n])
        _ (mx/eval! a)
        ata (mx/add (mx/matmul (mx/transpose a) a)
                    (mx/multiply (mx/scalar (float n)) (mx/eye n)))
        b (mx/random-normal [n 1])]
    (mx/eval! ata b)
    (time-it (str n "x" n " linear solve")
      (fn []
        (let [x (mx/solve ata b)]
          (mx/eval! x)
          x)))))

;; ============================================================================
;; 2. Batched Array Operations
;; ============================================================================

(section "Batched Array Operations")

;; Element-wise ops on large arrays
(doseq [n [100000 1000000]]
  (let [a (mx/random-normal [n])
        b (mx/random-normal [n])]
    (mx/eval! a b)
    (time-it (str (/ n 1000) "K element-wise (add+mul+exp+sum)")
      (fn []
        (let [c (mx/add a b)
              d (mx/multiply c a)
              e (mx/exp d)
              s (mx/sum e)]
          (mx/eval! s)
          (mx/item s))))))

;; Softmax on large vectors
(doseq [n [10000 100000]]
  (let [logits (mx/random-normal [n])]
    (mx/eval! logits)
    (time-it (str (/ n 1000) "K softmax")
      (fn []
        (let [probs (mx/softmax logits)]
          (mx/eval! probs)
          probs)))))

;; Batched outer products
(let [n 1000
      d 100
      a (mx/random-normal [n d])
      b (mx/random-normal [n d])]
  (mx/eval! a b)
  (time-it (str n " batched " d "-dim dot products via matmul")
    (fn []
      (let [;; (n,d) @ (d,n) -> (n,n) but we just want diagonal
            ;; Use element-wise multiply + sum instead
            dots (mx/sum (mx/multiply a b) [1])]
        (mx/eval! dots)
        dots))))

;; ============================================================================
;; 3. Autograd through Large Computations
;; ============================================================================

(section "Autograd")

;; Gradient of a large quadratic form
(doseq [n [100 500 1000]]
  (let [A (let [a (mx/random-normal [n n])]
            (mx/eval! a)
            (mx/add (mx/matmul (mx/transpose a) a)
                    (mx/multiply (mx/scalar (float n)) (mx/eye n))))
        _ (mx/eval! A)
        f (fn [x] (mx/squeeze (mx/matmul (mx/reshape x [1 n])
                                          (mx/matmul A (mx/reshape x [n 1])))))
        grad-f (mx/grad f)
        x0 (mx/random-normal [n])]
    (mx/eval! x0)
    (time-it (str n "-dim quadratic gradient")
      (fn []
        (let [g (grad-f x0)]
          (mx/eval! g)
          g)))))

;; value-and-grad through neural-net-like computation
(let [n 256
      d-in 64
      d-hidden 128
      d-out 1
      W1 (mx/random-normal [d-in d-hidden])
      W2 (mx/random-normal [d-hidden d-out])
      x (mx/random-normal [n d-in])
      y (mx/random-normal [n d-out])]
  (mx/eval! W1 W2 x y)
  (let [loss-fn (fn [w1]
                  (let [h (mx/tanh (mx/matmul x w1))
                        pred (mx/matmul h W2)
                        diff (mx/subtract pred y)]
                    (mx/mean (mx/multiply diff diff))))
        vg (mx/value-and-grad loss-fn)]
    (time-it (str n "x" d-in "->" d-hidden "->" d-out " forward+backward")
      (fn []
        (let [[v g] (vg W1)]
          (mx/eval! v g)
          [v g])))))

;; ============================================================================
;; 4. Compiled Function Performance
;; ============================================================================

(section "Compiled vs Uncompiled")

(let [n 1000
      f (fn [x]
          (let [s (mx/sum (mx/square x))
                e (mx/exp (mx/negative (mx/divide s (mx/scalar 2.0))))]
            (mx/multiply (mx/scalar (/ 1.0 (js/Math.sqrt (* 2 js/Math.PI)))) e)))
      f-compiled (mx/compile-fn f)
      x (mx/random-normal [n])]
  (mx/eval! x)
  ;; Warm up compiled version
  (let [r (f-compiled x)] (mx/eval! r))

  (time-it (str n "-dim Gaussian density (uncompiled, 100 iters)")
    (fn []
      (dotimes [_ 100]
        (let [r (f x)] (mx/eval! r)))))

  (time-it (str n "-dim Gaussian density (compiled, 100 iters)")
    (fn []
      (dotimes [_ 100]
        (let [r (f-compiled x)] (mx/eval! r))))))

;; ============================================================================
;; 5. Multivariate Normal (High-Dimensional)
;; ============================================================================

(section "Multivariate Normal")

(doseq [k [10 50 100]]
  (let [mean-vec (mx/zeros [k])
        ;; Random SPD covariance
        a (mx/random-normal [k k])
        _ (mx/eval! a)
        cov (mx/add (mx/matmul (mx/transpose a) a)
                    (mx/multiply (mx/scalar (float k)) (mx/eye k)))
        _ (mx/eval! cov)
        mvn (dist/multivariate-normal mean-vec cov)]
    ;; Sample
    (time-it (str k "-dim MVN: 1000 samples")
      (fn []
        (dotimes [_ 1000]
          (let [v (dist/sample mvn)]
            (mx/eval! v)))))

    ;; Log-prob
    (let [x (dist/sample mvn)]
      (mx/eval! x)
      (time-it (str k "-dim MVN: 1000 log-probs")
        (fn []
          (dotimes [_ 1000]
            (let [lp (dist/log-prob mvn x)]
              (mx/eval! lp))))))))

;; ============================================================================
;; 6. HMC on High-Dimensional Model
;; ============================================================================

(section "HMC Inference")

;; HMC models must not call mx/eval! inside the body (breaks gradient tracing).
;; The score function only needs the log-density graph, not materialized values.

(doseq [d [10 20]]
  (let [model (gen [d]
                (let [params (mapv (fn [i]
                                    (dyn/trace (keyword (str "x" i))
                                               (dist/gaussian 0 1)))
                                  (range d))]
                  params))
        addresses (mapv #(keyword (str "x" %)) (range d))
        observations (reduce (fn [cm [i v]]
                               (cm/set-choice cm [(keyword (str "x" i))] (mx/scalar v)))
                             cm/EMPTY
                             (map-indexed vector (repeatedly d #(* 2.0 (- (js/Math.random) 0.5)))))]

    (time-it (str d "-dim HMC: 200 samples, 10 leapfrog steps")
      (fn []
        (mcmc/hmc {:samples 200 :step-size 0.01 :leapfrog-steps 10
                   :burn 50 :addresses addresses}
                  model [d] observations)))))

;; ============================================================================
;; 7. Variational Inference (ELBO Gradient via vmap)
;; ============================================================================

(section "Variational Inference")

(doseq [d [5 10 20]]
  (let [;; Target: d-dim standard normal
        log-density (fn [params]
                      (mx/multiply (mx/scalar -0.5)
                                   (mx/sum (mx/multiply params params))))
        init-params (mx/random-normal [d])]
    (mx/eval! init-params)
    (time-it (str d "-dim VI: 200 iterations, 10 ELBO samples")
      (fn []
        (vi/vi {:iterations 200 :learning-rate 0.01 :elbo-samples 10}
               log-density init-params)))))

;; ============================================================================
;; 8. Leapfrog Integration (Isolated)
;; ============================================================================

(section "Leapfrog Integration (Isolated)")

(doseq [d [10 50 100]]
  (let [;; Quadratic potential: U(q) = 0.5 * q^T q
        grad-neg-U (mx/compile-fn (mx/grad (fn [q] (mx/multiply (mx/scalar -0.5) (mx/sum (mx/square q))))))
        eps (mx/scalar 0.01)
        half-eps (mx/scalar 0.005)
        q0 (mx/random-normal [d])
        p0 (mx/random-normal [d])]
    (mx/eval! q0 p0)
    (time-it (str d "-dim leapfrog: 100 steps")
      (fn []
        (loop [i 0, q q0, p p0]
          (if (>= i 100)
            (do (mx/eval! q p) [q p])
            (let [g (grad-neg-U q)
                  p (mx/subtract p (mx/multiply half-eps g))
                  q (mx/add q (mx/multiply eps p))
                  g (grad-neg-U q)
                  p (mx/subtract p (mx/multiply half-eps g))]
              (mx/eval! q p)
              (recur (inc i) q p))))))))

;; ============================================================================
;; 9. Random Number Generation Throughput
;; ============================================================================

(section "Random Number Generation")

(doseq [n [10000 100000 1000000]]
  (time-it (str (/ n 1000) "K normal samples")
    (fn []
      (let [x (mx/random-normal [n])]
        (mx/eval! x)
        x))))

(doseq [n [10000 100000 1000000]]
  (time-it (str (/ n 1000) "K uniform samples")
    (fn []
      (let [x (mx/random-uniform [n])]
        (mx/eval! x)
        x))))

;; ============================================================================
;; 10. Logsumexp / Softmax (Numerically Stable Reductions)
;; ============================================================================

(section "Numerically Stable Reductions")

(doseq [n [10000 100000 1000000]]
  (let [x (mx/random-normal [n])]
    (mx/eval! x)
    (time-it (str (/ n 1000) "K logsumexp")
      (fn []
        (let [r (mx/logsumexp x)]
          (mx/eval! r)
          (mx/item r))))))

;; 2D logsumexp (particle weights)
(doseq [[particles obs] [[100 1000] [1000 100]]]
  (let [x (mx/random-normal [particles obs])]
    (mx/eval! x)
    (time-it (str particles " particles x " obs " obs logsumexp(axis=1)")
      (fn []
        (let [r (mx/logsumexp x [1])]
          (mx/eval! r)
          r)))))

(println "\n=== GPU Benchmarks Complete ===")
