(ns demo-shape-vectorization
  "DISTINCTIVE FEATURE: shape-based vectorization via MLX broadcasting.

   To run N particles, GenMLX does NOT transform the function (no vmap, no
   tracing N copies). It simply changes the array shapes at each trace site
   from scalar [] to [N]. The model BODY RUNS ONCE; MLX broadcasting carries
   the batch dimension through every downstream arithmetic op. The batched
   handler transition differs from the scalar one by literally one call:
   dist-sample-n instead of dist-sample.

   This file proves: (1) one body run yields [N]-shaped leaves, (2) the batch
   is statistically a correct prior sample, (3) the wall-clock payoff vs an
   N-iteration scalar loop, and (4) vgenerate batches with a scalar-pinned
   constrained leaf and an [N] unconstrained leaf."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.mlx.random :as rng]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ── A small model: mu ~ N(0,2);  y ~ N(mu,1) ──────────────────────────────
;; Prior on mu: mean 0, variance 2^2 = 4.
(def model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 2))]
      (trace :y (dist/gaussian mu 1))
      mu)))

(defn- num [v] (mx/item v))                 ; force one scalar -> JS number
(defn- leaf [tr addr]                        ; pull a (possibly [N]) leaf array
  (cm/get-value (cm/get-submap (:choices tr) addr)))

;; ===========================================================================
(println "\n=== (b) ONE body execution -> [N]-shaped leaves ===")
(let [n   10000
      vt  (dyn/vsimulate model [] n (rng/fresh-key))
      mu-n (leaf vt :mu)
      y-n  (leaf vt :y)]
  (mx/eval! mu-n y-n)
  (println "N particles requested:   " n)
  (println "shape of :mu leaf:        " (mx/shape mu-n)
           "<- ONE run of the body, not" n "runs")
  (println "shape of :y  leaf:        " (mx/shape y-n))
  (println "score array shape:        " (mx/shape (:score vt))
           "(one log-density per particle)")

  ;; ── (c) statistical correctness of the prior batch ──────────────────────
  (println "\n=== (c) The batch is a correct prior sample ===")
  (println "E[mu]   over N particles: " (num (mx/mean mu-n))     "  (prior mean 0)")
  (println "Var[mu] over N particles: " (num (mx/variance mu-n)) "  (prior var 4)")
  (println "E[y]    over N particles: " (num (mx/mean y-n))      "  (E[y]=E[mu]=0)")
  (println "Var[y]  over N particles: " (num (mx/variance y-n))
           "  (Var[mu]+1 = 5)"))

;; ===========================================================================
;; (d) The payoff: vectorized vsimulate(N) vs an N-iteration scalar loop.
(println "\n=== (d) Wall-clock payoff: vectorized vs scalar loop ===")
(let [n 4000]
  ;; warm up both paths (JIT + Metal kernel compile) so timing is fair.
  (mx/eval! (:score (dyn/vsimulate model [] 16 (rng/fresh-key))))
  (mx/eval! (:score (p/simulate (dyn/auto-key model) [])))

  ;; --- vectorized: body runs ONCE, [N] arrays throughout ---
  (let [t0  (js/performance.now)
        vt  (dyn/vsimulate model [] n (rng/fresh-key))
        _   (mx/eval! (:score vt) (leaf vt :mu) (leaf vt :y))
        t1  (js/performance.now)
        vec-ms (- t1 t0)

        ;; --- scalar loop: body runs N times ---
        s0  (js/performance.now)
        scalar-scores
        (let [m-keyed (dyn/auto-key model)]
          (loop [i 0, acc 0.0]
            (if (< i n)
              (let [tr (p/simulate m-keyed [])]
                (recur (inc i) (+ acc (num (:score tr)))))
              acc)))
        s1  (js/performance.now)
        scalar-ms (- s1 s0)]
    (println "N =" n "particles")
    (println "vectorized vsimulate:     " (.toFixed vec-ms 2) "ms  (ONE body run)")
    (println "scalar loop of p/simulate:" (.toFixed scalar-ms 2) "ms  (" n "body runs)")
    (println "speedup factor:           "
             (.toFixed (/ scalar-ms (max vec-ms 0.001)) 1) "x")
    (println "(scalar sanity: mean per-particle score ="
             (.toFixed (/ scalar-scores n) 4) ")")))

;; ===========================================================================
;; (e) vgenerate also batches: constrained leaf is scalar, the rest are [N].
(println "\n=== (e) vgenerate batches with a constrained leaf pinned scalar ===")
(let [n   5000
      obs (cm/choicemap :y (mx/scalar 1.5))      ; condition y = 1.5
      vt  (dyn/vgenerate model [] obs n (rng/fresh-key))
      mu-n (leaf vt :mu)
      y-c  (leaf vt :y)
      w    (:weight vt)]
  (mx/eval! mu-n y-c w)
  (println "constrained :y shape:     " (mx/shape y-c)
           "(scalar — pinned to the observation)")
  (println "constrained :y value:     " (num y-c) "  (= 1.5)")
  (println "unconstrained :mu shape:  " (mx/shape mu-n) "(still [N])")
  (println "importance weight shape:  " (mx/shape w) "(one log-weight per particle)")
  ;; Posterior of mu given y=1.5: N(mean = 1.5 * 4/5 = 1.2, var = 4/5 = 0.8).
  ;; Self-normalized importance estimate of E[mu | y=1.5]:
  (let [logw  w
        norm  (mx/subtract logw (mx/logsumexp logw))   ; log normalized weights
        wn    (mx/exp norm)
        e-mu  (num (mx/sum (mx/multiply wn mu-n)))]
    (println "IS estimate E[mu | y=1.5]:" (.toFixed e-mu 4)
             "  (analytic posterior mean 1.2)")))

(println "\n=== done ===")
