(ns genmlx.l2-gate-test
  "Level 2 Gate 0: Tensor-native score vs GFI-based score benchmark.

   Experiment: Build make-tensor-score for a 5-site static Gaussian model.
   Compare against make-score-fn (GFI-based) in two settings:
   (a) raw function call (not compiled)
   (b) inside mx/compile-fn chain

   Success criterion: (a) raw: >2x speedup."
  (:require [genmlx.gen :refer [gen]]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.compiled :as compiled]
            [genmlx.tensor-trace :as tt]
            [genmlx.inference.util :as iu]))

;; ---------------------------------------------------------------------------
;; Benchmark model: 5-site static Gaussian
;; ---------------------------------------------------------------------------

(def bench-model
  (gen []
    (let [a (trace :a (dist/gaussian 0 1))
          b (trace :b (dist/gaussian a 1))
          c (trace :c (dist/gaussian 0 2))
          d (trace :d (dist/gaussian (mx/add b c) 0.5))
          e (trace :e (dist/gaussian d 0.1))]
      e)))

;; ---------------------------------------------------------------------------
;; Timing helper
;; ---------------------------------------------------------------------------

(defn bench
  "Run f n-warmup times, then n-iter times. Return {:mean-us :min-us :max-us}."
  [f n-warmup n-iter]
  (dotimes [_ n-warmup] (f))
  (let [times (mapv (fn [_]
                      (let [start (js/performance.now)
                            _ (f)
                            end (js/performance.now)]
                        (* 1000.0 (- end start)))) ;; microseconds
                    (range n-iter))
        sorted (sort times)]
    {:mean-us (/ (reduce + sorted) (count sorted))
     :min-us (first sorted)
     :max-us (last sorted)
     :median-us (nth sorted (quot (count sorted) 2))}))

;; ---------------------------------------------------------------------------
;; Gate 0a: Raw score function comparison
;; ---------------------------------------------------------------------------

(println "\n===============================================")
(println "Gate 0: Tensor-Native Score vs GFI-Based Score")
(println "===============================================")

(let [model (dyn/auto-key bench-model)
      schema (:schema bench-model)
      source (:source bench-model)
      ;; Generate a trace to get observations
      trace (p/simulate model [])
      choices (:choices trace)
      e-val (cm/get-value (cm/get-submap choices :e))
      obs (cm/choicemap :e e-val)
      ;; Latent addresses (a, b, c, d)
      latent-addrs [:a :b :c :d]
      ;; Build GFI score
      gfi-score-fn (iu/make-score-fn model [] obs latent-addrs)
      ;; Build tensor-native score
      {:keys [score-fn latent-index]} (compiled/make-tensor-score-with-index
                                        schema source [] obs)
      ;; Pack latent values
      vm (into {} (map (fn [addr]
                         [addr (cm/get-value (cm/get-submap choices addr))])
                       latent-addrs))
      latent-tensor (tt/pack-values vm latent-index)
      _ (mx/eval! latent-tensor)]

  ;; Verify both produce same score
  (let [gfi-s (gfi-score-fn latent-tensor)
        ts-s (score-fn latent-tensor)]
    (mx/eval! gfi-s)
    (mx/eval! ts-s)
    (println (str "\nCorrectness check:"))
    (println (str "  GFI score:    " (mx/item gfi-s)))
    (println (str "  Tensor score: " (mx/item ts-s)))
    (println (str "  Match: " (< (js/Math.abs (- (mx/item gfi-s) (mx/item ts-s))) 1e-4))))

  (println "\n--- Gate 0a: Raw function calls (no mx/compile-fn) ---")
  (let [gfi-timing (bench #(let [s (gfi-score-fn latent-tensor)]
                              (mx/eval! s)) 100 500)
        ts-timing (bench #(let [s (score-fn latent-tensor)]
                            (mx/eval! s)) 100 500)
        speedup (/ (:median-us gfi-timing) (:median-us ts-timing))]
    (println (str "  GFI score:    median=" (.toFixed (:median-us gfi-timing) 1)
                  "us  mean=" (.toFixed (:mean-us gfi-timing) 1) "us"))
    (println (str "  Tensor score: median=" (.toFixed (:median-us ts-timing) 1)
                  "us  mean=" (.toFixed (:mean-us ts-timing) 1) "us"))
    (println (str "  Speedup: " (.toFixed speedup 2) "x"))
    (println (str "  Gate 0a criterion (>2x): " (if (> speedup 2.0) "PASS" "INVESTIGATE"))))

  ;; ---------------------------------------------------------------------------
  ;; Gate 0b: Inside mx/compile-fn
  ;; ---------------------------------------------------------------------------

  (println "\n--- Gate 0b: Inside mx/compile-fn ---")
  (let [compiled-gfi (mx/compile-fn gfi-score-fn)
        compiled-ts (mx/compile-fn score-fn)
        ;; Warm up both
        _ (mx/eval! (compiled-gfi latent-tensor))
        _ (mx/eval! (compiled-ts latent-tensor))
        gfi-timing (bench #(let [s (compiled-gfi latent-tensor)]
                              (mx/eval! s)) 100 500)
        ts-timing (bench #(let [s (compiled-ts latent-tensor)]
                            (mx/eval! s)) 100 500)
        speedup (/ (:median-us gfi-timing) (:median-us ts-timing))]
    (println (str "  Compiled GFI:    median=" (.toFixed (:median-us gfi-timing) 1)
                  "us  mean=" (.toFixed (:mean-us gfi-timing) 1) "us"))
    (println (str "  Compiled Tensor: median=" (.toFixed (:median-us ts-timing) 1)
                  "us  mean=" (.toFixed (:mean-us ts-timing) 1) "us"))
    (println (str "  Speedup: " (.toFixed speedup 2) "x"))
    (println (str "  Gate 0b criterion (measurable improvement): "
                  (cond
                    (> speedup 1.5) "CLEAR WIN"
                    (> speedup 1.1) "MARGINAL"
                    :else "NO DIFFERENCE (expected — mx/compile-fn captures the same graph)")))))

(println "\n--- Gate 0 complete ---")
