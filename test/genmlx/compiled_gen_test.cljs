(ns genmlx.compiled-gen-test
  "Tests for compiled gen functions.
   Verifies correctness and measures speedup of mx/compile-fn
   applied to the full vgenerate + gradient pipeline."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.compiled-gen :as cg]
            [genmlx.inference.differentiable :as diff]
            [genmlx.inference.fisher :as fisher])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn assert-true [label pred]
  (if pred
    (println (str "  PASS: " label))
    (println (str "  FAIL: " label))))

(defn assert-close [label expected actual tol]
  (let [ok (< (js/Math.abs (- expected actual)) tol)]
    (if ok
      (println (str "  PASS: " label " (expected " (.toFixed expected 3) ", got " (.toFixed actual 3) ")"))
      (println (str "  FAIL: " label " (expected " (.toFixed expected 3) ", got " (.toFixed actual 3) ")")))))

;; ---------------------------------------------------------------------------
;; Model setup (same as fisher_test.cljs)
;; ---------------------------------------------------------------------------

(def K-obs 10)
(def true-mu 3.0)
(def obs-data (mapv (fn [i] (+ true-mu (* 0.5 (- i 4.5)))) (range K-obs)))

(def model-1
  (gen []
    (let [mu (param :mu 0.0)]
      (doseq [i (range K-obs)]
        (trace (keyword (str "y" i))
               (dist/gaussian mu 1.0)))
      mu)))

(def obs-1
  (apply cm/choicemap
    (mapcat (fn [i] [(keyword (str "y" i)) (mx/scalar (nth obs-data i))])
            (range K-obs))))

;; ---------------------------------------------------------------------------
;; Test 1: Compiled log-ML matches uncompiled
;; ---------------------------------------------------------------------------

(println "\n=== Test 1: Compiled log-ML correctness ===")

(let [key (rng/fresh-key 42)
      params (mx/array [true-mu])
      ;; Uncompiled
      {:keys [log-ml]} (diff/log-ml-gradient
                          {:n-particles 5000 :key key}
                          model-1 [] obs-1 [:mu] params)
      _ (mx/materialize! log-ml)
      uncompiled-val (mx/item log-ml)
      ;; Compiled
      compiled-loss (cg/compile-log-ml
                      {:n-particles 5000 :key key}
                      model-1 [] obs-1 [:mu])
      compiled-val (- (mx/item (compiled-loss params)))]
  (println (str "  Uncompiled log-ML: " (.toFixed uncompiled-val 4)))
  (println (str "  Compiled log-ML:   " (.toFixed compiled-val 4)))
  (assert-close "Compiled ≈ uncompiled" uncompiled-val compiled-val 0.01))

;; ---------------------------------------------------------------------------
;; Test 2: Compiled gradient matches uncompiled
;; ---------------------------------------------------------------------------

(println "\n=== Test 2: Compiled gradient correctness ===")

(let [key (rng/fresh-key 42)
      params (mx/array [true-mu])
      ;; Uncompiled
      {:keys [grad]} (diff/log-ml-gradient
                        {:n-particles 5000 :key key}
                        model-1 [] obs-1 [:mu] params)
      _ (mx/materialize! grad)
      uncompiled-grad (mx/item (mx/index grad 0))
      ;; Compiled
      compiled-vg (cg/compile-log-ml-gradient
                    {:n-particles 5000 :key key}
                    model-1 [] obs-1 [:mu])
      [neg-lml grad-c] (compiled-vg params)
      _ (mx/materialize! neg-lml grad-c)
      compiled-grad (mx/item (mx/index grad-c 0))]
  (println (str "  Uncompiled grad: " (.toFixed uncompiled-grad 4)))
  (println (str "  Compiled grad:   " (.toFixed compiled-grad 4)))
  (assert-close "Compiled grad ≈ uncompiled" uncompiled-grad compiled-grad 0.01))

;; ---------------------------------------------------------------------------
;; Test 3: Compiled gradient determinism (same key → same result)
;; ---------------------------------------------------------------------------

(println "\n=== Test 3: Compiled determinism ===")

(let [key (rng/fresh-key 42)
      compiled-vg (cg/compile-log-ml-gradient
                    {:n-particles 5000 :key key}
                    model-1 [] obs-1 [:mu])
      params (mx/array [true-mu])
      [v1 g1] (compiled-vg params)
      [v2 g2] (compiled-vg params)]
  (mx/materialize! v1 v2 g1 g2)
  (let [val1 (mx/item v1)
        val2 (mx/item v2)]
    (println (str "  Call 1: " (.toFixed val1 6)))
    (println (str "  Call 2: " (.toFixed val2 6)))
    (assert-close "Deterministic" val1 val2 1e-6)))

;; ---------------------------------------------------------------------------
;; Test 4: Compiled gradient with different params (cache hit, same shape)
;; ---------------------------------------------------------------------------

(println "\n=== Test 4: Different params (same shape → cache hit) ===")

(let [key (rng/fresh-key 42)
      compiled-vg (cg/compile-log-ml-gradient
                    {:n-particles 5000 :key key}
                    model-1 [] obs-1 [:mu])
      [v1 _] (compiled-vg (mx/array [0.0]))
      [v2 _] (compiled-vg (mx/array [3.0]))
      [v3 _] (compiled-vg (mx/array [6.0]))]
  (mx/materialize! v1 v2 v3)
  (let [l1 (- (mx/item v1))
        l2 (- (mx/item v2))
        l3 (- (mx/item v3))]
    (println (str "  log-ML at mu=0: " (.toFixed l1 2)))
    (println (str "  log-ML at mu=3: " (.toFixed l2 2)))
    (println (str "  log-ML at mu=6: " (.toFixed l3 2)))
    ;; mu=3 should have highest log-ML (closest to data mean)
    (assert-true "log-ML(3) > log-ML(0)" (> l2 l1))
    (assert-true "log-ML(3) > log-ML(6)" (> l2 l3))))

;; ---------------------------------------------------------------------------
;; Test 5: Speedup benchmark
;; ---------------------------------------------------------------------------

(println "\n=== Test 5: Speedup benchmark ===")

(let [key (rng/fresh-key 42)
      params (mx/array [true-mu])
      n-calls 20
      ;; Uncompiled timing
      t0 (js/Date.now)
      _ (dotimes [_ n-calls]
          (let [{:keys [grad log-ml]} (diff/log-ml-gradient
                                        {:n-particles 5000 :key key}
                                        model-1 [] obs-1 [:mu] params)]
            (mx/materialize! grad log-ml)))
      t-uncompiled (- (js/Date.now) t0)
      ;; Compiled timing (compilation already done during this call)
      compiled-vg (cg/compile-log-ml-gradient
                    {:n-particles 5000 :key key}
                    model-1 [] obs-1 [:mu])
      ;; Warm up (first 2 calls)
      _ (dotimes [_ 2]
          (let [[v g] (compiled-vg params)] (mx/materialize! v g)))
      t1 (js/Date.now)
      _ (dotimes [_ n-calls]
          (let [[v g] (compiled-vg params)]
            (mx/materialize! v g)))
      t-compiled (- (js/Date.now) t1)
      speedup (/ t-uncompiled (max t-compiled 1))]
  (println (str "  Uncompiled: " t-uncompiled "ms (" n-calls " calls)"))
  (println (str "  Compiled:   " t-compiled "ms (" n-calls " calls)"))
  (println (str "  Speedup:    " (.toFixed speedup 1) "x"))
  (assert-true "Compiled is faster" (> speedup 1.0)))

;; ---------------------------------------------------------------------------
;; Test 6: 2D model (mu + log-sigma)
;; ---------------------------------------------------------------------------

(println "\n=== Test 6: 2D compiled gradient ===")

(def model-2
  (gen []
    (let [mu (param :mu 0.0)
          log-sigma (param :log-sigma 0.0)
          sigma (mx/exp log-sigma)]
      (doseq [i (range K-obs)]
        (trace (keyword (str "y" i))
               (dist/gaussian mu sigma)))
      mu)))

(let [key (rng/fresh-key 77)
      params (mx/array [true-mu 0.0])
      compiled-vg (cg/compile-log-ml-gradient
                    {:n-particles 5000 :key key}
                    model-2 [] obs-1 [:mu :log-sigma])
      [neg-lml grad] (compiled-vg params)]
  (mx/materialize! neg-lml grad)
  (let [g0 (mx/item (mx/index grad 0))
        g1 (mx/item (mx/index grad 1))]
    (println (str "  log-ML: " (.toFixed (- (mx/item neg-lml)) 2)))
    (println (str "  grad[mu]: " (.toFixed g0 4)))
    (println (str "  grad[log-sigma]: " (.toFixed g1 4)))
    ;; At mu=3 (MLE for mu), grad[mu] ≈ 0
    ;; log-sigma=0 (sigma=1) is NOT the MLE for sigma (data has spread)
    ;; so grad[log-sigma] is nonzero — just check it's finite and reasonable
    (assert-close "grad[mu] ≈ 0 at MLE" 0.0 g0 1.0)
    (assert-true "grad[log-sigma] is finite" (js/isFinite g1))))

;; ---------------------------------------------------------------------------
;; Test 7: Compiled Fisher — correctness and speedup
;; ---------------------------------------------------------------------------

(println "\n=== Test 7: Compiled Fisher ===")

(let [key (rng/fresh-key 42)
      params (mx/array [true-mu])
      ;; Uncompiled Fisher
      t0 (js/Date.now)
      {:keys [fisher log-ml]} (fisher/observed-fisher
                                 {:n-particles 5000 :key key}
                                 model-1 [] obs-1 [:mu] params)
      _ (mx/materialize! fisher)
      t-uncompiled (- (js/Date.now) t0)
      f-uncompiled (mx/item (mx/mat-get fisher 0 0))
      ;; Compiled Fisher
      t1 (js/Date.now)
      {:keys [fisher log-ml]} (fisher/observed-fisher
                                 {:n-particles 5000 :key key :compiled? true}
                                 model-1 [] obs-1 [:mu] params)
      _ (mx/materialize! fisher)
      t-compiled (- (js/Date.now) t1)
      f-compiled (mx/item (mx/mat-get fisher 0 0))
      speedup (/ t-uncompiled (max t-compiled 1))]
  (println (str "  Uncompiled Fisher: " (.toFixed f-uncompiled 3) " (" t-uncompiled "ms)"))
  (println (str "  Compiled Fisher:   " (.toFixed f-compiled 3) " (" t-compiled "ms)"))
  (println (str "  Speedup: " (.toFixed speedup 1) "x"))
  (assert-close "Compiled Fisher ≈ uncompiled" f-uncompiled f-compiled 0.1)
  ;; Note: first compiled call includes compilation overhead,
  ;; so speedup may be < 1x. The real speedup is on subsequent calls.
  (assert-true "Both ≈ analytical (K=10)" (< (js/Math.abs (- f-compiled 10.0)) 2.0)))

(println "\nDone.")
