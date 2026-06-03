;; Unit tests for genmlx.agents.helpers — the ergonomic PPL helpers used across
;; the agentmodels examples library (factor-dist, softmax-action, value-carrying
;; draws, boxed-choice). Self-contained; no test framework.
;;
;; Ground truth: factor-dist injects the exact scalar log-weight into the trace
;; :score path; softmax-action reproduces a hand-computed Boltzmann distribution.
;;
;; Run: bun run --bun nbb test/genmlx/agentmodels_helpers_test.cljs

(ns genmlx.agentmodels-helpers-test
  (:require [genmlx.agents.helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.gen :refer-macros [gen]]))

;; -----------------------------------------------------------------------------
;; Test infrastructure
;; -----------------------------------------------------------------------------

(def passed (volatile! 0))
(def failed (volatile! 0))

(defn assert-true [msg cond]
  (if cond
    (do (vswap! passed inc) (println " PASS" msg))
    (do (vswap! failed inc) (println " FAIL" msg))))

(defn assert-close [msg expected actual tol]
  (let [diff (Math/abs (- expected actual))]
    (if (<= diff tol)
      (do (vswap! passed inc) (println " PASS" msg "  =" actual))
      (do (vswap! failed inc)
          (println " FAIL" msg "  expected:" expected "  got:" actual "  diff:" diff)))))

(defn assert-equal [msg expected actual]
  (if (= expected actual)
    (do (vswap! passed inc) (println " PASS" msg "  =" actual))
    (do (vswap! failed inc)
        (println " FAIL" msg "  expected:" expected "  got:" actual))))

(def tol 1e-5)

;; -----------------------------------------------------------------------------
;; Section 1: factor-dist injects the exact scalar into :score
;; -----------------------------------------------------------------------------

(println "\n== Section 1: factor-dist score injection ==")

;; A factor-only model: simulate starts score at 0 and adds log-prob = w, so the
;; trace score must equal w exactly — proving the injection is exact, not merely
;; monotone.
(let [w     2.3
      model (dyn/auto-key (gen [] (let [_ (trace :soft (h/factor-dist (mx/scalar w)))] 0)))
      tr    (p/simulate model [])]
  (assert-close "single factor: score == w" w (mx/item (:score tr)) tol))

;; Two factors accumulate additively: score == w1 + w2.
(let [w1 1.5 w2 -0.75
      model (dyn/auto-key
              (gen []
                (let [_ (trace :f1 (h/factor-dist (mx/scalar w1)))
                      _ (trace :f2 (h/factor-dist (mx/scalar w2)))]
                  0)))
      tr (p/simulate model [])]
  (assert-close "two factors: score == w1 + w2" (+ w1 w2) (mx/item (:score tr)) tol))

;; The draw is deterministically 0 regardless of w (point support at 0).
(let [d (h/factor-dist (mx/scalar 9.9))]
  (assert-close "factor-dist draws 0" 0.0 (mx/item (dist/sample d)) tol)
  (assert-close "factor-dist log-prob == w for any value"
                9.9 (mx/item (dist/log-prob d (mx/scalar 123.0))) tol))

;; -----------------------------------------------------------------------------
;; Section 2: softmax-action == hand-computed Boltzmann
;; -----------------------------------------------------------------------------

(println "\n== Section 2: softmax-action Boltzmann policy ==")

;; eu = [1, 2, 0], alpha = 0.5 -> logits z = [0.5, 1.0, 0.0].
;; Boltzmann probs = softmax(z):
;;   exp = [e^0.5, e^1, e^0] = [1.6487212707, 2.7182818285, 1.0], sum = 5.3670030992
;;   p   = [0.30719739, 0.50647924, 0.18632337]
(let [eu      (mx/array #js [1.0 2.0 0.0])
      d       (h/softmax-action 0.5 eu)
      exps    [(Math/exp 0.5) (Math/exp 1.0) (Math/exp 0.0)]
      z       (reduce + exps)
      hand    (mapv #(/ % z) exps)]
  (doseq [i (range 3)]
    (let [p-i (Math/exp (mx/item (dist/log-prob d (mx/scalar i mx/int32))))]
      (assert-close (str "softmax-action P(action " i ") matches Boltzmann")
                    (nth hand i) p-i tol))))

;; alpha = ##Inf -> deterministic argmax over eu (index 1, eu = 2.0). The
;; resulting categorical places all mass on the argmax: P(1) == 1.
(let [eu (mx/array #js [1.0 2.0 0.0])
      d  (h/softmax-action ##Inf eu)]
  (assert-close "alpha=Inf: P(argmax index 1) == 1"
                1.0 (Math/exp (mx/item (dist/log-prob d (mx/scalar 1 mx/int32)))) tol))

;; -----------------------------------------------------------------------------
;; Section 3: value-carrying draws (uniform-draw / weighted-draw)
;; -----------------------------------------------------------------------------

(println "\n== Section 3: value-carrying draws ==")

(let [box (h/weighted-draw [:a :b :c] [1.0 2.0 1.0])]
  (assert-equal "weighted-draw stores values" [:a :b :c] (:values box))
  (assert-true  "weighted-draw :dist is a Distribution" (some? (:dist box)))
  ;; gather by MLX index scalar (as a categorical sample would yield)
  (assert-equal "draw-value gathers by MLX index" :b (h/draw-value box (mx/scalar 1 mx/int32)))
  ;; gather by plain integer
  (assert-equal "draw-value gathers by plain int" :c (h/draw-value box 2))
  ;; weights normalize correctly: P = [1/4, 2/4, 1/4]
  (doseq [[i pi] (map vector (range 3) [0.25 0.5 0.25])]
    (assert-close (str "weighted-draw P(" i ")") pi
                  (Math/exp (mx/item (dist/log-prob (:dist box) (mx/scalar i mx/int32)))) tol)))

(let [box (h/uniform-draw [:x :y :z :w])]
  (assert-equal "uniform-draw stores values" [:x :y :z :w] (:values box))
  (doseq [i (range 4)]
    (assert-close (str "uniform-draw P(" i ") == 1/4") 0.25
                  (Math/exp (mx/item (dist/log-prob (:dist box) (mx/scalar i mx/int32)))) tol)))

;; -----------------------------------------------------------------------------
;; Section 4: distribution-as-value via boxed-choice (SwitchCombinator)
;; -----------------------------------------------------------------------------

(println "\n== Section 4: boxed-choice sub-model selection ==")

(let [model-a (dyn/auto-key (gen [] (let [_ (trace :v (dist/delta (mx/scalar 10.0)))] (mx/scalar 10.0))))
      model-b (dyn/auto-key (gen [] (let [_ (trace :v (dist/delta (mx/scalar 20.0)))] (mx/scalar 20.0))))
      choose  (h/boxed-choice [model-a model-b])
      tr0     (p/simulate choose [0])
      tr1     (p/simulate choose [1])]
  (assert-close "boxed-choice kind 0 runs branch A" 10.0 (mx/item (:retval tr0)) tol)
  (assert-close "boxed-choice kind 1 runs branch B" 20.0 (mx/item (:retval tr1)) tol))

;; -----------------------------------------------------------------------------
;; Summary
;; -----------------------------------------------------------------------------

(println "\n========================================")
(println "  PASSED:" @passed "   FAILED:" @failed)
(println "========================================")
(when (pos? @failed)
  (js/process.exit 1))
