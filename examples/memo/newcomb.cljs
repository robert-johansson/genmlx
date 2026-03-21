(ns newcomb
  "Newcomb's Problem — two decision-theoretic formulations.

   Setup: A perfect predictor (God) has placed money in two boxes:
     Box A: always contains $1,000
     Box B: contains $1,000,000 IF God predicted you'd take only B

   Payout matrix [2 actions, 2 god-states]:
     Take both (A+B): [$1,001,000 if God wrong, $1,000 if God right]
     Take B only:     [$1,000,000 if God right, $0 if God wrong]

   Two formulations via exact enumeration:

   1. FEARFUL: God's prediction DEPENDS on your choice (causal link).
      God mirrors your action. Taking B only → God predicts B → $1M.
      Result: take B only (expected utility dominates).

   2. REALIST: God's prediction is INDEPENDENT (coin flip, no causal link).
      Regardless of God's prediction, A+B always pays $1,000 more.
      Result: take both (dominance reasoning)."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.inference.exact :as exact])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Payout matrix
;; ---------------------------------------------------------------------------

;; Rows: y=0 (take A+B), y=1 (take B only)
;; Cols: god=0 (predicts A+B), god=1 (predicts B only)
(def payout (mx/array #js [#js [1000 0] #js [1001000 1000000]]))

;; ---------------------------------------------------------------------------
;; Fearful formulation: God mirrors your choice
;; ---------------------------------------------------------------------------

(println "Newcomb's Problem")
(println "=================")
(println)
(println "Payout matrix:")
(println "                  God predicts A+B   God predicts B")
(println "  Take A+B:       $1,000             $1,001,000")
(println "  Take B only:    $0                 $1,000,000")
(println)

(println "--- Fearful: God's prediction depends on your choice ---")

(let [;; God model: predicts action y deterministically (mirrors y)
      god-model (fn [y-val]
                  (gen []
                    (trace :g (dist/categorical
                      (mx/log (mx/eq? (mx/array #js [0 1]) y-val))))))
      ;; Fearful agent: compute EU for each action using causal god
      fearful (gen []
                (let [god-y0 (splice :god0 (exact/Exact (god-model 0)))
                      god-y1 (splice :god1 (exact/Exact (god-model 1)))
                      eu-y0 (mx/sum (mx/multiply god-y0
                              (mx/idx payout 0 -1)))
                      eu-y1 (mx/sum (mx/multiply god-y1
                              (mx/idx payout 1 -1)))
                      y (trace :y (exact/categorical-argmax (mx/stack [eu-y0 eu-y1])))]
                  y))
      r (exact/exact-posterior fearful [] nil)
      p-ab (get-in r [:marginals :y 0])
      p-b  (get-in r [:marginals :y 1])]
  (println (str "  P(take A+B) = " (.toFixed p-ab 4)))
  (println (str "  P(take B)   = " (.toFixed p-b 4)))
  (println "  Decision: Take B only (God mirrors you, so B → $1M)")
  (println))

;; ---------------------------------------------------------------------------
;; Realist formulation: God is independent (coin flip)
;; ---------------------------------------------------------------------------

(println "--- Realist: God's prediction is independent (50/50) ---")

(let [realist (gen []
                (let [god-probs (splice :god (exact/Exact
                                  (gen [] (trace :g (dist/bernoulli 0.5)))))
                      eu-y0 (mx/sum (mx/multiply god-probs
                              (mx/idx payout 0 -1)))
                      eu-y1 (mx/sum (mx/multiply god-probs
                              (mx/idx payout 1 -1)))
                      y (trace :y (exact/categorical-argmax (mx/stack [eu-y0 eu-y1])))]
                  y))
      r (exact/exact-posterior realist [] nil)
      p-ab (get-in r [:marginals :y 0])
      p-b  (get-in r [:marginals :y 1])]
  (println (str "  P(take A+B) = " (.toFixed p-ab 4)))
  (println (str "  P(take B)   = " (.toFixed p-b 4)))
  (println "  Decision: Take both (A+B always pays $1,000 more)")
  (println))

;; ---------------------------------------------------------------------------
;; Verification
;; ---------------------------------------------------------------------------

(println "Verification:")

(let [god-model (fn [y-val]
                  (gen []
                    (trace :g (dist/categorical
                      (mx/log (mx/eq? (mx/array #js [0 1]) y-val))))))
      fearful (gen []
                (let [god-y0 (splice :god0 (exact/Exact (god-model 0)))
                      god-y1 (splice :god1 (exact/Exact (god-model 1)))
                      eu-y0 (mx/sum (mx/multiply god-y0
                              (mx/idx payout 0 -1)))
                      eu-y1 (mx/sum (mx/multiply god-y1
                              (mx/idx payout 1 -1)))
                      y (trace :y (exact/categorical-argmax (mx/stack [eu-y0 eu-y1])))]
                  y))
      rf (exact/exact-posterior fearful [] nil)

      realist (gen []
                (let [god-probs (splice :god (exact/Exact
                                  (gen [] (trace :g (dist/bernoulli 0.5)))))
                      eu-y0 (mx/sum (mx/multiply god-probs
                              (mx/idx payout 0 -1)))
                      eu-y1 (mx/sum (mx/multiply god-probs
                              (mx/idx payout 1 -1)))
                      y (trace :y (exact/categorical-argmax (mx/stack [eu-y0 eu-y1])))]
                  y))
      rr (exact/exact-posterior realist [] nil)
      pass? (atom true)
      check (fn [name expected actual]
              (let [ok (< (abs (- expected actual)) 1e-5)]
                (when-not ok (reset! pass? false))
                (println (str "  " (if ok "PASS" "FAIL") ": " name
                              " (expected " expected ", got " (.toFixed actual 4) ")"))))]
  (check "fearful P(A+B)" 0.0 (get-in rf [:marginals :y 0]))
  (check "fearful P(B)"   1.0 (get-in rf [:marginals :y 1]))
  (check "realist P(A+B)" 1.0 (get-in rr [:marginals :y 0]))
  (check "realist P(B)"   0.0 (get-in rr [:marginals :y 1]))
  (println)
  (if @pass?
    (println "All checks passed.")
    (do (println "Some checks FAILED.")
        (js/process.exit 1))))
