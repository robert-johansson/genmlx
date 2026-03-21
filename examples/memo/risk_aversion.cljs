(ns risk-aversion
  "Risk aversion and value-at-risk, after Terry Tao's Mastodon example.

   An agent (Terry) chooses between two actions:
     Safe: payout = 5 + 3*o     (mean 5, std 3)
     Bold: payout = 9 + 10*o    (mean 9, std 10)

   Outcomes o are drawn from a discretized N(0,1).

   Terry minimizes VALUE-AT-RISK: sqrt(Var[u]) - E[u].
   Safe has lower variance, so Terry prefers Safe despite Bold's higher mean.

   Adding an external shock (mean -5, std 10) that affects both actions
   equally increases the baseline variance. Now Bold's higher mean outweighs
   its marginal extra variance, and Terry switches to Bold.

   Reproduces the memo demo-risk-aversion.ipynb results."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.inference.exact :as exact])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Discretized N(0,1) outcome space: 101 points from -10 to 10
;; ---------------------------------------------------------------------------

(def n-outcomes 101)
(def outcome-vals (mx/linspace -10 10 n-outcomes))

;; Normal pdf weights (unnormalized): exp(-x^2/2)
;; Use as logits for categorical: -x^2/2
(def norm-logits
  (mx/negative (mx/divide (mx/multiply outcome-vals outcome-vals)
                          (mx/scalar 2.0))))

;; ---------------------------------------------------------------------------
;; Utility: means[a] + stdvs[a] * o
;; ---------------------------------------------------------------------------

(def means #js [5 9])     ;; safe=5, bold=9
(def stdvs #js [3 10])    ;; safe=3, bold=10

(defn utility-1d
  "Payout for action a-idx and outcome array o.
   Returns array same shape as o."
  [a-idx o]
  (let [mu (mx/scalar (aget means a-idx))
        sd (mx/scalar (aget stdvs a-idx))]
    (mx/add mu (mx/multiply sd o))))

;; ---------------------------------------------------------------------------
;; Helper: compute value-at-risk from exact probability table over outcomes
;; ---------------------------------------------------------------------------

(defn value-at-risk
  "Compute sqrt(Var[u]) - E[u] given probability vector p and utility vector u."
  [p u]
  (let [eu  (mx/sum (mx/multiply p u))
        eu2 (mx/sum (mx/multiply p (mx/multiply u u)))
        var (mx/subtract eu2 (mx/multiply eu eu))]
    (mx/subtract (mx/sqrt var) eu)))

;; ---------------------------------------------------------------------------
;; Scenario 1: No external shock
;; ---------------------------------------------------------------------------

(println "Risk Aversion (Terry Tao)")
(println "=========================\n")
(println "Actions: Safe (mean=5, std=3), Bold (mean=9, std=10)")
(println "Terry minimizes value-at-risk: sqrt(Var[u]) - E[u]\n")

(println "--- Scenario 1: No external shock ---\n")

(let [;; World model: an outcome index drawn from discretized N(0,1)
      world-model (gen []
                    (trace :o (dist/categorical norm-logits)))

      ;; Terry decides: enumerate over actions, compute VR for each
      terry (gen []
              (let [;; Exact probability table over outcomes
                    probs (splice :world (exact/thinks world-model))

                    ;; Utility values for each outcome index
                    u-safe (utility-1d 0 outcome-vals)
                    u-bold (utility-1d 1 outcome-vals)

                    ;; Value-at-risk for each action
                    vr-safe (value-at-risk probs u-safe)
                    vr-bold (value-at-risk probs u-bold)

                    ;; Terry minimizes VR = maximizes -VR
                    a (trace :a (exact/categorical-argmax
                                  (mx/negative (mx/stack [vr-safe vr-bold]))))]
                a))

      r (exact/exact-posterior terry [] nil)
      p-safe (get-in r [:marginals :a 0])
      p-bold (get-in r [:marginals :a 1])]
  (println (str "  P(Safe) = " (.toFixed p-safe 4)))
  (println (str "  P(Bold) = " (.toFixed p-bold 4)))
  (println "  Terry chooses Safe (lower variance outweighs lower mean).\n"))

;; ---------------------------------------------------------------------------
;; Scenario 2: External shock (mean -5, std 10) affects both actions
;; ---------------------------------------------------------------------------

(println "--- Scenario 2: External shock (mean=-5, std=10) ---\n")

(defn utility-with-shock
  "Payout for action a-idx, outcome o, shock s:
   means[a] + stdvs[a]*o + (-5 + 10*s)"
  [a-idx o s]
  (mx/add (utility-1d a-idx o)
          (mx/add (mx/scalar -5) (mx/multiply (mx/scalar 10) s))))

(let [;; World model: both outcome and shock drawn from N(0,1)
      world-model (gen []
                    (let [o (trace :o (dist/categorical norm-logits))
                          s (trace :s (dist/categorical norm-logits))]
                      [o s]))

      ;; Terry decides with shock
      terry (gen []
              (let [;; Exact joint over (o, s) — probs has shape [101, 101]
                    probs (splice :world (exact/thinks world-model))

                    ;; Utility grids: broadcast outcome-vals [101,1] x [1,101]
                    o-grid (mx/reshape outcome-vals [n-outcomes 1])
                    s-grid (mx/reshape outcome-vals [1 n-outcomes])

                    u-safe-grid (utility-with-shock 0 o-grid s-grid)
                    u-bold-grid (utility-with-shock 1 o-grid s-grid)

                    ;; Value-at-risk over the full joint
                    vr-safe (value-at-risk probs u-safe-grid)
                    vr-bold (value-at-risk probs u-bold-grid)

                    a (trace :a (exact/categorical-argmax
                                  (mx/negative (mx/stack [vr-safe vr-bold]))))]
                a))

      r (exact/exact-posterior terry [] nil)
      p-safe (get-in r [:marginals :a 0])
      p-bold (get-in r [:marginals :a 1])]
  (println (str "  P(Safe) = " (.toFixed p-safe 4)))
  (println (str "  P(Bold) = " (.toFixed p-bold 4)))
  (println "  Terry switches to Bold! External shock raises baseline variance,")
  (println "  so Bold's higher mean now outweighs its marginal extra risk.\n"))

;; ---------------------------------------------------------------------------
;; Verification
;; ---------------------------------------------------------------------------

(println "Verification:")

(let [pass? (atom true)
      check (fn [name expected actual]
              (let [ok (< (abs (- expected actual)) 1e-3)]
                (when-not ok (reset! pass? false))
                (println (str "  " (if ok "PASS" "FAIL") ": " name
                              " (expected " expected ", got " (.toFixed actual 4) ")"))))

      ;; Scenario 1: no shock
      world-1 (gen [] (trace :o (dist/categorical norm-logits)))
      terry-1 (gen []
                (let [p (splice :w (exact/thinks world-1))
                      u-s (utility-1d 0 outcome-vals)
                      u-b (utility-1d 1 outcome-vals)
                      a (trace :a (exact/categorical-argmax
                                    (mx/negative (mx/stack [(value-at-risk p u-s)
                                                            (value-at-risk p u-b)]))))]
                  a))
      r1 (exact/exact-posterior terry-1 [] nil)

      ;; Scenario 2: with shock
      world-2 (gen []
                (let [o (trace :o (dist/categorical norm-logits))
                      s (trace :s (dist/categorical norm-logits))]
                  [o s]))
      terry-2 (gen []
                (let [p (splice :w (exact/thinks world-2))
                      og (mx/reshape outcome-vals [n-outcomes 1])
                      sg (mx/reshape outcome-vals [1 n-outcomes])
                      us (utility-with-shock 0 og sg)
                      ub (utility-with-shock 1 og sg)
                      a (trace :a (exact/categorical-argmax
                                    (mx/negative (mx/stack [(value-at-risk p us)
                                                            (value-at-risk p ub)]))))]
                  a))
      r2 (exact/exact-posterior terry-2 [] nil)]

  (check "no-shock P(Safe)"  1.0 (get-in r1 [:marginals :a 0]))
  (check "no-shock P(Bold)"  0.0 (get-in r1 [:marginals :a 1]))
  (check "shock P(Safe)"     0.0 (get-in r2 [:marginals :a 0]))
  (check "shock P(Bold)"     1.0 (get-in r2 [:marginals :a 1]))

  (println)
  (if @pass?
    (println "All checks passed.")
    (do (println "Some checks FAILED.")
        (js/process.exit 1))))
