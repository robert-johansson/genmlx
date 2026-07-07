;; @tier fast
(ns genmlx.support-guard-test
  "genmlx-7oen: out-of-support log-probs must be -Inf, never finite garbage
   or NaN.

   Audit genmlx-ansg (VERIFIED) found nine families missing the support-
   membership guard that uniform/exponential/discrete-uniform/truncated-normal
   already carry: geometric scored an impossible value HIGHER than a legal
   one; bernoulli/poisson/binomial/neg-binomial returned finite garbage on
   non-integer/negative v; gamma/beta/log-normal/inv-gamma returned NaN
   (log of a negative) — silently poisoning IS/SMC weights and MH accept
   comparisons. This suite pins: out-of-support -> exactly -Inf (finite
   check + NaN check), in-support values unchanged/finite, and the specific
   audit repros (geometric ordering, bernoulli lp(0.5)).

   Run: bunx --bun nbb@1.4.208 test/genmlx/support_guard_test.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))

(defn assert-true [label v]
  (if v
    (do (swap! pass inc) (println "  PASS" label))
    (do (swap! fail inc) (println "  FAIL" label))))

(def NEG-INF js/Number.NEGATIVE_INFINITY)

(defn- lp [d v] (mx/realize (dc/dist-log-prob d (mx/scalar v))))

(defn- check-dist [label d bad-vs good-vs]
  (doseq [v bad-vs]
    (let [x (lp d v)]
      (assert-true (str label ": lp(" v ") = -Inf (got " x ")")
                   (= x NEG-INF))))
  (doseq [v good-vs]
    (let [x (lp d v)]
      (assert-true (str label ": lp(" v ") finite (got " x ")")
                   (js/isFinite x)))))

;; ===========================================================================
(println "\n-- discrete families: integer + range masks --")

(check-dist "geometric(0.3)" (dist/geometric 0.3)
            [-1 -0.5 0.5 1.7] [0 1 2 10])

;; the audit's headline: an impossible value must NOT outscore a legal one
(let [d (dist/geometric 0.3)]
  (assert-true "geometric: lp(-1) no longer beats lp(0)"
               (< (lp d -1) (lp d 0))))

(check-dist "bernoulli(0.4)" (dist/bernoulli 0.4)
            [0.5 -1 2 0.999] [0 1])

(let [d (dist/bernoulli 0.4)]
  (assert-true "bernoulli: lp(1) = log(p) exactly"
               (< (js/Math.abs (- (lp d 1) (js/Math.log 0.4))) 1e-6))
  (assert-true "bernoulli: lp(0) = log(1-p) exactly"
               (< (js/Math.abs (- (lp d 0) (js/Math.log 0.6))) 1e-6)))

(check-dist "poisson(3)" (dist/poisson 3.0)
            [-1 -0.5 1.5 2.0000305] [0 1 5 20])

(check-dist "binomial(10, 0.4)" (dist/binomial 10 0.4)
            [-1 0.5 3.5 11 15] [0 3 10])

(check-dist "neg-binomial(5, 0.4)" (dist/neg-binomial 5 0.4)
            [-1 -0.5 2.5] [0 2 12])

;; ===========================================================================
(println "\n-- continuous families: positivity / interval masks (NaN class) --")

(check-dist "gamma(2, 1.5)" (dist/gamma-dist 2.0 1.5)
            [-1 -0.001 0] [0.1 1.0 7.0])

(check-dist "inv-gamma(2, 1.5)" (dist/inv-gamma 2.0 1.5)
            [-1 -0.001 0] [0.1 1.0 7.0])

(check-dist "log-normal(0, 1)" (dist/log-normal 0.0 1.0)
            [-1 -0.001 0] [0.1 1.0 7.0])

(check-dist "beta(2, 3)" (dist/beta-dist 2.0 3.0)
            [-0.1 1.1 2 -1] [0.2 0.5 0.8])

;; the NaN class specifically: no bounded-support dist may return NaN
(println "\n-- no NaN from any bounded-support family, in or out of domain --")
(let [cases [["gamma" (dist/gamma-dist 2.0 1.5) [-3 -1 0 0.5 2]]
             ["inv-gamma" (dist/inv-gamma 2.0 1.5) [-3 -1 0 0.5 2]]
             ["log-normal" (dist/log-normal 0.0 1.0) [-3 -1 0 0.5 2]]
             ["beta" (dist/beta-dist 2.0 3.0) [-1 -0.1 0.5 1.1 2]]
             ["geometric" (dist/geometric 0.3) [-2 -0.5 0.5 3]]
             ["poisson" (dist/poisson 3.0) [-2 -0.5 1.5 4]]
             ["binomial" (dist/binomial 10 0.4) [-2 0.5 11 4]]
             ["neg-binomial" (dist/neg-binomial 5 0.4) [-2 0.5 3]]
             ["bernoulli" (dist/bernoulli 0.4) [-1 0.5 2 1]]]]
  (doseq [[nm d vs] cases]
    (assert-true (str nm ": no NaN over " (pr-str vs))
                 (not-any? #(js/isNaN (lp d %)) vs))))

;; ===========================================================================
(println (str "\n== support-guard: " @pass " passed, " @fail " failed =="))
(when (pos? @fail) (set! (.-exitCode js/process) 1))
