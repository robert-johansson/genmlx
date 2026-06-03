;; Headless tests for inverse inference over agent BIAS (agentmodels Ch 5d/5e,
;; capability-only slice — bean genmlx-z4si).
;; Run: bun run --bun nbb test/genmlx/agentmodels_biased_inverse_test.cljs

(ns genmlx.agentmodels-biased-inverse-test
  (:require [agentmodels.biased-inverse :as bi]
            [genmlx.agents.inverse :as inv]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]))

(def passed (volatile! 0))
(def failed (volatile! 0))
(defn assert-true [msg c]
  (if c (do (vswap! passed inc) (println " PASS" msg))
        (do (vswap! failed inc) (println " FAIL" msg))))
(defn assert-close [msg expected actual tol]
  (if (<= (Math/abs (- expected actual)) tol)
    (do (vswap! passed inc) (println " PASS" msg "  =" actual))
    (do (vswap! failed inc) (println " FAIL" msg "  expected:" expected "  got:" actual))))

(def mdp (bi/temptation-mdp))
(def THIRD (/ 1.0 3.0))      ; 8/3 ÷ 8 → δ(2); the veg value 8·δ(2)=8/3
(def EU-VEG (/ 8.0 3.0))     ; 2.6667

;; cfg helpers (α=##Inf unless overridden)
(defn cfg-inf [states actions]
  {:mdp mdp :alpha ##Inf :discount 1.0 :n-iters 10 :states states :actions actions})
;; action 0 = head toward temptation gate ; action 1 = take the safe route

;; ===========================================================================
(println "\n== Section 1: forward agents — EU rows & policy at the start ==")
(def agents (bi/bias-agents {:mdp mdp :alpha ##Inf :discount 1.0 :n-iters 10}))
(def naive-ag (:naive agents))
(def soph-ag  (:sophisticated agents))
(def eu-n (bi/eu-row naive-ag 0))
(def eu-s (bi/eu-row soph-ag 0))
(println "  naive EU(start) =" eu-n "   soph EU(start) =" eu-s)
(assert-close "naive EU(a0)=8/3 (believes future self resists)" EU-VEG (nth eu-n 0) 1e-6)
(assert-close "naive EU(a1)=8/3"                                 EU-VEG (nth eu-n 1) 1e-6)
(assert-close "soph EU(a0)=2.5 (foresees succumbing to donut)"   2.5    (nth eu-s 0) 1e-6)
(assert-close "soph EU(a1)=8/3"                                  EU-VEG (nth eu-s 1) 1e-6)

(defn pi-prob [agent s a] (Math/exp (inv/action-loglik agent s a)))
(assert-close "π_naive(a0)=0.5 (exact tie, load-bearing)" 0.5 (pi-prob naive-ag 0 0) 1e-6)
(assert-close "π_naive(a1)=0.5"                           0.5 (pi-prob naive-ag 0 1) 1e-6)
(assert-close "π_soph(a0)=0 (NEVER heads for temptation)" 0.0 (pi-prob soph-ag 0 0) 1e-9)
(assert-close "π_soph(a1)=1"                              1.0 (pi-prob soph-ag 0 1) 1e-6)

;; ===========================================================================
(println "\n== Section 2: prior decomposition (assess on {:bias} alone) ==")
(def model-prior (bi/biased-agent-model {:mdp mdp :alpha ##Inf :discount 1.0
                                         :n-iters 10 :states []}))
(assert-close "log-prior P0(:naive)=log 1/2"
              (Math/log 0.5)
              (mx/item (:weight (p/assess (dyn/auto-key model-prior) []
                                          (cm/choicemap :bias 0))))
              1e-6)
(assert-close "log-prior P0(:sophisticated)=log 1/2"
              (Math/log 0.5)
              (mx/item (:weight (p/assess (dyn/auto-key model-prior) []
                                          (cm/choicemap :bias 1))))
              1e-6)

;; ===========================================================================
(println "\n== Section 3: exact posterior — the three closed-form cases ==")
(def pA (bi/bias-posterior (cfg-inf [0]     [1])))        ; safe ×1
(def pB (bi/bias-posterior (cfg-inf [0]     [0])))        ; tempt ×1
(def pC (bi/bias-posterior (cfg-inf [0 0 0] [1 1 1])))    ; safe ×3
(println "  P(safe×1) =" pA)
(println "  P(tempt×1)=" pB)
(println "  P(safe×3) =" pC)
(assert-close "Case A  P(:naive | safe×1) = 1/3"   THIRD          (:naive pA)         1e-6)
(assert-close "Case A  P(:soph  | safe×1) = 2/3"   (/ 2.0 3.0)    (:sophisticated pA) 1e-6)
(assert-close "Case B  P(:naive | tempt×1) = 1.0"  1.0            (:naive pB)         1e-9)
(assert-close "Case B  P(:soph  | tempt×1) = 0.0"  0.0            (:sophisticated pB) 1e-9)
(assert-close "Case C  P(:naive | safe×3) = 1/9"   (/ 1.0 9.0)    (:naive pC)         1e-6)
(assert-close "Case C  P(:soph  | safe×3) = 8/9"   (/ 8.0 9.0)    (:sophisticated pC) 1e-6)

;; ===========================================================================
(println "\n== Section 4: identifiability invariants ==")
;; The likelihood ratio per "safe" observation is naive:soph = π_naive(a1):π_soph(a1)
;; = 0.5:1.0 = 1:2 — i.e. each safe action is twice as likely under sophisticated.
;; That ratio, driven by the perceivedDelay mechanism (naive is indifferent at the
;; start, soph is decisive), is what produces the closed-form posteriors below.
(assert-true  "heading for temptation ⇒ MAP bias is :naive"   (> (:naive pB) 0.5))
(assert-true  "repeated safe route   ⇒ MAP bias is :sophisticated" (> (:sophisticated pC) 0.5))
(assert-true  "evidence accumulates: P(:soph|×3) > P(:soph|×1)" (> (:sophisticated pC) (:sophisticated pA)))
(def pC2 (bi/bias-posterior (cfg-inf [0 0] [1 1])))   ; safe ×2
(assert-true  "monotone increase across n: P(:soph|×1) < P(:soph|×2) < P(:soph|×3)"
              (< (:sophisticated pA) (:sophisticated pC2) (:sophisticated pC)))
(assert-close "monotone law P(:soph | n safe) = 1/(1+(1/2)^n) at n=2"
              (/ 1.0 (+ 1.0 (Math/pow 0.5 2)))
              (:sophisticated pC2) 1e-6)
(assert-close "monotone law P(:soph | n safe) = 1/(1+(1/2)^n) at n=3"
              (/ 1.0 (+ 1.0 (Math/pow 0.5 3)))
              (:sophisticated pC) 1e-6)

;; ===========================================================================
(println "\n== Section 5: normalization ==")
(doseq [[lbl pp] [["A" pA] ["B" pB] ["C" pC]]]
  (assert-true (str "posterior " lbl " sums to 1")
               (< (Math/abs (- 1.0 (+ (:naive pp) (:sophisticated pp)))) 1e-9)))

;; ===========================================================================
(println "\n== Section 6: consistency — joint GF == inverse.cljs idiom ==")
(def pA' (bi/bias-posterior-via-policy (cfg-inf [0]     [1])))
(def pC' (bi/bias-posterior-via-policy (cfg-inf [0 0 0] [1 1 1])))
(assert-close "joint-GF assess == policy-idiom posterior (safe×1)"
              (:sophisticated pA) (:sophisticated pA') 1e-9)
(assert-close "joint-GF assess == policy-idiom posterior (safe×3)"
              (:sophisticated pC) (:sophisticated pC') 1e-9)

;; ===========================================================================
(println "\n== Section 7: edge cases ==")
(def pEmpty (bi/bias-posterior (cfg-inf [] [])))
(assert-close "empty obs ⇒ posterior == prior (:naive 0.5)" 0.5 (:naive pEmpty)         1e-9)
(assert-close "empty obs ⇒ posterior == prior (:soph 0.5)"  0.5 (:sophisticated pEmpty) 1e-9)
;; asymmetric prior [0.3 0.7] on safe×1: P(:soph)=0.7·1/(0.7·1+0.3·0.5)=0.7/0.85
(def pAsym (bi/bias-posterior {:mdp mdp :alpha ##Inf :discount 1.0 :n-iters 10
                               :states [0] :actions [1] :prior [0.3 0.7]}))
(assert-close "asymmetric prior [0.3 0.7], safe×1 ⇒ P(:soph)=0.8235"
              (/ 0.7 0.85) (:sophisticated pAsym) 1e-6)
;; misaligned :states/:actions must be rejected, not silently mis-scored
(assert-true "mismatched :states/:actions lengths throw"
             (try (bi/bias-posterior (cfg-inf [0 0] [1])) false
                  (catch :default _ true)))

;; ===========================================================================
(println "\n== Section 8: exact ≈ importance-sampling (finite α=2) ==")
(defn cfg-2 [states actions]
  {:mdp mdp :alpha 2.0 :discount 1.0 :n-iters 10 :states states :actions actions})
(def N 5000)
(def exact2-safe  (bi/bias-posterior  (cfg-2 [0 0 0] [1 1 1])))
(def exact2-tempt (bi/bias-posterior  (cfg-2 [0 0 0] [0 0 0])))
(def is2-safe     (bi/is-bias-posterior (cfg-2 [0 0 0] [1 1 1]) N (rng/fresh-key 42)))
(def is2-tempt    (bi/is-bias-posterior (cfg-2 [0 0 0] [0 0 0]) N (rng/fresh-key 7)))
(println "  exact P(:soph|safe×3) @α=2 =" (:sophisticated exact2-safe)
         "   IS =" (:sophisticated (:posterior is2-safe)) "   ESS =" (:ess is2-safe))
(println "  exact P(:naive|tempt×3) @α=2 =" (:naive exact2-tempt)
         "   IS =" (:naive (:posterior is2-tempt)) "   ESS =" (:ess is2-tempt))
(assert-true  "finite-α: exact P(:soph|safe×3) > 0.5 (still identifiable, softer)"
              (> (:sophisticated exact2-safe) 0.5))
(assert-true  "finite-α: exact P(:naive|tempt×3) > 0.5"
              (> (:naive exact2-tempt) 0.5))
;; the exact α=2 posterior is deterministic (assess only scores), so the fidelity
;; check against the math-verifier's reference value is tight (1e-3 absorbs only the
;; 4-sig-fig rounding of the reference constant).
(assert-close "world fidelity: exact P(:soph|safe×3) @α=2 ≈ 0.5515"
              0.5515 (:sophisticated exact2-safe) 1e-3)
(assert-close "world fidelity: exact P(:naive|tempt×3) @α=2 ≈ 0.5638"
              0.5638 (:naive exact2-tempt) 1e-3)
(assert-close "exact ≈ IS  P(:soph|safe×3) @α=2 (N=5000)"
              (:sophisticated exact2-safe) (:sophisticated (:posterior is2-safe)) 0.05)
(assert-close "exact ≈ IS  P(:naive|tempt×3) @α=2 (N=5000)"
              (:naive exact2-tempt) (:naive (:posterior is2-tempt)) 0.05)
;; ESS here is a FIXED value (seeds 42/7 are hardcoded and MLX RNG is bit-reproducible),
;; not a probabilistic claim. The ~4× margin over N/4 reflects soft α=2 likelihoods.
(assert-true  "IS ESS > N/4 (safe×3)"  (> (:ess is2-safe)  (/ N 4)))
(assert-true  "IS ESS > N/4 (tempt×3)" (> (:ess is2-tempt) (/ N 4)))

;; ===========================================================================
(println "\n== Section 9: agent/cache reuse across calls on one model instance ==")
;; One model instance carries precomputed agents + memoized EU caches; scoring it
;; for both bias hypotheses (and against bias-posterior, which builds its own model)
;; must agree — exercising the 'built once and reused' claim.
(def shared-model (bi/biased-agent-model (cfg-inf [0] [1])))
(defn- assess-w [idx a]
  (mx/item (:weight (p/assess (dyn/auto-key shared-model) [] (bi/full-cm idx [a])))))
(let [wn (assess-w 0 1) ws (assess-w 1 1)              ; naive(safe), soph(safe)
      z  (+ (Math/exp wn) (Math/exp ws))]
  (assert-true  "reused model scores naive(safe) finite" (js/isFinite wn))
  (assert-true  "reused model scores soph(safe)  finite" (js/isFinite ws))
  (assert-close "reused-model posterior P(:soph|safe×1) = 2/3 (matches bias-posterior)"
                (/ 2.0 3.0) (/ (Math/exp ws) z) 1e-6))

;; ===========================================================================
(println (str "\n" @passed " passed, " @failed " failed"))
(when (pos? @failed) (js/process.exit 1))
