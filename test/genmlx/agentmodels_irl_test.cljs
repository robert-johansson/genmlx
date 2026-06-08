;; @tier slow
;; Headless tests for agentmodels Ch 4 — Inverse Reinforcement Learning.
;; Part A: MDP IRL (utility table + timeCost + alpha) — agentmodels Equation 1.
;; Part B: POMDP-IRL (factorSequence belief threading) — agentmodels Equation 2.
;; Run: bunx nbb@1.4.206 test/genmlx/agentmodels_irl_test.cljs

(ns genmlx.agentmodels-irl-test
  (:require [agentmodels.irl :as irl]))

(def passed (volatile! 0))
(def failed (volatile! 0))
(defn assert-true [msg c]
  (if c (do (vswap! passed inc) (println " PASS" msg))
        (do (vswap! failed inc) (println " FAIL" msg))))
(defn assert-close [msg expected actual tol]
  (if (<= (Math/abs (- expected actual)) tol) (do (vswap! passed inc) (println " PASS" msg "  =" actual))
      (do (vswap! failed inc) (println " FAIL" msg "  expected:" expected "  got:" actual))))

(def food irl/food-vals)

;; ===========================================================================
(println "\n== Phase A.1: utility-table inference (Eq 1) — donut favoured, Veg/Noodle unidentifiable ==")
(let [uspec   (irl/utility-only-spec)       ; alpha=2, timeCost=-0.04 fixed (chapter's first example)
      uagents (irl/build-agents uspec)
      prior   (irl/prior-summary uagents)
      post    (irl/summarize (irl/joint-posterior uagents irl/single-left-obs) uagents)]
  (println "  prior  P(donut/veg/noodle Fav)=" (.toFixed (:p-donut-favorite prior) 3)
           "/" (.toFixed (:p-veg-favorite prior) 3) "/" (.toFixed (:p-noodle-favorite prior) 3))
  (println "  1-left P(donut/veg/noodle Fav)=" (.toFixed (:p-donut-favorite post) 3)
           "/" (.toFixed (:p-veg-favorite post) 3) "/" (.toFixed (:p-noodle-favorite post) 3))
  ;; prior is symmetric over the three restaurants
  (assert-close "prior P(donutFav) = P(vegFav)"   (:p-donut-favorite prior) (:p-veg-favorite prior) 1e-9)
  (assert-close "prior P(vegFav)   = P(noodleFav)" (:p-veg-favorite prior)  (:p-noodle-favorite prior) 1e-9)
  ;; one leftward step (toward Donut South) → Donut favoured
  (assert-true "1 leftward step: P(donutFav) rises above prior"
               (> (:p-donut-favorite post) (:p-donut-favorite prior)))
  (assert-true "1 leftward step: Donut is the favoured restaurant"
               (and (> (:p-donut-favorite post) (:p-veg-favorite post))
                    (> (:p-donut-favorite post) (:p-noodle-favorite post))))
  ;; agentmodels' unidentifiability: the step gives no evidence distinguishing Veg from Noodle
  (assert-true "Veg vs Noodle UNIDENTIFIABLE (|P(vegFav) − P(noodleFav)| < 0.05)"
               (< (Math/abs (- (:p-veg-favorite post) (:p-noodle-favorite post))) 0.05)))

;; ===========================================================================
(println "\n== Phase A.2: soft rationality — low alpha (noise) washes out the evidence ==")
;; agentmodels: 'if the agent has a low value for alpha, this step to the left is
;; fairly likely even if the agent prefers Noodle or Veg' — a controlled low-vs-high
;; alpha comparison on the SAME single step.
(let [base   {:donut-vals food :veg-vals food :noodle-vals food :time-cost-vals [-0.04]}
      lo-ag  (irl/build-agents (assoc base :alpha-vals [0.1]))
      hi-ag  (irl/build-agents (assoc base :alpha-vals [100.0]))
      prior  (:p-donut-favorite (irl/prior-summary lo-ag))
      lo     (:p-donut-favorite (irl/summarize (irl/joint-posterior lo-ag irl/single-left-obs) lo-ag))
      hi     (:p-donut-favorite (irl/summarize (irl/joint-posterior hi-ag irl/single-left-obs) hi-ag))]
  (println "  P(donutFav | 1-left): prior" (.toFixed prior 3) "  low-α(0.1)" (.toFixed lo 3) "  high-α(100)" (.toFixed hi 3))
  (assert-true "low-α agent: a single step is ~uninformative (P(donutFav) ≈ prior)"
               (< (Math/abs (- lo prior)) 0.03))
  (assert-true "high-α agent: the same step IS diagnostic (P(donutFav) ≫ low-α)"
               (> hi (+ lo 0.1))))

;; ===========================================================================
(println "\n== Phase A.3: more evidence sharpens the posterior (joint utility+timeCost+alpha) ==")
(let [jspec   (irl/joint-spec)
      jagents (irl/build-agents jspec)
      p1 (irl/summarize (irl/joint-posterior jagents irl/single-left-obs) jagents)
      p4 (irl/summarize (irl/joint-posterior jagents irl/donut-south-obs) jagents)]
  (println "  P(donutFav): 1-left" (.toFixed (:p-donut-favorite p1) 3) "→ donut-south(4)" (.toFixed (:p-donut-favorite p4) 3)
           "   P(noodleFav):" (.toFixed (:p-noodle-favorite p1) 3) "→" (.toFixed (:p-noodle-favorite p4) 3))
  (assert-true "4-step Donut-South trajectory sharpens P(donutFav) over the single step"
               (> (:p-donut-favorite p4) (:p-donut-favorite p1)))
  (assert-true "4-step trajectory suppresses Noodle (the far restaurant)"
               (< (:p-noodle-favorite p4) (:p-noodle-favorite p1))))

;; ===========================================================================
(println "\n== Phase A.4: generate-and-compare (the chapter's motivating example) ==")
;; At high alpha the agent is near-deterministic; keep tables whose predicted path
;; matches. Donut South is the NEAREST restaurant, so many tables (even all-equal)
;; route through it — which is exactly why the factorized softmax likelihood (A.1-A.3)
;; is the preferred method.
(let [gc (irl/generate-and-compare {:donut-vals food :veg-vals food :noodle-vals food} irl/donut-south-obs)]
  (println "  gen-compare matched" (count gc) "of 27 tables (incl all-equal [0 0 0] — less discriminative)")
  (assert-true "gen-compare keeps the donut-preferring table [2 0 0]" (contains? gc [2 0 0]))
  (assert-true "gen-compare is LESS discriminative than factorized (≥ half the tables match)"
               (>= (count gc) 10)))

;; ===========================================================================
;; PART B — POMDP-IRL (factorSequence / Equation 2): the Bandit testbed
;; ===========================================================================
(require '[agentmodels.pomdp-irl :as pi])

(println "\n== Phase B.1: forward VOI agent + the Switch two-level belief prior ==")
;; Switch combinator selecting the initial belief (the two-level prior's belief axis).
(assert-close "Switch idx 0 → informed belief (P(arm1=champagne)=1.0)"  1.0 (pi/switch-initial-belief 0) 1e-9)
(assert-close "Switch idx 1 → misinformed belief"  pi/MISINFORMED-P (pi/switch-initial-belief 1) 1e-9)
;; the belief-space VOI planner must value exploration only when the horizon is long.
(let [champ-inf (pi/make-bandit-voi-agent {:utility pi/likes-champagne :belief 1.0})
      champ-mis (pi/make-bandit-voi-agent {:utility pi/likes-champagne :belief pi/MISINFORMED-P})
      choc      (pi/make-bandit-voi-agent {:utility pi/likes-chocolate :belief pi/MISINFORMED-P})
      q (fn [ag arm b n] ((:q ag) arm b n))]
  (assert-true "informed champagne-lover pulls arm1 (knows champagne 5 > chocolate 3)"
               (> (q champ-inf 1 1.0 6) (q champ-inf 0 1.0 6)))
  (assert-true "misinformed champagne-lover EXPLORES arm1 at a long horizon (VOI)"
               (> (q champ-mis 1 pi/MISINFORMED-P 6) (q champ-mis 0 pi/MISINFORMED-P 6)))
  (assert-true "misinformed champagne-lover pulls arm0 at a SHORT horizon (no VOI)"
               (> (q champ-mis 0 pi/MISINFORMED-P 2) (q champ-mis 1 pi/MISINFORMED-P 2)))
  (assert-true "chocolate-lover pulls arm0 (chocolate is the known best)"
               (> (q choc 0 pi/MISINFORMED-P 6) (q choc 1 pi/MISINFORMED-P 6))))

(println "\n== Phase B.2: preference-vs-belief UNIDENTIFIABILITY (short horizon) ==")
(let [u (pi/unidentifiability 2)]
  (println "  N=2 all-arm0: P(likesChocolate)=" (.toFixed (:p-likes-chocolate u) 3)
           " P(informed)=" (.toFixed (:p-informed u) 3))
  ;; arm0-pulls are explained by EITHER a chocolate preference OR a misinformed belief,
  ;; so neither is identified — P(likesChocolate) is well below 1 (≈ 2/3).
  (assert-true "P(likesChocolate) is NOT identified after short-horizon arm0 pulls (< 0.8)"
               (< (:p-likes-chocolate u) 0.8))
  (assert-true "...but the informed-champagne-lover IS ruled out (P(likesChocolate) > prior 0.5)"
               (> (:p-likes-chocolate u) 0.5)))

(println "\n== Phase B.3: identifiability GROWS with the horizon ==")
(let [sweep (pi/horizon-sweep [2 3 4 6 8])
      pc    (fn [N] (:p-likes-chocolate (first (filter #(= N (:horizon %)) sweep))))]
  (doseq [{:keys [horizon p-likes-chocolate]} sweep]
    (println "  N=" horizon " P(likesChocolate)=" (.toFixed p-likes-chocolate 3)))
  (assert-true "short horizon (N=2) is unidentified (P < 0.8)" (< (pc 2) 0.8))
  (assert-true "long horizon (N=8) identifies the chocolate preference (P ≥ 0.99)" (>= (pc 8) 0.99))
  (assert-true "P(likesChocolate) is non-decreasing in the horizon (2 → 8)"
               (<= (pc 2) (pc 3) (pc 4) (pc 6) (pc 8))))

(println "\n== Phase B.4: factorSequence belief threading via the Scan combinator ==")
;; A sequence where the agent EXPLORES arm1 (reveals champagne) then pulls arm0.
(let [obs   [{:arm 1 :prize :champagne} {:arm 0 :prize :chocolate} {:arm 0 :prize :chocolate}]
      scanned (pi/belief-trajectory-via-scan pi/MISINFORMED-P obs)
      hosted  (pi/belief-trajectory-host     pi/MISINFORMED-P obs)]
  (println "  Scan-threaded beliefs:" (vec scanned) "  host-threaded:" (vec hosted))
  (assert-true "Scan belief thread matches the host recursion" (= (vec scanned) (vec hosted)))
  (assert-true "belief reveals arm1=champagne after the exploring pull (→ 1.0 thereafter)"
               (and (= pi/MISINFORMED-P (first hosted)) (= 1.0 (second hosted)))))

(println (str "\n" @passed " passed, " @failed " failed"))
(when (pos? @failed) (js/process.exit 1))
