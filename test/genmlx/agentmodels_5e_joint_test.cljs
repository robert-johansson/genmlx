;; Headless tests for agentmodels Ch 5e — full [immediate,delayed] Restaurant
;; geometry + joint inference of biases and preferences (genmlx-69s8).
;; Run: bunx nbb@1.4.206 test/genmlx/agentmodels_5e_joint_test.cljs

(ns genmlx.agentmodels-5e-joint-test
  (:require [genmlx.agents.gridworld :as gw]
            [genmlx.agents.biased-planners :as bp]
            [genmlx.mlx :as mx]))

(def passed (volatile! 0))
(def failed (volatile! 0))
(defn assert-true [msg c]
  (if c (do (vswap! passed inc) (println " PASS" msg))
        (do (vswap! failed inc) (println " FAIL" msg))))
(defn assert-equal [msg expected actual]
  (if (= expected actual) (do (vswap! passed inc) (println " PASS" msg "  =" (pr-str actual)))
      (do (vswap! failed inc) (println " FAIL" msg "  expected:" (pr-str expected) "  got:" (pr-str actual)))))
(defn assert-close [msg expected actual tol]
  (if (<= (Math/abs (- expected actual)) tol) (do (vswap! passed inc) (println " PASS" msg "  =" actual))
      (do (vswap! failed inc) (println " FAIL" msg "  expected:" expected "  got:" actual))))

(defn- argmax-idx [xs] (first (apply max-key second (map-indexed vector xs))))

;; agentmodels disU reference (the donutTempting/value math): with γ=1 and the
;; 2-step stay, EU(L@0,a,d) must equal dis(d)·imm + dis(d+1)·del.
(defn dis [k d] (/ 1.0 (+ 1.0 (* (double k) (double d)))))
(defn disU [k [imm del] d] (+ (* (dis k d) imm) (* (dis k (inc d)) del)))

;; ===========================================================================
(println "\n== Phase 1.1: the [imm,del] Restaurant geometry (state augmentation) ==")
(def tmdp (bp/restaurant-temptation-mdp {}))
;; agentmodels plans until terminal (no fixed t-cap); we approximate that with a
;; horizon comfortably longer than the longest route (the soph long route is ~10
;; transitions), so the t≤1 base case never binds before a restaurant is reached.
(def H 18)
(def t-start (:start-idx tmdp))
(assert-equal "grid is 6×8 (48 grid states)"            48 (:grid-S tmdp))
(assert-equal "augmented S = 48 grid + 4 restaurant twins" 52 (:S tmdp))
(assert-equal "start cell = [3,6] → idx 39"             39 t-start)
(assert-equal "four restaurants"                        #{:veg :donut-n :donut-s :noodle}
              (set (vals (:restaurants tmdp))))
(assert-equal "ONLY the 4 L@1 twins are terminal"       4 (count (:terminals tmdp)))
(assert-true  "Donut-N grid cell is idx 14"             (= :donut-n (get (:restaurants tmdp) 14)))
(assert-true  "Veg grid cell is idx 4"                  (= :veg (get (:restaurants tmdp) 4)))
(assert-true  "T rows each sum to 1 (S*A = 208)"
              (< (Math/abs (- 208.0 (mx/item (mx/sum (:T tmdp))))) 1e-3))

;; ===========================================================================
(println "\n== Phase 1.2: 2-step stay delivers δ(d)·imm + δ(d+1)·del (= disU) ==")
;; Build a naive k=1 agent and read its memoized EU at the Donut-N arrival state.
(defn t-agent [alpha bias k]
  (bp/make-biased-mdp-agent {:mdp tmdp :alpha alpha :gamma 1.0 :n-iters H}
                            {:discount k :bias bias}))
(let [naive (t-agent ##Inf :naive 1.0)
      eu    (:eu naive)
      dn    14                         ; Donut-N arrival (L@0)
      vg    4]                         ; Veg arrival (L@0)
  ;; EU(L@0,a,d) at the arrival state IS the agentmodels 2-step discounted value.
  (assert-close "EU(DonutN@0, ·, d=0) = disUDonut(0) = +5"
                (disU 1.0 [10 -10] 0) (eu dn 0 H 0) 1e-6)
  (assert-close "EU(DonutN@0, ·, d=2) = disUDonut(2)"
                (disU 1.0 [10 -10] 2) (eu dn 0 H 2) 1e-6)
  (assert-close "EU(Veg@0, ·, d=0) = disUVeg(0) = -10 + 20/2 = 0"
                (disU 1.0 [-10 20] 0) (eu vg 0 H 0) 1e-6)
  (assert-close "EU(Veg@0, ·, d=3) = disUVeg(3)"
                (disU 1.0 [-10 20] 3) (eu vg 0 H 3) 1e-6)
  ;; the delayed component lands one step later, discounted by δ(d+1).
  (let [dn-twin (get (:twin tmdp) dn)]
    (assert-close "EU(DonutN@1 twin, ·, d=1) = δ(1)·(delayed -10) = -5"
                  (* (dis 1.0 1) -10) (eu dn-twin 0 H 1) 1e-6)))

;; ===========================================================================
(println "\n== Phase 1.3: endpoints at α=##Inf — rational→Veg, naive→DonutN, soph→Veg ==")
(let [rational (t-agent ##Inf :naive 0.0)   ; k=0 ⇒ unbiased
      naive    (t-agent ##Inf :naive 1.0)
      soph     (t-agent ##Inf :sophisticated 1.0)]
  (assert-equal "rational (k=0) reaches Veg (net-best, no temptation)"
                :veg (bp/restaurant-endpoint rational t-start H))
  (assert-equal "Naive (k=1) is captured by Donut-North (temptation)"
                :donut-n (bp/restaurant-endpoint naive t-start H))
  (assert-equal "Sophisticated (k=1) reaches Veg (routes around the temptation)"
                :veg (bp/restaurant-endpoint soph t-start H))
  ;; preference reversal: the Naive agent PLANS to reach Veg but DOES reach Donut-N.
  (let [planned (get (:terminals tmdp) (last (:states (bp/planned-rollout naive t-start H))))
        did     (bp/restaurant-endpoint naive t-start H)]
    (println "  naive planned endpoint:" planned "  actual endpoint:" did)
    (assert-equal "Naive PLANS to reach Veg"   :veg planned)
    (assert-true  "Naive's plan ≠ what it does (time-inconsistency)" (not= planned did)))
  ;; sophisticated is time-consistent: plan == do
  (let [planned (get (:terminals tmdp) (last (:states (bp/planned-rollout soph t-start H))))]
    (assert-equal "Sophisticated plan == do (both Veg)" :veg planned))
  ;; the equal-net-utility Donut-South decoy is reached by nobody
  (assert-true  "Donut-South (net-equal decoy) is never chosen"
                (not-any? #(= :donut-s (bp/restaurant-endpoint % t-start H))
                          [rational naive soph])))

;; ===========================================================================
(println "\n== Phase 1.4: temptation is DECISIVE at FINITE α (does NOT wash out) ==")
;; The bean's prior failure: scalar utilities → naive/soph differ only as an argmax
;; tie-break (gap 0) → washes out at finite α. With paired [imm,del] utilities the
;; decision gaps are bounded away from 0, so a finite-α softmax is near-deterministic.
;; junction idx 15 (=[3,2]): action 0 = left → Donut-N (idx 14); action 2 = up → Veg.
;; crossover idx 27 (=[3,4]): action 1 = right → long route; action 2 = up → short route.
(def junction 15)
(def crossover 27)
(let [naive (t-agent 500.0 :naive 1.0)
      soph  (t-agent 500.0 :sophisticated 1.0)
      euN   (:eu naive)
      euS   (:eu soph)
      ;; junction: at d=0 re-planning, the present-biased agent prefers Donut-N.
      gap-j (- (euN junction 0 H 0) (euN junction 2 H 0))
      ;; routing divergence at the crossover (the soph pre-commits to the long route).
      naive-up?   (> (euN crossover 2 H 0) (euN crossover 1 H 0))
      soph-right? (> (euS crossover 1 H 0) (euS crossover 2 H 0))]
  (println "  junction EU(→DonutN) − EU(→Veg) =" (.toFixed gap-j 4)
           "  ⇒ softmax(α=500·gap) ≈" (.toFixed (/ 1.0 (+ 1.0 (Math/exp (* -500.0 gap-j)))) 4))
  (assert-true "junction: defection gap > 0 (finite-α agent still prefers Donut-N)" (> gap-j 0.0))
  (assert-true "junction: gap is decisive (>0.05 ⇒ α=500 softmax ≈ 1, no wash-out)" (> gap-j 0.05))
  (assert-true "crossover: Naive heads UP the short route (toward the temptation)" naive-up?)
  (assert-true "crossover: Sophisticated branches RIGHT onto the long route" soph-right?)
  ;; and the finite-α (α=500) rollout endpoints match the α=##Inf ones (decisive).
  (assert-equal "finite-α Naive still ends at Donut-North"
                :donut-n (bp/restaurant-endpoint naive t-start H))
  (assert-equal "finite-α Sophisticated still ends at Veg"
                :veg (bp/restaurant-endpoint soph t-start H)))

;; ===========================================================================
;; PHASE 2 — joint inference of biases and preferences (agentmodels 5e)
;; ===========================================================================
(require '[agentmodels.restaurant-joint-inference :as r])
(defn E-alpha [s] (reduce (fn [a [al pr]] (+ a (* al pr))) 0.0 (:alpha-marginal s)))
(def nv @r/naive-trajectory)
(def vd @r/veg-direct-trajectory)

;; Each model's agents (and their EU caches) are scoped to a single `let` so they
;; become garbage-collectable before the next model builds — peak heap stays at one
;; model's grid (the 512-agent discounting model fits comfortably).

(println "\n== Phase 2.1-2.3: Discounting model — the P(donutTempting) headline + repeats + VegDirect ==")
(let [spec   (r/discounting-spec)
      agents (r/build-agents spec)
      p1     (r/joint-posterior (assoc spec :states (:states nv) :actions (:actions nv)) agents)
      p3     (r/joint-posterior (assoc spec :states (:states nv) :actions (:actions nv) :number-repeats 2) agents)
      pvd    (r/joint-posterior (assoc spec :states (:states vd) :actions (:actions vd)) agents)
      pr     (:prior p1) po (:posterior p1) po3 (:posterior p3)]
  ;; 2.1 — headline
  (println "  prior  P(tempt)=" (.toFixed (:p-donut-tempting pr) 4) " E[vmd]=" (.toFixed (:e-veg-minus-donut pr) 2))
  (println "  post   P(tempt)=" (.toFixed (:p-donut-tempting po) 4) " E[vmd]=" (.toFixed (:e-veg-minus-donut po) 2)
           " P(naive)=" (.toFixed (:p-naive po) 3))
  (assert-true "prior P(donutTempting) < 0.1 (agentmodels: 'less than 0.1')"   (< (:p-donut-tempting pr) 0.1))
  (assert-true "posterior P(donutTempting) ≥ 0.7 (agentmodels: 'closer to 0.9')" (>= (:p-donut-tempting po) 0.7))
  (assert-true "prior E[vegMinusDonut] ≈ 0 (equal utility most likely a priori)"  (< (Math/abs (:e-veg-minus-donut pr)) 0.5))
  (assert-true "posterior E[vegMinusDonut] > 0 (net Veg preference; rules out equal/donut)" (> (:e-veg-minus-donut po) 0.0))
  (assert-true "posterior P(naive) > 0.8 (the path is explained by naivety)"      (> (:p-naive po) 0.8))
  ;; 2.2 — graded with repeats
  (println "  graded P(naive):" (.toFixed (:p-naive po) 3) "→" (.toFixed (:p-naive po3) 3)
           "   P(tempt):" (.toFixed (:p-donut-tempting po) 3) "→" (.toFixed (:p-donut-tempting po3) 3))
  (assert-true "P(naive) non-decreasing with repeats (1× → 3×)"         (>= (:p-naive po3) (- (:p-naive po) 1e-9)))
  (assert-true "P(donutTempting) non-decreasing with repeats (1× → 3×)" (>= (:p-donut-tempting po3) (- (:p-donut-tempting po) 1e-9)))
  (assert-true "3× observation is (near-)decisive about naivety (≥ 0.99)" (>= (:p-naive po3) 0.99))
  ;; 2.3 — VegDirect uninformative about bias
  (println "  VegDirect P(naive) =" (.toFixed (:p-naive (:posterior pvd)) 3) "  (prior 0.5)")
  (assert-true "P(naive | VegDirect) ≈ prior 0.5 (no information about bias)"
               (< (Math/abs (- (:p-naive (:posterior pvd)) 0.5)) 0.1)))

(println "\n== Phase 2.4: Optimal (non-discounting) model — the Naive path looks like noise ==")
(let [spec   (r/optimal-spec)
      agents (r/build-agents spec)
      o1     (r/joint-posterior (assoc spec :states (:states nv) :actions (:actions nv)) agents)
      o3     (r/joint-posterior (assoc spec :states (:states nv) :actions (:actions nv) :number-repeats 2) agents)
      pri (E-alpha (:prior o1)) a1 (E-alpha (:posterior o1)) a3 (E-alpha (:posterior o3))
      vmd (:e-veg-minus-donut (:posterior o1))]
  (println "  E[vegMinusDonut]=" (.toFixed vmd 2) "  E[alpha]: prior" (.toFixed pri 1)
           " 1×" (.toFixed a1 1) " 3×" (.toFixed a3 1))
  (assert-true "optimal model infers a DONUT preference (E[vegMinusDonut] < 0)" (< vmd 0.0))
  (assert-true "optimal model explains the path via NOISE (E[alpha] 1× < prior)" (< a1 pri))
  (assert-true "repeats make noise less plausible (E[alpha] 3× > 1×)" (> a3 a1))
  (assert-true "no discounting ⇒ P(donutTempting) = 0" (< (:p-donut-tempting (:posterior o1)) 1e-9)))

(println "\n== Phase 2.5: Full joint model — discount∈{0,1}, naive/soph, alpha, utilities ==")
;; NOTE on fidelity: agentmodels describes the full model at 1× as 'ambiguous, like
;; the optimal model'. GenMLX's forward agents are faithful and DECISIVE (α=1000), so
;; the high-likelihood naive+discounting+temptation explanation already wins at 1×.
;; We therefore assert the robust DIRECTIONS (favors naive+discounting+temptation on
;; the naive path; sharpens with repeats) rather than that exact qualitative nuance.
(let [spec   (r/full-spec)
      agents (r/build-agents spec)
      f1     (r/joint-posterior (assoc spec :states (:states nv) :actions (:actions nv)) agents)
      f3     (r/joint-posterior (assoc spec :states (:states nv) :actions (:actions nv) :number-repeats 2) agents)
      pri (:prior f1) po1 (:posterior f1) po3 (:posterior f3)]
  (println "  prior P(naive)=" (.toFixed (:p-naive pri) 2) " P(disc)=" (.toFixed (:p-discounting pri) 2))
  (println "  1×    P(naive)=" (.toFixed (:p-naive po1) 3) " P(disc)=" (.toFixed (:p-discounting po1) 3)
           " P(tempt)=" (.toFixed (:p-donut-tempting po1) 3))
  (println "  3×    P(naive)=" (.toFixed (:p-naive po3) 3) " P(tempt)=" (.toFixed (:p-donut-tempting po3) 3))
  (assert-close "prior P(naive) = 0.5"      0.5 (:p-naive pri) 1e-6)
  (assert-close "prior P(discounting) = 0.5" 0.5 (:p-discounting pri) 1e-6)
  (assert-true "1×: posterior favors naive over prior"        (> (:p-naive po1) (:p-naive pri)))
  (assert-true "1×: posterior favors discounting over prior"  (> (:p-discounting po1) (:p-discounting pri)))
  (assert-true "1×: posterior P(donutTempting) up over prior" (> (:p-donut-tempting po1) (:p-donut-tempting pri)))
  (assert-true "3×: P(naive) sharpens (≥ 1× and ≥ 0.99)"      (and (>= (:p-naive po3) (- (:p-naive po1) 1e-9))
                                                                   (>= (:p-naive po3) 0.99))))

(println (str "\n" @passed " passed, " @failed " failed"))
(when (pos? @failed) (js/process.exit 1))
