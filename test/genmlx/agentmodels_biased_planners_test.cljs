;; @tier slow
;; Headless tests for the agentmodels biased/bounded planners (Ch 5).
;; Run: bun run --bun nbb test/genmlx/agentmodels_biased_planners_test.cljs

(ns genmlx.agentmodels-biased-planners-test
  (:require [genmlx.agents.gridworld :as gw]
            [genmlx.agents.agent :as agent]
            [genmlx.agents.biased-planners :as bp]
            [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]))

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

;; ===========================================================================
(println "\n== Section 1: hyperbolic discount δ(k,d) = 1/(1+kd) ==")
(assert-true  "δ(0,d)=1 for all d"        (every? #(= 1.0 (bp/delta 0 %)) (range 5)))
(assert-close "δ(1,0)=1"                  1.0 (bp/delta 1 0) 1e-12)
(assert-close "δ(2,3)=1/7"                (/ 1.0 7.0) (bp/delta 2 3) 1e-12)
(assert-true  "δ(1,·) strictly decreasing" (apply > (mapv #(bp/delta 1 %) (range 6))))

;; ===========================================================================
(println "\n== Section 6: limit recovery — biased(k=0, C_g=∞) == standard agent ==")
(def eq-mdp (gw/build-mdp {:grid [[:empty :G] [:empty :empty]]
                           :utilities {:G 2.0 :timeCost -0.1} :start [0 1] :gamma 1.0}))
(doseq [[label alpha n] [["soft (α=1.0)" 1.0 4] ["hard (α=##Inf)" ##Inf 6]]]
  (let [std (agent/make-mdp-agent {:mdp eq-mdp :alpha alpha :gamma 1.0 :n-iters n})
        bi  (bp/make-biased-mdp-agent {:mdp eq-mdp :alpha alpha :gamma 1.0 :n-iters n}
                                      {:discount 0.0 :bias :sophisticated})
        Qh  (mx/->clj (:Q std))
        eus (:expected-utility std)
        eub (:expected-utility bi)
        S   (:S eq-mdp) A (:A eq-mdp)
        e-rec (apply max (for [s (range S) a (range A)] (Math/abs (- (eus s a) (eub s a)))))
        e-Q   (apply max (for [s (range S) a (range A)] (Math/abs (- (get-in Qh [s a]) (eub s a)))))]
    (assert-true (str "  " label ": biased k=0 matches recursive EU (max err < 1e-4)") (< e-rec 1e-4))
    (assert-true (str "  " label ": biased k=0 matches tensor Q   (max err < 1e-4)") (< e-Q 1e-4))
    (assert-equal (str "  " label ": first action agrees at start")
                  (argmax-idx (get Qh (:start-idx eq-mdp)))
                  (argmax-idx (mapv #(eub (:start-idx eq-mdp) %) (range A))))))

;; ===========================================================================
(println "\n== Section 2: Procrastination MDP — hyperbolic preference reversal ==")
(def pmdp (bp/procrastination-mdp {}))            ; reward 4.5, work -1, wait -0.1, deadline 10
(defn work-day [bias k]
  (bp/procrastination-work-day
    (bp/make-biased-mdp-agent {:mdp pmdp :alpha ##Inf :gamma 1.0 :n-iters 14}
                              {:discount k :bias bias})))
(def naive-days (mapv #(work-day :naive %) [0.0 0.5 1.0 2.0 4.0]))
(println "  naive work-day vs k[0,0.5,1,2,4]:" naive-days "  (nil = never worked)")
(println "  naive k=8 work-day:" (work-day :naive 8.0))
(println "  soph  work-day vs k[0,0.5,1,2,4]:" (mapv #(work-day :sophisticated %) [0.0 0.5 1.0 2.0 4.0]))
(defn- when-worked [d] (if (nil? d) 1e9 d))      ; nil (never worked) = maximally late
(assert-equal "k=0 works on day 0" 0 (work-day :naive 0.0))
(assert-true  "naive work-day non-decreasing in k (procrastinates more)"
              (apply <= (map when-worked naive-days)))
;; preference reversal (the bean's headline): the Naive agent's choice flips from
;; work-now (low k) to wait-forever (high k) as the discount crosses a threshold.
(assert-true  "preference reversal: Naive works at k=0 but NEVER completes at high k"
              (and (some? (work-day :naive 0.0)) (nil? (work-day :naive 4.0))))
(assert-true  "Naive eventually fails to complete by the deadline (high k)"
              (nil? (work-day :naive 8.0)))
;; sophistication advantage: foreseeing the spiral, Sophisticated still completes
;; (works) at a discount where the Naive agent already procrastinates to failure.
(assert-true  "Sophisticated completes where Naive fails (k=2: soph works, naive never)"
              (and (some? (work-day :sophisticated 2.0)) (nil? (work-day :naive 2.0))))
(assert-true  "Sophisticated works no later than Naive at every swept k"
              (every? true? (map (fn [k] (<= (when-worked (work-day :sophisticated k))
                                             (when-worked (work-day :naive k))))
                                 [0.0 0.5 1.0 2.0 4.0])))
;; the MECHANISM behind the reversal (not just the outcome): a Naive high-k agent
;; PLANS to work but never does (plan ≠ do), and its own preference flips between
;; delay 0 (wait now) and delay 1 (believes it will work later).
(let [n4 (bp/make-biased-mdp-agent {:mdp pmdp :alpha ##Inf :gamma 1.0 :n-iters 14}
                                   {:discount 4.0 :bias :naive})
      H  14, eu (:eu n4)]                       ; action 0 = wait, 1 = work; W_0 = state 0
  (assert-true "Naive (k=4) PLANS to work (plan contains a work action)"
               (some #(= 1 %) (:actions (bp/planned-rollout n4 0 10))))
  (assert-true "...but NEVER works in reality (plan ≠ do — the time-inconsistency)"
               (not-any? #(= 1 %) (:actions (bp/simulate-biased-mdp n4 0 10))))
  (assert-true "W_0 preference reverses: WAIT preferred at d=0 but WORK believed at d=1"
               (and (> (eu 0 0 H 0) (eu 0 1 H 0))        ; d=0: EU(wait) > EU(work)
                    (> (eu 0 1 H 1) (eu 0 0 H 1)))))     ; d=1: EU(work) > EU(wait)

;; ===========================================================================
(println "\n== Section 3: Restaurant-Choice — Naive succumbs, Sophisticated avoids ==")
(def rmdp (bp/restaurant-mdp {}))
(def r-start (:start-idx rmdp))
(assert-equal "restaurant grid is 8×6 (S=48)" 48 (:S rmdp))
(assert-equal "three restaurant terminals" #{:veg :donut-n :donut-s} (set (vals (:terminals rmdp))))
(assert-true  "T rows each sum to 1 (total S*A=192)"
              (< (Math/abs (- 192.0 (mx/item (mx/sum (:T rmdp))))) 1e-3))
(defn rest-agent [bias k]
  (bp/make-biased-mdp-agent {:mdp rmdp :alpha ##Inf :gamma 1.0 :n-iters 16} {:discount k :bias bias}))
(def K 3.0)
(let [rational (rest-agent :naive 0.0)              ; k=0 ⇒ unbiased
      naive    (rest-agent :naive K)
      soph     (rest-agent :sophisticated K)]
  (assert-equal "rational (k=0) reaches Veg"                :veg     (bp/restaurant-endpoint rational r-start 16))
  (assert-equal "Naive (k=3) is captured by Donut-North"    :donut-n (bp/restaurant-endpoint naive r-start 16))
  (assert-equal "Sophisticated (k=3) reaches Veg"           :veg     (bp/restaurant-endpoint soph  r-start 16))
  ;; plan ↔ do: the Naive agent PLANS to reach Veg but DOES reach Donut-North
  (let [planned (get (:terminals rmdp) (last (:states (bp/planned-rollout naive r-start 16))))
        did     (bp/restaurant-endpoint naive r-start 16)]
    (println "  naive planned endpoint:" planned "  actual endpoint:" did)
    (assert-equal "Naive PLANS to reach Veg"        :veg     planned)
    (assert-true  "Naive's plan ≠ what it does"      (not= planned did)))
  ;; sophisticated is time-consistent: plan == do
  (let [planned (get (:terminals rmdp) (last (:states (bp/planned-rollout soph r-start 16))))]
    (assert-equal "Sophisticated plan == do (both Veg)" :veg planned))
  ;; the equal-utility Donut-South decoy is reached by NOBODY
  (assert-true  "Donut-South (equal util) is never chosen"
                (not-any? #(= :donut-s (bp/restaurant-endpoint (rest-agent % K) r-start 16))
                          [:naive :sophisticated])))

;; ===========================================================================
(println "\n== Section 4: Reward-myopia (C_g) — far reward beyond the look-ahead ==")
(def lmdp (bp/line-mdp {}))     ; small reward 1 step left of start, big reward 5 steps right
(def l-start (:start-idx lmdp))
(defn line-first-action [cg]
  ((:act (bp/make-biased-mdp-agent {:mdp lmdp :alpha ##Inf :gamma 1.0 :n-iters 10}
                                   {:discount 0.0 :bias :sophisticated :reward-myopic-bound cg}))
   l-start))
;; action 0 = left (toward small/near), 1 = right (toward big/far)
(println "  first action: unbounded=" (line-first-action ##Inf) " C_g=2=" (line-first-action 2))
(assert-equal "unbounded agent walks toward the BIG far reward (right)"  1 (line-first-action ##Inf))
(assert-equal "C_g=2 agent (can't see 5 steps) takes the SMALL near reward (left)" 0 (line-first-action 2))

;; ===========================================================================
(println "\n== Section 5: Update-myopia (C_m) — bounded value of information (POMDP) ==")
(def voi (bp/voi-world {}))                       ; prior {:A 0.55 :B 0.45}, true-world :B
(defn pomdp-agent [cm]
  (bp/make-biased-pomdp-agent (assoc voi :alpha ##Inf :n-iters 8)
                              {:discount 0.0 :bias :sophisticated :update-myopic-bound cm}))
(def opt (pomdp-agent ##Inf))                     ; optimal: values information
(def myo (pomdp-agent 0))                          ; update-myopic: never updates in look-ahead
(def s0 (:start-idx opt))
(def pv (:prior-vec opt))
(println "  start-idx" s0 " signpost" (:signpost voi) " prior-vec" pv)
(println "  optimal first action:" ((:act opt) pv s0) "  myopic first action:" ((:act myo) pv s0))
;; action 3 = down = the 1-step detour onto the signpost (which reveals the world)
(assert-equal "optimal (C_m=∞) walks to the signpost first (down=3)" 3 ((:act opt) pv s0))
(assert-true  "myopic (C_m=0) sees no VOI — does NOT detour to the signpost" (not= 3 ((:act myo) pv s0)))
(let [orun (bp/simulate-biased-pomdp opt :B s0 8 pv)
      mrun (bp/simulate-biased-pomdp myo :B s0 8 pv)
      oend (get (:terminals opt) (last (:states orun)))
      mend (get (:terminals myo) (last (:states mrun)))]
  (println "  optimal path" (:states orun) "→" oend "   myopic path" (:states mrun) "→" mend)
  (assert-equal "optimal walks-and-checks, then reaches the TRUE goal B"   :B oend)
  (assert-equal "myopic commits blind to the higher-prior goal A (wrong)"  :A mend))
;; mechanical: with C_m large the agent's planned belief DOES update at the signpost
;; (optimal prefers the detour); with C_m=0 it doesn't — already exercised by the
;; divergent first actions above, but assert the optimal/myopic EU ordering directly.
(let [eu-opt (:eu opt) eu-myo (:eu myo)
      ;; value of stepping toward the signpost (down=3) vs toward goal A (left=0)
      d-opt (- (eu-opt pv s0 3 8 0) (eu-opt pv s0 0 8 0))
      d-myo (- (eu-myo pv s0 3 8 0) (eu-myo pv s0 0 8 0))]
  (println "  EU(down)-EU(left): optimal" (.toFixed d-opt 3) " myopic" (.toFixed d-myo 3))
  (assert-true "optimal values the detour more than the myopic agent does" (> d-opt d-myo)))
;; stronger, unambiguous information value: the optimal agent values the SAME detour
;; action far more than the myopic one (≈2.25 = collect-true-reward vs commit-blind),
;; which is information value, not a one-step time-cost artifact.
(assert-true "optimal values the detour itself >1.0 more than myopic (genuine VOI, not geometry)"
             (> (- ((:eu opt) pv s0 3 8 0) ((:eu myo) pv s0 3 8 0)) 1.0))
;; robustness: an un-normalised prior must yield the SAME policy as its normalisation
;; (the belief recursion requires a distribution; make-biased-pomdp-agent normalises).
(let [opt2 (pomdp-agent ##Inf)
      scaled (bp/make-biased-pomdp-agent (assoc voi :prior {:A 1.1 :B 0.9} :alpha ##Inf :n-iters 8)
                                         {:discount 0.0 :bias :sophisticated :update-myopic-bound ##Inf})]
  (assert-true  "prior-vec normalised to sum 1" (< (Math/abs (- 1.0 (reduce + (:prior-vec scaled)))) 1e-9))
  (assert-equal "un-normalised prior {1.1 0.9} acts identically to {0.55 0.45}"
                ((:act opt2) pv s0) ((:act scaled) (:prior-vec scaled) s0)))
;; the POMDP :act now respects alpha (mirrors the MDP agent): finite alpha ⇒ a soft
;; Boltzmann policy, not a hard argmax. Assert via assess (deterministic): a NON-best
;; action still has finite log-prob under the soft policy (it would be -inf at ##Inf).
(let [soft (bp/make-biased-pomdp-agent (assoc voi :alpha 2.0 :n-iters 8)
                                       {:discount 0.0 :bias :sophisticated :update-myopic-bound ##Inf})
      lp0  (mx/item (:weight (p/assess (dyn/auto-key (:policy soft)) [(:prior-vec soft) s0]
                                       (cm/choicemap :action 0))))]    ; action 0 is not the argmax
  (assert-true "finite-α POMDP agent exposes a GFI :policy" (some? (:policy soft)))
  (assert-true "finite-α (α=2) POMDP policy is soft: a non-best action has finite log-prob" (js/isFinite lp0)))

(println (str "\n" @passed " passed, " @failed " failed"))
(when (pos? @failed) (js/process.exit 1))
