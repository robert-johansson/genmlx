;; @tier fast
;; Contract test for the PROMOTED genmlx.agents Layer-9 API (Lever A, bean genmlx-ssui).
;; Proves the 8 promoted namespaces load + are reachable AS A SRC LIBRARY, that an agent
;; IS a generative function (GFI ops work), and that the flagship tensor-VI <-> recursive-EU
;; equivalence holds at the new namespace. Deep behavior of each module is covered by the
;; re-pointed suites (agentmodels_*_test, bandit_test); this is the library-level contract.
;; Run: bunx nbb@1.4.206 test/genmlx/agents_api_test.cljs

(ns genmlx.agents-api-test
  (:require [genmlx.agents.gridworld :as gw]
            [genmlx.agents.agent :as agent]
            [genmlx.agents.helpers :as h]
            [genmlx.agents.inverse :as inv]
            [genmlx.agents.biased-planners :as bp]
            [genmlx.agents.pomdp :as pomdp]
            [genmlx.agents.pomdp-env :as penv]
            [genmlx.agents.presentation :as present]
            [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.mlx.random :as rng]
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

(println "\n== genmlx.agents API contract (Layer-9 src library) ==")

;; ---- all 8 promoted namespaces load + expose their public API ----
;; (a failed require would abort the run; these confirm the public fns are resolvable)
(assert-true "genmlx.agents.gridworld reachable"  (every? fn? [gw/build-mdp gw/parse-grid gw/transition-tensor]))
(assert-true "genmlx.agents.agent reachable"      (every? fn? [agent/make-mdp-agent agent/value-iteration agent/recursive-eu agent/simulate-mdp]))
(assert-true "genmlx.agents.helpers reachable"    (every? fn? [h/softmax-action h/uniform-draw h/weighted-draw h/draw-value]))
(assert-true "genmlx.agents.inverse reachable"    (every? fn? [inv/goal-agents inv/action-loglik inv/normalize-logs inv/posterior-sequence]))
(assert-true "genmlx.agents.biased-planners reachable" (every? fn? [bp/make-biased-mdp-agent bp/biased-eu bp/delta bp/simulate-biased-mdp]))
(assert-true "genmlx.agents.pomdp reachable"      (every? fn? [pomdp/make-pomdp-agent pomdp/simulate-pomdp pomdp/make-bandit-agent pomdp/simulate-bandit]))
(assert-true "genmlx.agents.pomdp-env reachable"  (every? fn? [penv/bandit-pomdp penv/restaurant-gridworld]))
(assert-true "genmlx.agents.presentation reachable" (every? fn? [present/state->frame present/env->trajectory present/marginals->bars present/render-frame-text]))

;; ---- environment + agent + GFI: an agent IS a generative function ----
(def mdp (gw/build-mdp {:grid [[:empty :G] [:empty :empty]]
                        :utilities {:G 2.0 :timeCost -0.1} :start [0 1] :gamma 1.0}))
(assert-true "gridworld/build-mdp returns :T :R :term :terminals, S=4" (and (every? mdp [:T :R :term :terminals]) (= 4 (:S mdp))))

(def ag (agent/make-mdp-agent {:mdp mdp :alpha 1.0 :gamma 1.0 :n-iters 6}))
(assert-true "make-mdp-agent exposes :policy (GF), :Q, :expected-utility"
             (and (some? (:policy ag)) (some? (:Q ag)) (fn? (:expected-utility ag))))
;; p/simulate returns a Trace directly (not {:trace ...}); its :choices hold :action
(let [tr (p/simulate (dyn/auto-key (:policy ag)) [(:start-idx mdp)])]
  (assert-true "agent is a GF: p/simulate yields an :action choice + a score"
               (and (some? (cm/get-choice (:choices tr) [:action])) (some? (:score tr)))))
;; GFI inference works: p/assess gives a finite log-weight
(let [w (mx/item (:weight (p/assess (dyn/auto-key (:policy ag)) [(:start-idx mdp)]
                                    (cm/choicemap :action 0))))]
  (assert-true "agent is a GF: p/assess yields a finite log-weight" (js/isFinite w)))
;; flagship invariant — tensor value-iteration Q == recursive expected-utility
(let [Qh (mx/->clj (:Q ag)) eu (:expected-utility ag) S (:S mdp) A (:A mdp)
      err (apply max (for [s (range S) a (range A)] (Math/abs (- (get-in Qh [s a]) (eu s a)))))]
  (assert-true "tensor-VI Q == recursive-EU (max err < 1e-4)" (< err 1e-4)))

;; ---- helpers ----
(assert-true "helpers/softmax-action returns a distribution"
             (some? (h/softmax-action 1.0 (mx/array #js [1.0 2.0 0.5] mx/float32))))
(let [box (h/uniform-draw [:a :b :c])]
  (assert-true "helpers value-carrying draw: box + draw-value by index"
               (and (:dist box) (= 3 (count (:values box))) (= :b (h/draw-value box 1)))))

;; ---- genmlx-xpbm regressions ----
(println "\n== genmlx-xpbm regressions ==")
;; recursive-eu without numeric :alpha throws (pre-fix: (* nil q) -> 0 -> silently
;; uniform softmax policy when callers bypass make-mdp-agent)
(assert-true "recursive-eu without :alpha throws actionably"
             (try (agent/recursive-eu mdp) false
                  (catch :default e (boolean (re-find #"alpha" (str (.-message e)))))))
;; simulate-mdp :key now threads on the :host path (pre-fix: accepted but ignored)
(let [k  (rng/fresh-key 7)
      r1 (agent/simulate-mdp ag (:start-idx mdp) 6 {:key k})
      r2 (agent/simulate-mdp ag (:start-idx mdp) 6 {:key k})]
  (assert-true "simulate-mdp: same :key -> identical host trajectory"
               (and (= (:states r1) (:states r2)) (= (:actions r1) (:actions r2)))))
;; keyless host path still works (custom 1-arity :act agents keep working)
(let [r (agent/simulate-mdp ag (:start-idx mdp) 4)]
  (assert-true "simulate-mdp: keyless host path still rolls out"
               (and (vector? (:states r)) (vector? (:actions r)))))

;; ---- biased-planners: k=0 limit recovery (biased == unbiased) ----
(assert-close "biased-planners/delta(0,d)=1" 1.0 (bp/delta 0 3) 1e-12)
(let [bi (bp/make-biased-mdp-agent {:mdp mdp :alpha 1.0 :gamma 1.0 :n-iters 6}
                                   {:discount 0.0 :bias :sophisticated})
      eub (:expected-utility bi) eus (:expected-utility ag) S (:S mdp) A (:A mdp)
      err (apply max (for [s (range S) a (range A)] (Math/abs (- (eub s a) (eus s a)))))]
  (assert-true "biased agent at k=0 recovers the unbiased agent (max err < 1e-4)" (< err 1e-4)))

;; ---- inverse: goal-inference posterior normalizes ----
(let [gas (inv/goal-agents {:grid [[:A :empty :B]] :goals [:A :B] :alpha 2.0})
      seq (inv/posterior-sequence gas {:A 0.5 :B 0.5} [])]
  (assert-true "inverse/goal-agents builds one agent per goal" (= #{:A :B} (set (keys gas))))
  (assert-close "inverse/posterior-sequence prior sums to 1" 1.0 (reduce + (vals (first seq))) 1e-9))

;; ---- inverse: BATCHED [G,S,A] posterior == per-cell host action-loglik (bean genmlx-y2hh) ----
;; Ground truth = the original per-cell algorithm, re-implemented here from the
;; UNCHANGED inv/action-loglik + inv/normalize-logs. The shape-batched
;; posterior-sequence must reproduce it to float32 tolerance on every prefix.
(defn host-posterior-sequence
  [goal-agents prior observations]
  (let [goals (keys goal-agents)]
    (loop [obs observations
           logp (into {} (map (fn [g] [g (Math/log (prior g))]) goals))
           acc  [(inv/normalize-logs logp)]]
      (if (empty? obs)
        acc
        (let [[s a] (first obs)
              logp' (into {} (map (fn [g] [g (+ (logp g) (inv/action-loglik (goal-agents g) s a))]) goals))]
          (recur (rest obs) logp' (conj acc (inv/normalize-logs logp'))))))))
(defn max-posterior-err [a b]
  (apply max 0.0 (for [[ma mb] (map vector a b), g (keys ma)]
                   (Math/abs (- (double (ma g)) (double (mb g)))))))
;; (a) 2-goal slice fixture
(let [grid [[:empty :empty :empty :empty :empty]
            [:empty :empty :empty :empty :empty]
            [:A     :empty :empty :empty :B]]
      gas  (inv/goal-agents {:grid grid :goals [:A :B] :alpha 2.0})
      obs  [[2 3] [7 3] [12 1] [13 1]]
      bat  (inv/posterior-sequence gas {:A 0.5 :B 0.5} obs)
      hos  (host-posterior-sequence gas {:A 0.5 :B 0.5} obs)]
  (assert-true  "batched: one posterior per prefix (2-goal)" (= (count hos) (count bat) 5))
  (assert-true  "batched == host action-loglik to 1e-4 (2-goal)" (< (max-posterior-err bat hos) 1e-4)))
;; (b) 4-goal corner fixture
(let [grid [[:A :empty :B]
            [:empty :empty :empty]
            [:C :empty :D]]
      gas  (inv/goal-agents {:grid grid :goals [:A :B :C :D] :alpha 2.0})
      obs  [[4 0] [4 1] [1 3] [2 2]]
      pri  {:A 0.25 :B 0.25 :C 0.25 :D 0.25}
      bat  (inv/posterior-sequence gas pri obs)
      hos  (host-posterior-sequence gas pri obs)]
  (assert-true  "batched: one posterior per prefix (4-goal)" (= (count hos) (count bat) 5))
  (assert-true  "every batched posterior sums to 1 (4-goal)"
                (every? (fn [m] (< (Math/abs (- 1.0 (reduce + (vals m)))) 1e-6)) bat))
  (assert-true  "batched == host action-loglik to 1e-4 (4-goal)" (< (max-posterior-err bat hos) 1e-4)))

(println (str "\n" @passed " passed, " @failed " failed"))
(when (pos? @failed) (js/process.exit 1))
