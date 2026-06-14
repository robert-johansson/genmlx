;; @tier medium
;; genmlx.agents API CONTRACTS — "tests as contracts" for the v1.0 freeze
;; (ROADMAP Phase 3 item 3, the genmlx-youg promotion criterion; bean genmlx-p4jo).
;;
;; agents_api_test.cljs proves reachability + agent-as-GF + VI<->recursive-EU. This
;; suite pins the things that would silently DRIFT across an API freeze and that the
;; scoping (wf_177d7bb0-65f) flagged: the per-constructor return-map shapes, the
;; family-specific :act signatures, and the POMDP/bandit agent contracts. Belief
;; host<->tensor + nil/impossible-obs MATH equivalence is already covered by
;; belief_tensor_test.cljs (36/36) — not duplicated here.
;;
;; THE HONEST CONTRACT (encoded below, documented in src/genmlx/agents/CONTRACTS.md):
;;   * The ONLY keys guaranteed on every agent are {:act, :params}.
;;   * :act has FAMILY-SPECIFIC signatures — state-based (s)/(s key) for MDP/biased,
;;     belief-based (belief s) for POMDP, (belief key) for bandit. There is no single
;;     uniform :act arity.
;;   * :Q/:V are MDP-only (biased agents are recursion-only, no tensor value table);
;;     :policy/:expected-utility exist for the planning families but NOT for the bandit;
;;     :belief-Q/:update-belief are the partially-observed families' surface.
;;
;; Run: bun run --bun nbb test/genmlx/agents_contracts_test.cljs

(ns genmlx.agents-contracts-test
  (:require [genmlx.agents.gridworld :as gw]
            [genmlx.agents.agent :as agent]
            [genmlx.agents.biased-planners :as bp]
            [genmlx.agents.pomdp :as pomdp]
            [genmlx.agents.pomdp-env :as penv]
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
(defn- act-int? [a] (and (integer? a) (<= 0 a)))

;; ---------------------------------------------------------------------------
;; build one agent per family
;; ---------------------------------------------------------------------------

(def mdp (gw/build-mdp {:grid [[:empty :empty] [:empty :goal]]
                        :utilities {:goal 1.0 :timeCost -0.1} :start [0 0] :gamma 1.0 :noise 0.0}))
(def start (:start-idx mdp))
(def A (:A mdp))

(def mdp-ag    (agent/make-mdp-agent {:mdp mdp :alpha 3.0 :gamma 1.0 :n-iters 12}))
(def biased-ag (bp/make-biased-mdp-agent {:mdp mdp :alpha 3.0 :gamma 1.0 :n-iters 12}
                                         {:discount 0.0 :bias :sophisticated}))

(def pgrid [[:A    :empty :B]
            [:wall :empty :wall]
            [:wall :empty :wall]
            [:wall :empty :wall]
            [:wall :empty :wall]])
(def penv-e   (penv/restaurant-gridworld {:grid pgrid :goals [:A :B] :signpost 7
                                          :true-world :A :start [1 4]}))
(def pomdp-ag (pomdp/make-pomdp-agent (assoc penv-e :alpha 2.0 :gamma 1.0 :n-iters 40)))
(def pstart   (:start-idx penv-e))

(def bandit-ag     (pomdp/make-bandit-agent {:strategy :thompson :alpha 4.0}))
(def bandit-belief {:arms [[1 1] [1 1]]})

;; ---------------------------------------------------------------------------
;; 1. the common minimal contract — EVERY agent has {:act fn, :params map}
;; ---------------------------------------------------------------------------
(println "\n== 1. common minimal contract: {:act, :params} on every agent ==")
(doseq [[nm a] [["mdp" mdp-ag] ["biased" biased-ag] ["pomdp" pomdp-ag] ["bandit" bandit-ag]]]
  (assert-true (str nm ": :act is callable")  (fn? (:act a)))
  (assert-true (str nm ": :params is a map")  (map? (:params a))))

;; ---------------------------------------------------------------------------
;; 2. per-family return-map keys (the documented, asserted differences)
;; ---------------------------------------------------------------------------
(println "\n== 2. per-family return-map keys ==")
;; MDP: tensor value table + GF policy + recursive EU
(assert-true "mdp: has :Q :V :policy :expected-utility"
             (and (some? (:Q mdp-ag)) (some? (:V mdp-ag))
                  (some? (:policy mdp-ag)) (fn? (:expected-utility mdp-ag))))
;; biased: GF policy + EU, but NO tensor :Q/:V (recursion-only) — the key drift
(assert-true "biased: has :policy :expected-utility but NO :Q/:V (recursion-only)"
             (and (some? (:policy biased-ag)) (fn? (:expected-utility biased-ag))
                  (nil? (:Q biased-ag)) (nil? (:V biased-ag))))
;; POMDP: belief-space surface
(assert-true "pomdp: has :belief-Q :update-belief :expected-utility :worlds"
             (and (fn? (:belief-Q pomdp-ag)) (fn? (:update-belief pomdp-ag))
                  (fn? (:expected-utility pomdp-ag)) (some? (:worlds pomdp-ag))))
;; bandit: arm-values + conjugate update, and NO GF :policy (Thompson/softmax-of-means)
(assert-true "bandit: has :arm-values :update-belief but NO :policy"
             (and (fn? (:arm-values bandit-ag)) (fn? (:update-belief bandit-ag))
                  (nil? (:policy bandit-ag))))

;; ---------------------------------------------------------------------------
;; 3. :act signatures are FAMILY-SPECIFIC (state-based vs belief-based)
;; ---------------------------------------------------------------------------
(println "\n== 3. :act signatures (family-specific) ==")
;; MDP/biased :act — state-based: (s) and (s key); the keyed form is deterministic
(doseq [[nm a] [["mdp" mdp-ag] ["biased" biased-ag]]]
  (assert-true (str nm ": (:act s) -> action int") (act-int? ((:act a) start)))
  (let [k (rng/fresh-key 11)]
    (assert-true (str nm ": (:act s key) is deterministic in key")
                 (= ((:act a) start k) ((:act a) start k)))))
;; POMDP :act — belief-based: (belief s) -> action int
(assert-true "pomdp: (:act belief s) -> action int"
             (act-int? ((:act pomdp-ag) (:prior pomdp-ag) pstart)))
;; bandit :act — belief-based: (belief key) -> arm int
(assert-true "bandit: (:act belief key) -> arm int in {0,1}"
             (contains? #{0 1} ((:act bandit-ag) bandit-belief (rng/fresh-key 3))))

;; ---------------------------------------------------------------------------
;; 4. MDP/biased agents ARE generative functions (policy GF) via the GFI
;; ---------------------------------------------------------------------------
(println "\n== 4. planning agents are GFs (policy through the GFI) ==")
(doseq [[nm a] [["mdp" mdp-ag] ["biased" biased-ag]]]
  (let [tr (p/simulate (dyn/auto-key (:policy a)) [start])]
    (assert-true (str nm ": p/simulate yields an :action choice + a score")
                 (and (some? (cm/get-choice (:choices tr) [:action])) (some? (:score tr)))))
  (let [w (mx/item (:weight (p/assess (dyn/auto-key (:policy a)) [start] (cm/choicemap :action 0))))]
    (assert-true (str nm ": p/assess yields a finite log-weight") (js/isFinite w))))

;; ---------------------------------------------------------------------------
;; 5. POMDP belief contract — belief is {world->prob}; filtering keeps it normalized
;; ---------------------------------------------------------------------------
(println "\n== 5. POMDP belief contract ==")
(let [ub    (:update-belief pomdp-ag)
      prior (:prior pomdp-ag)]
  (assert-true "belief is a {world->prob} map" (and (map? prior) (every? keyword? (keys prior))))
  (assert-close "prior sums to 1" 1.0 (reduce + (vals prior)) 1e-9)
  ;; nil-obs is the identity (absence is non-informative — the unified contract)
  (assert-true "nil observation leaves the belief unchanged (agent level)"
               (= prior (ub prior 10 nil)))
  ;; an informative observation snaps + stays normalized
  (let [b' (ub prior 7 :A)]
    (assert-close "reveal :A at the signpost -> P(:A)=1" 1.0 (:A b') 1e-9)
    (assert-close "filtered belief stays normalized" 1.0 (reduce + (vals b')) 1e-9))
  ;; belief-Q is an [A] MLX row over actions
  (let [q (:belief-Q pomdp-ag)]
    (assert-true "belief-Q returns an [A] action row" (= [A] (mx/shape (q prior pstart))))))

;; ---------------------------------------------------------------------------
;; 6. bandit contract — arm-values + conjugate Beta update
;; ---------------------------------------------------------------------------
(println "\n== 6. bandit contract ==")
(let [vals (:arm-values bandit-ag)
      ub   (:update-belief bandit-ag)]
  (assert-true "arm-values returns one mean per arm" (= 2 (count (vals bandit-belief))))
  (assert-close "Beta(1,1) arm mean = 0.5" 0.5 (first (vals bandit-belief)) 1e-9)
  (let [b' (ub bandit-belief 0 1)]            ; success on arm 0
    (assert-true "conjugate update: arm 0 success -> alpha+1" (= [2 1] (get-in b' [:arms 0])))
    (assert-true "conjugate update: other arms unchanged"     (= [1 1] (get-in b' [:arms 1])))))

(println (str "\n" @passed " passed, " @failed " failed"))
(when (pos? @failed) (js/process.exit 1))
