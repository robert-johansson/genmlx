;; Headless tests for the ADJACENCY-REVEAL restaurant POMDP (agentmodels Ch 3c
;; makeGridWorldPOMDP) — the real local-observation model, vs the single-signpost
;; restaurant-gridworld. The latent is a per-restaurant OPEN/CLOSED vector (2^k
;; worlds); the agent learns a restaurant's status only when ADJACENT to it, heads
;; to the preferred restaurant, and DETOURS to the backup if it observes the
;; preferred one closed. Deterministic (alpha=Inf, noise 0), so paths/beliefs are
;; reproducible. Run: bun run --bun nbb test/genmlx/agentmodels_pomdp_adjacency_test.cljs

(ns genmlx.agentmodels-pomdp-adjacency-test
  (:require [genmlx.agents.pomdp :as pomdp]
            [genmlx.agents.pomdp-env :as env]))

(def passed (volatile! 0))
(def failed (volatile! 0))
(defn assert-true [msg c]
  (if c (do (vswap! passed inc) (println " PASS" msg))
        (do (vswap! failed inc) (println " FAIL" msg))))
(defn assert-equal [msg expected actual]
  (if (= expected actual) (do (vswap! passed inc) (println " PASS" msg "  =" (pr-str actual)))
      (do (vswap! failed inc) (println " FAIL" msg "  expected:" (pr-str expected) "  got:" (pr-str actual)))))
(defn- close? [a b tol] (<= (Math/abs (- a b)) tol))

;; 4x3 grid: A (idx 0) top-left, the PREFERRED restaurant; B (idx 3) top-right, the
;; backup. Start idx 9 (bottom, col 1). A's adjacent cells {1,4}; B's {2,7}.
;;   A . . B      0 1 2 3
;;   . . . .      4 5 6 7
;;   . @ . .      8 9 10 11   start at 9 = [1,2]
(def grid [[:a     :empty :empty :b]
           [:empty :empty :empty :empty]
           [:empty :empty :empty :empty]])
(def UTIL {:a 5.0 :b 3.0})
(def OPEN-PROB {:a 0.6 :b 0.9})
(def A-CELL 0) (def B-CELL 3) (def START 9)

(defn build [true-world]
  (let [e  (env/restaurant-pomdp {:grid grid :utilities UTIL :open-prob OPEN-PROB
                                  :true-world true-world :start [1 2]})
        pa (pomdp/make-pomdp-agent (assoc e :alpha ##Inf :gamma 1.0 :noise 0.0 :n-iters 40))]
    [e pa]))
(defn- P-a-open [b] (reduce + (map (fn [[w p]] (if (:a w) p 0.0)) b)))

;; ---------------------------------------------------------------------------
(println "\n-- world structure: per-restaurant open/closed latent vector --")
(let [[e _] (build {:a true :b true})]
  (assert-equal "2 restaurants {:a :b}" [:a :b] (:restaurants e))
  (assert-equal "2^2 = 4 open/closed worlds" 4 (count (:worlds e)))
  (assert-true  "worlds are the {restaurant -> open?} configs"
                (= #{{:a true :b true} {:a true :b false} {:a false :b true} {:a false :b false}}
                   (set (:worlds e))))
  (assert-equal "world-utils: A open -> 5, B closed -> 0 (+ timeCost)"
                {:a 5.0 :b 0.0 :timeCost -0.1} (get (:world-utils e) {:a true :b false}))
  ;; prior is the product of the independent open-probs
  (assert-true  "prior P({A-open,B-open}) = 0.6*0.9 = 0.54"
                (close? 0.54 (get (:prior e) {:a true :b true}) 1e-9))
  (assert-true  "prior is normalized" (close? 1.0 (reduce + (vals (:prior e))) 1e-9)))

(println "\n-- observation is ADJACENCY-gated (not a single signpost) --")
(let [[e _] (build {:a false :b true})
      obs   (:observe e)]
  (assert-equal "adjacent to A (idx 1): observe A's status only" [[:a false]] (obs {:a false :b true} 1))
  (assert-equal "adjacent to B (idx 2): observe B's status only" [[:b true]]  (obs {:a false :b true} 2))
  (assert-equal "not adjacent to any restaurant (idx 5): no observation" nil  (obs {:a false :b true} 5))
  (assert-equal "on the start cell (idx 9): no observation" nil (obs {:a false :b true} START)))

(println "\n-- belief filter: an adjacency observation reveals only that restaurant --")
(let [[e pa] (build {:a true :b true})
      ub     (:update-belief pa)
      prior  (:prior pa)]
  (assert-true "obs=nil leaves the belief unchanged" (= prior (ub prior 5 nil)))
  (let [b' (ub prior 1 [[:a true]])]   ; observe A open at A's neighbour
    (assert-true "observe A-open -> P(A-open) = 1" (close? 1.0 (P-a-open b') 1e-9))
    (assert-true "...but B stays uncertain (P(B-open) = prior 0.9)"
                 (close? 0.9 (reduce + (map (fn [[w p]] (if (:b w) p 0.0)) b')) 1e-6)))
  (let [b' (ub prior 1 [[:a false]])]
    (assert-true "observe A-closed -> P(A-open) = 0" (close? 0.0 (P-a-open b') 1e-9))))

;; ---------------------------------------------------------------------------
(println "\n-- contingent re-planning: head to A, detour to B if A is closed --")
(let [[e pa] (build {:a true :b true})            ; A is open
      {:keys [states beliefs]} (pomdp/simulate-pomdp pa e START 14)]
  (println "   A-open  | path" states "| P(A-open)" (mapv #(.toFixed (P-a-open %) 2) beliefs))
  (assert-equal "A open: the agent reaches A (the preferred restaurant)" A-CELL (last states))
  (assert-true  "belief starts at the prior (P(A-open) = 0.6)" (close? 0.6 (P-a-open (first beliefs)) 1e-6))
  (assert-true  "belief snaps to A-open once adjacent to A" (close? 1.0 (P-a-open (last beliefs)) 1e-9)))

(let [[e pa] (build {:a false :b true})           ; A closed, B open
      {:keys [states observations beliefs]} (pomdp/simulate-pomdp pa e START 14)]
  (println "   A-closed| path" states "| obs" (vec (remove nil? observations))
           "| P(A-open)" (mapv #(.toFixed (P-a-open %) 2) beliefs))
  (assert-equal "A closed, B open: the agent DETOURS and reaches B (the backup)" B-CELL (last states))
  (assert-true  "belief still flat before reaching A's neighbour (P(A-open) = prior)"
                (close? 0.6 (P-a-open (nth beliefs 1)) 1e-6))
  (assert-true  "belief collapses to A-closed after observing A" (close? 0.0 (P-a-open (last beliefs)) 1e-9))
  (assert-true  "the agent observed A then B (adjacency reveals, in order)"
                (= [[[:a false]] [[:b true]]] (vec (remove nil? observations)))))

;; the latent genuinely matters: same prior/agent, different open/closed world ->
;; different restaurant reached (learned only through the adjacency observations).
(let [a-open  (last (:states (pomdp/simulate-pomdp (second (build {:a true  :b true})) (first (build {:a true  :b true})) START 14)))
      a-closed (last (:states (pomdp/simulate-pomdp (second (build {:a false :b true})) (first (build {:a false :b true})) START 14)))]
  (assert-true "the open/closed latent changes the outcome (A vs B)" (and (= A-CELL a-open) (= B-CELL a-closed))))

(println (str "\n== Results: " @passed " passed, " @failed " failed =="))
(when (pos? @failed) (js/process.exit 1))
