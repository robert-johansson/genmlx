;; Headless tests for the FUSED MDP rollout (bean genmlx-5zdd):
;; genmlx.agents.rollout/rollout-mdp, reached via agent/simulate-mdp :rollout-mode
;; :fused. The whole trajectory is one lazy Metal graph (state threaded as an MLX
;; tensor, action + next-state sampled in-graph, one materialize at the end) — no
;; per-step mx/item. At alpha=##Inf/noise=0 it must produce vector-IDENTICAL
;; :states/:actions to the host loop; the host (:host, default) path is unchanged.
;;
;; Run: bunx nbb@1.4.206 test/genmlx/rollout_fused_test.cljs

(ns genmlx.rollout-fused-test
  (:require [genmlx.agents.gridworld :as gw]
            [genmlx.agents.agent :as agent]
            [genmlx.mlx.random :as rng]))

(def passed (volatile! 0))
(def failed (volatile! 0))
(defn assert-true [msg c]
  (if c (do (vswap! passed inc) (println " PASS" msg))
        (do (vswap! failed inc) (println " FAIL" msg))))
(defn assert-equal [msg expected actual]
  (if (= expected actual) (do (vswap! passed inc) (println " PASS" msg "  =" (pr-str actual)))
      (do (vswap! failed inc) (println " FAIL" msg "  expected:" (pr-str expected) "  got:" (pr-str actual)))))

(defn mk [grid utilities alpha noise]
  (agent/make-mdp-agent {:mdp (gw/build-mdp {:grid grid :utilities utilities :noise noise})
                         :alpha alpha :n-iters 40}))

;; ---------------------------------------------------------------------------
(println "\n== Section 1: deterministic (alpha=Inf, noise=0) fused == host exactly ==")
;; TIE-FREE worlds (single-corridor geometry / asymmetric utilities) — every state
;; has a UNIQUE optimal action, so the fused argmax and the host's categorical-argmax
;; agree and :states/:actions are vector-IDENTICAL.
(def GRIDS-TIEFREE
  {"corridor" {:grid [[:A :empty :empty :empty :B]] :utils {:A 3.0 :B 1.0 :timeCost -0.1}}
   "tall"     {:grid [[:A    :empty :B]
                      [:wall :empty :wall]
                      [:wall :empty :wall]
                      [:empty :empty :empty]]
               :utils {:A 4.0 :B 3.0 :timeCost -0.1}}})
(doseq [[label {:keys [grid utils]}] GRIDS-TIEFREE]
  (let [ag (mk grid utils ##Inf 0.0)
        S  (:S (:mdp ag))
        all-ok (every? (fn [start]
                         (let [h (agent/simulate-mdp ag start 30)
                               f (agent/simulate-mdp ag start 30 {:rollout-mode :fused})]
                           (and (= (:states h) (:states f)) (= (:actions h) (:actions f)))))
                       (remove (:walls (:mdp ag)) (range S)))]
    (assert-true (str label ": fused == host :states/:actions for every start state") all-ok)))

(println "\n== Section 1b: tied world — fused reaches the same goal in the same length ==")
;; An open 2D world has MULTIPLE equal-length optimal paths to the corner goal, so
;; host and fused legitimately pick different (equally optimal) routes (the tie-break
;; divergence). Equivalence then is: same terminal reached, same trajectory length.
(let [ag (mk [[:A :empty :B] [:empty :empty :empty] [:empty :empty :C]]
             {:A 5.0 :B 2.0 :C -1.0 :timeCost -0.1} ##Inf 0.0)
      S  (:S (:mdp ag))
      ok (every? (fn [start]
                   (let [h (agent/simulate-mdp ag start 30)
                         f (agent/simulate-mdp ag start 30 {:rollout-mode :fused})]
                     (and (= (last (:states h)) (last (:states f)))         ; same goal
                          (= (count (:states h)) (count (:states f))))))    ; same (optimal) length
                 (remove (:walls (:mdp ag)) (range S)))]
  (assert-true "open hike: fused reaches the same goal in the same #steps (ties allowed)" ok))

(println "\n== Section 2: terminal truncation matches host length ==")
(let [{:keys [grid utils]} (GRIDS-TIEFREE "corridor")
      ag (mk grid utils ##Inf 0.0)]
  (doseq [start [1 2 3]]
    (let [h (agent/simulate-mdp ag start 30)
          f (agent/simulate-mdp ag start 30 {:rollout-mode :fused})]
      (assert-equal (str "start " start ": same trajectory length") (count (:states h)) (count (:states f)))
      (assert-true  (str "start " start ": fused ends at a terminal")
                    (contains? (:terminals (:mdp ag)) (last (:states f))))
      (assert-true  (str "start " start ": one action per transition")
                    (= (count (:actions f)) (dec (count (:states f))))))))

(println "\n== Section 3: already-terminal start ==")
(let [{:keys [grid utils]} (GRIDS-TIEFREE "corridor")
      ag (mk grid utils ##Inf 0.0)]
  (let [f (agent/simulate-mdp ag 0 30 {:rollout-mode :fused})]   ; idx 0 = :A terminal
    (assert-equal "terminal start -> single state, no actions" {:states [0] :actions []} f)))

(println "\n== Section 4: noisy / soft fused rollout stays valid + terminates ==")
;; finite alpha + noise => stochastic; can't match the host sequence (different RNG
;; thread), but the fused path must stay in-bounds (off walls) and reach a terminal.
(let [ag (mk [[:A :empty :B] [:empty :empty :empty] [:empty :empty :C]]
             {:A 5.0 :B 2.0 :C -1.0 :timeCost -0.1} 50.0 0.1)
      walls (:walls (:mdp ag))
      S (:S (:mdp ag)) terms (:terminals (:mdp ag))]
  (let [results (mapv (fn [seed] (agent/simulate-mdp ag 6 40 {:rollout-mode :fused :key (rng/fresh-key seed)}))
                      (range 8))]
    (assert-true "noisy fused stays in-bounds and off walls"
                 (every? (fn [r] (every? #(and (<= 0 % (dec S)) (not (walls %))) (:states r))) results))
    (assert-true "noisy fused reaches a terminal within horizon (high alpha)"
                 (every? (fn [r] (contains? terms (last (:states r)))) results))))

(println "\n== Section 5: host (default) path byte-identical (no regression) ==")
(let [{:keys [grid utils]} (GRIDS-TIEFREE "corridor")
      ag (mk grid utils ##Inf 0.0)
      h1 (agent/simulate-mdp ag 2 30)
      h2 (agent/simulate-mdp ag 2 30 {:rollout-mode :host})]
  (assert-equal "explicit :host == default" h1 h2)
  (assert-true  "default reaches goal" (contains? (:terminals (:mdp ag)) (last (:states h1)))))

(println (str "\n== Results: " @passed " passed, " @failed " failed =="))
(when (pos? @failed) (js/process.exit 1))
