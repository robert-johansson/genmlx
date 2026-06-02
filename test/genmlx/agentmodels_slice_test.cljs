;; Headless tests for the agentmodels TUI vertical slice — everything BELOW the
;; render seam: gridworld geometry/tensors, tensor value iteration + the
;; softmax-action policy rollout, and the pure Frame producer + ASCII renderer.
;; No Ink, no terminal — pure data, proven by asserts. If this passes, the live
;; TUI renders the same data correctly by construction.
;;
;; Run: bun run --bun nbb test/genmlx/agentmodels_slice_test.cljs

(ns genmlx.agentmodels-slice-test
  (:require [agentmodels.gridworld :as gw]
            [agentmodels.agent :as agent]
            [agentmodels.presentation :as pres]
            [genmlx.mlx :as mx]
            [clojure.string :as str]))

(def passed (volatile! 0))
(def failed (volatile! 0))
(defn assert-true [msg c]
  (if c (do (vswap! passed inc) (println " PASS" msg))
        (do (vswap! failed inc) (println " FAIL" msg))))
(defn assert-equal [msg expected actual]
  (if (= expected actual) (do (vswap! passed inc) (println " PASS" msg "  =" (pr-str actual)))
      (do (vswap! failed inc) (println " FAIL" msg "  expected:" (pr-str expected) "  got:" (pr-str actual)))))

;; -- The demo world (top row first; screen coords, y down) --------------------
;;   A . .        idx:  0 1 2
;;   . # .              3 4 5
;;   . . B              6 7 8
;; A (util 1) top-left, B (util 5) bottom-right, wall in the centre (idx 4).
(def grid [[:A     :empty :empty]
           [:empty :wall  :empty]
           [:empty :empty :B]])
(def mdp (gw/build-mdp {:grid grid
                        :utilities {:A 1.0 :B 5.0 :timeCost -0.1}
                        :start [1 0]            ; top-middle, idx 1
                        :gamma 1.0}))

(println "\n== Section 1: gridworld geometry + tensors ==")
(assert-equal "S = 9"            9 (:S mdp))
(assert-equal "A = 4"            4 (:A mdp))
(assert-equal "start-idx = 1"    1 (:start-idx mdp))
(assert-equal "wall at idx 4"    {4 :wall} (select-keys (zipmap (:walls mdp) (repeat :wall)) [4]))
(assert-equal "terminals {0:A 8:B}" {0 :A 8 :B} (:terminals mdp))
(assert-equal "T shape [9 4 9]"  [9 4 9] (mx/shape (:T mdp)))
(assert-equal "R shape [9 4]"    [9 4]   (mx/shape (:R mdp)))
(assert-equal "term shape [9]"   [9]     (mx/shape (:term mdp)))
;; every T[s,a,:] is a probability row that sums to 1  => total == S*A == 36
(assert-true  "T rows each sum to 1 (total 36)" (< (Math/abs (- 36.0 (mx/item (mx/sum (:T mdp))))) 1e-4))
;; known transitions: right from 1 -> 2; down from 1 hits the wall -> stays at 1; left from 0 clamps -> 0
(assert-equal "right(1) -> 2"        2 ((:ns-fn mdp) 1 1))
(assert-equal "down(1) into wall -> 1 (stays)" 1 ((:ns-fn mdp) 1 3))
(assert-equal "left(0) clamps -> 0"  0 ((:ns-fn mdp) 0 0))

(println "\n== Section 2: value iteration + optimal-policy rollout ==")
(def opt (agent/make-mdp-agent {:mdp mdp :alpha ##Inf :gamma 1.0 :n-iters 24}))
(assert-equal "Q shape [9 4]" [9 4] (mx/shape (:Q opt)))
(assert-equal "V shape [9]"   [9]   (mx/shape (:V opt)))
;; V(B) should be the largest value in the grid (highest-reward terminal)
(let [vs (vec (mx/->clj (:V opt)))]
  (assert-true "V is maximal at B (idx 8)" (= 8 (apply max-key #(nth vs %) (range 9)))))
;; the deterministic optimal agent must reach B (idx 8), routing around the wall
(let [{:keys [states]} (agent/simulate-mdp opt 1 8)]
  (println "  optimal path (alpha=Inf):" states)
  (assert-equal "optimal rollout ends at B (idx 8)" 8 (last states))
  (assert-true  "optimal rollout does NOT end at A (idx 0)" (not= 0 (last states)))
  (assert-true  "path stepped around the wall (never visits idx 4)" (not (some #{4} states))))

(println "\n== Section 3: the pure Frame producer + ASCII seam ==")
(def frame0 (pres/state->frame mdp 1 {:step 0 :vs (vec (mx/->clj (:V opt)))
                                      :vlo 0.0 :vhi 1.0}))
(assert-equal "frame :cells count == W*H == 9" 9 (count (:cells frame0)))
(assert-equal "cell idx 1 is the agent" :agent (:role (nth (:cells frame0) 1)))
(assert-equal "cell idx 4 is the wall"  :wall  (:role (nth (:cells frame0) 4)))
(assert-equal "cell idx 0 is a goal (A)" :goal  (:role (nth (:cells frame0) 0)))
(let [txt (pres/render-frame-text frame0)]
  (println "\n" (str/replace txt "\n" "\n  "))
  (assert-true "ASCII frame shows the agent @"  (str/includes? txt "@"))
  (assert-true "ASCII frame shows terminal A"   (str/includes? txt "A"))
  (assert-true "ASCII frame shows terminal B"   (str/includes? txt "B")))
;; the trajectory is one Frame per state and crosses the seam via (mx/->clj V) once
(let [roll   (agent/simulate-mdp opt 1 8)
      frames (pres/env->trajectory mdp roll (:V opt))]
  (assert-equal "trajectory length == #states" (count (:states roll)) (count frames))
  (assert-true  "final frame's agent sits on B" (= :agent (:role (nth (:cells (last frames)) 8)))))

(println (str "\n" @passed " passed, " @failed " failed"))
(when (pos? @failed) (js/process.exit 1))
