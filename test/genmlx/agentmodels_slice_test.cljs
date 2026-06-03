;; Headless tests for the agentmodels TUI vertical slice — everything BELOW the
;; render seam: gridworld geometry/tensors, tensor value iteration + the
;; softmax-action policy rollout, and the pure Frame producer + ASCII renderer.
;; No Ink, no terminal — pure data, proven by asserts. If this passes, the live
;; TUI renders the same data correctly by construction.
;;
;; Run: bun run --bun nbb test/genmlx/agentmodels_slice_test.cljs

(ns genmlx.agentmodels-slice-test
  (:require [genmlx.agents.gridworld :as gw]
            [genmlx.agents.agent :as agent]
            [genmlx.agents.presentation :as pres]
            [genmlx.agents.inverse :as inv]
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

(println "\n== Section 4: transition noise (orthogonal slip) ==")
;; A wall-free 3x3 so the slip lands on three distinct cells. Center state 4 =
;; (1,1); action right(1): intended -> 5, slips up -> 1, slips down -> 7.
(def open-mdp (gw/build-mdp {:grid [[:empty :empty :empty]
                                    [:empty :empty :empty]
                                    [:empty :empty :B]]
                             :utilities {:B 5.0 :timeCost -0.1}
                             :start [0 0] :noise 0.2}))
(let [row (vec (mx/->clj (mx/idx (mx/idx (:T open-mdp) 4) 1)))]   ; T[4, right, :]
  (assert-true "row T[4,right,:] sums to 1"            (< (Math/abs (- 1.0 (reduce + row))) 1e-5))
  (assert-true "intended (->5) gets 1-noise = 0.8"     (< (Math/abs (- 0.8 (nth row 5))) 1e-5))
  (assert-true "slip up   (->1) gets noise/2 = 0.1"    (< (Math/abs (- 0.1 (nth row 1))) 1e-5))
  (assert-true "slip down (->7) gets noise/2 = 0.1"    (< (Math/abs (- 0.1 (nth row 7))) 1e-5))
  (assert-true "all mass is on intended + the two slips"
               (< (Math/abs (- 1.0 (+ (nth row 5) (nth row 1) (nth row 7)))) 1e-5)))
(assert-true "noisy T rows each still sum to 1 (total 36)"
             (< (Math/abs (- 36.0 (mx/item (mx/sum (:T open-mdp))))) 1e-4))

(println "\n== Section 5: richer maze grid (wall forces a detour) ==")
;;   . . . . . .
;;   . # # # # .
;;   . . . . # .
;;   # # # . # .
;;   A . . . . B     start top-left; the only route to B hugs the right edge
(def maze [[:empty :empty :empty :empty :empty :empty]
           [:empty :wall  :wall  :wall  :wall  :empty]
           [:empty :empty :empty :empty :wall  :empty]
           [:wall  :wall  :wall  :empty :wall  :empty]
           [:A     :empty :empty :empty :empty :B]])
(def maze-mdp (gw/build-mdp {:grid maze :utilities {:A 1.0 :B 5.0 :timeCost -0.1}
                             :start [0 0] :gamma 1.0}))   ; noise 0 -> deterministic assert
(def maze-opt (agent/make-mdp-agent {:mdp maze-mdp :alpha ##Inf :gamma 1.0 :n-iters 40}))
(assert-equal "maze S = 30"     30 (:S maze-mdp))
(assert-equal "B at idx 29"     {29 :B} (select-keys (:terminals maze-mdp) [29]))
(let [vs (vec (mx/->clj (:V maze-opt)))]
  (assert-true "V maximal at B (idx 29)" (= 29 (apply max-key #(nth vs %) (range 30)))))
(let [{:keys [states]} (agent/simulate-mdp maze-opt (:start-idx maze-mdp) 30)
      frames (pres/env->trajectory maze-mdp {:states states :actions []} (:V maze-opt))]
  (println "  optimal maze path:" states)
  (println "\n " (str/replace (pres/render-frame-text (last frames)) "\n" "\n  "))
  (assert-equal "optimal maze rollout ends at B (idx 29)" 29 (last states))
  (assert-true  "optimal path avoids every wall" (not-any? (:walls maze-mdp) states)))

;; a noisy rollout (the path the live demo walks) must still produce a valid
;; trajectory: every visited cell is in-bounds and never a wall.
(let [nm (gw/build-mdp {:grid maze :utilities {:A 1.0 :B 5.0 :timeCost -0.1}
                        :start [0 0] :gamma 1.0 :noise 0.3})
      ag (agent/make-mdp-agent {:mdp nm :alpha 5.0 :gamma 1.0 :n-iters 40})
      {:keys [states]} (agent/simulate-mdp ag (:start-idx nm) 30)]
  (assert-true "noisy rollout stays in-bounds and off walls"
               (every? (fn [s] (and (<= 0 s) (< s 30) (not ((:walls nm) s)))) states)))

(println "\n== Section 6: inverse goal inference (assess-based) ==")
;;   . . . . .
;;   . . . . .
;;   A . . . B     start top-centre (idx 2); A/B are mirror-symmetric about col 2
(def inv-grid [[:empty :empty :empty :empty :empty]
               [:empty :empty :empty :empty :empty]
               [:A     :empty :empty :empty :B]])
(def goal-ags (inv/goal-agents {:grid inv-grid :goals [:A :B] :alpha 2.0}))
;; hand-specified observations from idx 2: DOWN, DOWN (on the symmetry axis, so
;; uninformative), then RIGHT, RIGHT toward B (idx 14). down = 3, right = 1.
(def obs [[2 3] [7 3] [12 1] [13 1]])
(def posts (inv/posterior-sequence goal-ags {:A 0.5 :B 0.5} obs))
(assert-equal "one posterior per prefix (prior + 4 obs)" 5 (count posts))
(assert-true  "every posterior sums to 1"
              (every? (fn [m] (< (Math/abs (- 1.0 (reduce + (vals m)))) 1e-6)) posts))
(assert-true  "prior is uniform"  (< (Math/abs (- 0.5 (:B (nth posts 0)))) 1e-6))
(assert-true  "two DOWN moves are symmetric -> still ~0.5"
              (< (Math/abs (- 0.5 (:B (nth posts 2)))) 0.02))
(assert-true  "after turning RIGHT toward B, P(B) jumps above 0.8"
              (> (:B (nth posts 4)) 0.8))
(assert-true  "evidence accumulates: P(B) after rights > after downs"
              (> (:B (nth posts 4)) (:B (nth posts 2))))
(println "  P(goal=B) over time:" (mapv #(.toFixed (:B %) 3) posts))
;; dist->bars marks only the highlighted (true) goal
(let [pb   (pres/dist->bars "P(goal)" {:A 0.3 :B 0.7} :B)
      bar  (fn [lbl] (first (filter #(= lbl (:label %)) (:bars pb))))]
  (assert-true "dist->bars highlights the true goal (B)"      (:highlight (bar "B")))
  (assert-true "dist->bars leaves the other goal (A) unmarked" (not (:highlight (bar "A")))))

(println "\n== Section 7: recursive expectedUtility == tensor value iteration ==")
;; A tiny grid; both paths must compute identical Q-values and first action, for
;; a finite (soft) alpha AND the alpha = Inf (hard max) limit.
(def eq-mdp (gw/build-mdp {:grid [[:empty :G]
                                  [:empty :empty]]
                           :utilities {:G 2.0 :timeCost -0.1} :start [0 1] :gamma 1.0}))
(defn- argmax-idx [xs] (first (apply max-key second (map-indexed vector xs))))
(doseq [[label alpha n] [["soft (alpha=1.0)" 1.0 4] ["hard (alpha=Inf)" ##Inf 6]]]
  (let [ag   (agent/make-mdp-agent {:mdp eq-mdp :alpha alpha :gamma 1.0 :n-iters n})
        Qh   (mx/->clj (:Q ag))               ; tensor Q [S][A]
        eu   (:expected-utility ag)            ; recursive EU(s,a,horizon)
        errs (for [s (range (:S eq-mdp)) a (range (:A eq-mdp))]
               (Math/abs (- (get-in Qh [s a]) (eu s a))))
        s0   (:start-idx eq-mdp)]
    (assert-true (str "  " label ": recursive EU matches tensor Q (max err < 1e-4)")
                 (< (apply max errs) 1e-4))
    (assert-equal (str "  " label ": first action agrees at start")
                  (argmax-idx (get Qh s0))
                  (argmax-idx (mapv #(eu s0 %) (range (:A eq-mdp)))))))

(println (str "\n" @passed " passed, " @failed " failed"))
(when (pos? @failed) (js/process.exit 1))
