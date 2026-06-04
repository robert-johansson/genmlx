(ns ch3-demo
  "Ch 3 gridworld MDP demo — the end-to-end vertical slice.

   Real GenMLX inference drives the picture: tensor value iteration -> Q, a
   softmax-action rollout (decision noise from alpha) through a stochastic
   environment (transition noise from the gridworld slip), then the
   render-agnostic Frame producer. The agent walks the sampled path in the
   terminal. +/- change the rationality alpha (alpha = infinity -> deterministic
   choices); n cycles the transition noise. Pure data crosses the seam; this file
   only wires + renders.

   Demo state lives in one r/atom (the proven reactive model); the gallery routes
   keys here via on-key and starts/stops the demo via enter!/leave!. Frame
   stepping + autoplay are owned by a reusable views/trajectory-player — this
   file only decides WHEN to resample (alpha/noise/r) and what to render around
   the player."
  (:require ["ink" :refer [Text Box]]
            [reagent.core :as r]
            [genmlx.agents.gridworld :as gw]
            [genmlx.agents.agent :as agent]
            [genmlx.agents.presentation :as pres]
            [views]))

;; A maze where the wall belt forces a detour (top row first; screen coords).
;;   . . . . . .
;;   . # # # # .
;;   . . . . # .
;;   # # # . # .
;;   A . . . . B     start top-left; A = util 1, B = util 5
(def grid [[:empty :empty :empty :empty :empty :empty]
           [:empty :wall  :wall  :wall  :wall  :empty]
           [:empty :empty :empty :empty :wall  :empty]
           [:wall  :wall  :wall  :empty :wall  :empty]
           [:A     :empty :empty :empty :empty :B]])
(def utilities {:A 1.0 :B 5.0 :timeCost -0.1})

(def alpha-ladder [0.3 1.0 3.0 10.0 100.0 ##Inf])
(def noise-ladder [0.0 0.1 0.2 0.4])

(defonce ds (r/atom {:alpha-i 4 :noise-i 1 :frames []}))

;; The reusable player owns the frame index + autoplay; it reads our frames
;; through a thunk, so a resample is just (swap! ds ...) followed by :replay!.
(defonce player (views/make-trajectory-player {:frames-fn #(:frames @ds)}))

(defn- alpha-label [a] (if (= a ##Inf) "INF (optimal)" (str a)))

(defn- rollout!
  "Rebuild the MDP at the current noise, plan (value iteration -> Q), sample a
   softmax rollout, produce the frames, and rewind the player to frame 0."
  []
  (let [alpha  (nth alpha-ladder (:alpha-i @ds))
        noise  (nth noise-ladder (:noise-i @ds))
        mdp    (gw/build-mdp {:grid grid :utilities utilities :start [0 0]
                              :gamma 1.0 :noise noise})
        ag     (agent/make-mdp-agent {:mdp mdp :alpha alpha :gamma 1.0 :n-iters 40})
        roll   (agent/simulate-mdp ag (:start-idx mdp) 30)
        frames (pres/env->trajectory mdp roll (:V ag))]
    (swap! ds assoc :frames frames)
    ((:replay! player))))

(defn enter! [] (rollout!) ((:start! player)))

(defn leave! [] ((:stop! player)))

(defn on-key [k]
  (case k
    ;; r resamples a fresh path; alpha/noise also force a resample
    :replay (rollout!)
    :plus   (do (swap! ds update :alpha-i #(min (dec (count alpha-ladder)) (inc %))) (rollout!))
    :minus  (do (swap! ds update :alpha-i #(max 0 (dec %))) (rollout!))
    :noise  (do (swap! ds update :noise-i #(mod (inc %) (count noise-ladder))) (rollout!))
    ;; everything else (space stepping) belongs to the player
    ((:on-key player) k)))

(defn view []
  (let [{:keys [alpha-i noise-i frames]} @ds
        a (nth alpha-ladder alpha-i)
        e (nth noise-ladder noise-i)
        n (max 1 (count frames))
        i (min @(:index player) (dec n))]
    [:> Box {:flexDirection "column" :padding 1}
     [views/status-bar {:title "Ch 3: Gridworld MDP"
                        :status (str "alpha = " (alpha-label a)
                                     "    noise = " e
                                     "    step " i "/" (dec n))}]
     [(:view player)]
     [:> Text {:dimColor true}
      " [space] step  [r] resample  [+/-] alpha  [n] noise  [q/esc] menu"]
     [:> Text {:color "gray"}
      " @ agent   A/B goals (util 1 / 5)   block = wall   shading = value fn"]]))
