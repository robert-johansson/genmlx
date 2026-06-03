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
   keys here via on-key and starts/stops the autoplay timer via enter!/leave!."
  (:require ["ink" :refer [Text Box]]
            [reagent.core :as r]
            [agentmodels.gridworld :as gw]
            [agentmodels.agent :as agent]
            [agentmodels.presentation :as pres]
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

(defonce ds (r/atom {:alpha-i 4 :noise-i 1 :idx 0 :frames [] :timer nil}))

(defn- alpha-label [a] (if (= a ##Inf) "INF (optimal)" (str a)))

(defn- rollout!
  "Rebuild the MDP at the current noise, plan (value iteration -> Q), sample a
   softmax rollout, and produce the frames. The whole pipeline, in five lines."
  []
  (let [alpha  (nth alpha-ladder (:alpha-i @ds))
        noise  (nth noise-ladder (:noise-i @ds))
        mdp    (gw/build-mdp {:grid grid :utilities utilities :start [0 0]
                              :gamma 1.0 :noise noise})
        ag     (agent/make-mdp-agent {:mdp mdp :alpha alpha :gamma 1.0 :n-iters 40})
        roll   (agent/simulate-mdp ag (:start-idx mdp) 30)
        frames (pres/env->trajectory mdp roll (:V ag))]
    (swap! ds assoc :frames frames :idx 0)))

(defn- step! []
  (swap! ds update :idx #(min (dec (count (:frames @ds))) (inc %))))

(defn enter! []
  (rollout!)
  (when-not (:timer @ds)
    (swap! ds assoc :timer (js/setInterval step! 450))))

(defn leave! []
  (when-let [t (:timer @ds)] (js/clearInterval t) (swap! ds assoc :timer nil)))

(defn on-key [k]
  (case k
    :space  (step!)
    :replay (rollout!)
    :plus   (do (swap! ds update :alpha-i #(min (dec (count alpha-ladder)) (inc %))) (rollout!))
    :minus  (do (swap! ds update :alpha-i #(max 0 (dec %))) (rollout!))
    :noise  (do (swap! ds update :noise-i #(mod (inc %) (count noise-ladder))) (rollout!))
    nil))

(defn view []
  (let [{:keys [alpha-i noise-i idx frames]} @ds
        a (nth alpha-ladder alpha-i)
        e (nth noise-ladder noise-i)
        n (max 1 (count frames))
        i (min idx (dec n))
        f (when (seq frames) (nth frames i))]
    [:> Box {:flexDirection "column" :padding 1}
     [views/status-bar {:title "Ch 3: Gridworld MDP"
                        :status (str "alpha = " (alpha-label a)
                                     "    noise = " e
                                     "    step " i "/" (dec n))}]
     (when f [views/grid-view f])
     [:> Text {:dimColor true}
      (str " action: " (or (some-> f :meta :action name) "-")
           "   [space] step  [r] resample  [+/-] alpha  [n] noise  [q/esc] menu")]
     [:> Text {:color "gray"}
      " @ agent   A/B goals (util 1 / 5)   block = wall   shading = value fn"]]))
