(ns ch3-demo
  "Ch 3 gridworld MDP demo — the end-to-end vertical slice.

   Real GenMLX inference drives the picture: tensor value iteration -> Q, a
   softmax-action rollout to a terminal, then the render-agnostic Frame producer.
   The agent then walks the sampled path in the terminal. +/- change the
   rationality alpha (alpha = infinity -> deterministic optimal); the path
   visibly changes. Pure data crosses the seam; this file only wires + renders.

   Demo state lives in one r/atom (the proven reactive model); the gallery routes
   keys here via on-key and starts/stops the autoplay timer via enter!/leave!."
  (:require ["ink" :refer [Text Box]]
            [reagent.core :as r]
            [agentmodels.gridworld :as gw]
            [agentmodels.agent :as agent]
            [agentmodels.presentation :as pres]
            [views]))

;; The demo world (top row first; screen coords).  A = util 1, B = util 5.
;;   A . .
;;   . # .
;;   . . B
(def grid [[:A     :empty :empty]
           [:empty :wall  :empty]
           [:empty :empty :B]])

(def mdp (gw/build-mdp {:grid grid
                        :utilities {:A 1.0 :B 5.0 :timeCost -0.1}
                        :start [1 0] :gamma 1.0}))

(def alpha-ladder [0.3 1.0 3.0 10.0 100.0 ##Inf])

(defonce ds (r/atom {:alpha-i 4 :idx 0 :frames [] :timer nil}))

(defn- alpha-label [a] (if (= a ##Inf) "INF (optimal)" (str a)))

(defn- rollout!
  "Recompute the agent + a fresh rollout at the current alpha; reset to step 0.
   This is the whole pipeline: value-iteration -> Q -> softmax rollout -> frames."
  []
  (let [a      (nth alpha-ladder (:alpha-i @ds))
        ag     (agent/make-mdp-agent {:mdp mdp :alpha a :gamma 1.0 :n-iters 24})
        roll   (agent/simulate-mdp ag (:start-idx mdp) 12)
        frames (pres/env->trajectory mdp roll (:V ag))]
    (swap! ds assoc :frames frames :idx 0)))

(defn- step! []
  (swap! ds update :idx #(min (dec (count (:frames @ds))) (inc %))))

(defn enter! []
  (rollout!)
  (when-not (:timer @ds)
    (swap! ds assoc :timer (js/setInterval step! 550))))

(defn leave! []
  (when-let [t (:timer @ds)] (js/clearInterval t) (swap! ds assoc :timer nil)))

(defn on-key [k]
  (case k
    :space  (step!)
    :replay (rollout!)
    :plus   (do (swap! ds update :alpha-i #(min (dec (count alpha-ladder)) (inc %))) (rollout!))
    :minus  (do (swap! ds update :alpha-i #(max 0 (dec %))) (rollout!))
    nil))

(defn view []
  (let [{:keys [alpha-i idx frames]} @ds
        a (nth alpha-ladder alpha-i)
        n (max 1 (count frames))
        i (min idx (dec n))
        f (when (seq frames) (nth frames i))]
    [:> Box {:flexDirection "column" :padding 1}
     [views/status-bar {:title "Ch 3: Gridworld MDP"
                        :status (str "alpha = " (alpha-label a) "    step " i "/" (dec n))}]
     (when f [views/grid-view f])
     [:> Text {:dimColor true}
      (str " action: " (or (some-> f :meta :action name) "-")
           "   [space] step  [r] resample  [+/-] alpha  [q/esc] menu")]
     [:> Text {:color "gray"}
      " @ agent   A/B goals (util 1 / 5)   block = wall   shading = value fn"]]))
