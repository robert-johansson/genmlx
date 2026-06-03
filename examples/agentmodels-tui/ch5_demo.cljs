(ns ch5-demo
  "Ch 5 inverse goal inference — the live posterior viewer.

   Watch GenMLX infer what an agent WANTS from how it moves. A 'true' agent
   (valuing A or B) walks the grid; the observer maintains a posterior over the
   agent's goal and updates it after every observed action by inverting the
   forward model through the GFI (agentmodels.inverse uses p/assess as the
   likelihood). As the walk plays, the grid (left) and the P(goal) bars (right)
   advance in lockstep — the bars are uninformative while the agent heads down
   the symmetry axis, then snap to the true goal the moment it commits.

   space steps, r resamples a fresh walk, t toggles the true goal."
  (:require ["ink" :refer [Text Box]]
            [reagent.core :as r]
            [agentmodels.gridworld :as gw]
            [agentmodels.agent :as agent]
            [agentmodels.inverse :as inv]
            [agentmodels.presentation :as pres]
            [views]))

;; Open grid, A and B mirror-symmetric about the centre column; start top-centre.
;;   . . . . .
;;   . . . . .
;;   A . . . B
(def grid [[:empty :empty :empty :empty :empty]
           [:empty :empty :empty :empty :empty]
           [:A     :empty :empty :empty :B]])
(def goals [:A :B])
(def start [2 0])            ; top-centre
(def alpha 2.0)             ; soft rationality (shared by the true agent + observer)
(def prior {:A 0.5 :B 0.5})

(defonce ds (r/atom {:true-goal :B :idx 0 :frames [] :posts [] :timer nil}))

(defn- regen!
  "Build per-goal agents, roll out the TRUE goal's agent, and compute the
   incremental posterior over goals from the observed actions."
  []
  (let [agents (inv/goal-agents {:grid grid :goals goals :alpha alpha})
        truth  (:true-goal @ds)
        mdp    (:mdp (agents truth))
        roll   (agent/simulate-mdp (agents truth) (+ (first start) (* (:W mdp) (second start))) 20)
        obs    (inv/observe-rollout roll)
        posts  (inv/posterior-sequence agents prior obs)
        frames (pres/env->trajectory mdp roll (:V (agents truth)))]
    (swap! ds assoc :frames frames :posts posts :idx 0)))

(defn- step! []
  (swap! ds update :idx #(min (dec (count (:frames @ds))) (inc %))))

(defn enter! []
  (regen!)
  (when-not (:timer @ds)
    (swap! ds assoc :timer (js/setInterval step! 650))))

(defn leave! []
  (when-let [t (:timer @ds)] (js/clearInterval t) (swap! ds assoc :timer nil)))

(defn on-key [k]
  (case k
    :space  (step!)
    :replay (regen!)
    :toggle (do (swap! ds update :true-goal {:A :B :B :A}) (regen!))
    nil))

(defn view []
  (let [{:keys [true-goal idx frames posts]} @ds
        n (max 1 (count frames))
        i (min idx (dec n))
        f (when (seq frames) (nth frames i))
        post (when (seq posts) (nth posts (min i (dec (count posts)))))]
    [:> Box {:flexDirection "column" :padding 1}
     [views/status-bar {:title "Ch 5: Inverse goal inference"
                        :status (str "true goal = " (name true-goal)
                                     "    observed " i "/" (dec n) " actions")}]
     [:> Box {:flexDirection "row"}
      (when f [:> Box {:marginRight 4} [views/grid-view f]])
      (when post [:> Box {} [views/bars-view (pres/dist->bars "P(goal)" post true-goal) 18]])]
     [:> Text {:dimColor true}
      " [space] step   [r] resample walk   [t] toggle true goal   [q/esc] menu"]
     [:> Text {:color "gray"}
      " the observer sees only moves — P(goal) sharpens as the agent commits"]]))
