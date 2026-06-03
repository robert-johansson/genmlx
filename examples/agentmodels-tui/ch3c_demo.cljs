(ns ch3c-demo
  "Ch 3c POMDP belief filtering — the agent acts UNDER UNCERTAINTY about which
   goal pays, and we watch its belief update as it gathers evidence.

   A QMDP agent (belief-weighted over per-goal MDPs) walks a corridor toward two
   candidate goals. It does not know which is rewarding, so its P(world) bars sit
   flat at the prior while it crosses the corridor — then SNAP to the truth the
   instant it reaches the centre signpost, after which it commits to the right
   goal. Grid (left) and belief bars (right) advance in lockstep, exactly like the
   Ch5 viewer. Pure data crosses the seam; this file only wires + renders."
  (:require ["ink" :refer [Text Box]]
            [reagent.core :as r]
            [genmlx.agents.pomdp :as pomdp]
            [genmlx.agents.pomdp-env :as env]
            [genmlx.agents.presentation :as pres]
            [views]))

;;   A . B
;;   # . #
;;   # P #   P = signpost (idx 7): standing here reveals which goal is rewarding
;;   # . #
;;   # @ #   start (idx 13)
(def grid [[:A    :empty :B]
           [:wall :empty :wall]
           [:wall :empty :wall]
           [:wall :empty :wall]
           [:wall :empty :wall]])
(def signpost 7)
(def goals [:A :B])
(def alpha 8.0)            ; decisive (still soft) — the belief story, not alpha, is the point

(defonce ds (r/atom {:true-world :A :idx 0 :frames [] :beliefs [] :timer nil}))

(defn- regen!
  "Build the POMDP agent + a fresh belief-filtered rollout for the current true
   world; produce the grid Frames and the per-step beliefs (both from one roll)."
  []
  (let [e      (env/restaurant-gridworld {:grid grid :goals goals :signpost signpost
                                          :true-world (:true-world @ds) :start [1 4]})
        pa     (pomdp/make-pomdp-agent (assoc e :alpha alpha :gamma 1.0 :n-iters 40))
        roll   (pomdp/simulate-pomdp pa e (:start-idx e) 20)
        wa     ((:world-agents pa) (:true-world @ds))
        frames (pres/env->trajectory (:mdp wa) roll (:V wa))]
    (swap! ds assoc :frames frames :beliefs (:beliefs roll) :idx 0)))

(defn- step! []
  (swap! ds update :idx #(min (dec (count (:frames @ds))) (inc %))))

(defn enter! []
  (regen!)
  (when-not (:timer @ds)
    (swap! ds assoc :timer (js/setInterval step! 600))))

(defn leave! []
  (when-let [t (:timer @ds)] (js/clearInterval t) (swap! ds assoc :timer nil)))

(defn on-key [k]
  (case k
    :space  (step!)
    :replay (regen!)
    :toggle (do (swap! ds update :true-world {:A :B :B :A}) (regen!))
    nil))

(defn view []
  (let [{:keys [true-world idx frames beliefs]} @ds
        n (max 1 (count frames))
        i (min idx (dec n))
        f (when (seq frames) (nth frames i))
        b (when (seq beliefs) (nth beliefs (min i (dec (count beliefs)))))]
    [:> Box {:flexDirection "column" :padding 1}
     [views/status-bar {:title "Ch 3c: POMDP belief filtering"
                        :status (str "true world = " (name true-world) "    step " i "/" (dec n))}]
     [:> Box {:flexDirection "row"}
      (when f [:> Box {:marginRight 4} [views/grid-view f]])
      (when b [:> Box {} [views/bars-view (pres/dist->bars "P(world)" b true-world) 16]])]
     [:> Text {:dimColor true}
      " [space] step   [r] resample   [t] toggle true world   [q/esc] menu"]
     [:> Text {:color "gray"}
      " belief stays flat until the agent reaches the centre signpost, then snaps to the truth"]]))
