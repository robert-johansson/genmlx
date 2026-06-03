(ns ch3d-demo
  "Ch 3d Bandits — belief filtering with no spatial state.

   Three one-armed bandits with unknown payoff probabilities; the agent keeps a
   Beta posterior per arm, pulls an arm (Thompson posterior sampling), sees the
   reward, and conjugately updates that arm's belief. The per-arm posterior-mean
   bars (true-best arm highlighted) sharpen as pulls concentrate on the best arm
   and cumulative regret flattens. `t` toggles Thompson <-> softmax-greedy.

   Pure data crosses the seam (bandit-bars -> bars-view); this file only wires +
   renders. (Thompson draws are exact Beta via dist/beta-dist, now backed by the
   stable gamma-ratio sampler — genmlx-gcw4.)"
  (:require ["ink" :refer [Text Box]]
            [reagent.core :as r]
            [genmlx.agents.pomdp :as pomdp]
            [genmlx.agents.pomdp-env :as env]
            [genmlx.agents.presentation :as pres]
            [views]))

(def thetas [0.25 0.50 0.80])         ; true arm payoff probs; arm 2 is best
(def horizon 30)

(defonce ds (r/atom {:strategy :thompson :idx 0 :roll nil :env nil :ag nil :timer nil}))

(defn- regen!
  "A fresh bandit + a full Thompson/softmax rollout at the current strategy."
  []
  (let [e    (env/bandit-pomdp {:thetas thetas :horizon horizon})
        ag   (pomdp/make-bandit-agent {:strategy (:strategy @ds)})
        roll (pomdp/simulate-bandit ag e)]   ; fresh key each regen -> a new run on `r`
    (swap! ds assoc :roll roll :idx 0 :env e :ag ag)))

(defn- step! []
  (swap! ds update :idx #(min (dec (count (:beliefs (:roll @ds)))) (inc %))))

(defn enter! []
  (regen!)
  (when-not (:timer @ds)
    (swap! ds assoc :timer (js/setInterval step! 450))))

(defn leave! []
  (when-let [t (:timer @ds)] (js/clearInterval t) (swap! ds assoc :timer nil)))

(defn on-key [k]
  (case k
    :space  (step!)
    :replay (regen!)
    :toggle (do (swap! ds update :strategy {:thompson :softmax :softmax :thompson}) (regen!))
    nil))

(defn view []
  (let [{:keys [strategy idx roll env]} @ds
        bs   (:beliefs roll)
        n    (max 1 (count bs))
        i    (min idx (dec n))
        b    (nth bs i)
        cum  (nth (:cum-reward roll) (max 0 (dec i)) 0)
        reg  (nth (:regret roll) (max 0 (dec i)) 0)
        pull (when (pos? i) (nth (:arms roll) (dec i)))]
    [:> Box {:flexDirection "column" :padding 1}
     [views/status-bar {:title "Ch 3d: Bandits (posterior sampling)"
                        :status (str "strategy=" (name strategy) "    step " i "/" (dec n)
                                     "    reward " cum "    regret " (.toFixed reg 2))}]
     (when b [views/bars-view (pres/bandit-bars b (:true-best env)) 20])
     [:> Text {:dimColor true}
      (str " pulled arm " (if (some? pull) pull "-")
           "    counts " (into (sorted-map) (frequencies (take i (:arms roll)))))]
     [:> Text {:dimColor true}
      " [space] step   [r] resample   [t] thompson<->softmax   [q/esc] menu"]
     [:> Text {:color "gray"}
      " bars = per-arm posterior means; green ◄true is arm 2 (theta=0.8) — pulls concentrate as belief sharpens"]]))
