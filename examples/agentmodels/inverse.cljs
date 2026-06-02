(ns agentmodels.inverse
  "Inverse goal inference (agentmodels Ch 4/5): observe an agent's actions and
   infer which goal it values — by INVERTING the forward agent model through the
   GFI.

   The likelihood of an observed action under a hypothesized goal is *exactly*
   `p/assess` on that goal's softmax policy: no bespoke likelihood function, just
   the same generative policy the agent acts with, scored against the observed
   action. Bayesian updating is incremental, so the posterior can be revealed one
   observation at a time — which is what the live TUI viewer animates.

       P(goal | a_1..a_t) ∝ P(goal) · Π_i  exp( assess(policy_goal, s_i){:action a_i} )

   Soft rationality matters here: the policy must be a Boltzmann softmax (finite
   alpha), not a hard argmax, or an observed sub-optimal action would have zero
   likelihood and collapse the posterior (agentmodels' softmax assumption)."
  (:require [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [agentmodels.gridworld :as gw]
            [agentmodels.agent :as agent]))

(defn goal-agents
  "Build one agent per candidate goal: the agent that VALUES that goal gives it
   `high` utility and every other goal `low`. Same grid/noise for all. Returns an
   ordered map {goal -> agent} (agents carry :policy and :Q)."
  [{:keys [grid goals high low time-cost alpha noise gamma]
    :or   {high 5.0 low 0.0 time-cost -0.1 alpha 2.0 noise 0.0 gamma 1.0}}]
  (reduce
    (fn [m g]
      (let [utils (assoc (zipmap goals (repeat low)) g high :timeCost time-cost)
            mdp   (gw/build-mdp {:grid grid :utilities utils :start [0 0]
                                 :gamma gamma :noise noise})]
        (assoc m g (agent/make-mdp-agent {:mdp mdp :alpha alpha :gamma gamma :n-iters 40}))))
    {} goals))

(defn action-loglik
  "log P(action = a | state = s) under `agent`'s policy — the GFI assess weight.
   assess fully constrains :action, so no sampling happens; the auto-key is just
   to satisfy the handler and does not affect the (deterministic) score."
  [agent s a]
  (mx/item (:weight (p/assess (dyn/auto-key (:policy agent)) [s] (cm/choicemap :action a)))))

(defn- normalize-logs
  "{goal -> log-weight} -> {goal -> probability} (stable softmax)."
  [logm]
  (let [hi (apply max (vals logm))
        ex (into {} (map (fn [[g l]] [g (Math/exp (- l hi))]) logm))
        z  (reduce + (vals ex))]
    (into {} (map (fn [[g e]] [g (/ e z)]) ex))))

(defn posterior-sequence
  "Incremental Bayesian update. `observations` is a seq of [state action] pairs.
   Returns a vector of posterior maps {goal -> prob}, one per prefix length:
   index 0 is the prior, index k is the posterior after k observed actions."
  [goal-agents prior observations]
  (let [goals (keys goal-agents)]
    (loop [obs  observations
           logp (into {} (map (fn [g] [g (Math/log (prior g))]) goals))
           acc  [(normalize-logs logp)]]
      (if (empty? obs)
        acc
        (let [[s a] (first obs)
              logp' (into {} (map (fn [g] [g (+ (logp g) (action-loglik (goal-agents g) s a))]) goals))]
          (recur (rest obs) logp' (conj acc (normalize-logs logp'))))))))

(defn observe-rollout
  "Turn an agent rollout {:states :actions} into [state action] observation pairs."
  [{:keys [states actions]}]
  (mapv vector states actions))
