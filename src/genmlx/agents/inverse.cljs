(ns genmlx.agents.inverse
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
            [genmlx.agents.gridworld :as gw]
            [genmlx.agents.agent :as agent]))

(defn goal-agents
  "Build one agent per candidate goal: the agent that VALUES that goal gives it
   `high` utility and every other goal `low`. Same grid/noise for all. Returns an
   ordered map {goal -> agent} (agents carry :policy and :Q).

   `:fixed` is a map of utilities merged into EVERY hypothesis (terrain that is
   known and shared, not part of the inferred preference) — e.g. a hiking Hill
   cliff {:hill -40} so all peak-preference hypotheses still avoid the cliff.
   `:start` only affects rollout, not the policy/Q used for action-loglik.
   `:n-iters` is the value-iteration horizon (default 40); pass it to keep the
   inverted agents at the same horizon as a sibling differentiable/forward agent."
  [{:keys [grid goals high low time-cost alpha noise gamma fixed start n-iters]
    :or   {high 5.0 low 0.0 time-cost -0.1 alpha 2.0 noise 0.0 gamma 1.0 fixed {} start [0 0]
           n-iters 40}}]
  (reduce
    (fn [m g]
      (let [utils (merge (assoc (zipmap goals (repeat low)) g high :timeCost time-cost) fixed)
            mdp   (gw/build-mdp {:grid grid :utilities utils :start start
                                 :gamma gamma :noise noise})]
        (assoc m g (agent/make-mdp-agent {:mdp mdp :alpha alpha :gamma gamma :n-iters n-iters}))))
    {} goals))

(defn action-loglik
  "log P(action = a | state = s) under `agent`'s policy — the GFI assess weight.
   assess fully constrains :action, so no sampling happens; the auto-key is just
   to satisfy the handler and does not affect the (deterministic) score."
  [agent s a]
  (mx/item (:weight (p/assess (dyn/auto-key (:policy agent)) [s] (cm/choicemap :action a)))))

(defn normalize-logs
  "{goal -> log-weight} -> {goal -> probability} (stable softmax). Public so the
   POMDP belief filter (genmlx.agents.pomdp) reuses the same normalization."
  [logm]
  (let [hi (apply max (vals logm))
        ex (into {} (map (fn [[g l]] [g (Math/exp (- l hi))]) logm))
        z  (reduce + (vals ex))]
    (into {} (map (fn [[g e]] [g (/ e z)]) ex))))

(defn- stack-log-policies
  "Stack the goal policies into one [G,S,A] log-action-probability tensor (bean
   genmlx-y2hh). LOGP[g,s,a] = log_softmax(alpha · Q_g[s,:])[a] — which is EXACTLY
   action-loglik(agent_g,s,a): the agent's policy is Categorical(alpha·Q_g[s,:])
   (agent.cljs make-mdp-agent), and the categorical assess weight is dist's
   logits->logprobs = logits - logsumexp(logits, last-axis). Computing it for all
   goals/states at once lets each observed [s,a] be scored against EVERY goal in
   one gather, with no per-cell mx/item. Requires a finite, shared alpha (a hard
   argmax would give -inf likelihoods for sub-optimal actions and collapse the
   posterior — the soft-rationality invariant of inverse inference)."
  [goals goal-agents]
  (let [alpha (:alpha (:params (goal-agents (first goals))))]
    (assert (and (number? alpha) (js/isFinite alpha))
            "posterior-sequence: batched inverse inference needs a finite alpha (hard argmax is degenerate)")
    (doseq [g (rest goals)]
      (assert (== alpha (:alpha (:params (goal-agents g))))
              "posterior-sequence: all goal agents must share one alpha to batch the policy stack"))
    (let [qstack (mx/stack (mapv #(:Q (goal-agents %)) goals))          ; [G,S,A]
          logits (mx/multiply (mx/scalar alpha) qstack)                 ; [G,S,A]
          logp   (mx/subtract logits (mx/expand-dims (mx/logsumexp logits [-1]) -1))]
      (mx/materialize! logp)                                            ; eval the stack once
      logp)))

(defn- softmax->map
  "Stable tensor softmax of a [G] log-weight vector -> {goal -> prob} map. Same
   max-shift recipe as normalize-logs, in tensor space (exp(-inf)=0 for zero-prior
   goals); the single host extraction per prefix posterior."
  [goals logp]
  (let [e (mx/exp (mx/subtract logp (mx/amax logp)))]
    (zipmap goals (mx/->clj (mx/divide e (mx/sum e))))))

(defn posterior-sequence
  "Incremental Bayesian update. `observations` is a seq of [state action] pairs.
   Returns a vector of posterior maps {goal -> prob}, one per prefix length:
   index 0 is the prior, index k is the posterior after k observed actions.

   Shape-batched over the goal axis (bean genmlx-y2hh): the goal policies are
   stacked into one [G,S,A] log-prob tensor and each observation is scored against
   ALL goals at once, accumulating into a single [G] log-weight tensor. Extraction
   is one [G] readout per prefix posterior (T+1 total), not G·T per-cell mx/item.
   Numerically identical (to float32) to the per-cell action-loglik path, which is
   retained above as the ground truth."
  [goal-agents prior observations]
  (let [goals (vec (keys goal-agents))                 ; pin order: same axis for stack + readout
        logp-policies (stack-log-policies goals goal-agents)
        logp0 (mx/log (mx/array (clj->js (mapv #(double (prior %)) goals)) mx/float32))]
    (loop [obs  observations
           logp logp0
           acc  [(softmax->map goals logp0)]]
      (if (empty? obs)
        acc
        (let [[s a] (first obs)
              cell  (mx/idx (mx/idx logp-policies s 1) a 1)   ; [G,S,A] -> [G,A] -> [G]
              logp' (mx/add logp cell)]
          (recur (rest obs) logp' (conj acc (softmax->map goals logp'))))))))

(defn observe-rollout
  "Turn an agent rollout {:states :actions} into [state action] observation pairs."
  [{:keys [states actions]}]
  (mapv vector states actions))
