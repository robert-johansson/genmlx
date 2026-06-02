(ns agentmodels.pomdp-env
  "POMDP environments (bean genmlx-m9m9). Each constructor returns the data
   bundle make-pomdp-agent / simulate-pomdp consume. The per-world MDP tensors
   themselves are produced inside make-pomdp-agent via inverse/goal-agents ->
   gridworld/build-mdp, so the env stays pure geometry/config.

   Shipped here: the hidden-goal RESTAURANT GRIDWORLD (the slice). The multi-armed
   BANDIT (conjugate Beta/Normal belief filtering) is the planned second env and
   a follow-up — noted at the bottom — kept out of the slice to avoid coupling it
   to the conjugate-update API before that demo is built."
  (:require [genmlx.mlx :as mx]))

(defn restaurant-gridworld
  "Hidden-goal POMDP: a grid with candidate goals, exactly one of which is the
   true rewarding restaurant (the latent world). The agent learns which only by
   STANDING ON the signpost cell — elsewhere observe -> nil, so belief stays at
   the prior (flat-then-snap). Pure geometry; the per-world MDP tensors come from
   gridworld/build-mdp via inverse/goal-agents inside make-pomdp-agent.

   Options: :grid :goals :signpost (state idx) :true-world (a goal keyword)
            :start [x y]. Returns the env bundle + an :observe model."
  [{:keys [grid goals signpost true-world start] :or {goals [:A :B] start [0 0]}}]
  (let [W (count (first grid))
        [sx sy] start]
    {:kind       :gridworld
     :grid       grid
     :goals      goals
     :worlds     goals                 ; latent = which goal is rewarding
     :true-world true-world
     :signpost   signpost
     :start-idx  (+ sx (* W sy))
     :prior      (zipmap goals (repeat (/ 1.0 (count goals))))
     ;; deterministic location-gated reveal: standing on the signpost reveals the
     ;; world identity; everywhere else there is no information (obs = nil).
     :observe    (fn [world loc] (when (= loc signpost) world))}))

;; ---------------------------------------------------------------------------
;; FOLLOW-UP (genmlx-m9m9): multi-armed bandit POMDP — latent = each arm's payoff
;; parameter; belief = independent conjugate posteriors per arm (Beta-Bernoulli /
;; Normal-Normal via genmlx.inference.conjugate). Pulling an arm reveals its prize
;; and updates only that arm's posterior. A separate visual idiom (arm bars rather
;; than a grid), so it ships after the gridworld slice rather than gating it.
;; ---------------------------------------------------------------------------
