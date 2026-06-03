(ns genmlx.agents.pomdp-env
  "POMDP environments (bean genmlx-m9m9). Each constructor returns the data
   bundle make-pomdp-agent / simulate-pomdp consume. The per-world MDP tensors
   themselves are produced inside make-pomdp-agent via inverse/goal-agents ->
   gridworld/build-mdp, so the env stays pure geometry/config.

   Shipped here: the hidden-goal RESTAURANT GRIDWORLD (latent = which goal pays)
   and the multi-armed BANDIT (latent = per-arm payoff parameters)."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]))

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

(defn bandit-pomdp
  "Multi-armed bandit POMDP (agentmodels Ch 3c/3d). The hidden state is the
   per-arm payoff parameter theta_i (FIXED for the episode); pulling arm i is the
   action AND the only observation channel for theta_i; the reward r ~
   Bernoulli(theta_i) IS the observation; belief factorizes into one independent
   Beta(alpha_i, beta_i) per arm (Beta-Bernoulli conjugacy, filtered host-side in
   genmlx.agents.pomdp). Returns the bundle make-bandit-agent / simulate-bandit
   consume — pure config plus a reward sampler; no MDP tensors (no spatial latent).

   Options: :thetas [p ...] true Bernoulli params (latent, fixed);
            :prior [[a b] ...] per-arm Beta priors (default Beta(1,1) each);
            :horizon H."
  [{:keys [thetas prior horizon] :or {horizon 30}}]
  (let [k     (count thetas)
        thv   (vec thetas)
        prior (or prior (vec (repeat k [1.0 1.0])))]
    {:kind      :bandit
     :arms      k
     :thetas    thv
     :true-best (apply max-key thv (range k))      ; index of the highest-theta arm
     :theta*    (apply max thetas)
     :prior     {:arms prior}
     :horizon   horizon
     ;; reward = observation: pulling arm i samples r ~ Bernoulli(theta_i).
     :pull      (fn [i key]
                  (int (mx/item (dist/sample (dist/bernoulli (nth thv i)) key))))}))
