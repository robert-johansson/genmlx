(ns genmlx.agents.pomdp-env
  "POMDP environments (bean genmlx-m9m9). Each constructor returns the data
   bundle make-pomdp-agent / simulate-pomdp consume. The per-world MDP tensors
   themselves are produced inside make-pomdp-agent via gridworld/build-mdp (from
   inverse/goal-agents, or per-world :world-utils), so the env stays pure
   geometry/config.

   Shipped here: the hidden-goal RESTAURANT GRIDWORLD (latent = which goal pays;
   single-signpost reveal), the ADJACENCY-REVEAL restaurant POMDP (latent = a
   per-restaurant open/closed vector; agentmodels' makeGridWorldPOMDP), and the
   multi-armed BANDIT (latent = per-arm payoff parameters)."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.agents.gridworld :as gw]))

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

(defn- neighbors-4
  "In-bounds 4-neighbour cell indices of `idx` on a W×H grid."
  [W H idx]
  (let [x (mod idx W), y (quot idx W)]
    (set (for [[dx dy] [[-1 0] [1 0] [0 -1] [0 1]]
               :let [nx (+ x dx), ny (+ y dy)]
               :when (and (<= 0 nx (dec W)) (<= 0 ny (dec H)))]
           (+ nx (* W ny))))))

(defn- open-closed-configs
  "All {restaurant -> open?} maps over `restaurants` (2^k configs)."
  [restaurants]
  (reduce (fn [acc r] (mapcat (fn [m] [(assoc m r true) (assoc m r false)]) acc))
          [{}] restaurants))

(defn restaurant-pomdp
  "Adjacency-reveal restaurant POMDP (agentmodels Ch 3c makeGridWorldPOMDP). Unlike
   the single-signpost restaurant-gridworld above, EACH restaurant is independently
   OPEN or CLOSED — the latent is a per-restaurant vector — and the agent learns a
   restaurant's status only when ADJACENT to it (a 4-neighbour of its cell). The
   belief is over the 2^k open/closed configs; an OPEN restaurant pays its utility,
   a CLOSED one pays 0 (a dead terminal the agent avoids). So the agent heads to the
   preferred restaurant and, on observing it CLOSED when adjacent, re-plans toward
   the backup — the canonical local-observation POMDP behaviour.

   Options:
     :grid       — restaurant cells are terminal keywords (e.g. :a :b);
     :utilities  — {restaurant -> open-utility} (default {:a 5.0 :b 3.0});
     :open-prob  — {restaurant -> P(open)} independent prior (default {:a 0.6 :b 0.9});
     :true-world — {restaurant -> open?} the actual config;
     :time-cost  — per-step cost (default -0.1);  :start — [x y].

   Returns the make-pomdp-agent/simulate-pomdp bundle, carrying :world-utils (so
   make-pomdp-agent builds one MDP per open/closed config) and an adjacency :observe."
  [{:keys [grid utilities open-prob true-world time-cost start]
    :or   {utilities {:a 5.0 :b 3.0} open-prob {:a 0.6 :b 0.9} time-cost -0.1 start [0 0]}}]
  (let [{:keys [W H terminals]} (gw/parse-grid grid)
        restaurants (vec (keys utilities))
        cell-of     (into {} (map (fn [[idx kw]] [kw idx])) terminals)   ; restaurant -> cell idx
        adj         (into {} (map (fn [r] [r (neighbors-4 W H (cell-of r))])) restaurants)
        worlds      (open-closed-configs restaurants)
        world-utils (into {} (map (fn [w]
                                    [w (assoc (into {} (map (fn [r] [r (if (w r) (utilities r) 0.0)])) restaurants)
                                              :timeCost time-cost)]))
                          worlds)
        prior       (into {} (map (fn [w]
                                    [w (reduce * (map (fn [r] (if (w r) (open-prob r) (- 1.0 (open-prob r)))) restaurants))]))
                          worlds)
        [sx sy]     start]
    {:kind        :restaurant-pomdp
     :grid        grid
     :restaurants restaurants
     :worlds      worlds
     :world-utils world-utils
     :true-world  true-world
     :prior       prior
     :start       start
     :start-idx   (+ sx (* W sy))
     ;; adjacency-gated local reveal: observe the open/closed status of every
     ;; restaurant the agent is currently adjacent to (nil if none).
     :observe     (fn [world loc]
                    (not-empty (vec (for [r restaurants :when (contains? (adj r) loc)] [r (world r)]))))}))

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
