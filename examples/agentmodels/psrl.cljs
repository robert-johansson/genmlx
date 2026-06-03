(ns agentmodels.psrl
  "Posterior Sampling Reinforcement Learning on a gridworld (agentmodels Ch 3d).

   The agent knows the transition structure but NOT the reward: exactly one (free)
   cell of a 4×4 gridworld is rewarding, and the agent has a posterior over which.
   Each EPISODE it runs the canonical Osband PSRL loop:

     1. SAMPLE a reward model (a goal cell) from the posterior  (Thompson sampling)
     2. SOLVE that model optimally by value iteration → a policy
     3. ACT — run one episode following that policy, observing each visited cell's
        true reward
     4. UPDATE the posterior on the observations (exact Bayes)

   When the posterior is uncertain, sampling yields diverse goals → the agent visits
   diverse cells → it learns (explores); as the posterior concentrates, it exploits.
   Cumulative regret (V*_true(start) − achieved return per episode) decreases as the
   posterior concentrates on the true goal. (agentmodels plots regret only for its
   bandit example and merely visualizes gridworld-PSRL trajectories — the regret curve
   here is the faithful Osband-theory extension the bean asks for.)

   Reward model: per-cell reward applied each step over a finite horizon, with a 5th
   'stay' action so the agent can dwell on the goal — the clean realization of
   agentmodels' finite-horizon utility(state)=rewardGrid[loc]. The planner reuses
   agent/value-iteration (value-iteration ≡ agentmodels' recursive expectedUtility).
   The shipped multi-armed bandit (genmlx.agents.pomdp/make-bandit-agent, bandit_test
   18/18) already covers the chapter's Thompson/softmax-greedy bandit + bandit regret.

   Reuse: gw/parse-grid + gw/next-state (geometry), agent/value-iteration +
   make-mdp-agent + simulate-mdp (planning/rollout). Zero engine change."
  (:require [genmlx.mlx :as mx]
            [genmlx.agents.gridworld :as gw]
            [genmlx.agents.agent :as agent]))

;; 5 actions: left right up down STAY (the stay action lets the agent dwell on the
;; rewarding cell to collect its per-step reward over the finite horizon).
(def action-deltas [[-1 0] [1 0] [0 -1] [0 1] [0 0]])

;; agentmodels 4×4 Restaurant-free RL grid (top-first; '#' walls). Start [0,0]
;; (idx 0); the true reward sits in the bottom-right free cell (idx 15).
(def grid
  [[:empty :empty :wall  :empty]
   [:empty :empty :empty :empty]
   [:wall  :empty :wall  :wall]
   [:empty :empty :empty :empty]])

(def W 4)
(def START-IDX 0)
(def TRUE-GOAL 15)         ; bottom-right [x=3,y=3]
(def HORIZON 9)

(defn free-cells
  "Reachable (non-wall) cell indices of `grid`."
  [grid]
  (let [{:keys [S walls]} (gw/parse-grid grid)]
    (vec (remove walls (range S)))))

;; ===========================================================================
;; Per-cell-reward MDP (non-terminal, finite-horizon, 5-action, deterministic)
;; ===========================================================================

(defn reward-grid-mdp
  "MDP over `grid` whose reward is `reward` ([S] of per-cell rewards, applied each
   step — NOT terminal). Deterministic 5-action geometry (incl. stay). Finite-horizon
   value iteration accumulates reward over the horizon."
  [grid reward]
  (let [{:keys [W H S walls]} (gw/parse-grid grid)
        A       (count action-deltas)
        ns-fn   (fn [s a]
                  (let [x (mod s W) y (quot s W)
                        [dx dy] (nth action-deltas a)
                        nx (max 0 (min (dec W) (+ x dx)))
                        ny (max 0 (min (dec H) (+ y dy)))
                        n  (+ nx (* W ny))]
                    (if (contains? walls n) s n)))
        ns-rows (vec (for [s (range S)] (vec (for [a (range A)] (ns-fn s a)))))
        T       (mx/array (clj->js (vec (for [s (range S)]
                                          (vec (for [a (range A)]
                                                 (let [s' (get-in ns-rows [s a])]
                                                   (mapv #(if (= % s') 1.0 0.0) (range S))))))))
                          mx/float32)
        R       (mx/array (clj->js (vec (for [s (range S)] (vec (repeat A (double (nth reward s))))))) mx/float32)
        term    (mx/array (clj->js (vec (repeat S 0.0))) mx/float32)]
    (mx/eval! T R term)
    {:W W :H H :S S :A A :T T :R R :term term :terminals {} :walls walls
     :start-idx START-IDX :ns-fn ns-fn :action-kw [:left :right :up :down :stay] :gamma 1.0 :noise 0.0}))

(defn- one-hot [S i] (vec (for [s (range S)] (if (= s i) 1.0 0.0))))

(defn optimal-return
  "V*(start) over `horizon` steps under the true reward (the best achievable return)."
  [grid reward start horizon]
  (let [mdp (reward-grid-mdp grid reward)
        {:keys [V]} (agent/value-iteration (assoc mdp :gamma 1.0) ##Inf horizon)]
    (mx/item (mx/idx V start))))

;; ===========================================================================
;; Reward posterior (categorical over which free cell is the goal) + exact update
;; ===========================================================================

(defn uniform-posterior
  "Uniform one-hot prior over the free cells (exactly one is the goal)."
  [free]
  (let [p (/ 1.0 (count free))] (into {} (map (fn [c] [c p]) free))))

(defn update-posterior
  "Exact Bayes update on observing reward `r` at cell `c`: hypotheses inconsistent
   with the observation are dropped (h=c requires r=1; h≠c requires r=0) and the rest
   renormalized. Observing the reward (r=1 at c) collapses the posterior to c."
  [post c r]
  (let [keep (into {} (filter (fn [[h p]] (= (if (= h c) 1.0 0.0) (double r))) post))
        z    (reduce + (vals keep))]
    (if (pos? z) (into {} (map (fn [[h p]] [h (/ p z)]) keep)) post)))

;; deterministic LCG, so Thompson sampling is reproducible in tests
(defn- lcg [s] (mod (+ (* s 1103515245) 12345) 2147483648))
(defn- unit [s] (/ (lcg (lcg s)) 2147483648.0))

(defn sample-goal
  "Thompson sample: draw a goal cell from the posterior with seed `s`."
  [post s]
  (let [cells (vec (keys post)), u (unit s)]
    (loop [i 0, acc 0.0]
      (let [acc' (+ acc (post (nth cells i)))]
        (if (or (>= i (dec (count cells))) (< u acc')) (nth cells i) (recur (inc i) acc'))))))

;; ===========================================================================
;; The PSRL episode loop
;; ===========================================================================

(defn- argmax-first
  "Index of the first maximal entry (deterministic; no random tie-break)."
  [xs]
  (reduce (fn [best i] (if (> (nth xs i) (nth xs best)) i best)) 0 (range 1 (count xs))))

(defn- run-episode
  "Plan optimally on `sampled-goal` (value iteration), roll out one episode in the
   TRUE world with a DETERMINISTIC greedy policy (argmax over Q, first-index tie-break,
   noise-0 transitions), and return {:states :return} — the visited states and realized
   true return. Deterministic so a seed fully reproduces the PSRL run (the only
   randomness is the Thompson model-sample)."
  [grid true-reward sampled-goal horizon S]
  (let [mdp   (reward-grid-mdp grid (one-hot S sampled-goal))
        {:keys [Q]} (agent/value-iteration (assoc mdp :gamma 1.0) ##Inf horizon)
        Qh    (mx/->clj Q)
        ns-fn (:ns-fn mdp)
        states (loop [s START-IDX, step 0, acc [START-IDX]]
                 (if (>= step horizon)
                   acc
                   (let [a  (argmax-first (nth Qh s))
                         s' (ns-fn s a)]
                     (recur s' (inc step) (conj acc s')))))
        ret    (reduce + (map #(nth true-reward %) (take horizon states)))]
    {:states states :return ret}))

(defn psrl
  "Run PSRL for `n-episodes`. Returns per-episode {:sampled :posterior :return :regret}
   plus :cum-regret and :final-posterior. `learn?` false = no-learning baseline (keep
   sampling from the fixed prior, never update) — for comparison."
  [{:keys [grid true-goal horizon n-episodes seed learn?]
    :or {grid grid true-goal TRUE-GOAL horizon HORIZON n-episodes 10 seed 1 learn? true}}]
  (let [{:keys [S]} (gw/parse-grid grid)
        free        (free-cells grid)
        true-reward (one-hot S true-goal)
        v-star      (optimal-return grid true-reward START-IDX horizon)]
    (loop [ep 0, post (uniform-posterior free), eps []]
      (if (>= ep n-episodes)
        (let [regrets (map :regret eps)]
          {:episodes eps
           :cum-regret (vec (reductions + regrets))
           :final-posterior post
           :v-star v-star})
        (let [g   (sample-goal post (+ (* seed 1000) ep))
              {:keys [states return]} (run-episode grid true-reward g horizon S)
              post' (if learn?
                      (reduce (fn [b s] (update-posterior b s (nth true-reward s))) post states)
                      post)]
          (recur (inc ep) post'
                 (conj eps {:sampled g :return return :regret (- v-star return)
                            :posterior post' :reached-goal? (boolean (some #(= true-goal %) states))})))))))
