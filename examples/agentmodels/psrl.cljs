(ns agentmodels.psrl
  "Posterior Sampling Reinforcement Learning on a gridworld (agentmodels Ch 3d).

   The agent knows the transitions but not the reward (one free cell is rewarding);
   each episode it Thompson-samples a goal from the posterior, solves that model by
   value iteration, acts greedily, and exactly Bayes-updates on the observed cells.
   Cumulative regret decreases as the posterior concentrates on the true goal."
  ;; When the posterior is uncertain, sampling yields diverse goals -> the agent
  ;; visits diverse cells -> it explores; as the posterior concentrates, it exploits.
  ;; Regret = V*_true(start) - achieved return per episode. (agentmodels plots regret
  ;; only for its bandit example and merely visualizes gridworld-PSRL trajectories;
  ;; the regret curve here is the faithful Osband-theory extension the bean asks for.)
  ;;
  ;; Reward model: per-cell reward applied each step over a finite horizon, with a 5th
  ;; 'stay' action so the agent can dwell on the goal -- the clean realization of
  ;; agentmodels' finite-horizon utility(state)=rewardGrid[loc]. The planner reuses
  ;; agent/value-iteration (value-iteration == agentmodels' recursive expectedUtility).
  ;; The shipped multi-armed bandit (genmlx.agents.pomdp/make-bandit-agent, bandit_test
  ;; 18/18) already covers the chapter's Thompson/softmax-greedy bandit + bandit regret.
  ;;
  ;; Reuse: gw/parse-grid + gw/next-state (geometry), agent/value-iteration +
  ;; make-mdp-agent + simulate-mdp (planning/rollout). Zero engine change.
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
;; Per-cell-reward MDP (non-terminal, finite-horizon, 5-action)
;; ===========================================================================

;; Orthogonal slip for the 5-action grid: left/right slip to up/down and vice
;; versa; the stay action (4) never slips. (Same convention as gridworld.cljs.)
(def ^:private perp {0 [2 3] 1 [2 3] 2 [0 1] 3 [0 1]})

(defn- t-row
  "Transition distribution [S] for (s,a): mass (1-noise) on the intended next
   state and noise/2 on each orthogonal slip target (no slip for stay or noise 0).
   Slipping into a wall/edge keeps the agent put, since ns-rows encodes that."
  [S ns-rows s a noise]
  (let [mass (if (and (pos? noise) (perp a))
               (let [[p q] (perp a)]
                 (merge-with +
                   {(get-in ns-rows [s a]) (- 1.0 noise)}
                   {(get-in ns-rows [s p]) (* 0.5 noise)}
                   {(get-in ns-rows [s q]) (* 0.5 noise)}))
               {(get-in ns-rows [s a]) 1.0})]
    (mapv #(get mass % 0.0) (range S))))

(defn reward-grid-mdp
  "MDP over `grid` whose reward is `reward` ([S] of per-cell rewards, applied each
   step — NOT terminal). 5-action geometry (incl. stay); with `noise` > 0 each
   non-stay action slips orthogonally with probability `noise`. Finite-horizon
   value iteration accumulates reward over the horizon."
  ([grid reward] (reward-grid-mdp grid reward 0.0))
  ([grid reward noise]
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
                                           (vec (for [a (range A)] (t-row S ns-rows s a noise))))))
                           mx/float32)
         R       (mx/array (clj->js (vec (for [s (range S)] (vec (repeat A (nth reward s)))))) mx/float32)
         term    (mx/array (clj->js (vec (repeat S 0.0))) mx/float32)]
     (mx/eval! T R term)
     {:W W :H H :S S :A A :T T :R R :term term :terminals {} :walls walls
      :start-idx START-IDX :ns-fn ns-fn :action-kw [:left :right :up :down :stay] :gamma 1.0 :noise noise})))

(defn- one-hot [S i] (vec (for [s (range S)] (if (= s i) 1.0 0.0))))

(defn optimal-return
  "V*(start) over `horizon` steps under the true reward (the best achievable
   expected return). Pass `noise` for the stochastic variant."
  ([grid reward start horizon] (optimal-return grid reward start horizon 0.0))
  ([grid reward start horizon noise]
   (let [mdp (reward-grid-mdp grid reward noise)
         {:keys [V]} (agent/value-iteration (assoc mdp :gamma 1.0) ##Inf horizon)]
     (mx/item (mx/idx V start)))))

;; ===========================================================================
;; Reward posterior (categorical over which free cell is the goal) + exact update
;; ===========================================================================

(defn uniform-posterior
  "Uniform one-hot prior over the free cells (exactly one is the goal)."
  [free]
  (let [p (/ 1.0 (count free))] (zipmap free (repeat p))))

(defn update-posterior
  "Exact Bayes update on observing reward `r` at cell `c`: hypotheses inconsistent
   with the observation are dropped (h=c requires r=1; h≠c requires r=0) and the rest
   renormalized. Observing the reward (r=1 at c) collapses the posterior to c."
  [post c r]
  (let [keep (into {} (filter (fn [[h _]] (= (if (= h c) 1.0 0.0) (double r))) post))
        z    (reduce + (vals keep))]
    (if (pos? z) (update-vals keep #(/ % z)) post)))

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

(defn- sample-cat
  "Index sampled from categorical `probs` using uniform `u` ∈ [0,1)."
  [probs u]
  (loop [i 0, acc 0.0]
    (let [acc' (+ acc (nth probs i))]
      (if (or (>= i (dec (count probs))) (< u acc')) i (recur (inc i) acc')))))

(defn- run-episode
  "Plan optimally on `plan-reward` (value iteration over the same `noise` MDP the
   agent acts in), roll out one episode with a greedy policy (argmax over Q,
   first-index tie-break), and return {:states :return} scored under `true-reward`.
   Transitions are deterministic when noise = 0, else sampled from the noisy T with
   a `seed`-driven LCG — so the whole PSRL run stays reproducible."
  [grid plan-reward true-reward horizon S noise seed]
  (let [mdp   (reward-grid-mdp grid plan-reward noise)
        {:keys [Q]} (agent/value-iteration (assoc mdp :gamma 1.0) ##Inf horizon)
        Qh    (mx/->clj Q)
        Th    (when (pos? noise) (mx/->clj (:T mdp)))
        ns-fn (:ns-fn mdp)
        states (loop [s START-IDX, step 0, acc [START-IDX], r (lcg (+ (* seed 31) 7))]
                 (if (>= step horizon)
                   acc
                   (let [a  (argmax-first (nth Qh s))
                         [s' r'] (if (pos? noise)
                                   [(sample-cat (nth (nth Th s) a) (/ r 2147483648.0)) (lcg r)]
                                   [(ns-fn s a) r])]
                     (recur s' (inc step) (conj acc s') r'))))
        ret    (reduce + (map #(nth true-reward %) (take horizon states)))]
    {:states states :return ret}))

(defn combined-reward
  "Per-cell reward [S]: +1 on the `goal`, `penalty` on each known `lava` cell, 0
   elsewhere. Lava is a known feature shared by every reward hypothesis; only the
   +1 goal is the latent the posterior is over."
  [S lava penalty goal]
  (mapv (fn [s] (cond (= s goal) 1.0
                      (contains? lava s) (double penalty)
                      :else 0.0))
        (range S)))

(defn psrl
  "Run PSRL for `n-episodes`. Returns per-episode {:sampled :posterior :return :regret}
   plus :cum-regret, :final-posterior, :v-star. `learn?` false = no-learning baseline
   (keep sampling from the fixed prior, never update) — for comparison.

   Goal+lava variant (genmlx.agents.worlds/lava-world): `:lava` (set of known
   penalty cells, excluded from the goal posterior), `:penalty` (their per-step
   reward), and `:noise` (orthogonal-slip transition probability). Under noise,
   crossing near the lava risks the penalty, so exploring toward the wrong cell is
   costly and learning the goal visibly reduces regret. Defaults (lava #{}, penalty
   0, noise 0) reproduce the bland-grid PSRL exactly."
  [{:keys [grid true-goal horizon n-episodes seed learn? lava penalty noise]
    :or {grid grid true-goal TRUE-GOAL horizon HORIZON n-episodes 10 seed 1 learn? true
         lava #{} penalty 0.0 noise 0.0}}]
  (let [{:keys [S]} (gw/parse-grid grid)
        free        (vec (remove lava (free-cells grid)))     ; goal candidates exclude known lava
        true-reward (combined-reward S lava penalty true-goal)
        v-star      (optimal-return grid true-reward START-IDX horizon noise)]
    (loop [ep 0, post (uniform-posterior free), eps []]
      (if (>= ep n-episodes)
        (let [regrets (map :regret eps)]
          {:episodes eps
           :cum-regret (vec (reductions + regrets))
           :final-posterior post
           :v-star v-star})
        (let [g       (sample-goal post (+ (* seed 1000) ep))
              plan-r  (combined-reward S lava penalty g)
              {:keys [states return]} (run-episode grid plan-r true-reward horizon S noise
                                                   (+ (* seed 1000) ep))
              ;; the posterior is over the +1 goal only; lava is known, so update on
              ;; the goal signal (1 at the true goal, 0 elsewhere), not the penalty.
              post'   (if learn?
                        (reduce (fn [b s] (update-posterior b s (if (= s true-goal) 1.0 0.0))) post states)
                        post)]
          (recur (inc ep) post'
                 (conj eps {:sampled g :return return :regret (- v-star return)
                            :posterior post' :states states
                            :reached-goal? (boolean (some #(= true-goal %) states))})))))))
