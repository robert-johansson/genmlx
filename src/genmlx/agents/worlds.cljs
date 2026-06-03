(ns genmlx.agents.worlds
  "Canonical agentmodels.org environments as reusable data, compiled through
   genmlx.agents.gridworld/build-mdp. The two worlds that OPEN the curriculum:

   - `line-mdp` — Ch 3a integer line-world: a 1-D corridor with a single goal.
     The minimal, hand-traceable MDP (value iteration + softmax policy on a line).

   - `hike-mdp` — Ch 3b Hiking gridworld: West (U=1) and East (U=10) peaks are
     walled pockets in the middle row; the bottom row is a Hill cliff (U=-10).
     Deterministic and stochastic (orthogonal-slip) variants. Under noise the
     agent values cliff-adjacent cells LESS and routes away from the Hill — the
     canonical noise-induced-detour demonstration. The grid, utilities, start and
     noise are re-derived from agentmodels' makeHikeMDP (5×5, start [0,1],
     East 10 / West 1 / Hill -10, timeCost -0.1, noise 0 or 0.1).

   These are pure data; the heavy numeric work (value iteration, rollout) lives in
   genmlx.agents.agent. New worlds belong here, keeping gridworld.cljs the compiler."
  (:require [genmlx.agents.gridworld :as gw]))

;; ===========================================================================
;; Ch 3a — integer line-world (the minimal teaching MDP)
;; ===========================================================================

(defn line-grid
  "A 1×n corridor (one row, screen coords) with a single :goal terminal at
   column `goal-idx`. Every other cell is empty."
  [n goal-idx]
  [(vec (for [i (range n)] (if (= i goal-idx) :goal :empty)))])

(defn line-mdp
  "Build the Ch 3a integer line-world: an n-state 1-D corridor with a single
   :goal reward and a per-step timeCost (so the agent reaches the goal quickly).
   Movement is left/right; up/down clamp to a stay since the grid is one row high.

   Options: :n (default 7) :goal-idx (default n-1) :reward (default 1.0)
            :time-cost (default -0.1) :start (default [0 0]) :gamma (default 1.0)."
  [{:keys [n goal-idx reward time-cost start gamma]
    :or {n 7 reward 1.0 time-cost -0.1 gamma 1.0}}]
  (let [goal-idx (or goal-idx (dec n))]
    (gw/build-mdp {:grid (line-grid n goal-idx)
                   :utilities {:goal reward :timeCost time-cost}
                   :start (or start [0 0]) :gamma gamma :noise 0.0})))

;; ===========================================================================
;; Ch 3b — Hiking gridworld (the noise-induced-detour demonstration)
;; ===========================================================================

(def hike-grid
  "agentmodels' makeHikeMDP, a 5×5 grid (rows top-first; screen coords).
   West (idx 12) and East (idx 14) peaks sit in the middle row, each a pocket
   walled off (walls at idx 6, 11, 13). The whole bottom row (idx 20–24) is a
   Hill cliff. Start is [0,1] = idx 5."
  [[:empty :empty :empty :empty :empty]
   [:empty :wall  :empty :empty :empty]
   [:empty :wall  :west  :wall  :east]
   [:empty :empty :empty :empty :empty]
   [:hill  :hill  :hill  :hill  :hill]])

(def hike-utilities
  "agentmodels hike payoffs: East 10 (the prize peak), West 1 (the lesser peak),
   Hill -10 (the cliff), and a -0.1 per-step timeCost (preference for short hikes)."
  {:east 10.0 :west 1.0 :hill -10.0 :timeCost -0.1})

(defn hike-mdp
  "Build the Hiking MDP. `:noise` selects the agentmodels variant:
     0.0  → deterministic (totalTime 12);
     0.1  → stochastic orthogonal slip (totalTime 13), where stepping along the
            cliff-adjacent bottom rows risks slipping into a Hill, so a
            noise-aware agent keeps its distance.

   Options: :noise (default 0.0) :utilities (default hike-utilities)
            :start (default [0 1]) :gamma (default 1.0)."
  [{:keys [noise utilities start gamma]
    :or {noise 0.0 utilities hike-utilities start [0 1] gamma 1.0}}]
  (gw/build-mdp {:grid hike-grid :utilities utilities
                 :start start :gamma gamma :noise noise}))

(def big-hike-grid
  "agentmodels' makeBigHikeMDP — a 6×6 grid (rows top-first; screen coords). West
   (idx 21) and East (idx 23) peaks sit in row 3 as walled pockets (walls at idx
   14, 20, 22); the whole bottom row (idx 30–35) is the Hill cliff. Start [1,1] = idx 7."
  [[:empty :empty :empty :empty :empty :empty]
   [:empty :empty :empty :empty :empty :empty]
   [:empty :empty :wall  :empty :empty :empty]
   [:empty :empty :wall  :west  :wall  :east]
   [:empty :empty :empty :empty :empty :empty]
   [:hill  :hill  :hill  :hill  :hill  :hill]])

(def big-hike-utilities
  "Big-hike payoffs: East 10, West 7 (a closer-valued peak than the 5×5's West=1,
   so the choice is genuinely contestable), a steep Hill cliff (-40), and a larger
   -0.4 per-step timeCost."
  {:east 10.0 :west 7.0 :hill -40.0 :timeCost -0.4})

(defn big-hike-mdp
  "Build the Big-Hiking MDP (agentmodels makeBigHikeMDP, totalTime 12). `:noise`
   defaults to 0.03 (the agentmodels value). Options: :noise :utilities
   (default big-hike-utilities) :start (default [1 1]) :gamma (default 1.0)."
  [{:keys [noise utilities start gamma]
    :or {noise 0.03 utilities big-hike-utilities start [1 1] gamma 1.0}}]
  (gw/build-mdp {:grid big-hike-grid :utilities utilities
                 :start start :gamma gamma :noise noise}))

;; ===========================================================================
;; Ch 3d — goal+lava exploration world (for Posterior Sampling RL)
;; ===========================================================================

(def lava-world
  "A goal+lava exploration world for the PSRL driver (agentmodels.psrl). A 5×5
   open grid (rows top-first; start is idx 0, top-left) with:
     - LAVA flanking the only row-2 crossing: idx {10,11,13,14}, so the single gap
       at idx 12 is bracketed by lava on both sides. Descending through the gap, an
       orthogonal slip lands on lava — KNOWN, fixed per-step penalty cells;
     - the rewarding GOAL in the SAFE top region (idx 3) — reachable without ever
       crossing the lava;
     - transition NOISE (orthogonal slip, default 0.2).

   The PSRL reward is per-cell, applied each step (NOT terminal), matching the
   driver's finite-horizon utility(state); lava is a known feature shared by every
   reward hypothesis, while the +1 goal is the unknown the posterior is over.

   Why it makes exploration visibly meaningful: while the agent is UNCERTAIN it
   samples goals all over — including the bottom — and chasing a bottom goal forces
   it across the lava gap, where it slips into the penalty. Once it LEARNS the true
   goal is in the safe top, it stops crossing. So learning REDUCES both the lava
   exposure and the regret versus the no-learning baseline (empirically ~halves the
   lava hits and cuts regret several-fold).

   Consume via (agentmodels.psrl/psrl (merge worlds/lava-world {:seed s :n-episodes n}))."
  {:grid       (vec (repeat 5 (vec (repeat 5 :empty))))
   :lava       #{10 11 13 14}
   :true-goal  3
   :penalty    -3.0
   :noise      0.2
   :horizon    12})
