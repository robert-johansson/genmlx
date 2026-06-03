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
