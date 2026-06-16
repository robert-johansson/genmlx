(ns agentmodels.hike-irl
  "Hiker inverse planning (agentmodels Ch 4) on the Big-Hiking terrain — a SECOND
   IRL domain beyond the Restaurant-Choice grid. Observe a hiker's path and infer
   which PEAK they value more (East vs West) by INVERTING the same softmax MDP
   agent through the GFI. Reuses genmlx.agents.inverse — goal-agents /
   action-loglik / posterior-sequence, the very machinery used for restaurant goal
   inference — pointed at the hiking world.

   The Big-Hike peaks are close in value (East 10 / West 7) with a steep Hill cliff
   (-40) every hiker avoids, so a hiker's destination is genuinely informative: the
   posterior over {:east :west} starts uniform and sharpens as the path commits
   toward one peak. The Hill is shared KNOWN terrain, passed to goal-agents as
   :fixed (so each peak hypothesis still avoids the cliff).

   The observed hiker is the deterministic (argmax-Q) near-optimal path of the
   peak-preferring hypothesis, so a fixed seed is not needed — the demonstration is
   reproducible. Reuse, zero engine change: worlds/big-hike-grid (terrain),
   inverse/goal-agents + posterior-sequence (the inversion)."
  (:require [genmlx.mlx :as mx]
            [genmlx.agents.worlds :as w]
            [genmlx.agents.inverse :as inv]))

(def ALPHA 3.0)            ; softmax rationality of the modelled hiker
(def TIME-COST -0.4)
(def NOISE 0.03)
(def HORIZON 12)

(defn peak-agents
  "One hiker hypothesis per peak: :east values East (10) > West (7); :west values
   West (10) > East (7). Both carry the fixed Hill cliff (-40) and the timeCost.
   Built via inverse/goal-agents (the restaurant goal-inference machinery)."
  []
  (inv/goal-agents {:grid w/big-hike-grid :goals [:east :west]
                    :high 10.0 :low 7.0 :fixed {:hill -40.0}
                    :time-cost TIME-COST :alpha ALPHA :noise NOISE :start [1 1]}))

(defn- argmax-first [xs]
  (reduce (fn [best i] (if (> (nth xs i) (nth xs best)) i best)) 0 (range 1 (count xs))))

(defn greedy-observations
  "The deterministic argmax-Q trajectory of `agent` from its start, as [state
   action] observation pairs (one per step until a terminal). RNG-free, so it is a
   reproducible 'observed hiker' path."
  [agent]
  (let [{:keys [Q mdp]} agent
        Qh (mx/->clj Q)
        {:keys [ns-fn terminals start-idx]} mdp
        terms (set (keys terminals))]
    (loop [s start-idx, step 0, obs []]
      (if (or (>= step HORIZON) (terms s))
        obs
        (let [a  (argmax-first (nth Qh s))
              s' (ns-fn s a)]
          (recur s' (inc step) (conj obs [s a])))))))

(defn infer-peak
  "Posterior over {:east :west} after each observed action (a vector of maps, one
   per prefix; index 0 is the uniform prior)."
  [agents obs]
  (inv/posterior-sequence agents {:east 0.5 :west 0.5} obs))

(defn demo
  "Build the hypotheses, observe a true East-preferring hiker, and report how the
   peak-preference posterior sharpens toward :east along the path."
  []
  (let [agents   (peak-agents)
        obs      (greedy-observations (:east agents))
        post-seq (infer-peak agents obs)]
    {:observations obs
     :posterior-sequence post-seq
     :final (last post-seq)}))
