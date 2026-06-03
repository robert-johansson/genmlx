(ns agentmodels.irl
  "Inverse Reinforcement Learning over an MDP agent (agentmodels Ch 4, 'Reasoning
   about agents') — inferring an agent's full UTILITY TABLE, timeCost, and softmax
   alpha from observed Gridworld trajectories.

   This generalizes the shipped discrete-goal inverse-planning (genmlx.agents.inverse)
   to a multi-valued utility table per restaurant plus the timeCost and alpha
   nuisance parameters — agentmodels' Equation 1:

       P(U,α | (s,a)_{0:n}) ∝ P(U,α) · Π_i P(a_i | s_i, U, α)

   The per-timestep likelihood P(a_i|s_i,U,α) is EXACTLY inverse/action-loglik — the
   GFI `p/assess` weight of the observed action under that agent's softmax policy (no
   bespoke likelihood). Inference is exact host-side enumeration over the finite
   prior grid, the same pattern as the 5d/5e joint inference. The forward agent is the
   standard (non-discounting) MDP agent (agent/make-mdp-agent) over the Restaurant-
   Choice grid (reused from the Ch 5e geometry); utilities are scalar terminals.

   Two inference styles (both in the chapter):
     - FACTORIZED SOFTMAX (Equation 1): score each observed [state,action] via
       action-loglik; the substantive joint inference of utility+timeCost+alpha.
     - GENERATE-AND-COMPARE: at high alpha, predict the trajectory and keep tables
       whose prediction matches the observation (the chapter's motivating example).

   Faithful (qualitative) targets: (1) one leftward step (α=2 fixed) → the agent's
   FAVOURITE is inferred to be Donut, but Veg vs Noodle stay ~equal (unidentifiable —
   the step gives no evidence distinguishing them); (2) jointly inferring timeCost+α,
   that same single step no longer strongly favours Donut (a high timeCost or low α
   explains it), pulling the posterior back toward the prior; (3) a longer trajectory
   sharpens the posterior.

   Reuse, zero engine change: agent/make-mdp-agent + gw/build-mdp (forward model),
   bp/restaurant-temptation-grid (geometry), inverse/action-loglik + normalize-logs."
  (:require [genmlx.agents.gridworld :as gw]
            [genmlx.agents.agent :as agent]
            [genmlx.agents.biased-planners :as bp]
            [genmlx.agents.inverse :as inv]))

;; The Restaurant-Choice grid (agentmodels Ch4 uses the same grid as Ch5b/5e).
(def grid bp/restaurant-temptation-grid)
(def start-xy [3 6])                 ; agentmodels start [3,1] (y-up) in top-first coords
(def N-ITERS 15)                     ; value-iteration sweeps (paths ≤ ~11 steps)

;; agentmodels actions: 0=left 1=right 2=up 3=down (gw/action-deltas).
;; The canonical Donut-South observation: from the start, step left ×3 then down to
;; Donut-South (idx 42). The single-step example uses just its first pair.
(def start-idx (+ 3 (* 6 6)))        ; = 39
(def single-left-obs   [[start-idx 0]])
(def donut-south-obs   [[39 0] [38 0] [37 0] [36 3]])

;; agentmodels Ch4 prior support
(def food-vals      [0 1 2])
(def time-cost-vals [-0.1 -0.3 -0.6])
(def alpha-vals     [0.1 1.0 10.0 100.0])

;; ===========================================================================
;; Forward agents over the joint finite prior (Donut-N = Donut-S, agentmodels)
;; ===========================================================================

(defn- restaurant-mdp [utils]
  (gw/build-mdp {:grid grid :utilities utils :start start-xy :gamma 1.0 :noise 0.0}))

(defn build-agents
  "Map {[di vi ni ti ai] -> {:agent :donut :veg :noodle :time-cost :alpha}} over the
   joint prior grid. One standard MDP agent per (utility-table, timeCost, alpha). The
   utility table is scalar per restaurant; Donut-N and Donut-S share the donut value."
  [{:keys [donut-vals veg-vals noodle-vals time-cost-vals alpha-vals n-iters]
    :or {n-iters N-ITERS}}]
  (into {}
        (for [di (range (count donut-vals)), vi (range (count veg-vals)), ni (range (count noodle-vals))
              ti (range (count time-cost-vals)), ai (range (count alpha-vals))]
          (let [donut (nth donut-vals di) veg (nth veg-vals vi) noodle (nth noodle-vals ni)
                tc (nth time-cost-vals ti) alpha (nth alpha-vals ai)
                mdp (restaurant-mdp {:donut-n donut :donut-s donut :veg veg :noodle noodle :timeCost tc})]
            [[di vi ni ti ai]
             {:agent (agent/make-mdp-agent {:mdp mdp :alpha alpha :gamma 1.0 :n-iters n-iters})
              :donut donut :veg veg :noodle noodle :time-cost tc :alpha alpha}]))))

;; ===========================================================================
;; Factorized-softmax posterior (Equation 1) + summary statistics
;; ===========================================================================

(defn joint-posterior
  "Exact P(U,timeCost,alpha | observations). observations = seq of [state action].
   Weight = Σ_i action-loglik (the factorized softmax likelihood); uniform prior
   cancels in normalize-logs. Returns {tuple -> probability}."
  [agents observations]
  (inv/normalize-logs
    (into {} (for [[tup {:keys [agent]}] agents]
               [tup (reduce (fn [acc [s a]] (+ acc (inv/action-loglik agent s a))) 0.0 observations)]))))

(defn- fav [{:keys [donut veg noodle]} which]
  (case which
    :donut  (and (> donut veg)  (> donut noodle))
    :veg    (and (> veg donut)  (> veg noodle))
    :noodle (and (> noodle donut) (> noodle veg))))

(defn summarize
  "Posterior summary statistics over a {tuple -> prob} distribution."
  [weighted agents]
  (reduce (fn [acc [tup pr]]
            (let [a (agents tup)]
              (-> acc
                  (update :p-donut-favorite  + (* pr (if (fav a :donut)  1.0 0.0)))
                  (update :p-veg-favorite    + (* pr (if (fav a :veg)    1.0 0.0)))
                  (update :p-noodle-favorite + (* pr (if (fav a :noodle) 1.0 0.0)))
                  (update :e-donut     + (* pr (:donut a)))
                  (update :e-alpha     + (* pr (:alpha a)))
                  (update :e-time-cost + (* pr (:time-cost a))))))
          {:p-donut-favorite 0.0 :p-veg-favorite 0.0 :p-noodle-favorite 0.0
           :e-donut 0.0 :e-alpha 0.0 :e-time-cost 0.0}
          weighted))

(defn prior-summary
  "Summary statistics under the uniform prior (each tuple equally likely)."
  [agents]
  (let [n (count agents)]
    (summarize (mapv (fn [t] [t (/ 1.0 n)]) (keys agents)) agents)))

;; ===========================================================================
;; Generate-and-compare (the chapter's motivating example; high-alpha only)
;; ===========================================================================

(defn generate-and-compare
  "agentmodels' literal generate-and-compare: at high alpha the agent is near-
   deterministic, so for each candidate utility table predict the argmax trajectory
   and keep tables whose predicted [state,action] prefix matches the observation.
   Returns {[di vi ni] -> prob} (uniform over the matching tables)."
  [{:keys [donut-vals veg-vals noodle-vals time-cost alpha horizon] :or {time-cost -0.04 alpha 100.0 horizon 16}} observations]
  (let [obs-len (count observations)
        matches (for [di (range (count donut-vals)), vi (range (count veg-vals)), ni (range (count noodle-vals))
                      :let [donut (nth donut-vals di) veg (nth veg-vals vi) noodle (nth noodle-vals ni)
                            mdp (restaurant-mdp {:donut-n donut :donut-s donut :veg veg :noodle noodle :timeCost time-cost})
                            ag  (agent/make-mdp-agent {:mdp mdp :alpha alpha :gamma 1.0 :n-iters N-ITERS})
                            {:keys [states actions]} (agent/simulate-mdp ag start-idx horizon)
                            pred (mapv vector states actions)]
                      :when (= (vec observations) (vec (take obs-len pred)))]
                  [di vi ni])
        n (count matches)]
    (into {} (map (fn [t] [t (/ 1.0 n)]) matches))))

;; ===========================================================================
;; Model specs (agentmodels Ch4)
;; ===========================================================================

(defn utility-only-spec
  "Infer the utility table only; timeCost fixed -0.04, alpha fixed 2 (chapter's first
   single-step example)."
  []
  {:donut-vals food-vals :veg-vals food-vals :noodle-vals food-vals
   :time-cost-vals [-0.04] :alpha-vals [2.0]})

(defn joint-spec
  "Joint inference of utility + timeCost + alpha (chapter's joint example)."
  []
  {:donut-vals food-vals :veg-vals food-vals :noodle-vals food-vals
   :time-cost-vals time-cost-vals :alpha-vals alpha-vals})

(defn analyze
  "Build agents for `spec`, condition on `observations`, return {:posterior :prior}."
  [spec observations]
  (let [agents (build-agents spec)]
    {:posterior (summarize (joint-posterior agents observations) agents)
     :prior     (prior-summary agents)}))
