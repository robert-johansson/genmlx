(ns agentmodels.restaurant-joint-inference
  "Restaurant-Choice JOINT inference of biases AND preferences (agentmodels Ch 5e).

   From a single observed Gridworld trajectory we jointly infer the agent's
   [immediate,delayed] utility TABLE, whether it is naive or sophisticated, its
   hyperbolic discount, and its softmax alpha — by exact host-side enumeration over
   finite priors, exactly the 5d (`joint_5d_inference.cljs`) pattern lifted to the
   full restaurant geometry (`bp/restaurant-temptation-mdp`, Phase 1).

   The headline (agentmodels 5e): conditioning on the Naive path (short route, ends
   at Donut-North) the posterior P(donutTempting) rises from a prior < 0.1 to ~0.9,
   E[vegMinusDonut] turns positive (the agent net-prefers Veg yet was tempted), and
   repeating the observation makes the discounting explanation beat the high-noise one.

   Three nested models (agentmodels 5e):
     :discounting  discount=1 fixed, alpha=1000 fixed, infer naive/soph + the full
                   4^4 utility table over {-10,0,10,20}.            (the headline)
     :optimal      discount=0 (no time-inconsistency), infer utilities (delayed=0)
                   + alpha over {.1,10,100,1000}. Explains the path only via noise.
     :full         infer discount∈{0,1}, naive/soph, alpha∈{.1,10,1000}, utilities
                   (delayed fixed donut=-10/veg=20 to bound the grid).

   The likelihood of an observed action is the softmax policy probability of the
   forward biased planner, scored with `p/assess` — no bespoke likelihood. agentmodels
   observes `act(state, 0)`, so every action is scored at delay 0 and the full
   planning horizon (the agent re-plans from delay 0 each real step).

   `donutTempting` / `vegMinusDonut` are the verbatim webppl-agents library predicates
   (getRestaurantHyperbolicInfer): PARAMETER conditions on the sampled table+discount,
   not behavioral — see `donut-tempting?`.

   Reuse, zero engine change: bp/restaurant-temptation-mdp (Phase-1 geometry),
   bp/make-biased-mdp-agent (forward model), bp/eu-row (EU vector via the frozen
   :expected-utility accessor), inv/normalize-logs (stable softmax),
   h/uniform-draw (finite priors), h/action-choicemap (action choicemap)."
  (:require [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.agents.biased-planners :as bp]
            [genmlx.agents.helpers :as h]
            [genmlx.agents.inverse :as inv])
  (:require-macros [genmlx.gen :refer [gen]]))

(def H 18)            ; planning + scoring horizon (≈ agentmodels plan-until-terminal)

;; ===========================================================================
;; Geometry reuse: the temptation MDP's T/terminals/twins are identical across
;; utility tables — only R (the restaurant [imm,del] values) changes. Build the
;; base ONCE, then swap R per candidate table (avoids 512× grid-parse + T build).
;; ===========================================================================

(def base-mdp (bp/restaurant-temptation-mdp {}))

(defn- mdp-with-utilities
  "Return the base temptation MDP with R recomputed for `utilities`
   {:donut-n [imm del] :donut-s [imm del] :veg [imm del] :noodle [imm del]
    :timeCost t}. Reuses the base :T/:terminals/:twin/:restaurants (same geometry)."
  [{:keys [S A terminals restaurants] :as base} utilities]
  (let [time-cost (get utilities :timeCost 0.0)
        comp-of   (fn [kw i] (let [u (get utilities kw)]
                               (if (sequential? u) (nth u i) u)))
        reward-of (fn [s]
                    (cond
                      (contains? terminals s)   (comp-of (terminals s) 1)    ; L@1: delayed
                      (contains? restaurants s) (comp-of (restaurants s) 0)  ; L@0: immediate
                      :else                     time-cost))
        R (mx/array (clj->js (for [s (range S)] (repeat A (reward-of s)))) mx/float32)]
    (assoc base :R R)))

;; ===========================================================================
;; The webppl-agents library summary statistics (verbatim parameter predicates)
;; ===========================================================================

(defn veg-minus-donut
  "sum(Veg) − sum(DonutN): the net-utility advantage of Veg over Donut."
  [{:keys [donut veg]}]
  (- (+ (veg 0) (veg 1)) (+ (donut 0) (donut 1))))

(defn donut-tempting?
  "agentmodels' getRestaurantHyperbolicInfer predicate (verbatim):
     dis(d)      = 1/(1 + discount·d)
     disU(u,d)   = dis(d)·u[0] + dis(d+1)·u[1]
     donutTempting = (disUDonut(4) < disUVeg(6)) ∧ (disUDonut(0) > disUVeg(2))
   True iff the discounted utilities encode a preference reversal: Veg preferred at
   distance (start) but Donut-North preferred when adjacent. discount=0 ⇒ never true."
  [{:keys [donut veg discount]}]
  (let [dis  (fn [d] (/ 1.0 (+ 1.0 (* (double discount) d))))
        disU (fn [u d] (+ (* (dis d) (u 0)) (* (dis (inc d)) (u 1))))]
    (and (< (disU donut 4) (disU veg 6))
         (> (disU donut 0) (disU veg 2)))))

;; ===========================================================================
;; Forward agents over the joint finite prior  (7 latent dims; fixed dims = 1 value)
;; ===========================================================================
;; spec keys: :donut-imm-vals :donut-del-vals :veg-imm-vals :veg-del-vals
;;            :discount-vals :bias-vals :alpha-vals :noodle :time-cost :n-iters
;; latent index tuple = [di dd vi vd dc bi ai]

(defn- tuples-of [spec]
  (let [{:keys [donut-imm-vals donut-del-vals veg-imm-vals veg-del-vals
                discount-vals bias-vals alpha-vals]} spec]
    (for [di (range (count donut-imm-vals)), dd (range (count donut-del-vals))
          vi (range (count veg-imm-vals)),   vd (range (count veg-del-vals))
          dc (range (count discount-vals)),  bi (range (count bias-vals))
          ai (range (count alpha-vals))]
      [di dd vi vd dc bi ai])))

(defn build-agents
  "Map {[di dd vi vd dc bi ai] -> {:agent :donut :veg :discount :bias :alpha}} over
   the full joint prior grid. One temptation agent per parameter tuple (geometry
   reused; only R varies). Donut-S shares Donut-N's table (agentmodels)."
  [{:keys [donut-imm-vals donut-del-vals veg-imm-vals veg-del-vals
           discount-vals bias-vals alpha-vals noodle time-cost n-iters]
    :or {noodle [-10 -10] time-cost -0.01 n-iters H} :as spec}]
  (into {}
        (for [[di dd vi vd dc bi ai :as tup] (tuples-of spec)]
          (let [donut [(nth donut-imm-vals di) (nth donut-del-vals dd)]
                veg   [(nth veg-imm-vals vi)   (nth veg-del-vals vd)]
                k     (nth discount-vals dc)
                bias  (nth bias-vals bi)
                alpha (nth alpha-vals ai)
                mdp   (mdp-with-utilities base-mdp
                        {:donut-n donut :donut-s donut :veg veg :noodle noodle :timeCost time-cost})]
            [tup {:agent    (bp/make-biased-mdp-agent
                              {:mdp mdp :alpha alpha :gamma 1.0 :n-iters n-iters}
                              {:discount k :bias bias})
                  :donut donut :veg veg :discount k :bias bias :alpha alpha}]))))

;; ===========================================================================
;; The joint generative function (extends the 5d multi-latent model)
;; ===========================================================================

(defn joint-restaurant-model
  "Joint GF tracing the 7 latent indices; the sampled tuple selects a precomputed
   agent; one softmax-action site :a0 :a1 ... per observed state, scored at horizon
   H, delay 0 (agentmodels' observe(act(state,0), action)). EU-rows for every agent
   are precomputed before `gen` (the body is re-run per enumerated tuple, so it only
   indexes)."
  [{:keys [donut-imm-vals donut-del-vals veg-imm-vals veg-del-vals
           discount-vals bias-vals alpha-vals] :as spec}
   states agents]
  (let [n-actions (:A base-mdp)
        boxes [(h/uniform-draw donut-imm-vals) (h/uniform-draw donut-del-vals)
               (h/uniform-draw veg-imm-vals)   (h/uniform-draw veg-del-vals)
               (h/uniform-draw discount-vals)  (h/uniform-draw bias-vals)
               (h/uniform-draw alpha-vals)]
        rows  (into {} (for [[tup {:keys [agent]}] agents]
                         [tup (mapv (fn [s] (mx/array (clj->js (bp/eu-row agent s n-actions)) mx/float32))
                                    states)]))]
    (gen []
      (let [di (trace :di (:dist (nth boxes 0)))
            dd (trace :dd (:dist (nth boxes 1)))
            vi (trace :vi (:dist (nth boxes 2)))
            vd (trace :vd (:dist (nth boxes 3)))
            dc (trace :dc (:dist (nth boxes 4)))
            bi (trace :bi (:dist (nth boxes 5)))
            ai (trace :ai (:dist (nth boxes 6)))
            tup [di dd vi vd dc bi ai]
            er    (rows tup)
            alpha (:alpha (agents tup))]
        (dotimes [i (count states)]
          (trace (keyword (str "a" i)) (h/softmax-action alpha (nth er i))))
        tup))))

(defn- full-cm
  "Choicemap {:di .. :dd .. :vi .. :vd .. :dc .. :bi .. :ai .. :a0 .. :a1 ..}."
  [[di dd vi vd dc bi ai] actions]
  (-> (h/action-choicemap actions)
      (cm/set-choice [:di] di) (cm/set-choice [:dd] dd)
      (cm/set-choice [:vi] vi) (cm/set-choice [:vd] vd)
      (cm/set-choice [:dc] dc) (cm/set-choice [:bi] bi)
      (cm/set-choice [:ai] ai)))

;; ===========================================================================
;; Exact enumeration + summary statistics
;; ===========================================================================

(defn- summarize
  "Reduce a seq of [tuple prob] into the chapter's summary statistics, using the
   agents map for each tuple's (donut, veg, discount, bias)."
  [weighted agents]
  (reduce (fn [acc [tup pr]]
            (let [a (agents tup)]
              (-> acc
                  (update :p-donut-tempting + (* pr (if (donut-tempting? a) 1.0 0.0)))
                  (update :e-veg-minus-donut + (* pr (veg-minus-donut a)))
                  (update :p-naive           + (* pr (if (= :naive (:bias a)) 1.0 0.0)))
                  (update :p-discounting     + (* pr (if (pos? (:discount a)) 1.0 0.0)))
                  (update :alpha-marginal    (fn [m] (update m (:alpha a) (fnil + 0.0) pr))))))
          {:p-donut-tempting 0.0 :e-veg-minus-donut 0.0 :p-naive 0.0
           :p-discounting 0.0 :alpha-marginal {}}
          weighted))

(defn joint-posterior
  "Exact P(params | trajectory). Enumerate the joint finite prior; for each tuple
   assess the joint GF on the full choicemap (latent indices + observed actions);
   normalize. `:number-repeats` r conditions on the SAME action sequence r+1 times
   (agentmodels numberRepeats — strengthens the evidence). Returns
   {:joint {tuple prob} :posterior <summary> :prior <summary> :n-tuples n}."
  [{:keys [states actions number-repeats] :or {number-repeats 0} :as spec} agents]
  (assert (= (count states) (count actions))
          (str "5e joint: states/actions length mismatch " (count states) " vs " (count actions)))
  (let [model   (dyn/auto-key (joint-restaurant-model spec states agents))
        ;; numberRepeats: condition on the same observed path (r+1) times. The assess
        ;; weight w = log-prior + log-likelihood; since EVERY latent prior here is a
        ;; uniform-draw, log-prior is the SAME constant for all tuples, so the correct
        ;; reps-posterior weight (log-prior + reps·log-likelihood) equals reps·w up to
        ;; a tuple-independent constant that cancels in normalize-logs.
        reps    (inc number-repeats)
        tuples  (keys agents)
        logw    (into {}
                      (for [tup tuples]
                        [tup (* reps (mx/item (:weight (p/assess model []
                                                                 (full-cm tup actions)))))]))
        post    (inv/normalize-logs logw)
        uniform (let [n (count tuples)] (mapv (fn [t] [t (/ 1.0 n)]) tuples))]
    {:joint     post
     :posterior (summarize post agents)
     :prior     (summarize uniform agents)
     :n-tuples  (count tuples)}))

;; ===========================================================================
;; The observed Naive trajectory (the agent's own short-route-to-Donut-North path)
;; ===========================================================================

(defn observed-trajectory
  "Roll a ground-truth forward agent out from the start and return
   {:states <non-terminal states> :actions <actions>} for scoring (one (s,a) pair
   per transition; the terminal twin is dropped since no action is taken there)."
  [bias k]
  (let [ag (bp/make-biased-mdp-agent {:mdp base-mdp :alpha ##Inf :gamma 1.0 :n-iters H}
                                     {:discount k :bias bias})
        {:keys [states actions]} (bp/simulate-biased-mdp ag (:start-idx base-mdp) H)]
    {:states (pop (vec states)) :actions (vec actions)}))

(def naive-trajectory         (delay (observed-trajectory :naive 1.0)))
(def sophisticated-trajectory (delay (observed-trajectory :sophisticated 1.0)))
(def veg-direct-trajectory    (delay (observed-trajectory :naive 0.0)))  ; rational → straight Veg

;; ===========================================================================
;; The three model priors (agentmodels 5e)
;; ===========================================================================

(defn discounting-spec
  "Assume discounting (k=1), alpha=1000; infer naive/soph + the full 4^4 utility
   table over {-10,0,10,20}. (The headline P(donutTempting) model.)"
  []
  (let [u [-10 0 10 20]]
    {:donut-imm-vals u :donut-del-vals u :veg-imm-vals u :veg-del-vals u
     :discount-vals [1.0] :bias-vals [:naive :sophisticated] :alpha-vals [1000.0]
     :noodle [-10 -10] :time-cost -0.01 :n-iters H}))

(defn optimal-spec
  "Assume non-discounting (k=0, sophisticated inert); infer utilities (delayed
   omitted = 0) over {-10,0,10,20,30,40} + alpha ∈ {.1,10,100,1000}. The only way to
   explain an anomalous path is softmax noise (low alpha)."
  []
  (let [u [-10 0 10 20 30 40]]
    {:donut-imm-vals u :donut-del-vals [0] :veg-imm-vals u :veg-del-vals [0]
     :discount-vals [0.0] :bias-vals [:sophisticated] :alpha-vals [0.1 10.0 100.0 1000.0]
     :noodle [-10 -10] :time-cost -0.01 :n-iters H}))

(defn full-spec
  "Full joint: discount ∈ {0,1}, naive/soph, alpha ∈ {.1,10,1000}, utilities over
   {-10,0,10,20,30} with delayed fixed (donut=-10, veg=20) to bound the grid."
  []
  (let [u [-10 0 10 20 30]]
    {:donut-imm-vals u :donut-del-vals [-10] :veg-imm-vals u :veg-del-vals [20]
     :discount-vals [0.0 1.0] :bias-vals [:naive :sophisticated] :alpha-vals [0.1 10.0 1000.0]
     :noodle [-10 -10] :time-cost -0.01 :n-iters H}))

(defn analyze
  "Build agents for `spec`, condition on (states, actions[, number-repeats]), and
   return the joint posterior + prior/posterior summary statistics."
  [spec {:keys [states actions number-repeats] :or {number-repeats 0}}]
  (let [agents (build-agents spec)]
    (joint-posterior (assoc spec :states states :actions actions
                            :number-repeats number-repeats)
                     agents)))
