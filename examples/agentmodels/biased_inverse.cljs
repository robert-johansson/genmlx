(ns agentmodels.biased-inverse
  "Inverse inference over an agent's BIAS (agentmodels Ch 5d/5e — joint inference of
   biases and preferences), capability-only slice: the bias as a FIRST-CLASS TRACED
   LATENT in a single joint generative function, inverted through the GFI.

   The whole point of `genmlx.agents`: an agent is a generative function; the
   inference method used to reason about it (exact enumeration, importance sampling,
   …) is pluggable and orthogonal to the agent definition. Here we trace the agent's
   bias (:naive vs :sophisticated) as the one latent random choice, wire it into the
   forward biased EU recursion from `biased-planners.cljs` (no duplicated recursion —
   we reuse `make-biased-mdp-agent` verbatim; `bp/eu-row` is a thin accessor over the
   agent's public `:expected-utility`), and recover P(:bias | observed actions) two ways on the
   SAME generative function:

     1. EXACT — enumerate the finite bias prior, score the full trajectory
        {:bias, :a0, …} with `p/assess`, normalize. This is the closed-form
        posterior (the bias set is finite, assess only scores).
     2. IMPORTANCE SAMPLING — sample :bias, constrain the actions, group the
        normalized particle weight by sampled bias. A sampling cross-check that
        agrees with the exact posterior (run at finite α; argmax likelihoods make
        IS degenerate at α=##Inf).

   The likelihood of an observed action under a hypothesized bias is exactly the
   softmax policy probability of the forward biased planner — no bespoke likelihood,
   just the same generative policy the agent acts with, scored against the action.
   This mirrors `inverse.cljs`'s goal-inference idiom, but indexes hypotheses by
   bias value rather than goal; `bias-posterior-via-policy` reuses that idiom as an
   independent consistency check.

   Scope: ONE latent (:bias). The full Ch 5d/5e joint over preferences/discount/α
   (procrastination time-series, restaurant utility table) is bean genmlx-4zfy."
  (:require [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.importance :as is]
            [genmlx.inference.util :as iu]
            [genmlx.agents.biased-planners :as bp]
            [genmlx.agents.helpers :as h]
            [genmlx.agents.inverse :as inv])
  (:require-macros [genmlx.gen :refer [gen]]))

(def bias-values
  "The finite bias prior support, in index order (the categorical samples an index)."
  [:naive :sophisticated])

(defn- check-aligned
  "Guard: :states and :actions must be the same length (one observed action per
   visited state). A mismatch would silently drop/mis-bind action sites."
  [{:keys [states actions]}]
  (when-not (= (count states) (count actions))
    (throw (ex-info (str "biased-inverse: :states and :actions must be the same length — got "
                         (count states) " states and " (count actions) " actions")
                    {:n-states (count states) :n-actions (count actions)}))))

;; ===========================================================================
;; The tiny temptation MDP (hand-built)
;; ===========================================================================
;;
;; gw/build-mdp only produces 4-action gridworlds, so the minimal 5-state /
;; 2-action world that yields clean closed-form posteriors is built as a literal
;; MDP map fed straight to make-biased-mdp-agent (which destructures :A :terminals
;; :R :T and assoc's :gamma/:alpha itself).
;;
;;   states  0=start  1=tempt(gate)  2=safe1  3=donut(term)  4=veg(term)
;;   actions 0, 1
;;   start  : a0 → tempt(1)   a1 → safe1(2)       ; a0 heads for the temptation gate
;;   tempt  : a0 → donut(3)   a1 → veg(4)         ; a0 = succumb (eat), a1 = continue
;;   safe1  : a0,a1 → veg(4)                       ; the safe route to veg
;;   donut,veg : absorbing
;;   R: tempt[a0]=Rd=5 (donut), veg[*]=Rv=8 ; everything else 0
;;
;; With k=1, γ=1: both routes reach veg in 2 steps → value 8·δ(2)=8/3 for a future
;; self that resists. The crossover Rd>Rv/2 (5>4) makes the *delay-0* self at the
;; gate succumb (donut value δ(1)·5=2.5), while Rd<2Rv/3 (5<5.33) makes the
;; *delay-1* self resist. So:
;;   naive  EU(start) = [8/3, 8/3]  (TIE — believes its future self resists, both
;;                                   routes look like they reach veg)  → π=[0.5,0.5]
;;   soph   EU(start) = [2.5, 8/3]  (foresees succumbing on the tempting route)
;;                                   → π=[0,1] (NEVER heads for temptation)
;; Hence: heading for temptation (a0) is diagnostic of naive; repeatedly taking the
;; safe route (a1) is evidence for sophisticated.

(defn temptation-mdp
  "Build the minimal 5-state / 2-action temptation MDP (see namespace notes).
   Returns an MDP map {:S :A :T :R :terminals} feeding make-biased-mdp-agent."
  []
  (let [S 5
        A 2
        ;; T[s][a][s'] one-hot transitions
        T-rows [[[0 1 0 0 0] [0 0 1 0 0]]    ; start : a0→tempt, a1→safe1
                [[0 0 0 1 0] [0 0 0 0 1]]    ; tempt : a0→donut(eat), a1→veg(continue)
                [[0 0 0 0 1] [0 0 0 0 1]]    ; safe1 : both → veg
                [[0 0 0 1 0] [0 0 0 1 0]]    ; donut : absorbing
                [[0 0 0 0 1] [0 0 0 0 1]]]   ; veg   : absorbing
        R-rows [[0 0]                         ; start
                [5 0]                         ; tempt : a0 eat donut (Rd=5), a1 continue
                [0 0]                         ; safe1
                [0 0]                         ; donut
                [8 8]]                        ; veg   : Rv=8
        T (mx/array (clj->js T-rows) mx/float32)
        R (mx/array (clj->js R-rows) mx/float32)]
    ;; No eager mx/eval! — T/R are constant arrays consumed once via mx/->clj in
    ;; build-biased-eu, which materializes them there (the proper boundary).
    {:S S :A A :T T :R R :terminals {3 :donut 4 :veg}}))

;; ===========================================================================
;; Forward agents per bias + EU rows
;; ===========================================================================

(defn bias-agents
  "One forward biased MDP agent per bias value, sharing the same mdp/α/k/C_g.
   Returns {:naive agent :sophisticated agent}. Built once and reused (each agent
   carries its own memoized EU cache), so importance sampling does not rebuild."
  [{:keys [mdp alpha discount reward-myopic-bound n-iters]
    :or   {alpha ##Inf discount 1.0 reward-myopic-bound ##Inf n-iters 10}}]
  (into {}
        (for [b bias-values]
          [b (bp/make-biased-mdp-agent
               {:mdp mdp :alpha alpha :gamma 1.0 :n-iters n-iters}
               {:discount discount :bias b :reward-myopic-bound reward-myopic-bound})])))

;; ===========================================================================
;; The joint generative function — :bias is a first-class traced latent
;; ===========================================================================

(defn biased-agent-model
  "The joint generative function. ONE traced latent — :bias — drawn from the prior
   (uniform, or weighted by `:prior`, a weight vector aligned to `bias-values`).
   The sampled bias selects a precomputed forward agent; one action site :a0 :a1 …
   per state in `:states` is then a softmax-action over that agent's EU row.

   Inverting:
     • EXACT     — assess on a full choicemap {:bias i, :a0 a0, …}
     • SAMPLING  — generate constraining only the actions (IS over the latent :bias)

   cfg keys: :mdp :alpha :discount :reward-myopic-bound :n-iters :states [:prior].
   Returns a DynamicGF of zero user args."
  [{:keys [states alpha prior mdp] :or {alpha ##Inf} :as cfg}]
  (let [agents    (bias-agents cfg)
        n-actions (:A mdp)
        box       (if prior
                    (h/weighted-draw bias-values prior)
                    (h/uniform-draw bias-values))
        ;; Precompute the policy logit array per (bias, state) ONCE — the gen body
        ;; is re-run per IS particle, so it must only index, never rebuild arrays.
        rows      (into {} (for [b bias-values]
                             [b (mapv #(mx/array (clj->js (bp/eu-row (agents b) % n-actions)) mx/float32)
                                      states)]))]
    (gen []
      (let [bi   (trace :bias (:dist box))
            bias (h/draw-value box bi)
            er   (rows bias)]
        (doseq [t (range (count states))]
          (trace (keyword (str "a" t))
                 (h/softmax-action alpha (nth er t))))
        bi))))

;; ===========================================================================
;; Choicemap builders
;; ===========================================================================

(defn full-cm
  "Choicemap constraining the full trajectory: {:bias i, :a0 a0, …}."
  [bias-idx actions]
  (cm/set-choice (h/action-choicemap actions) [:bias] bias-idx))

;; ===========================================================================
;; Exact posterior — enumerate the finite bias prior via p/assess
;; ===========================================================================

(defn bias-posterior
  "Exact P(:bias | actions). Enumerate the bias prior; for each value, assess the
   joint GF on {:bias i, :a0 a0, …} (weight = log P0(bias) + Σ log π_bias(a_t));
   normalize via inv/normalize-logs. Closed-form (finite bias set, assess only
   scores — no sampling). Returns {:naive p :sophisticated q}.

   With `:states []` (no actions) this returns the prior — assess on {:bias i}
   scores only the prior log-prob."
  [{:keys [actions] :as cfg}]
  (check-aligned cfg)
  (let [model (biased-agent-model cfg)
        logw  (into {}
                    (for [[i b] (map-indexed vector bias-values)]
                      [b (mx/item (:weight (p/assess (dyn/auto-key model) []
                                                     (full-cm i actions))))]))]
    (inv/normalize-logs logw)))

(defn- prior-prob
  "Normalized prior probability per bias value from a weight vector (nil = uniform)."
  [prior]
  (let [ws (or prior (vec (repeat (count bias-values) 1.0)))
        z  (apply + ws)]
    (zipmap bias-values (map #(/ % z) ws))))

(defn bias-posterior-via-policy
  "The SAME exact posterior via the established inverse.cljs idiom: one forward
   agent per bias, inv/action-loglik scoring each observed [state action] against
   that agent's :policy, plus the log-prior, normalized. An independent
   consistency check that the joint GF agrees with the goal-inference idiom
   (compose, don't duplicate)."
  [{:keys [states actions prior] :as cfg}]
  (check-aligned cfg)
  (let [agents (bias-agents cfg)
        p0     (prior-prob prior)
        logw   (into {}
                     (for [b bias-values]
                       [b (reduce + (Math/log (p0 b))
                                  (map (fn [s a] (inv/action-loglik (agents b) s a)) states actions))]))]
    (inv/normalize-logs logw)))

;; ===========================================================================
;; Importance-sampling cross-check
;; ===========================================================================

(defn is-bias-posterior
  "Importance-sampling estimate of P(:bias | actions) on the joint GF: sample
   :bias from the prior, constrain the observed actions, group the normalized
   particle weight by each particle's sampled bias. Returns
   {:posterior {:naive p :sophisticated q} :ess ess}.

   Run at FINITE α — at α=##Inf the argmax likelihoods are 0/1 indicators and IS
   degenerates (zero-weight particles). Pass a fixed `key` (e.g. (rng/fresh-key 42))
   for reproducibility; MLX RNG is bit-reproducible."
  [{:keys [actions] :as cfg} n key]
  (check-aligned cfg)
  (let [model (biased-agent-model cfg)
        obs   (h/action-choicemap actions)
        {:keys [traces log-weights]} (is/importance-sampling {:samples n :key key} model [] obs)
        {:keys [probs]} (iu/normalize-log-weights log-weights)
        ess   (iu/compute-ess log-weights)
        post  (reduce (fn [m [bias w]] (update m bias + w))
                      {:naive 0.0 :sophisticated 0.0}
                      (map (fn [tr w]
                             [(nth bias-values (int (mx/item (cm/get-choice (:choices tr) [:bias])))) w])
                           traces probs))]
    {:posterior post :ess ess}))
