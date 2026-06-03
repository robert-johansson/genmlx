(ns agentmodels.biased-planners
  "Biased / bounded planners (agentmodels Ch 5) — GenMLX-native, zero engine change.

   agentmodels Ch 5 turns the rational MDP/POMDP agent into a family of HUMANLY
   BIASED agents by adding ONE subjective `delay` axis to the expected-utility
   recursion and a handful of pure knobs over it. Everything here composes the
   existing pieces — `agent.cljs`'s host-side memoized recursion (the faithful
   `dp.cache` port), `exact/with-cache`, the GFI policy via `helpers/softmax-action`,
   and MLX tensors for the worlds. No handler/runtime/engine change.

   The recursion (faithful to agentmodels' makeAgent in 5a/5b):

       δ(d)        = 1 / (1 + k·d)                      ; hyperbolic discount, k = :discount
       EU(s,a,t,d) = δ(d)·u(s,a)
                     + ( terminal(s) ∨ t≤1 ∨ d ≥ C_g     ; C_g = reward-myopia cap
                         ? 0
                         : Σ_s' T(s,a,s') · V(s', t-1, d+1) )
       V(s',t',d') = Σ_a π(a)·EU(s',a,t',d')
         π = softmax( α · [ EU(s',a,t', perceivedDelay) ]_a )   ; POLICY at perceivedDelay
         and the averaged EU is at the TRUE d'                  ; VALUE  at d'

   TWO axes, distinct and both load-bearing (agentmodels is explicit about this):
   `t` is OBJECTIVE time-left (drives the finite horizon / termination); `d` is the
   SUBJECTIVE delay (drives discounting). They are not the same variable.

   Naive vs Sophisticated is the single line `perceivedDelay = naive ? d+1 : 0`
   (verbatim agentmodels). It is the delay the agent hands to its SIMULATED future
   self when predicting that self's action — NOT the delay it discounts its own
   reward with. The continuation VALUE always uses the true d+1; only the policy
   that picks the simulated future action uses perceivedDelay. That asymmetry is the
   whole of time-inconsistency:
   - Naive (d+1): wrongly models its future self as keeping the same ever-growing
     (so ever-flatter, ever more patient) delay clock → predicts it will resist
     temptation → but at execution `:act` re-plans from d=0, the temptation is now
     undiscounted, and it SUCCUMBS.
   - Sophisticated (0): correctly models its future self as re-planning from delay 0
     (same present-bias it has now) → foresees it would succumb → pre-commits / routes
     around the tempting state.

   Real execution always re-plans from d=0 (`:act` evaluates EU at d=0 each step,
   agentmodels' `var delay = delay || 0`); that is what makes the Naive agent's
   actual trajectory diverge from the one its own plan predicted.

   Reward-myopia (C_g): bound the look-ahead — base case `d ≥ C_g` returns only the
   discounted immediate utility (everything past C_g steps is valued 0).
   Update-myopia (C_m): a POMDP-only bound — see the belief-space section below.

   At k=0 (δ≡1) every bias collapses to the unbiased agent: the recursion is then
   independent of d, so perceivedDelay is irrelevant and `biased-eu` reproduces
   `agent/recursive-eu` and the tensor `Q` exactly (the limit-recovery anchor)."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.exact :as exact]
            [agentmodels.gridworld :as gw]
            [agentmodels.agent :as agent]
            [agentmodels.helpers :as h])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ===========================================================================
;; Section 1 — discount + the perceivedDelay branch (the two knobs)
;; ===========================================================================

(defn delta
  "Hyperbolic discount factor at subjective delay `d`: 1/(1 + k·d).
   δ(k,0)=1; strictly decreasing in d for k>0; k=0 ⇒ δ≡1 (no discounting)."
  [k d]
  (/ 1.0 (+ 1.0 (* (double k) (double d)))))

(defn bias->perceived-delay
  "The perceivedDelay function the agent uses for its SIMULATED future self when
   choosing that self's action (agentmodels: perceivedDelay = naive ? delay+1 : 0).
     :naive         → (fn [d] (inc d))   ; believes future self stays patient
     :sophisticated → (fn [_] 0)         ; models future self's present bias
   Only the policy that picks the simulated future action uses this; the
   continuation VALUE always uses the true d+1. At k=0 (δ≡1) both coincide and
   equal the unbiased agent. (Default → sophisticated/0 for any other key.)"
  [bias]
  (case bias
    :naive         inc
    :sophisticated (constantly 0)
    (constantly 0)))

(defn- softmax-vec
  "Host-side softmax over a vector of logits (stable). Returns probabilities."
  [xs]
  (let [m  (apply max xs)
        es (mapv #(Math/exp (- % m)) xs)
        z  (reduce + es)]
    (mapv #(/ % z) es)))

;; ===========================================================================
;; Section 2 — the delay-indexed biased EU recursion (MDP)
;; ===========================================================================

(defn- build-biased-eu
  "Core host-side recursion shared by the soft (finite-α) and hard (α=##Inf)
   agents. `backup-kind` ∈ {:soft :hard} chooses the value backup. Returns
   {:eu (fn [s a t d]) :backup (fn [s t value-delay policy-delay])}.

   `eu` is memoized via exact/with-cache on the JS-primitive key [s a t d] (a
   fresh cache per call — no cross-agent leakage). Rh/Th are dereferenced ONCE up
   front (pure host arithmetic — no MLX graph growth in the hot loop)."
  [{:keys [A gamma terminals alpha] :as mdp}
   {:keys [discount bias reward-myopic-bound]
    :or   {discount 0.0 bias :sophisticated reward-myopic-bound ##Inf}}
   backup-kind]
  (let [k       (double discount)
        gamma   (or gamma 1.0)
        cg      reward-myopic-bound
        pd-fn   (bias->perceived-delay bias)
        Rh      (mx/->clj (:R mdp))                 ; [S][A] JS numbers
        Th      (mx/->clj (:T mdp))                 ; [S][A][S'] JS numbers
        terms   (set (keys terminals))
        eu-atom (atom nil)
        ;; backup at state s', horizon t', VALUE delay dv, POLICY delay dp.
        ;; The policy logits read EU at dp (perceivedDelay); the averaged EU is at
        ;; the true dv. For Naive (pd-fn=inc) dp = dv at every node, so q-pol is
        ;; reused as q-val; the dp≠dv split that needs a distinct EU row is the
        ;; Sophisticated case (pd-fn=0, so dp=0 while dv=d+1≥1).
        backup  (fn [s t dv dp]
                  (let [eu    @eu-atom
                        q-pol (mapv (fn [a] (eu s a t dp)) (range A))
                        q-val (if (== dp dv) q-pol (mapv (fn [a] (eu s a t dv)) (range A)))]
                    (case backup-kind
                      :soft
                      (let [w (softmax-vec (mapv #(* alpha %) q-pol))]
                        (reduce + (map * w q-val)))
                      :hard
                      ;; argmax over the policy row; uniform tie-break, value read
                      ;; from the value row at the argmax set.
                      (let [m    (apply max q-pol)
                            idxs (filterv #(> (nth q-pol %) (- m 1e-9)) (range A))]
                        (/ (reduce + (map #(nth q-val %) idxs)) (count idxs))))))
        eu      (exact/with-cache
                  (fn [s a t d]
                    (let [u (* (delta k d) (get-in Rh [s a]))]
                      (if (or (terms s) (<= t 1) (>= d cg))
                        u
                        (+ u (* gamma
                                (reduce-kv
                                  (fn [acc s' pr]
                                    (if (pos? pr)
                                      (+ acc (* pr (backup s' (dec t) (inc d) (pd-fn d))))
                                      acc))
                                  0.0 (get-in Th [s a]))))))))]
    (reset! eu-atom eu)
    {:eu eu :backup backup}))

(defn biased-eu
  "Soft (finite-α) delay-indexed biased EU. See build-biased-eu."
  [mdp bias]
  (build-biased-eu mdp bias :soft))

(defn biased-eu-inf
  "Hard (α=##Inf) delay-indexed biased EU — argmax policy, uniform tie-break.
   Kept separate to avoid Inf·Q NaNs and to make rollouts deterministic."
  [mdp bias]
  (build-biased-eu mdp bias :hard))

;; ===========================================================================
;; Section 3 — agent constructor + rollout
;; ===========================================================================

(defn make-biased-mdp-agent
  "Build a biased MDP agent over `mdp` with a `bias` map
     {:discount k :bias :naive|:sophisticated :reward-myopic-bound C_g}.
   Returns {:mdp :policy :act :expected-utility :eu :params} — the same shape as
   agent/make-mdp-agent MINUS :Q/:V (the tensor value-iteration path has no delay
   axis; the biased agent is recursion-only). `:act` re-plans from delay 0 each
   call (agentmodels' `delay || 0`), which is what produces the Naive plan↔do
   divergence. `simulate-biased-mdp` (= agent/simulate-mdp) rolls it out."
  [{:keys [mdp alpha gamma n-iters] :or {alpha ##Inf gamma 1.0 n-iters 24}}
   {:keys [discount bias reward-myopic-bound]
    :or   {discount 0.0 bias :sophisticated reward-myopic-bound ##Inf} :as biasm}]
  (let [mdp    (assoc mdp :gamma gamma :alpha alpha)
        {:keys [eu]} (if (= alpha ##Inf) (biased-eu-inf mdp biasm) (biased-eu mdp biasm))
        H      n-iters
        A      (:A mdp)
        eu-row (fn [s] (mapv (fn [a] (eu s a H 0)) (range A)))
        policy (gen [s] (trace :action
                               (h/softmax-action alpha (mx/array (clj->js (eu-row s)) mx/float32))))]
    {:mdp mdp
     :policy policy
     :act (fn [s] (int (mx/item (:retval (p/simulate (dyn/auto-key policy) [s])))))
     :expected-utility (fn [s a] (eu s a H 0))
     :eu eu
     :params {:alpha alpha :gamma gamma :horizon H
              :discount discount :bias bias :reward-myopic-bound reward-myopic-bound}}))

(def simulate-biased-mdp
  "Roll a biased MDP agent out from `start` for ≤ horizon steps (= agent/simulate-mdp,
   which re-plans via :act each step — the d=0 re-planning that drives Naive
   time-inconsistency)."
  agent/simulate-mdp)

(defn- argmax-of [xs] (first (apply max-key second (map-indexed vector xs))))

(defn planned-rollout
  "The trajectory the agent BELIEVES it will follow — predicting each future
   action with the perceivedDelay its OWN recursion uses (Naive: the policy delay
   grows along the plan, so the far future looks patient; Sophisticated: it stays
   0). Contrast with `simulate-biased-mdp`, which re-plans from delay 0 each REAL
   step. For a Naive agent the two DIVERGE — its plan reaches the patient goal while
   it actually defects to the nearby temptation — which is exactly time-inconsistency.
   For a Sophisticated (or k=0) agent plan == do. Deterministic: `argmax-of` breaks
   ties by the last max index — the biased worlds here have a strict argmax at every
   act-state, so the tie-break never bites (the stochastic policies use the uniform
   tie-break of helpers/softmax-action instead)."
  [{:keys [mdp eu params]} start horizon]
  (let [{:keys [T terminals A]} mdp
        pd-fn (bias->perceived-delay (:bias params))
        H     (:horizon params)]
    (loop [s start, d 0, step 0, states [start], actions []]
      (if (or (>= step horizon) (contains? terminals s))
        {:states states :actions actions}
        (let [a  (argmax-of (mapv #(eu s % H d) (range A)))    ; predicted action at policy-delay d
              s' (agent/sample-next T s a)]
          (recur s' (pd-fn d) (inc step) (conj states s') (conj actions a)))))))

;; ===========================================================================
;; Section 4 — World A: the Procrastination MDP (agentmodels 5a/5b)
;; ===========================================================================
;;
;; Time-augmented, non-spatial. State = (loc, waitSteps): wait states W_0..W_D
;; (having waited 0..D steps) and goal states G_0..G_D (reached the reward after
;; working at waitSteps w). Actions: wait(0), work(1).
;;   W_w --wait--> W_{min(w+1,D)}   (cost 0)
;;   W_w --work--> G_w              (immediate cost work-cost)
;;   G_w terminal, payoff = reward + w·wait-cost     (wait-cost < 0 erodes it)
;; Working pays its cost NOW (delay d, ~undiscounted) but the reward lands one
;; step later (delay d+1, discounted) — so a present-biased agent defers. The
;; Naive agent keeps believing tomorrow-self will work; Sophisticated foresees the
;; spiral and works early; the k=0 agent works on day 0.

(defn procrastination-mdp
  "Build the Procrastination MDP. Options (agentmodels defaults):
     :reward 4.5  :work-cost -1.0  :wait-cost -0.1  :deadline 10.
   Returns an MDP map (MLX :T/:R/:term, :terminals) feeding biased-eu directly,
   plus bookkeeping (:nW :deadline :w-idx :g-idx)."
  [{:keys [reward work-cost wait-cost deadline]
    :or   {reward 4.5 work-cost -1.0 wait-cost -0.1 deadline 10}}]
  (let [D     deadline
        nW    (inc D)                       ; W_0..W_D
        S     (* 2 nW)                       ; + G_0..G_D
        A     2                              ; 0 = wait, 1 = work
        w-idx (fn [w] w)
        g-idx (fn [w] (+ nW w))
        ns-fn (fn [s a]
                (if (< s nW)
                  (if (= a 0) (w-idx (min (inc s) D)) (g-idx s))   ; wait→W_{s+1}; work→G_s
                  s))                                              ; goal absorbing
        Trows (vec (for [s (range S)]
                     (vec (for [a (range A)]
                            (let [s' (ns-fn s a)]
                              (mapv #(if (= % s') 1.0 0.0) (range S)))))))
        T     (mx/array (clj->js Trows) mx/float32)
        R     (mx/array
                (clj->js (vec (for [s (range S)]
                                (if (< s nW)
                                  [0.0 work-cost]                          ; W_s: wait, work
                                  (let [w (- s nW)]
                                    (repeat A (+ reward (* w wait-cost))))))))  ; G_w payoff
                mx/float32)
        term  (mx/array (clj->js (vec (for [s (range S)] (if (>= s nW) 1.0 0.0)))) mx/float32)
        terminals (into {} (for [w (range nW)] [(g-idx w) :done]))]
    (mx/eval! T R term)
    {:S S :A A :T T :R R :term term :terminals terminals
     :nW nW :deadline D :w-idx w-idx :g-idx g-idx
     :reward reward :work-cost work-cost :wait-cost wait-cost}))

(defn procrastination-work-day
  "Roll the agent out from W_0 for `deadline` steps and return the step index at
   which it first chose `work` (= the waitSteps at which it worked), or nil if it
   never worked (procrastinated past the deadline)."
  [{:keys [mdp] :as ag}]
  (let [{:keys [actions]} (simulate-biased-mdp ag 0 (:deadline mdp))]
    (first (keep-indexed (fn [i a] (when (= a 1) i)) actions))))

;; ===========================================================================
;; Section 5 — World B: the Restaurant-Choice grid (agentmodels 5a)
;; ===========================================================================
;;
;; 8×6 grid. Donut-North and Donut-South have IDENTICAL utility; Vegetarian Cafe
;; has higher utility but the SHORT route to it passes adjacent to Donut-North,
;; while a LONGER route reaches veg without ever being adjacent to a donut. The
;; rational (k=0) agent takes the short route to veg (no temptation). The Naive
;; agent heads for veg, is captured by Donut-North when adjacent, and ends there.
;; The Sophisticated agent foresees the capture and takes the long route to veg.
;; (Final wall/utility values tuned so the §3 test assertions hold.)

(def restaurant-grid
  "The 8×6 Restaurant-Choice grid literal (rows top-first; screen coords).
   Two corridors reach Veg (top): a SHORT left corridor (col 1) whose climb passes
   adjacent to the Donut-North pocket (2,2), and a LONGER right corridor (col 6)
   that avoids it. Donut-South is a reachable equal-utility DECOY in a dead-end
   pocket (4,1) near Veg: no agent picks it (Veg dominates that close), but it is a
   genuine equal-reward terminal. (Placing Donut-South near the start — agentmodels'
   exact layout, making the 'Naive ends at a donut even though Donut-South is equally
   good and closer' inefficiency vivid — needs a larger grid so its approach touches
   neither route; that is a pe5l refinement, not required for the bias test.)"
  [[:wall  :empty   :empty   :veg   :empty   :empty :empty :wall]
   [:wall  :empty   :wall    :wall  :donut-s :wall  :empty :wall]
   [:wall  :empty   :donut-n :wall  :wall    :wall  :empty :wall]
   [:wall  :empty   :wall    :wall  :wall    :wall  :empty :wall]
   [:wall  :empty   :wall    :wall  :wall    :wall  :empty :wall]
   [:wall  :empty   :empty   :empty :empty   :empty :empty :wall]])

(defn restaurant-mdp
  "Build the Restaurant-Choice MDP. Options:
     :utilities (default {:veg 3.0 :donut-n 1.0 :donut-s 1.0 :timeCost -0.05})
     :start (default [3 5])  :gamma (default 1.0)  :noise (default 0.0)."
  [{:keys [utilities start gamma noise]
    :or   {utilities {:veg 3.0 :donut-n 1.0 :donut-s 1.0 :timeCost -0.05}
           start [3 5] gamma 1.0 noise 0.0}}]
  (gw/build-mdp {:grid restaurant-grid :utilities utilities
                 :start start :gamma gamma :noise noise}))

(defn restaurant-endpoint
  "Roll the agent out from the grid start and return the restaurant keyword it
   ends on (or nil if it never reaches a terminal)."
  [{:keys [mdp] :as ag} start-idx horizon]
  (let [{:keys [states]} (simulate-biased-mdp ag start-idx horizon)]
    (get (:terminals mdp) (last states))))

;; ===========================================================================
;; Section 6 — World C: a reward-myopia line MDP (agentmodels 5c)
;; ===========================================================================
;;
;; A 1-row corridor with a small near reward and a big far reward. A C_g-bounded
;; agent that can only see C_g steps ahead misses the far reward and takes the
;; near one; the unbounded agent takes the far reward.

(def line-grid
  "1×7 corridor: small reward (idx 0) ── start (idx 1) ──→ big reward (idx 6)."
  [[:small :empty :empty :empty :empty :empty :big]])

(defn line-mdp
  "Build the reward-myopia corridor MDP. small near (1 step left of start),
   big far (5 steps right). Options :utilities :start (default [1 0])."
  [{:keys [utilities start gamma]
    :or   {utilities {:small 1.0 :big 5.0 :timeCost -0.05} start [1 0] gamma 1.0}}]
  (gw/build-mdp {:grid line-grid :utilities utilities :start start :gamma gamma :noise 0.0}))

;; ===========================================================================
;; Section 7 — World D: update-myopia (bounded VOI) over a belief-space POMDP
;; ===========================================================================
;;
;; Update-myopia (agentmodels 5c) is a POMDP bias: the agent plans AS IF it will
;; stop updating its belief after C_m steps, so it under-values information it can
;; only gather farther ahead (it under-explores). Formally the simulated future
;; belief uses  b'(s') ∝ I_{d<C_m}(s',a,o)·Σ_s T(s,a,s')·b(s)  where the
;; observation factor O(s',a,o) is kept while d < C_m and REPLACED BY 1 (zeroed as
;; information) once d ≥ C_m. C_m=∞ recovers the optimal information-valuing agent;
;; C_m=0 never updates in look-ahead (no VOI at all).
;;
;; This needs a genuine BELIEF-SPACE look-ahead that values information — which the
;; QMDP `make-pomdp-agent` deliberately does NOT (QMDP assumes the world is revealed
;; after one step). So this is a new faithful Ch-3c belief-space planner too (it
;; also advances bean gw5s). pomdp.cljs's QMDP path is left untouched.

(defn- obs-dist
  "Distribution over observations at next-state s' under `belief`:
   {o → Σ_{w: observe(w,s')=o} belief[w]}. `o` may be nil (uninformative cell)."
  [worlds observe belief s']
  (reduce (fn [m [bw w]]
            (if (pos? bw) (update m (observe w s') (fnil + 0.0) bw) m))
          {} (map vector belief worlds)))

(defn- bayes-update
  "Exact Bayes filter on a discrete world belief: b'(w) ∝ b(w)·[observe(w,s')=o].
   With a deterministic reveal this collapses to the revealed world (and is the
   identity when o=nil, since every world is consistent). Returns a prob vector
   aligned to `worlds`. If o is impossible under b, keeps b (defensive)."
  [worlds observe belief s' o]
  (let [raw (mapv (fn [bw w] (if (= (observe w s') o) bw 0.0)) belief worlds)
        z   (reduce + raw)]
    (if (pos? z) (mapv #(/ % z) raw) belief)))

(defn- build-biased-eu-belief
  "Belief-space delay-indexed biased EU (the faithful information-valuing Ch-3c
   look-ahead, extended with δ(d), perceivedDelay, C_g and the C_m update gate).
   `env` = {:worlds :Th :Rwh :terminals :A :gamma :alpha :observe}. Returns
   {:eu (fn [belief s a t d]) :backup (fn [belief s t dv dp])}. The belief (a prob
   vector aligned to :worlds) is part of the with-cache key; with deterministic
   reveals only finitely many beliefs are reachable, so the cache stays small."
  [{:keys [worlds Th Rwh terminals A gamma alpha observe]}
   {:keys [discount bias reward-myopic-bound update-myopic-bound]
    :or   {discount 0.0 bias :sophisticated reward-myopic-bound ##Inf update-myopic-bound ##Inf}}
   backup-kind]
  (let [k       (double discount)
        gamma   (or gamma 1.0)
        cg      reward-myopic-bound
        cm      update-myopic-bound
        pd-fn   (bias->perceived-delay bias)
        terms   (set (keys terminals))
        eu-atom (atom nil)
        ;; I_{d<C_m}: keep the observation factor while d<C_m, else ignore it (the
        ;; simulated future self stops learning — push-forward only, which for a
        ;; world-independent transition leaves the belief unchanged).
        gated   (fn [belief s' o d]
                  (if (< d cm) (bayes-update worlds observe belief s' o) belief))
        backup  (fn [belief s t dv dp]
                  (let [eu    @eu-atom
                        q-pol (mapv #(eu belief s % t dp) (range A))
                        q-val (if (== dp dv) q-pol (mapv #(eu belief s % t dv) (range A)))]
                    (case backup-kind
                      :soft (let [w (softmax-vec (mapv #(* alpha %) q-pol))]
                              (reduce + (map * w q-val)))
                      :hard (let [m    (apply max q-pol)
                                  idxs (filterv #(> (nth q-pol %) (- m 1e-9)) (range A))]
                              (/ (reduce + (map #(nth q-val %) idxs)) (count idxs))))))
        eu      (exact/with-cache
                  (fn [belief s a t d]
                    (let [u (* (delta k d)
                              (reduce + (map (fn [bw Rh] (* bw (get-in Rh [s a]))) belief Rwh)))]
                      (if (or (terms s) (<= t 1) (>= d cg))
                        u
                        (+ u (* gamma
                                (reduce-kv
                                  (fn [acc s' pr]
                                    (if (pos? pr)
                                      (+ acc (* pr
                                                ;; expectation over the observation at s'
                                                (reduce-kv
                                                  (fn [acc2 o po]
                                                    (+ acc2 (* po (backup (gated belief s' o d)
                                                                          s' (dec t) (inc d) (pd-fn d)))))
                                                  0.0 (obs-dist worlds observe belief s'))))
                                      acc))
                                  0.0 (get-in Th [s a])))))))) ]
    (reset! eu-atom eu)
    {:eu eu :backup backup}))

(defn make-biased-pomdp-agent
  "Belief-space biased POMDP agent (update-myopia / bounded VOI). Builds one MDP
   per latent world (shared geometry T, per-world reward R_w) via gridworld, then a
   belief-space look-ahead with the bias knobs. The `bias` map adds
   :update-myopic-bound C_m (default ##Inf = optimal info-valuing agent; small C_m =
   under-explores). Returns {:worlds :prior :observe :signpost :start-idx :eu :act
   :update-belief :params}. `:act` plans on the (C_m-gated) belief look-ahead;
   `:update-belief` is the REAL (ungated) filter the actual rollout uses — the gate
   only biases PLANNING, never the true belief."
  [{:keys [grid goals signpost observe start prior alpha gamma n-iters high low time-cost noise]
    :or   {goals [:A :B] start [0 0] alpha ##Inf gamma 1.0 n-iters 24
           high 5.0 low 0.0 time-cost -0.05 noise 0.0}}
   {:keys [discount bias reward-myopic-bound update-myopic-bound]
    :or   {discount 0.0 bias :sophisticated reward-myopic-bound ##Inf update-myopic-bound ##Inf} :as biasm}]
  (let [worlds    (vec goals)
        mdps      (mapv (fn [w]
                          (gw/build-mdp {:grid grid
                                         :utilities (assoc (zipmap goals (repeat low)) w high :timeCost time-cost)
                                         :start start :gamma gamma :noise noise}))
                        worlds)
        m0        (first mdps)
        T         (:T m0)
        env       {:worlds worlds :Th (mx/->clj T) :Rwh (mapv #(mx/->clj (:R %)) mdps)
                   :terminals (:terminals m0) :A (:A m0) :gamma gamma :alpha alpha :observe observe}
        {:keys [eu]} (build-biased-eu-belief env biasm (if (= alpha ##Inf) :hard :soft))
        H         n-iters
        A         (:A m0)
        prior     (or prior (zipmap goals (repeat (/ 1.0 (count goals)))))
        ;; belief as a NORMALISED prob vector aligned to `worlds` (the recursion's
        ;; belief form). A belief is a distribution; the belief-space recursion
        ;; weights both the immediate utility and the observation continuation by
        ;; these masses, so an un-normalised or partial :prior would silently
        ;; compound a wrong EU. Normalising here makes the agent robust to either.
        prior-vec (let [raw (mapv #(double (get prior % 0.0)) worlds)
                        z   (reduce + raw)]
                    (assert (pos? z) "make-biased-pomdp-agent: :prior must have positive total mass")
                    (mapv #(/ % z) raw))
        eu-row    (fn [bvec s] (mapv #(eu bvec s % H 0) (range A)))
        ;; action selection through the GFI, mirroring make-biased-mdp-agent: a
        ;; softmax(alpha·EU) policy — Boltzmann at finite alpha, argmax at ##Inf.
        policy    (gen [bvec s] (trace :action
                                       (h/softmax-action alpha (mx/array (clj->js (eu-row bvec s)) mx/float32))))
        act       (fn [bvec s] (int (mx/item (:retval (p/simulate (dyn/auto-key policy) [bvec s])))))]
    {:worlds worlds :prior prior :prior-vec prior-vec :observe observe :signpost signpost
     :start-idx (:start-idx m0) :T T :terminals (:terminals m0)
     :eu eu :act act :policy policy
     ;; the ACTUAL belief filter (ungated): the agent really does keep learning.
     :update-belief (fn [bvec s' o] (bayes-update worlds observe bvec s' o))
     :params {:alpha alpha :gamma gamma :horizon H :discount discount :bias bias
              :reward-myopic-bound reward-myopic-bound :update-myopic-bound update-myopic-bound}}))

(defn simulate-biased-pomdp
  "Roll the belief-space biased POMDP agent out from `start` (state idx) under the
   TRUE world. Each step: ACT from belief; transition via the world geometry;
   OBSERVE at the new cell; FILTER (real, ungated). Returns
   {:states :actions :observations :beliefs} (beliefs as prob vectors)."
  [{:keys [act update-belief observe T terminals]} true-world start horizon prior-vec]
  (loop [s start, b prior-vec, step 0, states [start], actions [], obss [], beliefs [prior-vec]]
    (if (or (>= step horizon) (contains? terminals s))
      {:states states :actions actions :observations obss :beliefs beliefs}
      (let [a  (act b s)
            s' (agent/sample-next T s a)
            o  (observe true-world s')
            b' (update-belief b s' o)]
        (recur s' b' (inc step) (conj states s') (conj actions a) (conj obss o) (conj beliefs b'))))))

(defn voi-world
  "A small 'walk-and-check' POMDP where information is a deliberate DETOUR. Two
   goals A (idx 0) and B (idx 4) sit at opposite ends of the top row; a signpost
   one step BELOW the start reveals which goal is rewarding. The optimal (large
   C_m) agent values the 1-step detour to the signpost and then commits to the
   right goal; an update-myopic (small C_m) agent sees no value in the detour and
   commits to the higher-prior goal blind. Options :prior :true-world."
  [{:keys [prior true-world] :or {prior {:A 0.55 :B 0.45} true-world :B}}]
  (let [grid     [[:A :empty :empty :empty :B]
                  [:empty :empty :empty :empty :empty]]
        W        5
        signpost (+ 2 (* W 1))]                 ; (2,1) — one step below the (2,0) start
    {:grid grid :goals [:A :B] :start [2 0] :signpost signpost
     :prior prior :true-world true-world
     :observe (fn [world loc] (when (= loc signpost) world))}))
