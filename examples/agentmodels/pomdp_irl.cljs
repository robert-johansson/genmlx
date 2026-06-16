(ns agentmodels.pomdp-irl
  "POMDP Inverse Reinforcement Learning (agentmodels Ch 4, 'Learning about agents in
   POMDPs') — jointly inferring an agent's UTILITIES and its BELIEFS from observed
   actions in a partially-observed world, and the preference-vs-belief
   UNIDENTIFIABILITY that results.

   agentmodels' Equation 2 (the POMDP analog of Equation 1):

       P(U,α,b₀ | (s,o,a)_{0:n}) ∝ P(U,α,b₀) · Π_i P(a_i | s_i, b_i, U, α)

   The belief b_i is THREADED through the observed trajectory (b_i = b_{i-1} | s_i,o_i,
   a_{i-1}) — agentmodels' `factorSequence`. Here that thread is a host-side recursion
   (the literal factorSequence), with the per-step action likelihood P(a_i|b_i,U,α)
   scored by the proven GFI path inverse/action-loglik (`p/assess` on the agent's
   softmax policy). The two-level prior over (utility, initial belief) is expressed
   with the Switch combinator (select-belief-branch); the belief thread is also
   exposed as a Scan in simulate mode. (We deliberately do NOT route the likelihood
   through p/generate over a tracing combinator — see bug genmlx-7bm6 — and instead
   score via assess, which is robust.)

   Testbed (the canonical agentmodels example): a 2-armed bandit. arm0 deterministically
   yields a KNOWN prize (chocolate); arm1 yields champagne (true) or nothing. The agent
   has prize utilities (likes-champagne or likes-chocolate) and a belief about arm1
   (informed = knows champagne; misinformed = thinks arm1 is probably nothing). A
   finite-horizon belief-space VOI planner: with few trials left, exploring arm1 is not
   worth it, so a misinformed champagne-lover still pulls arm0 — making the observed
   arm0-pull explainable by EITHER a chocolate preference OR a misinformed belief
   (unidentifiable). As the horizon grows, exploration becomes valuable, so a
   misinformed champagne-lover WOULD explore — and persistent arm0-pulls then identify
   a genuine chocolate preference.

   Reuse: inverse/action-loglik + normalize-logs, helpers/softmax-action, the Switch &
   Scan combinators. Zero engine change."
  (:require [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.combinators :as comb]
            [genmlx.agents.helpers :as h]
            [genmlx.agents.inverse :as inv])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ===========================================================================
;; Bandit prizes + the agent's utility hypotheses (agentmodels)
;; ===========================================================================
(def likes-champagne {:nothing 0.0 :champagne 5.0 :chocolate 3.0})
(def likes-chocolate {:nothing 0.0 :champagne 3.0 :chocolate 5.0})

;; arm 0 always yields :chocolate; arm 1 yields :champagne (true world) or :nothing.
;; belief b = P(arm1 = :champagne). informed → 1.0; misinformed → small p.
;; A misinformed belief of ~0.3 puts the exploration threshold inside a short horizon
;; (so the identifiability-vs-horizon effect is visible in N ∈ [4..9]); agentmodels
;; uses 0.05 with a much longer horizon. (Documented parameter choice.)
(def MISINFORMED-P 0.3)

;; ===========================================================================
;; Finite-horizon belief-space VOI value (exact, memoized; tiny reachable belief set)
;; ===========================================================================

(defn make-bandit-voi-agent
  "Exact finite-horizon belief-space VOI bandit agent. `utility` is a prize→value map;
   `belief` = P(arm1 = :champagne); `alpha` = softmax rationality. Returns
     {:q (fn [arm b n] -> Q) :policy (gen [b n]) :act-loglik (fn [b n arm] -> logp)}.
   Reachable beliefs are {belief, 0.0, 1.0} (arm0 reveals nothing; arm1 reveals arm1),
   so the recursion is small. Pulling arm0 yields :chocolate; pulling arm1 yields its
   true prize and REVEALS arm1 (belief → 0 or 1).

   NOTE: this exact belief-space VOI planner lives OUTSIDE the frozen four-constructor
   bandit family (it does not call pomdp/make-bandit-agent) — it is a chapter-local
   agent specialized to the 2-armed reveal-on-pull structure of this example."
  [{:keys [utility belief alpha] :or {alpha 1000.0}}]
  (let [u      (fn [prize] (double (get utility prize 0.0)))
        u-choc (u :chocolate)
        ;; exploit value once arm1 is revealed: pull the better arm for all n trials
        exploit (fn [arm1-prize n] (* n (max u-choc (u arm1-prize))))
        q-atom (atom nil)
        soft-v (fn [b n]
                 (if (<= n 0) 0.0
                     (let [q0 (@q-atom 0 b n)
                           q1 (@q-atom 1 b n)
                           m  (max q0 q1)
                           e0 (Math/exp (* alpha (- q0 m)))
                           e1 (Math/exp (* alpha (- q1 m)))
                           z  (+ e0 e1)]
                       (+ (* (/ e0 z) q0) (* (/ e1 z) q1)))))
        q     (memoize
                (fn [arm b n]
                  (case arm
                    0 (+ u-choc (soft-v b (dec n)))                          ; arm0: known, belief unchanged
                    1 (+ (+ (* b (u :champagne)) (* (- 1.0 b) (u :nothing))) ; arm1: expected prize now
                         (* b (exploit :champagne (dec n)))                   ; + exploit after reveal
                         (* (- 1.0 b) (exploit :nothing (dec n)))))))]
    (reset! q-atom q)
    (let [policy (gen [b n] (trace :arm (h/softmax-action alpha
                                          (mx/array #js [(q 0 b n) (q 1 b n)] mx/float32))))]
      {:q q
       :policy policy
       :act-loglik (fn [b n arm]
                     (mx/item (:weight (p/assess (dyn/auto-key policy) [b n] (cm/choicemap :arm arm)))))})))

;; ===========================================================================
;; factorSequence — thread belief through observed pulls, accumulate action log-lik
;; ===========================================================================

(defn reveal-belief
  "Belief update on an observed pull: arm0 reveals nothing about arm1; arm1 reveals it
   (belief collapses to 1.0 for :champagne, 0.0 otherwise)."
  [belief arm prize]
  (if (= arm 0) belief (if (= prize :champagne) 1.0 0.0)))

;; NOTE: this mirrors inverse/action-loglik but scores the :arm policy trace site
;; (not :action) — the deliberate bandit-domain naming for the agent's pull choice.
(defn factor-sequence-loglik
  "agentmodels' factorSequence (Equation 2): thread the agent's belief through the
   observed sequence and sum the per-step action log-likelihoods. `observed` is a seq
   of {:arm :prize} pulls; `horizon` is the total number of trials (so the pull at
   index t is scored at the remaining horizon horizon-t, delay 0). The belief updates
   on the observed prize (arm1 reveals arm1; arm0 reveals nothing)."
  [agent initial-belief horizon observed]
  (loop [obs observed, t 0, b initial-belief, ll 0.0]
    (if (empty? obs)
      ll
      (let [{:keys [arm prize]} (first obs)
            n   (- horizon t)
            ll' (+ ll ((:act-loglik agent) b n arm))]
        (recur (rest obs) (inc t) (reveal-belief b arm prize) ll')))))

;; ===========================================================================
;; Two-level prior (utility × initial belief) via the Switch combinator
;; ===========================================================================
;; The initial belief is selected by a Switch over two branches (informed /
;; misinformed); the utility is selected alongside. We enumerate the 2×2 joint exactly.

(def utility-hyps [[:champagne likes-champagne] [:chocolate likes-chocolate]])
(def belief-hyps  [[:informed 1.0] [:misinformed MISINFORMED-P]])

;; Switch realization of the two-level belief prior: each branch returns its initial
;; belief value (a deterministic GF), the Switch index selects which.
;; These branches are keyed once at load time (dyn/auto-key) and reused as fixed GFs.
(def ^:private belief-branch-informed    (dyn/auto-key (gen [] 1.0)))
(def ^:private belief-branch-misinformed (dyn/auto-key (gen [] MISINFORMED-P)))
(def belief-prior-switch
  (comb/switch-combinator belief-branch-informed belief-branch-misinformed))

(defn switch-initial-belief
  "Select the initial belief through the Switch combinator (idx 0 = informed,
   1 = misinformed). Demonstrates the two-level prior's belief branch as a GF.
   The branch retval is a plain double, so no mx/item extraction is needed."
  [idx]
  (:retval (p/simulate (dyn/auto-key belief-prior-switch) [idx])))

(defn- marginal-prob
  "Sum the probabilities in the joint posterior over entries whose [util-kw belief-kw]
   key satisfies `pred`."
  [post pred]
  (reduce (fn [s [k pr]] (+ s (if (pred k) pr 0.0))) 0.0 post))

(defn joint-posterior
  "Exact P(utility, initial-belief | observed pulls) over the 2×2 joint prior, via the
   factorSequence likelihood. Returns
     {:joint {[util-kw belief-kw] -> prob}
      :p-likes-chocolate p :p-informed p}."
  [{:keys [observed horizon alpha] :or {alpha 1000.0}}]
  (let [logw (into {}
               (for [[uk utility] utility-hyps
                     [bk b0]      belief-hyps]
                 (let [agent (make-bandit-voi-agent {:utility utility :belief b0 :alpha alpha})]
                   [[uk bk] (factor-sequence-loglik agent b0 horizon observed)])))
        post (inv/normalize-logs logw)]
    {:joint post
     :p-likes-chocolate (marginal-prob post (fn [[uk _]] (= uk :chocolate)))
     :p-informed        (marginal-prob post (fn [[_ bk]] (= bk :informed)))}))

;; ===========================================================================
;; Observations + analyses
;; ===========================================================================

(defn all-arm0
  "The observed sequence: the agent pulls arm0 (the known arm) every trial. arm0
   always yields :chocolate, so the belief never updates."
  [n]
  (vec (repeat n {:arm 0 :prize :chocolate})))

(defn unidentifiability
  "Observe all-arm0 over a short horizon: the pull is explained by EITHER a chocolate
   preference OR a misinformed belief about arm1, so neither is identified."
  [horizon]
  (joint-posterior {:observed (all-arm0 horizon) :horizon horizon}))

(defn horizon-sweep
  "P(likes-chocolate | all-arm0) as the horizon grows. With more trials, exploring
   arm1 becomes valuable, so a misinformed champagne-lover WOULD explore — persistent
   arm0-pulls then identify a genuine chocolate preference."
  [horizons]
  (mapv (fn [N] {:horizon N :p-likes-chocolate (:p-likes-chocolate (unidentifiability N))}) horizons))

;; ===========================================================================
;; factorSequence's belief thread, realized via the Scan combinator (simulate mode)
;; ===========================================================================
;; The kernel carries the belief and updates it on each observed prize; the per-step
;; OUTPUT is the belief held BEFORE the pull (the belief the action was chosen under) —
;; exactly the belief factor-sequence-loglik scores against. Run in simulate mode (no
;; constraints) so it is unaffected by the p/generate-over-combinator bug (genmlx-7bm6).

;; The kernel is keyed once at load time (dyn/auto-key) and reused as a fixed GF.
(def ^:private belief-scan-kernel
  (dyn/auto-key (gen [belief obs] [(reveal-belief belief (:arm obs) (:prize obs)) belief])))
(def ^:private belief-scan (comb/scan-combinator belief-scan-kernel))

(defn belief-trajectory-via-scan
  "The belief held at each pull, threaded through the Scan combinator (carry = belief).
   Demonstrates factorSequence's belief threading as a GFI combinator."
  [initial-belief observed]
  (:outputs (:retval (p/simulate (dyn/auto-key belief-scan) [initial-belief (vec observed)]))))

(defn belief-trajectory-host
  "Reference host-side belief thread (the same recursion factor-sequence-loglik uses)."
  [initial-belief observed]
  (loop [obs observed, b initial-belief, acc []]
    (if (empty? obs) acc
        (let [{:keys [arm prize]} (first obs)]
          (recur (rest obs) (reveal-belief b arm prize) (conj acc b))))))
