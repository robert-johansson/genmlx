(ns agentmodels.ch06-pluggable-inference
  "agentmodels.org Chapter 6 (\"Efficient inference\") — authored NATIVELY as the
   genmlx.agents pluggable-inference chapter.

   ════════════════════════════════════════════════════════════════════════════
   THE THROUGH-LINE: planning is inference; the backend is a swap.
   ════════════════════════════════════════════════════════════════════════════

   agentmodels.org treats the inference method as fixed scaffolding around each
   model. GenMLX's differentiator is the opposite stance, and it is the heart of
   `genmlx.agents`: an agent IS a generative function, and the inference backend
   used to reason about it — exact enumeration, importance sampling, MCMC, or
   gradient/amortized — is a PLUGGABLE seam, orthogonal to the agent definition.
   This is the inference analog of GenMLX's compilation creed: \"the handler is
   ground truth, compilation is optimization.\" Here: the agent GF is ground
   truth, and every inference backend is just a different way to read its
   posterior off — they must AGREE.

   This chapter takes ONE forward gridworld agent and reasons about it four ways:

     (6a) EFFICIENT EXACT — forward planning, two backends.  The agent's plan is
          computed by tensor value iteration (:Q) AND by the faithful recursive
          expected-utility (:expected-utility).  They agree to float32.  Even
          FORWARD planning has a backend swap (dynamic programming vs recursion).

     (6b) SAMPLING — inverse planning, three backends that AGREE.  Observe the
          agent act; infer which goal it values.  The posterior P(goal | actions)
          is read three ways on the SAME joint generative function:
            • EXACT      — enumerate the finite goal prior; p/assess the full
                           trajectory; normalize.  Closed form (ground truth).
            • IMPORTANCE — sample the goal, constrain the actions, group weights.
            • MCMC (MH)  — Metropolis-Hastings over the goal latent.
          All three converge to the same posterior (a single sampling run is an
          ESTIMATE, never the posterior).  An INDEPENDENT exact oracle
          (inverse/posterior-sequence — a genuinely different code path: per-cell
          action-loglik, no joint GF) cross-checks the closed form, and the
          posterior is shown SHARPENING one observed action at a time.

     (6c) AMORTIZED / GRADIENT — when the latent is CONTINUOUS (a utility vector
          rather than a discrete goal), the backend swaps again: recover the
          agent's utilities by Adam through the differentiable planner
          (genmlx.agents.differentiable).  Same forward agent, gradient backend.

   ZERO engine change: every backend is an existing genmlx.inference / agents
   piece composed on the one forward agent GF.  Compose, don't duplicate.

   Run: bun run --bun nbb examples/agentmodels/ch06_pluggable_inference.cljs
   Self-checks on load (asserts every claim; exits non-zero on any failure)."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.importance :as is]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.util :as iu]
            [genmlx.agents.gridworld :as gw]
            [genmlx.agents.helpers :as h]
            [genmlx.agents.inverse :as inv]
            [genmlx.agents.differentiable :as diff]
            [agentmodels.harness :as chk])
  (:require-macros [genmlx.gen :refer [gen]]))

;; Self-checks use the shared agentmodels.harness (chk/check-true,
;; chk/check-close, chk/report!) — one copy of the assert helpers across the
;; chapter scripts.

;; ===========================================================================
;; The world: a small 5×5 gridworld with three distinct goals
;; ===========================================================================
;;
;; actions (gw/action-deltas): 0=left 1=right 2=up 3=down ; state idx = x + W*y.
;;   :a at (0,0)=idx0   — UP-LEFT of the start
;;   :b at (0,4)=idx20  — DOWN-LEFT of the start  (shares "left" with :a)
;;   :c at (4,2)=idx14  — RIGHT of the start
;; start (2,2)=idx12 (centre).  :a and :b both lie to the LEFT, so leftward moves
;; are ambiguous between them (and rule out :c); only a vertical move disambiguates
;; :a (up) from :b (down).  This makes the posterior SHARPEN in two stages — the
;; pedagogical payoff of an incremental Bayesian filter.

(def grid
  [[:a     :empty :empty :empty :empty]
   [:empty :empty :empty :empty :empty]
   [:empty :empty :empty :empty :c    ]
   [:empty :empty :empty :empty :empty]
   [:b     :empty :empty :empty :empty]])

(def goals      [:a :b :c])      ; pinned order = the latent index axis
(def start      [2 2])
(def start-idx  12)              ; 2 + 5*2
(def ALPHA      2.0)             ; finite Boltzmann rationality (argmax makes IS/MH degenerate)
(def HIGH       5.0)             ; utility of the valued goal
(def LOW        0.0)             ; utility of the other goals
(def TIME-COST  -0.1)
(def N-ITERS    40)              ; value-iteration horizon — the SINGLE source of
                                 ; truth, threaded into BOTH the discrete-goal
                                 ; forward agents (6a/6b, via inv/goal-agents) and
                                 ; the differentiable planner (6c). One agent, one
                                 ; horizon, four backends.

;; One forward MDP agent per candidate goal (the agent valuing that goal). Each
;; carries :Q (tensor value iteration) and :policy (softmax gen fn). Built once.
(def goal-agents
  (inv/goal-agents {:grid grid :goals goals :high HIGH :low LOW
                    :time-cost TIME-COST :alpha ALPHA :start start :n-iters N-ITERS}))

;; A diagnostic observed trajectory heading for :a (up-left): left, left, up.
;;   (2,2)→(1,2)→(0,2)→(0,1) .  The two "left"s are ambiguous between :a and :b;
;;   the final "up" disambiguates toward :a (and away from :b at the bottom-left).
(def obs-states  [12 11 10])     ; non-terminal states where an action is taken
(def obs-actions [0  0  2])      ; left left up
(def observations (mapv vector obs-states obs-actions))

;; ===========================================================================
;; (6a) EFFICIENT EXACT — forward planning by two backends that agree
;; ===========================================================================
;;
;; make-mdp-agent solves the plan by tensor value iteration (:Q [S,A]) and ALSO
;; exposes the faithful recursive expected-utility (:expected-utility (s a)).
;; Q[s,a] is exactly EU(s,a) at the same horizon — DP and recursion agree. This
;; is the simplest instance of \"the backend is a swap\": forward planning itself.

(defn check-6a-vi-eq-recursive []
  (println "\n-- (6a) Efficient exact: value iteration :Q == recursive expected-utility --")
  (let [ag       (goal-agents :a)
        Q        (:Q ag)
        eu       (:expected-utility ag)
        ;; compare over the non-terminal observed states (and both actions there)
        max-diff (reduce max 0
                         (for [s obs-states a [0 1 2 3]]
                           (Math/abs (- (mx/item (mx/idx (mx/idx Q s) a)) (eu s a)))))]
    (chk/check-true "value iteration Q[s,a] == recursive EU(s,a) to 1e-3"
                    (< max-diff 1e-3))
    (println (str "       max |Q - EU| over observed (s,a) = " (.toExponential max-diff 2)))))

;; ===========================================================================
;; The joint inverse model — :goal is a first-class traced latent
;; ===========================================================================
;;
;; ONE generative function.  The latent :goal (a categorical INDEX into `goals`,
;; uniform prior) selects which forward agent generated the behaviour; each
;; observed state then contributes one softmax-action site over THAT agent's Q
;; row.  The action likelihood is the agent's own policy — no bespoke likelihood.
;; Inverting:
;;   EXACT     — assess the full choicemap {:goal i, :a0 a0, …}
;;   SAMPLING  — generate / MH constraining only the actions

(defn goal-inference-model
  "The joint GF (zero user args). Traces :goal ~ uniform(goals); for each observed
   state, traces :a0 :a1 … ~ softmax-action(ALPHA, Q_goal[state]).  The policy
   logits per (goal,state) are precomputed ONCE (the gen body re-runs per IS/MH
   particle, so it must only index, never rebuild arrays).

   Scalar-only by design: `h/draw-value` performs a host-side `mx/item` on the
   sampled :goal index to pick the agent, so this model is NOT shape-batchable
   (the established discrete-latent agents idiom; fine for the per-particle
   exact/IS/MH backends here, which run the body once per particle)."
  []
  (let [box  (h/uniform-draw goals)
        rows (into {} (for [g goals]
                        [g (mapv #(mx/idx (:Q (goal-agents g)) %) obs-states)]))]
    (gen []
      (let [gi   (trace :goal (:dist box))
            goal (h/draw-value box gi)
            er   (rows goal)]
        (doseq [t (range (count obs-states))]
          (trace (keyword (str "a" t))
                 (h/softmax-action ALPHA (nth er t))))
        gi))))

(defn- full-cm
  "Choicemap constraining the full trajectory: {:goal i, :a0 a0, …}."
  [goal-idx actions]
  (cm/set-choice (h/action-choicemap actions) [:goal] goal-idx))

(defn normalize-map
  "{goal -> non-negative weight} -> {goal -> probability}."
  [m]
  (let [z (reduce + (vals m))]
    (if (pos? z) (update-vals m #(/ % z)) m)))

;; Chapter-local total-variation diagnostic over the {goal -> prob} posteriors
;; (kept private — a one-off check, not part of the inverse-inference API).
(defn- tv
  "Total-variation distance between two {goal -> prob} maps."
  [p q]
  (* 0.5 (reduce + (map #(Math/abs (- (get p % 0.0) (get q % 0.0))) goals))))

;; ===========================================================================
;; (6b) BACKEND 1 — EXACT enumeration via p/assess (ground truth)
;; ===========================================================================

(defn exact-goal-posterior
  "Exact P(:goal | actions): enumerate the finite goal prior, assess the joint GF
   on {:goal i, :a0 a0, …} (weight = log P0(goal) + Σ log π_goal(a_t)), normalize.
   Closed form — assess only scores; no sampling.  Returns {:a p :b q :c r}."
  [actions]
  (let [model (goal-inference-model)
        logw  (into {}
                    (for [[i g] (map-indexed vector goals)]
                      [g (mx/item (:weight (p/assess (dyn/auto-key model) []
                                                     (full-cm i actions))))]))]
    (inv/normalize-logs logw)))

;; ===========================================================================
;; (6b) BACKEND 2 — IMPORTANCE SAMPLING
;; ===========================================================================

(defn is-goal-posterior
  "Importance-sampling estimate of P(:goal | actions) on the joint GF: sample
   :goal from the prior, constrain the observed actions, group the normalized
   particle weight by each particle's sampled goal.  Returns {:posterior {..}
   :ess ess}.  Run at finite ALPHA; pass a fixed key for reproducibility."
  [actions n key]
  (let [model (goal-inference-model)
        obs   (h/action-choicemap actions)
        {:keys [traces log-weights]} (is/importance-sampling {:samples n :key key} model [] obs)
        {:keys [probs]} (iu/normalize-log-weights log-weights)
        ess   (iu/compute-ess log-weights)
        post  (reduce (fn [m [tr w]]
                        (let [i (int (mx/item (cm/get-choice (:choices tr) [:goal])))]
                          (update m (nth goals i) + w)))
                      (zipmap goals (repeat 0.0))
                      (map vector traces probs))]
    {:posterior (normalize-map post) :ess ess}))

;; ===========================================================================
;; (6b) BACKEND 3 — METROPOLIS-HASTINGS (MCMC)
;; ===========================================================================

(defn mh-goal-posterior
  "Metropolis-Hastings estimate of P(:goal | actions) on the SAME joint GF.  The
   default selection is the complement of the observed (action) sites — i.e. the
   {:goal} latent — so each step regenerates :goal from its prior and accepts via
   the MH ratio (independent-proposal MH over the discrete latent).  The empirical
   frequency of the sampled goal over the chain estimates the posterior.  Returns
   {:posterior {..} :n-samples n}."
  [actions {:keys [samples burn thin key]}]
  (let [model  (goal-inference-model)
        obs    (h/action-choicemap actions)
        traces (mcmc/mh {:samples samples :burn (or burn 0) :thin (or thin 1) :key key}
                        model [] obs)
        counts (reduce (fn [m tr]
                         (let [i (int (mx/item (cm/get-choice (:choices tr) [:goal])))]
                           (update m (nth goals i) inc)))
                       (zipmap goals (repeat 0))
                       traces)]
    {:posterior (normalize-map counts) :n-samples (count traces)}))

;; ===========================================================================
;; Independent exact oracles (cross-check the closed form)
;; ===========================================================================

(defn posterior-sharpening
  "Independent exact posteriors via the inverse.cljs idiom (one forward agent per
   goal, per-cell action-loglik, accumulated and normalized — no joint GF, a
   genuinely different code path).  Returns a vector of {goal -> prob} maps, one
   per prefix: index 0 is the prior, index k is the posterior after k observed
   actions.  The last entry is the independent exact oracle for the full
   trajectory; the sequence shows the posterior sharpening one action at a time."
  []
  (let [prior (zipmap goals (repeat (/ 1.0 (count goals))))]
    (inv/posterior-sequence goal-agents prior observations)))

;; ===========================================================================
;; (6c) AMORTIZED / GRADIENT — recover a continuous utility vector
;; ===========================================================================
;;
;; Swap the discrete-goal hypothesis for a CONTINUOUS utility vector over the same
;; goals, and the inference backend swaps to gradient ascent: recover the planted
;; utilities by Adam through the differentiable planner (utilities → R → lazy value
;; iteration → Q → log-softmax → action log-likelihood), at fixed rationality.
;; Identifiable up to scale/offset, so success = LIKELIHOOD equivalence (loss at
;; the recovered params ≤ loss at the plant) plus the correct utility ORDERING.

(defn gradient-recovery
  "Recover the agent's utility vector by Adam through the differentiable planner
   (the continuous-latent backend, genmlx.agents.differentiable). Alpha is held at
   the true value (utilities are identifiable only up to a scale × alpha
   interaction, so we fix one and learn the other). Returns
   {:plant-utils :rec-utils :plant-loss :rec-loss :loss-history}."
  []
  (let [dmdp        (diff/build-diff-mdp {:grid grid :goals goals :start start})
        plant-utils [HIGH LOW LOW]                   ; :a valued, :b/:c not
        log-alpha   (Math/log ALPHA)
        plant-loss  (diff/loss-at dmdp TIME-COST plant-utils log-alpha N-ITERS observations)
        res         (diff/recover-params dmdp TIME-COST N-ITERS observations
                                         {:iterations 250 :lr 0.1
                                          :fixed-log-alpha log-alpha
                                          :key (rng/fresh-key 7)})
        rec-utils   (vec (mx/->clj (:theta-u res)))
        rec-loss    (diff/loss-at dmdp TIME-COST rec-utils log-alpha N-ITERS observations)]
    {:plant-utils plant-utils :rec-utils rec-utils
     :plant-loss plant-loss :rec-loss rec-loss
     :loss-history (vec (:loss-history res))}))

(defn check-6c-gradient-recovery []
  (println "\n-- (6c) Amortized/gradient: recover utilities by Adam through the planner --")
  (let [{:keys [plant-utils rec-utils plant-loss rec-loss loss-history]} (gradient-recovery)
        a-util    (first rec-utils)
        max-other (apply max (rest rec-utils))]
    (println (str "       planted utilities  = " plant-utils))
    (println (str "       recovered utilities = " (mapv #(js/Number (.toFixed % 3)) rec-utils)))
    (println (str "       loss: plant=" (.toFixed plant-loss 4) "  recovered=" (.toFixed rec-loss 4)))
    (chk/check-true "recovered :a utility is the largest (correct ordering)"
                    (> a-util max-other))
    (chk/check-true "recovered loss <= plant loss + 1e-2 (likelihood-equivalent)"
                    (<= rec-loss (+ plant-loss 1e-2)))
    (chk/check-true "loss decreased from init (gradient actually descended)"
                    (< (last loss-history) (first loss-history)))))

;; ===========================================================================
;; -main — the chapter self-check
;; ===========================================================================

(defn -main []
  (println "\n=== agentmodels Ch 6 — Planning is inference; the backend is a swap (GenMLX) ===")

  ;; (6a)
  (check-6a-vi-eq-recursive)

  ;; Exact ground truth + the independent oracle, shown sharpening per action
  (println "\n-- Exact posterior P(goal | actions), sharpening one action at a time --")
  (let [sharpen (posterior-sharpening)
        oseq    (last sharpen)
        exact   (exact-goal-posterior obs-actions)]
    (doseq [[k post] (map-indexed vector sharpen)]
      (println (str "       after " k " obs " (if (zero? k) "(prior)    " "          ")
                    ": " (pr-str (zipmap goals (map #(js/Number (.toFixed (get post %) 3)) goals))))))
    (println (str "       exact (assess-enum) final: " (pr-str (normalize-map exact))))
    (chk/check-true "exact posterior favors the true goal :a"
                    (= :a (key (apply max-key val exact))))
    (chk/check-true "posterior is NON-degenerate (:a favored but < 0.99; :b keeps real mass — backends must really agree)"
                    (and (< (get exact :a) 0.99) (> (get exact :b) 0.02)))
    (chk/check-close "independent oracle (posterior-sequence) == exact (TV)" 0.0 (tv exact oseq) 1e-4)

    ;; (6b) the three backends AGREE
    (println "\n-- (6b) Three inference backends on the SAME agent GF, all agreeing --")
    (let [is-res  (is-goal-posterior obs-actions 5000 (rng/fresh-key 42))
          mh-res  (mh-goal-posterior obs-actions {:samples 5000 :burn 500 :key (rng/fresh-key 43)})
          is-post (:posterior is-res)
          mh-post (:posterior mh-res)]
      (println (str "       EXACT      : " (pr-str (normalize-map exact))))
      (println (str "       IMPORTANCE : " (pr-str is-post)
                    "   (ESS " (.toFixed (:ess is-res) 1) ")"))
      (println (str "       MCMC (MH)  : " (pr-str mh-post)
                    "   (" (:n-samples mh-res) " samples)"))
      (chk/check-close "IS posterior == exact (TV < 0.03)"  0.0 (tv exact is-post) 0.03)
      (chk/check-close "MH posterior == exact (TV < 0.05)"  0.0 (tv exact mh-post) 0.05)
      (chk/check-true "IS effective sample size is healthy (> 10% of N)"
                      (> (:ess is-res) 500))))

  ;; (6c)
  (check-6c-gradient-recovery)

  (chk/report!))

;; Auto-run the self-check ONLY when this file is the script entry point (run
;; directly) — so the bench and tests can `require` the public model + backends
;; without triggering the full chapter run.  The test wrapper calls (-main)
;; explicitly to keep the self-check a hard test-failure signal.
;; Match the example's own path ("…/ch06_pluggable_inference.cljs") but NOT the
;; test wrapper ("…_ch06_pluggable_inference_test.cljs") or the bench — so a
;; `require` from either never double-runs the self-check.
(defn- run-as-script? []
  (some #(re-find #"ch06_pluggable_inference\.cljs" (str %)) (.-argv js/process)))

(when (run-as-script?) (-main))
