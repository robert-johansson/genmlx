(ns genmlx.agents.pomdp
  "POMDP agent (agentmodels Ch 3c) — belief filtering + belief-space action,
   GenMLX-native by reuse.

   The agent is uncertain about a discrete latent WORLD (e.g. which goal is the
   rewarding one). It keeps a belief over worlds, acts on that belief, observes,
   and Bayesian-updates the belief (filtering). The pieces all compose on what is
   already built:

   - one MDP planner PER world via genmlx.agents.inverse/goal-agents (each carries
     its solved :Q [S,A]);
   - the belief-space value is QMDP: Q_QMDP(b,s) = Σ_w b(w) · Q_w[s] — a plain MLX
     reduction over the per-world :Q rows;
   - act = softmax-action over that mixed Q-row, scored through the GFI exactly
     like the MDP policy;
   - the belief filter reuses inverse/normalize-logs with an observation
     likelihood (here a deterministic location-gated reveal).

   FAITHFULNESS / SCOPE: this is the QMDP approximation, NOT agentmodels' full
   belief-space lookahead. QMDP assumes uncertainty resolves after one step, so it
   does not value information-gathering. That is acceptable here because the
   slice's observation is gated by GEOMETRY, not chosen as an action — there is no
   explore-to-learn decision whose value QMDP would miss, so QMDP and full
   lookahead pick the same path. Full finite-horizon belief-space planning is a
   future increment. No exact/expectation in any recursion (project ethos)."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.agents.inverse :as inv]
            [genmlx.agents.agent :as agent]
            [genmlx.agents.helpers :as h])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn make-pomdp-agent
  "Build a POMDP agent over a discrete latent world. Options:
     :grid :goals :alpha :noise :gamma :n-iters — passed through to the per-world
       MDP planners (inverse/goal-agents);
     :prior   {world -> prob}  (defaults to uniform over :goals);
     :observe (fn [world loc] -> obs|nil)  the observation model.
   Returns
     {:worlds :world-agents {world->agent} :prior :observe
      :belief-Q (fn [belief s] -> [A] MLX)         ; QMDP belief-space Q row
      :update-belief (fn [belief loc obs] -> belief') ; one Bayes filtering step
      :act (fn [belief s] -> action-int)            ; softmax over belief-Q
      :expected-utility (fn [belief s a] -> float)  ; belief-weighted EU
      :params {...}}."
  [{:keys [grid goals alpha noise gamma prior observe n-iters]
    :or   {alpha 2.0 noise 0.0 gamma 1.0 n-iters 40}}]
  (let [world-agents (inv/goal-agents {:grid grid :goals goals :alpha alpha
                                       :noise noise :gamma gamma})
        A         (:A (:mdp (val (first world-agents))))
        prior     (or prior (zipmap goals (repeat (/ 1.0 (count goals)))))
        ;; QMDP belief-space Q at state s: Σ_w b(w) · Q_w[s]  (plain mx reduction)
        belief-Q  (fn [belief s]
                    (reduce (fn [acc [w b]]
                              (mx/add acc (mx/multiply (mx/scalar b)
                                                       (mx/idx (:Q (world-agents w)) s))))
                            (mx/zeros #js [A]) belief))
        ;; one Bayes filtering step: b'(w) ∝ b(w) · P(obs | w, loc).
        ;; obs = nil (uninformative location) => belief unchanged (flat-then-snap).
        update-belief (fn [belief loc obs]
                        (if (nil? obs)
                          belief
                          (inv/normalize-logs
                            (into {} (map (fn [[w b]]
                                            [w (+ (Math/log b)
                                                  (if (= (observe w loc) obs) 0.0 ##-Inf))])
                                          belief)))))
        ;; act = softmax-action over the belief-mixed Q row, as a real GFI choice
        act (fn [belief s]
              (let [q      (belief-Q belief s)
                    policy (gen [] (trace :action (h/softmax-action alpha q)))]
                (int (mx/item (:retval (p/simulate (dyn/auto-key policy) []))))))]
    {:worlds (vec goals) :world-agents world-agents :prior prior :observe observe
     :belief-Q belief-Q :update-belief update-belief :act act
     :expected-utility (fn [belief s a]
                         (reduce + (map (fn [[w b]]
                                          (* b ((:expected-utility (world-agents w)) s a)))
                                        belief)))
     :params {:alpha alpha :gamma gamma :noise noise :horizon n-iters}}))

(defn simulate-pomdp
  "Roll the POMDP agent out from `start` over a finite horizon, threading
   (state, belief, action, observation). Each step: (1) the agent ACTS from its
   current belief; (2) the world transitions via the TRUE world's T; (3) the agent
   OBSERVES at the new location; (4) it FILTERS its belief. Returns
     {:states [...] :actions [...] :observations [...] :beliefs [b0 b1 ...]}
   where :beliefs index k is the belief held at state k (when choosing action k) —
   so :states and :beliefs align and both feed the seam (env->trajectory / dist->bars)."
  [{:keys [act update-belief observe world-agents prior]} env start horizon]
  (let [true-world (:true-world env)
        true-mdp   (:mdp (world-agents true-world))
        T          (:T true-mdp)
        terminals  (:terminals true-mdp)]
    (loop [s start, b prior, step 0
           states [start], actions [], obss [], beliefs [prior]]
      (if (or (>= step horizon) (contains? terminals s))
        {:states states :actions actions :observations obss :beliefs beliefs}
        (let [a  (act b s)
              s' (agent/sample-next T s a)
              o  (observe true-world s')
              b' (update-belief b s' o)]
          (recur s' b' (inc step)
                 (conj states s') (conj actions a)
                 (conj obss o) (conj beliefs b')))))))

;; ---------------------------------------------------------------------------
;; Multi-armed bandit POMDP (agentmodels Ch 3c/3d)
;; ---------------------------------------------------------------------------
;;
;; A bandit is a POMDP whose latent never changes: the hidden state is the
;; per-arm payoff parameter (fixed for the episode); pulling an arm is both the
;; action and the only observation of that arm; the reward is the observation;
;; belief factorizes into one independent Beta(alpha,beta) per arm.
;;
;; The Beta-Bernoulli update is the trivial conjugate increment kept HOST-SIDE on
;; a plain per-arm map. This is the same identity as
;; genmlx.inference.conjugate/bb-update, but that namespace is analytical-inference
;; *marginalization* middleware (array in/out, co-computes a marginal LL a filter
;; discards) — not an acting agent's online belief store. Keeping it host-side
;; matches how the gridworld POMDP filter above stays host-side via
;; inverse/normalize-logs rather than forcing the inference engine.

(defn update-arm
  "Conjugate Beta-Bernoulli posterior update on arm `i`: a success (reward 1)
   bumps alpha, a failure bumps beta; every other arm is untouched (independence)."
  [belief i reward]
  (update-in belief [:arms i]
             (fn [[a b]] (if (== reward 1) [(inc a) b] [a (inc b)]))))


(defn make-bandit-agent
  "Bandit POMDP agent. Belief = {:arms [[alpha beta] ...]} (per-arm Beta).
     :strategy :thompson (default) | :softmax ;  :alpha inverse-temp for :softmax.
   Returns {:act (fn [belief key] -> arm-int) :update-belief (fn [belief arm reward])
            :arm-values (fn [belief] -> [mean ...]) :params}.

   Thompson sampling is POSTERIOR SAMPLING: each step draws one theta_i ~
   Beta(alpha_i,beta_i) per arm and pulls the argmax. The explore->exploit
   behaviour is emergent — the posterior draw is wide while the belief is
   uncertain and collapses as the Beta sharpens — NOT the optimal Bayes-adaptive
   (information-valuing) policy."
  [{:keys [strategy alpha] :or {strategy :thompson alpha 4.0}}]
  (let [arm-values (fn [{:keys [arms]}] (mapv (fn [[a b]] (/ a (+ a b))) arms))]
    {:arm-values    arm-values
     :update-belief update-arm
     :act (fn [{:keys [arms]} key]
            (case strategy
              :thompson
              ;; posterior sampling: draw theta_i ~ Beta(alpha_i,beta_i) per arm and
              ;; pull the argmax. The action is the argmax of K draws (not one random
              ;; choice), so we sample directly. dist/beta-dist now uses the stable
              ;; gamma-ratio sampler (genmlx-gcw4 fixed), so this is exact + crash-free
              ;; at the high concentration a converging bandit reaches.
              (let [ks (rng/split-n key (count arms))
                    ss (mapv (fn [k [a b]] (mx/item (dist/sample (dist/beta-dist a b) k))) ks arms)]
                (int (apply max-key #(nth ss %) (range (count ss)))))
              :softmax
              (let [eu  (mx/array (clj->js (arm-values {:arms arms})))
                    pol (gen [] (trace :action (h/softmax-action alpha eu)))]
                (int (mx/item (:retval (p/simulate (dyn/with-key pol key) [])))))))
     :params {:strategy strategy :alpha alpha}}))

(defn simulate-bandit
  "Roll the bandit agent over the horizon, threading (belief, arm, reward). Each
   step: ACT from belief -> arm; PULL -> reward (the observation); FILTER ->
   belief'. Returns {:arms [...] :rewards [...] :beliefs [b0 b1 ...] :cum-reward
   [...] :regret [...]}; :beliefs index k is the belief held when choosing pull k.
   Pass `key0` (e.g. (rng/fresh-key 42)) for a reproducible rollout."
  [{:keys [act update-belief]} {:keys [pull prior theta* thetas horizon]} & [key0]]
  (loop [b prior, step 0, key (or key0 (rng/fresh-key))
         arms [], rewards [], beliefs [prior], cum 0.0, reg 0.0, cum-reward [], regret []]
    (if (>= step horizon)
      {:arms arms :rewards rewards :beliefs beliefs :cum-reward cum-reward :regret regret}
      (let [[k1 krest] (rng/split key)
            [k2 k3]    (rng/split krest)
            i  (act b k1)
            r  (pull i k2)
            b' (update-belief b i r)
            cum' (+ cum r)
            reg' (+ reg (- theta* (nth thetas i)))]   ; instantaneous regret theta* - theta_i
        (recur b' (inc step) k3
               (conj arms i) (conj rewards r) (conj beliefs b')
               cum' reg' (conj cum-reward cum') (conj regret reg'))))))
