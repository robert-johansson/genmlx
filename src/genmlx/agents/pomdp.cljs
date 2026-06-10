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
            [genmlx.agents.gridworld :as gw]
            [genmlx.agents.agent :as agent]
            [genmlx.agents.belief :as belief]
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
  [{:keys [grid goals alpha noise gamma prior observe n-iters world-utils start]
    :or   {alpha 2.0 noise 0.0 gamma 1.0 n-iters 40 start [0 0]}}]
  (let [;; Per-world MDP planners. Two ways to specify the worlds:
        ;;   :world-utils {world -> utilities-map} — general (e.g. open/closed
        ;;     restaurant configs, where each world has different cell utilities), or
        ;;   :goals [...] — the single-rewarding-goal shorthand via goal-agents.
        world-agents (if world-utils
                       (into {} (map (fn [[w utils]]
                                       [w (agent/make-mdp-agent
                                            {:mdp (gw/build-mdp {:grid grid :utilities utils
                                                                 :start start :gamma gamma :noise noise})
                                             :alpha alpha :gamma gamma :n-iters n-iters})])
                                     world-utils))
                       (inv/goal-agents {:grid grid :goals goals :alpha alpha
                                         :noise noise :gamma gamma}))
        worlds    (if world-utils (vec (keys world-utils)) goals)
        A         (:A (:mdp (val (first world-agents))))
        prior     (or prior (zipmap worlds (repeat (/ 1.0 (count worlds)))))
        worlds-vec (vec worlds)
        W         (count worlds-vec)
        ;; QMDP belief-space Q at state s: Σ_w b(w) · Q_w[s]. Tensorized (bean
        ;; genmlx-4ifp): stack the per-world solved Q into [W,S,A] ONCE, then
        ;; contract the [W] belief vector against the [W,A] Q-rows at s as a single
        ;; fused reduction, instead of a host reduce over the belief map. Agrees
        ;; with the old reduce to float32 (1e-5) — only the summation reassociates.
        Qstack    (mx/stack (mapv #(:Q (world-agents %)) worlds-vec))    ; [W,S,A]
        belief-Q  (fn [belief s]
                    (let [bvec (mx/array (clj->js (mapv #(double (get belief % 0.0)) worlds-vec)) mx/float32)]
                      (mx/sum (mx/multiply (mx/reshape bvec [W 1]) (mx/idx Qstack s 1)) [0])))
        ;; one Bayes filtering step: b'(w) ∝ b(w) · P(obs | w, loc).
        ;; obs = nil (uninformative location) => belief unchanged (flat-then-snap).
        ;; Observation-model contract (unified across pomdp/belief/biased filters,
        ;; genmlx-xpbm): obs = nil is unconditionally uninformative (identity), and
        ;; an obs impossible under the current belief keeps b unchanged (defensive)
        ;; instead of NaN-ing through normalize-logs.
        update-belief (fn [belief loc obs]
                        (if (nil? obs)
                          belief
                          (let [logm (into {} (map (fn [[w b]]
                                                     [w (+ (Math/log b)
                                                           (if (= (observe w loc) obs) 0.0 ##-Inf))])
                                                   belief))]
                            (if (every? #(= % ##-Inf) (vals logm))
                              belief
                              (inv/normalize-logs logm)))))
        ;; act = softmax-action over the belief-mixed Q row, as a real GFI choice.
        ;; policy is hoisted and parameterized on q: constructing a gen fn per act
        ;; call re-ran schema extraction every rollout step (genmlx-xpbm).
        policy (gen [q] (trace :action (h/softmax-action alpha q)))
        act (fn [belief s]
              (let [q (belief-Q belief s)]
                (int (mx/item (:retval (p/simulate (dyn/auto-key policy) [q]))))))]
    {:worlds (vec worlds) :world-agents world-agents :prior prior :observe observe
     :belief-Q belief-Q :update-belief update-belief :act act
     ;; tensor belief kernel (bean genmlx-kpuo): same map-in/map-out contract as
     ;; :update-belief but the filter runs as pure MLX [W] ops (no per-step host
     ;; map arithmetic, no mx/item). Opt-in via simulate-pomdp :belief-mode :tensor.
     :update-belief-tensor (fn [belief loc obs]
                             (belief/update-belief-map observe (vec worlds) belief loc obs))
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
   so :states and :beliefs align and both feed the seam (env->trajectory / dist->bars).

   `:belief-mode` (optional trailing opts) selects the belief filter: :host
   (default — the original Clojure-map normalize-logs filter, byte-identical seam)
   or :tensor (the pure-MLX kernel genmlx.agents.belief; bean genmlx-kpuo). Both
   produce {world -> prob} beliefs, so the seam is unchanged either way."
  [{:keys [act update-belief update-belief-tensor observe world-agents prior]} env start horizon
   & [{:keys [belief-mode] :or {belief-mode :host}}]]
  (let [true-world (:true-world env)
        true-mdp   (:mdp (world-agents true-world))
        T          (:T true-mdp)
        terminals  (:terminals true-mdp)
        update-belief (if (= belief-mode :tensor) update-belief-tensor update-belief)]
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

(defn tensor-bb-increment
  "One-hot-masked Beta-Bernoulli conjugate increment on [K] alpha/beta MLX tensors
   (bean genmlx-4ifp) — the pure-tensor form of update-arm, same algebra as
   genmlx.inference.conjugate/bb-update. For chosen arm `i` with reward r∈{0,1}:
     alpha' = alpha + onehot_i · r ,  beta' = beta + onehot_i · (1 - r).
   Returns [alpha' beta']. Element-wise and differentiable; produces the same
   numbers as update-arm. (The live :thompson path keeps the {:arms [[a b]…]} map
   for seam compatibility; the [N,K] batched form drives genmlx-tl6p.)"
  [alpha beta i reward k]
  (let [mask (mx/where (mx/equal (mx/arange k) (mx/scalar (int i))) (mx/scalar 1.0) (mx/scalar 0.0))
        r    (mx/scalar (double reward))]
    [(mx/add alpha (mx/multiply mask r))
     (mx/add beta  (mx/multiply mask (mx/subtract (mx/scalar 1.0) r)))]))


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
  (let [arm-values (fn [{:keys [arms]}] (mapv (fn [[a b]] (/ a (+ a b))) arms))
        ;; hoisted + parameterized on eu: a per-act (gen ...) re-ran schema
        ;; extraction every pull (genmlx-xpbm)
        softmax-pol (gen [eu] (trace :action (h/softmax-action alpha eu)))]
    {:arm-values    arm-values
     :update-belief update-arm
     :act (fn [{:keys [arms]} key]
            (case strategy
              :thompson
              ;; posterior sampling: draw theta_i ~ Beta(alpha_i,beta_i) per arm and
              ;; pull the argmax. Tensorized (bean genmlx-4ifp): one [K] Beta draw via
              ;; the per-element gamma-ratio sampler (genmlx-gcw4-stable, crash-free at
              ;; high concentration) + a single mx/argmax — replacing the K per-arm
              ;; mx/item draws with ONE extraction. Belief stays {:arms [[a b]…]} for
              ;; seam compatibility (RNG path now draws all arms in one op — the seeded
              ;; convergence/regret tests assert invariants, not the exact pull order).
              (let [av    (mx/array (clj->js (mapv first arms)) mx/float32)    ; [K] alpha
                    bv    (mx/array (clj->js (mapv second arms)) mx/float32)   ; [K] beta
                    theta (dist/beta-sample-vec av bv key)]                    ; [K] posterior draw
                (int (mx/item (mx/argmax theta))))
              :softmax
              (let [eu (mx/array (clj->js (arm-values {:arms arms})))]
                (int (mx/item (:retval (p/simulate (dyn/with-key softmax-pol key) [eu])))))))
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

(defn simulate-bandit-batched
  "Run N independent Thompson bandit episodes at once via shape-based batching
   (bean genmlx-tl6p). Belief is [N,K] alpha/beta tensors; each step is ONE [N,K]
   Beta draw + [N] argmax pull + [N] Bernoulli reward + a one-hot-masked [N,K]
   conjugate increment — the N×K particle dimension is fully tensorized, so the
   only host loop is over the (small) horizon and there is no per-step mx/item.

   Equivalence to N independent simulate-bandit calls is DISTRIBUTIONAL (aggregate
   means over N), not per-episode: the batched path draws all N×K Beta values in
   one op rather than the host's per-arm split tree, so individual trajectories
   differ but the N-distribution matches. Reward at the pulled arm uses the TRUE
   thetas (the observation channel), exactly like the host `pull`. Returns
     {:arms [N,H] :rewards [N,H] :regret [N,H] (cumulative) :final-means [N,K]
      :cum-reward [N] :n N}  — every non-:n leaf has a leading [N] axis."
  [{:keys [thetas theta* horizon prior]} n master-key]
  (let [K        (count thetas)
        true-th  (mx/array (clj->js (vec thetas)) mx/float32)            ; [K] true payoffs
        thstar   (mx/scalar (double (or theta* (apply max thetas))))
        prior-ab (or (:arms prior) (vec (repeat K [1.0 1.0])))           ; per-arm Beta prior
        a0       (mx/array (clj->js (vec (repeat n (mapv first prior-ab)))) mx/float32)  ; [N,K]
        b0       (mx/array (clj->js (vec (repeat n (mapv second prior-ab)))) mx/float32)
        arange-K (mx/arange K)]
    (loop [t 0, alpha a0, beta b0
           cum-reward (mx/zeros [n]), cum-regret (mx/zeros [n])
           key (rng/ensure-key master-key)
           arms [], rewards [], regrets []]
      (if (>= t horizon)
        {:arms (mx/stack arms 1) :rewards (mx/stack rewards 1) :regret (mx/stack regrets 1)
         :final-means (mx/divide alpha (mx/add alpha beta)) :cum-reward cum-reward :n n}
        (let [[k-th k-rest] (rng/split key)
              [k-rew k-next] (rng/split k-rest)
              theta   (dist/beta-sample-vec alpha beta k-th)             ; [N,K] posterior draw
              arm     (mx/argmax theta 1)                               ; [N] pulled arm
              p-chos  (mx/take-idx true-th arm 0)                        ; [N] true theta of pull
              r       (mx/where (mx/less (rng/uniform k-rew [n]) p-chos)
                                (mx/scalar 1.0) (mx/scalar 0.0))         ; [N] Bernoulli reward
              mask    (mx/where (mx/equal arange-K (mx/reshape arm [n 1]))
                                (mx/scalar 1.0) (mx/scalar 0.0))         ; [N,K] one-hot of pull
              r-col   (mx/reshape r [n 1])
              alpha'  (mx/add alpha (mx/multiply mask r-col))            ; conjugate increment
              beta'   (mx/add beta  (mx/multiply mask (mx/subtract (mx/scalar 1.0) r-col)))
              cum-reward' (mx/add cum-reward r)
              cum-regret' (mx/add cum-regret (mx/subtract thstar p-chos))]  ; theta* - theta_pull
          ;; break the lazy graph each step (carries + recorded outputs) so horizon×N
          ;; does not accumulate one giant graph; matches value-iteration's cadence.
          (mx/materialize! alpha' beta' cum-reward' cum-regret' arm r)
          (recur (inc t) alpha' beta' cum-reward' cum-regret' k-next
                 (conj arms arm) (conj rewards r) (conj regrets cum-regret')))))))
