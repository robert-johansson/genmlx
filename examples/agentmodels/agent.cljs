(ns agentmodels.agent
  "MDP agent for the agentmodels port — GenMLX-native, with TWO paths that agree.

   agentmodels.org defines an agent by the mutual recursion act / expectedUtility
   under a softmax (Boltzmann) action choice with rationality `alpha`. That agent
   is SOFT-rational: the value of a state is the expectation of expectedUtility
   under the agent's OWN softmax policy (not a hard max). GenMLX realises this two
   ways that compute the identical value function and Q:

   1. TENSOR VALUE ITERATION (Layer B, GPU-native) — the Bellman backup
        Q = R + gamma*(1-term) * (T . V),    V = backup(Q)
      where backup is the soft policy-expectation  V(s) = Σ_a softmax(alpha*Q[s])[a]*Q[s,a]
      for finite alpha, and the hard max_a Q in the optimal limit alpha = ##Inf.

   2. RECURSIVE expectedUtility (faithful agentmodels) — memoized with
      exact/with-cache (the dp.cache analog), mutually recursive with the soft
      policy, exactly mirroring act/expectedUtility over a finite horizon.

   Both produce the same first action / Q-values (certified by the equivalence
   test). The policy is itself a generative function: act = a GFI random choice,
        (gen [s] (trace :action (softmax-action alpha Q[s])))
   which is agentmodels' factor(alpha*EU) in one composable line.

   Scope: MDP planning + rollout + the recursive≡tensor equivalence. POMDP
   (make-pomdp-agent, belief filtering, :update-belief, simulate-pomdp) remains."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.exact :as exact]
            [agentmodels.helpers :as h])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Backups: hard max (optimal limit) vs soft policy-expectation (faithful)
;; ---------------------------------------------------------------------------

(defn- hard-value [_alpha Q] (mx/amax Q [1]))            ; V = max_a Q

(defn- soft-value
  "agentmodels' state value: the expectation of Q under the agent's softmax
   policy, V(s) = Σ_a softmax(alpha*Q[s])[a] * Q[s,a]. (Expectation over the
   policy — NOT logsumexp/mellowmax.)"
  [alpha Q]
  (let [pi (mx/softmax (mx/multiply (mx/scalar alpha) Q) 1)]   ; [S,A] over actions
    (mx/sum (mx/multiply pi Q) [1])))                          ; [S]

(defn- value-fn-for [alpha] (if (= alpha ##Inf) hard-value soft-value))

;; ---------------------------------------------------------------------------
;; Path 1: tensor value iteration
;; ---------------------------------------------------------------------------

(defn bellman-step
  "One synchronous Bellman backup (pure MLX graph). V:[S] -> [Q:[S,A], V':[S]].
   next-V[s,a] = Σ_s' T[s,a,s'] * V[s']  via a flat matrix-vector product."
  [{:keys [S A T R term]} gamma alpha value-of V]
  (let [V-col  (mx/reshape V #js [S 1])
        T-flat (mx/reshape T #js [(* S A) S])
        next-V (mx/reshape (mx/matmul T-flat V-col) #js [S A])           ; [S,A]
        cont   (mx/reshape (mx/subtract (mx/scalar 1.0) term) #js [S 1]) ; [S,1] -> bcast over A
        Q      (mx/add R (mx/multiply (mx/scalar gamma)
                                      (mx/multiply cont next-V)))]
    [Q (value-of alpha Q)]))

(defn value-iteration
  "Run `n` Bellman sweeps from V = 0. Backup is the soft policy-expectation for
   finite alpha and hard max at alpha = ##Inf. Returns {:Q [S,A] :V [S]}.
   Materializes V each sweep to break lazy-graph accumulation."
  [{:keys [S gamma] :as mdp} alpha n]
  (let [value-of (value-fn-for alpha)]
    (loop [V (mx/zeros #js [S]), i 0, Q nil]
      (if (>= i n)
        {:Q Q :V V}
        (let [[Q' V'] (bellman-step mdp gamma alpha value-of V)]
          (mx/materialize! V')
          (recur V' (inc i) Q'))))))

;; ---------------------------------------------------------------------------
;; Path 2: faithful recursive expectedUtility (exact/with-cache + soft policy)
;; ---------------------------------------------------------------------------

(defn- softmax-vec [xs]
  (let [m  (apply max xs)
        es (mapv #(Math/exp (- % m)) xs)
        z  (reduce + es)]
    (mapv #(/ % z) es)))

(defn recursive-eu
  "agentmodels' expectedUtility / act, memoized with exact/with-cache (the
   dp.cache analog). SOFT-rational and mutually recursive: the future value is
   the expectation of EU under the agent's own softmax(alpha*EU) policy. Operates
   on host-side scalars (state, action, timeLeft) — with-cache keys on JS args.

       EU(s,a,t) = u(s,a) + gamma*[ terminal(s) or t<=1 ? 0
                                    : Σ_s' T(s,a,s') * V(s', t-1) ]
       V(s,t)    = Σ_a  softmax(alpha*EU(s,·,t))[a] * EU(s,a,t)

   Returns {:eu (fn [s a t]) :soft-v (fn [s t])}; EU(s,a,horizon) is the t-step Q
   that `value-iteration` computes with `n = horizon` sweeps."
  [{:keys [A gamma terminals] :as mdp}]
  (let [Rh      (mx/->clj (:R mdp))       ; [S][A] JS numbers
        Th      (mx/->clj (:T mdp))       ; [S][A][S'] JS numbers
        terms   (set (keys terminals))
        eu-atom (atom nil)
        soft-v  (fn [s t]
                  (let [qs (mapv (fn [a] (@eu-atom s a t)) (range A))]
                    (reduce + (map * (softmax-vec (mapv #(* (:alpha mdp) %) qs)) qs))))
        eu      (exact/with-cache
                  (fn [s a t]
                    (let [u (get-in Rh [s a])]
                      (if (or (terms s) (<= t 1))
                        u
                        (+ u (* gamma
                                (reduce-kv
                                  (fn [acc s' pr]
                                    (if (pos? pr) (+ acc (* pr (soft-v s' (dec t)))) acc))
                                  0.0 (get-in Th [s a]))))))))]
    (reset! eu-atom eu)
    {:eu eu :soft-v soft-v}))

(defn- recursive-eu-inf
  "alpha = ##Inf limit of recursive-eu: the soft policy expectation collapses to
   the hard max, so V(s,t) = max_a EU(s,a,t). Kept separate to avoid Inf*Q NaNs."
  [{:keys [A gamma terminals] :as mdp}]
  (let [Rh (mx/->clj (:R mdp))
        Th (mx/->clj (:T mdp))
        terms (set (keys terminals))
        eu-atom (atom nil)
        max-v (fn [s t] (apply max (mapv (fn [a] (@eu-atom s a t)) (range A))))
        eu (exact/with-cache
             (fn [s a t]
               (let [u (get-in Rh [s a])]
                 (if (or (terms s) (<= t 1))
                   u
                   (+ u (* gamma (reduce-kv
                                   (fn [acc s' pr]
                                     (if (pos? pr) (+ acc (* pr (max-v s' (dec t)))) acc))
                                   0.0 (get-in Th [s a]))))))))]
    (reset! eu-atom eu)
    {:eu eu :soft-v max-v}))

;; ---------------------------------------------------------------------------
;; Agent constructor + rollout
;; ---------------------------------------------------------------------------

(defn make-mdp-agent
  "Build an MDP agent over `mdp`. Returns
     {:mdp :Q [S,A] :V [S] :policy gen-fn :act (fn [s]->action)
      :expected-utility (fn [s a]) :params}.
   :Q comes from tensor value iteration; :expected-utility is the faithful
   recursive path at the same horizon (lazy — only computed if called, so demos
   pay nothing). Both agree (see the equivalence test)."
  [{:keys [mdp alpha gamma n-iters]
    :or   {alpha 100.0 gamma 1.0 n-iters 24}}]
  (let [mdp        (assoc mdp :gamma gamma :alpha alpha)
        {:keys [Q V]} (value-iteration mdp alpha n-iters)
        policy     (gen [s] (trace :action (h/softmax-action alpha (mx/idx Q s))))
        rec        (delay (if (= alpha ##Inf) (recursive-eu-inf mdp) (recursive-eu mdp)))]
    {:mdp mdp :Q Q :V V :policy policy
     :act (fn [s] (int (mx/item (:retval (p/simulate (dyn/auto-key policy) [s])))))
     :expected-utility (fn [s a] ((:eu @rec) s a n-iters))   ; recursive EU(s,a,horizon)
     :params {:alpha alpha :gamma gamma :horizon n-iters}}))

;; The environment transition is itself a generative function: given the T-row
;; probabilities for (s,a), sample the next state. With a deterministic (noise=0)
;; MDP the row is one-hot, so this reduces to the obvious deterministic step.
(def ^:private env-step
  (gen [probs] (trace :s (dist/weighted probs))))

(defn- sample-next [T s a]
  (let [probs (vec (mx/->clj (mx/idx (mx/idx T s) a)))]      ; T[s,a,:] -> [S'] probs
    (int (mx/item (:retval (p/simulate (dyn/auto-key env-step) [probs]))))))

(defn simulate-mdp
  "Roll the agent's policy out from `start` for at most `horizon` steps, stopping
   at a terminal. The action comes from the softmax policy (decision noise) and
   the next state is sampled from T (environment / transition noise). Returns
   {:states [s0 s1 ...] :actions [a0 a1 ...]} (JS ints); one action per transition."
  [{:keys [mdp act]} start horizon]
  (let [{:keys [T terminals]} mdp]
    (loop [s start, step 0, states [start], actions []]
      (if (or (>= step horizon) (contains? terminals s))
        {:states states :actions actions}
        (let [a  (act s)
              s' (sample-next T s a)]
          (recur s' (inc step) (conj states s') (conj actions a)))))))
