(ns agentmodels.agent
  "MDP agent for the agentmodels port — GenMLX-native.

   agentmodels.org defines an agent by the mutual recursion `act` / `expectedUtility`
   under a softmax (Boltzmann) action choice with rationality `alpha`. Here that
   becomes two things that compose cleanly with the GFI:

   1. VALUE ITERATION as a pure MLX graph — the Bellman backup
        Q = R + gamma*(1-term) * (T . V),   V = max_a Q
      run for a fixed number of sweeps. This is the GPU-native realisation of
      `expectedUtility` (the hard/argmax limit; soft backup swaps max for
      logsumexp — a follow-up for the faithful equivalence test, bean genmlx-m4pr).

   2. THE POLICY AS A GENERATIVE FUNCTION — `act` is literally
        (gen [s] (trace :action (softmax-action alpha Q[s])))
      so the agent's action is a GFI random choice. `softmax-action` (helpers.cljs)
      is Categorical(softmax(alpha*Q[s])); at alpha = ##Inf it is the deterministic
      argmax. This IS agentmodels' factor(alpha*EU), expressed in one line and
      fully composable with simulate / generate / assess.

   Vertical-slice scope: the tensor-Bellman path + rollout. The faithful
   exact/with-cache recursive path and the recursive≡tensor equivalence test are
   the next increment (bean genmlx-m4pr)."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [agentmodels.helpers :as h])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn bellman-step
  "One synchronous Bellman backup (pure MLX graph). V:[S] -> [Q:[S,A], V':[S]].
   next-V[s,a] = sum_s' T[s,a,s'] * V[s']  via a flat matrix-vector product."
  [{:keys [S A T R term]} gamma V]
  (let [V-col  (mx/reshape V #js [S 1])
        T-flat (mx/reshape T #js [(* S A) S])
        next-V (mx/reshape (mx/matmul T-flat V-col) #js [S A])           ; [S,A]
        cont   (mx/reshape (mx/subtract (mx/scalar 1.0) term) #js [S 1]) ; [S,1] -> bcast over A
        Q      (mx/add R (mx/multiply (mx/scalar gamma)
                                      (mx/multiply cont next-V)))]
    [Q (mx/amax Q [1])]))

(defn value-iteration
  "Run `n` Bellman sweeps from V = 0. Returns {:Q [S,A] :V [S]}. Materializes V
   each sweep to break lazy-graph accumulation (the Layer C eval boundary)."
  [{:keys [S gamma] :as mdp} n]
  (loop [V (mx/zeros #js [S]), i 0, Q nil]
    (if (>= i n)
      {:Q Q :V V}
      (let [[Q' V'] (bellman-step mdp gamma V)]
        (mx/materialize! V')
        (recur V' (inc i) Q')))))

(defn make-mdp-agent
  "Build an MDP agent over `mdp`. Returns
     {:mdp :Q [S,A] :V [S] :policy gen-fn :act (fn [s] -> action-int) :params}.
   `:policy` is the softmax-action generative function; `:act` samples one action
   for a state by running it through the GFI (p/simulate)."
  [{:keys [mdp alpha gamma n-iters]
    :or   {alpha 100.0 gamma 1.0 n-iters 24}}]
  (let [mdp        (assoc mdp :gamma gamma)
        {:keys [Q V]} (value-iteration mdp n-iters)
        policy     (gen [s] (trace :action (h/softmax-action alpha (mx/idx Q s))))]
    {:mdp mdp :Q Q :V V :policy policy
     :act (fn [s] (int (mx/item (:retval (p/simulate (dyn/auto-key policy) [s])))))
     :params {:alpha alpha :gamma gamma}}))

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
