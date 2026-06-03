(ns genmlx.agents.differentiable
  "Differentiable MDP utility/alpha learning (bean genmlx-j5um): recover an agent's
   planted utilities and rationality by GRADIENT through the planner + policy
   log-likelihood — the payoff of the tensor-native rewrite.

   SCOPE. This is the MDP (no-belief) case. It does NOT differentiate through a
   stochastic rollout (the env-step sample uses mx/item — non-differentiable). It
   differentiates through the PLANNER (lazy value iteration) and the policy
   log-likelihood at FIXED observed (s,a) pairs — exactly the inverse/action-loglik
   quantity the IRL posterior is built from. That graph
       utilities[G] -> R[S,A] -> lazy VI -> Q[S,A] -> log_softmax -> gather
   is fully lazy/differentiable with no Scan combinator or belief filter, so it
   lands independently of genmlx-5zdd / genmlx-kpuo. (The POMDP-belief-gradient and
   expected-fused-rollout-gradient variants are follow-ons: genmlx-5x3f, genmlx-h833.)

   IDENTIFIABILITY. utilities are identifiable only up to scale/offset and a
   multiplicative interaction with alpha (alpha·Q enters the policy). So recovery
   is to LIKELIHOOD-EQUIVALENCE (loss at recovered ≈ loss at the plant), not exact
   params — plant a 2-goal world for relative ordering, or fix one of {alpha,
   utility-scale} and learn the other.

   KEY INVARIANT. ZERO mx/item / mx/eval! / mx/materialize! inside the loss — they
   sever the backward pass (silent zero gradients). Materialization happens only in
   learning/train between iterations, after autograd has run."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.agents.gridworld :as gw]
            [genmlx.agents.agent :as agent]
            [genmlx.learning :as learn]))

(defn build-diff-mdp
  "Static MDP scaffolding for differentiable learning: the geometry tensors
   (T/term/S/A/gamma) — which DON'T depend on the utilities — plus the [S,G]
   goal-onehot. The utilities become the learnable [G] tensor theta-u fed to
   gridworld/diff-reward. `:goals` fixes the (deterministic) goal ordering for
   theta-u's axis."
  [{:keys [grid goals start gamma noise] :or {gamma 1.0 noise 0.0 start [0 0]}}]
  (let [base      (gw/build-mdp {:grid grid :utilities {} :start start :gamma gamma :noise noise})
        goals-vec (vec goals)]
    (assoc base
           :goal-onehot (gw/goal-onehot grid goals-vec)
           :goals-vec   goals-vec
           :G           (count goals-vec))))

(defn diff-q
  "Differentiable finite-horizon Q[S,A] for utilities `theta-u` ([G] tensor), time
   cost `tc` (scalar) and `alpha` (finite number or scalar tensor): build R via
   diff-reward, then lazy value iteration. Pure lazy graph (no materialize)."
  [{:keys [goal-onehot S A G] :as dmdp} theta-u tc alpha n-iters]
  (let [R   (gw/diff-reward goal-onehot theta-u tc S A G)
        mdp (assoc dmdp :R R)]
    (:Q (agent/value-iteration-lazy mdp alpha n-iters))))

(defn action-loglik-loss
  "Negative marginal action-log-likelihood of the observed (s,a) pairs under the
   softmax policy induced by diff-q — inverse/action-loglik made a tensor, summed
   over observations. `states-obs`/`actions-obs` are host int vectors (constants
   outside the gradient); the gather is index-only, so the whole graph stays lazy
   and differentiable in (theta-u, log-alpha). Returns a [] scalar (the loss).
   `alpha = exp(log-alpha)` keeps alpha positive and unconstrained for Adam."
  [{:keys [S A] :as dmdp} theta-u tc log-alpha n-iters states-obs actions-obs]
  (let [alpha  (mx/exp log-alpha)
        Q      (diff-q dmdp theta-u tc alpha n-iters)                    ; [S,A]
        logits (mx/multiply alpha Q)                                     ; [S,A]
        logp   (mx/subtract logits (mx/expand-dims (mx/logsumexp logits [-1]) -1))  ; log_softmax over actions
        flat   (mx/reshape logp #js [(* S A)])
        idx    (mx/array (clj->js (mapv (fn [s a] (+ (* s A) a)) states-obs actions-obs)) mx/int32)
        ll     (mx/sum (mx/take-idx flat idx 0))]                        ; Σ_m log π(a_m|s_m)
    (mx/negative ll)))

(defn- pack   [theta-u-vec log-alpha] (mx/array (clj->js (conj (vec theta-u-vec) log-alpha)) mx/float32))
(defn- unpack [params G] {:theta-u (mx/take-idx params (mx/arange G) 0)  ; [G]
                          :log-alpha (mx/idx params G)})                 ; scalar

(defn recover-params
  "Recover (utilities, alpha) by Adam on the action-loglik loss through the
   differentiable planner. Params = [theta-u(0..G-1), log-alpha] as one [G+1] MLX
   array. Returns {:theta-u [G] MLX :alpha number :log-history [...]}.
   Options: :iterations (300) :lr (0.05) :init-utils ([G] of 0.0) :init-log-alpha (0)
            :fixed-log-alpha (when set, alpha is held fixed and only utilities learn).
   `observations` is a seq of [state action] pairs (non-terminal states)."
  [dmdp tc n-iters observations
   {:keys [iterations lr init-utils init-log-alpha fixed-log-alpha key]
    :or   {iterations 300 lr 0.05}}]
  (let [G          (:G dmdp)
        states-obs (mapv first observations)
        actions-obs (mapv second observations)
        init-la    (or fixed-log-alpha init-log-alpha 0.0)
        init       (pack (or init-utils (repeat G 0.0)) init-la)
        ;; loss as a fn of the packed [G+1] params; if alpha is fixed, override the
        ;; log-alpha slot with the constant so its gradient slot is inert.
        raw    (fn [params _key]
                 (let [{:keys [theta-u log-alpha]} (unpack params G)
                       la (if fixed-log-alpha (mx/scalar (double fixed-log-alpha)) log-alpha)]
                   (action-loglik-loss dmdp theta-u tc la n-iters states-obs actions-obs)))
        vg     (mx/value-and-grad raw [0])
        lgf    (fn [params k] (let [[loss grad] (vg params (rng/ensure-key k))]
                                {:loss loss :grad grad}))
        {:keys [params loss-history]}
        (learn/train {:iterations iterations :optimizer :adam :lr lr :key (rng/ensure-key key)} lgf init)
        {:keys [theta-u log-alpha]} (unpack params G)]
    {:theta-u theta-u
     :alpha   (mx/item (mx/exp (if fixed-log-alpha (mx/scalar (double fixed-log-alpha)) log-alpha)))
     :loss-history loss-history}))

(defn loss-at
  "The loss evaluated at given host utilities + log-alpha (a convenience for the
   likelihood-equivalence success check: loss(recovered) ≤ loss(plant) + tol)."
  [dmdp tc utils-vec log-alpha n-iters observations]
  (mx/item (action-loglik-loss dmdp (mx/array (clj->js (vec utils-vec)) mx/float32)
                               tc (mx/scalar (double log-alpha)) n-iters
                               (mapv first observations) (mapv second observations))))
