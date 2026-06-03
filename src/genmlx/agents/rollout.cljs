(ns genmlx.agents.rollout
  "Fused-rollout combinator (bean genmlx-5zdd). The host rollout loops sample an
   action and an environment transition each step and extract BOTH to the host via
   mx/item (~2 GPU→CPU round-trips per step). This re-expresses the MDP rollout as a
   single lazy Metal graph: the state is threaded as an MLX int tensor and the
   action + next-state are sampled IN-GRAPH (argmax at alpha=##Inf / noise=0, else
   Gumbel-max via rng/categorical), with exactly ONE mx/materialize! + one ->clj at
   the end, then a host truncation at the first terminal. No per-step mx/item.

   Scope. Only the MDP rollout fuses cleanly. A POMDP/biased rollout must call the
   host `observe` geometry at the new state each step, which needs s' as a host int
   — so it cannot shed the per-step extraction and stays the host loop. The bandit
   rollout's tensor form is exactly genmlx.agents.pomdp/simulate-bandit-batched at
   N=1 (bean genmlx-tl6p). So this namespace fuses the MDP case; the others are
   covered or inherently host-bound (see genmlx-5zdd notes).

   EQUIVALENCE. alpha=##Inf (argmax policy) + noise=0 (one-hot transitions) =>
   vector-IDENTICAL :states/:actions to agent/simulate-mdp (no tolerance, on
   non-degenerate Q). Finite-alpha / noisy paths match the host DISTRIBUTION
   (Gumbel-max == the host categorical draw), not the exact per-step sequence (the
   noise threads differently).

   TIE-BREAK. At an EXACT Q tie the host's exact/categorical-argmax breaks uniformly
   at random, while the fused mx/argmax takes the first index — a measure-zero
   divergence on symmetric worlds. Exact-equivalence holds on non-tied Q."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]))

(defn- step-action
  "In-graph action sample at lazy int state `s`: argmax(Q[s]) at alpha=##Inf, else
   Gumbel-max(alpha·Q[s]) via rng/categorical. Returns a lazy int32 [] index (cast
   so it stacks with the int32 state carry — argmax/categorical return uint32)."
  [Q s alpha key]
  (let [q (mx/idx Q s)]                                       ; [A]
    (mx/astype
      (if (= alpha ##Inf)
        (mx/argmax q)
        (rng/categorical key (mx/multiply (mx/scalar alpha) q)))
      mx/int32)))

(defn- step-next
  "In-graph env transition at lazy (s,a): argmax(T[s,a]) at noise=0 (one-hot rows),
   else Gumbel-max(log T[s,a]). Returns a lazy int32 [] index."
  [T s a noise key]
  (let [trow (mx/idx (mx/idx T s) a)]                         ; [S'] transition row
    (mx/astype
      (if (zero? noise)
        (mx/argmax trow)
        (rng/categorical key (mx/log trow)))
      mx/int32)))

(defn rollout-mdp
  "Fused MDP rollout (drop-in for agent/simulate-mdp). Threads the state as a lazy
   MLX int tensor for `horizon` steps, samples action + next-state in-graph,
   materializes ONCE, then host-truncates at the first terminal so the output
   matches the host loop exactly. `agent` is a make-mdp-agent (carries :mdp with
   :T/:terminals/:alpha/:noise and the solved :Q). Returns {:states [...] :actions
   [...]} (JS ints). Pass {:key k} to seed the finite-alpha / noisy draws."
  [{:keys [mdp Q]} start horizon & [{:keys [key]}]]
  (let [{:keys [T terminals alpha noise]} mdp
        noise     (or noise 0.0)
        det?      (and (= alpha ##Inf) (zero? noise))
        keys      (when-not det? (rng/split-n (rng/ensure-key key) (* 2 (max 1 horizon))))
        s0        (mx/array (int start) mx/int32)]
    (loop [s s0, t 0, states [s0], actions []]
      (if (>= t horizon)
        (let [st (mx/stack states)
              ac (when (seq actions) (mx/stack actions))]
          (mx/materialize! st)
          (when ac (mx/materialize! ac))
          (let [svec (mapv int (mx/->clj st))
                avec (if ac (mapv int (mx/->clj ac)) [])
                ;; host truncation: stop at the first terminal (terminals are
                ;; absorbing one-hot rows, so post-terminal tensor steps dwell and
                ;; are discarded) — reproduces the host loop's variable length.
                term-idx (or (first (keep-indexed (fn [i sv] (when (contains? terminals sv) i)) svec))
                             (count avec))]
            {:states  (subvec svec 0 (inc term-idx))
             :actions (subvec avec 0 (min term-idx (count avec)))}))
        (let [a  (step-action Q s alpha (when keys (nth keys (* 2 t))))
              s' (step-next T s a noise (when keys (nth keys (inc (* 2 t)))))]
          (recur s' (inc t) (conj states s') (conj actions a)))))))
