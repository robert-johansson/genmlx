# Sequential Decisions: MDPs and the Maze

> **Ports** agentmodels.org Ch 3a (MDPs).

So far Pac-Man has made *one* choice at a time. But the interesting thing about a maze is that choices are linked: where you step now decides which steps are available next, and the pellet at the end is only worth chasing if the corridor between here and there isn't too long. To reason about a *sequence* of decisions we need a model of how the world unfolds — a **Markov decision process**, or MDP.

The glyphs and the −1 step cost are fixed on the shared [legend](./legend.md); this chapter is about what to *do* with them.

## The minimal maze: a corridor

Strip the maze down to a single row and put one power pellet at the far right. That is the **corridor** — agentmodels' integer line-world, the smallest MDP you can fully trace by hand. Pac-Man starts at the left; every step costs −1; reaching the pellet ends the episode with its reward. The only question is whether the pellet is worth walking to.

`line-mdp` builds exactly this world as plain data, then compiles it through the gridworld builder:

```clojure
(defn line-mdp
  "Build the Ch 3a integer line-world: an n-state 1-D corridor with a single
   :goal reward and a per-step timeCost (so the agent reaches the goal quickly).
   Movement is left/right; up/down clamp to a stay since the grid is one row high.

   Options: :n (default 7) :goal-idx (default n-1) :reward (default 1.0)
            :time-cost (default -0.1) :start (default [0 0]) :gamma (default 1.0)."
  [{:keys [n goal-idx reward time-cost start gamma]
    :or {n 7 reward 1.0 time-cost -0.1 gamma 1.0}}]
  (let [goal-idx (or goal-idx (dec n))]
    (gw/build-mdp {:grid (line-grid n goal-idx)
                   :utilities {:goal reward :timeCost time-cost}
                   :start (or start [0 0]) :gamma gamma :noise 0.0})))
```

Five things define the MDP, and they are all right here: the **states** (the `n` cells of the grid), the **actions** (left/right, with up/down clamping to a stay), the **transition function** (`:noise 0.0`, so a move lands deterministically one cell over), the **per-step time cost** (`:timeCost`, the pressure to finish), and the **terminal reward** (`:goal`, the cell that ends the episode and pays out). `build-mdp` turns these into the dense tensors `T` (transition), `R` (reward), and `term` (terminal mask) that the planner consumes.

Here is the corridor in motion — Pac-Man walking rightward, one frame per step, to the power pellet that closes the episode:

![Animation of Pac-Man walking rightward along a one-row corridor, cell by cell, until he reaches the power pellet at the far right end.](figures/corridor-rollout.gif)

The walk looks obvious, but *why* Pac-Man heads right rather than left is a value computation — and that is the heart of the chapter.

## Value: the same answer, computed two ways

The value `V(s)` of a state is the total future reward Pac-Man expects from there, acting by his own policy. agentmodels defines this with two mutually recursive functions: `expectedUtility(s, a)` — the value of taking action `a` in state `s` — and the policy, a *softmax* over those expected utilities. Softmax (controlled by a rationality parameter `alpha`) means the agent strongly prefers high-value actions but isn't a perfect maximizer; in the `alpha = ##Inf` limit it becomes the hard `max`.

GenMLX computes this value **two ways that agree by construction**, and the agreement is the point: it certifies that the fast tensor path means the same thing as the faithful recursive one.

The first path is **tensor value iteration** — the Bellman backup, run as MLX graph ops on the GPU. Start from `V = 0` and sweep:

```clojure
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
```

Each sweep computes `Q[s,a] = R + gamma·(1−term)·(T·V)` and then folds `Q` back into a new `V` — the soft policy-expectation `V(s) = Σ_a softmax(alpha·Q[s])[a]·Q[s,a]`. The `mx/materialize!` call forces each sweep's `V` to a concrete array so the lazy graph doesn't grow without bound across iterations. After enough sweeps `V` stops changing: it has converged.

The second path is the **faithful recursive `expectedUtility`** — agentmodels' `act`/`expectedUtility` recursion almost line for line, on host scalars, memoized with `exact/with-cache` (the `dp.cache` analog):

```clojure
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
  ...
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
```

Read the `EU` formula in the docstring as a sentence: the value of doing `a` in `s` is the immediate reward `u(s,a)`, plus — unless `s` is terminal or we've run out of time — the discounted average over next states of *their* value one step shorter. The `V` formula closes the loop by mixing the `EU`s under the agent's own softmax policy. That mutual recursion is exponential if computed naively: each state expands into all its successors, which expand again. `exact/with-cache` is what tames it — every `(s, a, t)` triple is computed once and remembered, turning the exponential tree into a polynomial table. This is the chapter's load-bearing lesson, and the reason memoization isn't an optimization detail but a correctness-of-scale requirement.

Both paths return the same `Q` and `V` (this is certified by the equivalence test). `EU(s, a, horizon)` from the recursive path equals `Q[s,a]` from `n = horizon` value-iteration sweeps.

## Watching value propagate

Why does value iteration need *several* sweeps rather than one? Because reward information has to travel backward through the maze. On sweep one, only cells adjacent to the goal learn they're valuable. On sweep two, their neighbors learn it. The goal's worth ripples outward one cell per sweep until it reaches the start.

The plot below tracks `V(start)` — the value Pac-Man assigns to where he's standing — across successive sweeps:

![Line chart of V(start) versus value-iteration sweep number: the value sits flat and low for several early sweeps while the goal reward has not yet propagated back across the corridor, then jumps sharply once the information reaches the start cell and plateaus at the converged value.](figures/vi-convergence.png)

The line tracks one cell; the maze shows them all. Here value iteration runs on the
classic maze, the floor shaded by `V(s)` after each sweep — the worth of the fruit
spilling outward one ring of cells at a time until the whole maze knows the way home:

![Animation of value iteration on the classic maze: the floor begins uniformly dark (every cell at the bare step cost), then over successive sweeps brightens outward from the fruit as the goal's value propagates back cell by cell, converging to the value heatmap.](figures/ch03-vi-propagation.gif)

The flat early region is Pac-Man not yet knowing the pellet is worth walking to — the reward signal simply hasn't reached the start. The jump is the moment the backup chain connects start to goal; the plateau is convergence. The number of sweeps you need is the distance the reward must travel, which is exactly why the corridor's horizon (`n-iters`) is set generously.

## Acting it out

`make-mdp-agent` packages all of this: it runs value iteration to get `Q` and `V`, wraps the softmax policy as a generative function, and exposes `:act` (sample one action) and `:expected-utility` (the recursive value, computed lazily only if you ask). The policy being a generative function is the unifying idea of the whole book — choosing an action is just `(trace :action (softmax-action alpha (Q s)))`, a GFI random choice like any other.

To produce a full trajectory, `simulate-mdp` rolls the policy forward from the start until a terminal cell:

```clojure
(defn simulate-mdp
  "Roll the agent's policy out from `start` for at most `horizon` steps, stopping
   at a terminal. The action comes from the softmax policy (decision noise) and
   the next state is sampled from T (environment / transition noise). Returns
   {:states [s0 s1 ...] :actions [a0 a1 ...]} (JS ints); one action per transition."
  [{:keys [mdp act] :as agent} start horizon & [{:keys [rollout-mode key]}]]
  (if (= rollout-mode :fused)
    (rollout/rollout-mdp agent start horizon {:key key})
    (let [{:keys [T terminals]} mdp]
      (loop [s start, step 0, k key, states [start], actions []]
        (if (or (>= step horizon) (contains? terminals s))
          {:states states :actions actions}
          (let [[k-act k-next k'] (if k (rng/split-n k 3) [nil nil nil])
                a  (if k-act (act s k-act) (act s))
                s' (sample-next T s a k-next)]
            (recur s' (inc step) k' (conj states s') (conj actions a))))))))
```

Two kinds of randomness meet in that loop: the **decision noise** of the softmax policy (`act`) and the **environment noise** of the transition (`sample-next` draws the next state from `T[s,a,:]`). For the corridor both are tame — `noise 0.0` makes the transition one-hot — so the rollout is the clean rightward walk you saw in the GIF. The whole thing in practice is one builder and one roll:

```clojure
(let [agent (make-mdp-agent {:mdp (line-mdp {:n 7})})]
  (simulate-mdp agent 0 12))
;; => {:states [0 1 2 3 4 5 6] :actions [...]}
```

That is the corridor solved: a line of states, a backed-up value function, and a policy that walks the gradient to the pellet.

Next we widen the corridor into two dimensions — and discover that with walls and a cliff, the *shortest* path and the *best* path stop being the same thing.
