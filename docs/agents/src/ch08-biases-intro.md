# Cognitive Biases and Bounded Rationality

> **Ports** agentmodels.org Ch 5 (biases intro).

So far Pac-Man has been a *good* optimiser — sometimes a perfect one, sometimes a noisy one, but always optimal in expectation. When we let him be stochastic, we did it with softmax: he picks better actions more often, worse actions less often, and the temperature α tunes how sharply. That single knob carried us a long way. This chapter is about where it runs out.

The glyphs, rewards, and the −1 step cost are all on the [shared legend](./legend.md); we lean on them here without redefining them.

## The baseline a bias departs from

Here is the agent we have been studying — the rational planner, rolled out from the centre of the maze straight to the highest-value cache.

![Animation of the optimal Pac-Man walking left then down through the maze to reach the fruit, ignoring nearer but smaller pellets along the way.](figures/classic-rollout.gif)

Watch what he *doesn't* do. He passes within a step of a small pellet on the way and never turns aside for it; he commits to the long walk to the fruit because the fruit, net of the steps it costs, is worth more. This is the behaviour every earlier chapter recovered, and it is the reference frame for everything below: **a bias is a systematic, reproducible departure from this path.**

## Two kinds of departure

Now suppose we watch a *human* play, and they do turn aside for the near pellet — every single time, in every maze where a tempting reward sits close to the start. How should our model account for it?

The softmax agent already deviates from optimal. But it deviates *randomly*. Crank α down and Pac-Man wobbles: sometimes he grabs the near pellet, sometimes the far fruit, sometimes he wanders into a wall. Over many rollouts the errors average out and centre on the optimal path. That is the signature of softmax noise — **unbiased scatter** around the right answer.

A real player's diversion is different. It is not scatter; it is a *lean*. The near pellet wins not occasionally but consistently, and the size of the pull tracks how close the temptation is, not how warm the temperature is. No setting of α reproduces this, because α only controls the *width* of the deviation, never its *direction*. A symmetric noise model cannot manufacture an asymmetric, repeatable detour. To capture systematic deviation you have to change the agent's *expected-utility computation itself*, not the noise wrapped around it.

That is exactly what the biased planners do. The whole family is built by adding one subjective axis — a *delay* — to the value recursion, and then a few pure knobs over it. The cleanest way to see the principle is the discount factor:

```clojure
(defn delta
  "Hyperbolic discount factor at subjective delay `d`: 1/(1 + k·d).
   δ(k,0)=1; strictly decreasing in d for k>0; k=0 ⇒ δ≡1 (no discounting)."
  [k d]
  (/ 1.0 (+ 1.0 (* (double k) (double d)))))
```

`delta` is the entire mechanism of one bias in three lines. With `k = 0` it returns `1.0` for every delay `d`, so future reward counts exactly as much as present reward — that is the rational agent, and at that limit the biased recursion reproduces the unbiased one exactly. Crank `k` above zero and `delta` falls off *hyperbolically* with delay: the agent values a reward one step away at `1/(1+k)`, two steps away at `1/(1+2k)`, and so on. The fruit, being far, gets discounted hard; the near pellet, being close, barely at all. The temptation now *outscores* the fruit in the agent's own arithmetic — and it does so every time, in the same direction. That is systematic deviation, produced inside the value computation rather than bolted on as noise.

A second bias enriches the model with *myopia* — a cap on how far ahead the agent looks at all. The biased agent constructor exposes both as plain options:

```clojure
(defn make-biased-mdp-agent
  "Build a biased MDP agent over `mdp` with a `bias` map
     {:discount k :bias :naive|:sophisticated :reward-myopic-bound C_g}.
   ...
   `:act` re-plans from delay 0 each
   call (agentmodels' `delay || 0`), which is what produces the Naive plan↔do
   divergence. `simulate-biased-mdp` (= agent/simulate-mdp) rolls it out."
  [{:keys [mdp alpha gamma n-iters] :or {alpha ##Inf gamma 1.0 n-iters 24}}
   {:keys [discount bias reward-myopic-bound]
    :or   {discount 0.0 bias :sophisticated reward-myopic-bound ##Inf} :as biasm}]
  ...)
```

The `bias` map is the whole vocabulary of bounded rationality this book uses: `:discount k` is time inconsistency (the hyperbolic pull of `delta`), `:reward-myopic-bound C_g` is myopia (look no further than `C_g` steps), and `:bias :naive` vs `:sophisticated` decides whether the agent *knows* it is time-inconsistent. Note the defaults — `discount 0.0`, `reward-myopic-bound ##Inf` — are precisely the *un*-biased agent. A biased Pac-Man is the rational one with one or two knobs nudged off their neutral settings, which is what makes "bias as a systematic departure from the baseline" a literal statement about the code.

## Why this matters two ways

There are two reasons to care, and they are mirror images of each other.

The first is **solving problems**: if you want an agent that *acts well* in a maze, you want it unbiased — you turn the knobs to their neutral defaults and let it plan optimally. Bias is a defect to be engineered away.

The second is **learning preferences**: if you want to *infer* what a real player wants from watching them play, you must model the bias, because a player who skips the fruit is not telling you the fruit is worthless — they are telling you they discount it. An inverse planner that assumes the player is unbiased will misread that detour as "prefers the near pellet" and recover the wrong utilities. Only an agent model rich enough to contain the bias can subtract it back out and reveal the true preference underneath. Bias here is not a defect; it is the thing the data is shaped by, and modelling it is the price of reading the data honestly.

The same generative function serves both: turn the knobs off to act, turn them on (and infer their settings) to understand. That duality — decision models as tools for *doing* and as models for *interpreting* — is the spine of the chapters that follow.

Next we make the first of these biases concrete, and watch a Naive Pac-Man's plan diverge from what he actually does.
