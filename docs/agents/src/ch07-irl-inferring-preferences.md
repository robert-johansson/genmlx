# Reasoning About Agents: Inverse Reinforcement Learning

> **Ports** agentmodels.org Ch 4, *Reasoning about agents*.

Every chapter so far has run the agent *forward*: give Pac-Man a utility
table, plan, watch him walk. This chapter runs the arrow backwards. We sit in
the gallery, watch a single step, and ask the inverse question — *what does he
want?* The glyphs and the −1-per-step time cost are on [the legend](./legend.md);
here the only thing we add is an observer.

## The maze: two caches, one corridor

The setup is the `two-caches` maze from the legend — a vertical corridor with a
small **pellet** to the north and a fat **power pellet** to the south. Pac-Man
starts in the middle. We have not been told which cache he prefers; we only get
to watch him move. He takes one step **south**.

![The two-caches maze: a vertical corridor with a pellet to the north and a cyan power pellet to the south. Pac-Man has taken one step south from the centre start cell, which is marked as part of his path; the floor is shaded by the value function so the southern half glows brighter.](figures/ch07-twocaches.png)

The floor is shaded by the value function. Notice that the brighter half is the
southern one — but that shading is *our* reconstruction, not something Pac-Man
announced. All we actually observed is a single `[state, action]` pair: *he was
in the centre, he stepped down*. Everything else has to be inferred.

## The likelihood is already written

The crucial move is that we do **not** write a new likelihood function for the
observer. The probability of an observed action under a hypothesised goal is
*exactly* the score of that action under the goal's own softmax policy — which
is the GFI's `p/assess`. The forward policy that the agent *acts* with is the
same object the observer *scores* with.

```clojure
(defn action-loglik
  "log P(action = a | state = s) under `agent`'s policy — the GFI assess weight.
   assess fully constrains :action, so no sampling happens; the auto-key is just
   to satisfy the handler and does not affect the (deterministic) score."
  [agent s a]
  (mx/item (:weight (p/assess (dyn/auto-key (:policy agent)) [s] (cm/choicemap :action a)))))
```

Read that carefully. `p/assess` takes a generative function, its arguments
`[s]`, and a choicemap that pins `:action` to the observed `a`. Because the
action is fully constrained, no sampling happens — `assess` just returns the
log-probability the policy assigns to that choice. That returned `:weight` *is*
the per-step likelihood. There is no bespoke inverse-RL code: the likelihood is
a corollary of the agent being a generative function.

Soft rationality is what makes this work. The policy is a Boltzmann softmax with
finite temperature `alpha`, not a hard argmax. If Pac-Man planned by argmax, any
sub-optimal observed step would have probability zero and instantly collapse the
posterior. With a finite `alpha`, every action has positive probability — so
evidence accumulates smoothly instead of vetoing hypotheses.

## One agent per hypothesis

To turn the likelihood into a posterior, we instantiate one forward agent for
each candidate preference. The agent that values a given cache gives it `high`
utility and every rival cache `low`; the grid, noise, and softmax `alpha` are
shared.

```clojure
(defn goal-agents
  [{:keys [grid goals high low time-cost alpha noise gamma fixed start n-iters]
    :or   {high 5.0 low 0.0 time-cost -0.1 alpha 2.0 noise 0.0 gamma 1.0 fixed {} start [0 0]
           n-iters 40}}]
  (reduce
    (fn [m g]
      (let [utils (merge (assoc (zipmap goals (repeat low)) g high :timeCost time-cost) fixed)
            mdp   (gw/build-mdp {:grid grid :utilities utils :start start
                                 :gamma gamma :noise noise})]
        (assoc m g (agent/make-mdp-agent {:mdp mdp :alpha alpha :gamma gamma :n-iters n-iters}))))
    {} goals))
```

Each hypothesis is a complete MDP agent carrying its own `:policy` and value
table `:Q`. Inference is then Bayes' rule over this finite set: prior times the
product of per-step likelihoods, normalised. `posterior-sequence` does this
incrementally, returning one posterior map per prefix length so the belief can be
revealed one observed step at a time — index 0 is the prior, index `k` the
posterior after `k` actions.

## The posterior after one southward step

Conditioning the two-caches hypotheses on Pac-Man's single step south gives this:

![Bar chart of the posterior over Pac-Man's favourite cache after one southward step: the power pellet at 0.748 and the northern pellet at 0.252.](figures/cache-posterior.png)

A southward step is roughly three-to-one evidence for the power pellet:
**P(power) = 0.748** against **P(pellet) = 0.252**. One step has moved a
uniform prior decisively, but not to certainty — under soft rationality a
power-pellet-lover *usually* heads south, and a pellet-lover *occasionally*
does too, so the posterior keeps mass on both.

## When one step is not enough

The companion tests run the richer Restaurant-Choice geometry, where three
caches replace two and we also infer the time cost and `alpha`. They make the
identifiability story precise. After one leftward step toward the near (Donut)
cache, the posterior favourite shifts to Donut — but the two *far* caches stay
tied:

> Veg vs Noodle is **unidentifiable**, with `|P(vegFav) − P(noodleFav)| < 0.05`.

The single step distinguishes near from far, but gives no evidence to separate
two caches that lie symmetrically beyond it. Their paths have not yet diverged.
This is exactly why an observer wants *multiple* trajectories — and why an
*active* learner would steer Pac-Man toward a junction that splits the
hypotheses. Feed the longer four-step trajectory all the way to the Donut cache
and the tests confirm the posterior sharpens: `P(donutFav)` rises over the
single-step value while `P(noodleFav)` — the far cache — is suppressed.

## Preference and noise, jointly

Because `alpha` is itself a hypothesis dimension, the same enumeration recovers
preference *and* rationality together. The factorised softmax posterior is built
by scoring each observation against every candidate `(utility, timeCost, alpha)`
tuple, accumulating `action-loglik`:

```clojure
(defn joint-posterior
  "Exact P(U,timeCost,alpha | observations). observations = seq of [state action].
   Weight = Σ_i action-loglik (the factorized softmax likelihood); uniform prior
   cancels in normalize-logs. Returns {tuple -> probability}."
  [agents observations]
  (inv/normalize-logs
    (into {} (for [[tup {:keys [agent]}] agents]
               [tup (reduce (fn [acc [s a]] (+ acc (inv/action-loglik agent s a))) 0.0 observations)]))))
```

The tests pin down why joint inference matters. On the *same* single leftward
step, a low-noise agent and a high-noise agent disagree about how much to read
into it. With `alpha = 0.1`, the step is nearly uninformative — `P(donutFav)`
stays within `0.03` of the prior, because a noisy agent might step left for no
reason at all. With `alpha = 100`, the very same step is sharply diagnostic,
pushing `P(donutFav)` well above the low-`alpha` value. A behaviour only counts
as evidence to the extent the agent is believed to be acting deliberately — and
the model infers that deliberateness from the same data.

There is a tempting shortcut — generate-and-compare: at high `alpha` just predict
each table's argmax path and keep the ones that match the observation. The tests
include it and show why it is the *worse* tool: because Donut South is the
nearest cache, far too many tables (even the all-equal one) route through it, so
at least 10 of the 27 candidates survive. The factorised softmax likelihood,
which weighs *how likely* each step was rather than merely *whether* it could
happen, is the discriminating one.

The agent was a generative function; that is the whole trick. Forward, it plans
and acts. Backward, the very same policy *is* the likelihood, and inference is
just Bayes' rule bolted onto `p/assess` — no new machinery, only the arrow
reversed. Next we let the world itself hide things from Pac-Man, and infer his
*beliefs* alongside his preferences.
