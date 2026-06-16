# Time Inconsistency II: Procrastination and Changing Plans

> **Ports** agentmodels.org Ch 5b (time inconsistency II).

Pac-Man has a power pellet to eat and a deadline to eat it by. Every step he can
**work** (grab the pellet now, paying a small cost up front) or **wait** (drift
one step closer to the deadline, paying a tiny patience cost). The reward for the
pellet lands one step *after* he works — so the cost is felt now, the payoff
later. A patient agent works on day zero. A present-biased agent keeps telling
itself *I'll do it tomorrow* — and tomorrow it says the same thing.

This chapter is about the gap between what a hyperbolic agent **believes it will
do** and what it **actually does**. For a Naive agent those two stories diverge,
and that divergence *is* time inconsistency. The glyphs and scoring are on the
[shared legend](./legend.md); here we only need *work* and *wait*.

## Why the reversal happens

The hyperbolic discount factor weights a reward by how many subjective steps away
it sits. In `biased_planners.cljs` it is one line:

```clojure
(defn delta
  "Hyperbolic discount factor at subjective delay `d`: 1/(1 + k·d).
   δ(k,0)=1; strictly decreasing in d for k>0; k=0 ⇒ δ≡1 (no discounting)."
  [k d]
  (/ 1.0 (+ 1.0 (* (double k) (double d)))))
```

At `k = 0` this is flat — `δ(0, d) = 1` for every `d`, recovering the unbiased
exponential agent. For `k > 0` it drops fast and then *flattens*: the companion
test pins `δ(2, 3) = 1/7` and checks `δ(1, ·)` is strictly decreasing. The
flattening is the whole story. Seen from far away (large delay) two future
rewards look almost equally distant, so the agent happily prefers the larger
later one. Seen from up close (delay 0) the near reward towers over the far one.
As the agent walks toward a decision, the curve under its feet steepens and its
preference *reverses*.

![Hyperbolic versus exponential discount curves: the exponential curve decays at a constant proportional rate while the hyperbolic curve drops steeply at first and then flattens, so two distant rewards look nearly equal but a near reward dominates — the crossing that produces preference reversal.](figures/ch09-discount-curves.png)

The exponential curve never crosses itself under a shift in viewpoint; the
hyperbolic one does. That crossing is exactly the procrastination spiral: from
tomorrow's vantage, working looks worth it; from today's, waiting always wins.

## Believe versus do

The agent's recursion carries two clocks. Objective time `t` drives the deadline;
subjective delay `d` drives `delta`. The single line that separates a Naive agent
from a Sophisticated one is which delay it hands to its *simulated future self*
when predicting that self's action:

```clojure
(defn bias->perceived-delay
  "The perceivedDelay function the agent uses for its SIMULATED future self when
   choosing that self's action (agentmodels: perceivedDelay = naive ? delay+1 : 0).
     :naive         → (fn [d] (inc d))   ; believes future self stays patient
     :sophisticated → (fn [_] 0)         ; models future self's present bias
   Only the policy that picks the simulated future action uses this; the
   continuation VALUE always uses the true d+1. At k=0 (δ≡1) both coincide and
   equal the unbiased agent. (Default → sophisticated/0 for any other key.)"
  [bias]
  (case bias
    :naive         inc
    :sophisticated (constantly 0)
    (constantly 0)))
```

A **Naive** agent (`inc`) models its future self as keeping the same ever-growing
delay clock — so the future always looks patient, and the agent predicts it will
resist temptation later. A **Sophisticated** agent (`constantly 0`) correctly
models its future self as re-planning from delay 0, carrying the same present
bias it has now, so it *foresees* that it would succumb.

This belief is captured by `planned-rollout` — the trajectory the agent *thinks*
it will follow. At each predicted future step it advances the policy delay through
its own `pd-fn`:

```clojure
(defn planned-rollout
  "The trajectory the agent BELIEVES it will follow — predicting each future
   action with the perceivedDelay its OWN recursion uses (Naive: the policy delay
   grows along the plan, so the far future looks patient; Sophisticated: it stays
   0). Contrast with `simulate-biased-mdp`, which re-plans from delay 0 each REAL
   step. For a Naive agent the two DIVERGE — its plan reaches the patient goal while
   it actually defects to the nearby temptation — which is exactly time-inconsistency.
   For a Sophisticated (or k=0) agent plan == do. ..."
  [{:keys [mdp eu params]} start horizon]
  (let [{:keys [T terminals A]} mdp
        pd-fn (bias->perceived-delay (:bias params))
        H     (:horizon params)]
    (loop [s start, d 0, step 0, states [start], actions []]
      (if (or (>= step horizon) (contains? terminals s))
        {:states states :actions actions}
        (let [a  (argmax-of (mapv #(eu s % H d) (range A)))    ; predicted action at policy-delay d
              s' (agent/sample-next T s a)]
          (recur s' (pd-fn d) (inc step) (conj states s') (conj actions a)))))))
```

Notice the recursion: after each predicted step the delay becomes `(pd-fn d)`.
For a Naive agent that is `(inc d)`, so the believed future drifts ever more
patient and the plan eventually grabs the pellet. The plan is a fiction the agent
tells itself.

What the agent *actually* does is a different function, and the difference is one
word — it re-plans from delay 0 on every real step:

```clojure
(defn simulate-biased-mdp
  "Roll a biased MDP agent out from `start` for ≤ horizon steps (= agent/simulate-mdp's
   :host path, which re-plans via :act each step — the d=0 re-planning that drives
   Naive time-inconsistency). :rollout-mode :fused is unsupported: biased agents
   carry no tensor :Q, so rollout/rollout-mdp would deref nil (genmlx-m3nn) — use
   the default :host rollout."
  [ag start horizon & [opts]]
  (when (= (:rollout-mode opts) :fused)
    (throw (ex-info (str "simulate-biased-mdp does not support :rollout-mode :fused — "
                         "biased agents have no tensor :Q; use the default :host rollout")
                    {:agent-keys (keys ag)})))
  (agent/simulate-mdp ag start horizon opts))
```

`simulate-biased-mdp` delegates to the ordinary host rollout, whose `:act`
evaluates expected utility at delay 0 each step (the agent's `:act` is built to
re-plan from delay 0 — agentmodels' `delay || 0`). So `planned-rollout` lets the
delay grow along the imagined future; `simulate-biased-mdp` resets it to 0 every
real step. For a Naive agent these read out two different trajectories from the
*same* expected-utility function. That mismatch is time inconsistency made into
two callable functions.

## What the numbers say

Build the procrastination MDP (reward 4.5, work cost −1, wait cost −0.1, deadline
10) and sweep the discount rate `k`. At `k = 0` the Naive agent **works on day 0**
— it is just the unbiased agent. As `k` climbs the first work-day is
non-decreasing (it procrastinates more), and the test pins the headline reversal:
the Naive agent **works at `k = 0` but never completes at `k = 4`**, and still
fails to finish by the deadline at `k = 8`. The preference flips.

The mechanism is asserted directly at the start state `W_0`. With `k = 4`:

```clojure
(let [n4 (bp/make-biased-mdp-agent {:mdp pmdp :alpha ##Inf :gamma 1.0 :n-iters 14}
                                   {:discount 4.0 :bias :naive})
      H  14, eu (:eu n4)]                       ; action 0 = wait, 1 = work
  ;; PLANS to work — its planned-rollout contains a work action
  (some #(= 1 %) (:actions (bp/planned-rollout n4 0 10)))
  ;; but NEVER works in reality — simulate-biased-mdp contains no work action
  (not-any? #(= 1 %) (:actions (bp/simulate-biased-mdp n4 0 10)))
  ;; W_0 preference reverses:
  (and (> (eu 0 0 H 0) (eu 0 1 H 0))     ; at d=0: EU(wait) > EU(work)
       (> (eu 0 1 H 1) (eu 0 0 H 1))))   ; at d=1: EU(work) > EU(wait)
```

The same `eu` function says *wait beats work at delay 0* and *work beats wait at
delay 1*. From tomorrow's view (`d = 1`) the agent prefers to have worked; from
today's view (`d = 0`) it prefers to wait. `planned-rollout` reads the optimistic
`d = 1`-and-beyond story and reports a plan that works; `simulate-biased-mdp`
re-reads `d = 0` every step and reports a trajectory that never does.

The Sophisticated agent, seeing this in advance, behaves differently — and the
test quantifies the advantage: at `k = 2` the Sophisticated agent **completes
(works)** where the Naive agent **never does**, and across every swept `k` in
`[0, 0.5, 1, 2, 4]` the Sophisticated agent works no later than the Naive one.
Foreseeing the spiral is what lets it commit.

The contrast survives in space, too. In the Restaurant-Choice grid (Section 3 of
the same test) a Naive agent with `k = 3` **plans to reach the high-value Veg
cache but is actually captured by the adjacent Donut**, while the Sophisticated
agent's plan equals its action — both reach Veg. `planned-rollout` returns Veg;
`restaurant-endpoint` returns Donut-North; the two disagree, and that disagreement
is, once more, the whole of the phenomenon.

Next: when the agent cannot even see far enough to plan — bounding the look-ahead
itself, with reward-myopia and bounded value of information.
