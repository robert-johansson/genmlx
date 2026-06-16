# Joint Inference II: Was That Detour Noise, Bias, or Taste?

> **Ports** agentmodels.org Ch 5e (joint inference II).

Watch a Pac-Man pass right by a power pellet — fifty points, glowing cyan, one corridor away — and stop to eat a plain ten-point pellet first. Why? Three stories fit equally well from a single glance:

- He is **noisy**: a softmax actor with low `alpha` who simply slipped.
- He is **biased**: a *Naive* hyperbolic discounter, tempted by the nearer reward because his future self looks (to him) like it will hold out.
- He has **taste**: he genuinely values plain pellets more than power pellets.

One observation cannot tell these apart. This chapter is about what *can*: watching the same detour happen again, and giving taste a seat at the table so it has to compete with bias. (For the glyphs and the −1-per-step scoring, see [the legend](./legend.md).)

## The temptation maze as an MDP

We strip the maze down to its decision spine. From the start, Pac-Man can head toward the **temptation gate** (action `0`) or take the **safe route** (action `1`). At the gate he can succumb and eat the donut now (Rd = 5), or continue on to the more valuable veg (Rv = 8). The safe route reaches veg with no temptation along the way. Here is the hand-built MDP, copied verbatim from `examples/agentmodels/biased_inverse.cljs`:

```clojure
(defn temptation-mdp
  "Build the minimal 5-state / 2-action temptation MDP (see namespace notes).
   Returns an MDP map {:S :A :T :R :terminals} feeding make-biased-mdp-agent."
  []
  (let [S 5
        A 2
        ;; T[s][a][s'] one-hot transitions
        T-rows [[[0 1 0 0 0] [0 0 1 0 0]]    ; start : a0→tempt, a1→safe1
                [[0 0 0 1 0] [0 0 0 0 1]]    ; tempt : a0→donut(eat), a1→veg(continue)
                [[0 0 0 0 1] [0 0 0 0 1]]    ; safe1 : both → veg
                [[0 0 0 1 0] [0 0 0 1 0]]    ; donut : absorbing
                [[0 0 0 0 1] [0 0 0 0 1]]]   ; veg   : absorbing
        R-rows [[0 0]                         ; start
                [5 0]                         ; tempt : a0 eat donut (Rd=5), a1 continue
                [0 0]                         ; safe1
                [0 0]                         ; donut
                [8 8]]                        ; veg   : Rv=8
        T (mx/array (clj->js T-rows) mx/float32)
        R (mx/array (clj->js R-rows) mx/float32)]
    {:S S :A A :T T :R R :terminals {3 :donut 4 :veg}}))
```

The reward crossover is the whole trick. With hyperbolic discount `k = 1`, a self standing *at the gate* (delay 0) values the donut at `δ(1)·5 = 2.5` and veg at `δ(2)·8 = 8/3 ≈ 2.667` — so the gate-self would prefer veg, barely. But a self planning *from the start* (delay 1) values the donut even less. The result is a textbook preference reversal, and it splits the two biases cleanly. The companion test `agentmodels_biased_inverse_test.cljs` pins the expected utilities at the start state: the **Naive** agent sees `EU(start) = [8/3, 8/3]` — a dead tie, because he believes his future self will resist temptation, so both routes look like they reach veg. The **Sophisticated** agent sees `EU(start) = [2.5, 8/3]` — he foresees his own succumbing on the tempting route, and discounts it.

Those expected utilities become softmax policies. At the decisive softmax limit (`alpha = ##Inf`):

- `π_naive(a0) = 0.5`, `π_naive(a1) = 0.5` — the Naive agent is *indifferent* at the start and may head for temptation.
- `π_soph(a0) = 0.0`, `π_soph(a1) = 1.0` — the Sophisticated agent *never* heads for temptation.

Heading for the gate is therefore diagnostic of Naivety; taking the safe route is evidence — but not proof — of Sophistication.

## Bias as a traced latent

The point of `genmlx.agents` is that an agent is a generative function, and the thing we want to infer about him — his bias — is just one more traced random choice inside it. The joint model below draws `:bias` from the prior, lets that draw *select* a precomputed forward planner, and then emits one softmax-action site per observed state:

```clojure
(defn biased-agent-model
  [{:keys [states alpha prior mdp] :or {alpha ##Inf} :as cfg}]
  (let [agents    (bias-agents cfg)
        n-actions (:A mdp)
        box       (if prior
                    (h/weighted-draw bias-values prior)
                    (h/uniform-draw bias-values))
        ;; Precompute the policy logit array per (bias, state) ONCE — the gen body
        ;; is re-run per IS particle, so it must only index, never rebuild arrays.
        rows      (into {} (for [b bias-values]
                             [b (mapv #(mx/array (clj->js (bp/eu-row (agents b) % n-actions)) mx/float32)
                                      states)]))]
    (gen []
      (let [bi   (trace :bias (:dist box))
            bias (h/draw-value box bi)
            er   (rows bias)]
        (doseq [t (range (count states))]
          (trace (keyword (str "a" t))
                 (h/softmax-action alpha (nth er t))))
        bi))))
```

The likelihood of an observed action is *just the agent's own softmax policy probability* — there is no bespoke likelihood function anywhere. Inverting the model is then mechanical: enumerate the finite bias prior, `assess` the full trajectory under each value, and normalize. Because the bias set is finite and `assess` only scores, this is the exact closed-form posterior — no sampling, no approximation. Inference is fully pluggable: the same model also accepts importance sampling as a cross-check, with no change to the agent.

## One detour proves little

Condition on a single safe-route observation. With a uniform prior over `{:naive :sophisticated}`, the test reports:

```
P(:naive | safe×1) = 1/3        P(:sophisticated | safe×1) = 2/3
```

That is the figure below: after watching the careful route exactly once, the odds are two-to-one that Pac-Man is Sophisticated rather than Naive (0.667 vs 0.333). The arithmetic is transparent — each safe step is twice as likely under Sophisticated (`π_soph(a1) = 1` vs `π_naive(a1) = 0.5`), so the 1:2 likelihood ratio tilts a 1:1 prior to a 1:2 posterior.

![Two posterior bars over Pac-Man's bias after one safe-route observation: the Naive bar at one-third and the taller Sophisticated bar at two-thirds, a two-to-one lead.](figures/ch13-bias-posterior.png)

The mirror case is sharper. If Pac-Man instead heads *for* the gate even once, the Sophisticated story is annihilated — `π_soph(a0) = 0` — so `P(:naive | tempt×1) = 1.0` exactly. A single move toward temptation is a smoking gun; a single careful move is only suggestive.

## Seeing it three times collapses the noise story

Now the crux. A genuinely careless agent would slip *occasionally*, not reliably. So repeat the safe-route observation and watch the explanations sort themselves out. The exact posterior follows a clean law, asserted in the test as `P(:soph | n safe) = 1/(1 + (1/2)ⁿ)`:

- n = 1: P(:soph) = 2/3 ≈ 0.667
- n = 2: P(:soph) = 4/5 = 0.800
- n = 3: P(:soph) = **8/9 ≈ 0.889**

Each repeated detour multiplies the likelihood ratio by another factor of two, and the posterior climbs monotonically — the test verifies `P(:soph|×1) < P(:soph|×2) < P(:soph|×3)`. A noise explanation cannot keep up with that, because noise would not produce the *same* deliberate choice three times running. Repetition is what separates a systematic bias from a random slip.

This identifiability survives softer rationality, too. Drop to a finite `alpha = 2`, where even the Sophisticated agent occasionally wanders toward temptation, and the posterior loosens but still points the right way: the test pins `P(:soph | safe×3) ≈ 0.5515` and the mirror `P(:naive | tempt×3) ≈ 0.5638`, both safely above one-half. An importance-sampling run of N = 5000 particles agrees with the exact answer to within 0.05, with effective sample size comfortably above N/4 — the pluggable sampler reproduces the closed-form posterior it had no part in deriving.

## Letting taste compete with bias

So far the only explanations on offer were bias and noise. But the original detour had a third reading — maybe Pac-Man just *likes plain pellets*. To let taste compete, we expand the latent space from one dimension to the full restaurant-twin preference table. The joint model in `restaurant_joint_inference.cljs` traces seven indices at once — the immediate and delayed utilities of donut and veg, the discount, the bias, and the softmax `alpha`:

```clojure
(defn joint-posterior
  "Exact P(params | trajectory). Enumerate the joint finite prior; for each tuple
   assess the joint GF on the full choicemap (latent indices + observed actions);
   normalize. `:number-repeats` r conditions on the SAME action sequence r+1 times
   (agentmodels numberRepeats — strengthens the evidence). Returns
   {:joint {tuple prob} :posterior <summary> :prior <summary> :n-tuples n}."
  [{:keys [states actions number-repeats] :or {number-repeats 0} :as spec} agents]
  (assert (= (count states) (count actions))
          (str "5e joint: states/actions length mismatch " (count states) " vs " (count actions)))
  (let [model   (dyn/auto-key (joint-restaurant-model spec states agents))
        reps    (inc number-repeats)
        tuples  (keys agents)
        logw    (into {}
                      (for [tup tuples]
                        [tup (* reps (mx/item (:weight (p/assess model []
                                                                 (full-cm tup actions)))))]))
        post    (inv/normalize-logs logw)
        uniform (let [n (count tuples)] (mapv (fn [t] [t (/ 1.0 n)]) tuples))]
    {:joint     post
     :posterior (summarize post agents)
     :prior     (summarize uniform agents)
     :n-tuples  (count tuples)}))
```

The summary statistics are the verbatim webppl-agents predicates — `donut-tempting?` tests whether the discounted utilities encode a preference reversal (veg preferred from afar, donut preferred up close), and `veg-minus-donut` measures net taste. With the discount fixed at `k = 1`, conditioning on the tempting detour raises `P(donutTempting)` from a prior under 0.1 toward ~0.9, and `E[vegMinusDonut]` turns positive — the agent net-prefers veg yet was tempted, exactly the discounting signature. And the `:number-repeats` knob does for the joint model what repetition did for the bias model: seeing the same path again strengthens the discounting explanation over the high-noise one, because noise, unlike a stable bias or a stable taste, does not repeat.

The same maze, the same GFI `assess`, three rival explanations — and the only thing that ever decided between them was *how many times you looked*.

Next: we have inferred over agents by exact enumeration throughout — the closing
chapters run the *same* inference through interchangeable backends (enumeration,
sampling, gradient) that must agree, making inference a pluggable axis of its own.
