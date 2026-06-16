# Beyond agentmodels: LLM Policies and Remote Environments

> **Beyond agentmodels.org.** This capstone is the unfrozen frontier of
> `genmlx.agents` — it goes past the textbook to two ideas the GFI makes almost
> free: the *policy* of an agent can be a language model, and the *environment* an
> agent acts in can live on the other end of a network connection. Both fall out of
> the same thesis we have leaned on all book — an agent is a generative function —
> with **zero engine change**.

Every chapter so far has hand-built Pac-Man's policy: a softmax over expected
utilities, a biased recursion, a belief-mixed Q-row. But the policy is just a
generative function with one `:action` site. Nothing says we have to write it
ourselves. What if the distribution over actions came from a *language model*? And
what if the maze Pac-Man walks were not a tensor in this process, but a service
answering over HTTP? This chapter shows both, on the same maze, through the same
GFI.

## An LLM in policy position

The idea is exact, not metaphorical: make the per-action policy logits *be* a
language model's own log-probabilities. We describe the state in words, and for
each move word — `" left" " right" " up" " down"`, aligned to the gridworld action
indices — we ask the LLM how likely it is to say that word next. That likelihood is
recovered through the GFI, with `p/assess` on the LLM generative function:

```clojure
(defn action-logprob
  "log p(action tokens | state prompt) via the LLM GF's GFI assess op."
  [llm-gf prompt-ids act-ids]
  (let [n           (count act-ids)
        constraints (reduce (fn [cm i]
                              (cm/set-value cm (keyword (str "t" i))
                                            (mx/scalar (nth act-ids i) mx/int32)))
                            cm/EMPTY (range n))]
    (->num (:weight (p/assess llm-gf [prompt-ids n] constraints)))))
```

The LLM generative function traces one categorical token per position (`:t0`,
`:t1`, …); constraining those sites to an action word's token ids and calling
`p/assess` returns exactly `log p(word | prompt)`. Do that for all four words and
you have four numbers — the policy logits at this state:

```clojure
(defn llm-action-logits
  "The four policy logits at a state: each is the LLM GF's assess log-prob of the
   corresponding action word given the (pre-encoded) state prompt."
  [llm-gf prompt-ids act-id-lists]
  {:logits (mapv #(action-logprob llm-gf prompt-ids %) act-id-lists)
   :ntoks  (mapv count act-id-lists)})
```

And now the punchline. The policy itself is the *same* categorical generative
function we have used since Chapter 2 — only its logits happen to come from a
transformer:

```clojure
(def action-policy
  (gen [logits] (trace :action (dist/categorical logits))))
```

That is the entire composition. `p/simulate` samples an action from the LLM's
distribution; `p/assess` scores one; `p/generate` constrains one — all through the
ordinary handler, with no custom dispatch and no change to the engine. The
companion check (`agents_llm_policy_test`, which skips cleanly when no model is
present) pins the GFI nesting: the policy's `p/assess` weight equals
`logsoftmax(LLM logits)[a]` to `1e-3`, the four `p/generate` weights exponentiate
to a distribution summing to `1`, the same key reproduces the same action, and the
distribution genuinely varies from cell to cell — the LLM is conditioning on the
state, not parroting a constant.

**One honest caveat.** A 0.6-billion-parameter base model is *not* a competent
gridworld planner, and we never assert that it is. What this proves is the
*composition* — a language model dropped into policy position, scored and sampled
through the GFI like any other distribution — not that the LLM plans well. The
inverse direction is just as natural: **LLM-over-agent** inference fuses the
agent-GF likelihood `P(action | goal)` with a language-model reasoner about goals,
so the posterior over what Pac-Man wants is informed by both its trajectory and a
model's prior over plausible intentions.

## A maze on the other end of a wire

The second frontier moves the *environment* out of process. The agent/environment
boundary — act, transition, observe — is exactly the boundary a network draws, so
`genmlx.agents.remote` reifies it as real HTTP over the `genmlx.world.net` membrane.
Bridging a Pac-Man maze into a served environment is a one-liner: compile the ASCII
maze to MDP tensors, hand them to the environment handler, and let the membrane host
it:

```clojure
(defn pacman-remote-handler
  "The cs188-layout -> genmlx-remote bridge: compile a Pac-Man ASCII maze into a
   build-mdp bundle and hand it to remote/mdp-env-handler — the single handler that
   genmlx.world.net/serve! hosts."
  [maze & [opts]]
  (remote/mdp-env-handler (pac/pacman-mdp {:ascii maze}) opts))
```

The world's true location now lives *behind* the membrane, in a server-side cell
the agent cannot see; the agent learns where it landed only by receiving the next
observation across the wire. Driving it is the same act/transition loop as
`simulate-mdp`, except the transition is a POST:

```clojure
(net/with-server (pacman-remote-handler pac/corridor {:start start})
  (fn [url]
    (pr/let [rem (remote/remote-mdp-rollout ag (remote/gym-transport url) H)]
      ;; rem => {:states [...] :actions [...] :rewards [...]}
      (= (:states in-proc) (:states rem)))))   ; => true
```

The decisive property is *faithfulness*: at the deterministic regime
(`alpha = ##Inf`, noise 0) the across-wire trajectory is **bit-identical** to the
in-process one. The agent code does not change at all — only where the transition
happens. Running the bridge confirms it: the corridor rollout `[0 1 2 3 4 5 6]`
comes back identical from the server, every step a real network round-trip.

![The optimal Pac-Man rolling out through the maze — the same trajectory whether the environment is an in-process tensor or a remote service answering over HTTP; at the deterministic regime the two are bit-identical, which is the proof the membrane is faithful.](figures/classic-rollout.gif)

The figure is the in-process rollout, but that is exactly the point: across the
wire it is the same frames. (The LLM-policy figures — the per-action distribution
and the fused inverse posterior — regenerate from real model output when a local
model is present; without one they are noted rather than shown, and the build is
never blocked.) `genmlx.agents.remote` is marked **provisional**: it sits outside
the frozen v1.0 surface, and its shapes may still change.

## Where the maze ends

We started with a coin for a ghost's turn and end with a transformer choosing
Pac-Man's move over a network. Nothing about the agent had to change along the way,
because the whole book rests on one idea: an agent is a generative function, and
everything else — the planner, the belief filter, the bias, the inference backend,
the policy's *source*, the environment's *location* — is a pluggable seam around it.
A model written in Chapter 2 still runs here. That is the resource-rational,
GFI-native picture of agency this book set out to show — minds inferred, and now
also *spoken* and *served*, all in the same maze.
