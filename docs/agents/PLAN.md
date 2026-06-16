# GenMLX Agents — Authoring Contract

The methodology every chapter follows. Mirrors `docs/tutorial-v2/PLAN.md`, adapted
for the agents book.

## Design principles

- **One running environment** — the Pac-Man maze (`genmlx.agents.pacman`). Pellets
  are rewards, the power pellet a delayed payoff, a hidden ghost is latent state.
- **Show, then explain** — every concept starts with runnable code before theory.
- **Code is quoted, never re-typed** — every listing is lifted verbatim from a
  runnable `examples/agentmodels/*.cljs`, which is the single source of truth (it is
  linted to the `genmlx.agents` public API and runs green; see milestone E0).
- **Every figure is live** — a PNG/GIF regenerated from a real rollout or posterior
  via the in-repo capture pipeline (milestone E2). No hand-drawn diagrams.
- **Hybrid figures** — Pac-Man screenshots/GIFs for behaviour & environment; bar/
  line charts (from real inference output) for posteriors. Both are data-driven.
- **Pac-Man-primary, with honest exceptions** — ghosts are POMDP latent state, never
  joint planner state; the PPL primer and RSA stay plain; adversarial multi-ghost
  uses classical tree search, labeled non-GFI.

## Per-chapter authoring template

```markdown
# <Chapter Title>

> **Ports** agentmodels.org <Ch X>.

<1-2 paragraphs: the question this chapter answers, framed in the maze.>

## <Concept>

<prose>  -->  ```clojure  <code lifted from examples/agentmodels/<file>.cljs>  ```

![alt text describing the figure](figures/<chNN-name>.png)

<interpretation of the figure / result>
```

## Figures

- Live PNG/GIF only, stored in `src/figures/`, referenced as `figures/<name>.png`.
- Every figure has descriptive **alt text** (accessibility + the "what am I seeing").
- One command regenerates all figures deterministically (milestone E2).

## Test strategy

- Each chapter's code runs from its `examples/agentmodels/*.cljs` with **zero FAIL**
  under `bun run --bun nbb`, and is covered by a companion `test/genmlx/agentmodels_*`
  (most already exist and pass).

## Verification protocol (per chapter)

1. Author the Markdown, quoting code from the example file.
2. Confirm the example runs green; cross-check the quoted code matches the file.
3. Generate the chapter's figures from real runs; embed with alt text.
4. `cd docs/agents && mdbook build` exits 0; visual check.

## Chapter ↔ example ↔ test mapping (source of truth)

Every chapter quotes code from the file(s) below (the runnable, idiom-/API-linted
`examples/agentmodels/*` from milestone E0, or library code directly where no
dedicated demo script exists — the capability lives in the library + its tests).

| Chapter | Quotes from | Companion test(s) |
|---|---|---|
| ch01 PPL primer | `ch01_intro.cljs`, `ch02_webppl.cljs` | `agentmodels_ch01_intro_test`, `agentmodels_ch02_webppl_test` |
| ch02 agents as programs | `helpers.cljs` (factor-dist, softmax-action) | `agentmodels_helpers_test` |
| ch03 MDPs | `worlds.cljs` (line-mdp), `agent.cljs` | `agentmodels_worlds_test`, `agentmodels_slice_test` |
| ch04 gridworld | `worlds.cljs` (hike), `agent.cljs`, `pacman.cljs` | `agentmodels_worlds_test` |
| ch05 POMDPs | `belief.cljs`, `pomdp.cljs`, `pomdp_env.cljs` | `agentmodels_pomdp_test`, `agentmodels_pomdp_adjacency_test`, `agents_fused_pomdp_test` |
| ch06 bandits & PSRL | `psrl.cljs`, `pomdp.cljs` (bandit), `worlds.cljs` (lava) | `agentmodels_psrl_test`, `agentmodels_psrl_lava_test` |
| ch07 IRL | `irl.cljs`, `pomdp_irl.cljs`, `hike_irl.cljs`, `inverse.cljs` | `agentmodels_irl_test`, `agentmodels_hike_irl_test` |
| ch08 biases-intro | narrative (sets up `biased_planners.cljs`) | — |
| ch09–ch11 biases | `biased_planners.cljs` | `agentmodels_biased_planners_test` |
| ch12 joint I | `joint_5d_inference.cljs` | `agentmodels_5d_joint_test` |
| ch13 joint II | `restaurant_joint_inference.cljs`, `biased_inverse.cljs` | `agentmodels_5e_joint_test`, `agentmodels_biased_inverse_test` |
| ch14 efficient/pluggable | `ch06_pluggable_inference.cljs`, `differentiable.cljs` | `agentmodels_ch06_pluggable_inference_test`, `agentmodels_diff_learn_test` |
| ch15 multi-agent | `multi_agent.cljs` | `agentmodels_multi_agent_test` |
| ch16 library guide | `ch08_library_guide.cljs` | `agentmodels_ch08_library_test`, `agents_api_test`, `agents_contracts_test` |
| ch17 LLM capstone | `remote.cljs`, llm policy/inverse | `agents_llm_policy_test`, `agents_llm_inverse_test` (model-gated) |

Chapters ch02–ch05 and ch09–ch11 quote library namespaces directly (plus their
tests); dedicated per-chapter demo scripts are optional and may be added later.
All quoted example files pass `bun run --bun nbb` with zero FAIL (verified in E0.2).
