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
