# GenMLX — self-demonstrating terminal slide deck

A **full-screen, split-pane** [ink](https://github.com/vadimdemedes/ink)
(React-for-the-terminal) slide deck that runs under **nbb**, where every "figure" is
a **live GenMLX computation in the same process** — models simulate/condition/vectorize
on screen, real LLMs generate, and the deck's own navigation is itself a generative
function.

It takes over the whole terminal (alternate screen buffer + a full-height root box),
and each slide is split: **narrative on the left, a live visual on the right** — an
ASCII histogram/bar chart for the numeric slides, the LLM-generated ClojureScript
source for the synthesis slide, or live numeric output. It presents the eight
distinctive features from `examples/distinctive/` as live slides.

## Run

```bash
# one-time: install ink/react (shared with the TUI prototype)
(cd ../genmlx-tui && npm install)

# interactive deck (needs a real terminal)
./run.sh
```

Keys: **← / →** (or `n`/`p`/space) move · **r** (or Enter) run the current slide's
demo · **q** quit.

The LLM slides (03/04/05) use `qwen3.5-4b-mlx-bf16` from `~/.cache/models`; it loads
asynchronously at startup (a `loading…/ready ✓` badge shows in the header), so the
pure-GenMLX slides are usable immediately while it loads.

### Headless self-test (no TTY)

Verifies every figure actually computes, without starting the interactive UI:

```bash
DECK_SELFTEST=1    ./run.sh   # the 5 pure-GenMLX figures + the nav gen function (fast)
DECK_SELFTEST=full ./run.sh   # also loads the LLM and runs 03/04/05
```

## How it works

- **Stack:** `reagent` (bundled inside nbb) for hiccup components and reactive
  `r/atom` state; `ink` for terminal rendering + the `useInput`/`useApp` hooks
  (used via a reagent `:f>` function component); `genmlx.*` for the probabilistic core.
- **Classpath:** `run.sh` runs node `nbb` from the **repo root** (so `nbb.edn`'s
  `:paths` put `genmlx.*` on the classpath) with `NODE_PATH` pointing at
  `../genmlx-tui/node_modules` (for ink/react/ink-spinner).
- **Live figures:** each slide's demo is a function that *returns output lines as
  data* (never `println`, which would corrupt ink's render). Pure-GenMLX figures are
  synchronous (run on a key-press behind a brief spinner); LLM figures return a
  promise and update the view when it resolves.

## Slides

| # | Feature | Live demo |
|---|---------|-----------|
| 01 | Homoiconicity — programs are data | one source: simulate, inspect (L1-M3), vsimulate `[1000]`, auto-conjugacy |
| 02 | Verified compilation ladder | 3 tiers from structure; compiled == handler, `|Δ|=0` |
| 06 | Auto-analytical from source | conjugacy detected; exact log-ML vs hand-derived MVN; IS converges |
| 07 | Shape-based vectorization | `[10000]` from one body run; ~4000× speedup vs scalar loop |
| 08 | Value-semantics through GPU | graph build ms vs eval ms; trace `:score` is an `MxArray` |
| 03 | LLMs as distributions | same `p/simulate`/`p/generate` on coin, Gaussian, LLM |
| 04 | Code as a conditioned RV | grammar masks any categorical; constrained text + a plain die |
| 05 | ClojureScript writing ClojureScript | LLM proposes a model → SCI evals → GFI scores by log-ML |
| — | The deck as a gen function | navigation as a discriminated operant; your path is a trace |

## Notes / limitations

- The interactive TTY rendering must be run in a real terminal; the figure
  computations and full module loading are verified by `DECK_SELFTEST`.
- For the strongest synthesis on slide 05, set `MODEL-NAME` in `deck.cljs` to
  `Qwen3.6-35B-A3B-4bit` (slower to load).
- "Live" numbers vary run-to-run (fresh PRNG keys); seed with `dyn/with-key` if you
  want stable figures for a recorded talk.
