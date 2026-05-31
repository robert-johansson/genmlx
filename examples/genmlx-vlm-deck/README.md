# GenMLX VLM deck — perceiving ARC-AGI-3 scenes

A full-screen, split-pane [ink](https://github.com/vadimdemedes/ink) slide deck
(under nbb) that demonstrates GenMLX's vision-language capability on real
**ARC-AGI-3** game frames:

```
render a scene  →  VLM describes it (live)  →  parse the description → {facts}
```

Each scene slide shows an ARC frame as coloured blocks on the **left**; press `r`
and the **qwen3.5-4b VLM** describes it live (right), and we parse that free-form
text into structured facts we control (grid layout, colours, player, walls/goal/border).

A final **"perception as inference"** slide closes the loop back to the GFI: it turns
a per-cell reading of a scene into a generative function (`vision/make-grid-gf`),
feeds it as constraints (`labels->constraints` → `cm/from-map`) to `p/generate`, and
shows that the perceived scene — now a scored **trace** — is far more probable than a
random grid under a structured prior. Perception becomes GFI evidence.

The VLM is handed the rendered PNG (`../genmlx-lab/dev/arc_frames/<game>_deck.png`); the terminal
grid is drawn from the same frame JSON with the same palette — so what you see is
what the model saw.

## Setup & run

```bash
# 1. ink/react (shared with the other decks), once:
(cd ../genmlx-tui && npm install)

# 2. render the scene PNGs the VLM consumes (needs Pillow), once:
python3 examples/genmlx-vlm-deck/render_frames.py      # from the repo root

# 3. run the deck (needs a real terminal):
./run.sh
```

Keys: **← / →** (or n/p/space) move · **r** (or Enter) ask the VLM · **q** quit.
The VLM loads asynchronously at startup (a `loading…/ready ✓` badge shows in the
header); a live description takes ~15s.

### Headless self-test (no TTY)

```bash
DECK_SELFTEST=1    ./run.sh   # load + ASCII-preview the 4 ARC grids
DECK_SELFTEST=full ./run.sh   # also load the VLM and describe+parse every scene
```

## Scenes

Four ARC-AGI-3 games captured in `../genmlx-lab/dev/arc_frames/` (`*.json` = 31 steps of 64×64
grids; `_deck.png` = the rendered frame fed to the VLM). The deck shows each game's
most colourful frame: `sk48` (frame 0), `g50t` (18), `re86` (1), `bp35` (1).

## How it works

- **Stack:** reagent (bundled in nbb) for hiccup; ink for the terminal UI + the
  `useInput`/`useApp` hooks; `genmlx.llm.vision/load-vlm` for the VLM session.
- **Grid render:** the 64×64 frame is downsampled 2× and drawn with upper-half-block
  glyphs (`▀`, fg = top cell, bg = bottom cell), run-length-compressed per row.
- **VLM call:** `(.send session prompt #js {:images #js [png-bytes] :config …})` —
  the same image API `vision.cljs` exposes. Loaded once; called on `r`.
- **Parse:** `parse-desc` pulls structured facts from the prose. It uses word-boundary
  matching (so "coloured" isn't read as "red") and **negation handling** (so "no
  visible walls" reads as absent, not present) — the point of the deck is that the
  VLM's description is just text *we* parse however downstream code needs.

This is a perception front-end pattern on the GFI: a VLM is a generative function
that *sees*, and its output is structured evidence for everything that follows.

## Notes / limitations

- Interactive TTY rendering must run in a real terminal; the grid loading, VLM call,
  and parser are verified headlessly via `DECK_SELFTEST`.
- Descriptions vary run-to-run (the VLM samples); the parser is written to cope.
- `OS_ACTIVITY_MODE=disable` + a stdout filter suppress macOS "Context leak" noise.
- For the largest/strongest VLM, set `VLM-NAME` in `deck.cljs` to `Qwen3.6-35B-A3B-4bit`.
