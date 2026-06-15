# The LLM forward seam (f6ov)

GenMLX runs LLMs as generative functions. The only thing it needs from a model is
a forward pass: `(token-context) -> logits[vocab]`. Historically GenMLX got that
by calling four methods that live **inside** mlx-node's per-model structs
(`Qwen3Model` / `Qwen35Model`): `.forward`, `.forwardWithCache`, `.initCaches`,
`.resetCaches`. Those are GenMLX additions to mlx-node, not part of its sanctioned
API — and upstream reorganizes the model files almost every release, so each
resync forced us to re-derive them by hand (the worst case, the 2026-05-31 rebase
`genmlx-dbce`, mis-aligned 700+-line conflict blocks). That recurring cost is the
"forward-seam tax."

**f6ov removes the tax by owning the forward.** GenMLX now computes the Qwen3 and
Qwen3.5 (GatedDeltaNet hybrid) forward passes in pure ClojureScript over stable
primitives, and `backend/load-model` makes that owned forward the **smart default**
for the families it implements (`forward/supported?`). The borrowed forward stays
reachable only as an explicit one-release fallback (`{:cljs-forward? false}`).

## The owned-path surface (what a resync can and cannot touch)

The owned forward (`genmlx.llm.forward`, `genmlx.llm.qwen3-forward`,
`genmlx.llm.qwen35-forward`) depends on exactly this surface:

| Dependency | Provided by | Rebase-stable? |
|---|---|---|
| `mx/*` array + nn ops: `rms-norm`, `rope`, `scaled-dot-product-attention`, `silu`, `sigmoid`, `matmul`, `take-idx`, `reshape`, `transpose`, `concatenate`, `slice`, `index`, `repeat-arr`, `arange`, `sum`, `exp`, `log1p`, `add`/`subtract`/`multiply`, `astype`, `ones`/`zeros`, `squeeze`, `dtype` | `genmlx.mlx` → `genmlx.rs` / `transforms.rs` / MLX `fast::`/core | **Yes** — upstream never edits `genmlx.rs`/`transforms.rs` |
| `mx/load-safetensors` (weights as `{name -> MxArray}`) | `genmlx.rs` (`loadSafetensors`) | **Yes** |
| `config.json` read | Node `fs` + `js/JSON.parse` | n/a (not mlx-node) |
| Tokenizer (`Qwen3Tokenizer.fromPretrained`), `detectModelType` | mlx-lm, **sanctioned/stable** API | Yes |

The owned forward calls **none** of the four borrowed model-struct methods.

## Why a simulated upstream resync touches only `genmlx.rs`/`transforms.rs`

Hypothetically delete `.forward`/`.forwardWithCache`/`.initCaches`/`.resetCaches`
from mlx-node's model files (the kind of change a resync makes). The owned default
is unaffected, because:

1. **The owned forward modules never call them** — verified statically by
   `test/genmlx/llm_forward_seam_test.cljs`
   (`owned-forward-modules-have-no-borrowed-calls`).
2. **`backend.cljs` reaches them only inside `cljs-forward-model?`-gated code** —
   the `{:cljs-forward? false}` fallback — verified by the same test
   (`backend-gates-every-borrowed-call-behind-the-fallback`; the one private leaf
   `forward-with-cache` is itself only invoked from the gated
   `forward-prefill`/`forward-step`).

So the conflict surface of a resync is confined to `genmlx.rs` / `transforms.rs`
(which upstream never touches) plus the MLX submodule pin — not the three
per-model `model.rs` files. The guard test makes this a **CI invariant**: if a
future edit re-introduces a borrowed-forward dependency into the owned path, it
fails.

## Remaining (deliberate) coupling

- The borrowed forward stays as the explicit fallback for one release, and as the
  automatic route for families/weight-layouts the owned forward does not yet
  implement (e.g. MoE — `genmlx-k199`; HF sharded / `index.json` checkpoints —
  `genmlx-o94r`, gated by `forward/loadable-weights?`).
- f6ov decouples from mlx-node's **model structs**, not from MLX itself: the owned
  path still requires `genmlx.rs` to be built. The install-experience hardening is
  a separate concern (`genmlx-91b3`).
