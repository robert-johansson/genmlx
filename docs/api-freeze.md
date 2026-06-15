# GenMLX v1.0 API Freeze

This document is the concrete realization of ROADMAP.md's *"What the v1.0 freeze
commits to"*. It is the **semver contract**: the surfaces below are frozen for the
1.x line — signatures do not change incompatibly, and extensions are **additive**.
Each commitment names its concrete surface (the actual protocols / functions /
files) and the test(s) and/or `gfi.cljs` laws that **pin** it — so a change that
breaks the contract turns a specific test red.

The freeze covers the *public* contract only. Internals (handler transition
internals, compiled-path implementations, membrane resource heuristics) may change
freely as long as the pinned public behavior holds. Surfaces explicitly fenced as
**provisional** (below) are outside the freeze.

> Counts in this document (protocols, laws, distributions, exports) are stated as
> "count the live definition" wherever a number would drift; where a number is
> given it is accurate as of the v1.0 freeze and re-derivable from the cited file.

---

## 1. The GFI protocols + edit interface

**Frozen surface.** `src/genmlx/protocols.cljs` defines **11 single-operation
protocols** — the 10 core GFI operations plus the vectorized-splice protocol:

| Protocol | Operation |
|---|---|
| `IGenerativeFunction` | `simulate` |
| `IGenerate` | `generate` |
| `IAssess` | `assess` |
| `IUpdate` | `update` |
| `IRegenerate` | `regenerate` |
| `IPropose` | `propose` |
| `IProject` | `project` |
| `IUpdateWithDiffs` | `update-with-diffs` |
| `IUpdateWithArgs` | `update-with-args` |
| `IHasArgumentGrads` | `has-argument-grads` |
| `IBatchedSplice` | `batched-splice` (vectorized inference) |

The **edit interface** is `src/genmlx/edit.cljs`: the `IEdit` protocol
(`edit [gf trace edit-request]`), the four `EditRequest` constructors
(`constraint-edit`, `selection-edit`, `args-update-edit`, `proposal-edit`), the
`edit-dispatch` multimethod, and the backward-request round-trip contract.

Protocol *signatures* are frozen; new generative-function/distribution
implementations of these protocols are additive.

**Pinned by.** The `gfi.cljs` `laws` vector (one law per weight-semantics theorem,
each citing the thesis), executed by `gfi_laws_test.cljs` (+ the `_p1..p10` splits)
which iterates `gfi/laws` and calls `gfi/verify`; the per-op suites
`gfi_simulate_test`, `gfi_generate_test`, `gfi_assess_test`, `gfi_update_test`,
`gfi_regenerate_test`, `gfi_project_test`, `update_args_test`, `vupdate_test`, and
the contract suites `gfi_contract_test` / `gfi_contracts_test` / `gfi_universal_test`.
The edit interface is pinned by `edit_property_test`, `edit_purity_test`,
`proposal_edit_test`, and the `:edit-backward-request-roundtrip` law.

---

## 2. The choicemap / trace / selection data algebra

**Frozen surface.** `choicemap.cljs`, `trace.cljs`, `selection.cljs` (and
`diff.cljs` for argdiffs): hierarchical keyword/vector addresses, leaf-value
identity, and the `get-value` / `get-submap` / `merge` / `filter` algebra. The
**durable-key discipline** (cross-session addressing) is realized in `memory.cljs`:
content-hash identity (`put-content!`, sha256 over a canonical serialization) and
named-key (program-name) upsert — explicitly mirroring the choicemap-address
discipline (content = leaf-value identity; named = the address slot).

**Pinned by.** `choicemap_algebra_test`(`2`), `choicemap_property_test`,
`choicemap_test`; `selection_algebra_test2`, `selection_property_test`,
`selection_test`, `selection_regenerate_test`; `trace_test`,
`trace_immutability_test`(`_property`), `tensor_trace_test`(`_property`); the
`:address-uniqueness` and `:simulate-address-set-consistency` laws. Durable keys:
`world_memory_test` / `memory_test` — the content hash is pinned to a golden hex
computed by an **independent** shell `sha256` over the hand-written canonical
string, and session checkpointing is proven by an actual close+reopen
(across-process-death).

---

## 3. `defdist`, `with-handler` / `with-dispatch`, and the dispatcher stack

**Frozen surface.** `defdist` (`dist/macros.cljc`) + the open multimethods
`dist-sample*` / `dist-log-prob` (`dist/core.cljs`) — the extension point for new
distributions. `with-handler` / `with-dispatch` (`dispatch.cljs`) — the
execution-strategy extension points. The **4-level dispatcher stack**
(custom → analytical → compiled → handler, `dynamic.cljs`), extended via
`::custom-dispatch` / `::custom-transition` metadata.

**Pinned by.** `defdist`: every distribution test transitively
(`dist_test`, `new_dist_test`, `dist_logprob_test`, `dist_property_test`,
`dist_normalization_test`, …) plus custom-distribution definitions in
`gfi_gradient_test` and `kalman_test`. `with-dispatch`: `gfi_laws_test_p7`
("Alg-16 proposal override via with-dispatch" — pins importance-sampling
correctness through the custom-dispatch path). The analytical/compiled dispatcher
levels: `exact_test`, `l3_5_gate_test`, `auto_analytical_test`, and the
`compiled_*` equivalence/parity suites.

> **Known gap (honest).** `with-handler` is pinned only *indirectly* — through
> `wrap-grammar` (`llm/grammar.cljs`, exercised by `llm_grammar_test` /
> `grammar_per_op_test`) and the analytical wrap — no test names
> `dispatch/with-handler` directly. (CLAUDE.md's "canonical example" for
> `with-handler` is `exact.cljs`, which actually uses `with-dispatch`; the real
> `with-handler` example is `grammar.cljs`.) The contract is behaviorally pinned;
> a direct unit test would harden it. Tracked, not a blocker.

---

## 4. The agents env / agent / solver / belief surface

**Frozen surface.** Frozen as **tests-as-contracts** (per-constructor return-map
shapes + family-specific `:act`/belief signatures), documented in
`src/genmlx/agents/CONTRACTS.md` — **not** formal `defprotocol`s. The minimal
guaranteed contract is `{:act :params}`; everything else is family-specific by
design: `make-mdp-agent` (`{:mdp :Q :V :policy :act :expected-utility :params}`),
`make-biased-mdp-agent` (no `:Q`/`:V`), `make-pomdp-agent` (belief keys),
`make-bandit-agent`. Belief filters in `agents/belief.cljs`; envs in
`agents/gridworld.cljs`, `pomdp_env.cljs`, `worlds.cljs`.

**Pinned by.** `agents_contracts_test` (per-constructor return-map shapes + family
`:act` signatures), `agents_api_test` (namespace reachability, agent-as-GF / GFI
ops, tensor-VI `Q == recursive-EU`, inverse-planning host==batched),
`belief_tensor_test` (belief filter math, host==tensor equivalence). All three are
named in `CONTRACTS.md`'s "Enforcing tests".

> **Provisional fence.** `genmlx.agents.remote` is explicitly **outside** the
> frozen surface (`CONTRACTS.md`) — pinned only by behavioral self-checks
> (`external_env_test`, `examples/external_env.cljs`), not a contracts test.
> Consumers must not treat `remote` as a v1.0 contract.

---

## 5. The two-membrane effect contract

**Frozen surface.** Everything above the two membranes is **pure values** (Layers
1–9 referentially transparent). The two effect faces:

- **Compute membrane** (`mlx.cljs`): `eval!` is the sole GPU side effect;
  `item` / `->clj` / `materialize!` / `realize` / `tidy-run` / `tidy-materialize`
  are the boundary wrappers, all dispatching through `eval!`.
- **World membrane** (Bun): `world/net.cljs` (network face — `serve!` /
  `with-server`), `world/proc.cljs` (scheduler face — `with-deadline`),
  `memory.cljs` (persistence face — `bun:sqlite`).

The mutable boundaries are the **audited, enumerated** set (CLAUDE.md: the runtime
`volatile!`, the `mlx.cljs` resource atoms, the KV cache, the `Bun.serve`
listener). The control scheduler (`world.proc/with-deadline`) is the sole
control-layer effect and is **never** a generative function (`control/CONTRACTS.md`).

**Pinned by.** Purity above the membrane: `mutation_boundary_test` (property tests
+ AST check via `verify/check-no-mutation`) and the `:no-mutation` /
`:no-external-randomness` laws. Compute-membrane honesty/drift:
`membrane_coverage_test` (the genmlx-0vwn two-directional drift guard — partitions
every `@mlx-node/core` function export into wrapped ∪ intentional-omission and
fails on any unaccounted upstream change), `membrane_honesty_test`,
`membrane_guard_test`, `membrane_churn_test`, `clip_contract_test`,
`det_contract_test`, `native_guard_test`. World membrane: `world_proc_test`
(deadline logic with an injected fake clock; `with-server`/`with-deadline`
teardown), `world_memory_test` / `memory_test`.

> **Note (honest).** "`eval!` is the sole side effect" is enforced *structurally*
> (the coverage partition leaves no unwrapped NAPI compute export; the mutation
> property tests find no hidden mutation) rather than by a single literal
> assertion. The structural guarantee is the stronger one.

---

## 6. Docs that match code + the tiered test runner as the release gate

**Frozen surface.** The doc-truth contract is enforced where it can be mechanically
pinned: `docs/membrane-coverage.md` (the coverage matrix, pinned to
`packages/core/index.d.cts`), `agents/CONTRACTS.md` and `control/CONTRACTS.md`
(every claim test-backed). The **tiered test runner** (`test/run.sh`, tiers
core/fast/medium/slow/bench/all/check, wired as `package.json` `test:*` scripts)
is the release gate; `test:check` is the zero-GPU gate. Test files carry `@tier`
markers.

**Pinned by.** `membrane_coverage_test` reads `packages/core/index.d.cts` directly
and fails on any upstream add/delete/rename not reflected in the membrane — the
live doc-vs-code pin for the compute surface. The CONTRACTS.md claims are pinned by
the agents/control contract suites. `membrane_honesty_test` pins the membrane
honesty fixes.

> **Known gap (honest).** Prose docs (CLAUDE.md / ROADMAP.md / ARCHITECTURE.md /
> README.md) are **not** mechanically lint-pinned; they are human-maintained and
> kept in sync by the Phase-1 truth pass (an adversarially-verified
> ARCHITECTURE.md↔code sweep). The membrane surface and the agents/control
> contracts *are* mechanically pinned. The tiered runner is operational, not
> test-pinned (no test asserts the runner config itself).

---

## What is **not** frozen (provisional / internal)

- `genmlx.agents.remote` — provisional external-environment bridge (§4).
- `switch-method` live SMC↔MCMC translation in the metareasoner — deferred
  (`control/meta_mdp.cljs` raises `:not-implemented`); the v1.0 control action set
  is `{:continue :stop}`.
- Compiled-path internals, handler transition internals, membrane resource-cleanup
  heuristics — free to change behind the pinned public behavior.
- The `gen.core` substrate-protocol / incremental-substrate (fGen) factoring — a
  post-1.0 direction; v1.0 deliberately does not freeze APIs that would block it.

---

## Extending without breaking the freeze

- **New distribution** → `defdist` + `dist-sample*` / `dist-log-prob`. Additive.
- **New execution strategy** → a handler transition `(fn [state addr dist])` via
  `with-handler`, or full op-level control via `with-dispatch`. Additive.
- **New combinator / generative function** → implement the GFI protocols (and
  `IBatchedSplice` for vectorized support). Additive.
- **New agent family** → a `make-*-agent` constructor returning at least
  `{:act :params}`, documented in `agents/CONTRACTS.md` with an enforcing test.

Every new execution path must land with a law pinning it to the handler (the
handler is ground truth; everything else is optimization) — the structural
mitigation for "semantics enforced by convention" (ROADMAP Risk 3).
