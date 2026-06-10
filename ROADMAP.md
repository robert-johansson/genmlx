# GenMLX Roadmap to v1.0

*Drafted 2026-06-09 from the full-codebase audit (bean `genmlx-ng02`, findings epic `genmlx-hqo2`), the pillar vision beans (`genmlx-youg`, `genmlx-y39l`, `genmlx-gsoi`, `genmlx-7qbr`, `genmlx-f6ov`, `genmlx-bod4`), and the TOPML milestone tree (`genmlx-wdop`). Beans are the source of truth for task state; this document is the map, not the tracker. Sizes: **S** = days, **M** = 1–2 weeks, **L** = multi-week.*

---

## What v1.0 means

GenMLX v1.0 is the release where the four verticals are complete enough to commit to, and the engine underneath them is *verified*, not just tested:

1. **Engine** — the full GFI contract (thesis-complete), with every semantic convention promoted to a checked algebraic law, and compiled/analytical paths provably matching the handler.
2. **genmlx.llm** — LLM-as-GF validated across all GFI operations over token traces, with a forward path GenMLX owns (no per-model upstream structs) and a golden-output oracle.
3. **genmlx.agents** — a stable env/agent/solver API across MDP/POMDP/game/multi-agent, inverse planning and biases included, proven against at least one *external* environment over the Bun network face.
4. **genmlx.control** — a working metareasoner: steppable/budgeted inference, a cost meter, a controller-as-agent-GF, and one seed-validated anytime benchmark.
5. **Two honest membranes** — mlx-node (compute effects) fully surface-audited; Bun (world effects) with at least the persistence and network faces real.
6. **A public API freeze** — documented, semver-committed, with docs that match the code.

The architectural thesis stays fixed throughout: *the handler is ground truth; compilation is optimization; an agent is a GF; inference is pluggable; effects live only at the two membranes.*

---

## Where we are (2026-06-09)

- **Engine**: L0–L4 compilation ladder built and certified; 64-law suite; 33 distributions; 10 combinators; 35+ inference algorithms. The 2026-06-09 audit found a cluster of semantic-convention bugs (selection algebra, SMC evidence, update/regenerate weight conventions, L3 detection false positives, compiled-path edge parity) — all line-cited in epic `genmlx-hqo2`, none yet verified/fixed.
- **llm**: LLM-as-GF works end to end (token trace sites, grammar/byte/reader constraints, codegen, MSA). Forward path still borrows mlx-node model structs; no numeric oracle for the forward pass; full-GFI-over-tokens validation pending (`genmlx-fayo`).
- **agents**: promoted to `src/genmlx/agents/` with MDP/POMDP constructors, gridworld/bandit/POMDP environments, biased planners, inverse planning, TUI gallery. Several agentmodels chapters remain (`genmlx-wsbk`, `genmlx-8yya`, `genmlx-5l29`); no external-environment agent yet.
- **control**: does not exist in `src/` — by design. Specs exist for the steppable substrate (`genmlx-rfal`), cost meter (`genmlx-i0s4`), and metareasoner (`genmlx-nrkq`).
- **Membranes**: mlx.cljs is thin and mostly honest; the full mlx-node surface has never been enumerated against it (`genmlx-0vwn`). The Bun world membrane (`genmlx-gsoi`) is a design, not code.

---

## Phase 0 — Correctness floor *(do first; gates everything)*

The audit's headline findings are exactly the kind of silent-wrong-results class that the project's rigor floor (`genmlx-p3uy`, `genmlx-9ocx`) exists to prevent. Every item below carries a verify-first checkbox in its bean (adversarial verification was descoped from the audit; the independent-oracle rule applies).

| Order | Bean | Item | Size |
|---|---|---|---|
| 1 | `genmlx-yey5` | Selection algebra: `SelectAddrs.get-subselection` returns `all` unconditionally; Complement mirror; flat selections at splices/combinators | M |
| 2 | `genmlx-vdpx` | SMC log-ML double-counts on skipped resamples; cSMC reference-particle weight scale; SMCP3 strip/observations | M |
| 3 | `genmlx-ovop` `genmlx-8l8j` | Regenerate-through-splice weight algebra; scalar-vs-batched update weight conventions | M |
| 4 | `genmlx-b470` | L3 false positives: bilinear block eligibility, non-NN affine coefficient drop, dirichlet-categorical eliminated with no handler, constrained-prior discard | M–L |
| 5 | `genmlx-b210` | Compiled parity: fused iid-gaussian-as-Delta (zero score), boundary log-prob guards, M4 zero-truthiness, binding-env shadowing | M |
| 6 | `genmlx-q3x2` | Schema walker map/set-literal blindness (sites vanish, model stays static) | S–M |
| 7 | `genmlx-7ca0` | `mh` default `sel/all` resamples observations; tensor-score layout mismatch | S |
| 8 | `genmlx-pkmx` | `strip-compiled` leaves prefix/analytical keys — law-suite soundness | S |
| 9 | `genmlx-2j40` | Honesty sweep: fake warm-ups, fabricated `:final-ess` feeding paper artifacts | S–M |
| 10 | `genmlx-njaq` `genmlx-v740` `genmlx-yeam` `genmlx-7sqe` `genmlx-000i` `genmlx-xwxh` `genmlx-ybw9` `genmlx-vd2j` `genmlx-xpbm` | Remaining audit beans (PRNG hygiene, combinators, dist boundaries, VI honesty, serialize, llm layer, core-data, membrane, agents) | M–L total |

Also in this phase: SBC restoration and expansion (`genmlx-18q9`, `genmlx-zi2f`, `genmlx-66er`), the batched test-runner mode (`genmlx-q69j`), and the open evidence-experiment items (`genmlx-9ocx`).

**Exit criterion:** every fixed path has an independent-oracle regression test, and the law suite actually forces the handler path when it claims to.

---

## Phase 1 — Engine contract complete

Close the gap between the implemented GFI and the thesis, and turn the conventions the audit caught drifting into checked contracts.

- **Update with new arguments** — `IUpdate` lacks the thesis's `x'` parameter; SMC with changing arguments needs it. Design-clear, touches protocols + handler + combinators. (M)
- **Regenerate through structure change** — handler currently throws when an unselected address is absent; Gen.jl samples fresh. Needed for branch-flipping MH. (M)
- **Score-type as enforced metadata** — `:joint`/`:marginal`/`:collapsed` checked at update/project boundaries instead of implicit convention (ARCHITECTURE §3.3's own admission). (S–M)
- **Selection redesign** — coherent descent semantics with property tests (`genmlx-ota8`), building on the Phase 0 fix. (S)
- **Trace translators** (`genmlx-oen5`, thesis §3.6–3.7) and **encapsulated randomness** (`genmlx-qbaa`, §4.5) — the two missing thesis mechanisms. (M each)
- **Analytical update mode for linear-Gaussian blocks** (`genmlx-6hcu`) and the dirichlet-categorical family done right (`genmlx-cf0d`). (M)
- **Doc truth pass** — ARCHITECTURE.md vs code (`genmlx-piqh`), law/dist count claims (`genmlx-8a78`, `genmlx-i1zt`, `genmlx-hpml`), README verticals section (`genmlx-44vq`). (S)

**Exit criterion:** a reader of ARCHITECTURE.md finds zero divergences from the code; every GFI operation has laws pinning its weight semantics on scalar, batched, compiled, and analytical paths.

---

## Phase 2 — genmlx.llm v1.0

Two tracks: correctness of the GF semantics over tokens, and ownership of the forward path.

1. **Golden-output oracle first** (gate inside `genmlx-f6ov`): pin forward-pass logits/argmax against Python mlx-lm for ≥1 prompt per supported model *while the current forward still works*. Without this, any forward rewrite can be silently wrong. (S)
2. **f6ov P0–P6** — the approved decoupling spec: fused-op primitives in genmlx.rs → safetensors weights → CLJS Qwen3 forward → GenMLX-owned KV cache → backend migration → parity gate. Removes the per-model upstream struct dependency and the thread-local-stream hazard class (`genmlx-7siy`). (L)
3. **Full 7-op GFI over tokens** (`genmlx-fayo`): fix grammar middleware running generate-semantics under update/regenerate (`genmlx-xwxh`), token-MCMC TV-to-exact validation + KV recompute hardening (`genmlx-3ob2`). (M)
4. **Replay optimization** (`genmlx-almu`): external stream → teacher-forcing rescore → handler replay; 1.5–2× on unconstrained simulate, and the general pattern for vision/database streams. (M)
5. **MSA consolidation**: knowledge mode as the keeper, retire the template-mode model dependency (`genmlx-n4ds`), strip thinking tokens (`genmlx-wumc`). (S)
6. **Flagship composition artifact**: LLM as a scored sub-component inside a structured gen fn (`genmlx-t3z5`); grammar/combinator program-synthesis worked example (`genmlx-ct1r`). (S–M)

**Research flags (llm):**
- *Byte-level score semantics* — byte-GF weights are per-step renormalized greedy-tokenization conditionals, not model evidence; define and document the contract before anyone uses them as evidence (`genmlx-h2ki` adjacent).
- *Retokenization consistency* — `generate-and-score` re-encodes text that need not match sampled tokens; decide whether to thread token ids through or accept and document the approximation.
- *MoE forward* (`genmlx-k199`, `genmlx-o94r`) — only if 30B-class models become a v1.0 requirement; otherwise post-1.0.

---

## Phase 3 — genmlx.agents v1.0

The pillar closest to done in spirit; the work is completion, hardening, and one genuinely new capability (external environments).

1. **Finish the chapter ports** (`genmlx-5l29`, `genmlx-wsbk`, `genmlx-8yya`, `genmlx-6t33`, `genmlx-xte5`) and author Ch 6 natively as the pluggable-inference chapter (`genmlx-b5hb`). (L, parallelizable)
2. **Fix the audit items** (`genmlx-xpbm`) and **define the observation-model contract**: the three belief filters disagree on nil-observation semantics — that is a design decision (when is absence informative?), not a bug fix. Unify, then make the tensor/host equivalence claims true. (M)
3. **API freeze**: env/agent/solver/belief protocols, with tests as contracts (the `genmlx-youg` "promotion" criterion). (M)
4. **agents × llm flagship** (`genmlx-youg`): an agent whose policy is an LLM, and an LLM reasoning over an agent model — both GFs, nested both ways. (M)
5. **First external environment** — the agent boundary as real I/O over the Bun network face (`genmlx-gsoi`): sensor = inbound constraint via generate, actuator = outbound sampled choice committed to the world. Candidate: a game server or remote simulator. This is where *interacting with external APIs and reasoning over them* lands — an API is an environment; model it (`p(return | query, state)` as a pure GF), never GF the service itself. Depends on Phase 4's network face. (M–L)

**Research flags (agents):**
- *Streaming/lazy joint-inference enumeration* (`genmlx-fpya`) — memory growth in exact agent inference.
- *Memoization policy* (`genmlx-twza`) — with-cache key-space boundedness under noisy/graded observation models (the audit flagged the silent blow-up case).
- *Differentiable belief-space planning* (`genmlx-5x3f`) — needs the safe-where rewrite first; then gradient correctness through near-zero-mass filter steps.

---

## Phase 4 — Membranes (runs alongside Phase 3)

**mlx-node (compute membrane):**
- Full surface audit and coverage matrix (`genmlx-0vwn`): every NAPI export classified wired/on-demand/intentionally-omitted; the matrix doubles as the upstream-drift guard. Confirmed already: LM training primitives (params/grads through the transformer forward) exist and are reachable. (M)
- MxArray ClojureScript protocols (`genmlx-wxvk`); install hardening (`genmlx-91b3`); engine loads without the native backend for pure graph/CI work (`genmlx-eulz`); membrane honesty fixes from `genmlx-vd2j`. (S–M each)

**Bun (world membrane, `genmlx-gsoi`):** same move as mlx.cljs, second substrate — thin, honest, one effect class per face.
- *Persistence face* — **genmlx.memory v0** (`genmlx-7qbr`): serialize ↔ bun:sqlite, save/restore traces, particle sets, parameter stores; synchronous checkpoint at session boundaries. Requires the serialize fixes (`genmlx-000i`) first. (M)
- *Network/IO face* — Bun.fetch/serve/WebSocket/spawn for the agents external environment (Phase 3.5). (M)
- *Process/worker face* — Worker/spawn/nanoseconds for the control scheduler (Phase 5). (S, as part of control)

**Design needed (not research):** cross-session addressing/identity — stable keys or content-addressing so a value saved in session 3 is retrievable in session 50; the choicemap-address discipline extended to durable keys. Decide before memory v0 ships.

---

## Phase 5 — genmlx.control v1.0 *(last; consumes everything below it)*

control = agents pointed at computation. One-way dependency: core → {llm, agents} → control. The scheduler is control's `eval!` — the sole side effect; never a GF.

Build order (all beans exist and are floored under `genmlx-tyqr`):
1. **Steppable substrate** (`genmlx-rfal`): step/peek/done? wrapper over the already-step-structured SMCP3; posterior/ESS/log-ML + budget as a resumable value. (S–M — but depends on Phase 0's SMC evidence fixes, or the controller optimizes a biased signal.)
2. **Cost meter** (`genmlx-i0s4`): inference effort as a first-class value. (S)
3. **Controller-as-agent-GF** (`genmlx-nrkq`): meta-MDP over {add particles, refine, switch, stop}, reusing make-mdp-agent + softmax-action; v0 allocator generalizes method_selection. (M)
4. **Seed-validated anytime microbenchmark** (`genmlx-gdtq`, plus `genmlx-6l31`). (S–M)

**Research register (control — the genuinely open questions):**

| Topic | Question | Anchors |
|---|---|---|
| Value of computation | Myopic VOC vs learned estimators; when does myopic VOC mislead the meta-MDP? | Russell & Wefald 1991; Gershman/Horvitz/Tenenbaum |
| Bounded regress | Where and how to truncate metareasoning-about-metareasoning; cost model for the metareasoner itself (dep-graph blast radius as cost meter) | `genmlx-y39l`, `genmlx-bod4` |
| Performance profiles | Modeling p(quality \| t) as environment GFs the metareasoner filters over | Dean & Boddy; Zilberstein |
| Substrate generality | Does the SMC-first step interface generalize to MCMC/VI without a redesign? | `genmlx-rfal` |
| AIKR open leg | Open-universe models: fresh-name distributions, ID-keyed choicemaps, factorial-corrected densities; the growing hypothesis space control alone does not deliver | Zaiser et al. 2026 (PLDI), `genmlx-bod4` |

**Pre-freeze spike (do during Phase 3, before the API freeze):** the gen.core substrate-protocol extraction sketch from `genmlx-bod4` — verify how cleanly the GFI core factors from MLX (grep evidence says: nearly free outside dist/handler/inference). v1.0 should not freeze APIs that would block the incremental-substrate (fGen) factoring later; factoring itself is post-1.0.

---

## Sequencing

```
Phase 0 (correctness floor)
   └─► Phase 1 (engine contract)
          ├─► Phase 2 (llm)        ─┐
          ├─► Phase 3 (agents) ◄────┼── gen.core spike before API freeze
          │      └─ external env ◄──┤
          ├─► Phase 4 (membranes) ──┘   (network face feeds agents; persistence face independent)
          └────────► Phase 5 (control)  (needs steppable + cost + agents API + Phase 0 SMC fixes)
```

The TOPML paper (`genmlx-wdop`, `genmlx-9ocx`) rides Phases 0–2: the rigor floor *is* Phase 0, the crown jewel (synthesis → exact evidence) hardens in Phase 1, the agents-axis and reach evidence come from Phases 3 and 5's first artifacts.

## What the v1.0 freeze commits to

- The 10 GFI protocols + edit interface (signatures frozen; extensions additive).
- choicemap/trace/selection data algebra and address semantics, including durable-key discipline.
- `defdist`, `with-handler`/`with-dispatch`, and the dispatcher-stack extension points.
- The agents env/agent/solver/belief protocol surface.
- The two-membrane effect contract: `eval!` for compute, the Bun faces for world; everything above is pure values.
- Docs that match code (enforced by the Phase 1 truth pass), and the tiered test runner as the release gate.

## Risks

1. **Verification debt** — every audit finding ships with a verify-first checkbox; the floor phase is only done when fixes have *independent* oracles (the ke9i lesson).
2. **Upstream drift tax** — mlx-node rebases; mitigated by f6ov (own the forward) and the 0vwn coverage matrix (drift becomes a visible diff).
3. **Convention pluralism** — the audit's root architectural finding: semantics enforced by convention where two code paths exist. The mitigation is structural: no new execution path lands without a law pinning it to the handler.
4. **Control research overrun** — Phase 5 has real unknowns; the floored scope (`genmlx-tyqr`) is deliberately small. If VOC research stalls, v1.0 ships the floored controller and the research register moves to v1.x.
5. **Metal test-environment fragility** — sustained-GPU wedges; the batched runner (`genmlx-q69j`) and zero-GPU gate stay first-class.
