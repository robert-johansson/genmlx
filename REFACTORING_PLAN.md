# GenMLX Refactoring Plan: Dispatcher + Malli Integration

*This document is the executable checklist for the ARCHITECTURE.md refactoring and the MALLI_INTEGRATION.md schema layer. Each item has explicit done criteria. Phases are sequential -- complete one before starting the next. All work on branch `refactor/dispatcher-malli`.*

---

## Pre-Flight

- [x] **Create branch.** `git checkout -b refactor/dispatcher-malli` from main.
- [x] **Verify baseline.** Run the full test suite on main and record pass counts. These are the numbers that must not change.

```bash
bun run --bun nbb test/genmlx/level0_certification_test.cljs       # 68/68
bun run --bun nbb test/genmlx/schema_test.cljs                     # 227/227
bun run --bun nbb test/genmlx/compiled_simulate_test.cljs           # 82/82
bun run --bun nbb test/genmlx/partial_compile_test.cljs             # 92/92
bun run --bun nbb test/genmlx/combinator_compile_test.cljs          # 92/92
bun run --bun nbb test/genmlx/l4_certification_test.cljs            # 41/41
bun run --bun nbb test/genmlx/gen_clj_compat_test.cljs              # 356/356
bun run --bun nbb test/genmlx/genjax_compat_test.cljs               # 73/73
```

- [x] **Verify Malli loads in nbb.** Via submodule at `malli/src` with IPrintWithWriter fix.

---

## Phase 1: Malli Schemas on Existing Code

*Goal: Define schemas that describe the current codebase. Zero behavior change. All tests pass with validation enabled.*

### 1.1 Create schemas.cljs

- [x] **Create `src/genmlx/schemas.cljs`.** Require `malli.core`, `malli.util`, `malli.error`, `genmlx.choicemap`, `genmlx.trace`.

- [x] **Define `BaseState` schema.** Keys: `:key`, `:choices`, `:score`, `:executor` (optional). Verified against real simulate init state.

- [x] **Define `HandlerState` multi-schema.** SimulateState, GenerateState, UpdateState, RegenerateState, ProjectState — each extends BaseState via `mu/merge`. Verified against real init states for each mode.

- [x] **Define `SubResult` schema.** `:weight` allows nil (combinators set it explicitly to nil in simulate mode). Verified against real sub-results.

- [x] **Define GFI return schemas.** All 7 verified against real GFI operation outputs. Custom `:error/message` on `:fn` predicates for clear error reporting.

- [x] **Define `ModelSchema` schema.** Verified against both static and dynamic models. Fixed: `:conjugate-pairs` is `[:or vector? map?]` (was `map?`).

- [x] **Define `validated` helper function.** Gated by `*validate?*` (default: false). Throws with `me/humanize` errors.

**Done criterion for 1.1:** All schemas evaluate without error. Each schema validated against real data from the REPL returns true for well-formed data and false for intentionally malformed data (missing key, wrong type).

### 1.2 Add Validation Points

- [x] **Add validation at `run-handler` entry.** In `runtime.cljs`, validates against `BaseState`. (Mode-specific validation deferred to Phase 2 dispatcher.)

- [x] **Add validation at `merge-sub-result` entry.** In `handler.cljs`, validates against `SubResult`.

- [x] **Add validation at GFI method exits.** In `dynamic.cljs`, all 7 GFI methods validate return values.

### 1.3 Verify Schemas Match Existing Code

- [x] **Run full test suite with `*validate?*` = true.** 1,031/1,031 assertions pass.

- [x] **Run all memo examples with validation enabled.** 26/26 pass.

- [x] **Run full test suite with `*validate?*` = false.** 1,031/1,031 — matches baseline exactly.

**Phase 1 complete.** Committed at `4d9b306`.

---

## Phase 2: The Dispatch Refactoring

*Goal: Replace the cond ladders in DynamicGF with a dispatcher stack. All tests pass. Malli catches any structural regressions.*

### 2.1 Create dispatch.cljs

- [x] **Define `IDispatcher` protocol.** `resolve-transition [this op schema opts]` in `dispatch.cljs`.
- [x] **TransitionSpec and ScoreType schemas** already defined in Phase 1 `schemas.cljs`.
- [x] **Implement `HandlerDispatcher`.** L0 fallback, always succeeds.
- [x] **Implement `CompiledDispatcher`.** L1 full + L1-M3 prefix paths.
- [x] **Implement `AnalyticalDispatcher`.** L3 generate/assess/regenerate.
- [x] **Implement `CustomTransitionDispatcher`.** Checks `::custom-transition` metadata.
- [x] **Implement `with-handler`.** Metadata-based handler substitution.
- [x] **Define `default-dispatcher-stack`.** `[custom analytical compiled handler]`.
- [x] **Implement `resolve`.** Stack walk, first non-nil wins.
- [x] **Implement `run-dispatched`.** Resolve + execute via `:run` function in dispatch-spec.

### 2.2 Modify DynamicGF

- [x] **Refactor `simulate`.** 68/68 L0, 82/82 compiled simulate.
- [x] **Refactor `generate`.** 356/356 Gen.clj, 73/73 GenJAX.
- [x] **Refactor `update`.** Fixed closure bug: `:run` fns used `(:trace opts)` from outer scope instead of destructured `trace`.
- [x] **Refactor `regenerate`.**
- [x] **Refactor `assess`.**
- [x] **Refactor `project`.**
- [x] **Full test suite:** 1,031/1,031. **26/26 memo examples.**

### 2.3 Clean Up

- [ ] **Remove dead helper functions.** Deferred — helpers are still called from dispatcher closures, not dead yet. Will inline in a future cleanup pass.

**Phase 2 complete.** Committed at `f8086b4`.

---

## Phase 3: ExactGF Migration

*Goal: Prove that `with-handler` replaces the ExactGF record. The 26 memo examples are the primary verification.*

### 3.1 Create `enumerate` Function

- [x] **Define `enumerate` in `inference/exact.cljs`.** Uses `dispatch/with-handler` + `dyn/auto-key` + `enumerate-run`.
- [x] **Verify in REPL.** All 6 GFI ops verified. Weights match ExactGF baseline exactly.

### 3.2 Update Memo Examples

- [x] **No memo examples used `exact/Exact` directly** — they use `exact/thinks` (updated) or `exact/Exact` inside `run-enumerate` (unchanged).
- [x] **chimp.cljs** `instance? ExactGF` check replaced with metadata check.
- [x] **26/26 memo examples pass.**

### 3.3 Remove ExactGF Record

- [x] **ExactGF defrecord deleted.** Replaced by `enumerate-run` + `enumerate`.
- [x] **`exact/Exact` kept** (lightweight metadata annotation for splice inside enumerate mode).
- [x] **`exact/thinks` updated** to use `enumerate`.
- [x] **`enumerate-executor` updated** to detect `::dispatch/custom-transition` metadata.
- [x] **All post-processing algebra, high-level API, utilities kept.**
- [x] **1,031/1,031 tests pass.** exact_test.cljs: 119/120 (test 42 fails identically on main).

**Phase 3 complete.** Committed at `07c56ba`.

---

## Phase 4: Additional Malli Schemas

*Goal: Add schemas for structures that are now stable after the refactoring.*

- [ ] **Define `TraceSite` schema.** Validate against real schema extraction output.

- [ ] **Define `ConjugacyEntry` schema.** Multi-dispatch on `:family`. Validate against the conjugacy table in `conjugacy.cljs`.

- [ ] **Define `DistributionParams` schema.** Multi-dispatch on `:type`. Validate against real distribution instances.

- [ ] **Add validation to distribution constructors (gated).** The `defdist`-generated constructors validate params against the schema during development.

- [ ] **Run full test suite with all validation enabled.** All tests pass.

**Done criterion for Phase 4:** All schemas from MALLI_INTEGRATION.md are defined and verified against real data.

---

## Phase 5: Domain Extension Proof

*Goal: Prove that `mu/merge` schema composition works for domain-specific state extensions. This is preparation for the LLM integration, not the implementation itself.*

- [ ] **Define `LLMGenerateState` as `mu/merge` of `GenerateState` with `[:map [:kv-cache some?] [:context vector?]]`.** Verify it validates maps with both base and LLM-specific keys.

- [ ] **Define `BeamGenerateState` as `mu/merge` of `GenerateState` with `[:map [:beam-states vector?] [:beam-logZ number?]]`.** Same verification.

- [ ] **Verify that base validation still applies in merged schemas.** A map missing `:choices` must fail validation even if it has `:kv-cache`.

- [ ] **Delete these proof-of-concept schemas** (or move to a test file). They are validation of the mechanism, not production schemas. The real schemas will be defined when the LLM integration is built.

**Done criterion for Phase 5:** `mu/merge` composition works correctly. Base keys are validated in merged schemas. Domain-specific keys are validated. The mechanism is proven.

---

## Post-Flight

- [ ] **Run the complete test suite one final time.** Record all pass counts. Compare against pre-flight baseline. Numbers must match exactly.

- [ ] **Run all memo examples one final time.** All 26 correct.

- [ ] **Review the diff.** New files: `dispatch.cljs`, `schemas.cljs`. Modified files: `dynamic.cljs` (simplified), `runtime.cljs` (validation point), `handler.cljs` (validation point), `inference/exact.cljs` (ExactGF deleted, `enumerate` added). No other files changed.

- [ ] **Verify net line count.** `dispatch.cljs` + `schemas.cljs` add lines. `dynamic.cljs` loses lines (cond ladders removed). `exact.cljs` loses lines (ExactGF record deleted). The refactoring should be roughly line-neutral or slightly negative.

- [ ] **Merge to main** (after review).

---

## Quality Gates

Each phase has an explicit quality gate. Do not proceed to the next phase until the gate passes.

| Phase | Gate | Mechanism |
|---|---|---|
| 1 | Schemas describe existing code | Full test suite passes with `*validate?*` = true |
| 2 | Dispatch refactoring is correct | Full test suite + memo examples pass in both validation modes |
| 3 | with-handler replaces ExactGF | All 26 memo examples produce correct results |
| 4 | All schemas defined and verified | Full test suite passes with all validation enabled |
| 5 | mu/merge composition works | REPL verification of merged schemas |
| Post | No regressions | Pass counts match pre-flight baseline exactly |

---

## What Is NOT In This Plan

- **LLM integration.** That is a separate project that builds on this foundation.
- **Grammar engine (instaparse).** That is a separate project.
- **AWRS sampler.** That is a separate inference algorithm.
- **Any changes to handler.cljs transitions.** The transitions are pure functions that stay unchanged.
- **Any changes to protocols.cljs.** The GFI protocols are the composition boundary and stay unchanged.
- **Any changes to inference algorithms.** They're written against the GFI and don't need changes.
- **Any changes to combinators.** They implement the GFI and compose through it.

This plan changes the dispatch mechanism inside DynamicGF, adds structural validation via Malli, and proves the `with-handler` mechanism via the ExactGF migration. Everything else stays unchanged.
