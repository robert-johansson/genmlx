# GenMLX Refactoring Plan: Dispatcher + Malli Integration

*This document is the executable checklist for the ARCHITECTURE.md refactoring and the MALLI_INTEGRATION.md schema layer. Each item has explicit done criteria. Phases are sequential -- complete one before starting the next. All work on branch `refactor/dispatcher-malli`.*

---

## Pre-Flight

- [ ] **Create branch.** `git checkout -b refactor/dispatcher-malli` from main.
- [ ] **Verify baseline.** Run the full test suite on main and record pass counts. These are the numbers that must not change.

```bash
bun run --bun nbb test/genmlx/level0_certification_test.cljs       # 68/68
bun run --bun nbb test/genmlx/schema_test.cljs                     # 174/174
bun run --bun nbb test/genmlx/compiled_simulate_test.cljs           # 82/82
bun run --bun nbb test/genmlx/partial_compile_test.cljs             # 92/92
bun run --bun nbb test/genmlx/combinator_compile_test.cljs          # 90/90
bun run --bun nbb test/genmlx/l4_certification_test.cljs            # 41/41
bun run --bun nbb test/genmlx/gen_clj_compat_test.cljs              # 165/165
bun run --bun nbb test/genmlx/genjax_compat_test.cljs               # 73/73
```

- [ ] **Verify Malli loads in nbb.** Confirm `(require '[malli.core :as m])` works. Record the require incantation for reference.

---

## Phase 1: Malli Schemas on Existing Code

*Goal: Define schemas that describe the current codebase. Zero behavior change. All tests pass with validation enabled.*

### 1.1 Create schemas.cljs

- [ ] **Create `src/genmlx/schemas.cljs`.** Require `malli.core`, `malli.util`, `malli.error`, `genmlx.choicemap`, `genmlx.trace`.

- [ ] **Define `BaseState` schema.** Keys: `:key`, `:choices`, `:score`, `:executor` (optional). Verify by evaluating against a real simulate init state from the REPL.

- [ ] **Define `HandlerState` multi-schema.** Dispatch on mode (simulate/generate/update/regenerate). Each branch extends `BaseState` via `mu/merge`. Verify by evaluating against real init states for each mode from the REPL.

- [ ] **Define `SubResult` schema.** Keys: `:choices` (required), `:score` (required), `:retval` (optional), `:weight` (optional), `:discard` (optional). Verify by capturing a real sub-result from `execute-sub` in the REPL.

- [ ] **Define GFI return schemas.** `SimulateReturn`, `GenerateReturn`, `UpdateReturn`, `RegenerateReturn`, `AssessReturn`, `ProposeReturn`, `ProjectReturn`. Verify each by running the corresponding GFI operation on a simple model in the REPL and validating the output.

- [ ] **Define `ModelSchema` schema.** Keys from `schema.cljs` output: `:trace-sites`, `:splice-sites`, `:static?`, `:dynamic-addresses?`, `:has-branches?`, `:dep-order`, plus optional compiled/analytical keys. Verify by validating the schema of an actual `DynamicGF`.

- [ ] **Define `validated` helper function.** Gated by `*validate?*` dynamic var. When enabled, validates and throws with humanized errors. When disabled, no-op.

**Done criterion for 1.1:** All schemas evaluate without error. Each schema validated against real data from the REPL returns true for well-formed data and false for intentionally malformed data (missing key, wrong type).

### 1.2 Add Validation Points

- [ ] **Add validation at `run-handler` entry.** In `runtime.cljs`, validate `init-state` against `HandlerState` before creating the volatile. Gated by `*validate?*`.

- [ ] **Add validation at `merge-sub-result` entry.** In `handler.cljs`, validate `sub-result` against `SubResult`. Gated by `*validate?*`.

- [ ] **Add validation at GFI method exits.** In `dynamic.cljs`, validate return values of `simulate`, `generate`, `update`, `regenerate`, `assess`, `project` against their respective schemas. Gated by `*validate?*`.

**Done criterion for 1.2:** Validation code compiles and the `*validate?*` flag defaults to false (no behavior change with default settings).

### 1.3 Verify Schemas Match Existing Code

- [ ] **Run full test suite with `*validate?*` = true.** Every test must pass. If any test fails, the schema is wrong -- fix the schema, not the code.

```bash
# Run each test file with validation enabled
# (mechanism TBD: dynamic binding at test entry, or env var, or wrapper)
```

- [ ] **Run all memo examples with validation enabled.** All 26 must produce correct output.

```bash
for f in examples/memo/*.cljs; do bun run --bun nbb "$f"; done
```

- [ ] **Run full test suite with `*validate?*` = false.** Confirm zero overhead -- same pass counts, no measurable slowdown.

**Done criterion for Phase 1:** All tests pass in both validation modes. Schemas are proven to describe the actual data shapes.

---

## Phase 2: The Dispatch Refactoring

*Goal: Replace the cond ladders in DynamicGF with a dispatcher stack. All tests pass. Malli catches any structural regressions.*

### 2.1 Create dispatch.cljs

- [ ] **Define `IDispatcher` protocol.** Single method: `(resolve-transition [this op schema opts])`. Returns a transition-spec map or nil.

- [ ] **Define `TransitionSpec` Malli schema.** Keys: `:transition` (fn), `:score-type` (enum), `:init-state-fn` (optional fn), `:post-fn` (optional fn), `:compiled-fn` (optional fn).

- [ ] **Define `ScoreType` Malli schema.** `[:enum :joint :marginal :collapsed :beam-marginal]`.

- [ ] **Implement `HandlerDispatcher`.** Always returns the appropriate base transition for the given op. This is the L0 fallback that always succeeds. Verify: `(resolve-transition handler-dispatcher :simulate schema {})` returns a valid TransitionSpec.

- [ ] **Implement `CompiledDispatcher`.** Checks schema for compiled paths (`:compiled-simulate`, `:compiled-generate`, etc.). Returns compiled fn if present, nil otherwise. Verify: returns non-nil for a static model's schema, nil for a dynamic model's schema.

- [ ] **Implement `AnalyticalDispatcher`.** Checks for `:auto-handlers` in schema and whether conjugate observations are constrained. Returns wrapped transition if applicable, nil otherwise. Verify: returns non-nil for a model with conjugate pairs and matching constraints, nil otherwise.

- [ ] **Implement `CustomTransitionDispatcher`.** Checks for `::custom-transition` in the gen-fn's metadata. Returns it if present, nil otherwise. Verify: returns the transition from a `with-handler`-annotated gen-fn.

- [ ] **Implement `with-handler`.** `(defn with-handler [gf transition] (vary-meta gf assoc ::custom-transition transition))`. Verify: metadata is correctly attached and retrievable.

- [ ] **Define `default-dispatcher-stack`.** Vector: `[custom-transition analytical compiled handler]`.

- [ ] **Implement `resolve`.** Walks the stack, returns first non-nil. Validate result against `TransitionSpec` schema (gated).

- [ ] **Implement `execute-spec`.** Takes a transition-spec and runs it -- either via `run-handler` (for transition specs) or directly (for compiled specs). This function replaces the ~20 `run-*-compiled` / `run-*-handler` / `run-*-prefix` / `run-*-analytical` helpers in `dynamic.cljs`.

**Done criterion for 2.1:** All dispatcher implementations return valid TransitionSpecs (verified by Malli). The stack resolves correctly for models at each compilation level. Tested in REPL against real models.

### 2.2 Modify DynamicGF

- [ ] **Add `dispatch.cljs` require to `dynamic.cljs`.**

- [ ] **Refactor `simulate` method.** Replace the cond ladder with `dispatch/resolve` + `execute-spec`. Keep the `ensure-key`, `rng/seed!`, and `mx/gfi-cleanup!` calls.

- [ ] **Run simulate-specific tests.** L0 certification (68/68), compiled simulate test (82/82), schema test (174/174). All must pass.

- [ ] **Refactor `generate` method.** Same pattern. The analytical path, compiled path, prefix path, and handler path all route through the dispatcher.

- [ ] **Run generate-specific tests.** Gen.clj compat (165/165), GenJAX compat (73/73). All must pass.

- [ ] **Refactor `update` method.** Same pattern.

- [ ] **Refactor `regenerate` method.** Same pattern. The L3.5 auto-regenerate-transition check moves into `AnalyticalDispatcher`.

- [ ] **Refactor `assess` method.** Same pattern.

- [ ] **Refactor `project` method.** Same pattern.

- [ ] **Run full test suite with validation enabled.** All tests must pass with `*validate?*` true.

- [ ] **Run full test suite with validation disabled.** All tests must pass with identical counts.

- [ ] **Run all memo examples.** All 26 must produce correct output.

**Done criterion for 2.2:** Every cond ladder in DynamicGF is replaced by `dispatch/resolve`. All tests pass in both validation modes. The diff shows `dynamic.cljs` getting simpler (fewer lines, no replicated dispatch logic).

### 2.3 Clean Up

- [ ] **Remove dead helper functions from dynamic.cljs.** The `run-simulate-compiled`, `run-simulate-prefix`, `run-simulate-handler`, `run-generate-analytical`, `run-generate-compiled`, `run-generate-prefix`, `run-generate-handler`, `run-update-compiled`, `run-update-prefix`, `run-update-handler`, `run-regen-analytical`, `run-regen-compiled`, `run-regen-prefix`, `run-regen-handler`, `run-assess-analytical`, `analytical-applicable?` functions. These are now inside dispatcher implementations or `execute-spec`.

- [ ] **Run full test suite one more time.** Confirm nothing broke during cleanup.

**Done criterion for 2.3:** `dynamic.cljs` is measurably shorter. No dead code remains. All tests pass.

---

## Phase 3: ExactGF Migration

*Goal: Prove that `with-handler` replaces the ExactGF record. The 26 memo examples are the primary verification.*

### 3.1 Create `enumerate` Function

- [ ] **Define `enumerate` in `inference/exact.cljs`.** `(defn enumerate [gf] (with-handler gf enumerate-transition))`. This replaces both `ExactGF` record usage and `exact/Exact` metadata annotation.

- [ ] **Verify in REPL.** `(p/simulate (enumerate model) args)` must return a valid trace. `(p/generate (enumerate model) args constraints)` must return `{:trace :weight}` with correct marginal likelihood.

### 3.2 Update Memo Examples

- [ ] **Update Sally-Anne.** Replace `(splice :anne (exact/Exact (sa-anne-model ...)))` with `(splice :anne (enumerate (sa-anne-model ...)))`.

- [ ] **Update any other examples using `exact/Exact` or `->ExactGF`.** Search for `ExactGF`, `exact/Exact`, `exact/thinks` across examples/.

- [ ] **Run all memo examples.** All 26 must produce identical output to baseline.

### 3.3 Remove ExactGF Record

- [ ] **Delete the `ExactGF` defrecord and its protocol implementations** from `inference/exact.cljs` (lines 137-189 approximately).

- [ ] **Delete `exact/Exact` function** (the metadata annotation, line 107-113). Replaced by `enumerate`.

- [ ] **Keep everything else in exact.cljs.** `enumerate-transition`, `run-enumerate`, `enumerate-executor`, `execute-exact`, all post-processing algebra (`marginal`, `condition-on`, `extract-table`, `expectation`, `entropy`, `variance`, `mutual-info`), all high-level API (`exact-posterior`, `exact-joint`, `pr`, `observes`), and utilities (`categorical-argmax`, `with-cache`).

- [ ] **Update `exact/thinks` to use `enumerate`.** `(defn thinks [model] (enumerate model))` instead of `(->ExactGF model)`.

- [ ] **Run all memo examples.** All 26 must match baseline.

- [ ] **Run full test suite.** All tests must pass.

**Done criterion for Phase 3:** ExactGF record is deleted. `enumerate` is the single entry point. All memo examples produce correct results. All tests pass.

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
