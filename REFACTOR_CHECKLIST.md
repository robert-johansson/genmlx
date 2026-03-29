# Refactor Branch Cleanup Checklist

Post-audit checklist for `refactor/dispatcher-malli` before merge to `main`.

---

## P0 — Must fix before merge

### 1. `with-handler` contract mismatch — DONE

Implemented option (a) + `with-dispatch`:

- [x] `with-handler` now genuinely accepts a handler transition
  `(fn [state addr dist])`. The `custom-dispatcher` wraps it through the
  standard run-handler machinery via parameterized `run-*` functions.
- [x] Added `with-dispatch` for full op-level dispatch override
  `(fn [op gf args key opts] -> gfi-result)`.
- [x] `enumerate` uses `with-dispatch` (needs custom init-state).
- [x] Handler `run-*` functions parameterized by transition (first arg).
  Handler table uses partial application with standard transitions.
- [x] Updated ARCHITECTURE.md Section 2.4 — documents both mechanisms.
- [x] Updated ARCHITECTURE.md Section 5.4 migration path.

### 2. Stale `ExactGF` references — DONE

- [x] `examples/memo/cheryl.cljs` — updated
- [x] `examples/memo/ultimatum.cljs` (3 locations) — updated
- [x] `examples/memo/trucks.cljs` (2 locations) — updated
- [x] `test/genmlx/exact_test.cljs` (16 locations) — updated
- [x] `README.md` — updated
- [x] `MALLI_INTEGRATION.md` — updated
- [x] `ARCHITECTURE.md` (2 remaining are migration docs, intentional)
- [x] `src/genmlx/inference/exact.cljs` — updated
- [x] `examples/memo/chimp.cljs` — updated (`::custom-dispatch` check)

---

## P1 — Should fix before merge

### 3. `inspect.cljs` uses actual dispatcher stack — DONE

- [x] Added `resolve-dispatch` public accessor to `dynamic.cljs`.
- [x] Added `:label` to each dispatcher's return value.
- [x] Rewrote `inspect.cljs` to use `dyn/resolve-dispatch` instead of
  its own hardcoded `resolve-dispatch` function.
- [x] Removed `dispatch` require (now only requires `dynamic`).

### 4. Remove `schemas` dependency from `handler.cljs` — DONE

- [x] Removed `schemas/validated` call from `merge-sub-result`.
- [x] Added validation at the two call sites in `runtime.cljs` (batched
  splice and scalar executor splice).
- [x] Removed `[genmlx.schemas :as schemas]` from `handler.cljs` requires.
- [x] `handler.cljs` is a pure leaf module again.

### 5. Uniform `run-*` function signatures — DONE

- [x] Option A applied: `_args` in all functions that use `(:args trace)`.
  Applied to 10 functions across compiled, prefix, analytical, and handler
  paths. Handler functions now parameterized (transition as first arg), so
  `_opts`/`_args` are explicit everywhere.

---

## P2 — Should fix soon after merge

### 6. ARCHITECTURE.md status annotations — DONE

- [x] Part I: added `Status: IMPLEMENTED` banner.
- [x] Part IV: added `Status: PLANNED` banner.
- [x] Part VI: added `Status: RESEARCH DIRECTIONS` banner.

### 7. Split `dynamic.cljs` by execution strategy — DEFERRED

Not needed at current scale (857 lines). Revisit when a 5th strategy
is added or the file grows past ~1200 lines.

### 8. ~~Evaluate Malli submodule weight~~ — DROPPED

Temporary bridge to upstream releases (~2 weeks).

### 9. CLAUDE.md updates — DONE

- [x] Added `dispatch.cljs`, `schemas.cljs`, `inspect.cljs` to project structure.
- [x] Updated `dynamic.cljs` description to mention dispatcher stack.
- [x] Added `dispatch` to Layer 2, `schemas` to Layer 3.
- [x] Added Malli to requirements.
- [x] Added "Adding an execution strategy" pattern with `with-handler`/`with-dispatch`.

---

## P3 — Nice to have

### 10. `enumerate-run` defensive check — DONE

- [x] Added default clause to `case` that throws with clear message.

### 11. `run-dispatched` nil-spec check — DONE

- [x] Added `(assert spec ...)` before calling `:run`.

### 12. Schema validation for mode-specific states — DEFERRED

Schemas exist but wiring adds complexity for a dev-only feature.
Revisit when `*validate?*` is used in testing.

### 13. Delete completed plan documents — DONE

- [x] Checked — no REFACTORING_PLAN.md exists (already cleaned up).

---

## Verification gate — PASSED

All test suites pass at or above expected counts:

| Suite | Result |
|---|---|
| L0 certification | 68/68 |
| Schema L1-M1 | 227/227 |
| Compiled simulate L1-M2 | 82/82 |
| Partial compile L1-M3 | 92/92 |
| Combinator compile L1-M5 | 92/92 |
| L4 certification | 41/41 |
| Gen.clj compat | 356/356 |
| GenJAX compat | 73/73 |
| Exact enumeration | 119/120 (pre-existing test 42 fail) |
| Core unit tests (7 suites) | all pass |
| chimp.cljs example | all pass |
