# L1-M5: Combinator-Aware Compilation — Spec

## Goal

When a combinator wraps a compilable kernel (a DynamicGF with `:compiled-simulate`),
the combinator's `simulate` bypasses per-step/element Trace and ChoiceMap construction
by calling the kernel's compiled-simulate function directly and building the output
Trace once at the end.

## Kernel eligibility

A kernel is eligible if `(:compiled-simulate (:schema kernel))` is non-nil:
- Kernel is a DynamicGF
- Compiled by L1-M2 (static) or L1-M4 (branch-rewritten)
- All trace sites have noise transforms

Non-eligible kernels fall back to existing handler path unchanged.

## Per-combinator compiled path

### 1. UnfoldCombinator

**Current per-step**: `(p/simulate kernel [t state extra...])` → DynamicGF.simulate →
compiled-sim → build ChoiceMap → build Trace → merge choices[t]

**Compiled per-step**: `(compiled-sim step-key [t state extra...])` →
`{:values :score :retval}` → collect in plain vector

**After loop**: Build one ChoiceMap from all collected values. Build one Trace.

**Key threading**: Initial key from `rng/fresh-key`. Split per step.

### 2. ScanCombinator

Same pattern as Unfold. Kernel args: `[carry input]`. Retval: `{:carry :outputs}`.

### 3. MapCombinator

**Compiled per-element**: `(compiled-sim key_i args_i)` → collect, build at end.

### 4. SwitchCombinator

If selected branch has compiled-simulate, call it directly. Build Trace once.

### 5. MixCombinator

Sample index, if selected component has compiled-simulate, call it directly.

## Architecture

No new compilation infrastructure needed. Each combinator's `simulate` gains a branch:

```
if kernel has :compiled-simulate in :schema
  → compiled path (call compiled-sim directly, defer Trace construction)
else
  → existing handler path (unchanged)
```

compiled-simulate is a pure function `(fn [key args-vec] -> {:values {addr->val} :score :retval})`.
Calling it directly bypasses:
- Protocol dispatch (p/simulate)
- DynamicGF.simulate setup (rng/seed!, key metadata)
- Per-step ChoiceMap construction (reduce-kv + cm/set-value)
- Per-step Trace construction (tr/make-trace)
- Per-step ChoiceMap merge in combinator (cm/set-choice choices [t] ...)

## Correctness requirements

- Trace structure: identical ChoiceMap shape (same addresses, nesting)
- Score: within ~1e-6 of handler path
- Retval: identical shape and type
- Step-scores metadata (::step-scores) preserved for update optimization
- Non-compilable kernels: bit-identical to current behavior

PRNG note: values differ numerically from handler path (split-per-step vs
fresh-key-per-step). Both statistically correct.

## What M5 does NOT do

- No mx/compile-fn fusion of combinator loops (host loop is fine)
- No compiled generate/update/regenerate (simulate only)
- No nested combinator compilation
- No IBatchedSplice compilation
- No splice-aware compilation in parent gen fns

## Files modified

- `src/genmlx/combinators.cljs` — compiled paths in simulate for all 5 combinators
- `src/genmlx/compiled.cljs` — utility `get-compiled-simulate`

## Test file

`test/genmlx/combinator_compile_test.cljs` (~60-80 tests)

1. Unfold with compilable kernel — trace structure, scores, retval, step-scores
2. Unfold with non-compilable kernel — fallback, regression safe
3. Scan with compilable kernel — carry threading, outputs
4. Map with compilable kernel — per-element compiled
5. Switch with compilable branch — selected branch compiled
6. Switch with non-compilable branch — fallback
7. Mix with compilable component — selected component compiled
8. Edge cases — single step, zero steps, delta-only, multi-site, extra args
9. Regression — existing combinator tests pass
10. Performance — timing comparison

## Performance expectation

- 1.5-3x for Unfold T=50-100 vs M2-compiled kernel through handler
- Scales with steps/elements
