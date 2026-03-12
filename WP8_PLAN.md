# WP-8: Combinator Compiled Update — Spec

## Goal

Extend the `update` method of all 5 combinators (Map, Unfold, Scan, Switch, Mix)
to dispatch to `compiled-update` on their kernel/branch when available, exactly as
WP-7 did for `generate`. The handler path is ground truth; the compiled path must
produce identical traces, scores, weights, and discards.

## Signatures

The compiled-update function (from WP-3) has this signature:

```
(fn [key args-vec constraints old-choices]
  -> {:values {addr->val} :score :discard {addr->old-val} :retval})
```

Each combinator's `update` method will check `(compiled/get-compiled-update kernel)`.
If present, call it directly instead of `p/update`. If nil, fall back to existing
handler path (no change).

## Per-Combinator Design

### 1. Map (~15 lines)

**Current**: Reconstructs per-element old traces, calls `p/update` on kernel per element.

**Compiled path**:
- `(if-let [cupd (compiled/get-compiled-update kernel)] ...)`
- Per element `i`: extract `elem-args`, `elem-constraints = cm/get-submap(constraints, i)`,
  `elem-old-choices = cm/get-submap(old-choices, i)`
- Call `(cupd key elem-args elem-constraints elem-old-choices)` → result
- Convert `(:values result)` → choices via `values->choices`
- Convert `(:discard result)` → discard via `values->choices`
- Accumulate `score`, per-element scores for metadata
- Weight = `total-new-score - old-trace-score` (same as current)
- Key splitting: `rng/split` per element (same pattern as WP-7 generate)
- **No prefix-skip**: Map has no sequential dependency, all elements independent

**Metadata**: `::element-scores` (vec of per-element scores), `::compiled-path true`

### 2. Unfold (~25 lines)

**Current**: Prefix-skip optimization (finds first changed step), then sequential
`p/update` from that point.

**Compiled path**:
- `(if-let [cupd (compiled/get-compiled-update kernel)] ...)`
- **Prefix-skip preserved**: Same first-changed detection using `::step-scores` metadata
- Steps `0..first-changed-1`: reuse from old trace (no re-execution)
- Steps `first-changed..n-1`: call
  `(cupd key [t state & extra] step-constraints step-old-choices)` → result
- `step-old-choices = cm/get-submap(old-choices, t)`
- Convert values/discard via `values->choices`
- Thread state: `new-state = (:retval result)`
- Key splitting per step
- Weight = `total-new-score - old-trace-score`

**Metadata**: `::step-scores`, `::compiled-path true`

### 3. Scan (~25 lines)

**Current**: Same prefix-skip as Unfold, but with carry/output threading.

**Compiled path**: Mirrors Unfold exactly, with these differences:
- Kernel args: `[carry (nth inputs t)]` instead of `[t state & extra]`
- Retval destructuring: `[new-carry output] = (:retval result)`
- Track both `step-scores` and `step-carries` in metadata

**Metadata**: `::step-scores`, `::step-carries`, `::compiled-path true`

### 4. Switch (~10 lines)

**Current**: Same-branch → `p/update`, different-branch → `p/generate`.

**Compiled path** (same-branch case only):
- `(if-let [cupd (compiled/get-compiled-update branch)] ...)`
- Call `(cupd key branch-args constraints old-choices)` → result
- Convert values/discard via `values->choices`
- Weight = new-score - old-trace-score

**Different-branch case**: Use `compiled/get-compiled-generate` (already done in
WP-7). If unavailable, fall back to `p/generate`.

**Metadata**: `::switch-idx`, `::compiled-path true`

### 5. Mix (~15 lines)

**Current**: Same-component → `p/update` inner, different-component → `p/generate` inner.

**Compiled path** (same-component case only):
- `(if-let [cupd (compiled/get-compiled-update component)] ...)`
- Inner old choices = everything except `:component-idx`
- Inner constraints = everything except `:component-idx`
- Call `(cupd key args inner-constraints inner-old-choices)` → result
- Total score = inner-score + idx-score
- Weight = new-total-score - old-trace-score

**Different-component case**: Use `compiled/get-compiled-generate` (already done
in WP-7). If unavailable, fall back to `p/generate`.

**Metadata**: `::compiled-path true`

## Helper: discard→choices

The compiled-update returns `{:discard {addr->old-val}}` — a plain map. Reuse
existing `values->choices` for both values and discard:

```clojure
(values->choices (:values result))   ;; → ChoiceMap of new values
(values->choices (:discard result))  ;; → ChoiceMap of discarded old values
```

No new helper needed.

## Code Changes

**File**: `src/genmlx/combinators.cljs` only (~90 lines added)

Each combinator's `update` method gets an `if-let` branch at the top (same
pattern as WP-7 generate). The existing handler path stays as the `else`
branch — zero changes to fallback logic.

## Tests (~60 assertions)

**File**: `test/genmlx/wp8_combinator_update_test.cljs`

**Test kernels**: Reuse WP-7 pattern — simple `gen` functions with 1-2 trace
sites using compilable distributions (gaussian, uniform, exponential).

### Per combinator (5 combinators × ~12 tests each):

1. **Compilation check** — `compiled/get-compiled-update` returns non-nil for
   compilable kernel
2. **No constraints** — weight=0, score unchanged, choices unchanged, discard=EMPTY
3. **Single site constrained** — constrained site gets new value, unconstrained
   sites kept, weight/score match handler
4. **All sites constrained** — all new values, weight/score match handler
5. **Discard correctness** — discard contains old values of constrained sites only
6. **Compiled vs handler equivalence** — `force-handler` comparison (score, weight,
   discard structure)
7. **Non-compilable fallback** — beta-dist kernel falls back to handler path
8. **Metadata** — `::compiled-path true`, `::element-scores`/`::step-scores`/etc.

### Combinator-specific tests:

- **Unfold/Scan**: Prefix-skip still works (constrain only step 2 of 3 → steps
  0,1 reused from old trace)
- **Unfold/Scan**: State threading through compiled steps
- **Switch**: Same-branch compiled update vs different-branch compiled generate
- **Mix**: Same-component compiled update vs different-component compiled generate
- **Map**: Partial constraints (some elements constrained, some not)

### Comparison pattern (standard for all):

```clojure
;; Create deterministic starting trace via generate with full constraints
(let [trace (make-trace-via-generate combinator args full-constraints)
      ;; Run compiled update with new constraints
      result (p/update combinator trace new-constraints)
      ;; Run handler update for comparison
      combinator-h (make-handler-combinator kernel)
      trace-h (make-trace-via-generate combinator-h args full-constraints)
      result-h (p/update combinator-h trace-h new-constraints)]
  (assert-close "weight matches" (:weight result-h) (:weight result) 1e-5)
  (assert-close "score matches" (:score (:trace result-h)) (:score (:trace result)) 1e-5))
```

## Invariants

1. **Compiled ≡ Handler**: Identical traces, scores, weights, discards
2. **Weight = new-score - old-score** (computed at combinator level)
3. **Discard**: Contains only old values of constrained sites
4. **Kept sites**: Retain exact values from input trace (no resampling)
5. **Prefix-skip**: Unfold/Scan skip unchanged prefix steps (same as handler)
6. **Graceful fallback**: Non-compilable kernels use handler path transparently
7. **Metadata preserved**: All existing metadata (element-scores, step-scores, etc.)
   plus `::compiled-path`

## Execution Order

1. Write test file with all ~60 assertions (most will fail initially)
2. Implement Map compiled update (~15 lines)
3. Implement Unfold compiled update (~25 lines)
4. Implement Scan compiled update (~25 lines)
5. Implement Switch compiled update (~10 lines)
6. Implement Mix compiled update (~15 lines)
7. Run tests → all green
8. Run regressions (L0 68/68, M1 174/174, M2 82/82, M3 92/92, M5 90/90,
   WP-3 56/56, WP-4 60/60, WP-7 86/86)

## Risks

- **Key threading in update**: The compiled-update function takes a key but
  doesn't split it (update never samples). `build-update-site-step` threads
  the key through state but never uses it. Safe.
- **Discard assembly for sequential combinators**: Unfold/Scan must only include
  discard entries for steps that were actually re-executed (not prefix-skipped).
  The existing handler path already handles this correctly.
- **Switch different-branch**: Different-branch update calls `p/generate`, which
  was already compiled in WP-7. The update path should try
  `compiled/get-compiled-generate` for the new branch, falling back to
  `p/generate`. Minor enhancement beyond just `compiled/get-compiled-update`.
