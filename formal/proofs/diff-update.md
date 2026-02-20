# Diff-Aware Update Correctness — TODO 10.13

> Theorem: for MapCombinator with VectorDiff and for Unfold/Scan with
> per-step prefix skipping, the incremental update produces the same
> result as full re-execution, but skips unchanged elements/steps.

---

## 1. Diff Types Recap

From `diff.cljs`:

```
Δ  ::=  NoChange                        -- value unchanged
     |  UnknownChange                   -- conservatively assume changed
     |  ValueChange(old, new)           -- specific old and new values
     |  VectorDiff(S : 𝒫(ℕ))           -- set of changed indices
     |  MapDiff(C, A, R : 𝒫(K))        -- changed, added, removed keys
```

Diff types annotate arguments to `update-with-diffs`, allowing combinators
to skip unchanged sub-computations.

---

## 2. MapCombinator with VectorDiff

### Statement

**Theorem (Map Diff-Aware Update).**

Let tr = Map(g).simulate(args) be a trace, let constraints be new
constraints, and let argdiffs = VectorDiff(S) where S is the set of
changed argument indices.

Let C = {i | constraints at index i is non-empty} be the set of
constrained indices.

Then:

```
update-with-diffs(Map(g), tr, constraints, VectorDiff(S))
```

produces the same result as:

```
update(Map(g), tr, constraints)
```

but only re-executes elements i ∈ S ∪ C, reusing cached results for
elements i ∉ S ∪ C.

Specifically:

**(a) Trace equality:**
```
update-with-diffs(…).trace = update(…).trace
```

**(b) Weight equality:**
```
update-with-diffs(…).weight = update(…).weight
```

**(c) Discard equality:**
```
update-with-diffs(…).discard = update(…).discard
```

**(d) Efficiency:**
Only |S ∪ C| calls to `g.update` are made (vs n for full update).

### Proof

**Score decomposition.** The Map score is the sum of per-element scores:
```
score_Map = Σᵢ₌₀ⁿ⁻¹ scoreᵢ
```

For the full update:
```
weight_full = score_new - score_old = Σᵢ (score_new_i - score_old_i)
```

**Unchanged elements (i ∉ S ∪ C):**

For unchanged elements:
1. Arguments are identical (i ∉ S ⇒ args_new[i] = args_old[i])
2. No constraints (i ∉ C ⇒ constraints at i = EMPTY)

Therefore, a full update at index i would:
- Keep old choices (no constraints, old choices exist)
- Compute the same score (same distribution, same values)
- Produce weight_i = 0 (score_new_i = score_old_i)
- Produce empty discard

The diff-aware path reuses the cached `score_old_i` from `::element-scores`
metadata (stored at simulate/generate/update time), which equals `score_new_i`.

So: score_new_i = score_old_i, weight_i = 0, discard_i = EMPTY. ✓

**Changed elements (i ∈ S ∪ C):**

For changed elements, the diff-aware path calls `g.update` exactly as the
full update would, producing the same trace, score, weight, and discard
for that element. ✓

**Total weight:**

```
weight_diff = score_new_total - score_old_total
            = Σᵢ score_new_i - Σᵢ score_old_i
```

For i ∉ S ∪ C: score_new_i = score_old_i (reused from cache)
For i ∈ S ∪ C: score_new_i from g.update

The sum is identical to the full update's weight. ✓

**Total discard:**
- Unchanged elements contribute EMPTY
- Changed elements contribute the same discard as full update
- Union is identical ✓

**Implementation:** `MapCombinator/update-with-diffs` at
`combinators.cljs:1112-1170`.

The key optimization: the loop at `combinators.cljs:1137-1155` checks
`(contains? update-set i)` — only elements in the update set are
re-executed.

---

## 3. Unfold/Scan with Prefix Skipping

### Statement

**Theorem (Unfold/Scan Prefix Skip).**

Let tr = Unfold(g).simulate([n, init, extras...]) be a trace with
`::step-scores` metadata. Let constraints affect only steps at indices
≥ t₀ (i.e., all constraints are at steps t₀, t₀+1, …).

Then `update(Unfold(g), tr, constraints)` correctly:

1. **Skips prefix** steps 0, 1, …, t₀-1 (reuses cached choices and scores)
2. **Re-executes** steps t₀, t₀+1, …, n-1 via g.update
3. Produces the same trace, weight, and discard as full re-execution

**Note:** Steps after the first constraint are ALL re-executed even if they
have no constraints, because the carry state may have changed.

### Proof

**Prefix correctness (steps 0 to t₀-1):**

For each step t < t₀:
- No constraints at step t (by assumption)
- Arguments are unchanged (same carry state from previous step)
- Therefore the step's choices, score, and carry state are identical to
  the original execution

The cached `::step-scores[t]` correctly records step t's score.
The prefix choices are copied directly from the old trace.
The prefix carry state is recovered from `(:retval trace)[t₀-1]` (for
Unfold) or `::step-carries[t₀-1]` (for Scan).

**Suffix correctness (steps t₀ to n-1):**

From step t₀ onward, each step is re-executed via `g.update` with:
- Old choices from the original trace at that step
- Constraints scoped to that step
- Carry state from the previous (re-executed) step

This is identical to what a full update would do from step t₀ onward. ✓

**Weight correctness:**

```
weight = score_new - score_old
```

where:
- score_new = Σ_{t<t₀} old_score_t + Σ_{t≥t₀} new_score_t
- score_old = Σ_{t=0}^{n-1} old_score_t

So:
```
weight = [Σ_{t<t₀} old_score_t + Σ_{t≥t₀} new_score_t] - Σ_t old_score_t
       = Σ_{t≥t₀} (new_score_t - old_score_t)
```

This correctly captures only the weight changes from re-executed steps. ✓

**Implementation:**
- Unfold: `UnfoldCombinator/update` at `combinators.cljs:183-248`
- Scan: `ScanCombinator/update` at `combinators.cljs:679-752`
- The prefix boundary `first-changed` is computed at `combinators.cljs:191-197`
  (Unfold) and `combinators.cljs:689-695` (Scan).

### Metadata Stripping

When arguments change (detected via `update-with-diffs` with non-NoChange
argdiffs), the `::step-scores` metadata is stripped to prevent invalid
prefix skipping:

```clojure
;; combinators.cljs:1182
(p/update this (with-meta trace (dissoc (meta trace) ::step-scores)) constraints)
```

This is correct because when arguments change:
- The carry state at each step may differ
- Old per-step scores are no longer valid
- Full re-execution from step 0 is required

Without this stripping, the prefix skip would incorrectly reuse scores
computed under different arguments. ✓

---

## 4. Correctness of NoChange Fast Path

For both Map and Unfold/Scan, when argdiffs = NoChange AND constraints
= EMPTY:

```clojure
{:trace trace :weight (mx/scalar 0.0) :discard cm/EMPTY}
```

This is correct because:
- No arguments changed → same distributions at all sites
- No constraints → no values replaced
- Therefore: trace unchanged, weight = 0, discard empty ✓

**Implementation:**
- Map: `combinators.cljs:1122-1123`
- Unfold: `combinators.cljs:1177-1178`
- Scan: `combinators.cljs:1200-1201`

---

## 5. Summary of Efficiency Gains

| Scenario | Full Update | Diff-Aware Update |
|----------|------------|-------------------|
| Map: 1 of n elements constrained | n calls to g.update | 1 call |
| Map: k elements changed (VectorDiff) | n calls | k calls |
| Unfold: constraint at step t₀ | n steps re-executed | n - t₀ steps |
| Unfold: no constraints | n steps re-executed | 0 steps (fast path) |
| Any combinator: NoChange + no constraints | Full re-execution | Immediate return |

These optimizations are critical for MCMC performance, where each MH step
typically changes only one or a few addresses. With diff-aware update, the
cost of an MH step is proportional to the number of changed elements rather
than the total model size.

---

## 6. Implementation Correspondence

| Formal Concept | Implementation | Location |
|----------------|---------------|----------|
| VectorDiff(S) | `{:diff-type :vector-diff :changed S}` | `diff.cljs:24-28` |
| NoChange | `{:diff-type :no-change}` | `diff.cljs:11-12` |
| UnknownChange | `{:diff-type :unknown-change}` | `diff.cljs:14-15` |
| Map update-with-diffs | `MapCombinator/update-with-diffs` | `combinators.cljs:1112-1170` |
| Unfold update (prefix skip) | `UnfoldCombinator/update` | `combinators.cljs:183-248` |
| Scan update (prefix skip) | `ScanCombinator/update` | `combinators.cljs:679-752` |
| Unfold update-with-diffs | `UnfoldCombinator/update-with-diffs` | `combinators.cljs:1172-1186` |
| Step scores metadata | `::step-scores` in trace meta | `combinators.cljs:149,169` |
| Step carries metadata (Scan) | `::step-carries` in trace meta | `combinators.cljs:640` |
| Metadata stripping | `(dissoc (meta trace) ::step-scores)` | `combinators.cljs:1182` |
| Element scores (Map) | `::element-scores` in trace meta | `combinators.cljs:52,69` |
