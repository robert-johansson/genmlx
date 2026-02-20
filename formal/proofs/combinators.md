# Combinator Compositionality — TODO 10.11

> Theorem: GFI contracts are preserved by all combinators. If the inner
> generative function satisfies the GFI contract, then the combinator-
> wrapped function also satisfies it.

---

## 1. GFI Contract

A generative function g : G_γ η satisfies the **GFI contract** if:

**C1 (Score consistency):** For any trace u from simulate{g}, the score
equals the log-density of u under the trace distribution:
```
score(u) = log(dμ/dν(u)) = Σ_a log density_{d_a}(u(a))
```

**C2 (Generate weight):** For constraints obs, the generate weight equals
the sum of log-densities at constrained addresses:
```
weight = Σ_{a ∈ dom(obs)} log density_{d_a}(obs(a))
```

**C3 (Update weight):** For old trace u and new constraints c, the
update weight equals the score difference:
```
weight = score(u_new) - score(u_old)
```

**C4 (Regenerate weight):** For selection s, the regenerate weight
satisfies:
```
weight = score(u_new) - score(u_old) - proposal_ratio
```
where proposal_ratio = Σ_{a ∈ selected(s)} (new_lp(a) - old_lp(a)).

**C5 (Discard completeness):** Update's discard contains the old value
at every address where a new constraint was applied.

---

## 2. Statement

### Theorem (Combinator Compositionality)

For each combinator C ∈ {Map, Unfold, Switch, Scan, Mask, Mix, Recurse,
Contramap, Dimap} and each GFI operation op ∈ {simulate, generate, update,
regenerate}:

If g satisfies GFI contracts C1-C5, then C(g) satisfies C1-C5.

---

## 3. Proof per Combinator

### 3.1 Map(g : G_γ η)

**Trace type:** Map(g) : [η₁] → G_{[i:γ]} [η]

Map applies g independently to each element of the input sequences.

**C1 (Score):** The Map score is the sum of per-element scores:
```
score_Map = Σᵢ score_i = Σᵢ Σ_{a ∈ γ} log density_{d_a}(uᵢ(a))
```
This equals the log-density under the product measure μ₁ ⊗ ⋯ ⊗ μₙ,
which is the trace distribution of Map(g). ✓

**C2 (Generate weight):** Constraints at element i scope to element i:
```
weight_Map = Σᵢ weight_i
= Σᵢ Σ_{a ∈ dom(obs_i)} log density(obs_i(a))
= Σ_{(i,a) ∈ dom(obs)} log density(obs(i,a))
```
This is the sum of log-densities at all constrained addresses in the
Map trace. ✓

**C3 (Update weight):**
```
weight_Map = score_new - score_old
= Σᵢ score_new_i - Σᵢ score_old_i
= Σᵢ (score_new_i - score_old_i)
= Σᵢ weight_i
```
By the IH, each weight_i equals the per-element score difference. ✓

**C4 (Regenerate weight):** Analogous to update — regenerate weight
decomposes as sum of per-element weights. ✓

**C5 (Discard):** Each element's discard contains its displaced old values.
The Map discard is the union of per-element discards, nested by index. ✓

**Implementation:** `MapCombinator` at `combinators.cljs:38-122`.

### 3.2 Unfold(g : G_γ η)

**Trace type:** Unfold(g) : (ℕ × η₁ × …) → G_{[t:γ]} η

Unfold applies g sequentially: each step receives the carry state from
the previous step. Steps are independent in their random choices
(different addresses 0, 1, 2, …) but dependent through the carry state.

**C1 (Score):**
```
score_Unfold = Σ_{t=0}^{n-1} score_t
```
Each score_t is the log-density of step t's choices under g's trace
distribution conditioned on the carry state from step t-1.

**Carry-state dependency.** Unlike Map (where elements are independent),
Unfold steps are sequentially dependent through the carry state c_t.
The trace distribution factors as:

```
μ_Unfold(u₀, u₁, …, u_{n-1})
  = μ_g(u₀ | c₀) · μ_g(u₁ | c₁(u₀)) · ⋯ · μ_g(u_{n-1} | c_{n-1}(u₀,…,u_{n-2}))
```

where c_t is the carry state after step t, which depends deterministically
on the choices at steps 0, …, t-1 (via the return values of g). The
log-density is:

```
log(dμ_Unfold/dν)(u₀,…,u_{n-1})
  = Σ_t log(dμ_g/dν_γ)(u_t | c_t)
  = Σ_t score_t
```

This factorization is valid because:
1. The **stock measure** ν_{[t:γ]} = ν_γ^{⊗n} is a product (no carry
   dependency in the reference measure)
2. The **trace measure** factors by the **chain rule of probability**:
   P(u₀,…,u_{n-1}) = Π_t P(u_t | u₀,…,u_{t-1}) = Π_t P(u_t | c_t)
   where the second equality holds because u_t is conditionally
   independent of u₀,…,u_{t-1} given c_t (the carry state is a
   sufficient statistic for the history)
3. Each conditional density dμ_g(u_t | c_t)/dν_γ = score_t by the IH
   on g (C1 applied to g with carry state c_t as argument)

The score sum therefore equals the log-density of the joint trace
under the Unfold trace distribution. ✓

**C2 (Generate weight):** Constraints at step t scope to step t:
```
weight_Unfold = Σ_t weight_t
```
Each weight_t is correct by the IH. The sum gives the total constrained
density. ✓

**C3 (Update weight):**
```
weight_Unfold = score_new - score_old = Σ_t (score_new_t - score_old_t)
```
Computed in `combinators.cljs:230` as `(mx/subtract score (:score trace))`.

Note: with per-step prefix skipping, only steps from `first-changed`
onward are re-executed. The prefix scores are reused from `::step-scores`
metadata. The total score is still Σ_t score_t, just computed more
efficiently. ✓

**C4, C5:** Analogous to Map but sequential. ✓

**Implementation:** `UnfoldCombinator` at `combinators.cljs:136-286`.

### 3.3 Switch(g₁, …, gₙ)

**Trace type:** Switch(g₁,…,gₙ) : ℤₙ → G_{γᵢ} η (heterogeneous)

Switch selects a single branch based on an index and delegates entirely
to that branch.

**C1 (Score):** score_Switch = score_{branch_idx}. Since only one branch
executes, the trace distribution is the selected branch's distribution.
The score is correct by the IH on that branch. ✓

**C2-C5:** All operations delegate to the selected branch with unchanged
constraints/selection/etc. Correctness follows directly from the IH on
the selected branch. ✓

**Implementation:** `SwitchCombinator` at `combinators.cljs:294-354`.

### 3.4 Scan(g : G_γ (C × O))

**Trace type:** Scan(g) : (C × [η₁]) → G_{[t:γ]} (C × [O])

Scan is like Unfold but takes an explicit input sequence and produces
both carry state and output sequence. The argument is identical to
Unfold: sequential composition with carry threading.

**C1-C5:** Same argument as Unfold — scores sum across steps, constraints
scope to individual steps, prefix skipping reuses cached scores. ✓

**Implementation:** `ScanCombinator` at `combinators.cljs:624-795`.

### 3.5 Mask(g : G_γ η)

**Trace type:** Mask(g) : 𝔹 → G_{γ|{}} η

Mask gates execution on a boolean: when active (true), delegates to g;
when inactive (false), produces an empty trace with zero score.

**Active case (true):**
All operations delegate to g with unchanged arguments. Correctness follows
from the IH on g. ✓

**Inactive case (false):**
```
choices = EMPTY, score = 0, weight = 0, discard = EMPTY, retval = nil
```
The empty trace has density 1 (empty product), log-density 0. All weights
are zero (no addresses constrained/changed/selected). ✓

**Implementation:** `MaskCombinator` at `combinators.cljs:363-436`.

### 3.6 Mix(ws, g₁, …, gₙ)

**Trace type:** Mix(ws, g₁,…,gₙ) : G_{{:component-idx}⊕γᵢ} η

Mix is a first-class mixture model: sample a component index from a
categorical distribution, then execute that component.

**C1 (Score):**
```
score_Mix = score_categorical(idx) + score_component(idx)
```
The trace distribution is the mixture: P(trace) = Σᵢ wᵢ · P_i(trace|idx=i).
The joint density of (idx, trace_inner) decomposes as:

```
p(idx, trace_inner) = p(idx) · p(trace_inner | idx)
                    = categorical(idx; ws) · p_{g_idx}(trace_inner)
```

Taking logs: log p = log categorical(idx; ws) + log p_{g_idx}(trace_inner)
= score_categorical(idx) + score_component(idx). By the IH on g_idx,
score_component(idx) is the correct log-density. ✓

**C2 (Generate weight):**
If `:component-idx` is constrained, its density contributes to the weight.
The inner component's generate weight adds inner constrained densities.
Total weight = weight_idx + weight_inner. ✓

**C3 (Update weight) — detailed treatment:**

Two cases arise depending on whether the component index changes.

**Case 1: Same component** (old_idx = new_idx = i).

The categorical address `:component-idx` is unchanged (or constrained
to the same value). The inner trace is updated via `g_i.update`:

```
weight_Mix = (score_cat_new + score_inner_new) - (score_cat_old + score_inner_old)
```

Since old_idx = new_idx: score_cat_new = score_cat_old (same categorical
log-prob). So:

```
weight_Mix = score_inner_new - score_inner_old = weight_inner
```

The discard contains displaced inner choices. The categorical address
is NOT in the discard (unchanged). ✓

**Case 2: Different component** (old_idx = i, new_idx = j ≠ i).

This happens when `:component-idx` is constrained to a new value j.
The old component i's entire trace is discarded, and a new component j
is generated from scratch (constrained by any provided constraints):

```
score_new = score_cat(j) + score_{g_j}(trace_new)
score_old = score_cat(i) + score_{g_i}(trace_old)
weight_Mix = score_new - score_old
```

The weight accounts for both the change in categorical probability
(different mixture weights) AND the complete replacement of the inner
trace. This is computed at `combinators.cljs:970-1020`:

1. `p/generate` the new component j with constraints → new inner trace
2. Total new score = categorical(j) + inner_new_score
3. Weight = total_new_score - old_trace_score

The discard contains:
- `:component-idx` → old_idx (the old component index)
- All old inner choices (entire old component trace displaced)

This is correct because `update`'s contract requires the discard to
contain old values at all changed addresses. When the component switches,
*every* address changes (old component fully removed, new component
fully created). ✓

**C4 (Regenerate weight):**

If `:component-idx` is in the selection, a new component index is
sampled from the categorical, potentially triggering a component switch.
The weight accounting follows the same two-case structure as C3. If
the component index is NOT selected, only inner addresses in the
selection are regenerated within the current component. ✓

**C5 (Discard):**
- Same component: discard from inner update (nested under component key)
- Different component: old `:component-idx` + old inner choices (entire
  old component trace displaced)
✓

**Implementation:** `MixCombinator` at `combinators.cljs:919-1062`.

### 3.7 Recurse(maker)

**Trace type:** Recurse(maker) : G_γ η (fixed point)

Recurse creates a fixed-point combinator: `maker(self)` returns a GF
that may call `self` recursively.

**Well-foundedness condition.** The correctness argument requires that
every execution of `fix(maker)` terminates, i.e., the recursive
unfolding reaches a base case after finitely many steps. Formally:

**Definition (Well-founded recursion).** A maker function is
*well-founded* if there exists a measure μ : Args → ℕ (mapping
arguments to natural numbers) such that every recursive call
`self(args')` within `maker(self)(args)` satisfies μ(args') < μ(args).

GenMLX does NOT enforce this statically — it is the programmer's
responsibility. If the recursion does not terminate, the handler will
diverge (infinite loop consuming stack), and no GFI contract applies.

**C1-C5 by induction on recursion depth.**

Let R_k = the k-th unfolding of `fix(maker)`:
- R₀ = ⊥ (undefined — represents non-terminating recursion)
- R_{k+1} = maker(R_k)

For well-founded recursion, every execution terminates at some finite
depth d, so we can prove correctness by induction on d.

**Base case (d = 0):** The body of `maker(self)` reaches a base case
without calling `self`. In this case, the execution is a sequence of
`trace` and pure operations — correctness follows from the handler
soundness theorem (see `handler-soundness.md`).

**Inductive case (d → d+1):** The body of `maker(self)` calls `self`
via `splice(k, self, args')`. By the induction hypothesis, the
recursive call `self(args')` satisfies C1-C5 at depth d. The splice
operation nests the sub-result under address k, preserving the trace
structure (see `handler-soundness.md` §5). The outer body's remaining
operations are correct by the handler soundness theorem.

The combined trace, score, weight, and discard are correct because:
1. The outer body's trace sites contribute their densities correctly
2. The recursive sub-GF's trace is nested under k and contributes its
   score/weight by the IH
3. The trace type at each recursion level uses distinct address prefixes
   (enforced by the `splice` address nesting), so there are no address
   collisions

**Score structure.** For a Recurse GF that unfolds to depth d:

```
score_Recurse = score_body + score_splice₁ + score_splice₂ + ⋯
```

where each `score_splice_j` is the score of a recursive sub-call, and
the total score is the sum of log-densities at all addresses in the
(potentially deep) trace tree. This sum equals log(dμ/dν) for the
full trace by the product structure of the stock measure over the
(finite but dynamic) set of addresses. ✓

**Remark on measure-theoretic subtlety.** The trace type γ of a
Recurse GF is not statically determined — it depends on the runtime
recursion depth, which may depend on random choices (e.g., a geometric
recursion). The stock measure is therefore defined over a countable
union of finite product spaces: ν = Σ_d ν_{γ_d} where γ_d is the
trace type at depth d. Absolute continuity is preserved at each depth
by the compositionality argument above.

**Implementation:** `RecurseCombinator` at `combinators.cljs:445-522`.

### 3.8 Contramap(f, g : G_γ η)

**Trace type:** Contramap(f, g) : G_γ η (args transformed)

Contramap transforms arguments before passing to g. The trace structure
is identical to g's.

**C1-C5:** All operations transform the arguments via f then delegate to g.
Since f does not affect the trace (choices, score, weight, discard), all
contracts are preserved by the IH on g. ✓

**Implementation:** `ContramapGF` at `combinators.cljs:802-886`.

### 3.9 Dimap(f, h, g : G_γ η)

**Trace type:** Dimap(f, h, g) : G_γ η' (args and return transformed)

Dimap = Contramap(f, MapRetval(h, g)). The argument transform f and
return transform h do not affect the trace structure.

**C1-C5:** f transforms arguments (does not affect trace), h transforms
return value (does not affect score/weight). By composition of Contramap
and MapRetval, both of which preserve the contract, Dimap preserves it. ✓

**Implementation:** `dimap` at `combinators.cljs:854-859`. ∎

---

## 4. Implementation Correspondence

| Combinator | Record | Lines | GFI Ops Implemented |
|-----------|--------|-------|---------------------|
| Map | `MapCombinator` | `combinators.cljs:38-122` | simulate, generate, update, regenerate |
| Unfold | `UnfoldCombinator` | `combinators.cljs:136-286` | simulate, generate, update, regenerate |
| Switch | `SwitchCombinator` | `combinators.cljs:294-354` | simulate, generate, update, regenerate |
| Scan | `ScanCombinator` | `combinators.cljs:624-795` | simulate, generate, update, regenerate |
| Mask | `MaskCombinator` | `combinators.cljs:363-436` | simulate, generate, update, regenerate |
| Mix | `MixCombinator` | `combinators.cljs:919-1062` | simulate, generate, update, regenerate |
| Recurse | `RecurseCombinator` | `combinators.cljs:445-522` | simulate, generate, update, regenerate |
| Contramap | `ContramapGF` | `combinators.cljs:802-886` | simulate, generate, update, regenerate |
| Dimap | (composition) | `combinators.cljs:854-859` | (via Contramap + MapRetval) |

All combinators additionally implement `IEdit` (via `edit-dispatch`),
`IUpdateWithDiffs`, and `IProject` — see `combinators.cljs:1064-1355`.
