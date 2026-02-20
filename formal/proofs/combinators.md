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
distribution (conditioned on the carry state from step t-1). The total
score equals the log-density under the sequential composition of
conditionals, which is the Unfold trace distribution by chain rule. ✓

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
The log-density decomposes as log P(idx) + log P(trace_inner|idx), which
is exactly the two-term sum above. ✓

**C2 (Generate weight):**
If `:component-idx` is constrained, its density contributes to the weight.
The inner component's generate weight adds inner constrained densities.
Total weight = weight_idx + weight_inner. ✓

**C3 (Update weight):**
Two cases: same component or different component.
- Same: update inner, weight = score_new - score_old (idx score unchanged)
- Different: generate new component from scratch, weight = score_new - score_old

Both computed in `combinators.cljs:970-1020`. ✓

**C5 (Discard):**
- Same component: discard from inner update
- Different component: old choices (entire inner trace displaced)
✓

**Implementation:** `MixCombinator` at `combinators.cljs:919-1062`.

### 3.7 Recurse(maker)

**Trace type:** Recurse(maker) : G_γ η (fixed point)

Recurse creates a fixed-point combinator: `maker(self)` returns a GF
that may call `self` recursively.

**C1-C5:** At each invocation, Recurse calls `maker(this)` to get the
inner GF, then delegates the operation to it. Correctness follows from
the IH on the inner GF (which itself may recurse, but each recursion
level delegates to a concrete GF body).

The fixed-point argument: if maker preserves the GFI contract (i.e.,
`maker(g)` satisfies C1-C5 whenever g does), then `fix(maker)` satisfies
C1-C5. This is because any finite execution of `fix(maker)` unfolds to
a finite composition of contract-preserving operations. ✓

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
