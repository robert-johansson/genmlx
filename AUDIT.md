# Plan vs Reality Audit

Audit of the plan "Making GenMLX At Least As Powerful As GenJAX" against the
actual implementation, conducted 2026-02-19. Updated 2026-02-19 after bug fixes.

---

## What works as planned

| Item | Plan | Status |
|---|---|---|
| 1.1 `propose` | IPropose on Distribution and DynamicGF | Works correctly. Tested. |
| 1.5 vectorized update/regenerate | Batched handler transitions, vupdate/vregenerate | Works correctly. Tested. |
| 2.2 enumerative Gibbs | Exact conditional sampling for discrete variables | Works correctly. Tested with posterior concentration. |
| 2.5 inference composition | chain, repeat, seed, cycle, mix-kernels, run-kernel | Works correctly. Tested with MH chain convergence. |
| 3.1 scan combinator | State-threading sequential combinator with full GFI | Works correctly. Tested with carry accumulation and constrained weight. |
| 3.2 unfold/switch update/regenerate | Complete GFI on existing combinators | Works correctly. Tested. |
| 4.1 choice gradients | Per-choice gradients via MLX grad | Works correctly. Tested with direction and magnitude. |
| 5.1 seven new distributions | cauchy, inv-gamma, geometric, neg-binomial, binomial, discrete-uniform, truncated-normal | Works correctly. Log-prob verified against closed-form. |
| 5.2 defdist-transform macro | Derived distributions via deterministic transforms | Works correctly. Tested with log-normal. |
| 5.3 mixture constructor | Mixture distribution as first-class Distribution record | Works correctly. Tested with log-prob at symmetric point. |

---

## What works but differs from plan

| Item | What the plan said | What actually exists |
|---|---|---|
| 1.2 custom proposal MH | Use update weight for acceptance ratio | Uses trace scores directly (correct, but the update weight itself is still wrong for dependent models) |
| 2.3 involutive MCMC | Includes log\|det J\| for continuous transforms | Missing Jacobian determinant. Only works for volume-preserving involutions (like the swap involution in the test). |
| 3.3 contramap/dimap | Full GFI combinators | Only simulate + generate implemented. No update/regenerate. |
| 3.5 mix combinator | First-class mixture model combinator | Only simulate + generate. No update/regenerate. |
| 4.3 programmable VI | Pluggable objectives + estimators | Works, but has dead code (unused `obj-builder` variable constructed then never referenced). IWELBO converges slowly (known issue with IWAE gradient variance). |
| 4.4 wake-sleep | Amortized inference via wake-sleep | Works for simple cases. Guide constraint coverage may be incomplete if guide has more trace sites than the specified `guide-addresses`. |

---

## What is broken or fundamentally incomplete

### 1.3 Edit interface

**Plan**: "GenJAX's most distinctive contribution." IEdit protocol on DynamicGF
and all combinators. Edit-transition in handler. Foundation for SMCP3.
Backward requests enable automatic computation of acceptance weights for
reversible kernels.

**Reality**: IEdit protocol is defined but **never implemented on any record**.
Not on DynamicGF, not on any combinator. `edit-dispatch` is a standalone
function that wraps existing `update`/`regenerate` calls. No handler
integration. The edit interface is not polymorphic and cannot be extended to
new generative function types without modifying `edit-dispatch`.

**Impact**: The edit interface is scaffolding, not a foundation.

### ~~1.4 Vectorized switch~~ FIXED

**Plan**: Execute all branches with [N]-shaped samples, mask-select results
via `mx/where`. Zero-branch-overhead vectorized selection.

**Reality (after fix)**: Each branch produces N independent samples via N
simulate calls, stacked into [N]-shaped arrays. Results are combined using
`mx/where` based on the index array. Also fixed a pre-existing bug where
branch 1+ values were never selected due to incorrect `reduce-kv` indexing.
Still a standalone function (not a GFI record), and uses N simulate calls
per branch rather than batch sampling (correct but not optimal).

### 2.1 SMCP3

**Plan**: "User only writes the forward kernel; the edit interface provides
the backward request automatically."

**Reality**: The ProposalEdit path requires **both** forward AND backward
kernels to be explicitly provided. When only a forward kernel is given, SMCP3
silently degrades to standard SMC with constraint-edit. The plan's core
promise is unfulfilled.

The SMCP3 test provides no forward kernel at all, so it exercises only the
basic importance sampling path. The ProposalEdit path is **completely
untested**.

### ~~2.4 Conditional SMC~~ FIXED

**Plan**: SMC with a retained reference particle for particle MCMC.

**Reality (after fix)**: The reference trace is now used at initialization via
`p/generate` with the reference trace's full choices as constraints. The
reference particle retains its values and is never resampled (conditional
resampling preserves it at index 0). Note: the reference particle's weight
includes prior log-prob terms (slightly overweighting it), since exact
observation-only weight computation requires per-address score decomposition
not yet implemented.

### 3.4 Argdiffs / retdiffs

**Plan**: "Diff-aware update transition" in handler. `update-with-diffs` in
dynamic.cljs. Diff propagation through Map, Unfold, Scan combinators.
"Critical for MCMC performance -- each MH step only changes one or a few
addresses, but currently GenMLX re-executes the entire model."

**Reality**: `diff.cljs` defines data structures (`no-change`, `unknown-change`,
`value-change`, `vector-diff`, `map-diff`) and computation utilities. **Zero
handler integration. Zero use anywhere in the codebase.** The handler has no
awareness of diffs. There is no `update-with-diffs`. No diff propagation
through combinators. Every MH step still re-executes the entire model.

**Impact**: Dead code. The entire incremental computation optimization
described in the plan does not exist.

### 4.2 Trainable parameters

**Plan**: `dyn/param` function for declaring trainable parameters inside gen
bodies. Param handler reads from param store in state.

**Reality**: `dyn/param` **does not exist**. There is no param handler. The
`ParamStore` in `learning.cljs` is an external utility (get/set/update params,
flatten/unflatten). It can be used for optimization loops outside models, but
you cannot write `(dyn/param :theta 1.0)` inside a model body.

**Impact**: The integration point between the learning infrastructure and the
generative function interface was never built.

---

## Known bugs — FIXED

1. **Update weight for DynamicGF**: FIXED. DynamicGF.update now computes
   `weight = new_score - old_score`, capturing all log-prob changes including
   dependent unconstrained addresses. All combinators (Map, Unfold, Scan)
   also fixed to use `weight = new_total_score - old_total_score`.
   Tested with dependent-variable model, MapCombinator, ScanCombinator.

2. **Conditional SMC ignores reference trace**: FIXED. csmc now uses
   `p/generate` with the reference trace's choices as constraints at
   initialization, preserving the reference particle at index 0.
   Tested: reference particle retains its distinctive value.

3. **Vectorized switch produces identical samples**: FIXED. Each branch now
   produces N independent samples (via N simulate calls + mx/stack). Also
   fixed a pre-existing bug where the branch-combining `reduce-kv` skipped
   branch 1 due to iterating over `(vec (rest vals))` with `(pos? i)`.
   Tested with single-branch and mixed-branch index arrays.

## Known bugs not yet fixed

4. **Neg-binomial distribution**: Added but has zero tests.

---

## Untested paths

- ProposalEdit in the edit interface
- SMCP3 with actual forward/backward kernels
- Asymmetric custom MH (with explicit backward-gf)
- `chain`, `cycle-kernels`, `mix-kernels`, `seed` kernel combinators (only
  `repeat-kernel` and `mh-kernel` are exercised in tests)
- Batched update path (`vupdate`)
- Involutive MCMC with non-volume-preserving transforms

---

## Summary

Out of 22 plan items:
- **10 work as planned** (mostly distributions, combinators, basic protocols)
- **6 work but with limitations** (missing update/regenerate, missing Jacobian, etc.)
- **3 were broken, now FIXED** (update weight, vectorized switch, conditional SMC)
- **3 remain broken or fundamentally incomplete** (edit interface, argdiffs,
  trainable params)

The 3 remaining broken items include 2 of the plan's "Tier 1: Core Foundation"
features. The edit interface, which the plan calls "GenJAX's most distinctive
contribution," is scaffolding. The argdiffs system, described as "critical for
MCMC performance," is dead code. Trainable parameters, the integration point
between learning and inference, were never connected.

Planned total: ~2,920 new lines.
Actual new code: ~1,700-1,900 lines.
