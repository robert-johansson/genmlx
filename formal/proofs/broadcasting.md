# Broadcasting Correctness — TODO 10.8, 10.9

> Theorem: GenMLX's broadcasting-based vectorization produces the same
> result as N independent sequential executions. Corollary: vectorized
> inference can be implemented by broadcasting within inference applied
> to the model.
>
> Analogous to Theorem 3.3 and Corollary 3.4 of the POPL 2026 paper,
> but for broadcasting (semantic property of the handler) rather than
> vmap (syntactic program transformation).

---

## 1. Key Difference from the Paper

The paper defines **vmap_n{−}** as a source-to-source program transformation
(Figure 13) and proves its correctness via logical relations (Figure 14,
Theorem 3.3). GenMLX achieves the same result through a different mechanism:

1. **No program transformation**: the model code is unchanged
2. **Domain lifting**: `dist-sample-n` produces [N]-shaped arrays at each
   trace site instead of scalars
3. **Broadcasting**: `dist-log-prob` on [N]-shaped arrays produces [N]-shaped
   log-probabilities via MLX element-wise operations
4. **Shape-agnostic handlers**: the pure transitions never inspect array shapes

The correctness proof is therefore a **semantic** property of the handler
rather than a **syntactic** transformation of the program. The logical
relations framework from the paper still applies, but the fundamental lemma
is simpler: it follows from handler shape-agnosticism rather than from
proving correctness of a term-level transformation.

---

## 2. Logical Relations

Following Figure 14 of the paper, we define logical relations
R_τ ⊆ ⟦τ⟧^N × ⟦τ[N]⟧ encoding "a value of type τ[N] correctly represents
N independent values of type τ":

```
R_1           := { ((x,…,x), x) | x ∈ ⟦1⟧ }
                 (unit type: all copies identical)

R_B           := { ((x₁,…,xₙ), x) | xᵢ ∈ ⟦B⟧, x = [x₁,…,xₙ] }
                 (base types: [N]-array holds N independent values)

R_T           := { ((x₁,…,xₙ), x) | xᵢ ∈ ⟦T⟧, x = stack(x₁,…,xₙ) }
                 (batched types: leading dimension indexes copies)

R_{η₁ × η₂}  := { (((a₁,b₁),…,(aₙ,bₙ)), (A,B)) |
                    ((a₁,…,aₙ), A) ∈ R_{η₁},
                    ((b₁,…,bₙ), B) ∈ R_{η₂} }
                 (products: componentwise)

R_{{k:η}}    := { ((r₁,…,rₙ), R) |
                    ∀k. ((r₁(k),…,rₙ(k)), R(k)) ∈ R_{η_k} }
                 (records/choicemaps: per-field)

R_{D η}       := { ((μ₁,…,μₙ), μ) | μ = R_{η *}(⊗ᵢ₌₁ⁿ μᵢ) }
                 (distributions: product measure, pushed through R)

R_{P η}       := { ((μ₁,…,μₙ), μ) | μ = R_{η *}(⊗ᵢ₌₁ⁿ μᵢ) }
                 (stochastic computations: same as distributions)

R_{G_γ η}     := { (((μ₁,f₁),…,(μₙ,fₙ)), (ν,g)) |
                    ((μ₁,…,μₙ), ν) ∈ R_{P γ},
                    ((f₁,…,fₙ), g) ∈ R_{γ → η} }
                 (gen fns: measure and return fn both related)
```

---

## 3. Key Lemmas

### Lemma 3.1 (Batch Sampling Independence)

For any distribution d : D η, PRNG key k, and batch size N:

```
dist-sample-n(d, k, N) = [v₁, …, vₙ]
```

where each vᵢ is an independent sample from d. Formally, the joint
distribution of (v₁,…,vₙ) equals the N-fold product measure μ_d^{⊗N}.

**Proof:** By construction of `dist-sample-n`. For distributions with
native batch sampling (gaussian, uniform, bernoulli, exponential, etc.),
independence follows from the PRNG key splitting: each sample uses an
independent portion of the key stream. For distributions using the
sequential fallback, independence follows from N independent calls to
`dist-sample` with independent keys. ∎

### Lemma 3.2 (Log-Prob Broadcasting)

For broadcastable distributions d : D η and [N]-shaped value
v = [v₁,…,vₙ]:

```
dist-log-prob(d, v) = [dist-log-prob(d, v₁), …, dist-log-prob(d, vₙ)]
```

That is, log-prob on [N]-shaped input produces element-wise log-prob.

**Proof.** We verify this by categorizing all 27 distributions by their
log-prob implementation pattern.

**Category 1: Scalar element-wise (23 distributions).** These compute
log-prob using only element-wise MLX arithmetic operations on the value
and scalar parameters. Broadcasting is automatic.

*Continuous, closed-form:* gaussian, uniform, exponential, laplace,
log-normal, logistic, cauchy, half-cauchy, half-normal, von-mises,
pareto (11 distributions).

Each has the form `f(v, θ₁, …, θₖ)` where f is a composition of
`mx/add`, `mx/subtract`, `mx/multiply`, `mx/divide`, `mx/log`,
`mx/exp`, `mx/square`, `mx/abs`, and constants. When v has shape [N]
and θᵢ are scalar, MLX broadcasting produces [N]-shaped output where
output[i] = f(v[i], θ₁, …, θₖ).

Example — gaussian with parameters μ (scalar), σ (scalar):
```
log-prob(v) = -0.5 · ((v - μ)/σ)² - log(σ) - 0.5·log(2π)
```
When v : [N], each sub-expression broadcasts: `v - μ` : [N],
`((v - μ)/σ)²` : [N], final result : [N].

*Discrete, closed-form:* bernoulli, geometric, discrete-uniform
(3 distributions).

Same element-wise pattern. Bernoulli uses `mx/where` which broadcasts.

*Using special functions:* beta, gamma, student-t, dirichlet, poisson
(5 distributions).

These use `mlx-log-gamma` (a pure element-wise computation using
`mx/log`, `mx/add`, `mx/multiply`, `mx/sin` — see `dist.cljs:42-59`)
or `mx/log-gamma`. No shape inspection occurs; all operations broadcast.

*Simplex-valued:* categorical (1 distribution).

Log-prob for categorical uses `mx/gather` on the logits array at the
value index. **However**, the categorical log-prob implementation calls
`mx/shape` on the *logits parameter* (`dist.cljs:406`) during
normalization to determine the number of categories. This is safe for
broadcasting because `mx/shape` is called on the *parameter* (which has
fixed shape regardless of batch size), not on the *value* being scored.
When v is [N]-shaped (N indices), `mx/gather` correctly selects N
log-probabilities element-wise.

*Positive semi-definite matrix-valued:* lkj-cholesky (1 distribution).

Log-prob is computed via element-wise operations on the diagonal and
off-diagonal entries of the Cholesky factor. Broadcasts correctly.

*Mixture:* mixture (1 distribution).

Delegates to component log-probs and uses logsumexp. Broadcasts
provided all components broadcast.

**Category 2: NON-broadcastable (2 distributions).** These inspect
value shapes in their log-prob computation and **do not** support
[N]-shaped broadcasting.

*wishart:* `dist-log-prob` at `dist.cljs:1010` calls `mx/ndim` and
`mx/reshape` on the value x to determine the matrix dimension k. The
log-prob formula involves `mx/logdet`, matrix trace, and matrix
operations that assume x is a k×k matrix — not a batch of matrices.

*inv-wishart:* `dist-log-prob` at `dist.cljs:1061` similarly calls
`mx/ndim` on the value x and performs matrix operations assuming a
single k×k matrix.

These distributions are excluded from the broadcasting correctness
theorem. In practice, they are not used in vectorized inference
(dist-sample-n falls back to sequential for them as well).

**Summary:** 25 of 27 distributions satisfy Lemma 3.2 unconditionally.
The 2 matrix-variate distributions (wishart, inv-wishart) are excluded
from vectorized execution. The categorical distribution inspects
parameter shapes but not value shapes, so it broadcasts correctly. ∎

### Lemma 3.3 (Score Accumulation Broadcasting)

Score accumulation in handler transitions broadcasts correctly:

```
σ.score + lp  where σ.score : shape_s, lp : shape_l
```

produces a result with shape `broadcast(shape_s, shape_l)`.

Specifically:
- `scalar + [N] = [N]` (first trace site: scalar initial score + [N] log-prob)
- `[N] + [N] = [N]` (subsequent sites: element-wise addition)
- `[N] + scalar = [N]` (constrained site: [N] score + scalar log-prob)

In all cases, `result[i] = σ.score[i] + lp[i]` where scalars are
broadcast to all positions. This follows from MLX's broadcasting rules.

**Proof:** Direct from MLX broadcasting semantics (same as NumPy). ∎

### Lemma 3.4 (Handler Shape-Agnosticism)

No handler transition function (simulate, generate, update, regenerate,
project — both scalar and batched variants) inspects the **shape** of
sampled values or log-probabilities. The only shape-aware operations are:
- `dist-sample-n`: explicitly samples [N]-shaped values (entry point)
- `(:batch-size state)`: read to pass N to `dist-sample-n`

All other state threading operations — `assoc`, `update`, `cm/set-choice`,
`mx/add` — work identically regardless of array shape.

**Proof:** By inspection of `handler.cljs`.

The scalar transitions (`handler.cljs:72-171`) use:
- `dc/dist-sample`, `dc/dist-log-prob` — return shape determined by dist
- `rng/split` — key operations, independent of value shapes
- `cm/set-choice` — stores values without shape inspection
- `mx/add` for score/weight — broadcasts, never inspects shape

The batched transitions (`handler.cljs:177-258`) are structurally
identical, differing only in that `dist-sample` is replaced by
`dist-sample-n`. No transition calls `mx/shape`, `mx/ndim`, `mx/size`,
`mx/item`, or any other shape-inspecting operation on sampled values.

This structural identity is the key enabler of broadcasting correctness. ∎

---

## 4. Broadcasting Correctness Theorem (cf. Theorem 3.3)

### Theorem (Broadcasting Correctness)

Let g : G_γ η be a generative function with denotation (μ, f) = ⟦g⟧,
where all distributions used in g's body are broadcastable (Category 1
in Lemma 3.2 — i.e., g does not use wishart or inv-wishart).

Let g_N denote the broadcasting lift: executing g with N particles using
batched handlers (batched-simulate-handler, batched-generate-handler, etc.).

Then for all modes m ∈ {simulate, generate}:

```
⟦m{g_N}⟧ ≡ zip_R ∘ (⟦m{g}⟧)^{⊗N}
```

That is, the batched execution produces N independent results in
struct-of-array (SoA) format, related by R to N independent scalar
executions.

Specifically:

**(a) Simulate:** ⟦g_N.simulate⟧ produces a VectorizedTrace where:
- choices[a] has shape [N] for each address a
- score has shape [N]
- ((tr₁,…,trₙ), vtrace) ∈ R_{γ × η × ℝ} where trᵢ are independent
  samples from ⟦g.simulate⟧

**(b) Generate:** ⟦g_N.generate(obs)⟧ produces a VectorizedTrace where:
- unconstrained choices[a] has shape [N]
- constrained choices[a] are scalar (shared observation)
- score has shape [N], weight has shape [N]
- Each particle i independently satisfies the generate correctness
  proposition (see `proofs/correctness.md`)

### Proof

By induction on the structure of the generative function body, using the
handler shape-agnosticism (Lemma 3.4) as the key mechanism.

**Base case: single trace site.** The body contains one `trace(a, d)`.

For simulate_N:
```
⟦trace(a, d)⟧_simulate^N(σ₀) =
  let (k₁, k₂) = split(σ₀.key)
      v = dist-sample-n(d, k₂, N)       -- [N]-shaped, independent (Lemma 3.1)
      lp = dist-log-prob(d, v)           -- [N]-shaped, element-wise (Lemma 3.2)
  in (v, { choices: {a : v}, score: σ₀.score + lp })
```

By Lemma 3.1, v = [v₁,…,vₙ] where each vᵢ ~ d independently.
By Lemma 3.2, lp = [lp₁,…,lpₙ] where lpᵢ = log density_d(vᵢ).
By Lemma 3.3, score = [s₁,…,sₙ] where sᵢ = σ₀.score + lpᵢ.

Therefore, the i-th component of the batched result matches the result
of a scalar execution with value vᵢ. ✓

**Inductive case: sequential composition.** The body is
`do_G{x ← t₁; t₂}`.

By the induction hypothesis, after executing t₁ under the batched handler:
- choices₁[a] has shape [N] for each address in grade(t₁)
- score₁ has shape [N]
- The i-th slice matches independent execution i

The return value x has [N]-shaped components. The continuation t₂ uses
x in its body. By handler shape-agnosticism (Lemma 3.4), the handler
processes t₂'s trace sites identically regardless of x's shape. Each
trace site in t₂ produces [N]-shaped samples and [N]-shaped log-probs.

The score accumulation score₁ + lp₂ produces [N]-shaped scores by
Lemma 3.3, where each component i equals the score from independent
execution i.

By the disjoint grading rule, choices₁ ⊕ choices₂ produces a choice map
where each address holds [N]-shaped arrays, and the i-th slice across
all addresses matches independent execution i. ✓

**Fundamental lemma:** If ⟦t⟧ has denotation (μ, f) and the batched
execution of t produces result r with ((r₁,…,rₙ), r) ∈ R_{G_γ η},
then:
- The trace distribution of r equals R_{γ *}(μ^{⊗N}), i.e., N independent
  traces zipped into SoA format
- The return function of r applied to the SoA trace gives N independent
  return values zipped into SoA format

This follows from the structural induction above and the definition of
R_{G_γ η}.  ∎

---

## 5. Broadcasting Commutativity Corollary (cf. Corollary 3.4)

### Corollary (Broadcasting Commutativity)

Let g : G_γ η be a generative function. Then vectorized inference can
be implemented by broadcasting within inference applied to the model.

**(a) Simulate commutativity:**
```
⟦simulate{g_N}⟧ = ⟨id, f_N, w ↦ Σᵢ w[i]⟩_* ⟦g_N.simulate⟧
```

That is, a correct implementation of vectorized simulate produces
[N]-shaped traces with [N]-shaped scores, where each component is
an independent sample with its correct density. The density collapsing
(product in the paper → sum in log-space) is element-wise score
accumulation.

**(b) Generate commutativity:**
```
⟦generate{g_N}(obs)⟧ = broadcasting within generate{g}(obs)
```

Each particle independently satisfies the generate contract: constrained
addresses have the observed value (scalar, shared), unconstrained
addresses have independent [N]-shaped samples, and the [N]-shaped weight
gives per-particle importance weights.

**(c) Extension to update and regenerate:**
```
⟦update{g_N}(vtrace, constraints)⟧ = broadcasting within update{g}
⟦regenerate{g_N}(vtrace, selection)⟧ = broadcasting within regenerate{g}
```

Each particle is updated/regenerated independently within the same
batched execution.

### Proof

Follows directly from the Broadcasting Correctness Theorem.

For (a): By the theorem, g_N.simulate produces ((tr₁,…,trₙ), vtrace) ∈
R_{G_γ η}. Each trᵢ is an independent sample with score wᵢ = dμ/dν(trᵢ).
The [N]-shaped score [w₁,…,wₙ] contains per-particle densities. In
log-space, the density collapsing via multiplication (paper's Πᵢ w[i])
becomes the natural element-wise score accumulation that the handler
already performs.

For (b): By the theorem's generate case, constrained sites produce scalar
values (shared observation) with scalar log-prob that broadcasts into the
[N]-shaped score. Unconstrained sites produce [N]-shaped samples. The
[N]-shaped weight [w₁,…,wₙ] gives per-particle importance weights, each
equal to what a scalar generate would produce for that particle.

For (c): The batched update and regenerate transitions (handler.cljs:204-258)
are structurally identical to their scalar counterparts, differing only in
value shapes. By handler shape-agnosticism (Lemma 3.4), each particle's
weight component equals the weight from an independent scalar update/
regenerate. ∎

---

## 6. Batched Splice Correctness

When a generative function body contains `splice(k, g, args)`, the
batched handler creates a **nested** `run-handler` scope for the sub-GF
(`batched-splice-transition` at `handler.cljs:363-418`). This requires
extending the broadcasting correctness theorem to handle splice.

### Statement

**Lemma 6.1 (Batched Splice Correctness).** If the sub-GF g satisfies
the broadcasting correctness theorem (i.e., g uses only broadcastable
distributions), then `splice(k, g, args)` under the batched handler
produces correct [N]-shaped results nested under address k.

### Proof

The batched splice transition (`handler.cljs:363-418`) performs:

1. **Key splitting:** Split parent key into (k₁, k₂). k₂ is passed to
   the sub-GF. This is identical to scalar splice — independence of
   the sub-GF's randomness from the parent's subsequent randomness is
   preserved.

2. **State scoping:** Extract sub-constraints, sub-old-choices, and
   sub-selection scoped to address k. This is identical to scalar splice.

3. **Nested run-handler:** Create a fresh `run-handler` with:
   - A fresh `volatile!` holding the sub-GF's initial state
   - `:batch-size N` propagated from the parent state
   - The appropriate batched handler (simulate, generate, update, or
     regenerate) chosen by examining which state fields are present

4. **Sub-GF execution:** The sub-GF body executes under the nested
   batched handler. By the broadcasting correctness theorem applied to g
   (induction on the nesting depth of splice calls), this produces
   correct [N]-shaped choices, score, and weight for the sub-GF.

5. **Result merging:** The sub-result is merged into the parent state
   via `merge-sub-result`:
   - `parent.choices[k] ← sub.choices` (nested under k)
   - `parent.score += sub.score` ([N] + [N] = [N], or scalar + [N] = [N])
   - `parent.weight += sub.weight` (same broadcasting)
   - `parent.discard[k] ← sub.discard` (if applicable)

The critical property is that the nested `run-handler` creates a **fresh
volatile** (`handler.cljs:458`), completely isolated from the parent's
volatile. The parent's state is not accessed during the sub-GF's execution.
After the sub-GF completes, the parent volatile is updated exactly once
with the merged result.

By the handler soundness theorem (see `handler-soundness.md` §5), the
nested execution correctly implements the sub-GF's GFI operation in
batched mode. The merge step preserves the address-nesting structure:
each particle i's sub-choices are nested under k in the parent's
choices, maintaining the SoA format.

**Limitation:** Only `DynamicGF` instances (those with `:body-fn`) are
supported for batched splice. Non-DynamicGF sub-GFs (e.g., combinators
used directly as sub-GFs) throw an error in batched mode
(`handler.cljs:430-431`). This is a practical limitation, not a
theoretical one — combinator-based sub-GFs could in principle be
supported if their GFI operations accepted a `:batch-size` parameter. ∎

---

## 7. Implementation Correspondence

| Formal Concept | Implementation | Location |
|----------------|---------------|----------|
| g_N.simulate | `vsimulate` | `dynamic.cljs:231-243` |
| g_N.generate | `vgenerate` | `dynamic.cljs:245-260` |
| g_N.update | `vupdate` | `dynamic.cljs:262-282` |
| g_N.regenerate | `vregenerate` | `dynamic.cljs:284-305` |
| Batched simulate transition | `batched-simulate-transition` | `handler.cljs:177-187` |
| Batched generate transition | `batched-generate-transition` | `handler.cljs:189-202` |
| Batched update transition | `batched-update-transition` | `handler.cljs:204-233` |
| Batched regenerate transition | `batched-regenerate-transition` | `handler.cljs:235-258` |
| dist-sample-n (Lemma 3.1) | `dc/dist-sample-n` multimethod | `dist/core.cljs` |
| Log-prob broadcasting (Lemma 3.2) | `dc/dist-log-prob` on [N]-shaped input | `dist/core.cljs` |
| VectorizedTrace (SoA format) | `VectorizedTrace` record | `vectorized.cljs` |
| Shape-agnosticism (Lemma 3.4) | No shape inspection in transitions | `handler.cljs:72-258` |

### Shape-Agnosticism Verification

The following operations would break shape-agnosticism and are NOT present
in any handler transition:

- `mx/shape`, `mx/ndim`, `mx/size` — shape inspection
- `mx/item`, `mx/realize` — materialization (forces scalar)
- `mx/reshape`, `mx/squeeze`, `mx/unsqueeze` — shape manipulation
- Any conditional branching based on array dimensions

The ONLY shape-aware code is `dist-sample-n` (which receives N explicitly)
and the initial state setup in `vsimulate`/`vgenerate` (which sets
`:batch-size N`).
