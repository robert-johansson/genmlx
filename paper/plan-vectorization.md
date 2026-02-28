# Paper 3: Vectorized Inference via Shape-Based Batching

**Title:** "Vectorized Inference Without vmap: Shape Broadcasting as a Batching Strategy for Probabilistic Programs"

**Target venue:** [ACM Transactions on Probabilistic Machine Learning (TOPML)](https://dl.acm.org/journal/topml) — regular submission

**Alternative venues:** POPL, PLDI (if the PL theory angle is stronger than the
ML systems angle)

**Status:** No draft. Formal proof exists in `formal/proofs/broadcasting.md`
(481 lines). System description in `paper/genmlx.md` Section 5.

---

## Venue Fit

TOPML scope includes Monte Carlo sampling and scalability. Vectorized particle
inference is a core technique for scaling SMC and importance sampling on GPUs.
The TOPML [PP special issue](https://dl.acm.org/pb-assets/static_journal_pages/topml/pdf/TOPML_CfP_SI_Probabilistic_Programming-1704743488800.pdf)
listed "approximate and exact inference algorithms" and "automatic
differentiation" — vectorized inference fits the first, and the broadcasting
technique is relevant to any differentiable PPL.

**TOPML format:** Double-anonymous, ACM template, no strict page limit.

---

## Thesis

> Vectorized particle inference in probabilistic programs can be achieved by
> changing the *shape* of sampled values from scalar to [N]-array and relying
> on broadcasting, rather than by transforming programs with vmap. This approach
> is simpler (no program transformation), works with arbitrary host-language
> control flow, and achieves comparable speedups. We prove broadcasting
> correctness via logical relations and demonstrate the approach on Apple
> Silicon via MLX.

---

## Why This Paper Works for TOPML

1. **Directly relevant to TOPML's audience.** Anyone running importance sampling
   or SMC on a GPU cares about vectorization. This paper provides a formally
   correct, simpler alternative to vmap.

2. **Formal correctness proof.** The logical relations proof is more rigorous
   than GenJAX's vectorization treatment (Awad et al. POPL 2026). TOPML values
   theoretical foundations.

3. **Practical technique.** 20 distributions with native batch sampling, shape-
   agnostic handlers, vectorized switch for discrete structure. This is
   immediately usable.

4. **Clean comparison.** Same interface (GFI), same algorithms, different
   vectorization strategy. The vmap-vs-broadcasting comparison is objective.

---

## Assessment: Standalone or Merge with Paper 1?

**Option A: Standalone TOPML paper (~25 pages).**
Focused on vectorization. Includes formal proof, implementation, and evaluation.
Strongest as independent contribution.

**Option B: Merged into Paper 1 as expanded Section 4.**
Paper 1 already has a vectorization section. Expanding it to 5-6 pages with the
formal proof and evaluation could be sufficient for TOPML.

**Recommendation: Option A if the benchmarks are strong enough to carry a paper.
Option B if the speedup story is incremental rather than dramatic.**

Run the benchmarks first. If vectorization provides >10x speedup consistently
across models and particle counts, it's a standalone paper. If it's 3-5x, fold
it into Paper 1.

---

## Technical Content

### The Broadcasting Approach (3 components)

**Component 1: Batch Sampling**
```
dist-sample-n : Distribution × Key × N → [N]-array
```
20 of 27 distributions provide native batch sampling. 7 use sequential fallback.

**Component 2: Shape-Agnostic Handlers**

Handler state transitions never inspect value shapes. Score accumulation via
`mx/add` broadcasts: scalar + [N] = [N]. Verified by code inspection — no
shape-dependent conditionals in handler.cljs.

**Component 3: Broadcasting Arithmetic**

All `dist-log-prob` implementations use element-wise MLX operations. [N]-shaped
input → [N]-shaped output automatically.

### The Correctness Proof

Logical relations R_τ encode "[N]-shaped values correctly represent N independent
scalar values." Four lemmas (batch sampling independence, log-prob broadcasting,
score accumulation, handler agnosticism) combine into the main theorem:

**Theorem:** Batched simulate/generate/update/regenerate = N independent scalar
executions (in distribution).

**Proof:** Induction on number of trace sites.

### Comparison with vmap

| Property | vmap (GenJAX) | Broadcasting (GenMLX) |
|----------|--------------|----------------------|
| Program transformation | Required | None |
| Host-language control flow | Must be JAX-traceable | Unrestricted |
| Data-dependent branching | Not supported | Not supported* |
| Combinator support | Native | Via vectorized switch |
| Splice support | Via traced sub-programs | Not in batched mode |
| Implementation complexity | High (tracing machinery) | Low (one multimethod) |
| Formal correctness | Awad et al. POPL 2026 | Logical relations |

*Discrete structure handled by vectorized switch: execute all branches, combine
with `mx/where` based on [N]-shaped index arrays.

---

## Paper Outline (target: 25-30 pages, TOPML journal format)

```
1. Introduction (2.5 pages)
   - Particle methods need vectorization for GPU efficiency
   - vmap: the standard approach and its limitations
   - Our approach: broadcasting as a simpler alternative
   - Contributions: formal proof, implementation, evaluation

2. Background (3 pages)
   2.1 Particle-Based Inference
       - Importance sampling, SMC, importance weighting
       - The per-particle overhead problem on GPUs
   2.2 The Generative Function Interface
       - simulate, generate, update, regenerate
       - Handler-based execution model
   2.3 vmap-Based Vectorization
       - How GenJAX/JAX uses jax.vmap
       - Traceability constraints

3. Shape-Based Vectorization (4 pages)
   3.1 The Key Insight
       - Change array shapes, not programs
       - MLX broadcasting handles all arithmetic
   3.2 Implementation
       - dist-sample-n: 20 native batch samplers
       - Batched handler transitions (unchanged from scalar)
       - VectorizedTrace data structure
   3.3 Handling Discrete Structure
       - Vectorized switch: execute all branches, mask-select results
       - Supports mixture models, clustering

4. Formal Correctness (5 pages)
   4.1 Logical Relations R_τ
       - Definition for all types (base, product, record, distribution, GF)
   4.2 Lemma 1: Batch Sampling Independence
   4.3 Lemma 2: Log-Prob Broadcasting
   4.4 Lemma 3: Score Accumulation Broadcasting
   4.5 Lemma 4: Handler Shape-Agnosticism
   4.6 Main Theorem: Broadcasting Correctness
   4.7 Corollary: Vectorized Inference Commutes

5. Implementation on Apple Silicon (2 pages)
   5.1 MLX Unified Memory Advantage
   5.2 Distribution Catalog (20 native, 7 fallback)
   5.3 Batched Splice with Nested Scoping

6. Evaluation (4 pages)
   6.1 Vectorization Speedup vs Particle Count
       - N = 1, 10, 50, 100, 500, 1000
       - 3 models: simple Gaussian, linear regression, hierarchical
   6.2 Statistical Equivalence
       - Batched vs scalar: posterior moments match within tolerance
   6.3 Expressiveness Comparison
       - Programs that work with broadcasting but not vmap
       - Programs that work with vmap but not broadcasting
   6.4 Vectorized Switch: Mixture Model
   6.5 Scaling Analysis

7. Related Work (2 pages)
   7.1 Vectorized Probabilistic Programming
       - Awad et al. 2025 (compositional vectorization via vmap, POPL 2026)
   7.2 Array Broadcasting in ML
   7.3 Logical Relations for Program Equivalence

8. Conclusion (1 page)
```

---

## Key Evaluation Experiments

| Experiment | Purpose | Data source |
|-----------|---------|-------------|
| Speedup vs N | Quantify overhead amortization | vectorized_benchmark.cljs (extend) |
| Statistical equivalence | Verify correctness empirically | vectorized_test.cljs |
| Expressiveness: doseq model | Show control flow advantage | New model (write) |
| Expressiveness: splice model | Show vmap advantage | New model (write) |
| Vectorized switch | Mixture model batching | combinators_test.cljs |

---

## Effort Estimate

- Write introduction and background: 1.5 days
- Write vectorization approach section: 1.5 days
- Adapt broadcasting.md proof for paper format: 2.5 days
- Write implementation section: 1 day
- Run benchmarks and write evaluation: 3 days
- Related work and conclusion: 1 day
- Anonymize and format: 0.5 days
- Polish: 1 day

**Total: ~12 days**
