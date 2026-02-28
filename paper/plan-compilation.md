# Paper 4: Loop Compilation for MCMC on Apple Silicon

**Title:** "Compiled Markov Chains: Metal Kernel Fusion for MCMC on Apple Silicon"

**Target venue:** Fold into Paper 1 (TOPML system paper) as an evaluation
highlight, OR submit standalone to a workshop/short paper venue.

**Alternative venues (if standalone):** PROBPROG workshop, LAFI (Languages for
Inference) workshop at POPL, NeurIPS workshop on Bayesian Deep Learning, or
as a short paper at AISTATS

**Status:** No draft. Implementation in `mcmc.cljs`. Formal treatment in
`formal/proofs/hmc-nuts.md`. Benchmarks in compiled_benchmark.cljs.

---

## Assessment for TOPML

**Not recommended as standalone TOPML submission.** Loop compilation is a
concrete optimization technique, but it's narrower than what a TOPML journal
paper expects. The contribution — "pre-generate noise, compile loop body,
dispatch once" — is clean but not deep enough for a 25+ page journal paper.

**Recommended path:** Integrate into Paper 1 as a substantial subsection of the
inference algorithms section (Section 3.3) and as a key evaluation experiment
(Section 6.3). The loop compilation speedup results strengthen Paper 1's
evaluation significantly.

If Paper 1 becomes too long, or if the compilation results are independently
strong (e.g., >10x speedup), consider a workshop paper.

---

## Content for Paper 1 Integration

### In Paper 1, Section 3.3 (Loop Compilation) — 1.5 pages

The overhead problem: per-step MCMC on GPU requires a separate Metal dispatch per
step. For short chains, dispatch overhead dominates.

The solution: pre-generate all randomness for K steps, compile the K-step loop
body via `mx/compile-fn`. The entire chain becomes a single Metal dispatch.

Three compiled algorithms: MH, MALA, HMC. Fused leapfrog for HMC reduces
gradient evaluations from 2L to L+1.

Interaction with adaptive tuning: warmup (per-step, adaptive) → sampling
(compiled, fixed parameters).

Correctness: compilation preserves the chain's stationary distribution because
`mx/compile-fn` preserves computational semantics with identical random inputs.

### In Paper 1, Section 6.3 (Loop Compilation Speedup) — 1.5 pages

**Experiments:**
1. Speedup vs chain length K (MH, MALA, HMC × K=10,50,100,500,1000)
2. Speedup vs parameter dimension D (K=100 × D=2,10,50,100,500)
3. Combined: compilation × vectorization (N parallel compiled chains)
4. Inference quality: compiled HMC on known posterior, verify ESS/R-hat match

---

## Standalone Workshop Paper (if needed)

If the results are strong enough for a standalone paper:

```
1. Introduction (1 page)
2. Background: MCMC on GPUs and MLX (1.5 pages)
3. Loop Compilation (2 pages)
   - Architecture, compiled MH/MALA/HMC, fused leapfrog
4. Correctness (1 page)
   - Distribution preservation, symplecticity
5. Evaluation (2 pages)
   - Speedup results, scaling analysis
6. Related Work (1 page)
7. Conclusion (0.5 pages)
```

Target: 8-10 pages. Workshop deadline dependent.

---

## Technical Content (unchanged from original plan)

### Three Compiled Algorithms

**Compiled MH:** Pre-generate K proposals + K uniforms. Loop body: propose,
score, accept/reject via `mx/where`.

**Compiled MALA:** Pre-generate K noise + K uniforms. Loop body: gradient,
Langevin proposal, asymmetric MH correction, accept/reject.

**Compiled HMC:** Pre-generate K momenta + K uniforms. Loop body: L fused
leapfrog steps, Hamiltonian difference, accept/reject. Fused leapfrog merges
adjacent half-kicks (L+1 gradient evals instead of 2L).

### Correctness

- Compilation preserves distribution (identical computation on identical inputs)
- Fused leapfrog preserves symplecticity (composition of shear maps)
- Dual averaging converges to target acceptance rate

---

## Effort Estimate

**If folded into Paper 1:** 0 additional days (covered by Paper 1's effort)

**If standalone workshop paper:** ~4 days
- Write paper: 2 days
- Run benchmarks: 1.5 days
- Polish: 0.5 days
