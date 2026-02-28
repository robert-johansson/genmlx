# Formal Specification of λ_MLX

> Complete formal foundation for GenMLX, extending λ_GEN (POPL 2026) with
> handler types, full GFI operations, broadcasting correctness, and
> algebraic effect semantics.

## Relationship to LAMBDA_MLX.md

`LAMBDA_MLX.md` (project root) is the high-level overview and motivation
document. It introduces λ_GEN, explains why a λ_MLX formalization is
valuable, and sketches the key ideas. The files in this directory contain
the complete, rigorous formal specifications and proofs.

## Reading Order

1. **`calculus.md`** — Full type grammar, term grammar, and typing rules
2. **`semantics.md`** — QBS denotational semantics and handler transition semantics
3. **`transformations.md`** — Program transformations for all GFI operations
4. **`proofs/correctness.md`** — Correctness of generate and update
5. **`proofs/handler-soundness.md`** — Handler soundness by induction
6. **`proofs/broadcasting.md`** — Broadcasting correctness theorem and corollary
7. **`proofs/combinators.md`** — Combinator compositionality
8. **`proofs/edit-duality.md`** — Edit/backward duality for reversible kernels
9. **`proofs/diff-update.md`** — Diff-aware update correctness
10. **`proofs/kernel-composition.md`** — Markov kernel stationarity and composition
11. **`proofs/adev.md`** — ADEV gradient estimation (reparam + REINFORCE)
12. **`proofs/deterministic-gf.md`** — CustomGradientGF and NeuralNetGF correctness
13. **`proofs/hmc-nuts.md`** — Adaptive HMC, NUTS, and symplectic integration
14. **`proofs/vi.md`** — Variational inference (ELBO, IWELBO, VIMCO, programmable VI)
15. **`proofs/smcp3.md`** — SMCP3 weight correctness and log-ML estimation
16. **`proofs/contracts-linkage.md`** — Contract-theorem mapping and verification power

## TODO Item Mapping

| File | TODO Items | Description |
|------|-----------|-------------|
| `calculus.md` | 10.4 | Full λ_MLX calculus (types, terms, typing rules) |
| `semantics.md` | 10.5 | QBS denotational semantics |
| `transformations.md` | 10.6 | Program transformations for all GFI operations |
| `proofs/correctness.md` | 10.7 | Proposition: generate & update correctness |
| `proofs/broadcasting.md` | 10.8, 10.9 | Broadcasting correctness theorem & corollary |
| `proofs/handler-soundness.md` | 10.10 | Handler soundness by induction |
| `proofs/combinators.md` | 10.11 | Combinator compositionality |
| `proofs/edit-duality.md` | 10.12 | Edit/backward duality |
| `proofs/diff-update.md` | 10.13 | Diff-aware update correctness |
| `proofs/kernel-composition.md` | 10.14 | Kernel stationarity and composition |
| `proofs/adev.md` | 10.15 | ADEV gradient estimation correctness |
| `proofs/deterministic-gf.md` | 10.16 | Deterministic GF wrapper correctness |
| `proofs/hmc-nuts.md` | 10.17 | HMC/NUTS detailed balance and adaptation |
| `proofs/vi.md` | 10.18 | Variational inference objectives |
| `proofs/smcp3.md` | 10.19 | SMCP3 weight and log-ML correctness |
| `proofs/contracts-linkage.md` | 10.20 | Contract-theorem mapping |

## Notation Conventions

| Symbol | Meaning |
|--------|---------|
| ⟦τ⟧ | Denotation of type τ (QBS interpretation) |
| ⟦t⟧ | Denotation of term t |
| 𝒫(X) | Probability measures on space X |
| 𝒫_≪(X) | Measures absolutely continuous w.r.t. stock measure |
| ν_η | Stock measure on ground type η |
| dμ/dν | Radon-Nikodym derivative (density) |
| γ ⊕ γ' | Trace type concatenation (monoid operation) |
| R_τ | Logical relation at type τ (for broadcasting correctness) |
| H(σ, τ) | Handler computation type (state monad) |
| δ_x | Dirac delta measure at x |
| ⊗ | Product measure |
| Σ, Π | Summation, product (also used for dependent types) |
| dom(f) | Domain of function/map f |
| Sel(γ) | Selection over trace type γ |
| Δ | Diff type (NoChange, UnknownChange, etc.) |
| K | Markov kernel K : Γ × Key → Γ |
| π | Target distribution |
| H(q, p) | Hamiltonian (potential + kinetic energy) |
| U(q) | Potential energy = -log π(q) |
| K(p) | Kinetic energy = ½p^T M^{-1} p |
| ELBO | Evidence lower bound = E_q[log p - log q] |
| IWELBO_K | K-sample importance-weighted ELBO |
| ∇_θ | Gradient with respect to parameters θ |
| stop_gradient(·) | Gradient barrier (zero in backward pass) |
| σ_adev | ADEV handler state (extends σ_sim with reinforce-lp) |

All scores are written in **multiplicative notation** (densities as products)
in the formal development, matching the paper. The implementation uses
**log-space** (scores as sums of log-densities), noted where relevant.

## References

- Becker et al., "Probabilistic Programming with Vectorized Programmable
  Inference," POPL 2026 — Section 3, Figures 10-14, Proposition 3.1,
  Theorem 3.3, Corollary 3.4
- LAMBDA_MLX.md — Overview and motivation
- `src/genmlx/handler.cljs` — Handler state transitions
- `src/genmlx/dynamic.cljs` — DynamicGF implementation
- `src/genmlx/edit.cljs` — Edit interface
- `src/genmlx/diff.cljs` — Diff types
- `src/genmlx/combinators.cljs` — Combinator implementations
- `src/genmlx/inference/kernel.cljs` — Kernel composition
- `src/genmlx/inference/adev.cljs` — ADEV gradient estimation
- `src/genmlx/inference/mcmc.cljs` — HMC, NUTS, MALA
- `src/genmlx/inference/vi.cljs` — Variational inference
- `src/genmlx/inference/smcp3.cljs` — SMCP3
- `src/genmlx/custom_gradient.cljs` — Custom gradient GFs
- `src/genmlx/nn.cljs` — Neural network GFs
- `src/genmlx/contracts.cljs` — GFI contracts
- Hastings 1970 — MH sampling
- Neal 2011 — HMC
- Hoffman & Gelman 2014 — NUTS, dual averaging
- Burda et al. 2015 — IWELBO
- Lew et al. 2023 — ADEV, SMCP3
