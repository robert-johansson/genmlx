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
