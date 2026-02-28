# GenJAX vs GenMLX Feature Comparison

> Date: 2026-02-28 (updated)
> GenJAX: Python/JAX, POPL 2026 artifact (~10k lines)
> GenMLX: ClojureScript/MLX on Apple Silicon (~14k lines)
>
> **Note:** Performance claims in this document have not been independently
> verified with a rigorous cross-platform benchmark. A proper comparison
> against Gen.jl and GenJAX is needed before trusting any speedup numbers.

## GenJAX features GenMLX lacks

### High priority (core missing capabilities)

1. ~~**ADEV (Automatic Differentiation of Expected Values)**~~ **PARTIALLY CLOSED** — GenMLX now has ADEV with reparameterization and REINFORCE gradient strategies, vectorized GPU execution (~50-60x over sequential), compiled optimization loops, and baseline variance reduction. What remains different:
   - GenJAX has per-distribution gradient strategy primitives (`flip_reinforce`, `normal_reparam`, `flip_enum`, `flip_mvd`, etc.) — GenMLX uses a simpler `has-reparam?` dispatch
   - GenJAX has enumeration and measure-valued derivative strategies — GenMLX only has REINFORCE and reparameterization
   - GenJAX has `Dual` number system (forward-mode AD) and `ADEVPrimitive` extensibility — GenMLX uses MLX reverse-mode AD
   - GenJAX handles `jax.lax.cond` inside ADEV programs — GenMLX does not support control flow in ADEV
   - **ADEV is GenJAX's primary research contribution (POPL 2026)**

2. ~~**`Vmap` as a first-class GFI combinator**~~ **CLOSED** — GenMLX now has `vmap-gf` / `repeat-gf` implementing full GFI (simulate, generate, update, regenerate, assess, propose, project). Supports `in-axes`, scalar constraint broadcast, nested vmap-of-vmap, per-element selection, batched fast path (~7x), and splice in batched mode via combinator fallback.

3. **PJAX / `seed` / `modular_vmap` staging infrastructure**
   - `sample_p` custom JAX primitive integrating into Jaxpr
   - `seed` transformation eliminates `sample_p`, injects explicit PRNG keys
   - `modular_vmap` extends `jax.vmap` to handle probabilistic primitives with automatic PRNG key splitting
   - Gen functions become JAX-transformable (JIT, vmap, grad compose freely)
   - GenMLX gen functions cannot be passed as black boxes to transforms

### Medium priority (useful missing features)

4. **State inspector (`@state` / `save` / `namespace`)**
   - JAX interpreter that collects tagged intermediate values from any JAX function
   - MCMC algorithms internally call `save(accept=...)`, `save(log_alpha=...)`
   - `chain` automatically collects and returns acceptance rates, R-hat, ESS
   - Works under `jax.jit` / `jax.vmap` / `jax.scan`

5. **`tfp_distribution()` bridge**
   - One-liner access to TensorFlow Probability's entire distribution catalog
   - GenMLX requires writing a `defdist` for each new distribution

6. **State space model extras**
   - `discrete_hmm`, `kalman_filter`, `kalman_smoother`, FFBS
   - Linear Gaussian SSM with exact inference
   - `DiscreteHMMTrace`, `LinearGaussianTrace` specialized trace types

7. **`trace.verify()` / `Fixed` infrastructure**
   - `Fixed[T]` / `NotFixedException` tracks which choices were constrained
   - Runtime checking that all random choices were properly constrained during inference

8. **`change` operation for SMC**
   - Explicit function for translating particle choices between different model parameterizations during SMC tempering

9. ~~**Address collision detection with file/line info**~~ **PARTIALLY CLOSED** — GenMLX now has `validate-gen-fn` which detects duplicate addresses via a validation handler. Does not yet include file/line info in the error message (GenJAX raises `ValueError` with precise location).

10. **`MCMCResult` with integrated diagnostics**
    - `chain` returns structured result with stacked traces, accepts, R-hat (bulk/tail ESS)
    - GenMLX has R-hat/ESS utilities but not integrated into the chain runner

11. **Variational families**
    - `mean_field_normal_family`, `full_covariance_normal_family`
    - Pre-built variational families for VI

12. **Visualization module (`genjax.viz`)**
    - `raincloud`, `horizontal_raincloud` (Matplotlib)
    - Publication-quality standardized color palettes and figure sizes

13. ~~**Runtime type checking**~~ **PARTIALLY CLOSED** — GenMLX now has parameter validation on all distributions (sigma>0, lo<hi, etc.) and helpful error messages for common mistakes. Does not have shape-level validation comparable to `beartype` + `jaxtyping`.

### Low priority / ecosystem

14. **Equinox/Flax/Haiku NN integration** — JAX's NN ecosystem works transparently with GenJAX via Pytree compatibility. GenMLX has its own `nn.cljs` but smaller scope.

15. **SPMD / multi-device** — `axis_name` and `spmd_axis_name` in Vmap for distributed computation.

16. **XLA compilation (GPU/TPU)** — CUDA/TPU via XLA. GenMLX is Apple Silicon only.

17. **HTML rendering of Pytrees** — Interactive HTML via Penzai (`pz.Struct`).

## GenMLX features GenJAX lacks

1. **NUTS** — No-U-Turn Sampler with adaptive step-size and mass matrix (neither Gen.jl nor GenJAX ship NUTS)
2. **Involutive MCMC** — with Jacobian computation
3. **Elliptical Slice Sampling**
4. **MAP Optimization** — vectorized with N random restarts simultaneously
5. **SMCP3** — Sequential Monte Carlo with Probabilistic Program Proposals
6. **Wake-Sleep learning**
7. **Full kernel algebra** — `cycle-kernels`, `mix-kernels`, `repeat-kernel` (GenJAX only has `chain`)
8. **Mix, Recurse, Mask, Unfold combinators** — GenJAX only has `Vmap`, `Scan`, `Cond`
9. **Contramap / Dimap** — argument/return value transformation combinators
10. **`defdist-transform` macro** — derived distributions via deterministic transforms
11. **`mixture` constructor** — first-class mixture distribution
12. **`propose` / `project` GFI operations** — not exposed in GenJAX
13. **Loop-compiled MCMC chains** — fuse entire chains into single Metal dispatches
14. **`CustomGradientGF`** — `IHasArgumentGrads` protocol for custom gradient gen functions
15. **MLX unified memory** — zero-copy CPU/GPU sharing on Apple Silicon
16. **27 distributions** vs GenJAX's ~20 (von Mises, piecewise uniform, discrete uniform, truncated normal, delta, etc.)
17. **GFI contract verification** — `verify-gfi-contracts` with 11 measure-theoretic contracts, tested on 13 canonical models (575 checks)
18. **Adaptive HMC/NUTS** — dual averaging step-size tuning + diagonal mass matrix estimation

## Core GFI protocol comparison

| Operation | GenJAX | GenMLX |
|-----------|--------|--------|
| `simulate` | Yes | Yes |
| `generate` | Yes | Yes |
| `assess` | Yes | Yes |
| `update` | Yes | Yes |
| `regenerate` | Yes | Yes |
| `propose` | No | Yes |
| `project` | No | Yes |
| `filter` | Yes (first-class GFI method) | Via selection algebra |

## Inference algorithm comparison

| Algorithm | GenJAX | GenMLX |
|-----------|--------|--------|
| MH (selection-based) | Yes | Yes |
| Compiled MH | No | Yes |
| MALA | Yes | Yes |
| HMC | Yes | Yes |
| NUTS | No | Yes |
| Enumerative Gibbs | Yes | Yes |
| Custom proposal MH | Yes | Yes |
| Involutive MCMC | No | Yes |
| Elliptical Slice | No | Yes |
| MAP | No | Yes |
| Importance Sampling | Yes | Yes |
| SMC | Yes | Yes |
| SMCP3 | No | Yes |
| ADEV | Yes (Jaxpr + modular_vmap) | Yes (shape-based batching) |
| VI / ELBO | Yes | Yes |
| Wake-Sleep | No | Yes |
| Kernel combinators | `chain` only | Full algebra |

## Combinator comparison

| Combinator | GenJAX | GenMLX |
|------------|--------|--------|
| Vmap (composable GFI) | Yes | Yes (`vmap-gf` / `repeat-gf` + shape-based batching) |
| Scan | Yes | Yes |
| Switch / Cond | Yes | Yes |
| Map | No (via Vmap) | Yes |
| Unfold | No | Yes |
| Mask | No | Yes |
| Mix | No | Yes |
| Recurse | No | Yes |
| Contramap / Dimap | No | Yes |

## Bottom line

**PJAX staging infrastructure is the biggest remaining architectural gap.** It enables Gen functions to compose freely with JAX transforms (JIT, vmap, grad) — something GenMLX cannot do because SCI-interpreted functions are opaque to MLX's tracer.

The ADEV gap is partially closed — GenMLX now has working ADEV with reparameterization, REINFORCE, vectorized GPU execution, and compiled optimization. GenJAX's ADEV is more sophisticated (enumeration, measure-valued derivatives, forward-mode Dual numbers), but for the common case of continuous latent variables, both systems work.

GenMLX compensates with broader inference algorithms (NUTS, SMCP3, Involutive MCMC, Elliptical Slice, MAP, etc.), richer combinators (Mix, Recurse, Mask, etc.), and GFI contract verification.

**TODO:** A rigorous cross-platform benchmark comparing GenMLX, GenJAX, and Gen.jl on identical models is needed before making any performance claims.
