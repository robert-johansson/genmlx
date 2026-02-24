# GenJAX vs GenMLX Feature Comparison

> Date: 2026-02-25
> GenJAX: Python/JAX, POPL 2026 artifact (~10k lines)
> GenMLX: ClojureScript/MLX on Apple Silicon (~14k lines)

## GenJAX features GenMLX lacks

### High priority (core missing capabilities)

1. **ADEV (Automatic Differentiation of Expected Values)**
   - `@expectation` decorator wrapping ADEV programs
   - `expectation.grad_estimate(params)` for unbiased gradient of E[f(X)]
   - Per-distribution gradient strategies:
     - REINFORCE: `flip_reinforce`, `normal_reinforce`, `geometric_reinforce`, `uniform_reinforce`, `multivariate_normal_reinforce`
     - Reparameterization: `normal_reparam`, `uniform_reparam`, `multivariate_normal_reparam`
     - Enumeration: `flip_enum`, `flip_enum_parallel`, `categorical_enum_parallel`
     - Measure-valued derivatives: `flip_mvd`
   - `Dual` number system (forward-mode AD with primal+tangent)
   - `ADEVPrimitive` base class for extensible stochastic primitives with custom JVP
   - Handles `jax.lax.cond` inside ADEV programs
   - **This is GenJAX's primary research contribution (POPL 2026)**

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

9. **Address collision detection with file/line info**
   - Raises `ValueError` with precise location when the same address is used twice
   - GenMLX does not detect duplicate addresses

10. **`MCMCResult` with integrated diagnostics**
    - `chain` returns structured result with stacked traces, accepts, R-hat (bulk/tail ESS)
    - GenMLX has R-hat/ESS utilities but not integrated into the chain runner

11. **Variational families**
    - `mean_field_normal_family`, `full_covariance_normal_family`
    - Pre-built variational families for VI

12. **Visualization module (`genjax.viz`)**
    - `raincloud`, `horizontal_raincloud` (Matplotlib)
    - Publication-quality standardized color palettes and figure sizes

13. **Runtime type checking**
    - `beartype` + `jaxtyping` for runtime type and shape validation
    - GenMLX has no runtime validation

### Low priority / ecosystem

14. **Equinox/Flax/Haiku NN integration** — JAX's NN ecosystem works transparently with GenJAX via Pytree compatibility. GenMLX has its own `nn.cljs` but smaller scope.

15. **SPMD / multi-device** — `axis_name` and `spmd_axis_name` in Vmap for distributed computation.

16. **XLA compilation (GPU/TPU)** — CUDA/TPU via XLA. GenMLX is Apple Silicon only.

17. **HTML rendering of Pytrees** — Interactive HTML via Penzai (`pz.Struct`).

## GenMLX features GenJAX lacks

1. **NUTS** — No-U-Turn Sampler (neither Gen.jl nor GenJAX ship NUTS)
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
13. **Loop-compiled MCMC chains** — fuse entire chains into single Metal dispatches (10-21x over GenJAX pre-compiled MH)
14. **`CustomGradientGF`** — `IHasArgumentGrads` protocol for custom gradient gen functions
15. **MLX unified memory** — zero-copy CPU/GPU sharing on Apple Silicon
16. **27 distributions** vs GenJAX's ~20 (von Mises, piecewise uniform, discrete uniform, truncated normal, delta, etc.)

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
| VI / ELBO | Yes (ADEV) | Yes (MLX autograd) |
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

**ADEV is the biggest gap.** It's the core theoretical contribution of the GenJAX POPL paper and enables sound, unbiased gradient estimation through arbitrary mixes of continuous and discrete distributions — something GenMLX fundamentally cannot do today. The PJAX staging infrastructure is the next most impactful missing feature, enabling Gen functions to compose freely with JAX transforms (JIT, vmap, grad).

The Vmap combinator gap is now closed — GenMLX has both composable `vmap-gf` (full GFI combinator with nesting, per-element selection, batched fast path) and shape-based batching (`vsimulate`/`vgenerate`) for dispatch amortization.

GenMLX compensates with broader inference algorithms (NUTS, SMCP3, Involutive MCMC, etc.), richer combinators (Mix, Recurse, Mask, etc.), and significantly faster execution on Apple Silicon (10-34x across benchmarks).
