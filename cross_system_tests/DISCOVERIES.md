# Cross-System Verification Discoveries

Gaps, bugs, and enhancement opportunities found during test writing.
These should drive improvements to GenMLX — tests should NOT be adapted to
work around limitations.

## MLX Wrapper Gaps

### Full audit: node-mlx exposes but mlx.cljs does not wrap

**mlx.cljs wraps 145 functions. node-mlx (@frost-beta/mlx 0.0.1-dev, wrapping MLX ~0.31.x) exposes ~60 additional ops that are available but unwrapped.**

#### Math ops (available in node-mlx, missing from mlx.cljs)

| Op | node-mlx name | Priority | Notes |
|---|---|---|---|
| `arcsin` | `.arcsin` | Low | Inverse sine. Have `arccos` but not `arcsin`. |
| `arctan` | `.arctan` | Low | Inverse tangent. |
| `arctan2` | `.arctan2` | Medium | Two-argument arctangent. Needed for directional statistics (von Mises etc). |
| `arctanh` | `.arctanh` | Low | Inverse hyperbolic tangent. |
| `sinh` | `.sinh` | Low | Hyperbolic sine. |
| `cosh` | `.cosh` | Low | Hyperbolic cosine. |
| `rsqrt` | `.rsqrt` | Medium | Reciprocal sqrt (1/sqrt(x)). Single op, faster than `divide(1, sqrt(x))`. Used in normalization. |
| `degrees` | `.degrees` | Low | Radians to degrees. |
| `radians` | `.radians` | Low | Degrees to radians. |
| `nan_to_num` | `.nanToNum` | **High** | Replace NaN/Inf with finite values. Critical for numerical safety in inference loops. |

#### Reduction ops

| Op | node-mlx name | Priority | Notes |
|---|---|---|---|
| `logcumsumexp` | `.logcumsumexp` | **High** | Log-cumulative-sum-exp. Critical for SMC/particle filtering (numerically stable running log-weights). Currently would need manual implementation. |
| `cumprod` | `.cumprod` | Medium | Cumulative product. Useful for survival functions, stick-breaking. |
| `cummin` | `.cummin` | Low | Cumulative minimum. |
| `cummax` | `.cummax` | Low | Cumulative maximum. |
| `median` | `.median` | Medium | Median reduction. Useful for robust statistics, diagnostics. |

#### Linear algebra / tensor ops

| Op | node-mlx name | Priority | Notes |
|---|---|---|---|
| `trace` | `.trace` | **High** | Matrix trace. Used in Fisher info, matrix distributions, KL divergences. |
| `einsum` | `.einsum` | **High** | Einstein summation. Replaces many manual matmul/trace/contraction patterns. Could simplify MVN, Wishart, batched inference code significantly. |
| `diagonal` | `.diagonal` | Medium | Extract diagonal of matrix (more general than `diag`). |
| `tri` | `.tri` | Medium | Lower-triangular ones matrix. Masking, triangular indexing. |
| `tril` | `.tril` | Medium | Extract lower triangle. Covariance matrix construction. |
| `kron` | `.kron` | Low | Kronecker product. Advanced linear algebra. |
| `tensordot` | `.tensordot` | Medium | Generalized tensor contraction. |
| `addmm` | `.addmm` | Low | Fused add + matmul (alpha*A@B + beta*C). |
| `hadamard_transform` | `.hadamardTransform` | Low | Fast Hadamard transform. Used in some randomized algorithms. |

#### Linalg (CPU stream)

| Op | node-mlx name | Priority | Notes |
|---|---|---|---|
| `det` | N/A (compose) | **High** | Not in MLX. `prod(diag(chol(A)))^2`. Verified working in test runner. |
| `logdet` | N/A (compose) | **High** | Not in MLX. `2*sum(log(diag(chol(A))))`. Verified working. |
| `pinv` | `.pinv` | Medium | Pseudoinverse. Useful for least-squares, ill-conditioned systems. |
| `lu` / `luFactor` | `.lu` / `.luFactor` | Medium | LU decomposition. det for non-PD matrices. |
| `choleskyInv` | `.choleskyInv` | **High** | Invert directly from Cholesky factor. Avoids separate `inv(chol(A))` — faster for MVN. |
| `eig` / `eigvals` | `.eig` / `.eigvals` | Low | General eigendecomposition (not just symmetric). Added in MLX ~Nov 2025. |

#### Indexing / slicing

| Op | node-mlx name | Priority | Notes |
|---|---|---|---|
| Native `slice` | `.slice` | **High** | `slice(start, stop, strides)`. Current mlx.cljs `slice` uses `arange+take` (creates intermediate). Native is zero-copy. |
| `sliceUpdate` | `.sliceUpdate` | Medium | In-place slice assignment. Needed for scatter-like ops. |
| `put_along_axis` | `.putAlongAxis` | Medium | Scatter values into array along axis. |

#### Shape / creation

| Op | node-mlx name | Priority | Notes |
|---|---|---|---|
| `ones_like` / `zeros_like` | `.onesLike` / `.zerosLike` | Medium | Create array matching shape/dtype of existing. Avoids manual `(zeros (shape a) (dtype a))`. |
| `identity` | `.identity` | Low | Identity matrix (alias for `eye`). |
| `unflatten` | `.unflatten` | Low | Inverse of flatten. |
| `moveaxis` / `swapaxes` | `.moveaxis` / `.swapaxes` | Medium | Axis manipulation. More readable than `transpose` with axis lists. |

#### Logic / comparison

| Op | node-mlx name | Priority | Notes |
|---|---|---|---|
| `logical_and` | `.logicalAnd` | Medium | Element-wise AND. Currently would use `multiply` on bool arrays. |
| `logical_or` | `.logicalOr` | Medium | Element-wise OR. |
| `logical_not` | `.logicalNot` | Medium | Element-wise NOT. |
| `allclose` | `.allclose` | Medium | Check if arrays are element-wise close. Useful for testing/assertions. |
| `isclose` | `.isclose` | Medium | Element-wise closeness check. |
| `isneginf` / `isposinf` | `.isneginf` / `.isposinf` | Low | Check for negative/positive infinity specifically. |

#### I/O

| Op | node-mlx name | Priority | Notes |
|---|---|---|---|
| `save` / `load` | `.save` / `.load` | Medium | Array serialization. GenMLX has its own serialize.cljs but these are native MLX format. |
| `saveSafetensors` | `.saveSafetensors` | Low | SafeTensors format (HuggingFace ecosystem). |

### Performance concerns

| Discovery | Status | Notes |
|---|---|---|
| Linalg decompositions are CPU-only | **CONFIRMED (MLX limitation)** | `cholesky`, `solve`, `inv`, `qr`, `svd`, `eigh`, `eigvalsh` all require `cpu-stream`. MLX C++ core: "This op is not yet supported on the GPU." Issue ml-explore/mlx#1781 (Jan 2025) fixed the crash but did NOT add GPU support. Issue #2708 (SVD/eig GPU request) still open as of Nov 2025 with no timeline. `norm` and `cross` are the only GPU-capable linalg ops. CPU stream usage in mlx.cljs is correct and necessary. |
| `slice` uses `arange` + `take` | **OPEN** | Creates intermediate index array per slice. MLX has native `.slice(start, stop, strides)` that avoids this. Could matter in inner loops. |
| `index` creates intermediate scalar | **OPEN** | `(index a i)` creates `(scalar i int32)` + `take` each call. For repeated indexing in loops, pre-compute index arrays. |
| `array?` uses try/catch | **OPEN** | Type check via `(try (some? (.-shape x)) (catch ...))`. In hot paths, exception-based dispatch is slow. Consider checking for MLX array constructor or prototype instead. |
| Missing `rsqrt` | **OPEN** | `1/sqrt(x)` requires two ops. `rsqrt` is a single GPU kernel. Used in layer normalization, Gaussian density. |
| Missing `choleskyInv` | **OPEN** | `inv(chol(A))` requires two CPU linalg calls. `choleskyInv` is a single call. Performance win for MVN-heavy models. |

## Distribution Gaps

| Discovery | Status | Notes |
|---|---|---|

## GFI Gaps

| Discovery | Status | Notes |
|---|---|---|

## Test Coverage Gaps

| Discovery | Status | Notes |
|---|---|---|
| GenJAX has no `geometric` distribution | INFO | Not a GenMLX issue. Geometric verified only vs Gen.jl (all 3 test points agree). |
| GenJAX `binomial` dtype issue | INFO | GenJAX TFP wrapper rejects mixed int32/float32. Binomial verified only vs Gen.jl (all 3 agree). |

## IEEE 754 Compliance (verified)

All edge cases pass — MLX correctly handles:
- `log(-1)` → NaN
- `log(0)` → -Inf
- `sqrt(-1)` → NaN
- `exp(100)` → Inf (float32 overflow)
- `exp(-100)` → 0 (underflow)
- `1/0` → Inf
- `0/0` → NaN
- `lgamma(0)` → Inf (pole)
- `lgamma(-1)` → Inf (pole)
- `sigmoid(±100)` → 0 or 1 (no NaN)

## Precision Notes

| Observation | Details |
|---|---|
| float32 vs float64 gap | MLX ops accurate to ~1e-6 relative error for individual ops. `lgamma(1)` returns `-2.38e-7` instead of exact `0`. Acceptable for single ops but compounds over chains. |
| `exp(10)` relative error | 4.5e-8 — well within float32 spec. Absolute error 0.001 at magnitude 22026. |
| `lgamma(100)` relative error | 3e-7 — excellent for float32 at magnitude 359. |
| `det` via Cholesky | `det([[4,2],[2,3]])` = 7.9999995 vs expected 8.0. Relative error 6e-8. |
