# Cross-System Verification Discoveries

Gaps, bugs, and enhancement opportunities found during test writing.
These should drive improvements to GenMLX — tests should NOT be adapted to
work around limitations.

## MLX Wrapper Gaps

### Missing ops

| Discovery | Status | Notes |
|---|---|---|
| `det` (determinant) not in mlx.cljs | **OPEN** | MLX has no native det. Implement via `prod(diag(chol(A)))^2`. Verified in runner. Needed for MVN, Wishart. |
| `logdet` (log-determinant) not in mlx.cljs | **OPEN** | `2 * sum(log(diag(chol(A))))` — numerically stable. Verified in runner. Critical for MVN log-prob. |
| `trace` (matrix trace) not in mlx.cljs | **OPEN** | MLX core exposes `.trace` — not wrapped. `trace(A) = sum(diag(A))`. Used in matrix distributions, Fisher info. |
| `tri` (triangular matrix) not in mlx.cljs | **OPEN** | MLX core exposes `.tri` — not wrapped. Returns lower-triangular ones matrix. Useful for masking. |
| `tril` (lower triangle) not in mlx.cljs | **OPEN** | MLX core exposes `.tril` — not wrapped. Extract lower triangle of matrix. |
| `einsum` not in mlx.cljs | **OPEN** | MLX core exposes `.einsum` — not wrapped. Powerful for tensor contractions, batched matmuls, traces. Could simplify MVN, Wishart code. |
| `pinv` (pseudoinverse) not in mlx.cljs | **OPEN** | MLX linalg exposes `.pinv` — not wrapped. Useful for least-squares, ill-conditioned systems. |
| `lu` / `luFactor` not in mlx.cljs | **OPEN** | MLX linalg exposes LU decomposition — not wrapped. Alternative to Cholesky for det (works for non-PD matrices). |
| `choleskyInv` not in mlx.cljs | **OPEN** | MLX linalg exposes `.choleskyInv` — not wrapped. Directly inverts from Cholesky factor (avoids separate `inv` call). |
| Native `slice` not used | **OPEN** | MLX core exposes `.slice` and `.sliceUpdate` — not wrapped. Current `slice` in mlx.cljs uses `arange` + `take` which creates intermediate index arrays. Native slice avoids this overhead. |

### Performance concerns

| Discovery | Status | Notes |
|---|---|---|
| Linalg decompositions are CPU-only | **CONFIRMED** | `cholesky`, `solve`, `inv`, `qr`, `svd`, `eigh`, `eigvalsh` all require `cpu-stream`. MLX error: "This op is not yet supported on the GPU." This is an MLX framework limitation, not a GenMLX bug. `norm` and `cross` DO work on GPU. CPU stream usage in mlx.cljs is correct. |
| `slice` uses `arange` + `take` | **OPEN** | Creates intermediate index array per slice. MLX has native `.slice(start, stop, strides)` that avoids this. Could matter in inner loops. |
| `index` creates intermediate scalar | **OPEN** | `(index a i)` creates `(scalar i int32)` + `take` each call. For repeated indexing in loops, pre-compute index arrays. |
| `array?` uses try/catch | **OPEN** | Type check via `(try (some? (.-shape x)) (catch ...))`. In hot paths, exception-based dispatch is slow. Consider checking for MLX array constructor or prototype instead. |

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
