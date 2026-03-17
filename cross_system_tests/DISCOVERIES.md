# Cross-System Verification Discoveries

Gaps, bugs, and enhancement opportunities found during test writing.
These should drive improvements to GenMLX — tests should NOT be adapted to
work around limitations.

## MLX Wrapper Gaps

| Discovery | Status | Notes |
|---|---|---|
| `det` (determinant) not in mlx.cljs | **OPEN** | MLX linalg has no native det. Verified via `prod(diag(chol(A)))^2` in runner — should be added to mlx.cljs. Needed for MVN, Wishart. |
| `logdet` (log-determinant) not in mlx.cljs | **OPEN** | `2 * sum(log(diag(chol(A))))` — numerically stable. Verified in runner. Critical for MVN log-prob. Should be added to mlx.cljs. |

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
