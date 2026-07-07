# RRPS second-family P0 — Gamma-Poisson changepoint structure search: heterogeneity gate + exact evidence + SBC (genmlx-l9nw, paper E2)

Metadata: git `1b0d65c41f7433099426405b5e4d631106df6188` · NVIDIA Thor (linux/arm64) · bun 1.3.14 / nbb 1.4.208 · seeds=45 (15/15/15 none/one/two) · SBC R=2000 · 2026-07-07T19:39:45.399Z

Task: count series T=24, test idx [3 7 11 15 19 23] (18 train / 6 test, interleaved). Structures = 0/1/2 changepoints on t∈{1..23}; per-segment rate λ ~ Gamma(2, 0.4) iid, y ~ Poisson(λ). Per-segment marginal evidence EXACT closed form (Pólya/negative-binomial). Structure prior: P(c)=1/3 × uniform over the C(23,c) fine-grid location sets (documented choice; coarse policies search subsets of the same hypothesis space under the same prior). Selection score = train evidence + log prior; held-out oracle = evidence(full) − evidence(train), EXACT for every structure — no IS anywhere in the gate loop (tighter than the first family's Cflex). Test points never enter selection: leakage-free by construction.

## [3] Exact-evidence cross-check (verification)
- Multi-obs Gamma-Poisson routes `:exact` through L3 conjugacy (single-rate + two-segment models).
- Host closed form == GenMLX `score-exact`: |Δ| = 3.21e-6, 1.44e-5 (float32 floor; asserted < 5e-5).
- Closed form == brute-force quadrature (200k-pt midpoint, float64): max |Δ| = 2.80e-13 over 4 small-T cases (asserted < 1e-5).
- Independent high-N IS (20k) agrees with exact: |Δ| = 0.0061 (asserted < 0.25).
- Stride mechanism (fixed :two instance, true cps [7 15]): best-on-s4-grid heldout -11.637 vs best-on-s1-grid -13.548 — fine-grid refinement worth -1.911 held-out nats; the location-resolution knob's intrinsic origin.

## [4] SBC over the conjugate scoring machinery (Talts 2018)
Rank histogram on the Gamma posterior the exact evidence normalizes (GenMLX gamma prior draw → GenMLX Poisson counts → batched GenMLX gamma draws at the closed-form Gamma(A0+Σy, B0+k) posterior; R=2000, 20 bins): chi2=14.06 (df=19, 95% crit=30.144) — uniform, no U-shape.

## [5] Pre-registered heterogeneity gate — VERDICT: **PASS**
Net-utility = held-out-LL(selected) − λ·compute; compute = per-segment marginal evaluations. Grid = ["c0" "c1/s4" "c1/s1" "c2/s4" "c2/s2" "c2/s1"] with seg-evals [1 11 47 41 188 806]. λ ∈ [0 0.003 0.01] (pre-registered). Criterion (pre-registered, the 3-type generalization of rrps_p0's): a cell SERVES a type iff it is that type's best-by-mean-NU or the paired deterministic-bootstrap 95% CI of [NU(type-best) − NU(cell)] includes 0; the gate passes at λ iff NO cell serves all three types; overall PASS ⇔ some λ passes.

### λ = 0 — served by: c1/s4, c1/s1, c2/s4, c2/s2, c2/s1

cell | NU :none | NU :one | NU :two | fails (CI-lo of type-best − cell)
---|---|---|---|---
c0 | -13.557 | -14.261 | -15.191 | one vs c2/s1 (lo=0.892); two vs c2/s2 (lo=0.026)
c1/s4 | -13.557 | -12.833 | -14.764 | —
c1/s1 | -13.557 | -12.223 | -14.924 | —
c2/s4 | -13.557 | -12.833 | -14.306 | —
c2/s2 | -13.557 | -12.659 | -14.291 | —
c2/s1 | -13.557 | -12.223 | -14.352 | —

Best per type: none=c2/s1, one=c2/s1, two=c2/s2. Headline pair (:none vs :two): none-best=c2/s1, two-best=c2/s2; :none Δ(own−other) = 0.000 [0.000, 0.000]; :two Δ(own−other) = 0.061 [-1.653, 1.797] → divergent optima with CI-lo>0 both ways? no

### λ = 0.003 — **no fixed cell serves all types**

cell | NU :none | NU :one | NU :two | fails (CI-lo of type-best − cell)
---|---|---|---|---
c0 | -13.560 | -14.264 | -15.194 | one vs c1/s1 (lo=0.766)
c1/s4 | -13.590 | -12.866 | -14.797 | none vs c0 (lo=0.030)
c1/s1 | -13.698 | -12.364 | -15.065 | none vs c0 (lo=0.138)
c2/s4 | -13.680 | -12.956 | -14.429 | none vs c0 (lo=0.120)
c2/s2 | -14.121 | -13.223 | -14.855 | none vs c0 (lo=0.561); one vs c1/s1 (lo=0.248); two vs c2/s4 (lo=0.201)
c2/s1 | -15.975 | -14.641 | -16.770 | none vs c0 (lo=2.415); one vs c1/s1 (lo=2.277); two vs c2/s4 (lo=0.631)

Best per type: none=c0, one=c1/s1, two=c2/s4. Headline pair (:none vs :two): none-best=c0, two-best=c2/s4; :none Δ(own−other) = 0.120 [0.120, 0.120]; :two Δ(own−other) = 0.766 [-0.144, 2.075] → divergent optima with CI-lo>0 both ways? no

### λ = 0.01 — **no fixed cell serves all types**

cell | NU :none | NU :one | NU :two | fails (CI-lo of type-best − cell)
---|---|---|---|---
c0 | -13.567 | -14.271 | -15.201 | one vs c1/s1 (lo=0.407)
c1/s4 | -13.667 | -12.943 | -14.874 | none vs c0 (lo=0.100)
c1/s1 | -14.027 | -12.693 | -15.394 | none vs c0 (lo=0.460)
c2/s4 | -13.967 | -13.243 | -14.716 | none vs c0 (lo=0.400)
c2/s2 | -15.437 | -14.539 | -16.171 | none vs c0 (lo=1.870); one vs c1/s1 (lo=1.291); two vs c2/s4 (lo=1.210)
c2/s1 | -21.617 | -20.283 | -22.412 | none vs c0 (lo=8.050); one vs c1/s1 (lo=7.590); two vs c2/s4 (lo=6.037)

Best per type: none=c0, one=c1/s1, two=c2/s4. Headline pair (:none vs :two): none-best=c0, two-best=c2/s4; :none Δ(own−other) = 0.400 [0.400, 0.400]; :two Δ(own−other) = 0.486 [-0.458, 1.705] → divergent optima with CI-lo>0 both ways? no

## Selection accuracy (diagnostic)

cell | type | count-acc | struct-acc
---|---|---|---
c0 | none | 1.00 | 1.00
c0 | one | 0.00 | 0.00
c0 | two | 0.00 | 0.00
c1/s4 | none | 1.00 | 1.00
c1/s4 | one | 0.53 | 0.13
c1/s4 | two | 0.00 | 0.00
c1/s1 | none | 1.00 | 1.00
c1/s1 | one | 0.67 | 0.20
c1/s1 | two | 0.00 | 0.00
c2/s4 | none | 1.00 | 1.00
c2/s4 | one | 0.53 | 0.13
c2/s4 | two | 0.53 | 0.00
c2/s2 | none | 1.00 | 1.00
c2/s2 | one | 0.53 | 0.13
c2/s2 | two | 0.53 | 0.00
c2/s1 | none | 1.00 | 1.00
c2/s1 | one | 0.67 | 0.20
c2/s1 | two | 0.53 | 0.00

## Honest findings (frozen full 45-seed run)

1. **The gate passes on the count-ladder (search-span) axis, with :one as the discriminating type.** At λ ∈ {0.003, 0.01}: c0 is :none-optimal and significantly under-serves :one (paired CI-lo = 0.766 / 0.407, mean gap 1.90 / 1.58 nats vs c1/s1), while every other cell significantly under-serves :none — on :none seeds all cells select the 0-changepoint structure (count-acc 1.00 everywhere; the Occam prior works), so their λ·Δcompute penalty is a near-constant paired delta with CI-lo > 0. Hence NO fixed cell serves all three types.
2. **The rrps_p0-style extreme-pair headline (:none vs :two) does NOT clear CI-lo>0 both ways** (:two Δ = 0.766 [−0.144, 2.075] at λ=0.003). Roughly half the :two instances are dip-contrast/low-rate and genuinely borderline under the c=2 Occam term (−ln(3·C(23,2)) ≈ −6.6 nats; count-acc at full search = 0.53 on :two), so :two's per-seed reward gaps are high-variance. Heterogeneity is real BOTH across types and WITHIN :two — headroom for per-instance adaptivity, but a variance burden the E2 controller sweep must budget for (more seeds and/or contrast stratification).
3. **The location-resolution (stride) axis is partially dominated, echoing rrps_p0's scoring-depth finding:** the fine c2/s1 cell is never cost-optimal at λ>0 (806 seg-evals swamp any refinement gain), and fine-grid location fitting can even hurt held-out LL (stride demo: −1.911 nats on the fixed instance; at λ=0 the :two optimum is c2/s2, not c2/s1). The live knobs going into the controller are the count ladder (cmax) plus a coarse-vs-mid stride choice — c2/s4 is :two's cost-adjusted optimum at both λ>0 — not fine-grid refinement.

## P0 verdict: **PASS — heterogeneity is REAL; the Gamma-Poisson changepoint family is GO for RRPS E2.** No fixed (search-span, location-stride) budget serves :none/:one/:two simultaneously. Proceed to the gated follow-ons (frozen per-(task,seed) proposer streams + search machinery, bean items 3-4).