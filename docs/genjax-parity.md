# GenJAX parity — keeper tables

North star (bean `genmlx-ecsi`): every GenMLX algorithm at least as fast as its
GenJAX/JAX equivalent on the RTX PRO 6000 (sm_120). This doc holds the
**distilled, pinned gap tables**; everything else — specs (source of truth),
runners, raw results, orchestrator — lives in the sibling repo
`~/mlx/genjax-parity` (see its README for the two-repo contract and fidelity
rules). The GenMLX side is `bench/genjax_parity.cljs` (bench tier: strictly
serial, card otherwise idle).

Method, both sides identically: shared JSON spec, float32, literal frozen data,
per-run pre-generated keys, device-synced timing, warmup reported separately
(never amortized), median/p10/p90 over 20 runs. Always-on weight-convention
guard: the generate weight must equal the hand-computed log-likelihood of the
constraints (caught genjax 1.0's per-component weight vectors for broadcast
sites; model uses one scalar site per observation on both sides).

## linreg_is — importance sampling (prior proposal), Bayesian linear regression, 10 obs

sm_120, measured 2026-07-28. Pins: genmlx `d4cca33` (src tree; bench file
landed the following commit), mlx-node `722bf55e`, genjax 1.0.13 (git,
femtomc/genjax), jax 0.7.2 + cuda13 wheels (nvcc 13.3.73 — same toolkit
version mlx-node builds against), driver 595.71.05. Raw results + spec in the
parity repo (`results/linreg_is.*.20260728-*.json`).

| N | GenJAX median | GenMLX median | steady gap | GenJAX warmup | GenMLX warmup |
|---|---|---|---|---|---|
| 1 | 0.018 ms | 1.76 ms | 98x | 326 ms | 4.0 ms |
| 100 | 0.019 ms | 1.77 ms | 96x | 376 ms | 6.7 ms |
| 3000 | 0.020 ms | 1.84 ms | 94x | 351 ms | 2.1 ms |
| 30000 | 0.019 ms | 2.08 ms | 110x | 356 ms | 1.5 ms |

logZ agreement at N=30000: −18.72 (GenJAX) vs −18.76 (GenMLX) — MC noise;
both sides estimate the same joint. (GenMLX N=3000 row reproduces the
cost-per-particle table's vgenerate ~1.6–1.8 ms host floor.)

**Reading the row.** Both sides are N-independent to 30000 — the GPU is idle
in both worlds; the contest is host floors. GenJAX's floor is one fused XLA
executable launch (~19 µs); GenMLX's is per-call lazy-graph construction in
nbb/SCI (~1.8 ms, 86% host graph-build per genmlx-sv3z). This is the
compile-fn verdict's predicted gap, now measured: **~95x steady-state** on a
microbenchmark-sized model. The other side of the ledger is warmup: JAX pays
~350 ms jit compile **per shape**, GenMLX ~nothing, so **GenMLX's total wall
is lower until ~173 same-shape calls**. Neither number is the whole story;
report both.

**Findings filed from this row:**
- The steady-state gap's cause is the already-diagnosed host graph-build floor
  (milestone genmlx-819v, compile-fn verdict + its three reopen triggers) —
  follow-up bean tracks whether/how to attack it (see genmlx-ecsi children).
- L3 analytical elimination did NOT trigger on the spec-driven model: the
  bench builds dist params from closure variables, not literals, and the
  unstripped generate weight stayed key-dependent. The benched path strips
  analytical anyway (the spec pins the ALGORITHM), but detection silently
  failing on closure params is a real conjugacy-coverage gap (cf. the
  genmlx-7zuq class).
