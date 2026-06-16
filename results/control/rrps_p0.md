# RRPS P0 — heterogeneity gate + exact oracle + SBC (genmlx-b27i)

Task: multi-observation conjugate-structure search, k=12 (2 groups of 6, 4 train + 2 test per group). Structures C1 (single Gaussian, exact), C2 (two-group Gaussian, exact), Cflex (two-group Student-t(2), non-conjugate IS). Stream order [C1, C2, Cflex]; EASY true=C1 (arrives first), HARD true=Cflex (arrives last).

## [3] Exact oracle cross-check (verification)
- `score-exact == nn-marginal-closed` over multi-obs groups: |Δ| ~ 1e-6 (C1), ~3e-6 (C2).
- Independent high-N IS (20k) oracle agrees with exact C2 (|Δ| < 0.02).
- Held-out predictive oracle = log-evidence(full) − log-evidence(train), EXACT for C1/C2 (closed-form posterior-predictive), high-N IS for Cflex; test points (idx 4,5,10,11) never enter selection (out-of-sample by construction).

## [4] SBC over the scoring loop (Talts 2018)
Rank histogram on the conjugate posterior the exact evidence normalizes (R=2000, 20 bins): chi2=11.46 (df=19, 95% crit=30.144) — uniform, no U-shape. Certifies evidence calibration BEFORE any net-utility claim.

## [5] Pre-registered heterogeneity gate — VERDICT: **PASS**
Seeds=40 (20 easy / 20 hard, paired). Net-utility = held-out-LL(selected) − λ·compute; compute = :llm-tokens (121/proposal) + :particles (IS depth). Grid = ["n1/d0" "n2/d64" "n2/d512" "n3/d64" "n3/d512"].

λ | EASY-optimal | HARD-optimal | EASY:Δ(own−other) 95%CI | HARD:Δ(own−other) 95%CI | no fixed pt serves both?
---|---|---|---|---|---
0 | n2/d512 | n3/d512 | 0.003 [0.000, 0.009] | 4.781 [0.420, 12.387] | no
0.002 | n1/d0 | n3/d64 | 0.615 [0.612, 0.621] | 10.087 [2.452, 21.081] | **YES**
0.006 | n1/d0 | n3/d64 | 1.839 [1.836, 1.845] | 8.863 [1.406, 20.774] | **YES**

**Gate PASS** = some λ has divergent per-type optima with the paired 95% CI excluding 0 in BOTH directions (the EASY-optimal budget is significantly worse on HARD and vice-versa). The divergence is driven by the **#proposals (arrival) axis** (EASY-optimal n=1; HARD-optimal n=3 to reach the late-arriving true model).

## Honest finding: the scoring-depth axis is dominated
At n=3 the shallow (d64) and deep (d512) cells give ~identical net-utility: shallow IS under-rates the heavy-tailed model (verification [3d] shows ~4 nats of downward bias at d32), but the flip region where deepening changes the SELECTED model is narrow and low-stakes (depth-flips-selection iff the competition is a near-tie iff the flip barely changes the held-out decision). So the title-B 'two coupled knobs' reduce in practice to ONE robust knob (#proposals); a CI-lo>0 win must come from per-instance #proposals adaptivity — the same single-knob regime gdtq found ties (mean-only) in. This is the load-bearing risk going into P4.

## P0 verdict: PASS — title-B REACHABLE (intrinsic heterogeneity exists). Proceed to P1–P4; the title is EARNED only by a measured CI-lo>0 frontier-dominance win at P4, else revert to title-A.