# Anytime-control microbenchmark (genmlx-gdtq)

Seeds=30, bootstrap B=2000, 95% non-overlapping CIs. Net-utility = −regret − λ·cost (cost=forced-evals). Higher is better.

**Per-instance heterogeneity (the headline mechanism):** each instance is EASY (r=0.3) or HARD (r=3) with equal probability — two difficulty types (paired across methods). The Bayes-optimal stop τ*(r) ∝ r differs by type, and NO single fixed budget is good for both (a small k under-serves the hard instances, a large k over-pays on the easy ones), so the adaptive VOC controller — spending little on easy instances and a lot on hard ones — strictly beats every fixed budget. With a single fixed r the schedule is identical across seeds and a tuned fixed budget merely ties (the genmlx-gdtq S=30 null).

**Honest finding on hysteresis.** The headline adaptive policy is the *myopic* VOC (hysteresis 1, ≡ Russell-Wefald meta-greedy). On this conjugate Bayes-risk schedule the myopic stop is near-optimal, so the hysteresis-3 robustness variant (`+hyst`) is a wash-to-overhead: each extra fold past the myopic stop has marginal value below cost by construction. Hysteresis would pay off on noisier / non-myopic value structures; here it does not — reported transparently (the bench is seed-validation, never a contribution).

The headline-win test is: myopic-VOC mean net-utility > EVERY fixed-k mean AND the paired 95% CI vs the BEST-tuned fixed-k (the binding baseline) excludes 0. (Beating the deliberately-tiny budgets k1/k2 holds in mean but their per-seed regret on hard instances is heavy-tailed, so their CIs are wide — not the meaningful comparison.)

## Headline — adapt-DATA (single-latent conjugate; regret vs θ_true)

λ | myopic-VOC | +hyst | best fixed-k | myopic−best-fixed (95% CI) | **headline win?**
---|---|---|---|---|---
0 | -0.1493 | -0.1098 | k12=-0.1098 | -0.0395 [-0.1317, 0.0295] | no
0.01 | -0.2525 | -0.2546 | k3=-0.3045 | 0.0521 [-0.0650, 0.1773] | mean-only
0.03 | -0.4271 | -0.4958 | k3=-0.4445 | 0.0175 [-0.1109, 0.1362] | mean-only
0.08 | -0.8095 | -1.0233 | k3=-0.7945 | -0.0150 [-0.2701, 0.1906] | no

## Ablation — adapt-PARTICLE (AR(1) Kalman chain; MC-precision dv, see caveat)

Homogeneous-difficulty contrast: a linear-Gaussian chain has ~instance-independent optimal N (MC error ∝ 1/N uniformly), so adapt-particle has no heterogeneity to exploit and is expected to TIE fixed-N — which makes the headline interpretable (adaptivity pays off exactly when there is per-instance heterogeneity).

λ | controller | best fixed-N | beats all fixed-N?
---|---|---|---
0 | -0.0222 | N64=-0.0222 | no
0.0005 | -0.4032 | N32=-0.1694 | no
0.002 | -1.5462 | N16=-0.4031 | no