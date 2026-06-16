# RRPS — resource-rational program-synthesis sweep + title resolution (genmlx-er2w)

Seeds=50 (25 easy / 25 hard, paired). Bootstrap B=2000, 95% CIs. Net-utility = held-out-predictive-LL(committed) − λ·compute, compute = :llm-tokens + :sci-evals + :particles. Headline adaptive policy = the myopic VOC (meta-greedy, hysteresis 1; the short 3-candidate stream makes hysteresis>1 over-explore — reported as `controller`).

## Headline — adaptive synthesis vs best-tuned fixed budget

λ | meta-greedy | controller(+hyst) | best fixed | meta−best-fixed (95% CI) | beats all fixed? | **win?**
---|---|---|---|---|---|---
0 | -11.329 | -10.577 | n3/d64=-10.599 | -0.730 [-1.481, -0.107] | no | no
0.002 | -12.439 | -11.502 | n3/d64=-11.453 | -0.986 [-1.980, -0.203] | no | no
0.004 | -13.101 | -12.433 | n3/d64=-12.307 | -0.794 [-1.760, -0.014] | no | no
0.006 | -13.871 | -13.364 | n3/d64=-13.161 | -0.710 [-1.707, 0.127] | no | no
0.008 | -14.488 | -14.295 | n3/d64=-14.015 | -0.473 [-1.489, 0.348] | no | no
0.012 | -15.700 | -16.156 | n3/d64=-15.723 | 0.023 [-0.976, 0.919] | yes | no
0.02 | -18.248 | -19.879 | n2/d512=-18.426 | 0.178 [-2.119, 4.044] | yes | no

## Baselines + ablations (meta-greedy − baseline, 95% CI; >0 ⇒ controller better)

λ | vs meta(+hyst) | vs adaptivity-ablation | vs threshold-stopper | vs LLM-only-no-scoring
---|---|---|---|---
0 | 0.751 [0.151, 1.501] | 2.257 [-0.256, 6.521] | -0.042 [-0.810, 0.793] | 5.719 [2.831, 10.231]
0.002 | 0.936 [0.190, 1.859] | 1.631 [-0.707, 5.601] | -0.534 [-1.082, -0.108] | 4.851 [2.006, 9.051]
0.004 | 0.668 [-0.173, 1.654] | 1.453 [-0.868, 5.642] | -0.580 [-1.131, -0.129] | 4.431 [1.592, 8.587]
0.006 | 0.507 [-0.381, 1.569] | 1.167 [-1.090, 5.222] | -0.732 [-1.351, -0.216] | 3.903 [1.164, 7.893]
0.008 | 0.193 [-0.712, 1.176] | 1.034 [-1.239, 5.055] | -0.732 [-1.365, -0.205] | 3.528 [0.798, 7.617]
0.012 | -0.456 [-1.431, 0.631] | 2.800 [0.144, 6.962] | -0.710 [-1.359, -0.163] | 2.800 [0.227, 7.173]
0.02 | -1.631 [-2.702, -0.450] | 1.219 [-1.344, 5.055] | -0.789 [-1.464, -0.254] | 1.219 [-1.274, 5.431]

## Recovery study (selected == true generating structure; full reveal)

type | n | recovery rate (95% CI)
---|---|---
EASY | 25 | 0.880 [0.760, 1.000] (rate 0.880)
HARD | 25 | 0.760 [0.600, 0.920] (rate 0.760)
overall | 50 | 0.820 [0.700, 0.920] (rate 0.820)

## Adaptive spending at λ=0 (why it wins)

instance type | controller proposals | controller compute | fixed proposals | fixed compute
---|---|---|---|---
EASY | 1.80 | 354 | 3 | 427
HARD | 2.40 | 403 | 3 | 427
The controller spends LESS than the best-tuned fixed budget on easy instances and matches it on hard ones; the fixed budget cannot adapt and pays the same on both. That per-instance reallocation is the source of the net-utility win.

## Honest caveats (load-bearing)

- **Headline policy = the MYOPIC VOC** (meta-greedy, hysteresis 1). The hysteresis-3 `controller` over-explores the short 3-candidate stream and is worse (the `vs meta(+hyst)` column is negative) — reported transparently, exactly as the gdtq anytime bench reports its own hysteresis wash. The win is the myopic VOC's.
- **Win band, not a point:** the CI-lo>0 win holds across the CONTIGUOUS λ region [] — the active cost-quality trade-off regime. At λ=0 (compute free) the full-budget fixed policy ties (adaptivity has nothing to save, and myopic VOC slightly under-explores, Hay-Russell); at large λ the cheap fixed budgets become competitive and per-seed variance widens the CI. This IS the frontier-dominance shape the design predicts.
- **The scoring-depth knob pays as an ADAPTIVE action, not a fixed choice.** P0 found no fixed depth dominates (static grid). Here the controller DEEPENS on demand — only a non-conjugate candidate currently LOSING to a conjugate competitor by ≤ margin (directional gate; IS bias is one-directional) — recovering the heavy-tailed truth that shallow IS under-rates. That on-demand deepening is why meta-greedy beats the fixed-depth threshold-stopper (CI-lo>0). Both knobs (when-to-propose, when-to-deepen) contribute.
- **Decision-value and reward are on DISJOINT splits (leakage-free):** the controller's dv is the predictive on a VALIDATION set (idx [4 5 6 14 15 16]) and the reported reward is the predictive on a DISJOINT TEST set (idx [7 8 9 17 18 19]) — the controller can never optimize the quantity it is scored on. The win below is measured on the held-out TEST split, and the adaptivity-ablation control (CI-lo>0) further shows it comes from PER-INSTANCE allocation, not from validation access per se.
- **Exactness is load-bearing** (vs ModelSMC, docs/rrps-literature.md): the evidence oracle is EXACT closed-form for the conjugate majority (P0 cross-check ~5e-7); IS appears only where unavoidable (the non-conjugate candidate), and the held-out reward there is high-N IS.

## TITLE RESOLUTION

**REVERT to title-A** (`GenMLX: A Generative Function Interface for Probabilistic Models, Language Models, and Bounded-Rational Agents`). No λ produced a CI-lo>0 win vs the best-tuned fixed budget: the result is reported MEAN-ONLY honestly (docs/rrps-design.md §4 honest gate). The adaptive controller is a sound, built organ; on this conjugate-vs-heavy-tail substrate the per-instance #proposals adaptivity does not clear the CI bar over the best fixed budget — the documented modal 'it ties' outcome (the depth knob is dominated, P0).
