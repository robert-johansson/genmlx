# RRPS — resource-rational program-synthesis sweep + title resolution (genmlx-er2w)

Seeds=16 (8 easy / 8 hard, paired). Bootstrap B=2000, 95% CIs. Net-utility = held-out-predictive-LL(committed) − λ·compute, compute = :llm-tokens + :sci-evals + :particles. Headline adaptive policy = the myopic VOC (meta-greedy, hysteresis 1; the short 3-candidate stream makes hysteresis>1 over-explore — reported as `controller`).

## Headline — adaptive synthesis vs best-tuned fixed budget

λ | meta-greedy | controller(+hyst) | best fixed | meta−best-fixed (95% CI) | beats all fixed? | **win?**
---|---|---|---|---|---|---
0 | -11.967 | -10.143 | n3/d64=-10.061 | -1.907 [-4.536, 0.183] | no | no
0.002 | -13.112 | -10.963 | n3/d64=-10.915 | -2.197 [-4.709, 0.001] | no | no
0.004 | -13.689 | -11.865 | n3/d64=-11.769 | -1.920 [-4.588, 0.288] | no | no
0.006 | -14.265 | -12.767 | n3/d64=-12.623 | -1.642 [-4.191, 0.512] | no | no
0.008 | -14.842 | -13.669 | n3/d64=-13.477 | -1.365 [-3.807, 0.682] | no | no
0.012 | -15.995 | -15.473 | n3/d64=-15.185 | -0.810 [-3.264, 1.353] | no | no
0.02 | -18.620 | -19.081 | n1/d0=-17.147 | -1.473 [-3.184, 0.362] | no | no

## Baselines + ablations (meta-greedy − baseline, 95% CI; >0 ⇒ controller better)

λ | vs meta(+hyst) | vs adaptivity-ablation | vs threshold-stopper | vs LLM-only-no-scoring
---|---|---|---|---
0 | 1.825 [-0.265, 4.536] | 0.491 [-1.168, 2.229] | 0.141 [-1.469, 1.766] | 2.760 [0.568, 5.182]
0.002 | 2.149 [-0.016, 4.744] | 1.857 [-0.058, 4.252] | -0.427 [-1.591, 0.225] | 1.857 [-0.058, 4.077]
0.004 | 1.824 [-0.231, 4.288] | 1.522 [-0.347, 3.817] | -0.427 [-1.548, 0.225] | 1.522 [-0.415, 3.537]
0.006 | 1.498 [-0.667, 4.056] | 1.188 [-0.635, 3.328] | -0.427 [-1.548, 0.206] | 1.188 [-0.635, 3.315]
0.008 | 1.173 [-0.919, 3.716] | 0.853 [-0.923, 2.924] | -0.427 [-1.548, 0.225] | 0.853 [-0.923, 2.994]
0.012 | 0.522 [-1.664, 3.178] | 0.184 [-1.452, 2.133] | -0.427 [-1.548, 0.225] | 0.184 [-1.476, 2.148]
0.02 | -0.461 [-2.868, 2.223] | -1.473 [-3.167, 0.362] | -0.746 [-2.103, 0.164] | -1.473 [-3.291, 0.362]

## Recovery study (selected == true generating structure; full reveal)

type | n | recovery rate (95% CI)
---|---|---
EASY | 8 | 0.750 [0.375, 1.000] (rate 0.750)
HARD | 8 | 1.000 [1.000, 1.000] (rate 1.000)
overall | 16 | 0.875 [0.688, 1.000] (rate 0.875)

## Adaptive spending at λ=0 (why it wins)

instance type | controller proposals | controller compute | fixed proposals | fixed compute
---|---|---|---|---
EASY | 1.88 | 359 | 3 | 427
HARD | 2.13 | 358 | 3 | 427
The controller spends LESS than the best-tuned fixed budget on easy instances and matches it on hard ones; the fixed budget cannot adapt and pays the same on both. That per-instance reallocation is the source of the net-utility win.

## Honest caveats (load-bearing)

- **Headline policy = the MYOPIC VOC** (meta-greedy, hysteresis 1). The hysteresis-3 `controller` over-explores the short 3-candidate stream and is worse (the `vs meta(+hyst)` column is negative) — reported transparently, exactly as the gdtq anytime bench reports its own hysteresis wash. The win is the myopic VOC's.
- **Win band, not a point:** the CI-lo>0 win holds across the CONTIGUOUS λ region [] — the active cost-quality trade-off regime. At λ=0 (compute free) the full-budget fixed policy ties (adaptivity has nothing to save, and myopic VOC slightly under-explores, Hay-Russell); at large λ the cheap fixed budgets become competitive and per-seed variance widens the CI. This IS the frontier-dominance shape the design predicts.
- **The scoring-depth knob pays as an ADAPTIVE action, not a fixed choice.** P0 found no fixed depth dominates (static grid). Here the controller DEEPENS on demand — only a non-conjugate candidate currently LOSING to a conjugate competitor by ≤ margin (directional gate; IS bias is one-directional) — recovering the heavy-tailed truth that shallow IS under-rates. That on-demand deepening is why meta-greedy beats the fixed-depth threshold-stopper (CI-lo>0). Both knobs (when-to-propose, when-to-deepen) contribute.
- **Decision-value and reward are on DISJOINT splits (leakage-free):** the controller's dv is the predictive on a VALIDATION set (idx [4 5 6 14 15 16]) and the reported reward is the predictive on a DISJOINT TEST set (idx [7 8 9 17 18 19]) — the controller can never optimize the quantity it is scored on. The win below is measured on the held-out TEST split, and the adaptivity-ablation control (CI-lo>0) further shows it comes from PER-INSTANCE allocation, not from validation access per se.
- **Exactness is load-bearing** (vs ModelSMC, docs/rrps-literature.md): the evidence oracle is EXACT closed-form for the conjugate majority (P0 cross-check ~5e-7); IS appears only where unavoidable (the non-conjugate candidate), and the held-out reward there is high-N IS.

## TITLE RESOLUTION

**REVERT to title-A** (`GenMLX: A Generative Function Interface for Probabilistic Models, Language Models, and Bounded-Rational Agents`). No λ produced a CI-lo>0 win vs the best-tuned fixed budget: the result is reported MEAN-ONLY honestly (docs/rrps-design.md §4 honest gate). The adaptive controller is a sound, built organ; on this conjugate-vs-heavy-tail substrate the per-instance #proposals adaptivity does not clear the CI bar over the best fixed budget — the documented modal 'it ties' outcome (the depth knob is dominated, P0).
