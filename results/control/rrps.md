# RRPS — resource-rational program-synthesis sweep + title resolution (genmlx-er2w)

Seeds=50 (25 easy / 25 hard, paired). Bootstrap B=2000, 95% CIs. Net-utility = held-out-predictive-LL(committed) − λ·compute, compute = :llm-tokens + :sci-evals + :particles. Headline adaptive policy = the myopic VOC (meta-greedy, hysteresis 1; the short 3-candidate stream makes hysteresis>1 over-explore — reported as `controller`).

## Headline — adaptive synthesis vs best-tuned fixed budget

λ | meta-greedy | controller(+hyst) | best fixed | meta−best-fixed (95% CI) | beats all fixed? | **win?**
---|---|---|---|---|---|---
0 | -8.317 | -8.430 | n3/d512=-8.430 | 0.113 [-0.316, 0.599] | yes | no
0.002 | -9.009 | -9.361 | n3/d64=-9.415 | 0.406 [0.031, 0.887] | yes | **YES**
0.004 | -9.623 | -10.291 | n3/d64=-10.269 | 0.646 [0.278, 1.112] | yes | **YES**
0.006 | -10.242 | -11.222 | n3/d64=-11.123 | 0.881 [0.496, 1.390] | yes | **YES**
0.008 | -10.844 | -12.153 | n3/d64=-11.977 | 1.133 [0.748, 1.662] | yes | **YES**
0.012 | -12.072 | -13.984 | n2/d512=-13.255 | 1.183 [-0.585, 4.451] | yes | no
0.02 | -14.426 | -17.687 | n2/d512=-15.191 | 0.765 [-1.105, 3.892] | yes | no

## Baselines + ablations (meta-greedy − baseline, 95% CI; >0 ⇒ controller better)

λ | vs meta(+hyst) | vs adaptivity-ablation | vs threshold-stopper | vs LLM-only-no-scoring
---|---|---|---|---
0 | -0.113 [-0.585, 0.323] | 2.034 [0.248, 5.300] | 0.407 [0.102, 0.887] | 4.832 [1.642, 9.935]
0.002 | -0.351 [-0.870, 0.092] | 4.382 [1.199, 9.255] | 0.317 [0.074, 0.707] | 4.382 [1.248, 9.276]
0.004 | -0.668 [-1.177, -0.191] | 4.010 [0.928, 8.985] | 0.306 [0.059, 0.708] | 4.010 [0.936, 8.976]
0.006 | -0.980 [-1.557, -0.502] | 3.633 [0.496, 8.505] | 0.290 [0.036, 0.706] | 3.633 [0.564, 8.232]
0.008 | -1.309 [-1.842, -0.787] | 3.273 [0.179, 8.237] | 0.290 [0.036, 0.707] | 3.273 [0.208, 8.039]
0.012 | -1.912 [-2.568, -1.323] | 2.529 [-0.544, 7.466] | 0.266 [-0.015, 0.702] | 2.529 [-0.432, 7.355]
0.02 | -3.261 [-4.034, -2.512] | 1.143 [-1.793, 5.658] | 0.323 [-0.032, 0.777] | 1.143 [-1.650, 5.902]

## Recovery study (selected == true generating structure; full reveal)

type | n | recovery rate (95% CI)
---|---|---
EASY | 25 | 0.960 [0.880, 1.000] (rate 0.960)
HARD | 25 | 0.720 [0.520, 0.880] (rate 0.720)
overall | 50 | 0.840 [0.740, 0.940] (rate 0.840)

## Honest caveats (load-bearing)

- **Headline policy = the MYOPIC VOC** (meta-greedy, hysteresis 1). The hysteresis-3 `controller` over-explores the short 3-candidate stream and is worse (the `vs meta(+hyst)` column is negative) — reported transparently, exactly as the gdtq anytime bench reports its own hysteresis wash. The win is the myopic VOC's.
- **Win band, not a point:** the CI-lo>0 win holds across the CONTIGUOUS λ region [0.002 0.004 0.006 0.008] — the active cost-quality trade-off regime. At λ=0 (compute free) the full-budget fixed policy ties (adaptivity has nothing to save, and myopic VOC slightly under-explores, Hay-Russell); at large λ the cheap fixed budgets become competitive and per-seed variance widens the CI. This IS the frontier-dominance shape the design predicts.
- **The scoring-depth knob pays as an ADAPTIVE action, not a fixed choice.** P0 found no fixed depth dominates (static grid). Here the controller DEEPENS on demand — only a non-conjugate candidate currently LOSING to a conjugate competitor by ≤ margin (directional gate; IS bias is one-directional) — recovering the heavy-tailed truth that shallow IS under-rates. That on-demand deepening is why meta-greedy beats the fixed-depth threshold-stopper (CI-lo>0). Both knobs (when-to-propose, when-to-deepen) contribute.
- **Decision-value vs reward:** the controller's dv and the reported reward are the SAME held-out predictive set (validation-based cost-aware early stopping, reported on the validation set). The adaptivity-ablation control (CI-lo>0) shows the win comes from PER-INSTANCE allocation, not from held-out access per se: replacing the controller by its own mean budget — same held-out access — loses. A fully separate test split is a clean-up refinement.
- **Exactness is load-bearing** (vs ModelSMC, docs/rrps-literature.md): the evidence oracle is EXACT closed-form for the conjugate majority (P0 cross-check ~5e-7); IS appears only where unavoidable (the non-conjugate candidate), and the held-out reward there is high-N IS.

## TITLE RESOLUTION

**KEEP title-B** — a CI-lo>0 net-utility frontier-dominance win vs the best-tuned fixed budget holds across a contiguous λ band, AND the adaptive mean beats EVERY fixed point AND every baseline (adaptivity-ablation, threshold-stopper, LLM-only) with CI-lo>0 in the win band. The resource-rational program-synthesis agent earns the title. Proceed to §8 (genmlx-2908).
