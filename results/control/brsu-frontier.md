# brsu oracle-allocator gate — verdict: **NO-GO**

Pre-registered GO/NO-GO for the title-B re-earn path (genmlx-ss7t Tier 0).
GO requires the leakage-safe per-family allocator to Pareto-dominate every
fixed arm AND round-robin on BOTH held-out cohorts.

## within-LOIO (n=15)

| policy | solve rate | mean cost (s/task) |
|---|---|---|
| per-family-allocator **<-** | 0.733 | 23.0 |
| per-task-oracle | 0.733 | 22.3 |
| grpo/loop | 0.467 | 27.9 |
| qwen35b/oneshot | 0.467 | 30.8 |
| sft600/loop | 0.267 | 36.9 |
| qwen35b/loop | 0.267 | 55.9 |
| round-robin | 0.233 | 31.2 |
| base/loop | 0.200 | 48.4 |
| grpo/oneshot | 0.133 | 14.2 |
| base/oneshot | 0.067 | 19.2 |
| sft600/oneshot | 0.000 | 16.3 |

best fixed arm: `grpo/loop` — allocator gate-pass: **False**
solve-margin vs best fixed: +0.268 (95% CI [+0.067, +0.467], paired bootstrap B=2000)
headroom to per-task oracle: +0.000

## held-out-family (n=12)

| policy | solve rate | mean cost (s/task) |
|---|---|---|
| per-task-oracle | 0.417 | 21.9 |
| sft600/loop | 0.250 | 36.9 |
| qwen35b/oneshot | 0.167 | 30.8 |
| qwen35b/loop | 0.167 | 55.9 |
| grpo/loop | 0.083 | 27.9 |
| round-robin | 0.083 | 31.2 |
| per-family-allocator **<-** | 0.083 | 27.9 |
| base/oneshot | 0.000 | 19.2 |
| base/loop | 0.000 | 48.4 |
| sft600/oneshot | 0.000 | 16.3 |
| grpo/oneshot | 0.000 | 14.2 |

best fixed arm: `sft600/loop` — allocator gate-pass: **False**
solve-margin vs best fixed: -0.166 (95% CI [-0.417, +0.000], paired bootstrap B=2000)
headroom to per-task oracle: +0.333

## all-27 (n=27)

| policy | solve rate | mean cost (s/task) |
|---|---|---|
| per-task-oracle | 0.593 | 22.1 |
| per-family-allocator **<-** | 0.444 | 25.2 |
| qwen35b/oneshot | 0.333 | 30.8 |
| grpo/loop | 0.296 | 27.9 |
| sft600/loop | 0.259 | 36.9 |
| qwen35b/loop | 0.222 | 55.9 |
| round-robin | 0.167 | 31.2 |
| base/loop | 0.111 | 48.4 |
| grpo/oneshot | 0.074 | 14.2 |
| base/oneshot | 0.037 | 19.2 |
| sft600/oneshot | 0.000 | 16.3 |

best fixed arm: `qwen35b/oneshot` — allocator gate-pass: **False**
solve-margin vs best fixed: +0.114 (95% CI [-0.074, +0.296], paired bootstrap B=2000)
headroom to per-task oracle: +0.148

## Assumptions

- per-task cost uniform within (tier,strategy): reports carry only aggregate cost
- oneshot/loop gen-time split proportional to samples (oneshot = 27x4)
- cost unit = measured gen-time seconds per tier (35B per-token cost inherent)
- within cohort: leave-one-instance-out lookup; held-out family: global-train-best fallback

metadata: git 9ebe0f4c, sources sha256/16 inloop_eval_base-full27.json=5dfdedc54c63fd59, inloop_eval_sft600-full27.json=fdef25ea73f05e60, inloop_eval_grpo-full27.json=f1a163a80b987f70, inloop_eval_qwen35b.json=fd6fac515abb105b
