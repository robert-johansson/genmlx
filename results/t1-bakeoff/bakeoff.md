# Ornith-35B vs Qwen3-Coder-Next bake-off — T1/T2 report (genmlx-8lm2)

2026-07-11, Thor (aarch64 Tegra, CUDA). All runs guarded (`~/genmlx-guarded-run.sh`,
FLOOR_MB=25000), strictly sequential, one GPU process at a time.

## Arms

| arm | model | path | forward |
|---|---|---|---|
| A | Qwen3-Coder-Next-80B-A3B 4-bit | `Qwen3-Coder-Next-4bit/snapshots/7b9321ea…` | native qwen3_next |
| B | Ornith-1.0-35B-A3B 8-bit | `models--mlx-community--Ornith-1.0-35B-8bit/snapshots/28d10f45…` | GenMLX owned |
| C | arm B + 25 steps evidence-reward GRPO | `~/genmlx-checkpoints/ornith35b-8bit-grpo-armc-merged` | GenMLX owned |

Arm C training: `scripts/grpo_student.cljs` — reward = Bayesian model evidence
(`world/train_reward.cljs` `model-evidence-reward`, gaussian-mean task), frozen packed
experts (genmlx-n32r path), GROUP_SIZE=8 MAX_COMPLETION=220 LR=1e-3 SGD seed=1,
LM_HEAD_CHUNK=1 FORWARD_CHUNK=2. Applied 25/25; reward mean(first 3) −7.69 →
mean(last 3) −2.77 (TREND UP). This was the **first GRPO run on an 8-bit
checkpoint** (previously verified 4-bit only). The trained non-expert save was
reconstituted into a loadable HF-layout checkpoint with
`scripts/reconstitute_moe_checkpoint.py` (genmlx-e2my; experts + vision
bit-identical to source by construction).

## T1 — instruction-prompted battery (17 tasks × K=4, seeded)

Frozen config: `K=4 SEED=42 TEMPERATURE=0.8 MAX_TOKENS=512`, battery =
`world/t1_battery.cljs` (12 distill + 5 lifted msa `:program` tasks), sample 0
greedy. Scoring: `scripts/t1_score.cljs` (sandboxed SCI verdicts, evidence via
`msa-score`, seeded bootstrap CI). Raw artifacts: `gen-{a,b,c}.jsonl`,
`score-{a,b,c}.json`, `compare-*.json` in this directory; generation configs in
`gen-*-meta.json` (git SHAs included).

| metric | A: 80B native | B: 35B-8bit owned | C: B + GRPO |
|---|---|---|---|
| parse rate | **97%** | 85% | 87% |
| keep rate | **91%** CI[82,97] | 75% CI[63,85] | 79% CI[69,88] |
| SCI pass (`:function`, n=32) | 88% CI[75,97] | 84% CI[69,97] | 84% CI[72,97] |
| program keep | **94%** | 67% | 75% |
| evidence mean (kept) | −4.65 | −4.42 | **−4.16** |
| evidence max | −0.35 | −0.00 | **+0.46** |
| tokens-to-first-valid (median) | 73.5 | **50.0** | 62.0 |
| gen wall (68 samples) | **349 s** | 571 s | 605 s |

Reading:
- **B matches A on function-writing** (SCI pass within CI overlap) at 2.3× fewer
  total parameters, but **loses on program reliability** — parse failures
  dominate its gap (10 vs 2 unparseable).
- **C (sharpening) moved every targeted metric the right way with no
  regressions**: keep +4pp, parse +2pp, evidence mean +0.26 nats, evidence max
  now the best of all three arms (a program better than the reference). The
  parse-reliability gap to A narrowed but did not close — the obvious scaling
  axes are more steps and a multi-task training battery (25 steps on one task
  here).

### Disclosed asymmetries and caveats

1. **FIM is excluded from T1 by design.** It is an 80B-only capability (Ornith
   has no FIM tokens); T1 is chat-instruction prompting for all arms. FIM
   remains a separate arm-A-only probe (`examples/fim_infill.cljs`).
2. **Train/test overlap.** Arm C's training task (gaussian-mean, obs ≈ 2.0) is
   near-identical to battery task `gaussian-mean-near2`. On that task: A
   3/4 pass, B **0/4**, C 1/4 with the best evidence of any arm (−1.15). On the
   16-task transfer set (overlap task removed): keep = B 79.7% vs C 82.8% —
   the transfer effect is real but smaller than the headline (+4pp) suggests.
3. **Compute framing (per-FLOP honesty).** Both models are A3B-class MoE
   (~3B active params/token), so **per-token FLOPs are roughly comparable**;
   the 2.3× is total parameters (memory footprint, cache pressure). Measured
   wall-clock favors the 80B *native* path on Thor today (349 s vs 571/605 s) —
   the owned 8-bit decode (~9 tok/s) is the bottleneck, an engineering axis
   (cf. the ps8a prefill work), not a model-quality one. The 35B's demonstrated
   resource-rational edge is **on-device trainability** (arm C trained on this
   box; the 80B cannot) plus footprint.
4. **One guard kill during T1-C generation** (MemAvailable 24.1 GB < 25 GB floor
   at sample 64/68): the run was resumed (`t1_bakeoff` is resume-stable) and
   completed; scoring covers all 68 records.

## T2 — in-the-loop (world.search proposer)

Runner: `scripts/t2_inloop.cljs` (in-process proposer, replay-trampoline;
NP=200 oracle; 9 `:program` tasks — the 8 `:function` tasks have behavioral
oracles with no evidence solve bar and are excluded, listed in the output
metadata).

- **Arm A: 9/9 solved**, median tokens-to-solution 432, mean 2.0 search steps,
  5770 completion tokens total, 0 errors (`t2-a.json`).
- **Arms B/C: blocked** — with the owned 8-bit 35B resident, the *first task's
  synchronous search/oracle phase* (before any LLM call) climbs to ~107 GB and
  is floor-killed every time. Not the oracle particle count (NP=2000 and 200
  identical), not MLX pool retention (`MLX_CACHE_LIMIT_GB=8` no effect), not
  missing GC (a per-step synchronous sweep was added and is insufficient), and
  not the model's normal first-forward transient (T1 on the same checkpoint
  peaks at ~88 GB and recovers). Full evidence trail and next probes:
  **genmlx-12w4**. T2 B/C numbers are deferred until that bug is fixed; no
  cross-arm T2 comparison is claimed.

## Feeds

Seeds, configs, and checkpoints above are frozen for the paper (8dfk E11 /
1vxi). Training logs: `~/genmlx-battery-logs/grpo-35b8-armc.jsonl`,
guarded-run logs `guarded-{t1,t2}-*.txt/.mem.csv`.
