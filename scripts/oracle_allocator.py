#!/usr/bin/env python3
"""genmlx-brsu: oracle-allocator gate — does a per-family best-arm lookup beat
every fixed single-arm policy on the cost/quality frontier?

Retrospective, non-learned analysis over the 2026-06-24 full-27 in-loop eval
reports (one JSON per model tier; identical 27 task ids in each). Arms =
{base, sft600, grpo, qwen35b} x {oneshot, loop}. This is the pre-registered
GO/NO-GO gate for the title-B re-earn path (work order genmlx-ss7t Tier 0):
GO only if the allocator Pareto-dominates every fixed arm AND round-robin on
the held-out cohorts. A null is a valid verdict.

Cost metric (documented approximations, see ASSUMPTIONS in the output):
  per-task cost of (tier, strategy) = tier gen-time seconds
      x phase share (oneshot samples = 27 x k-oneshot; loop = rest,
        split proportional to samples) / 27 tasks.
  gen-time is measured wall time per tier, so the 35B's real per-token cost
  is inherently reflected (the bean's "not sample-count-equal" requirement).
  Limitation: per-task cost variation within a (tier, strategy) is not
  recorded in the reports, so costs are uniform per arm.

Leakage-safe allocation (the ubqo lesson):
  WITHIN cohort (5 families x 3 instances): leave-one-instance-out — each
  task's family->arm lookup is derived only from the OTHER instances of its
  family (never itself). HELD-OUT-FAMILY cohort (segmented, 12 instances):
  the lookup has no entry by construction; the allocator falls back to the
  global best arm derived on ALL within tasks. Cohorts reported separately.

Usage: python3 scripts/oracle_allocator.py [json-dir] [out-dir]
Defaults: json-dir=~/json, out-dir=results/control
Deterministic: bootstrap seeded (B=2000, seed 20260707).
"""
import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone

JSON_DIR = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/json")
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "results/control"
TIERS = ["base", "sft600", "grpo", "qwen35b"]
FILES = {t: (f"inloop_eval_{t}-full27.json" if t != "qwen35b" else "inloop_eval_qwen35b.json")
         for t in TIERS}
B = 2000
SEED = 20260707

# ---------------------------------------------------------------- load
reports, checksums = {}, {}
for t, fn in FILES.items():
    path = os.path.join(JSON_DIR, fn)
    raw = open(path, "rb").read()
    checksums[fn] = hashlib.sha256(raw).hexdigest()[:16]
    reports[t] = json.loads(raw)

ids0 = [r["id"] for r in reports[TIERS[0]]["results"]]
for t in TIERS:
    assert [r["id"] for r in reports[t]["results"]] == ids0, f"id mismatch in {t}"
    assert reports[t]["config"]["k-oneshot"] == reports[TIERS[0]]["config"]["k-oneshot"]

K_ONESHOT = reports[TIERS[0]]["config"]["k-oneshot"]
N_TASKS = len(ids0)

# per-arm uniform per-task cost (seconds of measured gen wall time)
arm_cost = {}
for t in TIERS:
    c = reports[t]["cost"]
    oneshot_samples = N_TASKS * K_ONESHOT
    total_samples = c["samples"]
    loop_samples = total_samples - oneshot_samples
    assert loop_samples > 0, f"{t}: loop samples <= 0"
    arm_cost[(t, "oneshot")] = c["gen-time"] * (oneshot_samples / total_samples) / N_TASKS
    arm_cost[(t, "loop")] = c["gen-time"] * (loop_samples / total_samples) / N_TASKS

ARMS = [(t, s) for t in TIERS for s in ("oneshot", "loop")]
def arm_name(a): return f"{a[0]}/{a[1]}"

# tasks[i] = {id, family, cohort, instance, solved: {arm: bool}}
tasks = []
for i, rid in enumerate(ids0):
    r0 = reports[TIERS[0]]["results"][i]
    solved = {}
    for t in TIERS:
        r = reports[t]["results"][i]
        solved[(t, "oneshot")] = bool(r["oneshot-solved"])
        solved[(t, "loop")] = bool(r["loop-solved"])
    tasks.append({"id": rid, "family": r0["family"], "cohort": r0["cohort"],
                  "instance": rid.rsplit("-", 1)[-1], "solved": solved})

within = [x for x in tasks if x["cohort"] == "within"]
heldout_family = [x for x in tasks if x["cohort"] == "family"]

# ---------------------------------------------------------------- policies
def best_arm(task_subset):
    """Best arm on a task subset: max solve count, tie -> cheaper arm."""
    return max(ARMS, key=lambda a: (sum(x["solved"][a] for x in task_subset), -arm_cost[a]))

def policy_eval(assignments):
    """assignments: list of (task, arm) -> {solve-rate, mean-cost}."""
    n = len(assignments)
    return {"n": n,
            "solve-rate": sum(x["solved"][a] for x, a in assignments) / n,
            "mean-cost-s": sum(arm_cost[a] for _, a in assignments) / n}

def allocator_assignments(task_subset):
    """Leakage-safe per-family lookup: LOIO within; global-train fallback for
    families absent from train (the held-out-family cohort)."""
    out = []
    for x in task_subset:
        if x["cohort"] == "within":
            train = [y for y in within
                     if y["family"] == x["family"] and y["instance"] != x["instance"]]
            out.append((x, best_arm(train)))
        else:
            out.append((x, best_arm(within)))  # family unseen in train
    return out

def oracle_assignments(task_subset):
    """Per-task upper bound (not realizable): cheapest solving arm, else cheapest arm."""
    out = []
    for x in task_subset:
        solving = [a for a in ARMS if x["solved"][a]]
        out.append((x, min(solving, key=lambda a: arm_cost[a]) if solving
                    else min(ARMS, key=lambda a: arm_cost[a])))
    return out

def rr_eval(task_subset):
    """Round-robin = uniform expectation over arms."""
    n = len(task_subset)
    return {"n": n,
            "solve-rate": sum(sum(x["solved"][a] for a in ARMS) / len(ARMS)
                              for x in task_subset) / n,
            "mean-cost-s": sum(arm_cost.values()) / len(arm_cost)}

def frontier(task_subset):
    rows = {arm_name(a): policy_eval([(x, a) for x in task_subset]) for a in ARMS}
    rows["round-robin"] = rr_eval(task_subset)
    rows["per-family-allocator"] = policy_eval(allocator_assignments(task_subset))
    rows["per-task-oracle"] = policy_eval(oracle_assignments(task_subset))
    return rows

def pareto_dominated_by_allocator(rows):
    """Gate predicate: allocator >= solve and <= cost vs EVERY fixed arm and RR,
    strictly better somewhere vs each."""
    al = rows["per-family-allocator"]
    verdicts = {}
    for name in [arm_name(a) for a in ARMS] + ["round-robin"]:
        p = rows[name]
        geq = al["solve-rate"] >= p["solve-rate"] - 1e-12
        leq = al["mean-cost-s"] <= p["mean-cost-s"] + 1e-12
        strict = al["solve-rate"] > p["solve-rate"] + 1e-12 or \
                 al["mean-cost-s"] < p["mean-cost-s"] - 1e-12
        verdicts[name] = bool(geq and leq and strict)
    return verdicts

def bootstrap_margin(task_subset, fixed_best_name):
    """Paired bootstrap CI (B, seeded) on allocator solve-rate minus the
    best-tuned fixed arm's, resampling tasks."""
    rng = random.Random(SEED)
    fixed = next(a for a in ARMS if arm_name(a) == fixed_best_name)
    alloc = dict((x["id"], a) for x, a in allocator_assignments(task_subset))
    diffs = []
    n = len(task_subset)
    for _ in range(B):
        sample = [task_subset[rng.randrange(n)] for _ in range(n)]
        d = sum(x["solved"][alloc[x["id"]]] for x in sample) / n \
            - sum(x["solved"][fixed] for x in sample) / n
        diffs.append(d)
    diffs.sort()
    return {"mean": sum(diffs) / B,
            "ci95": [diffs[int(0.025 * B)], diffs[int(0.975 * B)]],
            "vs": fixed_best_name, "B": B, "seed": SEED}

# ---------------------------------------------------------------- run
result = {"assumptions": [
              "per-task cost uniform within (tier,strategy): reports carry only aggregate cost",
              f"oneshot/loop gen-time split proportional to samples (oneshot = {N_TASKS}x{K_ONESHOT})",
              "cost unit = measured gen-time seconds per tier (35B per-token cost inherent)",
              "within cohort: leave-one-instance-out lookup; held-out family: global-train-best fallback"],
          "arm-cost-s-per-task": {arm_name(a): round(c, 3) for a, c in arm_cost.items()},
          "cohorts": {}}

for label, subset in [("within-LOIO", within), ("held-out-family", heldout_family),
                      ("all-27", tasks)]:
    rows = frontier(subset)
    # best-tuned fixed arm: max solve, tie -> cheaper (the binding baseline)
    fixed_names = [arm_name(a) for a in ARMS]
    best_fixed = max(fixed_names,
                     key=lambda nm: (rows[nm]["solve-rate"], -rows[nm]["mean-cost-s"]))
    verd = pareto_dominated_by_allocator(rows)
    result["cohorts"][label] = {
        "frontier": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in rows.items()},
        "best-fixed": best_fixed,
        "allocator-dominates": verd,
        "gate-pass": all(verd.values()),
        "bootstrap-solve-margin-vs-best-fixed": {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in bootstrap_margin(subset, best_fixed).items()},
        "headroom-to-per-task-oracle": round(
            rows["per-task-oracle"]["solve-rate"]
            - rows["per-family-allocator"]["solve-rate"], 4)}

gate = result["cohorts"]["held-out-family"]["gate-pass"] and \
       result["cohorts"]["within-LOIO"]["gate-pass"]
result["verdict"] = "GO" if gate else "NO-GO"

result["metadata"] = {
    "bean": "genmlx-brsu", "script": "scripts/oracle_allocator.py",
    "source-files": checksums,
    "git-sha": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "eval-config": reports[TIERS[0]]["config"]}

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "brsu-frontier.json"), "w") as f:
    json.dump(result, f, indent=1)

# ---------------------------------------------------------------- markdown
L = [f"# brsu oracle-allocator gate — verdict: **{result['verdict']}**", "",
     "Pre-registered GO/NO-GO for the title-B re-earn path (genmlx-ss7t Tier 0).",
     "GO requires the leakage-safe per-family allocator to Pareto-dominate every",
     "fixed arm AND round-robin on BOTH held-out cohorts.", ""]
for label, c in result["cohorts"].items():
    L += [f"## {label} (n={c['frontier']['per-family-allocator']['n']})", "",
          "| policy | solve rate | mean cost (s/task) |", "|---|---|---|"]
    for name, row in sorted(c["frontier"].items(), key=lambda kv: -kv[1]["solve-rate"]):
        mark = " **<-**" if name == "per-family-allocator" else ""
        L.append(f"| {name}{mark} | {row['solve-rate']:.3f} | {row['mean-cost-s']:.1f} |")
    bm = c["bootstrap-solve-margin-vs-best-fixed"]
    L += ["", f"best fixed arm: `{c['best-fixed']}` — allocator gate-pass: **{c['gate-pass']}**",
          f"solve-margin vs best fixed: {bm['mean']:+.3f} (95% CI [{bm['ci95'][0]:+.3f}, {bm['ci95'][1]:+.3f}], paired bootstrap B={bm['B']})",
          f"headroom to per-task oracle: {c['headroom-to-per-task-oracle']:+.3f}", ""]
L += ["## Assumptions", ""] + [f"- {a}" for a in result["assumptions"]]
L += ["", f"metadata: git {result['metadata']['git-sha'][:8]}, sources sha256/16 "
      + ", ".join(f"{k}={v}" for k, v in checksums.items())]
with open(os.path.join(OUT_DIR, "brsu-frontier.md"), "w") as f:
    f.write("\n".join(L) + "\n")

print(f"verdict: {result['verdict']}")
for label, c in result["cohorts"].items():
    al = c["frontier"]["per-family-allocator"]
    bf = c["frontier"][c["best-fixed"]]
    print(f"  {label}: allocator {al['solve-rate']:.3f}@{al['mean-cost-s']:.1f}s"
          f" vs best-fixed {c['best-fixed']} {bf['solve-rate']:.3f}@{bf['mean-cost-s']:.1f}s"
          f" gate={c['gate-pass']}")
print("wrote", os.path.join(OUT_DIR, "brsu-frontier.{json,md}"))
