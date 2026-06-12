#!/usr/bin/env python3
"""Merge per-combo SBC fragments into results/sbc_results.json (genmlx-18q9).

Each fragment is one sbc_test.cljs run with SBC_ONLY=model:algo and
SBC_OUT=results/sbc/<model>_<algo>.json. A fragment whose own summary says
complete? true ran to a VERDICT — either passed (per-param rows present;
individual params may still fail chi2/ks) or FAILED (run-sbc exceeded the
sim-failure budget and produced no per-param rows). Fragments from
killed/wedged processes (absent, unparseable, or complete? false) are
reported missing.

Three-way classification per registry combo (genmlx-hojy):
  passed  — complete fragment with per-param rows for the combo
  failed  — complete fragment that ran to a failed verdict; surfaces in
            merged results as {model, algorithm, verdict: "failed", reason}
            and in summary.failed_combos
  missing — no trustworthy fragment; summary.missing_combos

summary.complete? is true only when registry coverage is full AND no combo
ran to a failed verdict. Pass/fail totals are recomputed from per-param
pass? flags; each failed combo adds one to fail (the same convention as
sbc_test.cljs's combo-level fail counter).

Usage:
  merge_sbc_results.py <fragment-dir> <output-json>
  merge_sbc_results.py --skip-ok <fragment> <model:algo>
      exit 0 iff the fragment ran to a PASSED verdict for that combo
      (used by test/run_sbc.sh skip-resume so failed combos re-run)
"""
import json
import pathlib
import subprocess
import sys


def combo_registry() -> list[str]:
    """Enumerate the full combo registry from sbc_test.cljs (SBC_LIST=1)."""
    proc = subprocess.run(
        ["bun", "run", "--bun", "nbb", "test/genmlx/sbc_test.cljs"],
        env={**__import__("os").environ, "SBC_LIST": "1"},
        capture_output=True, text=True, timeout=300,
    )
    return [line.removeprefix("COMBO ").strip()
            for line in proc.stdout.splitlines()
            if line.startswith("COMBO ")]


def classify_fragment(combo: str, frag_path: pathlib.Path):
    """Classify one registry combo against its fragment file.

    Returns (status, rows, config) where status is "passed" | "failed" |
    "missing", rows are the merged-results rows to ingest (a synthesized
    verdict row for legacy failed fragments that wrote no rows), and config
    is the fragment's config (None when missing).
    """
    if not frag_path.exists():
        return "missing", [], None
    try:
        data = json.loads(frag_path.read_text())
    except json.JSONDecodeError:
        return "missing", [], None
    if not data.get("summary", {}).get("complete?"):
        return "missing", [], None

    config = {k: v for k, v in data.get("config", {}).items() if k != "only"}
    model, _, algo = combo.partition(":")
    rows = [r for r in data.get("results", [])
            if r.get("model") == model and r.get("algorithm") == algo]

    if any(r.get("verdict") == "failed" for r in rows):
        return "failed", rows, config
    if rows:
        return "passed", rows, config
    if data.get("summary", {}).get("fail", 0) > 0:
        # Legacy failed fragment: ran to a failed verdict before
        # sbc_test.cljs recorded verdict rows — no per-param rows, only
        # the combo-level fail counter. Synthesize the verdict row.
        return "failed", [{
            "model": model,
            "algorithm": algo,
            "verdict": "failed",
            "reason": ("ran to failed verdict: no per-param rows "
                       "(sim-failure budget exceeded; legacy fragment "
                       "without verdict field)"),
            "params": [],
        }], config
    # complete? true but no rows and no failures recorded — inconsistent;
    # don't trust it.
    return "missing", [], None


def merge_fragments(registry: list[str], frag_dir: pathlib.Path) -> dict:
    results, config = [], None
    failed, missing = [], []
    provenance = set()
    for combo in registry:
        frag = frag_dir / (combo.replace(":", "_") + ".json")
        status, rows, frag_config = classify_fragment(combo, frag)
        if status == "missing":
            missing.append(combo)
            continue
        config = config or frag_config
        meta = (frag_config or {}).get("meta") or {}
        if meta.get("genmlx_commit") or meta.get("mlx_node_commit"):
            provenance.add((meta.get("genmlx_commit"),
                            meta.get("mlx_node_commit")))
        results.extend(rows)
        if status == "failed":
            failed.append(combo)

    n_pass = sum(1 for r in results for p in r.get("params", []) if p["pass?"])
    n_fail = (sum(1 for r in results for p in r.get("params", [])
                  if not p["pass?"])
              + len(failed))
    return {
        "config": config or {},
        "results": results,
        "summary": {
            "pass": n_pass,
            "fail": n_fail,
            "total": n_pass + n_fail,
            "complete?": not missing and not failed,
            "failed_combos": failed,
            "missing_combos": missing,
            # Freeze-gate honesty (genmlx-9ocx): every distinct
            # (genmlx_commit, mlx_node_commit) pair the ingested fragments
            # were produced on. A clean frozen run has exactly one entry;
            # more means fragments span code/binary states.
            "provenance": [
                {"genmlx_commit": g, "mlx_node_commit": m}
                for g, m in sorted(provenance, key=str)
            ],
        },
    }


def main() -> int:
    if sys.argv[1] == "--skip-ok":
        status, _, _ = classify_fragment(sys.argv[3], pathlib.Path(sys.argv[2]))
        return 0 if status == "passed" else 1

    frag_dir = pathlib.Path(sys.argv[1])
    out_path = pathlib.Path(sys.argv[2])

    registry = combo_registry()
    if not registry:
        print("ERROR: empty combo registry (SBC_LIST run failed)")
        return 1

    merged = merge_fragments(registry, frag_dir)
    out_path.write_text(json.dumps(merged, indent=2))
    s = merged["summary"]
    print(f"merged {len(merged['results'])} combos -> {out_path}  "
          f"({s['pass']}/{s['total']} params passed)")
    if s["failed_combos"]:
        print(f"FAILED VERDICT: {len(s['failed_combos'])} combos: "
              + ", ".join(s["failed_combos"]))
    if s["missing_combos"]:
        print(f"INCOMPLETE: {len(s['missing_combos'])} combos "
              "missing/unfinished: " + ", ".join(s["missing_combos"]))
    return 0 if s["complete?"] else 1


if __name__ == "__main__":
    sys.exit(main())
