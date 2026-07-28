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
  merge_sbc_results.py <fragment-dir> <output-json> [--force]
      --force overrides the shrink guard: without it, a merge covering fewer
      combos/param tests than an existing git-TRACKED output refuses to write
      (exit 2, nothing changed) — see shrink_guard, genmlx-ty5u.
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


def git_tracked(path: pathlib.Path) -> bool:
    """True iff path is a git-tracked file. False outside a checkout/without git."""
    try:
        return subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", str(path)],
            capture_output=True, timeout=30,
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def coverage(data: dict) -> tuple[int, int]:
    """(combos ingested, param tests recorded) — the size of the evidence."""
    return len(data.get("results", [])), (data.get("summary") or {}).get("total", 0)


def shrink_guard(out_path: pathlib.Path, merged: dict, force: bool) -> str | None:
    """Refuse to shrink git-TRACKED frozen evidence (genmlx-ty5u).

    On 2026-07-07 a timed-out single-combo CUDA run replaced 27 combos of
    frozen Mac evidence with 1, and only `git checkout` saved it. A merge that
    covers strictly fewer combos (or fewer param tests) than the file it would
    overwrite is refused unless --force is passed deliberately. Untracked
    outputs — every scratch run — are never guarded.
    """
    if force or not out_path.exists() or not git_tracked(out_path):
        return None
    try:
        old = json.loads(out_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    old_combos, old_params = coverage(old)
    new_combos, new_params = coverage(merged)
    if new_combos < old_combos or new_params < old_params:
        return (f"REFUSED: {out_path} is git-tracked frozen evidence with "
                f"{old_combos} combos / {old_params} param tests; this merge has "
                f"{new_combos} combos / {new_params}. Nothing written.\n"
                f"         Re-run the missing combos, or pass --force if the "
                f"smaller record is genuinely the one you mean to freeze.")
    return None


def main() -> int:
    if sys.argv[1] == "--skip-ok":
        status, _, _ = classify_fragment(sys.argv[3], pathlib.Path(sys.argv[2]))
        return 0 if status == "passed" else 1

    argv = [a for a in sys.argv[1:] if a != "--force"]
    force = "--force" in sys.argv[1:]
    frag_dir = pathlib.Path(argv[0])
    out_path = pathlib.Path(argv[1])

    registry = combo_registry()
    if not registry:
        print("ERROR: empty combo registry (SBC_LIST run failed)")
        return 1

    merged = merge_fragments(registry, frag_dir)
    refusal = shrink_guard(out_path, merged, force)
    if refusal:
        print(refusal)
        return 2
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
