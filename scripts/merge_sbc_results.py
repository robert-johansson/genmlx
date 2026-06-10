#!/usr/bin/env python3
"""Merge per-combo SBC fragments into results/sbc_results.json (genmlx-18q9).

Each fragment is one sbc_test.cljs run with SBC_ONLY=model:algo and
SBC_OUT=results/sbc/<model>_<algo>.json. Only fragments whose own summary
says complete? true are trusted — a fragment from a killed/wedged process
is ignored and its combo reported missing.

The merged file's summary.complete? is true only when every combo in the
registry (enumerated via SBC_LIST by test/run_sbc.sh) has a completed
fragment. Pass/fail totals are recomputed from the per-param pass? flags.

Usage: merge_sbc_results.py <fragment-dir> <output-json>
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


def main() -> int:
    frag_dir = pathlib.Path(sys.argv[1])
    out_path = pathlib.Path(sys.argv[2])

    registry = combo_registry()
    if not registry:
        print("ERROR: empty combo registry (SBC_LIST run failed)")
        return 1

    results, config, missing = [], None, []
    for combo in registry:
        frag = frag_dir / (combo.replace(":", "_") + ".json")
        if not frag.exists():
            missing.append(combo)
            continue
        try:
            data = json.loads(frag.read_text())
        except json.JSONDecodeError:
            missing.append(combo)
            continue
        if not data.get("summary", {}).get("complete?"):
            missing.append(combo)
            continue
        config = config or {k: v for k, v in data["config"].items() if k != "only"}
        results.extend(data.get("results", []))

    n_pass = sum(1 for r in results for p in r["params"] if p["pass?"])
    n_fail = sum(1 for r in results for p in r["params"] if not p["pass?"])
    merged = {
        "config": config or {},
        "results": results,
        "summary": {
            "pass": n_pass,
            "fail": n_fail,
            "total": n_pass + n_fail,
            "complete?": not missing,
            "missing_combos": missing,
        },
    }
    out_path.write_text(json.dumps(merged, indent=2))
    print(f"merged {len(results)} combos -> {out_path}  "
          f"({n_pass}/{n_pass + n_fail} params passed)")
    if missing:
        print(f"INCOMPLETE: {len(missing)} combos missing/unfinished: "
              + ", ".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
