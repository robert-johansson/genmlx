#!/usr/bin/env python3
"""Regression test for scripts/merge_sbc_results.py (genmlx-hojy).

The bug: a fragment that ran to a FAILED verdict (complete?: true, no
per-param rows) was ingested as zero rows and listed in neither results
nor missing_combos — an executed-and-failed combo was invisible in the
merged summary. These tests round-trip synthetic fragments through the
merge and assert the three-way passed/failed/missing classification.

Run: python3 test/merge_sbc_results_test.py
"""
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
import merge_sbc_results as m  # noqa: E402

FAILURES = 0


def check(desc: str, ok: bool):
    global FAILURES
    print(("PASS: " if ok else "FAIL: ") + desc)
    if not ok:
        FAILURES += 1


CONFIG = {"N": 500, "L": 200, "N_BINS": 10, "ALPHA": 0.01,
          "n_total_tests": 50, "only": "x"}


def write_frag(d: pathlib.Path, combo: str, body: dict):
    (d / (combo.replace(":", "_") + ".json")).write_text(json.dumps(body))


def green_frag(model: str, algo: str, n_params: int = 2) -> dict:
    params = [{"name": f"p{i}", "ranks": [], "chi2": {}, "ecdf": {},
               "pass?": True} for i in range(n_params)]
    return {"config": CONFIG,
            "results": [{"model": model, "algorithm": algo,
                         "params": params, "elapsed_s": 1.0}],
            "summary": {"pass": n_params, "fail": 0, "total": n_params,
                        "complete?": True}}


def legacy_failed_frag() -> dict:
    # Exact shape of the 2026-06-11 conjugate-linreg-elim_smc.json fragment:
    # ran to a failed verdict before verdict rows existed.
    return {"config": CONFIG, "results": [],
            "summary": {"pass": 0, "fail": 1, "total": 1, "complete?": True}}


def verdict_failed_frag(model: str, algo: str) -> dict:
    return {"config": CONFIG,
            "results": [{"model": model, "algorithm": algo,
                         "verdict": "failed",
                         "reason": "sim-failure budget exceeded (>5% of sims failed)",
                         "params": [], "elapsed_s": 2.0}],
            "summary": {"pass": 0, "fail": 1, "total": 1, "complete?": True}}


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        d = pathlib.Path(tmp)
        registry = ["m1:cmh", "m2:smc", "m3:hmc", "m4:is", "m5:mh"]
        write_frag(d, "m1:cmh", green_frag("m1", "cmh"))
        write_frag(d, "m2:smc", legacy_failed_frag())
        write_frag(d, "m3:hmc", verdict_failed_frag("m3", "hmc"))
        # m4:is — no fragment at all
        incomplete = green_frag("m5", "mh")
        incomplete["summary"]["complete?"] = False
        write_frag(d, "m5:mh", incomplete)

        merged = m.merge_fragments(registry, d)
        s = merged["summary"]
        rows = {(r["model"], r["algorithm"]): r for r in merged["results"]}

        print("\n-- three-way classification --")
        check("passed combo ingested with params",
              ("m1", "cmh") in rows and len(rows[("m1", "cmh")]["params"]) == 2)
        check("legacy failed fragment surfaces as FAILED row",
              ("m2", "smc") in rows
              and rows[("m2", "smc")].get("verdict") == "failed")
        check("legacy failed row carries a reason",
              bool(rows.get(("m2", "smc"), {}).get("reason")))
        check("verdict-format failed fragment surfaces as FAILED row",
              rows.get(("m3", "hmc"), {}).get("verdict") == "failed")
        check("failed_combos lists exactly the two failed combos",
              s["failed_combos"] == ["m2:smc", "m3:hmc"])
        check("missing = registry minus (passed + failed)",
              s["missing_combos"] == ["m4:is", "m5:mh"])

        print("\n-- summary counting --")
        check("per-param passes counted", s["pass"] == 2)
        check("failed combos counted in fail (fail > 0)", s["fail"] == 2)
        check("complete? false with failed combos present",
              s["complete?"] is False)

        print("\n-- all-green run --")
        registry2 = ["m1:cmh"]
        merged2 = m.merge_fragments(registry2, d)
        s2 = merged2["summary"]
        check("complete? true when all combos passed",
              s2["complete?"] is True and s2["failed_combos"] == []
              and s2["missing_combos"] == [])

        print("\n-- skip-resume classification (--skip-ok) --")
        check("passed fragment is skip-ok",
              m.classify_fragment("m1:cmh", d / "m1_cmh.json")[0] == "passed")
        check("legacy failed fragment is NOT skip-ok",
              m.classify_fragment("m2:smc", d / "m2_smc.json")[0] == "failed")
        check("verdict failed fragment is NOT skip-ok",
              m.classify_fragment("m3:hmc", d / "m3_hmc.json")[0] == "failed")
        check("incomplete fragment classified missing",
              m.classify_fragment("m5:mh", d / "m5_mh.json")[0] == "missing")

    shrink_guard_tests()

    print(f"\n{'ALL PASS' if FAILURES == 0 else str(FAILURES) + ' FAILURES'}")
    return 1 if FAILURES else 0


def _merged(n_combos: int, n_params: int) -> dict:
    """A merged-results shape with the given evidence size."""
    return {"config": {}, "results": [{"model": f"m{i}", "algorithm": "cmh",
                                       "params": []} for i in range(n_combos)],
            "summary": {"pass": n_params, "fail": 0, "total": n_params,
                        "complete?": True, "failed_combos": [],
                        "missing_combos": [], "provenance": []}}


def shrink_guard_tests():
    """The genmlx-ty5u guard: a merge may not shrink TRACKED frozen evidence."""
    print("\n-- shrink guard (tracked frozen evidence) --")
    import subprocess as sp
    with tempfile.TemporaryDirectory() as td:
        d = pathlib.Path(td)
        sp.run(["git", "init", "-q"], cwd=d, check=True)
        sp.run(["git", "config", "user.email", "t@t"], cwd=d, check=True)
        sp.run(["git", "config", "user.name", "t"], cwd=d, check=True)

        frozen = d / "frozen.json"
        frozen.write_text(json.dumps(_merged(27, 50)))
        sp.run(["git", "add", "frozen.json"], cwd=d, check=True)
        sp.run(["git", "commit", "-qm", "freeze"], cwd=d, check=True)

        untracked = d / "scratch.json"
        untracked.write_text(json.dumps(_merged(27, 50)))

        cwd = pathlib.Path.cwd()
        try:
            import os
            os.chdir(d)
            check("tracked file detected", m.git_tracked(pathlib.Path("frozen.json")))
            check("untracked file not detected",
                  not m.git_tracked(pathlib.Path("scratch.json")))

            # The 2026-07-07 incident shape: 1 combo replacing 27.
            check("shrink on tracked output is REFUSED",
                  m.shrink_guard(pathlib.Path("frozen.json"),
                                 _merged(1, 2), False) is not None)
            check("fewer param tests at equal combo count is REFUSED",
                  m.shrink_guard(pathlib.Path("frozen.json"),
                                 _merged(27, 49), False) is not None)
            check("equal-size merge is allowed",
                  m.shrink_guard(pathlib.Path("frozen.json"),
                                 _merged(27, 50), False) is None)
            check("larger merge is allowed",
                  m.shrink_guard(pathlib.Path("frozen.json"),
                                 _merged(28, 52), False) is None)
            check("--force overrides the refusal",
                  m.shrink_guard(pathlib.Path("frozen.json"),
                                 _merged(1, 2), True) is None)
            check("UNtracked output is never guarded (scratch runs)",
                  m.shrink_guard(pathlib.Path("scratch.json"),
                                 _merged(1, 2), False) is None)
            check("absent output is never guarded",
                  m.shrink_guard(pathlib.Path("nope.json"),
                                 _merged(1, 2), False) is None)
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    sys.exit(main())
