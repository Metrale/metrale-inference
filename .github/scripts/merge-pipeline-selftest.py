#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Self-test for merge_pipeline_logic.py: the stamp/seal evaluator, the queue
decision, the held-lane re-run and perf-path exclusivity.

Offline, no token, no network. Every rule is driven from the fixture JSON in
merge-pipeline-fixtures/, and every group carries NEGATIVE controls: inputs the
rule must refuse, not only inputs it should accept.

A suite that has only ever passed cannot be told apart from one that cannot
fail. So the last section MUTATES the logic -- one plausible bug at a time, in
a copy -- re-runs this suite against the copy, and requires it to FAIL. A
mutation the suite survives is reported as a failure of the suite.

  merge-pipeline-selftest.py                  run everything, mutations included
  merge-pipeline-selftest.py --logic FILE     run the cases against FILE only
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIXTURES = HERE / "merge-pipeline-fixtures"
LOGIC = HERE / "merge_pipeline_logic.py"
COVERAGE = ROOT / "crates" / "bench" / "src" / "gate" / "coverage.rs"

# (what the bug is, text in the logic, text it becomes). Each must turn at
# least one case red.
MUTATIONS = [
    ("marks from any app are trusted",
     "and m.get(\"app_client_id\") == app_client_id", "and True"),
    ("a seal carries across commits like a stamp",
     "seals = _qualifying(marks, SEAL_MARKS, app, {head})",
     "seals = _qualifying(marks, SEAL_MARKS, app, set(snap.get(\"commits\") or []) | {head})"),
    ("a missing required check reads as green",
     "ok = bool(run) and run.get(\"status\")", "ok = (not run) or run.get(\"status\")"),
    ("the wrong app's check satisfies a required context",
     "and (app_id is None or app_id in (-1, c.get(\"app_id\")))", ""),
    ("exclusivity ignores queue order",
     "and key(e) < key(mine)]", "]"),
    ("perf paths match on a bare prefix",
     "path.startswith(p + \"/\")", "path.startswith(p)"),
    ("an in-progress CI run is re-run",
     "if ci.get(\"status\") != \"completed\":", "if False:"),
    ("an upper stack layer is enqueued",
     "return None, \"an upper stack layer merges down, not into the queue\"", "pass"),
    ("a draft is enqueued", "if pr.get(\"draft\"):", "if False:"),
    ("a PR thrown out of the queue is put straight back", "if snap.get(\"dequeued_since_head\"):", "if False:"),
    ("unreadable required checks read as none required",
     "if snap.get(\"required\") is None:", "if False:"),
    ("an unchanged check run is rewritten", "return \"none\" if same else \"update\"", "return \"update\""),
    ("a queue ref that names no PR is guessed", "return int(m.group(1)) if m else None",
     "return int(m.group(1)) if m else 0"),
]


def load(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location("merge_pipeline_logic", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def merge(base, patch):
    if isinstance(base, dict) and isinstance(patch, dict):
        out = dict(base)
        for k, v in patch.items():
            out[k] = merge(base.get(k), v) if k in base else copy.deepcopy(v)
        return out
    return copy.deepcopy(patch)


class Suite:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def check(self, label: str, cond: bool, detail: str = "") -> None:
        if cond:
            self.passed += 1
            print(f"  ok   {label}")
        else:
            self.failed += 1
            print(f"  FAIL {label}{(' -- ' + detail) if detail else ''}")


def shape(status: dict) -> str:
    return "pending" if status["status"] != "completed" else status["conclusion"]


def run_cases(logic, s: Suite) -> None:
    cases = json.loads((FIXTURES / "cases.json").read_text())
    base = json.loads((FIXTURES / "base-snapshot.json").read_text())

    print("== evaluate: stamp status, seal status, re-run, queue ==")
    for case in cases["evaluate"]:
        d = logic.evaluate(merge(base, case["patch"]))
        got = {"stamp": shape(d["stamp_status"]), "seal": shape(d["seal_status"]),
               "rerun": d["rerun_job"], "queue": d["queue_action"]}
        exp = case["expect"]
        bad = [f"{k}={got[k]!r} want {exp[k]!r}" for k in ("stamp", "seal", "rerun", "queue")
               if k in exp and got[k] != exp[k]]
        if "queue_reason" in exp and exp["queue_reason"] not in d["queue_reason"]:
            bad.append(f"queue_reason={d['queue_reason']!r} lacks {exp['queue_reason']!r}")
        # A pending check must never carry a conclusion: GitHub would show it red.
        for key in ("stamp_status", "seal_status"):
            st = d[key]
            if st["status"] != "completed" and st["conclusion"] is not None:
                bad.append(f"{key} is pending with conclusion {st['conclusion']!r}")
        s.check(case["name"], not bad, "; ".join(bad))

    print("== status_write: idempotent check-run writes ==")
    for case in cases["status_write"]:
        got = logic.status_write(case["desired"], case["existing"])
        s.check(case["name"], got == case["expect"], f"got {got!r}")

    print("== exclusivity: one perf-path PR in the queue ==")
    for case in cases["exclusivity"]:
        ok, why = logic.exclusivity(case["me"], case["my_perf"], case["entries"])
        s.check(case["name"], ok == case["expect"], why)

    print("== perf paths: the gate's PERF_PATHS, read from coverage.rs ==")
    perf = logic.parse_perf_paths(COVERAGE.read_text())
    s.check("PERF_PATHS parses from crates/bench/src/gate/coverage.rs", len(perf) > 0 and "crates" in perf,
            f"got {perf}")
    for case in cases["perf_paths"]:
        got = logic.is_perf_path(case["path"], perf)
        s.check(f"{case['path']} is {'' if case['expect'] else 'not '}a perf path", got == case["expect"])
    s.check("an unlistable diff counts as touching a perf path", logic.touches_perf(None, perf) is True)
    s.check("control: an empty diff touches no perf path", logic.touches_perf([], perf) is False)

    good = "pub const PERF_PATHS: [&str; 2] = [\n    \"crates\",\n    \"kernels\",\n];\n"
    s.check("a well-formed declaration parses", logic.parse_perf_paths(good) == ["crates", "kernels"])
    for label, text in (
        ("control: a declared length that disagrees with the entries is refused",
         good.replace("[&str; 2]", "[&str; 3]")),
        ("control: a missing declaration is refused, with no fallback list", "fn main() {}\n"),
        ("control: an entry the parser cannot read is refused",
         good.replace("    \"kernels\",\n", "    KERNELS,\n")),
        ("control: an unterminated list is refused", good.replace("];\n", "")),
    ):
        try:
            logic.parse_perf_paths(text)
            s.check(label, False, "it parsed")
        except ValueError:
            s.check(label, True)

    print("== queue refs: which PR a merge group was built for ==")
    for case in cases["queue_refs"]:
        got = logic.pr_from_queue_ref(case["ref"])
        s.check(f"{case['ref'] or '(empty)'} -> {case['expect']}", got == case["expect"], f"got {got!r}")


def run_mutations(s: Suite) -> None:
    print("== mutation controls: each plausible bug must turn this suite red ==")
    source = LOGIC.read_text()
    with tempfile.TemporaryDirectory() as tmp:
        for label, old, new in MUTATIONS:
            if source.count(old) != 1:
                s.check(f"control: {label}", False, f"the mutation anchor occurs {source.count(old)} times")
                continue
            mutant = pathlib.Path(tmp) / "merge_pipeline_logic.py"
            mutant.write_text(source.replace(old, new))
            # A mutant that does not even compile fails for the wrong reason and
            # would make its control pass vacuously.
            try:
                compile(mutant.read_text(), str(mutant), "exec")
            except SyntaxError as e:
                s.check(f"control: {label}", False, f"the mutant does not compile: {e}")
                continue
            proc = subprocess.run([sys.executable, __file__, "--logic", str(mutant)],
                                  capture_output=True, text=True)
            s.check(f"control: the suite fails when {label}", proc.returncode == 1,
                    f"exit {proc.returncode}" + (f": {proc.stderr.strip()[-200:]}" if proc.stderr else ""))
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logic", type=pathlib.Path)
    args = ap.parse_args()
    s = Suite()
    try:
        run_cases(load(args.logic or LOGIC), s)
    except Exception as e:  # a crash is a failure of the logic under test, never a pass
        s.check("the logic ran without raising", False, f"{type(e).__name__}: {e}")
    if args.logic is None:
        run_mutations(s)
    print(f"\n  {s.passed} passed, {s.failed} failed")
    return 1 if s.failed else 0


if __name__ == "__main__":
    sys.exit(main())
