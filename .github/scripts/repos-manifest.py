#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Build this repository's numbers-only status manifest for the internal repos dashboard.

Run from a checkout of main with full history (CI: the `repos` job of site-dispatch.yml). Reads
`.benchmarks/` and the measured paths of crates/bench/src/gate/coverage.rs from the checkout,
and asks GitHub, through `gh api` (GH_TOKEN), for the certification check on main's head and on
open pull requests.

    .github/scripts/repos-manifest.py --checkout . --slug Metrale/metrale-inference --repo engine > manifest.json

Definitions (the same as the roadmap generator's `_campaign`):
  campaign  the commit named by the most recently recorded record; its gates are the
            gates with a record at that commit; a sharded group counts once as a gate
            and each shard as a unit.
Only numbers, shas, ids and fixed state words are emitted (the dashboard refuses anything else).
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys
import time

CHECK = "PR benchmark gate"
STATES = {"success", "failure", "cancelled", "skipped", "neutral"}


def git(checkout, *args):
    return subprocess.run(["git", "-C", str(checkout), *args], check=True, capture_output=True, text=True).stdout.strip()


def gh(path):
    out = subprocess.run(["gh", "api", path], check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def check_state(slug, sha):
    runs = gh(f"repos/{slug}/commits/{sha}/check-runs?check_name={CHECK.replace(' ', '%20')}&filter=latest&per_page=10")
    runs = runs.get("check_runs", [])
    if not runs:
        return "missing"
    run = max(runs, key=lambda r: r.get("started_at") or "")
    if run.get("status") != "completed":
        return "pending"
    c = run.get("conclusion")
    return c if c in STATES else "failure"


def records(checkout):
    out = []
    for p in sorted((checkout / ".benchmarks").glob("*/*.json")):
        d = json.loads(p.read_text())
        out.append({"gate": p.parent.name, "sha": d["git_sha"], "at": int(d["recorded_at"]), "verdict": str(d.get("verdict", "unknown"))})
    return out


def perf_paths(checkout):
    f = checkout / "crates/bench/src/gate/coverage.rs"
    if not f.exists():
        return None
    m = re.search(r"pub const PERF_PATHS: \[&str; \d+\] = \[(.*?)\];", f.read_text(), re.S)
    return re.findall(r'"([^"]+)"', m.group(1)) if m else None


def campaign(checkout, recs, head):
    if not recs:
        return None, []
    newest = max(recs, key=lambda r: r["at"])
    sha = newest["sha"]
    mine = [r for r in recs if r["sha"] == sha]
    gates = {}
    for r in mine:
        gates.setdefault(r["gate"], []).append(r)
    single = [rs for rs in gates.values() if len(rs) == 1]
    code_equal, changed = None, None
    paths = perf_paths(checkout)
    if paths:
        try:
            diff = git(checkout, "diff", "--name-only", f"{sha}..{head}", "--", *paths)
            changed = len([l for l in diff.splitlines() if l])
            code_equal = changed == 0
        except subprocess.CalledProcessError:
            code_equal, changed = None, None
    camp = {
        "sha": sha.lower(),
        "recorded_at": max(r["at"] for r in mine),
        "gates": len(gates),
        "units": len(mine),
        "single_gates": len(single),
        "single_gates_passed": sum(1 for rs in single if rs[0]["verdict"] == "PASS"),
        "group_gates": len(gates) - len(single),
        "code_equal_to_main": code_equal,
        "changed_perf_files": changed,
    }
    gate_list = []
    for gid in sorted(gates):
        counts = {}
        for r in gates[gid]:
            v = re.sub(r"[^A-Za-z]", "", r["verdict"])[:16] or "unknown"
            counts[v] = counts.get(v, 0) + 1
        gate_list.append({"id": gid, "units": len(gates[gid]), "verdicts": counts})
    return camp, gate_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkout", required=True, type=pathlib.Path)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--repo", required=True)
    a = ap.parse_args()
    head = git(a.checkout, "rev-parse", "HEAD")
    committed_at = int(git(a.checkout, "log", "-1", "--format=%ct", "HEAD"))
    camp, gates = campaign(a.checkout, records(a.checkout), head)
    pending = []
    for pr in gh(f"repos/{a.slug}/pulls?state=open&per_page=100"):
        if pr.get("draft"):
            continue
        state = check_state(a.slug, pr["head"]["sha"])
        if state != "success":
            pending.append({"number": int(pr["number"]), "head_sha": pr["head"]["sha"].lower(), "check": state})
    manifest = {
        "schema": 1,
        "repo": a.repo,
        "source": a.slug,
        "generated_at": int(time.time()),
        "main": {"sha": head.lower(), "committed_at": committed_at},
        "main_certification_check": check_state(a.slug, head),
        "campaign": camp,
        "gates": gates,
        "prs_awaiting_certification": sorted(pending, key=lambda p: p["number"]),
    }
    json.dump(manifest, sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
