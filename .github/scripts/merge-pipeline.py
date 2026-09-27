#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The merge pipeline's I/O: read GitHub, ask merge_pipeline_logic, apply.

  merge-pipeline.py evaluate --pr N [--pr M ...] | --sha SHA | --all
      Write `stamp status` / `seal status` on each PR's head, re-run the held
      certification lane when the stamp arrived after it read the marks, and
      put the PR into the merge queue once it is stamped, sealed and green.
  merge-pipeline.py exclusivity --queue-ref REF
      The `perf-path exclusivity` check on a merge-queue entry.
  merge-pipeline.py guard-enqueued --pr N
      Take a second perf-path PR back out of the queue, with a comment.

Every decision is made in merge_pipeline_logic.py; this file only moves data.
`--dry-run` prints what would be written and writes nothing.

Environment (all required, no defaults): GH_TOKEN, REPO, and for `evaluate`
MARK_APP_CLIENT_ID, the client id of the App that mints Stamp/Seal/Expedite.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import merge_pipeline_logic as logic  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
COVERAGE = ROOT / "crates" / "bench" / "src" / "gate" / "coverage.rs"
PAGE = 100
PR_FILES_CAP = 3000


class ApiError(RuntimeError):
    pass


def env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        sys.exit(f"merge-pipeline: {name} is not set")
    return value


def gh(*args: str, stdin: str | None = None) -> str:
    proc = subprocess.run(["gh", *args], input=stdin, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ApiError(f"gh {' '.join(args[:3])}: {proc.stderr.strip() or proc.stdout.strip()}")
    return proc.stdout


def get(path: str) -> dict | list:
    return json.loads(gh("api", path))


def paged(path: str, key: str | None = None, cap: int | None = None) -> list:
    """Every page, joined. `gh --paginate` prints one JSON document per page,
    which a single json.loads cannot read, so pages are walked here."""
    out: list = []
    sep = "&" if "?" in path else "?"
    page = 1
    while True:
        doc = get(f"{path}{sep}per_page={PAGE}&page={page}")
        items = doc[key] if key else doc
        out.extend(items)
        if len(items) < PAGE or (cap is not None and len(out) >= cap):
            return out
        page += 1


def graphql(query: str, **variables: str | int) -> dict:
    args = ["api", "graphql", "-f", f"query={query}"]
    for k, v in variables.items():
        args += ["-F" if isinstance(v, int) else "-f", f"{k}={v}"]
    doc = json.loads(gh(*args))
    if doc.get("errors"):
        raise ApiError(json.dumps(doc["errors"]))
    return doc["data"]


# ── reads ─────────────────────────────────────────────────────────────────────


def perf_paths() -> list[str]:
    return logic.parse_perf_paths(COVERAGE.read_text())


def pr_files(repo: str, n: int) -> list[str] | None:
    try:
        files = [f["filename"] for f in paged(f"repos/{repo}/pulls/{n}/files", cap=PR_FILES_CAP)]
    except ApiError as e:
        print(f"::warning::could not list the files of #{n}: {e}")
        return None
    return None if len(files) >= PR_FILES_CAP else files


def queue_entries(repo: str, branch: str, perf: list[str]) -> list[dict]:
    owner, name = repo.split("/", 1)
    q = ("query($o:String!,$r:String!,$b:String!){repository(owner:$o,name:$r){"
         "mergeQueue(branch:$b){entries(first:100){nodes{position enqueuedAt "
         "pullRequest{number}}}}}}")
    data = graphql(q, o=owner, r=name, b=branch)
    nodes = ((data["repository"].get("mergeQueue") or {}).get("entries") or {}).get("nodes") or []
    return [{"pr": e["pullRequest"]["number"], "position": e["position"],
             "enqueued_at": e["enqueuedAt"],
             "perf": logic.touches_perf(pr_files(repo, e["pullRequest"]["number"]), perf)}
            for e in nodes]


def check_runs(repo: str, sha: str, name: str | None = None) -> list[dict]:
    path = f"repos/{repo}/commits/{sha}/check-runs" + (f"?check_name={name}" if name else "")
    return [{"id": r["id"], "name": r["name"], "status": r["status"], "conclusion": r["conclusion"],
             "sha": sha, "app_id": (r.get("app") or {}).get("id"),
             "app_client_id": (r.get("app") or {}).get("client_id"),
             "started_at": r.get("started_at"), "created_at": r.get("completed_at") or r.get("started_at"),
             "title": (r.get("output") or {}).get("title")}
            for r in paged(path, key="check_runs")]


def statuses(repo: str, sha: str) -> list[dict]:
    doc = get(f"repos/{repo}/commits/{sha}/status")
    state = {"success": "success", "failure": "failure", "error": "failure", "pending": None}
    return [{"id": 0, "name": s["context"], "status": "completed" if s["state"] != "pending" else "in_progress",
             "conclusion": state.get(s["state"]), "app_id": None, "started_at": s.get("updated_at")}
            for s in doc.get("statuses") or []]


def lookup_job(repo: str, sha: str) -> dict | None:
    runs = [r for r in get(f"repos/{repo}/actions/runs?head_sha={sha}&event=pull_request&per_page=50")
            ["workflow_runs"] if r["name"] == "CI"]
    if not runs:
        return None
    run = max(runs, key=lambda r: r["created_at"])
    jobs = get(f"repos/{repo}/actions/runs/{run['id']}/jobs?per_page=100")["jobs"]
    job = next((j for j in jobs if j["name"] == logic.LOOKUP_JOB), None)
    return {"run_id": run["id"], "status": run["status"],
            "lookup_job": None if job is None else
            {"id": job["id"], "status": job["status"], "completed_at": job.get("completed_at")}}


def stack_of(repo: str, n: int) -> dict:
    for s in paged(f"repos/{repo}/stacks"):
        if not s.get("open"):
            continue
        members = s.get("pull_requests") or []
        numbers = [m["number"] for m in members]
        if n in numbers:
            i = numbers.index(n)
            return {"in_open_stack": True, "is_bottom": i == 0,
                    "open_above": sum(1 for m in members[i + 1:] if m.get("state") == "open")}
    return {"in_open_stack": False}


def snapshot(repo: str, n: int, app_client_id: str, perf: list[str]) -> dict:
    pr = get(f"repos/{repo}/pulls/{n}")
    head, default = pr["head"]["sha"], pr["base"]["repo"]["default_branch"]
    owner, name = repo.split("/", 1)
    in_queue = graphql("query($o:String!,$r:String!,$n:Int!){repository(owner:$o,name:$r){"
                       "pullRequest(number:$n){isInMergeQueue}}}", o=owner, r=name, n=n)
    commits = [c["sha"] for c in paged(f"repos/{repo}/pulls/{n}/commits")]
    head_checks = check_runs(repo, head) + statuses(repo, head)
    marks = [c for c in head_checks if c["name"] in logic.STAMP_MARKS + logic.SEAL_MARKS]
    if not any(m["name"] in logic.STAMP_MARKS for m in marks):
        # A stamp holds across commits: look back, newest first, until one is found.
        for sha in reversed(commits[:-1]):
            found = [c for name_ in logic.STAMP_MARKS for c in check_runs(repo, sha, name_)]
            if found:
                marks += found
                break
    try:
        protection = get(f"repos/{repo}/branches/{default}").get("protection") or {}
        required = (protection.get("required_status_checks") or {}).get("checks")
    except ApiError as e:
        print(f"::warning::could not read the required checks of {default}: {e}")
        required = None
    my_perf = logic.touches_perf(pr_files(repo, n), perf)
    made = get(f"repos/{repo}/commits/{head}")["commit"]["committer"]["date"]
    dequeued = any(e.get("event") == "removed_from_merge_queue" and (e.get("created_at") or "") > made
                   for e in paged(f"repos/{repo}/issues/{n}/timeline"))
    return {
        "pr": {"number": n, "state": pr["state"], "draft": pr.get("draft", False),
               "merged": pr.get("merged", False), "head_sha": head, "base_ref": pr["base"]["ref"],
               "default_branch": default, "mergeable_state": pr.get("mergeable_state"),
               "in_merge_queue": bool(in_queue["repository"]["pullRequest"]["isInMergeQueue"]),
               "node_id": pr["node_id"]},
        "mark_app_client_id": app_client_id,
        "commits": commits,
        "marks": marks,
        "head_checks": head_checks,
        "required": required,
        "ci": lookup_job(repo, head),
        "perf": my_perf,
        "queue": queue_entries(repo, default, perf) if my_perf else [],
        "stack": stack_of(repo, n),
        "dequeued_since_head": dequeued,
    }


# ── writes ────────────────────────────────────────────────────────────────────


def write_status(repo: str, sha: str, name: str, desired: dict, existing: dict | None, dry: bool) -> str:
    action = logic.status_write(desired, existing)
    if action == "none" or dry:
        return action
    fields = ["-f", f"status={desired['status']}",
              "-f", f"output[title]={desired['title']}", "-f", f"output[summary]={desired['summary']}"]
    if desired["conclusion"]:
        fields += ["-f", f"conclusion={desired['conclusion']}"]
    if action == "create":
        gh("api", "-X", "POST", f"repos/{repo}/check-runs", "-f", f"name={name}", "-f", f"head_sha={sha}", *fields)
    else:
        gh("api", "-X", "PATCH", f"repos/{repo}/check-runs/{existing['id']}", *fields)
    return action


def apply(repo: str, snap: dict, decision: dict, dry: bool) -> None:
    pr, app = snap["pr"], snap["mark_app_client_id"]
    for name, key in ((logic.STAMP_STATUS, "stamp_status"), (logic.SEAL_STATUS, "seal_status")):
        mine = [c for c in snap["head_checks"] if c["name"] == name and c.get("app_client_id") == app]
        existing = max(mine, key=lambda c: c["id"]) if mine else None
        done = write_status(repo, pr["head_sha"], name, decision[key], existing, dry)
        print(f"  {name}: {decision[key]['title']} ({done})")
    if decision["rerun_job"] is not None:
        print(f"  re-run job {decision['rerun_job']}: {decision['rerun_reason']}")
        if not dry:
            gh("api", "-X", "POST", f"repos/{repo}/actions/jobs/{decision['rerun_job']}/rerun")
    action = decision["queue_action"]
    print(f"  queue: {action or 'no'} ({decision['queue_reason']})")
    if dry or action is None:
        return
    try:
        if action == "merge-async":
            gh("api", "-X", "PUT", f"repos/{repo}/pulls/{pr['number']}/merge-async",
               "-f", "merge_action=merge_queue")
        else:
            graphql("mutation($id:ID!,$sha:GitObjectID!){enqueuePullRequest(input:"
                    "{pullRequestId:$id,expectedHeadOid:$sha}){mergeQueueEntry{position}}}",
                    id=pr["node_id"], sha=pr["head_sha"])
    except ApiError as e:
        # GitHub is the last word on whether a PR may enter the queue. A refusal
        # here is retried by the next evaluation, so it is reported, not fatal.
        print(f"::warning title=#{pr['number']} was not enqueued::{e}")


def prs_for_sha(repo: str, sha: str) -> list[int]:
    return [p["number"] for p in get(f"repos/{repo}/commits/{sha}/pulls") if p.get("state") == "open"]


def cmd_evaluate(args: argparse.Namespace) -> int:
    repo, app = env("REPO"), env("MARK_APP_CLIENT_ID")
    perf = perf_paths()
    if args.all:
        numbers = [p["number"] for p in paged(f"repos/{repo}/pulls?state=open")]
    elif args.sha:
        numbers = prs_for_sha(repo, args.sha)
    else:
        numbers = args.pr or []
    failed = 0
    for n in numbers:
        print(f"#{n}")
        try:
            snap = snapshot(repo, n, app, perf)
            if snap["pr"]["state"] != "open":
                print("  closed; nothing to do")
                continue
            apply(repo, snap, logic.evaluate(snap), args.dry_run)
        except ApiError as e:
            failed += 1
            print(f"::error title=Merge pipeline could not evaluate #{n}::{e}")
    return 1 if failed else 0


def exclusivity_for(repo: str, n: int) -> tuple[bool, str]:
    perf = perf_paths()
    pr = get(f"repos/{repo}/pulls/{n}")
    mine = logic.touches_perf(pr_files(repo, n), perf)
    if not mine:
        return logic.exclusivity(n, False, [])
    return logic.exclusivity(n, True, queue_entries(repo, pr["base"]["ref"], perf))


def cmd_exclusivity(args: argparse.Namespace) -> int:
    repo = env("REPO")
    n = logic.pr_from_queue_ref(args.queue_ref)
    if n is None:
        print(f"::warning::'{args.queue_ref}' does not name a queued PR; exclusivity is not judged here")
        return 0
    try:
        ok, why = exclusivity_for(repo, n)
    except ApiError as e:
        # An efficiency guard, not a correctness one: the certification job in
        # this same merge group still refuses a tree its records do not cover.
        print(f"::warning::could not judge exclusivity for #{n}, letting it through: {e}")
        return 0
    print(f"#{n}: {why}")
    if not ok:
        print(f"::error title=Another perf-path PR is ahead in the queue::{why}")
    return 0 if ok else 1


def cmd_guard(args: argparse.Namespace) -> int:
    repo, n = env("REPO"), args.pr
    ok, why = exclusivity_for(repo, n)
    print(f"#{n}: {why}")
    if ok or args.dry_run:
        return 0
    node = get(f"repos/{repo}/pulls/{n}")["node_id"]
    graphql("mutation($id:ID!){dequeuePullRequest(input:{id:$id}){clientMutationId}}", id=node)
    gh("api", "-X", "POST", f"repos/{repo}/issues/{n}/comments", "-f",
       f"body=**Taken out of the merge queue.** {why}\n\nOnly one pull request that touches a perf "
       "path (`PERF_PATHS` in `crates/bench/src/gate/coverage.rs`) is in the queue at a time.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("evaluate")
    who = ev.add_mutually_exclusive_group(required=True)
    who.add_argument("--pr", type=int, action="append")
    who.add_argument("--sha")
    who.add_argument("--all", action="store_true")
    ev.add_argument("--dry-run", action="store_true")
    ex = sub.add_parser("exclusivity")
    ex.add_argument("--queue-ref", required=True)
    gd = sub.add_parser("guard-enqueued")
    gd.add_argument("--pr", type=int, required=True)
    gd.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    return {"evaluate": cmd_evaluate, "exclusivity": cmd_exclusivity, "guard-enqueued": cmd_guard}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
