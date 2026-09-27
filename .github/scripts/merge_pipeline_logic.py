# SPDX-License-Identifier: MIT OR Apache-2.0
"""The merge pipeline's decisions, as pure functions over plain data.

Nothing here performs I/O. `merge-pipeline.py` reads GitHub into the snapshot
shapes documented on each function, calls these, and applies what they return.
The self-test (`merge-pipeline-selftest.py`) drives them from fixture JSON, so
every rule below is exercised without a network, a token or a pull request.

Three questions are answered here:

  * `evaluate`     what `stamp status` and `seal status` should say on a PR's
                   head, whether the held certification lane must be re-run,
                   and whether the PR should be put into the merge queue.
  * `exclusivity`  whether a perf-path PR may stay in the merge queue.
  * `status_write` whether a check run has to be written at all, so that an
                   evaluator re-run on an unchanged PR writes nothing.
"""
from __future__ import annotations

import re

STAMP_STATUS = "stamp status"
SEAL_STATUS = "seal status"
OWN_STATUSES = (STAMP_STATUS, SEAL_STATUS)

# The marks minted by `certification-commands.yml`. An Expedite is an admin's
# waiver and stands in for both, as it always has in ci.yml.
STAMP_MARKS = ("Stamp", "Expedite")
SEAL_MARKS = ("Seal", "Expedite")

# The ci.yml jobs the held lane is made of. `stamp lookup` reads the marks once,
# when it runs, and the certification job is skipped when it read "not stamped".
LOOKUP_JOB = "stamp lookup"
CERTIFICATION_CHECK = "PR Benchmark Certifications"

PASSING = ("success", "skipped", "neutral")

_QUEUE_REF = re.compile(r"^(?:refs/heads/)?gh-readonly-queue/[^/]+(?:/[^/]+)*/pr-([0-9]+)-[0-9a-f]+$")
_PERF_DECL = re.compile(r"^pub const PERF_PATHS: \[&str; ([0-9]+)\] = \[$")
_PERF_ITEM = re.compile(r'^\s*"([^"]+)",?$')


# ── perf paths ────────────────────────────────────────────────────────────────


def parse_perf_paths(coverage_rs: str) -> list[str]:
    """Read `pub const PERF_PATHS` out of crates/bench/src/gate/coverage.rs.

    The list is defined there and nowhere else. There is deliberately no
    fallback: a short list would answer "not a perf path" for a path that is
    one. The declared array length must equal the number of entries read.
    """
    lines = coverage_rs.splitlines()
    for i, line in enumerate(lines):
        m = _PERF_DECL.match(line)
        if not m:
            continue
        want = int(m.group(1))
        items: list[str] = []
        for body in lines[i + 1:]:
            if body.strip() == "];":
                break
            item = _PERF_ITEM.match(body)
            if item:
                items.append(item.group(1))
            elif body.strip() and not body.strip().startswith("//"):
                raise ValueError(f"PERF_PATHS has an entry this parser cannot read: {body.strip()!r}")
        else:
            raise ValueError("PERF_PATHS is never closed with `];`")
        if len(items) != want or want == 0:
            raise ValueError(f"PERF_PATHS declares {want} entries but {len(items)} were read")
        return items
    raise ValueError("no `pub const PERF_PATHS: [&str; N] = [` line in coverage.rs")


def is_perf_path(path: str, perf_paths: list[str]) -> bool:
    """git pathspec semantics, which is how the gate applies the list: an entry
    names that exact path or everything beneath it. `crates` covers
    `crates/x.rs`, never `cratesx/y` and never `docs/crates/z`."""
    return any(path == p or path.startswith(p + "/") for p in perf_paths)


def touches_perf(files: list[str] | None, perf_paths: list[str]) -> bool:
    """`files` is None when the diff could not be listed in full (an API error
    or a truncated list). That answers True: the rule it feeds only ever holds
    a PR back from the queue, and the gate re-checks the records either way."""
    if files is None:
        return True
    return any(is_perf_path(f, perf_paths) for f in files)


def pr_from_queue_ref(ref: str) -> int | None:
    """The PR a merge-queue branch was built for. GitHub names them
    gh-readonly-queue/<base>/pr-<N>-<sha>. Anything else is None, never a guess."""
    m = _QUEUE_REF.match(ref or "")
    return int(m.group(1)) if m else None


# ── marks ─────────────────────────────────────────────────────────────────────


def _qualifying(marks: list[dict], names: tuple[str, ...], app_client_id: str,
                shas: set[str]) -> list[dict]:
    """A mark counts only if the certification App wrote it. A check run named
    `Seal` can be created by any workflow's GITHUB_TOKEN on its own head, so the
    name alone proves nothing."""
    return [m for m in marks
            if m.get("name") in names
            and m.get("conclusion") == "success"
            and app_client_id
            and m.get("app_client_id") == app_client_id
            and m.get("sha") in shas]


def _upper_layer(pr: dict) -> bool:
    base, default = pr.get("base_ref") or "", pr.get("default_branch") or ""
    return bool(base) and bool(default) and base != default


def _status(ok: bool, title: str, summary: str) -> dict:
    if ok:
        return {"status": "completed", "conclusion": "success", "title": title, "summary": summary}
    return {"status": "in_progress", "conclusion": None, "title": title, "summary": summary}


def mark_statuses(snap: dict) -> tuple[dict, dict, list[dict]]:
    """(stamp status, seal status, the qualifying stamp marks)."""
    pr = snap["pr"]
    head = pr["head_sha"]
    app = snap.get("mark_app_client_id") or ""
    marks = snap.get("marks") or []
    if _upper_layer(pr):
        why = (f"The base is `{pr['base_ref']}`, not `{pr['default_branch']}`: this layer "
               "merges down into the one beneath it, and the stack's base carries the marks.")
        return (_status(True, "Upper stack layer: not held", why),
                _status(True, "Upper stack layer: sealed at the base", why), [])

    # A stamp only releases runners, so it holds for every commit of the PR.
    stamps = _qualifying(marks, STAMP_MARKS, app, set(snap.get("commits") or []) | {head})
    # A seal vouches for a diff, so only the current head counts.
    seals = _qualifying(marks, SEAL_MARKS, app, {head})

    if stamps:
        first = min(stamps, key=lambda m: m.get("created_at") or "")
        stamp = _status(True, f"Stamped ({first['name']})",
                        f"{first.get('title') or first['name']} Recorded on `{first['sha'][:10]}`.")
    else:
        stamp = _status(False, "Waiting for /stamp",
                        "Certification is held until someone with write access, or the PR's "
                        "author, comments `/stamp`. This check completes by itself when they do.")
    if seals:
        seal = _status(True, f"Sealed ({seals[0]['name']})",
                       f"{seals[0].get('title') or seals[0]['name']} Recorded on `{head[:10]}`.")
    else:
        seal = _status(False, "Waiting for /seal on this commit",
                       "A codeowner of every path in the diff comments `/seal`. A seal vouches for "
                       "one diff, so the next commit needs a new one. This check completes by "
                       "itself when it is recorded.")
    return stamp, seal, stamps


# ── the held lane ─────────────────────────────────────────────────────────────


def lane_rerun(snap: dict, stamps: list[dict]) -> tuple[int | None, str]:
    """The job to re-run so the held certification lane sees the stamp, or None.

    `stamp lookup` reads the marks once. If every qualifying stamp was minted
    after it finished, it read "not stamped" and the certification job was
    skipped. Re-running that one job re-runs the jobs that depend on it, and
    only then. A run still in progress cannot be re-run; the next evaluation,
    which its completion triggers, does it.
    """
    ci = snap.get("ci")
    if not stamps or not ci:
        return None, "no stamp yet, or no CI run for this commit"
    job = ci.get("lookup_job") or {}
    if job.get("status") != "completed" or not job.get("completed_at"):
        return None, "the stamp lookup has not finished; it will read the mark itself"
    first_mark = min(m.get("created_at") or "" for m in stamps)
    if first_mark <= job["completed_at"]:
        return None, "the stamp lookup ran after the stamp and released the lane"
    cert = _latest(snap.get("head_checks") or [], CERTIFICATION_CHECK, None)
    if not cert or cert.get("conclusion") != "skipped":
        return None, "certification was not held, so there is nothing to release"
    if ci.get("status") != "completed":
        return None, "the CI run is still in progress; it is re-run when it completes"
    return int(job["id"]), "the stamp lookup finished before the stamp existed"


# ── the queue ─────────────────────────────────────────────────────────────────


def _latest(checks: list[dict], name: str, app_id: int | None) -> dict | None:
    hits = [c for c in checks if c.get("name") == name
            and (app_id is None or app_id in (-1, c.get("app_id")))]
    return max(hits, key=lambda c: (c.get("started_at") or "", c.get("id") or 0)) if hits else None


def required_green(required: list[dict], checks: list[dict], own: dict) -> tuple[bool, list[str]]:
    """Every required context has a passing latest run. `own` carries this
    evaluation's own statuses, which are written before the queue is asked."""
    missing = []
    for req in required:
        name, app_id = req["context"], req.get("app_id")
        if name in own:
            ok = own[name]["conclusion"] in PASSING
        else:
            run = _latest(checks, name, app_id)
            ok = bool(run) and run.get("status") == "completed" and run.get("conclusion") in PASSING
        if not ok:
            missing.append(name)
    return not missing, missing


def perf_ahead(me: int, entries: list[dict]) -> dict | None:
    """The first perf-path entry ahead of `me` in the queue, or None. Ahead
    means a lower position; on equal positions the earlier enqueue wins, then
    the lower PR number, so two guards that race always agree on one winner."""
    mine = next((e for e in entries if e["pr"] == me), None)
    if mine is None:
        return None
    key = lambda e: (e["position"], e.get("enqueued_at") or "", e["pr"])  # noqa: E731
    ahead = [e for e in entries if e["pr"] != me and e.get("perf") and key(e) < key(mine)]
    return min(ahead, key=key) if ahead else None


def exclusivity(me: int, my_perf: bool, entries: list[dict]) -> tuple[bool, str]:
    """(may stay, why). A PR that touches no perf path always may."""
    if not my_perf:
        return True, "touches no perf path; any number of these may share the queue"
    blocker = perf_ahead(me, entries)
    if blocker is None:
        return True, "the only perf-path PR in the queue, or the first of them"
    return False, (f"#{blocker['pr']} is ahead of it in the queue and also touches a perf path. "
                   "Records measured apart never compose: once it lands, main has moved on a perf "
                   "path and this PR's records no longer cover the merged tree, so it would bounce "
                   "after a full CI run. Enqueue it again once "
                   f"#{blocker['pr']} has merged.")


def queue_action(snap: dict, stamp: dict, seal: dict) -> tuple[str | None, str]:
    """None, "enqueue" or "merge-async", and why."""
    pr = snap["pr"]
    if _upper_layer(pr):
        return None, "an upper stack layer merges down, not into the queue"
    if pr.get("state") != "open" or pr.get("merged"):
        return None, "not an open pull request"
    if pr.get("draft"):
        return None, "a draft"
    if pr.get("in_merge_queue"):
        return None, "already in the merge queue"
    if pr.get("mergeable_state") == "dirty":
        return None, "conflicts with its base"
    if snap.get("dequeued_since_head"):
        # Whatever took it out -- a red merge group, the exclusivity guard, a
        # person -- putting it straight back would repeat that, once per
        # evaluation. A new commit, or a person enqueueing it, is the retry.
        return None, "left the merge queue since this commit was made; enqueue it by hand or push"
    if stamp["conclusion"] != "success" or seal["conclusion"] != "success":
        return None, "waiting for the stamp and the seal"
    if snap.get("required") is None:
        return None, "the default branch's required checks could not be read"
    own = {STAMP_STATUS: stamp, SEAL_STATUS: seal}
    green, missing = required_green(snap["required"], snap.get("head_checks") or [], own)
    if not green:
        return None, "required checks not green yet: " + ", ".join(missing)
    if snap.get("perf"):
        entries = list(snap.get("queue") or []) + [{"pr": pr["number"], "position": 1 << 30, "perf": True}]
        blocker = perf_ahead(pr["number"], entries)
        if blocker is not None:
            return None, f"#{blocker['pr']} is a perf-path PR already in the queue; one at a time"
    stack = snap.get("stack") or {}
    if stack.get("in_open_stack"):
        if not stack.get("is_bottom"):
            return None, "a stack layer that is not the stack's base"
        if stack.get("open_above", 0) > 0:
            return None, "the stack still has open layers above its base; merge them down first"
        return "merge-async", "the base of a stack, with every layer above it merged down"
    return "enqueue", "stamped, sealed, and every required check is green"


def evaluate(snap: dict) -> dict:
    stamp, seal, stamps = mark_statuses(snap)
    rerun, rerun_why = lane_rerun(snap, stamps)
    action, why = queue_action(snap, stamp, seal)
    return {"stamp_status": stamp, "seal_status": seal,
            "rerun_job": rerun, "rerun_reason": rerun_why,
            "queue_action": action, "queue_reason": why}


def status_write(desired: dict, existing: dict | None) -> str:
    """"create", "update" or "none". Writing an unchanged check run would fire a
    check_run event for nothing, and the evaluator listens to those."""
    if existing is None:
        return "create"
    # A completed check run is not reopened; a fresh one supersedes it.
    if existing.get("status") == "completed" and desired["status"] != "completed":
        return "create"
    same = (existing.get("status") == desired["status"]
            and existing.get("conclusion") == desired["conclusion"]
            and existing.get("title") == desired["title"])
    return "none" if same else "update"
