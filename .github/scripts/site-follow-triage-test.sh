#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# 2026-09-27: Table-driven test for site-follow-triage.sh, the decision behind
# site-dispatch.yml's metrale.ai engine-pin PR. Seconds, no network.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
triage="$here/site-follow-triage.sh"
fails=0

A=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa   # the engine commit this run is for
B=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb   # an older engine commit
C=cccccccccccccccccccccccccccccccccccccccc   # a newer engine commit

check() {  # expected, why, target engine_head main_ref branch_ref behind ahead in_flight
  local expected="$1" why="$2" got
  shift 2
  got="$("$triage" "$@" 2>&1)" || got="EXIT$?"
  if [ "$got" != "$expected" ]; then
    printf 'FAIL  %-8s got %-8s (%s)\n' "$expected" "$got" "$why"
    fails=$((fails + 1))
  else
    printf 'ok    %-8s %s\n' "$expected" "$why"
  fi
}

# Only the run for the engine's current main may move the pin.
check stale   "a re-run of an older push never moves the pin back"      "$B" "$A" "$C" - 0 0 0
check stale   "stale wins even when the site already pins that commit"  "$B" "$A" "$B" - 0 0 0

# The site already reads this commit.
check current "pinned on main, no PR: nothing to do (idempotent)"        "$A" "$A" "$A" - 0 0 0
check close   "pinned on main by hand while a bump PR is open"           "$A" "$A" "$A" "$B" 0 1 0
check close   "pinned on main, open PR even mid-checks is superseded"    "$A" "$A" "$A" "$A" 3 1 2

# A new engine commit.
check build   "no bump PR open: open one"                                "$A" "$A" "$B" - 0 0 0
check build   "a PR for an older commit is rebuilt at once"              "$A" "$A" "$B" "$B" 0 1 0
check build   "...even while its checks run: the newest commit wins"     "$A" "$A" "$B" "$B" 0 1 4
check build   "...and when it is also behind"                            "$A" "$A" "$B" "$B" 2 1 4

# The PR already carries this commit.
check leave   "up to date and pinned right: re-running is a no-op"       "$A" "$A" "$B" "$A" 0 1 0
check leave   "checks running, not behind: leave them"                   "$A" "$A" "$B" "$A" 0 1 5
check wait    "behind main with checks running: no livelock"             "$A" "$A" "$B" "$A" 3 1 1
check build   "behind main and idle: rebuild from the new main"          "$A" "$A" "$B" "$A" 1 1 0

# A person pushed a fix to the bump branch (the PR comment asks them to): the
# hourly rebuild must not drop it. Only a new engine commit replaces the branch.
check leave   "behind and idle, but a person pushed: keep their commit"  "$A" "$A" "$B" "$A" 2 2 0
check leave   "a person pushed and it is not behind: leave it"           "$A" "$A" "$B" "$A" 0 3 0
check build   "a person pushed, then a new engine commit: rebuild"       "$A" "$A" "$B" "$B" 2 2 0

# Bad input is refused (exit 2), never read as a zero or a match.
check EXIT2   "a short sha is refused"                                    "${A:0:10}" "$A" "$B" - 0 0 0
check EXIT2   "an uppercase sha is refused"                               "${A^^}" "${A^^}" "$B" - 0 0 0
check EXIT2   "an empty engine.ref is refused, not read as 'differs'"     "$A" "$A" "" - 0 0 0
check EXIT2   "a branch_ref that is neither a sha nor - is refused"       "$A" "$A" "$B" none 0 1 0
check EXIT2   "a non-numeric behind_by is refused"                        "$A" "$A" "$B" "$A" x 1 0
check EXIT2   "a non-numeric ahead_by is refused"                         "$A" "$A" "$B" "$A" 0 -1 0
check EXIT2   "an empty in-flight count is refused"                       "$A" "$A" "$B" "$A" 0 1 ""
check EXIT2   "a missing argument is refused"                             "$A" "$A" "$B" "$A" 1 0

if [ "$fails" -gt 0 ]; then
  echo "$fails case(s) failed"
  exit 1
fi
echo "all cases pass"
