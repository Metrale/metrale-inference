#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# 2026-09-27: site-follow-triage.sh -- decide what site-dispatch.yml does about
# metrale.ai's engine pin (`site/engine.ref`) for one run. Prints exactly one word:
#
#   stale    -- this run's commit is no longer the engine's main (a re-run of an
#               older push); do nothing, the run for the newer commit carries it
#   current  -- metrale.ai's main already pins this commit and no bump PR is open
#   close    -- metrale.ai's main already pins this commit but a bump PR is still
#               open (a hand bump got there first); close the PR
#   build    -- (re)build the bump branch from metrale.ai's main at this commit
#   wait     -- the open bump PR is behind main but its checks are still running
#   leave    -- the open bump PR is up to date with this commit, or a person has pushed
#               to it; do nothing
#
# Usage: site-follow-triage.sh <target> <engine_head> <main_ref> <branch_ref|-> <behind_by> <ahead_by> <in_flight>
#
#   target       the engine commit this run is for (github.sha)
#   engine_head  the engine's main as the run reads it now
#   main_ref     site/engine.ref on metrale.ai's main
#   branch_ref   site/engine.ref at the head of the OPEN bump PR, `-` when none is open
#   behind_by    commits metrale.ai's main has that the PR head lacks (0 when none open)
#   ahead_by     commits the PR head has that main lacks; the bump itself is 1 (0 when none open)
#   in_flight    the PR head's check runs that have not completed (0 when none open)
set -euo pipefail

if [ "$#" -ne 7 ]; then
  echo "usage: $0 <target> <engine_head> <main_ref> <branch_ref|-> <behind_by> <ahead_by> <in_flight>" >&2
  exit 2
fi
target=$1 engine_head=$2 main_ref=$3 branch_ref=$4 behind=$5 ahead=$6 in_flight=$7

is_sha() { [[ "$1" =~ ^[0-9a-f]{40}$ ]]; }
for pair in "target:$target" "engine_head:$engine_head" "main_ref:$main_ref"; do
  is_sha "${pair#*:}" || { echo "${pair%%:*} must be a full lowercase commit sha, got '${pair#*:}'" >&2; exit 2; }
done
[ "$branch_ref" = - ] || is_sha "$branch_ref" \
  || { echo "branch_ref must be a full lowercase commit sha or -, got '$branch_ref'" >&2; exit 2; }
# 2026-09-27: Each count on its own: checked concatenated, `0` and `` read as `0`.
for count in "$behind" "$ahead" "$in_flight"; do
  case "$count" in
    '' | *[!0-9]*) echo "counts must be non-negative integers, got '$behind', '$ahead' and '$in_flight'" >&2; exit 2 ;;
  esac
done

# 2026-09-27: A re-run of an old push would otherwise point the site BACK at an
# older engine commit. Only the run whose commit is main right now may move the pin.
if [ "$target" != "$engine_head" ]; then
  echo stale
  exit 0
fi

if [ "$main_ref" = "$target" ]; then
  if [ "$branch_ref" = - ]; then echo current; else echo close; fi
  exit 0
fi

# 2026-09-27: No PR, or one for an older engine commit: the newest commit wins
# at once. Restarting that PR's checks is the point -- successive engine merges
# converge on the last of them instead of queueing one site PR each.
if [ "$branch_ref" != "$target" ]; then
  echo build
  exit 0
fi

# 2026-09-27: The right commit, but metrale.ai's main moved under it. Its branch
# protection is strict and GitHub does not update a branch for auto-merge, so it
# would sit unmergeable. Rebuild it -- but only once its checks are idle:
# rebuilding on sight restarts them on every unrelated merge to the site and the
# PR never lands (the livelock harvest-triage.sh guards against). And not
# at all once a person has pushed to it (more than the one bump commit): the PR
# comment tells them to fix a red bump there, and a rebuild would drop the fix.
# Updating the branch is theirs then; the next engine commit rebuilds it anyway.
if [ "$behind" -gt 0 ] && [ "$ahead" -le 1 ]; then
  if [ "$in_flight" -gt 0 ]; then echo wait; else echo build; fi
  exit 0
fi

# 2026-09-27: Up to date and pinned right. A red PR is also left alone: it waits
# for a person, and the next engine merge rebuilds it anyway.
echo leave
