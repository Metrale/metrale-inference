#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0

# 2026-10-15: Open one certification PR per model a change left owed.
#
# A change that re-opens a gate certifies on at least one model it reaches; the
# others it reaches are owed (crates/bench/src/gate/check_models.rs). After the
# change lands, each owed model gets one PR: branch cert/<model>/<sha7> at the
# merge commit, carrying .benchmarks/owed/<model>-<sha>.toml. Which PRs to open
# is decided by the owed_certs binary (crates/bench/src/gate/owed_prs.rs) from
# the existing cert/* branches and PR heads; this script only lists those and
# executes what the binary prints.
#
# usage: owed-certifications.sh <base> <head> --dry-run|--open
#   --dry-run  list, decide, print the plan; write nothing (the PR CI job)
#   --open     also create the branches, manifests and draft PRs (post-merge)
# env: GH_TOKEN (read for --dry-run; contents+pull-requests write for --open),
#      REPO (owner/name), DEFAULT_BRANCH (the PRs' base, --open only)
#
# 2026-10-15: Every write runs under set -e with nothing suppressing it, so a
# missing App permission fails the job loudly instead of skipping a model.

set -euo pipefail

if [ "$#" -ne 3 ] || { [ "$3" != "--dry-run" ] && [ "$3" != "--open" ]; }; then
  echo "usage: $0 <base> <head> --dry-run|--open" >&2
  exit 2
fi
base=$1 head=$2 mode=$3
: "${REPO:?REPO is required}"
if [ "$mode" = "--open" ]; then
  : "${DEFAULT_BRANCH:?DEFAULT_BRANCH is required with --open}"
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# 2026-10-15: Both lists come from this repository only: a fork's PR whose head
# happens to be named cert/... must not stand in for ours. A failed listing
# fails the script; an empty list would open duplicates.
gh api --paginate "repos/$REPO/git/matching-refs/heads/cert/" \
  | jq -r '.[].ref | sub("^refs/heads/"; "")' > "$work/branches"
gh api --paginate "repos/$REPO/pulls?state=all&per_page=100" \
  | jq -r --arg repo "$REPO" \
      '.[] | select(.head.repo.full_name == $repo) | .head.ref | select(startswith("cert/"))' \
  > "$work/pr_heads"
echo "existing cert/* branches: $(wc -l < "$work/branches"), cert/* PRs: $(wc -l < "$work/pr_heads")"

cargo run --locked -q -p metrale-bench --bin owed_certs -- \
  --root . --base "$base" --head "$head" \
  --branches "$work/branches" --pr-heads "$work/pr_heads" > "$work/owed.json"

jq -r '"head \(.head), kernel map \(.kernel_map)",
       (.statuses | to_entries[] | "  \(.key): \(.value)"),
       "owed (\(.owed | length)):",
       (.owed[] | "  \(.gate) — \(.subject.checkpoint) on \(.subject.hw) (\(.subject.model))"),
       "PRs to open (\(.prs | length)):",
       (.prs[] | "  \(.branch) create_branch=\(.create_branch) \(.manifest_path)")' "$work/owed.json"

if [ "$mode" = "--dry-run" ]; then
  exit 0
fi

n=$(jq '.prs | length' "$work/owed.json")
for i in $(seq 0 $((n - 1))); do
  jq -r ".prs[$i].manifest" "$work/owed.json" > "$work/manifest"
  branch=$(jq -r ".prs[$i].branch" "$work/owed.json")
  path=$(jq -r ".prs[$i].manifest_path" "$work/owed.json")
  if [ "$(jq -r ".prs[$i].create_branch" "$work/owed.json")" = "true" ]; then
    gh api -X POST "repos/$REPO/git/refs" -f ref="refs/heads/$branch" -f sha="$head" > /dev/null
    echo "created $branch at $head"
  fi
  # 2026-10-15: A run that died after the PUT left the manifest on the branch;
  # only a 404 means it is absent, any other error stops here.
  if ! out=$(gh api "repos/$REPO/contents/$path?ref=$branch" --jq .sha 2>&1); then
    case "$out" in
      *"HTTP 404"*)
        gh api -X PUT "repos/$REPO/contents/$path" \
          -f message="cert: owed certifications after ${head:0:7}" \
          -f content="$(base64 -w0 < "$work/manifest")" \
          -f branch="$branch" > /dev/null
        echo "committed $path to $branch"
        ;;
      *)
        echo "::error title=Cannot read $path on $branch::$out"
        exit 1
        ;;
    esac
  fi
  jq --arg base "$DEFAULT_BRANCH" ".prs[$i] | {title, body, head: .branch, base: \$base, draft: true}" \
    "$work/owed.json" > "$work/pr.json"
  url=$(gh api -X POST "repos/$REPO/pulls" --input "$work/pr.json" --jq .html_url)
  echo "opened $url"
done
