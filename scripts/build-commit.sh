#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# 2026-10-11: Print the commit a release or image build of `met` is stamped with: the full
# 40-hex commit of HEAD. The build passes it as METRALE_BUILD_COMMIT (cargo) or
# --build-arg METRALE_GIT_SHA (docker/*/Dockerfile). The server keys every
# InferenceUsage record and the /metrics build line by it, and an unstamped
# binary withholds the records (investor-mvp #28).
#
# It refuses, printing nothing on stdout, when HEAD is not a commit or when a
# tracked file differs from HEAD: a binary built from that tree is not the
# commit, and stamping it as one would be a false record. Untracked files do not
# count (build outputs live in target/).
#
# Usage: bash scripts/build-commit.sh [repo-dir]    (default: the current directory)
set -euo pipefail

dir="${1:-.}"
commit="$(git -C "$dir" rev-parse --verify --quiet 'HEAD^{commit}')" || {
  echo "build-commit: $dir has no HEAD commit" >&2
  exit 1
}
printf '%s' "$commit" | grep -Eqx '[0-9a-f]{40}' || {
  echo "build-commit: HEAD is '$commit', not a 40-hex commit" >&2
  exit 1
}
dirty="$(git -C "$dir" status --porcelain --untracked-files=no)"
if [ -n "$dirty" ]; then
  echo "build-commit: tracked files differ from $commit; refusing to stamp:" >&2
  printf '%s\n' "$dirty" | head -20 >&2
  exit 1
fi
printf '%s\n' "$commit"
