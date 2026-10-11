#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# 2026-10-11: Refresh the vendored platform-contract bundle (investor-mvp #41) in one command:
# replace `schemas/platform/` with Metrale/mfs `schemas/platform/` at <mfs-commit>,
# unchanged, and rewrite `schemas/platform.pin` to that commit and the SHA-256 of the
# copy's `SHA256SUMS`. The copy is checked with `sha256sum -c` before the old one is
# replaced, so a failed refresh leaves the repository as it was.
# `crates/qci-dump/tests/platform_contracts.rs` then holds the runtime's binding to the
# new fixtures; run it after a refresh.
#
# Usage: bash scripts/vendor-platform-schemas.sh <path-to-an-mfs-clone> <mfs-commit>
set -euo pipefail

[ "$#" -eq 2 ] || { echo "usage: $0 <path-to-an-mfs-clone> <mfs-commit>" >&2; exit 2; }
mfs="$1"
commit="$(git -C "$mfs" rev-parse --verify --quiet "$2^{commit}")" || {
  echo "vendor: $2 is not a commit in $mfs (fetch it first)" >&2
  exit 1
}
root="$(cd "$(dirname "$0")/.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

git -C "$mfs" archive "$commit" schemas/platform | tar -x -C "$tmp"
[ -f "$tmp/schemas/platform/SHA256SUMS" ] || {
  echo "vendor: $commit has no schemas/platform/SHA256SUMS" >&2
  exit 1
}
if command -v sha256sum >/dev/null; then sum=(sha256sum); else sum=(shasum -a 256); fi
(cd "$tmp/schemas/platform" && "${sum[@]}" -c --quiet SHA256SUMS) || {
  echo "vendor: the bundle at $commit does not match its own SHA256SUMS" >&2
  exit 1
}
sums="$("${sum[@]}" "$tmp/schemas/platform/SHA256SUMS" | cut -d' ' -f1)"

rm -rf "$root/schemas/platform"
mv "$tmp/schemas/platform" "$root/schemas/platform"
cat > "$root/schemas/platform.pin" <<PIN
# 2026-10-11: Pin of the vendored platform-contract bundle in \`schemas/platform/\` (investor-mvp #41).
# The copy is Metrale/mfs \`schemas/platform/\` at \`mfs_commit\`, unchanged. \`sha256sums\` is the
# SHA-256 of its \`SHA256SUMS\`. \`crates/qci-dump/tests/platform_contracts.rs\` checks both, and
# runs the runtime's binding over the bundle's fixtures. Refresh both with
# \`bash scripts/vendor-platform-schemas.sh <mfs-clone> <commit>\`; never edit the copy by hand.
mfs_commit = "$commit"
sha256sums = "$sums"
PIN
echo "vendored Metrale/mfs schemas/platform at $commit (SHA256SUMS $sums)"
