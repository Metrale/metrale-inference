#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0

# Vendor the files this repository derives from Metrale/metrale-assets (the brand kit, the single
# source of every brand asset), at a signed release tag, and pin them.
#
#   assets/brand/take-assets.sh <metrale-assets checkout> <tag>   # then: node assets/brand/derive.mjs, commit
#   assets/brand/take-assets.sh --check [--upstream]              # the vendored files are the pinned ones
#
# metrale-assets.pin records the tag, its commit and every file's git blob id. --check works offline
# (the Docs workflow runs it): it fails when a vendored file was edited, added or removed by hand.
# --upstream also asks GitHub that the pinned commit holds exactly those blobs (needs gh with access
# to the kit's repository).
set -euo pipefail
cd "$(dirname "$0")"
dir=metrale-assets
pin=metrale-assets.pin
# What this repository takes. Add a path here, re-run, then derive.
FILES=(
  tokens/brand.json
  svg/logo-horizontal.svg svg/logo-horizontal-ondark.svg svg/mark-compact.svg
  favicon.ico favicon.svg
  dark/favicon-16.png dark/favicon-32.png dark/favicon-48.png
  dark/apple-touch-icon-180.png dark/og-image-1200x630.png
  fonts/manrope-latin-wght-normal.woff2 fonts/MANROPE-LICENSE.txt fonts/manrope-fallback.css
)

if [ "${1:-}" = "--check" ]; then
  [ -f "$pin" ] || { echo "take-assets: $pin is missing" >&2; exit 1; }
  commit=$(sed -n 's/^commit=//p' "$pin")
  [[ "$commit" =~ ^[0-9a-f]{40}$ ]] || { echo "take-assets: $pin is malformed" >&2; exit 1; }
  want=$(grep -E '^[0-9a-f]{40}  ' "$pin" | sort -k2)
  have=$(cd "$dir" && find . -type f | sed 's|^\./||' | sort | while read -r f; do printf '%s  %s\n' "$(git hash-object "$f")" "$f"; done | sort -k2)
  [ "$want" = "$have" ] || { echo "take-assets: assets/brand/$dir differs from $pin (edited by hand? re-run take-assets.sh)" >&2; diff <(echo "$want") <(echo "$have") >&2 || true; exit 1; }
  if [ "${2:-}" = "--upstream" ]; then
    up=$(gh api "repos/Metrale/metrale-assets/git/trees/$commit?recursive=1" --jq '.tree[] | select(.type=="blob") | "\(.sha)  \(.path)"' | sort -k2)
    while read -r line; do grep -qxF "$line" <<<"$up" || { echo "take-assets: upstream $commit lacks: $line" >&2; exit 1; }; done <<<"$want"
  fi
  echo "take-assets: assets/brand/$dir == metrale-assets@$(sed -n 's/^tag=//p' "$pin") (${commit:0:12})${2:+, confirmed upstream}"
  exit 0
fi

[ $# -eq 2 ] || { echo "usage: take-assets.sh <metrale-assets checkout> <tag> | --check [--upstream]" >&2; exit 2; }
src=$1 tag=$2
git -C "$src" fetch -q --tags origin
commit=$(git -C "$src" rev-parse --verify "refs/tags/$tag^{commit}")
git -C "$src" tag -v "$tag" >/dev/null 2>&1 || { echo "take-assets: tag $tag is not a valid signed tag" >&2; exit 1; }
rm -rf -- "$dir"
mkdir -p "$dir"
git -C "$src" archive "$commit" "${FILES[@]}" | tar -x -C "$dir"
{
  printf 'tag=%s\ncommit=%s\n' "$tag" "$commit"
  for f in "${FILES[@]}"; do printf '%s  %s\n' "$(git -C "$src" rev-parse "$commit:$f")" "$f"; done
} > "$pin"
echo "vendored metrale-assets $tag (${commit:0:12}) into assets/brand/$dir; run node assets/brand/derive.mjs and commit"
