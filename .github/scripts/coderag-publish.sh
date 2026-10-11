#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0

# Publish every CodeRAG corpus as ONE orphan commit on gh-pages.
#
# The three corpora are built by separate jobs and any of them can fail, hit
# the free-tier rate limit, or be skipped as already current. A publish that
# only wrote the corpora built this run would delete the others from the
# branch, because the branch is one force-pushed commit. So this starts from
# what gh-pages holds NOW and overwrites only the corpora that were built,
# fit under the size cap, and have all three published files. Everything else
# keeps its last good copy.
#
# Inputs (all required, no defaults):
#   CORPORA        space-separated <base>:<status>, status = built|skipped|failed
#   INCOMING       directory holding the staged files of the corpora built
#   SIZE_CAP_BYTES per-file limit; GitHub rejects a pushed file over 100 MB
#   PUBLISH_REMOTE authenticated git URL of the repository
#   SOURCES_JSON   path of .github/ai-models.json (pinned external sources)
#   GITHUB_SHA     commit the corpora were built at
# Appends a per-corpus table to $GITHUB_STEP_SUMMARY when it is set.
set -euo pipefail

: "${CORPORA:?}" "${INCOMING:?}" "${SIZE_CAP_BYTES:?}" "${PUBLISH_REMOTE:?}" \
  "${SOURCES_JSON:?}" "${GITHUB_SHA:?}"
summary="${GITHUB_STEP_SUMMARY:-/dev/null}"
suffixes=(.jsonl.gz .jsonl.meta.json .meta.json)

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
prev="$work/prev" site="$work/site"
mkdir -p "$prev/coderag" "$site/coderag"

# `ls-remote --exit-code` exits 2 when the branch is absent (the first ever
# publish) and 128 on a real failure; only the first may fall through.
rc=0
git ls-remote --exit-code --heads "$PUBLISH_REMOTE" gh-pages >/dev/null 2>&1 || rc=$?
case "$rc" in
  0) git clone -q --depth 1 --branch gh-pages "$PUBLISH_REMOTE" "$work/clone"
     cp -a "$work/clone/coderag/." "$prev/coderag/" 2>/dev/null || true ;;
  2) echo "gh-pages does not exist yet; first publish." ;;
  *) echo "::error::cannot read gh-pages (git ls-remote exit $rc)" >&2; exit 1 ;;
esac

{
  echo "### CodeRAG corpora"
  echo ""
  echo "| corpus | build | published |"
  echo "|---|---|---|"
} >> "$summary"

refreshed=0
unrefreshed=()
for entry in $CORPORA; do
  base="${entry%%:*}" status="${entry#*:}"
  usable=false reason=""
  case "$status" in
    built)
      usable=true
      for s in "${suffixes[@]}"; do
        [ -s "$INCOMING/${base}${s}" ] || { usable=false; reason="staged file ${base}${s} missing"; }
      done
      if [ "$usable" = true ]; then
        bytes="$(stat -c%s "$INCOMING/${base}.jsonl.gz")"
        echo "${base}.jsonl.gz: ${bytes} bytes (cap ${SIZE_CAP_BYTES})"
        if [ "$bytes" -gt "$SIZE_CAP_BYTES" ]; then
          usable=false
          reason="${bytes} B exceeds the ${SIZE_CAP_BYTES} B cap; raise chunk-lines, lower vector-precision or trim paths"
        fi
      fi ;;
    skipped) reason="already current" ;;
    failed)  reason="build did not succeed" ;;
    *) echo "::error::unknown status '${status}' for ${base}" >&2; exit 1 ;;
  esac

  if [ "$usable" = true ]; then
    for s in "${suffixes[@]}"; do cp "$INCOMING/${base}${s}" "$site/coderag/${base}${s}"; done
    refreshed=$((refreshed + 1))
    echo "| ${base} | built | refreshed |" >> "$summary"
  elif [ -s "$prev/coderag/${base}.jsonl.gz" ]; then
    for s in "${suffixes[@]}"; do cp "$prev/coderag/${base}${s}" "$site/coderag/${base}${s}"; done
    echo "| ${base} | ${status} | kept previous copy (${reason}) |" >> "$summary"
    [ "$status" = skipped ] || unrefreshed+=("${base}: ${reason}")
  else
    echo "| ${base} | ${status} | NOT PUBLISHED (${reason}; no previous copy to keep) |" >> "$summary"
    unrefreshed+=("${base}: ${reason}, and no copy was published before")
  fi
done

# The invariant that matters, checked on the tree itself rather than on the
# per-corpus decisions above: nothing over the cap reaches git push.
while IFS= read -r f; do
  bytes="$(stat -c%s "$f")"
  if [ "$bytes" -gt "$SIZE_CAP_BYTES" ]; then
    echo "::error::${f#"$site"/} is ${bytes} bytes, over the ${SIZE_CAP_BYTES}-byte cap" >&2
    exit 1
  fi
done < <(find "$site" -type f)

if [ -z "$(find "$site/coderag" -type f -name '*.jsonl.gz' -print -quit)" ]; then
  echo "::error::no corpus to publish" >&2
  exit 1
fi

{
  echo "# CodeRAG corpora: sources and credits"
  echo ""
  echo "Built by \`.github/workflows/coderag.yml\`. Each corpus is a self-verifying"
  echo "lattice-jsonl file; its \`.meta.json\` names the repository, commit and"
  echo "embedding model it was built from."
  echo ""
  echo "| corpus | repository | commit | model | points | built |"
  echo "|---|---|---|---|---|---|"
  for m in "$site"/coderag/*.jsonl.meta.json; do
    jq -r --arg b "$(basename "$m" .jsonl.meta.json)" \
      '"| \($b) | \(.repo) | \(.commit_sha) | \(.model) | \(.points) | \(.generated_at) |"' "$m"
  done
  echo ""
  if [ -s "$site/coderag/books-rust.jsonl.gz" ]; then
    echo "## Third-party content"
    echo ""
    jq -r '.sources.rust_book | "- **books-rust** is chunked from *The Rust Programming Language* (https://github.com/\(.repo)), pinned at commit `\(.ref)`, licensed \(.license). Copyright the Rust Project Developers; the licence texts are LICENSE-MIT and LICENSE-APACHE in that repository at the pinned commit. The text is embedded unmodified, and every chunk names its source file and line range."' "$SOURCES_JSON"
  fi
} > "$site/coderag/SOURCES.md"

touch "$site/.nojekyll"
cd "$site"
git init -q -b gh-pages
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git add -A
git commit -q -m "coderag corpora @ ${GITHUB_SHA}"
git push --force "$PUBLISH_REMOTE" gh-pages

if [ "${#unrefreshed[@]}" -gt 0 ]; then
  {
    echo ""
    echo "### CodeRAG corpus NOT refreshed"
    echo ""
    for u in "${unrefreshed[@]}"; do echo "- ${u}"; done
    echo ""
    echo "The previously published copy stays live where one exists, so the site keeps working with slightly older knowledge."
  } >> "$summary"
  for u in "${unrefreshed[@]}"; do echo "::warning::CodeRAG corpus not refreshed: ${u}"; done
fi
echo "published ${refreshed} refreshed corpus(es) at ${GITHUB_SHA}"
