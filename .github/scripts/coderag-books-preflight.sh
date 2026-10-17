#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0

# Decide whether the published book corpus is already current.
#
# A pinned book changes only when someone bumps its ref, so re-embedding it on
# every run would spend the free-tier quota for byte-identical output. The
# published meta file records the commit the corpus was built from and the
# model that embedded it; the corpus is current when both equal what this run
# would build. Anything unexpected (404, bad JSON, curl failure) means BUILD:
# a wasted build is cheap, a stale book silently kept is not.
#
# Inputs (all required, no defaults):
#   META_URL    published <base>.jsonl.meta.json
#   PINNED_REF  commit the workflow would check out
#   EMBED_MODEL model the workflow would embed with
#   FORCE       'true' rebuilds regardless
# Writes `skip=true|false` to $GITHUB_OUTPUT and a line to $GITHUB_STEP_SUMMARY.
set -euo pipefail

: "${META_URL:?}" "${PINNED_REF:?}" "${EMBED_MODEL:?}" "${FORCE:?}" "${GITHUB_OUTPUT:?}"
summary="${GITHUB_STEP_SUMMARY:-/dev/null}"

emit() { # skip(true|false) message
  echo "skip=$1" >> "$GITHUB_OUTPUT"
  echo "$2"
  echo "$2" >> "$summary"
}

meta="$(mktemp)"
trap 'rm -f "$meta"' EXIT
code="$(curl -sS --retry 3 -o "$meta" -w '%{http_code}' "$META_URL" || true)"

case "$code" in
  200) ;;
  404) emit false "Book corpus: no published copy at ${META_URL} (HTTP 404); building."; exit 0 ;;
  *)   emit false "Book corpus: preflight of ${META_URL} failed (HTTP ${code:-none}); building."; exit 0 ;;
esac

if ! sha="$(jq -er '.commit_sha' "$meta")" || ! model="$(jq -er '.model' "$meta")"; then
  emit false "Book corpus: published meta lacks commit_sha/model; building."
  exit 0
fi

if [ "$FORCE" = "true" ]; then
  emit false "Book corpus: rebuild forced (published ${sha}, model ${model})."
elif [ "$sha" != "$PINNED_REF" ]; then
  emit false "Book corpus: pin moved (published ${sha}, pinned ${PINNED_REF}); building."
elif [ "$model" != "$EMBED_MODEL" ]; then
  emit false "Book corpus: embed model changed (published ${model}, now ${EMBED_MODEL}); building."
else
  emit true "Book corpus SKIPPED: published copy is current (commit ${sha}, model ${model})."
fi
