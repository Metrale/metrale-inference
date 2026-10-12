#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0

# Check one built corpus against the publish cap, right after it is built.
#
# 2026-10-11: the publish job used to be the only place the cap was enforced.
# The code corpus came out at 105,944,953 B against a 99,614,720 B cap, its
# build job ended green with nothing in its summary, and the verdict surfaced
# two jobs later as a warning. This gives every build job its own verdict:
# a step-summary line either way, an error annotation when over the cap (the
# step exits 1; the workflow marks it continue-on-error so the job stays
# green and the previous copy stays live), and a warning once the corpus has
# eaten the headroom a split is meant to leave.
#
# Inputs (all required, no defaults):
#   BASE               corpus name, for the messages
#   GZ                 the gzipped corpus the build produced
#   SIZE_CAP_BYTES     per-file limit; GitHub rejects a pushed file over 100 MB
#   SIZE_WARN_PERCENT  warn when the gzip reaches this percent of the cap
#   GITHUB_OUTPUT      receives over=true|false (the caller tells "over the
#                      cap" from "the check itself broke" by this, not by the
#                      exit code, which is 1 for both)
# Appends one line to $GITHUB_STEP_SUMMARY when it is set.
set -euo pipefail

: "${BASE:?}" "${GZ:?}" "${SIZE_CAP_BYTES:?}" "${SIZE_WARN_PERCENT:?}" "${GITHUB_OUTPUT:?}"
summary="${GITHUB_STEP_SUMMARY:-/dev/null}"

[ -s "$GZ" ] || { echo "::error::coderag-size-check: $GZ is missing or empty" >&2; exit 1; }
bytes="$(stat -c%s "$GZ")"
pct=$((bytes * 100 / SIZE_CAP_BYTES))

if [ "$bytes" -gt "$SIZE_CAP_BYTES" ]; then
  line="${BASE}: ${bytes} bytes (cap ${SIZE_CAP_BYTES}) — over cap, NOT published"
  echo "over=true" >> "$GITHUB_OUTPUT"
  echo "- ${line}" >> "$summary"
  echo "::error title=${BASE} is over the size cap::${line}. The previous copy stays live. Split the corpus by path (see the header of coderag.yml)."
  exit 1
fi

line="${BASE}: ${bytes} bytes (cap ${SIZE_CAP_BYTES}) — ${pct}% of cap, within cap"
echo "over=false" >> "$GITHUB_OUTPUT"
echo "- ${line}" >> "$summary"
echo "$line"
if [ "$pct" -ge "$SIZE_WARN_PERCENT" ]; then
  echo "::warning title=${BASE} is close to the size cap::${line}. Under $((100 - SIZE_WARN_PERCENT))% headroom; split it before it goes over."
fi
