#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0

# Name one built corpus the way it is published.
#
#   coderag-stage.sh <base> <corpus.gz> <meta.json> <outdir>
#
# AutoRepoRAG is given `output-name: <base>.jsonl`, so the metadata file is
# already <base>.jsonl.meta.json (the name the site's preflight fetches) but
# the gzip is <base>.jsonl.jsonl.gz. One meta file is published under two
# names, one per consumer:
#   <base>.jsonl.meta.json -- the site's OPFS preflight name.
#   <base>.meta.json       -- the sibling AutoRepoRAG v1 derived from a
#     previous corpus URL; kept so a pinned v1 consumer still finds it.
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "usage: $0 <base> <corpus.gz> <meta.json> <outdir>" >&2
  exit 2
fi
base="$1" gz="$2" meta="$3" outdir="$4"

for f in "$gz" "$meta"; do
  [ -s "$f" ] || { echo "::error::coderag-stage: $f is missing or empty" >&2; exit 1; }
done
mkdir -p "$outdir"
cp "$gz" "$outdir/${base}.jsonl.gz"
cp "$meta" "$outdir/${base}.jsonl.meta.json"
cp "$meta" "$outdir/${base}.meta.json"
