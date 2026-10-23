#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# 2026-10-22: Re-export the two GLM-5.3-Flash TP=3 GB10 f13 captures (KV≈1k and KV≈128k decode)
# as headroom traces under docs/kernel-perf/traces/. The SQLite files are on dgx1 at
# /home/claude/glm-bench/nsys (written by claude-state glm/agentic/prof/nsysrun.sh f13); the
# usage numbers below are that run's prof/f13-p{1k,128k}.usage.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
N=${NSYS_DIR:-/home/claude/glm-bench/nsys}
M=scripts/headroom/glm-5.3-flash-nvfp4-tp3--gb10.toml
SRC='nsys f13_%s.sqlite @ c5b9a80e6 (fast13: binary f4735654, profile glm53-tp3-512k-fast13 = fast12 recipe: --max-seq-len 524288 --kv-cache-dtype fp8_e4m3 --gpu-memory-utilization 0.82 --max-batch-size 16, MTP 1 draft forced, W4A4 experts, w4a16 dense, w8a8 DSA), dgx1 rank 0, 2026-10-22; nsys --cuda-graph-trace=node, 8 s window of the warm repeat of prof/%s-nothink.json (temperature 0.0, max_tokens 1500, enable_thinking false); single capture'
python3 scripts/headroom_trace.py "$N/f13_p1k.sqlite" --step-kernel screen_head_scan --attribution "$M" \
  --source "$(printf "$SRC" p1k p1k)" \
  --tokens-per-step 1.672 --tokens-basis "the request's usage: 1131 completion tokens, 454 accepted drafts => (1131-1)/(1131-1-454) tokens per verify step (prof/f13-p1k.usage)" \
  --kv-tokens 1646 --kv-basis "prompt 1038 + ~608 tokens decoded by the window midpoint (12 s into the request at its mean 51.1 tok/s; prof/f13-p1k.usage)" \
  > docs/kernel-perf/traces/glm-5.3-flash-nvfp4-tp3--gb10--f13-p1k.toml
python3 scripts/headroom_trace.py "$N/f13_p128k.sqlite" --step-kernel screen_head_scan --attribution "$M" \
  --source "$(printf "$SRC" p128k p128k)" \
  --tokens-per-step 1.800 --tokens-basis "the request's usage: 1500 completion tokens, 666 accepted drafts => (1500-1)/(1500-1-666) tokens per verify step (prof/f13-p128k.usage)" \
  --kv-tokens 128659 --kv-basis "prompt 128037 + ~622 tokens decoded by the window midpoint (12 s into the request at its mean 51.8 tok/s; prof/f13-p128k.usage)" \
  > docs/kernel-perf/traces/glm-5.3-flash-nvfp4-tp3--gb10--f13-p128k.toml
