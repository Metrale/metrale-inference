#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# 2026-10-22: Re-export GLM-5.3-Flash TP=3 GB10 nsys captures (KV≈1k and KV≈128k decode) as
# headroom traces under docs/kernel-perf/traces/. Usage: `$0 <capture>...` with captures from the
# table below (f13, f14a, f14b). The SQLite files are on dgx1 at /home/claude/glm-bench/nsys
# (written by claude-state glm/agentic/prof/nsysrun.sh <capture>); the usage numbers below are each
# run's prof/<capture>-p{1k,128k}.usage (f13, f14a and f14b decode the same token trajectories).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
[ $# -ge 1 ] || { echo "usage: $0 <capture: f13|f14a|f14b>..." >&2; exit 2; }
N=${NSYS_DIR:-/home/claude/glm-bench/nsys}
M=scripts/headroom/glm-5.3-flash-nvfp4-tp3--gb10.toml
# 2026-10-23: Every run ran on the GLM layer code: the circuit executor has no GLM emitters and the
# circuit plans the whole model, not a TP rank (claude-state glm/megakernel/STATE.md, P0 BLOCKED).
WHY='no GLM emitters in the circuit executor (crates/model-layers/src/circuit_exec/emitters/mod.rs), and the circuit plans the whole model, not a TP rank (kernels/circuits/INSTANCES.toml:176)'
EXEC=(--execution "verify n=2" legacy "$WHY" --execution "draft n=1" legacy "$WHY")
TAIL='nsys --cuda-graph-trace=node, 8 s window of the warm repeat of prof/%s-nothink.json (temperature 0.0, max_tokens 1500, enable_thinking false); single capture'
for cap in "$@"; do
  case $cap in
    f13) SRC="nsys f13_%s.sqlite @ c5b9a80e6 (fast13: binary f4735654, profile glm53-tp3-512k-fast13 = fast12 recipe: --max-seq-len 524288 --kv-cache-dtype fp8_e4m3 --gpu-memory-utilization 0.82 --max-batch-size 16, MTP 1 draft forced, W4A4 experts, w4a16 dense, w8a8 DSA), dgx1 rank 0, 2026-10-22; $TAIL" ;;
    f14a|f14b) SRC="nsys ${cap}_%s.sqlite (promo1: binary 6b362205, profile glm53-tp3-512k-promo1, recipe.env f45d87a6, recipe.flags e17fe83b; SM clocks 2504-2592 MHz on all 3 boxes), 2026-10-23, serve ${cap#f14} of two independent serves of the same configuration, dgx1 rank 0; $TAIL" ;;
    *) echo "unknown capture $cap" >&2; exit 2 ;;
  esac
  python3 scripts/headroom_trace.py "$N/${cap}_p1k.sqlite" --step-kernel screen_head_scan --attribution "$M" "${EXEC[@]}" \
    --source "$(printf "$SRC" p1k p1k)" \
    --tokens-per-step 1.672 --tokens-basis "the request's usage: 1131 completion tokens, 454 accepted drafts => (1131-1)/(1131-1-454) tokens per verify step (prof/$cap-p1k.usage)" \
    --kv-tokens 1646 --kv-basis "prompt 1038 + ~608 tokens decoded by the window midpoint (12 s into the request at its mean 51.1 tok/s; prof/$cap-p1k.usage)" \
    > "docs/kernel-perf/traces/glm-5.3-flash-nvfp4-tp3--gb10--$cap-p1k.toml"
  python3 scripts/headroom_trace.py "$N/${cap}_p128k.sqlite" --step-kernel screen_head_scan --attribution "$M" "${EXEC[@]}" \
    --source "$(printf "$SRC" p128k p128k)" \
    --tokens-per-step 1.800 --tokens-basis "the request's usage: 1500 completion tokens, 666 accepted drafts => (1500-1)/(1500-1-666) tokens per verify step (prof/$cap-p128k.usage)" \
    --kv-tokens 128659 --kv-basis "prompt 128037 + ~622 tokens decoded by the window midpoint (12 s into the request at its mean 51.8 tok/s; prof/$cap-p128k.usage)" \
    > "docs/kernel-perf/traces/glm-5.3-flash-nvfp4-tp3--gb10--$cap-p128k.toml"
done
