#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# The sorted set of PTX digests one build embeds, for comparing an uncached
# build with a cold-cache and a warm-cache one. Reads the newest
# metrale-kernels build-script output under <profile-dir>, takes its
# METRALE_PTX_DIR, and hashes every `t<idx>__<stem>.ptx` that `target_ptx.rs`
# embeds.
#
# Usage:
#   scripts/ptx-digest-set.sh <profile-dir> > digests.txt
#   scripts/ptx-digest-set.sh target/iter > before.txt
#
# stdout: `<sha256>  <file>` lines, sorted by file name.
# stderr: the PTX dir, the entry count and one SHA-256 over stdout.
#
# Gate (one target dir, so every leg compiles from the same OUT_DIR paths;
# a device `assert` embeds the staged source path in the PTX):
#   1. Uncached:   cargo build --profile iter            (cache unset)
#                  scripts/ptx-digest-set.sh target/iter > before.txt
#   2. Cold cache: rm -rf target/metrale-ptx-cache; touch a kernel source
#                  so the build script reruns; scripts/dev-build.sh <model>
#                  scripts/ptx-digest-set.sh target/iter > cold.txt
#   3. Warm cache: touch the source again; scripts/dev-build.sh <model>
#                  scripts/ptx-digest-set.sh target/iter > warm.txt
#   diff before.txt cold.txt && diff before.txt warm.txt
#   The warm leg's <OUT_DIR>/ptx-cache.log must read `hit` on every line.
#
# Negative control: append a comment to
# kernels/gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu, run
# `scripts/dev-build.sh glm-5.3-flash`, and the log must show `miss` for
# dflash_fp8_gemm.cu (it includes w4a16_gemm.cu through `-I`). Revert the
# edit and rebuild: every line is `hit` again.

set -euo pipefail

if [ $# -ne 1 ]; then
  echo "usage: scripts/ptx-digest-set.sh <profile-dir>   (e.g. target/iter)" >&2
  exit 2
fi
profile_dir="$1"

output="$(ls -t "$profile_dir"/build/metrale-kernels-*/output 2>/dev/null | head -n 1 || true)"
if [ -z "$output" ]; then
  echo "no metrale-kernels build-script output under $profile_dir/build" >&2
  exit 1
fi
ptx_dir="$(sed -n 's/^cargo:rustc-env=METRALE_PTX_DIR=//p' "$output" | tail -n 1)"
if [ -z "$ptx_dir" ] || [ ! -f "$ptx_dir/target_ptx.rs" ]; then
  echo "$output names no METRALE_PTX_DIR with a target_ptx.rs" >&2
  exit 1
fi

names="$(grep -o 't[0-9][0-9]*__[A-Za-z0-9_]*\.ptx' "$ptx_dir/target_ptx.rs" | sort -u)"
if [ -z "$names" ]; then
  echo "$ptx_dir/target_ptx.rs embeds no PTX (a METRALE_SKIP_BUILD stub?)" >&2
  exit 1
fi

digests="$(cd "$ptx_dir" && printf '%s\n' "$names" | xargs sha256sum | sort -k2)"
printf '%s\n' "$digests"
{
  echo "ptx dir: $ptx_dir"
  echo "entries: $(printf '%s\n' "$names" | wc -l)"
  echo "set sha256: $(printf '%s\n' "$digests" | sha256sum | cut -d' ' -f1)"
} >&2
