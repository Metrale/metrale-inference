#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Developer build with the content-hashed PTX cache on. Builds one model's
# kernels with `--profile iter`; a kernel whose nvcc argv, compiler, NVCC_*
# environment and include-closure bytes all match an earlier build is copied
# from <target-dir>/metrale-ptx-cache instead of recompiled.
#
# Usage:
#   scripts/dev-build.sh <model> [cargo build args...]
#   scripts/dev-build.sh glm-5.3-flash
#   scripts/dev-build.sh qwen3.6-27b -p met
#   METRALE_TARGET_HW=hopper scripts/dev-build.sh qwen3.6-27b
#
# <model> is required and has no default: it is exported as
# METRALE_TARGET_MODEL. METRALE_TARGET_HW and METRALE_TARGET_QUANT pass
# through as the caller set them; unset, the kernels build uses its own
# documented target defaults.
#
# Each run writes <OUT_DIR>/ptx-cache.log (`hit|miss|stale <key> <source>`)
# and prints one summary warning. CI and release builds never set
# METRALE_PTX_CACHE, so they compile every kernel exactly as before.

set -euo pipefail

if [ $# -lt 1 ] || [ -z "${1:-}" ] || [ "${1#-}" != "$1" ]; then
  echo "usage: scripts/dev-build.sh <model> [cargo build args...]" >&2
  exit 2
fi
model="$1"
shift

# 2026-10-24: The cache cap (required by the build, no default inside it), the
# iter profile's incremental + bundled-lld link and the PTX cache switch come from
# scripts/devloop/build_env.sh, which cargo-guard.sh shares.
export METRALE_TARGET_MODEL="$model"
. "$(dirname "$(readlink -f "$0")")/devloop/build_env.sh"
devloop_build_env build --profile iter "$@"

CARGO="${CARGO:-cargo}"
exec "$CARGO" build --profile iter "$@"
