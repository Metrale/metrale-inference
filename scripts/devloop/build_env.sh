#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# build_env.sh: the developer-build environment that cargo-guard.sh and dev-build.sh share. Source it, then call
#   devloop_build_env <cargo args...>
# which exports, for a cargo command that compiles (build check clippy test run bench doc rustc):
#   - METRALE_PTX_CACHE=1 unless the caller set it (0 opts out), with METRALE_PTX_CACHE_MAX_BYTES at
#     DEVLOOP_PTX_CACHE_MAX_BYTES unless the caller set it;
#   - METRALE_TARGET_HW/MODEL/QUANT from the lane's recipe when the caller names one in METRALE_LANE_RECIPE (a
#     recipe id, e.g. glm-5.3/glm-5.3-flash-nvfp4-tp3-512k): its `target` in kernels/circuits/INSTANCES.toml.
#     A caller's METRALE_TARGET_MODEL (`*` included) wins. Neither set is an error: the model is never guessed.
#   - for `--profile iter` only: CARGO_INCREMENTAL as the caller set it (the profile's incremental=true
#     otherwise) and the bundled rust-lld as the linker. Any other profile keeps CARGO_INCREMENTAL=0 unless the
#     caller set it.
# Release and CI builds do not source this file; they set METRALE_TARGET_MODEL themselves (`*` or one model).
# Returns 2 with a message on stderr when the environment cannot be resolved.

# 2026-10-17: room for many full kernel sets of every model a developer cycles through (was dev-build.sh's).
DEVLOOP_PTX_CACHE_MAX_BYTES=$((4 * 1024 * 1024 * 1024))

devloop_subcommand() {  # first argument that is not an option or a +toolchain
  local a
  for a in "$@"; do case "$a" in -*|+*) ;; *) echo "$a"; return;; esac; done
}

devloop_is_iter() {
  local prev= a
  for a in "$@"; do
    [ "$a" = --profile=iter ] && return 0
    [ "$prev" = --profile ] && [ "$a" = iter ] && return 0
    prev=$a
  done
  return 1
}

devloop_lane_target() {  # <workspace root> <recipe id> -> "hw model quant"
  python3 -I - "$1/kernels/circuits/INSTANCES.toml" "$2" <<'PY'
import sys, tomllib
path, recipe = sys.argv[1], sys.argv[2]
with open(path, "rb") as f:
    hits = [i for i in tomllib.load(f).get("instance", []) if i.get("recipe") == recipe]
if len(hits) != 1:
    sys.exit(f"METRALE_LANE_RECIPE={recipe}: {len(hits)} instances in {path} (need exactly 1)")
parts = hits[0].get("target", "").split("/")
if len(parts) != 3 or not all(parts):
    sys.exit(f"METRALE_LANE_RECIPE={recipe}: target {hits[0].get('target')!r} is not hw/model/quant")
print(*parts)
PY
}

devloop_target_env() {
  if [ -n "${METRALE_TARGET_MODEL:-}" ]; then
    echo "build_env: METRALE_TARGET_MODEL=$METRALE_TARGET_MODEL (caller)" >&2; return 0
  fi
  [ -n "${METRALE_LANE_RECIPE:-}" ] || {
    echo "build_env: refused: set METRALE_LANE_RECIPE=<recipe id> (the lane's model) or METRALE_TARGET_MODEL (* = every model)" >&2
    return 2; }
  local root t hw model quant v
  root=$(cargo locate-project --workspace --message-format plain 2>/dev/null) || {
    echo "build_env: refused: not inside a cargo workspace" >&2; return 2; }
  root=${root%/Cargo.toml}
  t=$(devloop_lane_target "$root" "$METRALE_LANE_RECIPE") || return 2
  read -r hw model quant <<< "$t"
  for v in "METRALE_TARGET_HW $hw" "METRALE_TARGET_QUANT $quant"; do
    set -- $v
    [ -z "${!1:-}" ] || [ "${!1}" = "$2" ] || {
      echo "build_env: refused: $1=${!1} but lane recipe $METRALE_LANE_RECIPE targets $2" >&2; return 2; }
  done
  export METRALE_TARGET_HW=$hw METRALE_TARGET_MODEL=$model METRALE_TARGET_QUANT=$quant
  echo "build_env: target $hw/$model/$quant (lane recipe $METRALE_LANE_RECIPE)" >&2
}

devloop_build_env() {
  case "$(devloop_subcommand "$@")" in
    build|b|check|c|clippy|test|t|run|r|bench|doc|d|rustc) ;;
    *) export CARGO_INCREMENTAL=${CARGO_INCREMENTAL:-0}; return 0;;
  esac
  export METRALE_PTX_CACHE=${METRALE_PTX_CACHE:-1}
  case "$METRALE_PTX_CACHE" in
    1) export METRALE_PTX_CACHE_MAX_BYTES=${METRALE_PTX_CACHE_MAX_BYTES:-$DEVLOOP_PTX_CACHE_MAX_BYTES};;
    0) ;;
    *) echo "build_env: refused: METRALE_PTX_CACHE must be 0 or 1, got $METRALE_PTX_CACHE" >&2; return 2;;
  esac
  devloop_target_env || return 2
  if devloop_is_iter "$@"; then
    local gccld
    gccld="$(rustc --print sysroot)/lib/rustlib/$(rustc -vV | sed -n 's/^host: //p')/bin/gcc-ld"
    [ -x "$gccld/ld.lld" ] || { echo "build_env: refused: no bundled ld.lld at $gccld" >&2; return 2; }
    export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-Clink-arg=-B$gccld -Clink-arg=-fuse-ld=lld"
    echo "build_env: --profile iter: incremental, linker rust-lld ($gccld)" >&2
  else
    export CARGO_INCREMENTAL=${CARGO_INCREMENTAL:-0}
  fi
}
