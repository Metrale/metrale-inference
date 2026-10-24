#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# cargo-guard.sh: a drop-in for `cargo` that never overlaps a GPU timing or capture leg and keeps the disk floor.
# Usage: cargo-guard.sh <cargo args...>     (waits, never skips; exit status is cargo's)
#
# The lock (bench_lock.sh) gives each holder a class; this script reads it on every tick (CARGO_GUARD_POLL_S):
#   timing / capture hold (or a legacy holder without a class): no cargo process runs. A build waits to start,
#     and a running build's process group is frozen with SIGSTOP until the hold ends (then SIGCONT).
#   correctness holds only: cargo runs at nice 19, pinned to the online CPUs outside the union of the holders'
#     serve CPUs, with CARGO_BUILD_JOBS = that CPU count unless the caller set it. No CPU left = blocked.
#   no hold: cargo runs as asked. A build re-pinned during a hold keeps nice 19 after it (it cannot be raised back).
# Every build registers in $BENCH_LOCK_BUILDS/<pgid> so a timing or capture bench_lock can wait for it to freeze.
# The environment comes from build_env.sh (PTX cache on, the lane's target model, iter-profile incremental + lld).
# Refuses (exit 75) if / has < ${CARGO_GUARD_MIN_GB:-15} GB free, (exit 2) if build_env.sh cannot resolve.
set -u
HERE=$(dirname "$(readlink -f "$0")")
. "$HERE/bench_lock.sh"
. "$HERE/build_env.sh"
MIN=${CARGO_GUARD_MIN_GB:-15}
POLL=${CARGO_GUARD_POLL_S:-1}
ts() { date -u +%FT%TZ; }
say() { echo "cargo-guard: $(ts) $*" >&2; }

guard_mode() {  # -> "block - <holders>" | "pin <cpulist> <holders>" | "clear - -"
  local h f c s n serve=() all keep=() cpu
  h=$(bench_holders)
  [ -n "$h" ] || { echo "clear - -"; return; }
  while read -r f c s n; do
    case "$c" in correctness) serve+=($(bench_cpulist_expand "$s"));; *) echo "block - $(echo "$h" | bench_holders_brief)"; return;; esac
  done <<< "$h"
  all=$(bench_cpulist_expand "$(cat /sys/devices/system/cpu/online)")
  for cpu in $all; do [[ " ${serve[*]} " == *" $cpu "* ]] || keep+=("$cpu"); done
  [ ${#keep[@]} -gt 0 ] || { echo "block - no-cpu-outside-serve:$(echo "$h" | bench_holders_brief)"; return; }
  echo "pin $(IFS=,; echo "${keep[*]}") $(echo "$h" | bench_holders_brief)"
}

devloop_build_env "$@" || exit 2

since=$SECONDS; last=
while :; do
  read -r mode cpus desc <<< "$(guard_mode)"
  if [ "$mode" = block ]; then
    [ "$desc" = "$last" ] || say "waiting, blocked by: $desc"
    last=$desc; sleep "$POLL"; continue
  fi
  free=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
  [ "$free" -ge "$MIN" ] || { say "refused, ${free} GB free < ${MIN} GB"; exit 75; }
  break
done
[ -z "$last" ] || say "waited $((SECONDS - since)) s"

if [ "$mode" = pin ]; then
  n=$(tr ',' '\n' <<< "$cpus" | wc -l)
  say "correctness hold ($desc): nice 19 on CPUs $cpus, ${CARGO_BUILD_JOBS:-$n} jobs: cargo $*"
  CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-$n} setsid nice -n 19 taskset -c "$cpus" cargo "$@" &
else
  say "clear (no bench hold, ${free} GB free): cargo $*"
  setsid cargo "$@" &
fi
CPID=$!
# The forked child becomes its own group only once setsid has run; wait for that (it is a few ms).
pg=
for _ in $(seq 500); do pg=$(ps -o pgid= -p "$CPID" | tr -d ' '); { [ -z "$pg" ] || [ "$pg" = "$CPID" ]; } && break; sleep 0.01; done
[ -z "$pg" ] || [ "$pg" = "$CPID" ] || { say "cargo is not its own process group ($pg); stopping it"; kill "$CPID"; exit 1; }
mkdir -p "$BENCH_LOCK_BUILDS"
echo "$$ $(ts) cargo $*" > "$BENCH_LOCK_BUILDS/$CPID"
trap 'rm -f "$BENCH_LOCK_BUILDS/$CPID"' EXIT
trap 'kill -CONT -- -$CPID 2>/dev/null; kill -TERM -- -$CPID 2>/dev/null' INT TERM

pin_group() {  # <cpulist>: every process of the build. A process that exits meanwhile is not an error.
  local p
  for p in $(pgrep -g "$CPID"); do taskset -pc "$1" "$p" >/dev/null 2>&1; done
}
cur=$mode; cur_cpus=$cpus; t0=
while kill -0 "$CPID" 2>/dev/null; do
  read -r mode cpus desc <<< "$(guard_mode)"
  case "$mode" in
    block)
      kill -STOP -- -"$CPID" 2>/dev/null
      [ "$cur" = block ] || { say "froze build $CPID, blocked by: $desc"; t0=$SECONDS; };;
    pin|clear)
      [ "$cur" = block ] && { kill -CONT -- -"$CPID" 2>/dev/null; say "resumed build $CPID after $((SECONDS - t0)) s"; }
      if [ "$mode" = pin ] && { [ "$cur" != pin ] || [ "$cpus" != "$cur_cpus" ]; }; then
        renice --priority 19 -g "$CPID" >/dev/null 2>&1; pin_group "$cpus"; say "re-pinned build $CPID to CPUs $cpus at nice 19 ($desc)"
      elif [ "$mode" = clear ] && [ "$cur" = pin ]; then
        pin_group "$(cat /sys/devices/system/cpu/online)"; say "build $CPID back on all CPUs (still nice 19)"
      fi;;
  esac
  cur=$mode; cur_cpus=$cpus
  sleep "$POLL"
done
wait "$CPID"
