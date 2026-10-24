#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# bench_lock.sh: the class-aware GPU benchmark lock. Source it, then call
#   bench_lock <owner> timing|capture
#   bench_lock <owner> correctness <serve-cpus>
# The class is required (no default). <serve-cpus> is the CPU list (e.g. 0-9) that this box's serve is pinned to;
# it is required for correctness and refused for the other classes. Pinning the serve to it is the caller's job.
#
# One GPU owner at a time: bench_lock waits while any other live holder exists, whatever its class, then writes
# BENCH_ACTIVE.d/<owner>.<pid> under an flock and removes only that file on EXIT. A timing or capture holder then
# also waits until every build that cargo-guard.sh runs is frozen (SIGSTOP), so no build overlaps its legs.
#
# What a class means to cargo-guard.sh: timing and capture freeze every build; during correctness holds a build
# runs at nice 19 on the online CPUs outside the union of the holders' serve CPUs.
#
# Holder file: "<owner> <pid> <utc> class=<class> serve_cpus=<list|->". A legacy holder line "<owner> <pid> <utc>"
# (no class) and the legacy BENCH_ACTIVE file count as timing, and the holder listing marks them "legacy".
# A class field that does not parse also counts as timing (fail closed) and is marked "unreadable".
BENCH_LOCK_DIR=${BENCH_LOCK_DIR:-/workspace/claude-state/glm/agentic/BENCH_ACTIVE.d}
BENCH_LOCK_LEGACY=${BENCH_LOCK_LEGACY:-/workspace/claude-state/glm/agentic/BENCH_ACTIVE}
# The registry of running cargo-guard builds: one file per build, named by the build's process group id.
BENCH_LOCK_BUILDS=${BENCH_LOCK_BUILDS:-$BENCH_LOCK_DIR.builds}
BENCH_LOCK_CLASSES="timing capture correctness"

bench_cpulist_expand() {  # "0-3,8" -> "0 1 2 3 8"; returns 1 on a malformed list
  local part a b out=() IFS=,
  [[ "$1" =~ ^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$ ]] || return 1
  for part in $1; do
    a=${part%-*}; b=${part#*-}
    [ "$a" -le "$b" ] || return 1
    while [ "$a" -le "$b" ]; do out+=("$a"); a=$((a + 1)); done
  done
  IFS=' '; echo "${out[*]}"
}

bench_holder_fields() {  # <file> -> "<class> <serve_cpus> <note>"; note is ok, legacy or unreadable
  local line class=- cpus=- tok n=0
  read -r line < "$1" || true
  for tok in $line; do
    n=$((n + 1))
    case "$tok" in class=*) class=${tok#class=};; serve_cpus=*) cpus=${tok#serve_cpus=};; esac
  done
  if [ "$class" = - ] && [ "$n" -le 3 ]; then echo "timing - legacy"; return; fi
  case "$class" in
    timing|capture) echo "$class - ok";;
    correctness) bench_cpulist_expand "$cpus" >/dev/null && echo "correctness $cpus ok" || echo "timing - unreadable";;
    *) echo "timing - unreadable";;
  esac
}

bench_holders() {  # [<pid to skip>] -> one line per live holder: "<path> <class> <serve_cpus> <note>"
  [ -e "$BENCH_LOCK_LEGACY" ] && echo "$BENCH_LOCK_LEGACY timing - legacy"
  local f pid
  for f in "$BENCH_LOCK_DIR"/*; do
    [ -f "$f" ] || continue
    pid=${f##*.}
    [ "$pid" = "${1:-}" ] && continue
    kill -0 "$pid" 2>/dev/null && echo "$f $(bench_holder_fields "$f")"
  done
}

bench_holders_brief() {  # stdin: bench_holders lines -> "name:class[(note)] ..."
  local f c s n
  while read -r f c s n; do
    [ "$n" = ok ] && printf '%s:%s ' "${f##*/}" "$c" || printf '%s:%s(%s) ' "${f##*/}" "$c" "$n"
  done
}

bench_build_frozen() {  # <pgid>: true when every process of the group is stopped, or the group is gone
  local p st
  for p in $(pgrep -g "$1"); do
    st=$(ps -o stat= -p "$p" 2>/dev/null) || continue
    case "$st" in T*|t*|Z*|'') ;; *) return 1;; esac
  done
  return 0
}

bench_builds_running() {  # -> the pgids of registered builds that still have a process not stopped
  local f
  for f in "$BENCH_LOCK_BUILDS"/*; do
    [ -f "$f" ] || continue
    bench_build_frozen "${f##*/}" || echo "${f##*/}"
  done
}

bench_lock() {
  local owner=${1:-} class=${2:-} cpus=${3:-} h b
  [ -n "$owner" ] || { echo "bench_lock: owner name required" >&2; return 2; }
  case "$class" in
    timing|capture|correctness) ;;
    *) echo "bench_lock: class required (one of: $BENCH_LOCK_CLASSES), got '${class}'" >&2; return 2;;
  esac
  if [ "$class" = correctness ]; then
    bench_cpulist_expand "$cpus" >/dev/null || { echo "bench_lock: correctness needs the serve's CPU list (e.g. 0-9), got '${cpus}'" >&2; return 2; }
  else
    [ -z "$cpus" ] || { echo "bench_lock: a $class hold takes no CPU list (builds are frozen), got '${cpus}'" >&2; return 2; }
    cpus=-
  fi
  [ $# -le 3 ] || { echo "bench_lock: too many arguments" >&2; return 2; }
  mkdir -p "$BENCH_LOCK_DIR"
  while :; do
    exec {BENCH_LOCK_FD}>"$BENCH_LOCK_DIR/.mutex"
    flock "$BENCH_LOCK_FD"
    h=$(bench_holders $$)
    if [ -z "$h" ]; then
      BENCH_LOCK_FILE="$BENCH_LOCK_DIR/$owner.$$"
      printf '%s %s %s class=%s serve_cpus=%s\n' "$owner" "$$" "$(date -u +%FT%TZ)" "$class" "$cpus" > "$BENCH_LOCK_FILE"
      flock -u "$BENCH_LOCK_FD"; exec {BENCH_LOCK_FD}>&-
      break
    fi
    flock -u "$BENCH_LOCK_FD"; exec {BENCH_LOCK_FD}>&-
    echo "bench_lock: $(date -u +%FT%TZ) $owner ($class) waiting on: $(echo "$h" | bench_holders_brief)" >&2
    sleep "${BENCH_LOCK_POLL_S:-60}"
  done
  trap 'rm -f "$BENCH_LOCK_FILE"' EXIT
  [ "$class" = correctness ] && return 0
  local since=$SECONDS waited=0
  while b=$(bench_builds_running) && [ -n "$b" ]; do
    [ $waited = 1 ] || echo "bench_lock: $owner ($class) waiting for builds to freeze: $(echo $b)" >&2
    waited=1; sleep 1
  done
  [ $waited = 0 ] || echo "bench_lock: builds frozen after $((SECONDS - since)) s" >&2
  return 0
}
