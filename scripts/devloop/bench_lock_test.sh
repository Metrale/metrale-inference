#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# bench_lock_test.sh: bench_lock.sh, cargo-guard.sh and build_env.sh. Runs in a temporary lock directory (never the
# live one) with a stand-in `cargo` on PATH for the guard cases: they test the guard's scheduling, not cargo. The
# stand-in records whether it started, its CPU affinity, nice value and CARGO_BUILD_JOBS, and ticks a counter so a
# freeze is observable. The last case re-runs the "timing hold blocks cargo" checks against a copy of cargo-guard.sh
# whose timing branch is removed, and requires them to FAIL (the test can go red). Exit 0 = PASS.
set -u
HERE=$(dirname "$(readlink -f "$0")")
T=$(mktemp -d); trap 'kill $(jobs -p) $(cat "$T/sleepers" 2>/dev/null) 2>/dev/null; rm -rf "$T"' EXIT
export BENCH_LOCK_DIR=$T/d BENCH_LOCK_LEGACY=$T/legacy BENCH_LOCK_BUILDS=$T/builds BENCH_LOCK_POLL_S=1
export CARGO_GUARD_POLL_S=0.2
mkdir -p "$BENCH_LOCK_DIR" "$T/bin"
RC=0
pass() { echo "PASS $*"; }
fail() { echo "FAIL $*"; RC=1; }
cat > "$T/bin/cargo" <<'EOF'
#!/bin/bash
{ echo started; grep Cpus_allowed_list /proc/self/status; echo "nice=$(ps -o ni= -p $$ | tr -d ' ')"; echo "jobs=${CARGO_BUILD_JOBS:-unset}"; } > "$FAKE_DIR/cargo.$FAKE_ID"
for _ in $(seq 1 "${FAKE_TICKS:-0}"); do echo x >> "$FAKE_DIR/ticks.$FAKE_ID"; sleep 0.1; done
exit "${FAKE_RC:-0}"
EOF
chmod +x "$T/bin/cargo"
export FAKE_DIR=$T
guard() {  # <id> <guard script> <cargo args...>: run a guard in the background with the stand-in cargo
  local id=$1 g=$2; shift 2
  FAKE_ID=$id PATH="$T/bin:$PATH" METRALE_TARGET_MODEL=test-model bash "$g" "$@" 2>"$T/guard.$id.log" &
}
holder() {  # <name> <line>: a live holder file owned by a background sleep (PID in <line> is its pid); prints the pid
  sleep 600 >/dev/null 2>&1 & local p=$!
  echo "$p" >> "$T/sleepers"; printf '%s\n' "${2//PID/$p}" > "$BENCH_LOCK_DIR/$1.$p"; echo "$p"
}
ticks() { wc -l < "$T/ticks.$1" 2>/dev/null || echo 0; }
take() { ( source "$HERE/bench_lock.sh"; bench_lock "$@" 2>>"$T/lock.log" && echo took ); }

# --- bench_lock: arguments ---------------------------------------------------------------------------------------
for args in "o" "o bogus" "o correctness" "o correctness 3-1" "o timing 0-3" "o capture 1 2"; do
  out=$(take $args); [ -z "$out" ] && pass "bench_lock refuses: $args" || fail "bench_lock accepted: $args"
done
[ -z "$(ls "$BENCH_LOCK_DIR")" ] && pass "a refused bench_lock writes no holder" || fail "refused call left $(ls "$BENCH_LOCK_DIR")"
line=$( ( source "$HERE/bench_lock.sh"; bench_lock own correctness 0-9 2>/dev/null; cat "$BENCH_LOCK_FILE" ) )
[[ "$line" == "own "*" class=correctness serve_cpus=0-9" ]] && pass "holder line: $line" || fail "holder line: $line"

# --- bench_lock: waits (the 2026-10-24 cases, now with classes) ----------------------------------------------------
touch "$BENCH_LOCK_DIR/dead.999999"
s=$SECONDS; out=$(timeout 10 bash -c "$(declare -f take); HERE=$HERE T=$T; take t1 timing"); e=$SECONDS
[[ "$out" == took && $((e - s)) -le 2 ]] && pass "dead holder does not block" || fail "dead holder blocked ($out)"
rm -f "$BENCH_LOCK_DIR/dead.999999"
sleep 4 & H=$!; echo "live $H 2026-01-01T00:00:00Z class=correctness serve_cpus=0-1" > "$BENCH_LOCK_DIR/live.$H"
s=$SECONDS; out=$(timeout 15 bash -c "$(declare -f take); HERE=$HERE T=$T; take t2 timing"); e=$SECONDS
[[ "$out" == took && $((e - s)) -ge 3 ]] && pass "a correctness holder blocks another holder ($((e - s)) s)" || fail "live holder did not block ($((e - s)) s: $out)"
rm -f "$BENCH_LOCK_DIR/live.$H"
touch "$BENCH_LOCK_LEGACY"
timeout 3 bash -c "$(declare -f take); HERE=$HERE T=$T; take t3 capture" >/dev/null && fail "legacy file did not block" || pass "legacy file blocks"
rm -f "$BENCH_LOCK_LEGACY"
ls "$BENCH_LOCK_DIR" | grep -q '^t' && fail "a test holder survived its exit: $(ls "$BENCH_LOCK_DIR")" || pass "entries removed on exit"

# --- holder parsing ---------------------------------------------------------------------------------------------
. "$HERE/bench_lock.sh"
f=$T/h; check() { printf '%s\n' "$1" > "$f"; [ "$(bench_holder_fields "$f")" = "$2" ] && pass "parse [$1] -> $2" || fail "parse [$1] -> $(bench_holder_fields "$f"), want $2"; }
check "glm-r1r4-X 3334651 2026-10-24T13:41:55Z" "timing - legacy"
check "o 1 t class=capture serve_cpus=-" "capture - ok"
check "o 1 t class=correctness serve_cpus=0-9,12" "correctness 0-9,12 ok"
check "o 1 t class=correctness serve_cpus=" "timing - unreadable"
check "o 1 t class=fast serve_cpus=-" "timing - unreadable"
check "o 1 t extra" "timing - unreadable"

# --- cargo-guard: a timing hold blocks cargo, a legacy hold counts as timing ------------------------------------
blocks_case() {  # <guard script> <tag>: 0 when the guard holds cargo back under a timing hold and a legacy hold
  local g=$1 tag=$2 ok=0 p
  rm -f "$BENCH_LOCK_DIR"/*
  p=$(holder tim "tim PID 2026-10-24T00:00:00Z class=timing serve_cpus=-")
  guard "blk$tag" "$g" build; local gp=$!
  sleep 2; [ -e "$T/cargo.blk$tag" ] && ok=1
  kill "$p"; wait "$gp"; [ -e "$T/cargo.blk$tag" ] || ok=1
  p=$(holder glm-old "glm-old PID 2026-10-24T00:00:00Z")
  guard "leg$tag" "$g" build; gp=$!
  sleep 2; [ -e "$T/cargo.leg$tag" ] && ok=1
  grep -q '(legacy)' "$T/guard.leg$tag.log" || ok=1
  kill "$p"; wait "$gp"
  return $ok
}
blocks_case "$HERE/cargo-guard.sh" "" && pass "timing and legacy holds keep cargo from starting; it runs once they end" || fail "a timing or legacy hold did not block cargo"

# --- cargo-guard: a correctness hold runs cargo niced on the CPUs outside the serve's ----------------------------
rm -f "$BENCH_LOCK_DIR"/*
p=$(holder cor "cor PID 2026-10-24T00:00:00Z class=correctness serve_cpus=0-9")
guard pin "$HERE/cargo-guard.sh" build; gp=$!; wait "$gp"; kill "$p"
grep -q 'Cpus_allowed_list:	10-19' "$T/cargo.pin" && grep -q 'nice=19' "$T/cargo.pin" && grep -q 'jobs=10' "$T/cargo.pin" \
  && pass "correctness hold: cargo on CPUs 10-19, nice 19, 10 jobs" || fail "correctness hold: $(tr '\n' ' ' < "$T/cargo.pin" 2>/dev/null)"

# --- cargo-guard: a timing hold that starts mid-build freezes it, and bench_lock waits for the freeze --------------
rm -f "$BENCH_LOCK_DIR"/*
FAKE_TICKS=60 guard frz "$HERE/cargo-guard.sh" build; gp=$!
sleep 1; a=$(ticks frz)
( source "$HERE/bench_lock.sh"; bench_lock frzholder timing 2>>"$T/lock.log"; echo "locked $(ls "$BENCH_LOCK_BUILDS")" > "$T/frz.locked"
  pg=$(ls "$BENCH_LOCK_BUILDS"); echo "frozen=$(bench_build_frozen "$pg" && echo yes || echo no)" >> "$T/frz.locked"; sleep 4 ) &
lp=$!; sleep 1.5; b=$(ticks frz); sleep 1; c=$(ticks frz)
wait "$lp"; sleep 1.5; d=$(ticks frz); wait "$gp"
grep -q 'frozen=yes' "$T/frz.locked" && [ "$a" -gt 0 ] && [ "$b" = "$c" ] && [ "$d" -gt "$c" ] \
  && pass "mid-build timing hold: build frozen ($a then $b=$c ticks) before bench_lock returned; resumed to $d after" \
  || { fail "mid-build freeze: ticks $a/$b/$c/$d, $(cat "$T/frz.locked" 2>/dev/null)"; cat "$T/guard.frz.log" "$T/lock.log"; }
[ -z "$(ls "$BENCH_LOCK_BUILDS")" ] && pass "build registry entry removed" || fail "registry left $(ls "$BENCH_LOCK_BUILDS")"

# --- cargo-guard: exit status and refusals ------------------------------------------------------------------------
FAKE_RC=7 guard rc "$HERE/cargo-guard.sh" build; wait $!; r=$?
[ "$r" = 7 ] && pass "cargo's exit status passes through" || { fail "exit status $r, want 7"; cat "$T/guard.rc.log"; }
( FAKE_ID=nomodel PATH="$T/bin:$PATH" bash "$HERE/cargo-guard.sh" build 2>"$T/guard.nomodel.log" ); r=$?
[ "$r" = 2 ] && [ ! -e "$T/cargo.nomodel" ] && pass "no lane recipe and no target model: refused (2), cargo not run" || fail "no-model: rc $r"

# --- build_env: the switches --------------------------------------------------------------------------------------
envof() { ( cd "$HERE/../.." && unset METRALE_PTX_CACHE METRALE_PTX_CACHE_MAX_BYTES CARGO_INCREMENTAL RUSTFLAGS METRALE_TARGET_HW METRALE_TARGET_QUANT
  . "$HERE/build_env.sh"; eval "$1"; devloop_build_env "${@:2}" 2>/dev/null || { echo "rc=$?"; exit; }
  echo "ptx=${METRALE_PTX_CACHE:-} max=${METRALE_PTX_CACHE_MAX_BYTES:-} inc=${CARGO_INCREMENTAL:-unset} model=${METRALE_TARGET_HW:-}/${METRALE_TARGET_MODEL:-}/${METRALE_TARGET_QUANT:-} lld=$([[ "${RUSTFLAGS:-}" == *-fuse-ld=lld* ]] && echo y || echo n)" ); }
expect() { local got; got=$(envof "$1" "${@:3}"); [ "$got" = "$2" ] && pass "env [$1 | ${*:3}] -> $got" || fail "env [$1 | ${*:3}] -> $got, want $2"; }
L=glm-5.3/glm-5.3-flash-nvfp4-tp3-512k
expect "export METRALE_LANE_RECIPE=$L" "ptx=1 max=4294967296 inc=0 model=gb10/glm-5.3-flash/nvfp4 lld=n" build -p met
expect "export METRALE_LANE_RECIPE=$L" "ptx=1 max=4294967296 inc=unset model=gb10/glm-5.3-flash/nvfp4 lld=y" build --profile iter -p met
expect "export METRALE_LANE_RECIPE=$L" "ptx=1 max=4294967296 inc=unset model=gb10/glm-5.3-flash/nvfp4 lld=y" build --profile=iter
expect "export METRALE_LANE_RECIPE=$L METRALE_PTX_CACHE=0" "ptx=0 max= inc=0 model=gb10/glm-5.3-flash/nvfp4 lld=n" check
expect "export METRALE_TARGET_MODEL='*' METRALE_LANE_RECIPE=$L" "ptx=1 max=4294967296 inc=0 model=/*/ lld=n" build
expect "export METRALE_LANE_RECIPE=glm-5.3/no-such-recipe" "rc=2" build
expect "export METRALE_LANE_RECIPE=$L METRALE_TARGET_QUANT=fp8" "rc=2" build
expect "export METRALE_LANE_RECIPE=$L METRALE_PTX_CACHE=yes" "rc=2" build
expect "true" "rc=2" build
expect "true" "ptx= max= inc=0 model=// lld=n" fmt --check

# --- the test can go red: the same timing checks against a guard whose timing branch is removed --------------------
mkdir -p "$T/sab"; cp "$HERE/bench_lock.sh" "$HERE/build_env.sh" "$T/sab/"
sed 's/\*) echo "block - \$(echo "\$h" | bench_holders_brief)"; return;;/*) ;;/' "$HERE/cargo-guard.sh" > "$T/sab/cargo-guard.sh"
cmp -s "$HERE/cargo-guard.sh" "$T/sab/cargo-guard.sh" && fail "sabotage did not apply"
blocks_case "$T/sab/cargo-guard.sh" sab && fail "sabotaged guard still passed the timing case (the test cannot go red)" \
  || pass "sabotaged guard (timing branch removed) fails the timing case: RED as required"
exit $RC
