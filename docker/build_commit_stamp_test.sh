#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# 2026-10-11: Every release and image build of `met` stamps the source commit as
# METRALE_BUILD_COMMIT (investor-mvp #28). The server keys each InferenceUsage
# record and the /metrics build line by it; an unstamped binary withholds the
# records, so a build path that forgets the stamp ships a server that silently
# reports no usage. No Docker daemon, no GPU, no network:
#   (a) each Dockerfile that compiles metrale-server declares
#       `ARG METRALE_GIT_SHA` in its build stage and passes it to that cargo
#       build as METRALE_BUILD_COMMIT;
#   (b) release-build.yml stamps from scripts/build-commit.sh before its first
#       cargo build, and datacenter-binaries.yml sets it on its build step;
#   (c) scripts/build-commit.sh prints HEAD's 40-hex commit for a clean tree
#       and refuses (nothing on stdout) a dirty tree or a directory with no commit.
#
# Usage: bash docker/build_commit_stamp_test.sh
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fail() { echo "ASSERT FAILED [$1]: $2" >&2; exit 1; }
ok() { echo "  ok [$1] $2"; }
instructions() { sed -e 's/[[:space:]]*#.*$//' -e '/^[[:space:]]*$/d' "$1"; }

# 2026-10-11: ── (a) Dockerfiles ───────────────────────────────────────────────────────────
checked=0
while IFS= read -r f; do
  rel="${f#"$ROOT"/}"
  build="$(instructions "$f" | grep -nE '^RUN[[:space:]].*cargo build .*-p metrale-server' || true)"
  [ -n "$build" ] || continue
  checked=$((checked + 1))
  line="${build%%:*}"
  printf '%s\n' "$build" | grep -qE 'RUN[[:space:]]+METRALE_BUILD_COMMIT="\$\{METRALE_GIT_SHA\}"[[:space:]]' \
    || fail a "$rel: the metrale-server build does not set METRALE_BUILD_COMMIT=\"\${METRALE_GIT_SHA}\""
  # 2026-10-11: The ARG must be declared in the same stage, before the RUN: an ARG is
  # scoped to the stage that declares it, so one from another stage is empty.
  stage_start="$(instructions "$f" | head -n "$line" | grep -nE '^FROM[[:space:]]' | tail -1 | cut -d: -f1)"
  instructions "$f" | sed -n "${stage_start},${line}p" | grep -qE '^ARG[[:space:]]+METRALE_GIT_SHA[[:space:]]*$' \
    || fail a "$rel: no \`ARG METRALE_GIT_SHA\` in the build stage before the metrale-server build"
  ok a "$rel stamps met from METRALE_GIT_SHA"
done < <(find "$ROOT/docker" -type f -name 'Dockerfile*' ! -name '*.dockerignore' | sort)
[ "$checked" -gt 0 ] || fail a "no Dockerfile under docker/ builds metrale-server; the walk is broken"

# 2026-10-11: ── (b) workflows ─────────────────────────────────────────────────────────────
wf="$ROOT/.github/workflows/release-build.yml"
stamp="$(grep -nE 'commit="\$\(bash scripts/build-commit.sh\)"' "$wf" | head -1 | cut -d: -f1)"
[ -n "$stamp" ] || fail b "release-build.yml never assigns commit=\"\$(bash scripts/build-commit.sh)\""
grep -qE 'echo "METRALE_BUILD_COMMIT=\$commit" >> "\$GITHUB_ENV"' "$wf" \
  || fail b "release-build.yml does not export the stamp through GITHUB_ENV"
first_build="$(grep -nE 'cargo build .*-p metrale-server' "$wf" | head -1 | cut -d: -f1)"
[ -n "$first_build" ] && [ "$stamp" -lt "$first_build" ] \
  || fail b "release-build.yml stamps at line $stamp, after its first metrale-server build (line $first_build)"
ok b "release-build.yml stamps before building"

wf="$ROOT/.github/workflows/datacenter-binaries.yml"
env_line="$(grep -nE '^[[:space:]]+METRALE_BUILD_COMMIT: \$\{\{ github\.sha \}\}' "$wf" | cut -d: -f1)"
build_line="$(grep -nE 'cargo build --release -p metrale-server' "$wf" | cut -d: -f1)"
step_line="$(grep -nE '^[[:space:]]+- name: Build met for' "$wf" | cut -d: -f1)"
[ -n "$env_line" ] && [ -n "$build_line" ] && [ -n "$step_line" ] \
  && [ "$step_line" -lt "$env_line" ] && [ "$env_line" -lt "$build_line" ] \
  || fail b "datacenter-binaries.yml does not set METRALE_BUILD_COMMIT on its build step"
ok b "datacenter-binaries.yml stamps its build step"

# 2026-10-11: ── (c) the script ────────────────────────────────────────────────────────────
script="$ROOT/scripts/build-commit.sh"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
g() { git -C "$tmp/repo" -c user.name=t -c user.email=t@example.invalid -c commit.gpgsign=false "$@"; }
mkdir -p "$tmp/repo" "$tmp/empty"
g init -q
if out="$(bash "$script" "$tmp/repo" 2>/dev/null)"; then fail c "a repo with no commit was stamped '$out'"; fi
[ -z "${out:-}" ] || fail c "a refusal printed '$out' on stdout"
echo one > "$tmp/repo/f"
g add f && g commit -q -m one
want="$(g rev-parse HEAD)"
got="$(bash "$script" "$tmp/repo")" || fail c "a clean tree was refused"
[ "$got" = "$want" ] || fail c "stamped '$got', HEAD is '$want'"
printf '%s' "$got" | grep -Eqx '[0-9a-f]{40}' || fail c "'$got' is not 40 lower-case hex"
echo untracked > "$tmp/repo/new"
got="$(bash "$script" "$tmp/repo")" || fail c "an untracked file made a clean tree refuse"
[ "$got" = "$want" ] || fail c "untracked file: stamped '$got'"
echo two > "$tmp/repo/f"
if out="$(bash "$script" "$tmp/repo" 2>/dev/null)"; then fail c "a dirty tree was stamped '$out'"; fi
[ -z "$out" ] || fail c "a refusal printed '$out' on stdout"
g add f
if bash "$script" "$tmp/repo" >/dev/null 2>&1; then fail c "a staged change was stamped"; fi
if bash "$script" "$tmp/empty" >/dev/null 2>&1; then fail c "a directory outside git was stamped"; fi
ok c "build-commit.sh stamps a clean HEAD and refuses everything else"

echo "build commit stamp: $checked Dockerfiles + 2 workflows + the script"
