#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# 2026-10-07: Provision native Ubuntu 24.04 (bare metal, no WSL) for the
# native-HIP Strix Halo target (`METRALE_TARGET_HW=strix-hip`, gfx1151). This
# is NOT the SCALE `strix` target; that path is docs/porting/amd-strix-halo-scale.md.
# The Windows + WSL variant of the same target is wsl-setup.sh.
#
# Installs or sets, at the pinned values below and nothing newer:
#   - ROCm userspace through amdgpu-install (`--usecase=rocm,hiplibsdk
#     --no-dkms`; the in-box amdgpu module of the Ubuntu kernel drives the GPU),
#   - the host build dependencies of `cargo build -p metrale-server`,
#   - render/video group membership for the build user,
#   - /etc/profile.d/rocm-native.sh (/opt/rocm/bin on PATH; no DXG variable,
#     ROCr reaches the GPU through /dev/kfd),
#   - the kernel command line in /etc/default/grub (GRUB_CMDLINE_LINUX_DEFAULT)
#     so the GPU can map almost all of RAM as GTT, then `update-grub`.
#
# It does NOT reboot. The new command line applies on the next boot; the final
# line says whether the running kernel already has it.
#
# Usage (as root on the target box):
#   sudo bash scripts/strix-hip/linux-setup.sh <linux-user>
#
# Idempotent: a rerun on a provisioned box re-verifies and changes nothing. A
# different ROCm or amdgpu-install already installed is a hard error; this
# script never upgrades or downgrades over it. The GTT pins are sized for a
# 128 GB box (124 GiB of GTT); on a box with less RAM the script refuses.
#
# Every pin is mirrored in the version table of docs/porting/strix-hip-setup.md;
# scripts/strix-hip/pins_test.py fails CI when they disagree, and also when the
# ROCm pins here drift from wsl-setup.sh.
#
# TODO: derive GTT_MIB from MemTotal for boxes other than 128 GB, once a second
# RAM size has been validated.
# TODO: add a `--check` mode that runs only the final verification block.
set -euo pipefail

ROCM_VERSION="7.2.1"
AMDGPU_INSTALL_VERSION="7.2.1.70201-1"
# The .deb file is named for the ROCm release; its dpkg Version field is not.
AMDGPU_INSTALL_PKG_VERSION="30.30.1.0.30300100-2303411.24.04"
AMDGPU_INSTALL_URL="https://repo.radeon.com/amdgpu-install/7.2.1/ubuntu/noble/amdgpu-install_7.2.1.70201-1_all.deb"
AMDGPU_INSTALL_SHA256="4c0338a241c15b12c14eb3aeb4012ea0d55dba681737ea8482248041a16c2afa"
UBUNTU_VERSION_ID="24.04"
# GTT size in MiB (124 GiB). TTM counts 4 KiB pages: 126976 MiB * 256 = 32505856.
GTT_MIB="126976"
TTM_PAGES="32505856"
# Kernel parameters, `key=value` each, exactly as on the validated box.
# amdgpu.gttsize is in MiB; the two ttm limits are in 4 KiB pages. amd_iommu=off
# is part of the validated configuration; it was not measured with the IOMMU on.
CMDLINE_PARAMS="amd_iommu=off amdgpu.gttsize=126976 ttm.pages_limit=32505856 ttm.page_pool_size=32505856"
GRUB_FILE="/etc/default/grub"
BUILD_DEPS=(curl wget ca-certificates build-essential pkg-config git cmake libclang-dev python3 python3-venv)
PROFILE_FILE="/etc/profile.d/rocm-native.sh"

die() { echo "linux-setup: FATAL: $*" >&2; exit 1; }
log() { echo "linux-setup: $*"; }

# dpkg version of an installed package, empty when absent.
pkg_version() { dpkg-query -W -f='${Status} ${Version}\n' "$1" 2>/dev/null | awk '$3 == "installed" { print $4 }'; }

# Installed ROCm release from /opt/rocm/.info/version (e.g. "7.2.1-81" -> "7.2.1").
rocm_installed() {
  [ -r /opt/rocm/.info/version ] || return 0
  sed -E 's/-.*$//' /opt/rocm/.info/version | head -n1
}

fetch_verified() { # url sha256 dest
  local url=$1 sha=$2 dest=$3
  if [ ! -f "$dest" ] || ! echo "$sha  $dest" | sha256sum -c --status; then
    log "downloading $url"
    curl -fsSL --retry 3 -o "$dest" "$url" || wget -q -O "$dest" "$url"
  fi
  echo "$sha  $dest" | sha256sum -c --status \
    || die "sha256 mismatch for $dest (expected $sha, got $(sha256sum "$dest" | cut -d' ' -f1))"
}

# Current GRUB_CMDLINE_LINUX_DEFAULT value (double-quoted form only).
grub_default_cmdline() {
  sed -n 's/^GRUB_CMDLINE_LINUX_DEFAULT="\(.*\)"[[:space:]]*$/\1/p' "$GRUB_FILE" | tail -n1
}

# $1 with every token whose key matches one of CMDLINE_PARAMS removed, then
# CMDLINE_PARAMS appended. Unrelated tokens (quiet, splash, ...) keep their order.
merged_cmdline() {
  local - cur=$1 tok key p out=() drop
  set -f
  for tok in $cur; do
    key=${tok%%=*}; drop=0
    for p in $CMDLINE_PARAMS; do [ "$key" = "${p%%=*}" ] && drop=1; done
    [ "$drop" -eq 1 ] || out+=("$tok")
  done
  for p in $CMDLINE_PARAMS; do out+=("$p"); done
  echo "${out[*]}"
}

# Preconditions --------------------------------------------------------------
[ "$(id -u)" -eq 0 ] || die "run as root (sudo)"
TARGET_USER=${1:-${SUDO_USER:-}}
[ -n "$TARGET_USER" ] || die "usage: linux-setup.sh <linux-user>"
id "$TARGET_USER" >/dev/null 2>&1 || die "user '$TARGET_USER' does not exist"
if grep -qi microsoft /proc/version; then
  die "this is a WSL kernel; use scripts/strix-hip/wsl-setup.sh there"
fi
# shellcheck source=/dev/null
. /etc/os-release
[ "${ID:-}" = "ubuntu" ] && [ "${VERSION_ID:-}" = "$UBUNTU_VERSION_ID" ] \
  || die "need Ubuntu $UBUNTU_VERSION_ID, found ${PRETTY_NAME:-unknown}"
[ $((GTT_MIB * 256)) -eq "$TTM_PAGES" ] || die "internal: TTM_PAGES is not GTT_MIB * 256"
case " $CMDLINE_PARAMS " in
  *" amdgpu.gttsize=$GTT_MIB "*" ttm.pages_limit=$TTM_PAGES ttm.page_pool_size=$TTM_PAGES "*) ;;
  *) die "internal: CMDLINE_PARAMS does not carry GTT_MIB/TTM_PAGES" ;;
esac
mem_mib=$(( $(awk '/^MemTotal:/ { print $2 }' /proc/meminfo) / 1024 ))
[ "$mem_mib" -ge "$GTT_MIB" ] \
  || die "MemTotal is $mem_mib MiB, below the pinned GTT of $GTT_MIB MiB; these pins are for a 128 GB box"
[ -f "$GRUB_FILE" ] || die "$GRUB_FILE not found; this script configures GRUB only"
command -v update-grub >/dev/null || die "update-grub not found"
grep -q '^GRUB_CMDLINE_LINUX_DEFAULT="' "$GRUB_FILE" \
  || die "$GRUB_FILE has no double-quoted GRUB_CMDLINE_LINUX_DEFAULT line; edit it by hand"
for f in /etc/default/grub.d/*.cfg; do
  [ -e "$f" ] || continue
  if grep -q 'GRUB_CMDLINE_LINUX_DEFAULT' "$f"; then
    die "$f also sets GRUB_CMDLINE_LINUX_DEFAULT and would override $GRUB_FILE; merge it by hand"
  fi
done
if ! lspci -n 2>/dev/null | grep -q ' 1002:1586'; then
  log "warning: no AMD 1002:1586 (Strix Halo GPU) PCI device seen; continuing"
fi

have_rocm=$(rocm_installed)
if [ -n "$have_rocm" ] && [ "$have_rocm" != "$ROCM_VERSION" ]; then
  die "ROCm $have_rocm is installed; this target is validated on $ROCM_VERSION only. Remove it (amdgpu-uninstall) first."
fi
have_ai=$(pkg_version amdgpu-install)
if [ -n "$have_ai" ] && [ "$have_ai" != "$AMDGPU_INSTALL_PKG_VERSION" ]; then
  die "amdgpu-install $have_ai is installed; expected $AMDGPU_INSTALL_PKG_VERSION (the $AMDGPU_INSTALL_VERSION release)"
fi
if [ -n "$(pkg_version amdgpu-dkms)" ]; then
  die "amdgpu-dkms is installed; this path uses the in-box kernel module (--no-dkms). Remove amdgpu-dkms first."
fi

export DEBIAN_FRONTEND=noninteractive
dpkg --configure -a
apt-get -f install -y -q

WORK=$(mktemp -d /tmp/strix-hip-setup.XXXXXX)
trap 'rm -rf "$WORK"' EXIT

# Build dependencies ----------------------------------------------------------
missing=()
for p in "${BUILD_DEPS[@]}"; do [ -n "$(pkg_version "$p")" ] || missing+=("$p"); done
if [ "${#missing[@]}" -gt 0 ]; then
  log "installing build deps: ${missing[*]}"
  apt-get update -q
  apt-get install -y -q "${missing[@]}"
fi

# ROCm userspace --------------------------------------------------------------
if [ -z "$have_ai" ]; then
  fetch_verified "$AMDGPU_INSTALL_URL" "$AMDGPU_INSTALL_SHA256" "$WORK/amdgpu-install.deb"
  apt-get install -y -q "$WORK/amdgpu-install.deb"
fi
if [ -z "$have_rocm" ] || ! command -v /opt/rocm/bin/hipcc >/dev/null; then
  log "installing ROCm $ROCM_VERSION userspace (no dkms)"
  amdgpu-install -y --usecase=rocm,hiplibsdk --no-dkms
fi
ldconfig

# Groups and environment -------------------------------------------------------
usermod -aG render,video "$TARGET_USER"
# Single-quoted on purpose: PATH expands at login, not now.
# shellcheck disable=SC2016
profile_body='# Written by scripts/strix-hip/linux-setup.sh. Native ROCr uses /dev/kfd; no DXG variable.
case ":$PATH:" in *:/opt/rocm/bin:*) ;; *) export PATH=/opt/rocm/bin:$PATH ;; esac'
if [ "$(cat "$PROFILE_FILE" 2>/dev/null)" != "$profile_body" ]; then
  printf '%s\n' "$profile_body" > "$PROFILE_FILE"
  chmod 0644 "$PROFILE_FILE"
fi

# Kernel command line -----------------------------------------------------------
cur_cmdline=$(grub_default_cmdline)
new_cmdline=$(merged_cmdline "$cur_cmdline")
if [ "$cur_cmdline" != "$new_cmdline" ]; then
  backup="$GRUB_FILE.strix-hip.$(date +%Y%m%d%H%M%S)"
  cp -a "$GRUB_FILE" "$backup"
  awk -v v="$new_cmdline" '
    /^GRUB_CMDLINE_LINUX_DEFAULT="/ { print "GRUB_CMDLINE_LINUX_DEFAULT=\"" v "\""; next }
    { print }' "$backup" > "$WORK/grub"
  cat "$WORK/grub" > "$GRUB_FILE"
  log "GRUB_CMDLINE_LINUX_DEFAULT: '$cur_cmdline' -> '$new_cmdline' (backup $backup)"
  update-grub
else
  log "GRUB_CMDLINE_LINUX_DEFAULT already carries the pinned parameters"
fi

# Verification (fails loudly on any drift) ------------------------------------
got_rocm=$(rocm_installed)
[ "$got_rocm" = "$ROCM_VERSION" ] || die "ROCm after install is '${got_rocm:-absent}', expected $ROCM_VERSION"
got_ai=$(pkg_version amdgpu-install)
[ "$got_ai" = "$AMDGPU_INSTALL_PKG_VERSION" ] || die "amdgpu-install after install is '${got_ai:-absent}'"
[ -x /opt/rocm/bin/hipcc ] || die "/opt/rocm/bin/hipcc missing after install"
/opt/rocm/bin/hipcc --version | grep -q "roc-$ROCM_VERSION" \
  || die "hipcc does not report roc-$ROCM_VERSION: $(/opt/rocm/bin/hipcc --version | head -n2 | tr '\n' ' ')"
[ "$(grub_default_cmdline)" = "$(merged_cmdline "$(grub_default_cmdline)")" ] \
  || die "$GRUB_FILE does not carry the pinned parameters after the edit"

running_ok=1
for p in $CMDLINE_PARAMS; do
  grep -qw -- "$p" /proc/cmdline || running_ok=0
done
log "OK: ROCm $got_rocm, amdgpu-install $got_ai, user $TARGET_USER in render,video"
if [ "$running_ok" -eq 1 ]; then
  log "running kernel already has: $CMDLINE_PARAMS"
  log "next: as $TARGET_USER, in a new login shell, run scripts/strix-hip/preflight.sh"
else
  log "REBOOT REQUIRED: the running kernel lacks some of: $CMDLINE_PARAMS"
  log "next: reboot, then as $TARGET_USER run scripts/strix-hip/preflight.sh"
fi
