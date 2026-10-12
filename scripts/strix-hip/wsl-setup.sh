#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# 2026-10-07: Provision WSL Ubuntu 24.04 for the native-HIP Strix Halo target
# (`METRALE_TARGET_HW=strix-hip`, gfx1151). This is NOT the SCALE `strix`
# target; that path is docs/porting/amd-strix-halo-scale.md.
#
# Installs, at the pinned versions below and nothing newer:
#   - ROCm userspace through amdgpu-install (`--usecase=rocm,hiplibsdk
#     --no-dkms`; WSL has no amdgpu kernel module to build),
#   - librocdxg (`rocdxg-roct`), the DXG bridge that lets ROCr reach the GPU
#     through the Windows driver,
#   - the host build dependencies of `cargo build -p metrale-server`,
#   - render/video group membership for the build user,
#   - /etc/profile.d/rocm-wsl.sh (HSA_ENABLE_DXG_DETECTION=1, /opt/rocm/bin).
#
# Usage (as root inside the distro; Windows prerequisites come first, see
# docs/porting/strix-hip-setup.md):
#   wsl -d Ubuntu-24.04 -u root -e bash /mnt/c/Users/<you>/wsl-setup.sh <linux-user>
#
# Idempotent: a rerun on a provisioned distro re-verifies and changes nothing.
# A different ROCm or librocdxg already installed is a hard error; this script
# never upgrades or downgrades over it.
#
# Every pin is mirrored in the version table of docs/porting/strix-hip-setup.md;
# scripts/strix-hip/pins_test.py fails CI when the two disagree.
#
# TODO: add a `--check` mode that runs only the final verification block.
set -euo pipefail

ROCM_VERSION="7.2.1"
AMDGPU_INSTALL_VERSION="7.2.1.70201-1"
# The .deb file is named for the ROCm release; its dpkg Version field is not.
AMDGPU_INSTALL_PKG_VERSION="30.30.1.0.30300100-2303411.24.04"
AMDGPU_INSTALL_URL="https://repo.radeon.com/amdgpu-install/7.2.1/ubuntu/noble/amdgpu-install_7.2.1.70201-1_all.deb"
AMDGPU_INSTALL_SHA256="4c0338a241c15b12c14eb3aeb4012ea0d55dba681737ea8482248041a16c2afa"
ROCDXG_VERSION="1.2.2"
ROCDXG_URL="https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-roct_1.2.2_amd64.deb"
ROCDXG_SHA256="28ded1254811192ebace1f76c0227580184af7b27ab2475fb9728295a702d541"
UBUNTU_VERSION_ID="24.04"
BUILD_DEPS=(curl wget ca-certificates build-essential pkg-config git cmake libclang-dev python3 python3-venv)
PROFILE_FILE="/etc/profile.d/rocm-wsl.sh"

die() { echo "wsl-setup: FATAL: $*" >&2; exit 1; }
log() { echo "wsl-setup: $*"; }

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

# Preconditions --------------------------------------------------------------
[ "$(id -u)" -eq 0 ] || die "run as root (wsl -u root)"
TARGET_USER=${1:-${SUDO_USER:-}}
[ -n "$TARGET_USER" ] || die "usage: wsl-setup.sh <linux-user>"
id "$TARGET_USER" >/dev/null 2>&1 || die "user '$TARGET_USER' does not exist"
grep -qi microsoft /proc/version || die "not a WSL kernel; this script provisions WSL only"
# shellcheck source=/dev/null
. /etc/os-release
[ "${ID:-}" = "ubuntu" ] && [ "${VERSION_ID:-}" = "$UBUNTU_VERSION_ID" ] \
  || die "need Ubuntu $UBUNTU_VERSION_ID, found ${PRETTY_NAME:-unknown}"

have_rocm=$(rocm_installed)
if [ -n "$have_rocm" ] && [ "$have_rocm" != "$ROCM_VERSION" ]; then
  die "ROCm $have_rocm is installed; this target is validated on $ROCM_VERSION only. Remove it (amdgpu-uninstall) first."
fi
have_dxg=$(pkg_version rocdxg-roct)
if [ -n "$have_dxg" ] && [ "$have_dxg" != "$ROCDXG_VERSION" ]; then
  die "rocdxg-roct $have_dxg is installed; expected $ROCDXG_VERSION"
fi

export DEBIAN_FRONTEND=noninteractive
# A WSL restart while apt runs (for example the Windows driver install below
# the distro) leaves dpkg half-configured. Finishing it is always safe.
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
have_ai=$(pkg_version amdgpu-install)
if [ -n "$have_ai" ] && [ "$have_ai" != "$AMDGPU_INSTALL_PKG_VERSION" ]; then
  die "amdgpu-install $have_ai is installed; expected $AMDGPU_INSTALL_PKG_VERSION (the $AMDGPU_INSTALL_VERSION release)"
fi
if [ -z "$have_ai" ]; then
  fetch_verified "$AMDGPU_INSTALL_URL" "$AMDGPU_INSTALL_SHA256" "$WORK/amdgpu-install.deb"
  apt-get install -y -q "$WORK/amdgpu-install.deb"
fi
if [ -z "$have_rocm" ] || ! command -v /opt/rocm/bin/hipcc >/dev/null; then
  log "installing ROCm $ROCM_VERSION userspace (no dkms)"
  amdgpu-install -y --usecase=rocm,hiplibsdk --no-dkms
fi

# librocdxg -------------------------------------------------------------------
if [ -z "$have_dxg" ]; then
  fetch_verified "$ROCDXG_URL" "$ROCDXG_SHA256" "$WORK/rocdxg-roct.deb"
  apt-get install -y -q "$WORK/rocdxg-roct.deb"
fi
ldconfig

# Groups and environment -------------------------------------------------------
usermod -aG render,video "$TARGET_USER"
# Single-quoted on purpose: PATH expands at login, not now.
# shellcheck disable=SC2016
profile_body='# Written by scripts/strix-hip/wsl-setup.sh: ROCr finds the GPU through DXG.
export HSA_ENABLE_DXG_DETECTION=1
case ":$PATH:" in *:/opt/rocm/bin:*) ;; *) export PATH=/opt/rocm/bin:$PATH ;; esac'
if [ "$(cat "$PROFILE_FILE" 2>/dev/null)" != "$profile_body" ]; then
  printf '%s\n' "$profile_body" > "$PROFILE_FILE"
  chmod 0644 "$PROFILE_FILE"
fi

# Verification (fails loudly on any drift) ------------------------------------
got_rocm=$(rocm_installed)
[ "$got_rocm" = "$ROCM_VERSION" ] || die "ROCm after install is '${got_rocm:-absent}', expected $ROCM_VERSION"
got_dxg=$(pkg_version rocdxg-roct)
[ "$got_dxg" = "$ROCDXG_VERSION" ] || die "rocdxg-roct after install is '${got_dxg:-absent}', expected $ROCDXG_VERSION"
[ -x /opt/rocm/bin/hipcc ] || die "/opt/rocm/bin/hipcc missing after install"
/opt/rocm/bin/hipcc --version | grep -q "roc-$ROCM_VERSION" \
  || die "hipcc does not report roc-$ROCM_VERSION: $(/opt/rocm/bin/hipcc --version | head -n2 | tr '\n' ' ')"
ls /opt/rocm/lib/librocdxg.so* /usr/lib/librocdxg.so* /usr/local/lib/librocdxg.so* >/dev/null 2>&1 \
  || ldconfig -p | grep -q librocdxg || die "librocdxg shared library not found after install"

log "OK: ROCm $got_rocm, rocdxg-roct $got_dxg, user $TARGET_USER in render,video"
log "next: from Windows run 'wsl --shutdown', reopen, then run scripts/strix-hip/preflight.sh as $TARGET_USER in a login shell"
