#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# 2026-10-07: Read-only fingerprint of a native-HIP Strix Halo host
# (`METRALE_TARGET_HW=strix-hip`, WSL Ubuntu 24.04). Not the SCALE `strix`
# target. Prints one JSON object on stdout and changes nothing on the box
# except a temporary directory for the device probe, removed on exit.
#
# Usage, as the build user in a LOGIN shell (so /etc/profile.d/rocm-wsl.sh
# applies; `wsl -e bash script` alone skips it):
#   bash -l scripts/strix-hip/preflight.sh > preflight.json
#   wsl -d Ubuntu-24.04 -u <you> -e bash -l /mnt/c/Users/<you>/preflight.sh
#
# Exit status: 0 when the host can run strix-hip builds, 2 when
# HSA_ENABLE_DXG_DETECTION is not 1, 3 when no gfx1151 agent/device is found
# (2 wins when both fail). The JSON is printed in every case; `failures`
# names each reason.
#
# The device probe compiles a small HIP program with the installed hipcc, so it
# reports what the HIP runtime sees, independent of the CUDA shim
# (docs/porting/hip-device-reporting.md covers the shim's mapping).
#
# TODO: compare the collected versions against the pins in wsl-setup.sh and
# report drift as a warning field.
set -uo pipefail

export LC_ALL=C
ROCM=/opt/rocm
WORK=$(mktemp -d "${TMPDIR:-/tmp}/strix-hip-preflight.XXXXXX")
trap 'rm -rf "$WORK"' EXIT

# Run a command with a time limit, capturing stdout+stderr into $WORK/<name>.
capture() { # name seconds cmd...
  local name=$1 secs=$2; shift 2
  timeout "$secs" "$@" > "$WORK/$name" 2>&1
  echo $? > "$WORK/$name.rc"
}

uname -r > "$WORK/uname" 2>&1
cat /etc/os-release > "$WORK/os-release" 2>/dev/null
cat "$ROCM/.info/version" > "$WORK/rocm-version" 2>/dev/null
dpkg-query -W -f='${Package} ${Version}\n' rocdxg-roct hsa-rocr hip-runtime-amd amdgpu-install > "$WORK/dpkg" 2>/dev/null
free -b > "$WORK/free" 2>&1
capture hipcc 30 "$ROCM/bin/hipcc" --version
capture rocminfo 60 "$ROCM/bin/rocminfo"

# Windows-side facts through WSL interop, when interop is enabled. Read-only.
if command -v powershell.exe >/dev/null 2>&1; then
  capture windows 30 powershell.exe -NoProfile -NonInteractive -Command \
    "\$g = Get-CimInstance Win32_VideoController | Where-Object Name -like '*Radeon*' | Select-Object -First 1; \$o = Get-CimInstance Win32_OperatingSystem; 'gpu_name=' + \$g.Name; 'driver_version=' + \$g.DriverVersion; 'os_caption=' + \$o.Caption; 'os_build=' + \$o.BuildNumber; 'visible_memory_kb=' + \$o.TotalVisibleMemorySize"
fi

# Device probe: what hipGetDeviceProperties / hipMemGetInfo report.
cat > "$WORK/probe.hip" <<'EOF'
// SPDX-License-Identifier: MIT OR Apache-2.0
#include <hip/hip_runtime.h>
#include <cstdio>
int main() {
  int n = 0;
  hipError_t e = hipGetDeviceCount(&n);
  if (e != hipSuccess) { std::printf("error=%s\n", hipGetErrorName(e)); return 1; }
  int rt = 0, drv = 0;
  hipRuntimeGetVersion(&rt);
  hipDriverGetVersion(&drv);
  std::printf("device_count=%d\nruntime_version=%d\ndriver_version=%d\n", n, rt, drv);
  for (int d = 0; d < n; ++d) {
    hipDeviceProp_t p;
    if (hipGetDeviceProperties(&p, d) != hipSuccess) continue;
    size_t free_b = 0, total_b = 0;
    hipSetDevice(d);
    hipMemGetInfo(&free_b, &total_b);
    std::printf("dev%d.name=%s\n", d, p.name);
    std::printf("dev%d.gcn_arch=%s\n", d, p.gcnArchName);
    std::printf("dev%d.multiprocessor_count=%d\n", d, p.multiProcessorCount);
    std::printf("dev%d.warp_size=%d\n", d, p.warpSize);
    std::printf("dev%d.integrated=%d\n", d, p.integrated);
    std::printf("dev%d.total_global_mem=%zu\n", d, p.totalGlobalMem);
    std::printf("dev%d.mem_get_info_free=%zu\n", d, free_b);
    std::printf("dev%d.mem_get_info_total=%zu\n", d, total_b);
  }
  return 0;
}
EOF
if [ -x "$ROCM/bin/hipcc" ]; then
  capture probe_build 180 "$ROCM/bin/hipcc" -O1 -o "$WORK/probe" "$WORK/probe.hip"
  if [ "$(cat "$WORK/probe_build.rc")" = 0 ]; then
    capture probe_run 60 "$WORK/probe"
  fi
fi

HSA_DXG="${HSA_ENABLE_DXG_DETECTION:-}" PROFILE_FILE_PRESENT=$([ -f /etc/profile.d/rocm-wsl.sh ] && echo 1 || echo 0) \
  python3 -I - "$WORK" <<'PY'
import json, os, re, sys

work = sys.argv[1]

def read(name):
    try:
        with open(os.path.join(work, name), encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None

def rc(name):
    v = read(name + ".rc")
    return int(v) if v and v.strip().lstrip("-").isdigit() else None

def kv(text):
    out = {}
    for line in (text or "").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out

failures = []

osr = kv(read("os-release"))
dpkg = {}
for line in (read("dpkg") or "").splitlines():
    parts = line.split()
    if len(parts) == 2:
        dpkg[parts[0]] = parts[1]

hipcc_text = read("hipcc") or ""
m_hip = re.search(r"HIP version:\s*(\S+)", hipcc_text)
m_clang = re.search(r"(AMD clang version .*)", hipcc_text)

# rocminfo: the GPU agent block, its gfx name and its first pool size.
ri = read("rocminfo") or ""
gpu = None
for block in re.split(r"\n\*{5,}\s*\n", ri):
    if re.search(r"Device Type:\s*GPU", block):
        name = re.search(r"^\s*Name:\s*(gfx\w+)", block, re.M)
        market = re.search(r"Marketing Name:\s*(.+?)\s*$", block, re.M)
        cu = re.search(r"Compute Unit:\s*(\d+)", block)
        pool = None
        pool_sec = block.split("Pool Info:", 1)
        if len(pool_sec) == 2:
            ps = re.search(r"Size:\s*(\d+)\(0x[0-9a-fA-F]+\)\s*KB", pool_sec[1])
            pool = int(ps.group(1)) if ps else None
        gpu = {
            "gfx_name": name.group(1) if name else None,
            "marketing_name": market.group(1) if market else None,
            "compute_units": int(cu.group(1)) if cu else None,
            "pool_size_kb": pool,
            "pool_size_gib": round(pool / 1048576, 2) if pool else None,
        }
        break

mem = {}
for line in (read("free") or "").splitlines():
    p = line.split()
    if p and p[0] == "Mem:" and len(p) >= 7:
        mem.update(total_bytes=int(p[1]), available_bytes=int(p[6]))
    elif p and p[0] == "Swap:" and len(p) >= 2:
        mem["swap_total_bytes"] = int(p[1])
if "total_bytes" in mem:
    mem["total_gib"] = round(mem["total_bytes"] / 2**30, 2)

probe = kv(read("probe_run"))
devices = []
for d in range(int(probe.get("device_count", "0") or 0)):
    pre = f"dev{d}."
    dev = {k[len(pre):]: v for k, v in probe.items() if k.startswith(pre)}
    for k in ("multiprocessor_count", "warp_size", "integrated", "total_global_mem",
              "mem_get_info_free", "mem_get_info_total"):
        if k in dev and dev[k].lstrip("-").isdigit():
            dev[k] = int(dev[k])
    devices.append(dev)

win = kv(read("windows")) if read("windows") is not None else None

hsa = os.environ.get("HSA_DXG", "")
if hsa != "1":
    failures.append("HSA_ENABLE_DXG_DETECTION is not 1 (run in a login shell or source /etc/profile.d/rocm-wsl.sh)")
gfx_seen = (gpu or {}).get("gfx_name") == "gfx1151" or any(
    str(d.get("gcn_arch", "")).startswith("gfx1151") for d in devices)
if not gfx_seen:
    failures.append("no gfx1151 agent in rocminfo or HIP device list")

doc = {
    "schema": "metrale-strix-hip-preflight/1",
    "ok": not failures,
    "failures": failures,
    "os": {"pretty_name": osr.get("PRETTY_NAME", "").strip('"') or None,
           "version_id": osr.get("VERSION_ID", "").strip('"') or None},
    "kernel": (read("uname") or "").strip() or None,
    "wsl": "microsoft" in (read("uname") or "").lower(),
    "env": {"HSA_ENABLE_DXG_DETECTION": hsa or None,
            "profile_d_rocm_wsl": os.environ.get("PROFILE_FILE_PRESENT") == "1"},
    "rocm": {"version": (read("rocm-version") or "").strip() or None,
             "rocdxg_roct": dpkg.get("rocdxg-roct"),
             "hsa_rocr": dpkg.get("hsa-rocr"),
             "hip_runtime_amd": dpkg.get("hip-runtime-amd"),
             "amdgpu_install": dpkg.get("amdgpu-install"),
             "hip_version": m_hip.group(1) if m_hip else None,
             "hipcc_clang": m_clang.group(1).strip() if m_clang else None},
    "rocminfo": dict(gpu or {}, exit=rc("rocminfo")),
    "memory": mem,
    "hip_probe": {"build_exit": rc("probe_build"), "run_exit": rc("probe_run"),
                  "runtime_version": probe.get("runtime_version"),
                  "driver_version": probe.get("driver_version"),
                  "error": probe.get("error"), "devices": devices},
    "windows": win,
}
json.dump(doc, sys.stdout, indent=2)
sys.stdout.write("\n")
code = 0
if hsa != "1":
    code = 2
elif not gfx_seen:
    code = 3
sys.exit(code)
PY
