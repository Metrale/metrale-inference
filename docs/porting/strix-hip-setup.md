# Strix Halo native-HIP host setup (`strix-hip`)

2026-10-07: How to build a host for the `kernels/strix-hip` target (AMD
gfx1151, Radeon 8060S) and how to prove it matches the validated
configuration. The validated box is a 128 GB Strix Halo machine that dual
boots Ubuntu 24.04 and Windows 11; both host paths below were measured on it.
The repository's HIP shims stand in for the CUDA libraries on either path.

This is **not** the SCALE `strix` target. That target has its own toolchain
and its own guide, [amd-strix-halo-scale.md](amd-strix-halo-scale.md). Nothing
here installs or uses SCALE.

What this guide certifies: the host stack (versions below), that ROCr and HIP
see one gfx1151 device, and the build environment for
`METRALE_TARGET_HW=strix-hip`. It does not certify any model or performance
number. The shim's device reporting contract is
[hip-device-reporting.md](hip-device-reporting.md).

## Choosing a host path

| Path | Host | GPU reached through | Role |
| --- | --- | --- | --- |
| Native Linux | Ubuntu 24.04 on bare metal, ROCm userspace | the in-box `amdgpu` kernel module and `/dev/kfd` | primary development path; recommended |
| Windows + WSL | Windows 11, WSL Ubuntu 24.04, ROCm userspace, librocdxg | the Windows display driver through DXG | the option for running on a Windows install |

**Use native Linux when memory matters**, which for model work is almost
always. Measured on the validated box (128 GB of RAM, the same physical
memory backs every column, so the columns do not add up):

| Setup | BIOS UMA Frame Buffer | GPU pool (rocminfo) | HIP `totalGlobalMem` | RAM Linux sees (`free` total) |
| --- | --- | --- | --- | --- |
| Native Linux, GTT 124 GiB | 1G | 130023424 KB (124 GiB) | 133143986176 (124.0 GiB) | 124.4 GiB |
| Windows + WSL | 1G | 67268878 KB (~64 GiB) | 68883331072 (64.15 GiB) | 117 GiB |
| Windows + WSL | 64G (BIOS default) | 100298866 KB (~95.6 GiB) | not measured | 31 GiB |

Natively the GPU maps system memory as GTT, and the kernel command line set
by `linux-setup.sh` sizes GTT at 124 GiB, so the GPU can address nearly all of
RAM while Linux still sees all of it. Under Windows the pool is capped by
Windows itself (see [Memory split under Windows](#memory-split-under-windows)),
and the WSL VM is a second claim on the same RAM.

Tooling lives in `scripts/strix-hip/`:

| File | Runs where | Purpose |
| --- | --- | --- |
| `linux-setup.sh` | native Ubuntu, as root | Installs the pinned ROCm userspace, build deps, groups and `/etc/profile.d/rocm-native.sh`; sets the GTT kernel parameters in `/etc/default/grub` and runs `update-grub`. Idempotent; refuses a different installed version; never reboots. |
| `wsl-setup.sh` | WSL, as root | Installs the pinned ROCm userspace, librocdxg, build deps, groups and `/etc/profile.d/rocm-wsl.sh`. Idempotent; refuses a different installed version. |
| `preflight.sh` | either, as the build user | Read-only fingerprint as JSON; detects native or WSL. Exits nonzero when the host is not usable. |
| `pins_test.py` | any host with python3 | Fails when the version table below, the two setup scripts and the preflight samples disagree. |

## Validated versions

Every row was read off the validated box (`preflight.sh` output, samples at
the end). Rows without a prefix apply to both paths; `Native:` and `WSL:`
rows to one. Rows marked "pinned" are enforced by the named script; change
both or `pins_test.py` fails.

| Component | Validated version | Enforced |
| --- | --- | --- |
| BIOS UMA Frame Buffer Size | 1G | no |
| Ubuntu | 24.04 | pinned (both scripts) |
| ROCm userspace | 7.2.1 | pinned (both scripts) |
| amdgpu-install package file | amdgpu-install_7.2.1.70201-1_all.deb | pinned (both scripts) |
| amdgpu-install dpkg version | 30.30.1.0.30300100-2303411.24.04 | pinned (both scripts) |
| amdgpu-install sha256 | 4c0338a241c15b12c14eb3aeb4012ea0d55dba681737ea8482248041a16c2afa | pinned (both scripts) |
| hsa-rocr | 1.18.0.70201-81~24.04 | no |
| HIP | 7.2.53211-e1a6bc5663 | no |
| hipcc | AMD clang 22.0.0git roc-7.2.1 26084 f58b06dce1f9 | checked (`roc-7.2.1`) |
| gcc | 13.3.0 | no |
| rustc | 1.93.1 | pinned (`rust-toolchain.toml`) |
| Native: Ubuntu point release | 24.04.4 LTS | no |
| Native: kernel | 7.0.0-38-generic (Ubuntu 24.04 HWE) | no |
| Native: amd_iommu | off | pinned (`linux-setup.sh`) |
| Native: amdgpu.gttsize | 126976 | pinned (`linux-setup.sh`) |
| Native: ttm.pages_limit | 32505856 | pinned (`linux-setup.sh`) |
| Native: ttm.page_pool_size | 32505856 | pinned (`linux-setup.sh`) |
| WSL: Windows | Windows 11 Pro, build 26200 | no |
| WSL: AMD Adrenalin | 26.9.2 (minimum 26.2.2) | no |
| WSL: Windows display driver | 32.0.32015.2008 | no |
| WSL: kernel | 6.18.40.1-microsoft-standard-WSL2 | no |
| WSL: librocdxg (rocdxg-roct) | 1.2.2 | pinned (`wsl-setup.sh`) |
| WSL: librocdxg sha256 | 28ded1254811192ebace1f76c0227580184af7b27ab2475fb9728295a702d541 | pinned (`wsl-setup.sh`) |

The librocdxg sha256 matches the digest GitHub publishes for the v1.2.2
release asset. The amdgpu-install `.deb` is named for the ROCm release but its
dpkg `Version` field is the driver bundle number; both are recorded.

## Native Linux

### 1. Firmware and disk

1. **BIOS: UMA Frame Buffer Size = 1G.** On the validated AMI Aptio board it
   is under Advanced > GFX Configuration. Natively the carve-out shows up as
   1 GiB of VRAM and everything else the GPU uses is GTT. A larger carve-out
   was not measured natively; it would reserve RAM Linux cannot use.
2. **Dual boot with an existing Windows install (optional).** The Ubuntu 24.04
   installer refuses a disk that holds a BitLocker volume. From an elevated
   Windows prompt, decrypt fully first and wait for it to finish:

   ```bat
   manage-bde -off C:
   manage-bde -status C:   :: repeat until "Fully Decrypted"
   ```

   Then install Ubuntu alongside Windows. If the firmware keeps booting
   Windows first, put the Ubuntu entry first from an elevated Windows prompt:

   ```bat
   bcdedit /enum firmware            :: note the identifier of the "ubuntu" entry
   bcdedit /set {fwbootmgr} displayorder {<ubuntu id>} /addfirst
   ```

   Turning BitLocker back on after the install was not tried.

### 2. Provisioning

Install Ubuntu 24.04 (the validated box runs the 24.04.4 HWE kernel,
7.0.0-38-generic), then run as root, naming the build user:

```bash
sudo bash scripts/strix-hip/linux-setup.sh <linux-user>
```

It installs ROCm through `amdgpu-install --usecase=rocm,hiplibsdk --no-dkms`
(the kernel's own `amdgpu` module drives the GPU; an installed `amdgpu-dkms`
stops the script), adds the user to `render` and `video`, writes
`/etc/profile.d/rocm-native.sh` (puts `/opt/rocm/bin` on `PATH`, nothing
else), and makes `GRUB_CMDLINE_LINUX_DEFAULT` in `/etc/default/grub` carry:

```text
amd_iommu=off amdgpu.gttsize=126976 ttm.pages_limit=32505856 ttm.page_pool_size=32505856
```

`amdgpu.gttsize` is in MiB (126976 MiB = 124 GiB); the two `ttm` limits are
in 4 KiB pages (126976 x 256 = 32505856). The `ttm` limits raise the
kernel's TTM cap, which otherwise defaults to half of RAM. Other tokens on the line (`quiet splash`) keep their order; an existing
value for any of these four keys is replaced. The previous file is kept as
`/etc/default/grub.strix-hip.<timestamp>` and `update-grub` runs only when
the line changed. The pins are for 128 GB of RAM: with less `MemTotal` than
the GTT size the script refuses.

No `HSA_ENABLE_DXG_DETECTION` is needed natively. The script does **not**
reboot; its last line says whether the running kernel already has the
parameters. Reboot when it says so, and log in again for the group change.

The validated box was provisioned by hand with the same `amdgpu-install`
command and grub line before the script existed. `linux-setup.sh` was then
checked read-only against that box: its grub merge leaves the box's line
unchanged and it refuses to run without root. TODO: run it end to end on a
fresh Ubuntu install and record the transcript here.

Install Rust as the build user with rustup; `rust-toolchain.toml` selects
1.93.1 on first `cargo` use.

### 3. Preflight

```bash
bash -l scripts/strix-hip/preflight.sh > preflight.json
```

Natively the script needs `/dev/kfd` readable and writable by the build user
(exit 2 otherwise) and a gfx1151 agent or HIP device (exit 3 otherwise). It
adds a `native` block: whether `/dev/kfd` is usable, the four kernel
parameters as the running kernel got them, and the amdgpu GTT and VRAM totals
from sysfs. A parameter that differs from the `linux-setup.sh` pins is listed
under `warnings` without failing the run, so a box with another RAM size can
still be fingerprinted. Sample at [the end](#sample-preflight-output-native-linux).

## Windows + WSL

### 1. Windows prerequisites

1. **Windows 11** with WSL2 available.
2. **AMD Adrenalin 26.2.2 or newer** (validated: 26.9.2, display driver
   `32.0.32015.2008`). 26.2.2 is the oldest driver this path is documented
   for; nothing older was tried.
3. **BIOS: UMA Frame Buffer Size = 1G.** See "Memory split under Windows"
   below for why.
4. **`%USERPROFILE%\.wslconfig`:**

   ```ini
   [wsl2]
   memory=120GB
   swap=0

   [experimental]
   autoMemoryReclaim=gradual
   ```

5. **Install the distro:** `wsl --install -d Ubuntu-24.04`, create the build
   user when prompted.
6. **Apply the config:** `wsl --shutdown`, then reopen the distro. WSL only
   reads `.wslconfig` on a cold start.

### Memory split under Windows

The GPU's memory pool is the BIOS carve-out plus the shared GPU memory
Windows grants, and Windows caps shared GPU memory at half of the RAM it can
see. Measured on the validated 128 GB box:

| UMA Frame Buffer Size | Windows sees | WSL `free` total | rocminfo GPU pool |
| --- | --- | --- | --- |
| 1G (validated) | 136.0 GB | 117 GiB | 67268878 KB (~64 GiB) |
| 64G (default) | 68.3 GB | 31 GiB | 100298866 KB (~95.6 GiB) |

Both pools agree, to within about 0.2 GiB, with carve-out + half of what
Windows sees (1 GiB + 68.0 GB, and 64 GiB + 34.2 GB). The 64G default gives
the GPU more memory but leaves WSL 31 GiB, too little for a release build
next to a checkpoint load. The 1G setting trades about 31 GiB of GPU pool for
a usable WSL; HIP then reports `totalGlobalMem` = 68883331072 bytes
(64.15 GiB), matching rocminfo. Neither comes close to the 124 GiB the native
path gives on the same box.

### 2. WSL provisioning

Copy the script somewhere WSL can read it and run it as root, naming the
build user:

```bash
wsl -d Ubuntu-24.04 -u root -e bash /mnt/c/Users/<you>/wsl-setup.sh <linux-user>
```

It installs ROCm through `amdgpu-install --usecase=rocm,hiplibsdk --no-dkms`
(no kernel module: the GPU is reached through the Windows driver), installs
`rocdxg-roct_1.2.2_amd64.deb` after checking its sha256, adds the user to
`render` and `video`, and writes `/etc/profile.d/rocm-wsl.sh`:

```bash
export HSA_ENABLE_DXG_DETECTION=1
export PATH=/opt/rocm/bin:$PATH   # guarded against duplicates
```

Without `HSA_ENABLE_DXG_DETECTION=1` ROCr finds no GPU agent. A rerun is safe;
an installed ROCm or librocdxg at any other version stops the script with an
error instead of being replaced. Run `wsl --shutdown` afterwards so the group
change applies.

Install Rust as the build user with rustup; `rust-toolchain.toml` selects
1.93.1 on first `cargo` use.

### 3. Preflight

```bash
bash -l scripts/strix-hip/preflight.sh > preflight.json
```

Use a login shell (`bash -l`). `wsl -e bash script` does not read
`/etc/profile.d`, so the DXG variable is missing and preflight fails with exit
2, which is the intended signal. Exit 3 means no gfx1151 agent or HIP device.
Under WSL the JSON also carries the Windows driver and visible memory
(through WSL interop, when enabled).

## Build environment

Both paths, once preflight passes:

```bash
export CUDARC_CUDA_VERSION=13000
export METRALE_NO_RDMA=1
export METRALE_TARGET_HW=strix-hip
export METRALE_TARGET_MODEL=<model>     # a directory under kernels/strix-hip/
export METRALE_TARGET_QUANT=<quant>     # its quant leaf, e.g. nvfp4
export METRALE_HIPCC=/opt/rocm/bin/hipcc
cargo build --release -p metrale-server --no-default-features --features cuda --bin met
```

Why each one, with the code that reads it:

- `CUDARC_CUDA_VERSION=13000`: `vendor/cudarc/build.rs` picks its bindings
  from this when no CUDA toolkit is installed, which is the case here.
- `METRALE_NO_RDMA=1`: `crates/gpu-sys/build.rs` skips the verbs shim, so no
  libibverbs is needed.
- `METRALE_TARGET_HW=strix-hip`: `crates/kernels/build.rs` compiles
  `kernels/strix-hip` with hipcc and builds the HIP shims (`libcuda.so`,
  `libcudart.so`, `libcublasLt.so`) into the `metrale-kernels` OUT_DIR;
  `crates/storage/build.rs` emits its empty registry for this target.
- `METRALE_HIPCC`: optional, `/opt/rocm/bin/hipcc` is the default; set it so
  the build log records which compiler ran.
- `--no-default-features --features cuda`: `crates/server/Cargo.toml`
  defaults to `cuda` + `nccl`, and there is no NCCL on this host.

At run time the shims must shadow any other `libcuda.so`. Point the loader at
the OUT_DIR the binary was linked against:

```bash
O=target/release/build/$(tr ' ' '\n' < target/release/met.d \
  | grep -oE 'metrale-kernels-[0-9a-f]+/out' | sort -u | head -1)
export LD_LIBRARY_PATH=$O:/opt/rocm/lib
```

This environment produced a working release `met` under WSL for a model
target that is not on `main` yet; the targets on `main` use the same
variables. TODO: record a native Linux build with this environment here.

## Recovery and known limits

Both paths:

- **Device counts differ by API.** rocminfo reports 40 compute units; HIP's
  `multiProcessorCount` is 20 (workgroup processors). Kernel launch sizing sees
  the HIP number through the shim.
- **Size budgets from the pool, not from `free`.** The shim's
  `cuMemGetInfo_v2` returns the same `hipMemGetInfo` numbers preflight
  reports.
- **The two paths report the device differently.** Natively HIP reports
  `integrated` = 1 and the name `AMD Radeon Graphics`; under WSL it reports
  `integrated` = 0 and `AMD Radeon(TM) 8060S Graphics`. Code that branches on
  either property sees two different devices.

Native Linux:

- **GTT parameters not applied.** If preflight warns about a kernel parameter,
  check `cat /proc/cmdline`; a `GRUB_CMDLINE_LINUX_DEFAULT` set in
  `/etc/default/grub.d/*.cfg` overrides `/etc/default/grub` (the setup script
  refuses to run in that case), and the change needs a reboot.
- **Windows boots first again.** Rerun the `bcdedit` command above from
  Windows to put the Ubuntu entry back in front.

Windows + WSL:

- **apt killed by a WSL restart.** Installing or updating the Windows driver
  restarts WSL and kills any apt run inside it. Rerun `wsl-setup.sh`; it starts
  with `dpkg --configure -a` and `apt-get -f install`.
- **Adrenalin installer hangs over ssh.** The GUI installer never finishes in
  a non-interactive session. Extract the package and install the display
  driver alone from an elevated prompt:
  `pnputil /add-driver <extracted>\Packages\Drivers\Display\WT6A_INF\*.inf /subdirs /install`,
  then reboot.
- **Background jobs get reaped.** A process started with `nohup` or `&` from an
  `ssh` → `wsl` command dies when that `wsl.exe` session exits. Start long jobs
  with `systemd-run --user --unit=<name> --collect bash -c '...'` and poll the
  unit.
- **No managed memory.** `hipMallocManaged` is not supported on this WSL path;
  anything that relies on `cuMemAllocManaged` through the shim fails. TODO:
  check managed memory on the native path.
- **Lazy allocation failure.** An allocation beyond the remaining GPU pool may
  not fail at `hipMalloc` but later, on use.

## Sample preflight output: native Linux

Validated box, native Ubuntu, 2026-10-07 (`bash -l preflight.sh`, exit 0).
The box was provisioned by hand before `linux-setup.sh` existed, so
`profile_d_rocm_native` is false; `PATH` came from a hand-written
`/etc/profile.d/rocm.sh`.

```json
{
  "schema": "metrale-strix-hip-preflight/2",
  "host": "native",
  "ok": true,
  "failures": [],
  "warnings": [],
  "os": {
    "pretty_name": "Ubuntu 24.04.4 LTS",
    "version_id": "24.04"
  },
  "kernel": "7.0.0-38-generic",
  "wsl": false,
  "env": {
    "HSA_ENABLE_DXG_DETECTION": null,
    "profile_d_rocm_wsl": false,
    "profile_d_rocm_native": false
  },
  "native": {
    "kfd_rw": true,
    "cmdline": {
      "amd_iommu": "off",
      "amdgpu.gttsize": "126976",
      "ttm.pages_limit": "32505856",
      "ttm.page_pool_size": "32505856"
    },
    "amdgpu": {
      "pci_device": "0x1586",
      "gtt_total": 133143986176,
      "vram_total": 1073741824,
      "vis_vram_total": 1073741824,
      "gtt_total_gib": 124.0,
      "vram_total_gib": 1.0
    }
  },
  "rocm": {
    "version": "7.2.1",
    "rocdxg_roct": null,
    "hsa_rocr": "1.18.0.70201-81~24.04",
    "hip_runtime_amd": "7.2.53211.70201-81~24.04",
    "amdgpu_install": "30.30.1.0.30300100-2303411.24.04",
    "hip_version": "7.2.53211-e1a6bc5663",
    "hipcc_clang": "AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.1 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)"
  },
  "rocminfo": {
    "gfx_name": "gfx1151",
    "marketing_name": "AMD Radeon Graphics",
    "compute_units": 40,
    "pool_size_kb": 130023424,
    "pool_size_gib": 124.0,
    "exit": 0
  },
  "memory": {
    "total_bytes": 133620641792,
    "available_bytes": 126705893376,
    "swap_total_bytes": 8589930496,
    "total_gib": 124.44
  },
  "hip_probe": {
    "build_exit": 0,
    "run_exit": 0,
    "runtime_version": "70253211",
    "driver_version": "70253211",
    "error": null,
    "devices": [
      {
        "name": "AMD Radeon Graphics",
        "gcn_arch": "gfx1151",
        "multiprocessor_count": 20,
        "warp_size": 32,
        "integrated": 1,
        "total_global_mem": 133143986176,
        "mem_get_info_free": 129913769984,
        "mem_get_info_total": 133143986176
      }
    ]
  },
  "windows": null
}
```

## Sample preflight output: Windows + WSL

Validated box, Windows + WSL, 2026-10-07 (`bash -l preflight.sh`, exit 0).
Captured with the first version of the script (schema 1), before the `host`,
`warnings` and `native` fields existed.

```json
{
  "schema": "metrale-strix-hip-preflight/1",
  "ok": true,
  "failures": [],
  "os": { "pretty_name": "Ubuntu 24.04.5 LTS", "version_id": "24.04" },
  "kernel": "6.18.40.1-microsoft-standard-WSL2",
  "wsl": true,
  "env": { "HSA_ENABLE_DXG_DETECTION": "1", "profile_d_rocm_wsl": true },
  "rocm": {
    "version": "7.2.1",
    "rocdxg_roct": "1.2.2",
    "hsa_rocr": "1.18.0.70201-81~24.04",
    "hip_runtime_amd": "7.2.53211.70201-81~24.04",
    "amdgpu_install": "30.30.1.0.30300100-2303411.24.04",
    "hip_version": "7.2.53211-e1a6bc5663",
    "hipcc_clang": "AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.1 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)"
  },
  "rocminfo": {
    "gfx_name": "gfx1151",
    "marketing_name": "AMD Radeon(TM) 8060S Graphics",
    "compute_units": 40,
    "pool_size_kb": 67268878,
    "pool_size_gib": 64.15,
    "exit": 0
  },
  "memory": {
    "total_bytes": 126613254144,
    "available_bytes": 125222469632,
    "swap_total_bytes": 0,
    "total_gib": 117.92
  },
  "hip_probe": {
    "build_exit": 0,
    "run_exit": 0,
    "runtime_version": "70253211",
    "driver_version": "70253211",
    "error": null,
    "devices": [
      {
        "name": "AMD Radeon(TM) 8060S Graphics",
        "gcn_arch": "gfx1151",
        "multiprocessor_count": 20,
        "warp_size": 32,
        "integrated": 0,
        "total_global_mem": 68883331072,
        "mem_get_info_free": 68325562368,
        "mem_get_info_total": 68883331072
      }
    ]
  },
  "windows": {
    "gpu_name": "AMD Radeon(TM) 8060S Graphics",
    "driver_version": "32.0.32015.2008",
    "os_caption": "Microsoft Windows 11 Pro",
    "os_build": "26200",
    "visible_memory_kb": "132799300"
  }
}
```

The same script run without `HSA_ENABLE_DXG_DETECTION` returns exit 2 with
`"ok": false` and both failures listed (no DXG variable, no gfx1151 device);
that run is the known-bad control for the exit-status check. A host that is
neither WSL nor has `/dev/kfd` (any machine without the GPU) returns exit 2
with `"host": "unknown"`.

TODO: rerun this guide on a second Strix Halo box and record its fingerprint
next to this one before calling the configuration portable across boards.
