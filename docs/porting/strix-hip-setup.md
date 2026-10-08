# Strix Halo native-HIP host setup (`strix-hip`)

2026-10-07: How to build a host for the `kernels/strix-hip` target (AMD
gfx1151, Radeon 8060S) from a clean Windows 11 machine, and how to prove it
matches the validated configuration. The path is Windows 11 + WSL Ubuntu 24.04
+ ROCm userspace + librocdxg, with the repository's HIP shims standing in for
the CUDA libraries.

This is **not** the SCALE `strix` target. That target has its own toolchain
and its own guide, [amd-strix-halo-scale.md](amd-strix-halo-scale.md). Nothing
here installs or uses SCALE.

What this guide certifies: the host stack (versions below), that ROCr and HIP
see one gfx1151 device through DXG, and the build environment for
`METRALE_TARGET_HW=strix-hip`. It does not certify any model or performance
number. The shim's device reporting contract is
[hip-device-reporting.md](hip-device-reporting.md).

Tooling lives in `scripts/strix-hip/`:

| File | Runs where | Purpose |
| --- | --- | --- |
| `wsl-setup.sh` | WSL, as root | Installs the pinned ROCm userspace, librocdxg, build deps, groups and `/etc/profile.d/rocm-wsl.sh`. Idempotent; refuses a different installed version. |
| `preflight.sh` | WSL, as the build user | Read-only fingerprint as JSON. Exits nonzero when the host is not usable. |
| `pins_test.py` | Any host with python3 | Fails when the version table below and the `wsl-setup.sh` pins disagree. |

## Validated versions

Every row was read off the validated host (`preflight.sh` output, sample at
the end). Rows marked "pinned" are enforced by `wsl-setup.sh`; change both or
`pins_test.py` fails.

| Component | Validated version | Enforced |
| --- | --- | --- |
| Windows | Windows 11 Pro, build 26200 | no |
| AMD Adrenalin | 26.9.2 (minimum 26.2.2) | no |
| Windows display driver | 32.0.32015.2008 | no |
| BIOS UMA Frame Buffer Size | 1G | no |
| WSL kernel | 6.18.40.1-microsoft-standard-WSL2 | no |
| Ubuntu | 24.04 | pinned |
| ROCm userspace | 7.2.1 | pinned |
| amdgpu-install package file | amdgpu-install_7.2.1.70201-1_all.deb | pinned |
| amdgpu-install dpkg version | 30.30.1.0.30300100-2303411.24.04 | pinned |
| amdgpu-install sha256 | 4c0338a241c15b12c14eb3aeb4012ea0d55dba681737ea8482248041a16c2afa | pinned |
| librocdxg (rocdxg-roct) | 1.2.2 | pinned |
| librocdxg sha256 | 28ded1254811192ebace1f76c0227580184af7b27ab2475fb9728295a702d541 | pinned |
| hsa-rocr | 1.18.0.70201-81~24.04 | no |
| HIP | 7.2.53211-e1a6bc5663 | no |
| hipcc | AMD clang 22.0.0git roc-7.2.1 26084 f58b06dce1f9 | checked (`roc-7.2.1`) |
| gcc | 13.3.0 | no |
| rustc | 1.93.1 | pinned (`rust-toolchain.toml`) |

The librocdxg sha256 matches the digest GitHub publishes for the v1.2.2
release asset. The amdgpu-install `.deb` is named for the ROCm release but its
dpkg `Version` field is the driver bundle number; both are recorded.

## 1. Windows prerequisites

1. **Windows 11** with WSL2 available.
2. **AMD Adrenalin 26.2.2 or newer** (validated: 26.9.2, display driver
   `32.0.32015.2008`). 26.2.2 is the oldest driver this path is documented
   for; nothing older was tried.
3. **BIOS: UMA Frame Buffer Size = 1G.** On the validated AMI Aptio board it
   is under Advanced > GFX Configuration. See "Memory split" below for why.
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

### Memory split

The GPU's memory pool is the BIOS carve-out plus the shared GPU memory
Windows grants, and Windows caps shared GPU memory at half of the RAM it can
see. Measured on the validated 128 GB box:

| UMA Frame Buffer Size | Windows sees | WSL `free` total | rocminfo GPU pool |
| --- | --- | --- | --- |
| 1G (validated) | 136.0 GB | 117 GiB | 67268878 KB (~64 GiB) |
| 64G (default) | 68.3 GB | 31 GiB | 100298866 KB (~95.6 GiB) |

Both pools agree, to within about 0.2 GiB, with carve-out + half of what
Windows sees (1 GiB + 68.0 GB, and 64 GiB + 34.2 GB). The 64G default gives the GPU more memory but leaves
WSL 31 GiB, too little for a release build next to a checkpoint load. The 1G
setting trades about 31 GiB of GPU pool for a usable WSL; HIP then reports
`totalGlobalMem` = 68883331072 bytes (64.15 GiB), matching rocminfo.

## 2. WSL provisioning

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

## 3. Preflight

```bash
bash -l scripts/strix-hip/preflight.sh > preflight.json
```

Use a login shell (`bash -l`). `wsl -e bash script` does not read
`/etc/profile.d`, so the DXG variable is missing and preflight fails with exit
2, which is the intended signal. Exit 3 means no gfx1151 agent or HIP device.
The JSON carries OS and WSL kernel, ROCm / librocdxg / HSA / HIP / hipcc
versions, the rocminfo gfx name and pool size, `free -b`, the Windows driver
and visible memory (through WSL interop, when enabled), and the properties a
small HIP program compiled on the spot reports for each device.

Attach the JSON to any result produced on the host.

## 4. Build environment

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

This environment produced a working release `met` on the validated host for a
model target that is not on `main` yet; the targets on `main` use the same
variables.

## 5. Recovery and known limits

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
  anything that relies on `cuMemAllocManaged` through the shim fails.
- **Lazy allocation failure.** An allocation beyond the remaining GPU pool may
  not fail at `hipMalloc` but later, on use. Size budgets from the pool
  preflight reports (the shim's `cuMemGetInfo_v2` returns the same
  `hipMemGetInfo` numbers), not from `free` in WSL.
- **Device counts differ by API.** rocminfo reports 40 compute units; HIP's
  `multiProcessorCount` is 20 (workgroup processors). Kernel launch sizing sees
  the HIP number through the shim.

## Sample preflight output

Validated host, 2026-10-07 (`bash -l preflight.sh`, exit 0):

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
that run is the known-bad control for the exit-status check.

TODO: rerun this guide on a second Strix Halo box and record its fingerprint
next to this one before calling the configuration portable across boards.
