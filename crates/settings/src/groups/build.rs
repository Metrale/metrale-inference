// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The `build` lever group: levers read when the binary is built.
//!
//! Owner: settings.
//! Invariants: rows in env-name order.

lever_group! {
    /// 2026-10-17: Build-time levers; no `met` process resolves them.
    pub specs BuildLevers { group: "build", class: Build }
    build_commit: Text = "unset", None, "METRALE_BUILD_COMMIT", "crates/server/src/tui/report.rs", "The commit a release build is stamped with (`option_env!`).";
    cuda_arch: Text = "unset", None, "METRALE_CUDA_ARCH", "crates/gpu-runtime/build.rs", "`METRALE_CUDA_ARCH` still wins when set explicitly.";
    extra_nvcc_flags: Text = "unset", None, "METRALE_EXTRA_NVCC_FLAGS", "crates/kernels/build_target.rs", "METRALE_EXTRA_NVCC_FLAGS — global override for kernel bisection tests.";
    hip_compat_include: Text = "unset", None, "METRALE_HIP_COMPAT_INCLUDE", "crates/kernels/build.rs", "(a) Stage the CUDA→HIP compat-header dir (metrale-kernels/hip/compat) and export METRALE_HIP_COMPAT_INCLUDE — HipTarget::compile force-includes hip/hip_runtime.h and `-I`s this dir so the...";
    kernel_image: Text = "fatbin", None, "METRALE_KERNEL_IMAGE", "crates/kernels/build_image.rs", "NVIDIA kernel image: `fatbin` (unset) embeds SASS for the HARDWARE.toml arch plus its PTX uncompressed, so module load skips the driver JIT; `ptx` embeds PTX text only, JIT-compiled at load. Any other value fails the build.";
    kernel_no_strict: Switch = "off", None, "METRALE_KERNEL_NO_STRICT", "crates/kernels/build_target.rs", "On by default; set METRALE_KERNEL_NO_STRICT=1 to disable locally (only the literal \"1\" disables, so METRALE_KERNEL_NO_STRICT=0 still means strict-on and never silently opts out). `1` turns it on.";
    kernel_tree_root: Path = "built-in", None, "METRALE_KERNEL_TREE_ROOT", "crates/model-layers/src/circuit_exec/sources.rs", "Root of the kernel tree, read with `env!` to embed files from `kernels/`. The build scripts of model-layers, circuit and met set it from metrale-kernel-root's `links` metadata (`DEP_METRALE_KERNEL_TREE_ROOT`), so a published crate needs no repository checkout; not a run-time setting.";
    no_rdma: Switch = "off", None, "METRALE_NO_RDMA", "crates/gpu-sys/build.rs", "RDMA-only opt-out: gate off the verbs shim (no libibverbs, no C compile) WITHOUT the nvcc/CUDA side effects that METRALE_SKIP_BUILD carries in metrale-storage. `1` turns it on.";
    no_warp_blockscale_mma: Switch = "off", None, "METRALE_NO_WARP_BLOCKSCALE_MMA", "crates/circuit/src/hardware/test_fixture.rs", "An nvcc define, not an environment variable: a class whose `[build] extra_nvcc_flags` carry `-DMETRALE_NO_WARP_BLOCKSCALE_MMA` compiles out the warp-level FP4 block-scale MMA paths its hardware lacks (sm_90a and sm_100a have no `mma.sync ... block_scale`). The circuit planner reads it as a `[[guard]]` in kernels/DEVICES.toml. Present in the flags turns it on.";
    predictor_arch: Text = "unset", None, "METRALE_PREDICTOR_ARCH", "crates/storage/build.rs", "`METRALE_PREDICTOR_ARCH` still wins when set explicitly, and the literal survives as the announced fallback for an unreadable file.";
    ptx_cache: Switch = "off", None, "METRALE_PTX_CACHE", "crates/kernels/build_ptx_cache.rs", "Content-hashed nvcc PTX cache for the kernel build. Unset or `0` leaves it off; any value other than `0` or `1` fails the build. Needs `METRALE_PTX_CACHE_MAX_BYTES`. `1` turns it on.";
    ptx_cache_max_bytes: Int = "unset", None, "METRALE_PTX_CACHE_MAX_BYTES", "crates/kernels/build_ptx_cache.rs", "LRU cap of the PTX cache in bytes: required, positive and decimal while `METRALE_PTX_CACHE=1`; there is no default cap.";
    target_closures: Text = "unset", None, "METRALE_TARGET_CLOSURES", "crates/kernels/src/lib.rs", "Closure set a kernel build selects (`option_env!`).";
}
