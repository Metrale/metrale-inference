// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-27: Launchers for the grouped NVFP4 MoE decode kernels
//! (`kernels/gb10/common/moe_nvfp4_grouped.cu`).
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.
//!
//! The layout is that of the grouped FP8 decode (`fp8_moe_grouped.rs`): rows grouped by
//! expert and the active experts listed by `moe_fp8_grouped_sort`, intermediates laid out
//! by sorted position, and the blend
//! `moe_weighted_sum_blend_fp8_grouped`. Only the weight format differs: row-major
//! NVFP4 (packed E2M1, E4M3 block scales of 16, per-tensor scale 2).

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::Fp8GroupedGeometry;
use crate::weight_map::QuantizedWeight;

/// 2026-09-27: Output columns per gate+up CTA. Must equal `NG_GU_COLS_PER_CTA` in the `.cu`.
pub const NVFP4_GROUPED_GATE_UP_COLS_PER_CTA: u32 = 16;

/// 2026-09-27: Rows per gate+up pass. Must equal `NG_GU_ROWS` in the `.cu`.
pub const NVFP4_GROUPED_GATE_UP_ROWS_PER_PASS: u32 = 4;

/// 2026-09-27: Output columns per down CTA. Must equal `NG_DOWN_COLS_PER_CTA` in the `.cu`.
pub const NVFP4_GROUPED_DOWN_COLS_PER_CTA: u32 = 128;

/// 2026-09-27: Rows per down pass. Must equal `NG_DOWN_ROWS` in the `.cu`.
pub const NVFP4_GROUPED_DOWN_ROWS_PER_PASS: u32 = 4;

/// 2026-10-02: `moe_expert_gate_up_act_nvfp4_grouped` (CUDA cores, FP32 SiLU products).
pub const NVFP4_GROUPED_GATE_UP_SCALAR: Fp8GroupedGeometry = Fp8GroupedGeometry {
    cols_per_cta: NVFP4_GROUPED_GATE_UP_COLS_PER_CTA,
    rows_per_pass: NVFP4_GROUPED_GATE_UP_ROWS_PER_PASS,
    threads: 128,
};

/// 2026-10-02: `moe_expert_down_act_nvfp4_grouped` (reads FP32 SiLU products).
pub const NVFP4_GROUPED_DOWN_SCALAR: Fp8GroupedGeometry = Fp8GroupedGeometry {
    cols_per_cta: NVFP4_GROUPED_DOWN_COLS_PER_CTA,
    rows_per_pass: NVFP4_GROUPED_DOWN_ROWS_PER_PASS,
    threads: 256,
};

/// 2026-10-02: Rows per pass of the tensor-core kernels (`moe_nvfp4_grouped_tc.cu`,
/// `TC_ROWS`), gate+up and down alike.
pub const NVFP4_GROUPED_TC_ROWS_PER_PASS: u32 = 8;

/// 2026-10-02: `moe_expert_gate_up_act_nvfp4_grouped_tc` (`NTC_GU_COLS` columns per CTA; writes
/// the SiLU product as BF16 hi + lo).
pub const NVFP4_GROUPED_GATE_UP_TC: Fp8GroupedGeometry = Fp8GroupedGeometry {
    cols_per_cta: 64,
    rows_per_pass: NVFP4_GROUPED_TC_ROWS_PER_PASS,
    threads: 128,
};

/// 2026-10-02: `moe_expert_down_act_nvfp4_grouped_tc` (`NTC_DOWN_COLS` columns per CTA; reads
/// the hi + lo SiLU rows).
pub const NVFP4_GROUPED_DOWN_TC: Fp8GroupedGeometry = Fp8GroupedGeometry {
    cols_per_cta: 128,
    rows_per_pass: NVFP4_GROUPED_TC_ROWS_PER_PASS,
    threads: 128,
};

/// 2026-10-14: `moe_expert_gate_up_act_nvfp4_grouped_tc_split` (`NTC_SPLIT_GU_COLS`: a column
/// tile's gate and up rows on two warps, so half the unsplit kernel's columns per CTA).
pub const NVFP4_GROUPED_GATE_UP_TC_SPLIT: Fp8GroupedGeometry = Fp8GroupedGeometry {
    cols_per_cta: 32,
    rows_per_pass: NVFP4_GROUPED_TC_ROWS_PER_PASS,
    threads: 128,
};

/// 2026-10-14: `moe_expert_down_act_nvfp4_grouped_tc_mt1` (`NTC_DOWN1_COLS`: one m-tile per warp).
pub const NVFP4_GROUPED_DOWN_TC_MT1: Fp8GroupedGeometry = Fp8GroupedGeometry {
    cols_per_cta: 64,
    rows_per_pass: NVFP4_GROUPED_TC_ROWS_PER_PASS,
    threads: 128,
};

/// 2026-10-14: `moe_expert_gate_up_act_nvfp4_grouped_hsm` (Hopper,
/// kernels/hopper/common/moe_nvfp4_decode_hopper.cu, `HSM_GU_*`): one gate warp and one up warp per
/// 16-column tile, each warp's weight tile staged whole in shared memory; the split pair's bytes.
pub const NVFP4_GROUPED_GATE_UP_HSM: Fp8GroupedGeometry = Fp8GroupedGeometry {
    cols_per_cta: 16,
    rows_per_pass: NVFP4_GROUPED_TC_ROWS_PER_PASS,
    threads: 64,
};

/// 2026-10-14: `moe_expert_down_act_nvfp4_grouped_hsm` (`HSM_DN_*`): four warps of one m-tile, each
/// warp's weight tile staged whole in shared memory; the `_tc_mt1` kernel's bytes.
pub const NVFP4_GROUPED_DOWN_HSM: Fp8GroupedGeometry = Fp8GroupedGeometry {
    cols_per_cta: 64,
    rows_per_pass: NVFP4_GROUPED_TC_ROWS_PER_PASS,
    threads: 128,
};

/// 2026-10-14: The dynamic shared memory one CTA may take (sm_90: 227 KiB).
pub const NVFP4_GROUPED_HSM_SMEM_MAX: u32 = 227 * 1024;

/// 2026-10-14: Dynamic shared bytes of an `_hsm` CTA of `warps` warps over a `k`-deep projection
/// with load groups of `k_step` and `xs_bytes` of split exchange (`hsm_head_bytes` + 9 * K per
/// warp in the `.cu`): one 8-byte mbarrier per warp and load group (rounded up to 16), the
/// exchange, then each warp's 16 x K / 2 packed and 16 x K / 16 scale bytes.
fn nvfp4_grouped_hsm_smem(warps: u32, k: u32, k_step: u32, xs_bytes: u32) -> u32 {
    (warps * (k / k_step) * 8).div_ceil(16) * 16 + xs_bytes + warps * 9 * k
}

/// 2026-10-14: [`nvfp4_grouped_hsm_smem`] of the `_hsm` gate+up over `k` = hidden (two warps,
/// the 2 KiB `gtc_split_xs_floats(1)` exchange).
pub fn nvfp4_grouped_hsm_gate_up_smem(k: u32) -> u32 {
    nvfp4_grouped_hsm_smem(
        NVFP4_GROUPED_GATE_UP_HSM.threads / 32,
        k,
        NVFP4_GROUPED_GATE_UP_TC_K_STEP,
        2048,
    )
}

/// 2026-10-14: [`nvfp4_grouped_hsm_smem`] of the `_hsm` down over `k` = inter.
pub fn nvfp4_grouped_hsm_down_smem(k: u32) -> u32 {
    nvfp4_grouped_hsm_smem(
        NVFP4_GROUPED_DOWN_HSM.threads / 32,
        k,
        NVFP4_GROUPED_DOWN_TC_K_STEP,
        0,
    )
}

/// 2026-10-14: Whether the `_hsm` pair takes an `inter` x `hidden` expert: the split pair's load
/// groups, whole CTAs of its geometries, and both CTAs within [`NVFP4_GROUPED_HSM_SMEM_MAX`].
pub fn nvfp4_grouped_hsm_pair_shape_ok(hidden: u32, inter: u32) -> bool {
    nvfp4_grouped_tc_pair_shape_ok(
        hidden,
        inter,
        NVFP4_GROUPED_GATE_UP_HSM,
        NVFP4_GROUPED_DOWN_HSM,
    ) && nvfp4_grouped_hsm_gate_up_smem(hidden) <= NVFP4_GROUPED_HSM_SMEM_MAX
        && nvfp4_grouped_hsm_down_smem(inter) <= NVFP4_GROUPED_HSM_SMEM_MAX
}

/// 2026-10-13: K of one tensor-core load group: 128-wide chunks times the projection's chunks
/// per group (`NTC_GU_G` = 2, `NTC_DOWN_G` = 1 in the `.cu`).
pub const NVFP4_GROUPED_GATE_UP_TC_K_STEP: u32 = 256;
pub const NVFP4_GROUPED_DOWN_TC_K_STEP: u32 = 128;

/// 2026-10-02: Whether the tensor-core kernels take an `n`-column, `k`-deep projection: `k` in
/// whole load groups of `k_step` (2026-10-13: the projection's own, so a 640-wide down
/// projection is admitted), `n` in whole CTAs of `geometry`.
pub fn nvfp4_grouped_tc_shape_ok(
    n: u32,
    k: u32,
    geometry: Fp8GroupedGeometry,
    k_step: u32,
) -> bool {
    n > 0
        && k > 0
        && k_step > 0
        && k.is_multiple_of(k_step)
        && n.is_multiple_of(geometry.cols_per_cta)
}

/// 2026-10-13: Whether the tensor-core pair takes an `inter` x `hidden` expert: gate+up
/// `[inter, hidden]` and down `[hidden, inter]`, each in its own load groups.
pub fn nvfp4_grouped_tc_expert_shape_ok(hidden: u32, inter: u32) -> bool {
    nvfp4_grouped_tc_pair_shape_ok(
        hidden,
        inter,
        NVFP4_GROUPED_GATE_UP_TC,
        NVFP4_GROUPED_DOWN_TC,
    )
}

/// 2026-10-14: [`nvfp4_grouped_tc_expert_shape_ok`] for a tensor-core pair of the given launch
/// geometries (the unsplit pair, or the split pair `NVFP4_GROUPED_*_TC_SPLIT` / `_MT1`, whose
/// load groups are the unsplit pair's).
pub fn nvfp4_grouped_tc_pair_shape_ok(
    hidden: u32,
    inter: u32,
    gate_up: Fp8GroupedGeometry,
    down: Fp8GroupedGeometry,
) -> bool {
    nvfp4_grouped_tc_shape_ok(inter, hidden, gate_up, NVFP4_GROUPED_GATE_UP_TC_K_STEP)
        && nvfp4_grouped_tc_shape_ok(hidden, inter, down, NVFP4_GROUPED_DOWN_TC_K_STEP)
}

/// 2026-09-27: One NVFP4 projection across the routed experts: device tables of each
/// expert's packed and block-scale pointers, and its per-tensor scales as f32.
#[derive(Clone, Copy, Debug)]
pub struct Nvfp4ExpertTables {
    pub packed_ptrs: DevicePtr,
    pub scale_ptrs: DevicePtr,
    pub scale2_vals: DevicePtr,
}

/// 2026-09-27: Grouped NVFP4 gate+up and SiLU over `k`-wide input rows. `cap` is
/// `fp8_grouped_active_cap`; the first `ceil(num_tokens / geometry.rows_per_pass)` block rows
/// are the shared expert. Writes the FP32 product `silu(bf16(gate)) * bf16(up)`,
/// `[positions, n]` for the routed experts into `act` and `[num_tokens, n]` for the shared
/// expert into `sh_act` (2026-10-02: the tensor-core kernel writes it as BF16 hi + lo in the
/// same bytes, which only its down kernel reads).
/// 2026-10-14: `shared_bytes`: the kernel's dynamic shared memory (0 but for the `_hsm` kernels,
/// [`nvfp4_grouped_hsm_gate_up_smem`]).
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gate_up_act_nvfp4_grouped(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    geometry: Fp8GroupedGeometry,
    shared_bytes: u32,
    input: DevicePtr,
    gate: Nvfp4ExpertTables,
    up: Nvfp4ExpertTables,
    act: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    active_experts: DevicePtr,
    active_count: DevicePtr,
    sh_gate: &QuantizedWeight,
    sh_up: &QuantizedWeight,
    sh_act: DevicePtr,
    n: u32,
    k: u32,
    cap: u32,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([
            div_ceil(n, geometry.cols_per_cta),
            cap + div_ceil(num_tokens, geometry.rows_per_pass),
            1,
        ])
        .block([geometry.threads, 1, 1])
        .shared_mem(shared_bytes)
        .arg_ptr(input)
        .arg_ptr(gate.packed_ptrs)
        .arg_ptr(gate.scale_ptrs)
        .arg_ptr(gate.scale2_vals)
        .arg_ptr(up.packed_ptrs)
        .arg_ptr(up.scale_ptrs)
        .arg_ptr(up.scale2_vals)
        .arg_ptr(act)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_ptr(active_experts)
        .arg_ptr(active_count)
        .arg_ptr(sh_gate.weight)
        .arg_ptr(sh_gate.weight_scale)
        .arg_f32(sh_gate.weight_scale_2)
        .arg_ptr(sh_up.weight)
        .arg_ptr(sh_up.weight_scale)
        .arg_f32(sh_up.weight_scale_2)
        .arg_ptr(sh_act)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(cap)
        .arg_u32(num_tokens)
        .launch(stream)
}

/// 2026-09-27: Grouped NVFP4 down over the SiLU product. `k` is the intermediate width; rows
/// of `act` and `output` are sorted positions, rows of `sh_act` and `sh_down_out` tokens.
#[allow(clippy::too_many_arguments)]
/// 2026-10-04: Repacks one row-major NVFP4 matrix `[n, k]` (packed E2M1 `[n, k / 2]` at `packed`,
/// E4M3 block scales `[n, k / 16]` at `scale`) in place into the lean layout the `_lean`
/// tensor-core expert kernels read (`nvfp4_tc_lean_repack`, moe_nvfp4_grouped_tc.cu): same bytes
/// and size, outputs byte-identical to the row-major point. Sets the device u32 at `bad` (zeroed
/// by the caller) on a negative or NaN scale, after which the matrix is unusable. Needs
/// [`nvfp4_lean_repack_shape_ok`].
pub fn nvfp4_tc_lean_repack(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    packed: DevicePtr,
    scale: DevicePtr,
    n: u32,
    k: u32,
    bad: DevicePtr,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        nvfp4_lean_repack_shape_ok(n, k),
        "nvfp4_tc_lean_repack: shape [{n}, {k}] not admitted"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([n / 16, 1, 1])
        .block([256, 1, 1])
        .shared_mem(NVFP4_LEAN_REPACK_SMEM_PER_K * k)
        .arg_ptr(packed)
        .arg_ptr(scale)
        .arg_u32(k)
        .arg_ptr(bad)
        .launch(stream)
}

/// 2026-10-04: Shared bytes `nvfp4_tc_lean_repack` stages per unit of K: a 16-row tile's packed
/// (16 * K / 2) and scale (16 * K / 16) bytes.
pub const NVFP4_LEAN_REPACK_SMEM_PER_K: u32 = 9;

/// 2026-10-04: `nvfp4_tc_lean_repack` admits whole 16-row tiles, whole 128-K chunks, and a tile
/// that fits the default 48 KiB of shared memory.
pub fn nvfp4_lean_repack_shape_ok(n: u32, k: u32) -> bool {
    n > 0
        && n.is_multiple_of(16)
        && k > 0
        && k.is_multiple_of(128)
        && NVFP4_LEAN_REPACK_SMEM_PER_K * k <= 48 * 1024
}

/// 2026-10-14: The grouped NVFP4 down (doc above `nvfp4_tc_lean_repack`); `shared_bytes` as for
/// [`moe_expert_gate_up_act_nvfp4_grouped`] ([`nvfp4_grouped_hsm_down_smem`] for `_hsm`).
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_down_act_nvfp4_grouped(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    geometry: Fp8GroupedGeometry,
    shared_bytes: u32,
    act: DevicePtr,
    down: Nvfp4ExpertTables,
    output: DevicePtr,
    expert_offsets: DevicePtr,
    active_experts: DevicePtr,
    active_count: DevicePtr,
    sh_act: DevicePtr,
    sh_down: &QuantizedWeight,
    sh_down_out: DevicePtr,
    n: u32,
    k: u32,
    cap: u32,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([
            div_ceil(n, geometry.cols_per_cta),
            cap + div_ceil(num_tokens, geometry.rows_per_pass),
            1,
        ])
        .block([geometry.threads, 1, 1])
        .shared_mem(shared_bytes)
        .arg_ptr(act)
        .arg_ptr(down.packed_ptrs)
        .arg_ptr(down.scale_ptrs)
        .arg_ptr(down.scale2_vals)
        .arg_ptr(output)
        .arg_ptr(expert_offsets)
        .arg_ptr(active_experts)
        .arg_ptr(active_count)
        .arg_ptr(sh_act)
        .arg_ptr(sh_down.weight)
        .arg_ptr(sh_down.weight_scale)
        .arg_f32(sh_down.weight_scale_2)
        .arg_ptr(sh_down_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(cap)
        .arg_u32(num_tokens)
        .launch(stream)
}

#[cfg(test)]
#[path = "nvfp4_moe_grouped_tests.rs"]
mod tests;
