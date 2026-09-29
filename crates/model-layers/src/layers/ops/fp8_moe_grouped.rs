// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the cross-row grouped FP8 MoE decode kernels
//! (`moe_shared_expert_fused_fp8_grouped.cu`, `moe_fp8_grouped_blend.cu`).
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.
//!
//! Rows are grouped by expert (`moe_sort_by_expert`). A gate+up or SiLU+down
//! CTA owns one active expert and applies each weight row it streams to every
//! row routed to that expert; the batch-2/3 kernels put one (token, slot) pair
//! on each `blockIdx.y` instead. Intermediates are laid out by sorted
//! position, and the blend maps each slot back through `token_to_perm`.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::Fp8Weight;

/// 2026-09-26: Output columns per gate+up CTA. Must equal `GU_COLS_PER_CTA` in the
/// `.cu`.
pub const FP8_GROUPED_GATE_UP_COLS_PER_CTA: u32 = 8;

/// 2026-09-25: Output columns per down CTA. Must equal `DOWN_CTA_COLS` in the
/// `.cu`: 2026-09-27, four column groups of 32.
pub const FP8_GROUPED_DOWN_COLS_PER_CTA: u32 = 128;

/// 2026-09-26: Rows per gate+up pass. Must equal `GU_GROUP_ROWS` in the `.cu`:
/// the shared expert takes `ceil(num_tokens / this)` block rows.
pub const FP8_GROUPED_GATE_UP_ROWS_PER_PASS: u32 = 4;

/// 2026-09-26: Rows per down pass. Must equal `GROUP_ROWS` in the `.cu`.
pub const FP8_GROUPED_DOWN_ROWS_PER_PASS: u32 = 4;

/// 2026-09-28: Rows per pass of the tensor-core kernels (`moe_fp8_grouped_tc.cu`,
/// `TC_ROWS`), gate+up and down alike.
pub const FP8_GROUPED_TC_ROWS_PER_PASS: u32 = 8;

/// 2026-09-28: Output columns per tensor-core gate+up CTA (`TC_GU_COLS`).
pub const FP8_GROUPED_TC_GATE_UP_COLS_PER_CTA: u32 = 64;

/// 2026-09-28: Output columns per tensor-core down CTA (`TC_DOWN_COLS`).
pub const FP8_GROUPED_TC_DOWN_COLS_PER_CTA: u32 = 128;

/// 2026-09-28: The launch shape of a grouped gate+up or down kernel: output columns per
/// CTA (`grid.x = ceil(n / cols_per_cta)`), rows per pass (the shared expert takes
/// `ceil(num_tokens / rows_per_pass)` block rows) and threads per CTA.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp8GroupedGeometry {
    pub cols_per_cta: u32,
    pub rows_per_pass: u32,
    pub threads: u32,
}

/// 2026-09-28: `moe_expert_gate_up_act_fp8_grouped` (FP32 SiLU products).
pub const FP8_GROUPED_GATE_UP_SCALAR: Fp8GroupedGeometry = Fp8GroupedGeometry {
    cols_per_cta: FP8_GROUPED_GATE_UP_COLS_PER_CTA,
    rows_per_pass: FP8_GROUPED_GATE_UP_ROWS_PER_PASS,
    threads: 128,
};

/// 2026-09-28: `moe_expert_down_act_fp8_grouped` (reads FP32 SiLU products).
pub const FP8_GROUPED_DOWN_SCALAR: Fp8GroupedGeometry = Fp8GroupedGeometry {
    cols_per_cta: FP8_GROUPED_DOWN_COLS_PER_CTA,
    rows_per_pass: FP8_GROUPED_DOWN_ROWS_PER_PASS,
    threads: 256,
};

/// 2026-09-28: `moe_expert_gate_up_act_fp8_grouped_tc` (FP32 SiLU products).
pub const FP8_GROUPED_GATE_UP_TC: Fp8GroupedGeometry = Fp8GroupedGeometry {
    cols_per_cta: FP8_GROUPED_TC_GATE_UP_COLS_PER_CTA,
    rows_per_pass: FP8_GROUPED_TC_ROWS_PER_PASS,
    threads: 128,
};

/// 2026-09-28: `moe_expert_down_act_fp8_grouped_tc` (reads FP32 SiLU products as BF16 hi + lo).
pub const FP8_GROUPED_DOWN_TC: Fp8GroupedGeometry = Fp8GroupedGeometry {
    cols_per_cta: FP8_GROUPED_TC_DOWN_COLS_PER_CTA,
    rows_per_pass: FP8_GROUPED_TC_ROWS_PER_PASS,
    threads: 128,
};

/// 2026-09-28: Whether the tensor-core kernels take an `n`-column, `k`-deep projection:
/// both in whole 128 blocks (the FP8 scale blocks) and `n` in whole CTAs of `geometry`.
pub fn fp8_grouped_tc_shape_ok(n: u32, k: u32, geometry: Fp8GroupedGeometry) -> bool {
    n > 0
        && k > 0
        && k.is_multiple_of(128)
        && n.is_multiple_of(128)
        && n.is_multiple_of(geometry.cols_per_cta)
}

/// 2026-09-25: Cap on active experts, which sizes the grouped grids' Y extent:
/// `num_tokens * top_k` rows can reach at most that many distinct experts. It
/// does not depend on the routing, so a captured graph stays valid for every
/// routing.
pub fn fp8_grouped_active_cap(num_tokens: u32, top_k: u32, num_experts: u32) -> u32 {
    (num_tokens * top_k).min(num_experts)
}

/// 2026-09-27: Experts `moe_fp8_grouped_sort` handles (`PMS_MAX_EXPERTS`).
pub const FP8_GROUPED_SORT_MAX_EXPERTS: u32 = 1024;

/// 2026-09-27: The outputs of [`moe_fp8_grouped_sort`]: `moe_sort_by_expert`'s
/// four and the active-expert list with its length.
pub struct Fp8GroupedSortOut {
    pub sorted_token_ids: DevicePtr,
    pub sorted_expert_ids: DevicePtr,
    pub expert_offsets: DevicePtr,
    pub token_to_perm: DevicePtr,
    pub active_experts: DevicePtr,
    pub active_count: DevicePtr,
}

/// 2026-09-27: Sorts the `total_expanded` slots of `topk_ids` by expert, as
/// `moe_sort_by_expert` does, and writes `active_experts[0..count]` in
/// ascending order with `active_count[0] = count`, in one launch on `stream`
/// (the grouped kernels that read them must follow on it).
#[allow(clippy::too_many_arguments)]
pub fn moe_fp8_grouped_sort(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    out: Fp8GroupedSortOut,
    topk_ids: DevicePtr,
    total_expanded: u32,
    num_experts: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        num_experts <= FP8_GROUPED_SORT_MAX_EXPERTS,
        "moe_fp8_grouped_sort: {num_experts} experts exceed {FP8_GROUPED_SORT_MAX_EXPERTS}"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(topk_ids)
        .arg_ptr(out.sorted_token_ids)
        .arg_ptr(out.sorted_expert_ids)
        .arg_ptr(out.expert_offsets)
        .arg_ptr(out.token_to_perm)
        .arg_ptr(out.active_experts)
        .arg_ptr(out.active_count)
        .arg_u32(total_expanded)
        .arg_u32(num_experts)
        .arg_u32(top_k)
        .launch(stream)
}

/// 2026-09-26: Grouped FP8 gate+up and SiLU. `cap` is [`fp8_grouped_active_cap`];
/// the first `ceil(num_tokens / geometry.rows_per_pass)` block rows are the shared
/// expert. Writes the product `silu(bf16(gate)) * bf16(up)`, `[positions, n]` for
/// the routed experts into `act` and `[num_tokens, n]` for the shared expert into
/// `sh_act`: FP32 from the scalar kernel ([`FP8_GROUPED_GATE_UP_SCALAR`]), BF16 from
/// the tensor-core one ([`FP8_GROUPED_GATE_UP_TC`], 2026-09-28).
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gate_up_act_fp8_grouped(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    geometry: Fp8GroupedGeometry,
    input: DevicePtr,
    gp_w: DevicePtr,
    gp_s: DevicePtr,
    up_w: DevicePtr,
    up_s: DevicePtr,
    act: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    active_experts: DevicePtr,
    active_count: DevicePtr,
    sh_gate: &Fp8Weight,
    sh_up: &Fp8Weight,
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
        .arg_ptr(input)
        .arg_ptr(gp_w)
        .arg_ptr(gp_s)
        .arg_ptr(up_w)
        .arg_ptr(up_s)
        .arg_ptr(act)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_ptr(active_experts)
        .arg_ptr(active_count)
        .arg_ptr(sh_gate.weight)
        .arg_ptr(sh_gate.row_scale)
        .arg_ptr(sh_up.weight)
        .arg_ptr(sh_up.row_scale)
        .arg_ptr(sh_act)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(cap)
        .arg_u32(num_tokens)
        .launch(stream)
}

/// 2026-09-26: Grouped FP8 down over the SiLU product, launched with `geometry`
/// ([`FP8_GROUPED_DOWN_SCALAR`] reads FP32 products, [`FP8_GROUPED_DOWN_TC`] BF16).
/// `k` is the intermediate width; rows of `act` and `output` are sorted
/// positions, rows of `sh_act` and `sh_down_out` tokens.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_down_act_fp8_grouped(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    geometry: Fp8GroupedGeometry,
    act: DevicePtr,
    down_w: DevicePtr,
    down_s: DevicePtr,
    output: DevicePtr,
    expert_offsets: DevicePtr,
    active_experts: DevicePtr,
    active_count: DevicePtr,
    sh_act: DevicePtr,
    sh_down: &Fp8Weight,
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
        .arg_ptr(act)
        .arg_ptr(down_w)
        .arg_ptr(down_s)
        .arg_ptr(output)
        .arg_ptr(expert_offsets)
        .arg_ptr(active_experts)
        .arg_ptr(active_count)
        .arg_ptr(sh_act)
        .arg_ptr(sh_down.weight)
        .arg_ptr(sh_down.row_scale)
        .arg_ptr(sh_down_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(cap)
        .arg_u32(num_tokens)
        .launch(stream)
}

/// 2026-09-25: Grouped blend, one `blockIdx.y` per token. `expert_out` rows
/// are sorted positions; `token_to_perm[token * top_k + k]` maps a slot to its
/// row. `gate_weight` may be NULL (ungated shared expert).
#[allow(clippy::too_many_arguments)]
pub fn moe_weighted_sum_blend_fp8_grouped(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    output: DevicePtr,
    expert_out: DevicePtr,
    expert_weights: DevicePtr,
    token_to_perm: DevicePtr,
    shared_out: DevicePtr,
    input: DevicePtr,
    gate_weight: DevicePtr,
    hidden: u32,
    top_k: u32,
    k: u32,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(hidden, 256), num_tokens, 1])
        .block([256, 1, 1])
        .arg_ptr(output)
        .arg_ptr(expert_out)
        .arg_ptr(expert_weights)
        .arg_ptr(token_to_perm)
        .arg_ptr(shared_out)
        .arg_ptr(input)
        .arg_ptr(gate_weight)
        .arg_u32(hidden)
        .arg_u32(top_k)
        .arg_u32(k)
        .launch(stream)
}

#[cfg(test)]
#[path = "fp8_moe_grouped_tests.rs"]
mod tests;
