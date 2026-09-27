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

/// 2026-09-25: Output columns per down CTA. Must equal `DOWN_COLS_PER_CTA` in the
/// `.cu`.
pub const FP8_GROUPED_DOWN_COLS_PER_CTA: u32 = 32;

/// 2026-09-26: Rows per gate+up pass. Must equal `GU_GROUP_ROWS` in the `.cu`:
/// the shared expert takes `ceil(num_tokens / this)` block rows.
pub const FP8_GROUPED_GATE_UP_ROWS_PER_PASS: u32 = 4;

/// 2026-09-26: Rows per down pass. Must equal `GROUP_ROWS` in the `.cu`.
pub const FP8_GROUPED_DOWN_ROWS_PER_PASS: u32 = 4;

/// 2026-09-25: Cap on active experts, which sizes the grouped grids' Y extent:
/// `num_tokens * top_k` rows can reach at most that many distinct experts. It
/// does not depend on the routing, so a captured graph stays valid for every
/// routing.
pub fn fp8_grouped_active_cap(num_tokens: u32, top_k: u32, num_experts: u32) -> u32 {
    (num_tokens * top_k).min(num_experts)
}

/// 2026-09-25: Builds the compacted active-expert list from `expert_offsets`:
/// `active_experts[0..count]` in ascending order, and `active_count[0] =
/// count`. Launch it on the stream of the grouped kernels that read it.
pub fn moe_fp8_grouped_compact(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    expert_offsets: DevicePtr,
    active_experts: DevicePtr,
    active_count: DevicePtr,
    num_experts: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(expert_offsets)
        .arg_ptr(active_experts)
        .arg_ptr(active_count)
        .arg_u32(num_experts)
        .launch(stream)
}

/// 2026-09-26: Grouped FP8 gate+up and SiLU. `cap` is [`fp8_grouped_active_cap`];
/// the first `ceil(num_tokens / FP8_GROUPED_GATE_UP_ROWS_PER_PASS)` block rows are
/// the shared expert. Writes the FP32 product
/// `silu(bf16(gate)) * bf16(up)`, `[positions, n]` for the routed experts into
/// `act` and `[num_tokens, n]` for the shared expert into `sh_act`.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gate_up_act_fp8_grouped(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
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
            div_ceil(n, FP8_GROUPED_GATE_UP_COLS_PER_CTA),
            cap + div_ceil(num_tokens, FP8_GROUPED_GATE_UP_ROWS_PER_PASS),
            1,
        ])
        .block([128, 1, 1])
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

/// 2026-09-26: Grouped FP8 down over the SiLU product: 256 threads, 8 warps of 4
/// output columns each. `k` is the intermediate width; rows of `act` and
/// `output` are sorted positions, rows of `sh_act` and `sh_down_out` tokens.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_down_act_fp8_grouped(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
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
            div_ceil(n, FP8_GROUPED_DOWN_COLS_PER_CTA),
            cap + div_ceil(num_tokens, FP8_GROUPED_DOWN_ROWS_PER_PASS),
            1,
        ])
        .block([256, 1, 1])
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
