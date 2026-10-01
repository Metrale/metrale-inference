// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: Launchers for the fused gate+up grouped MoE prefill GEMMs over NVFP4 pointer
//! tables (moved out of `moe_grouped_a.rs` unchanged; the grid and the pointer-table layout are
//! described there).
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::super::*;

/// 2026-09-25: [`moe_w4a16_fused_gate_up_n128`] for the kernels with a 64-wide K step.
#[allow(clippy::too_many_arguments)]
pub fn moe_w4a16_fused_gate_up_k64_n128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    gate_packed_ptrs: DevicePtr,
    gate_scale_ptrs: DevicePtr,
    gate_scale2_vals: DevicePtr,
    up_packed_ptrs: DevicePtr,
    up_scale_ptrs: DevicePtr,
    up_scale2_vals: DevicePtr,
    c_gate: DevicePtr,
    c_up: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n_out: u32,
    k: u32,
    max_m_tiles: u32,
    stream: u64,
) -> Result<()> {
    check_e4m3_grouped(
        gpu,
        kernel,
        a,
        expert_offsets,
        sorted_token_ids,
        num_experts,
        k,
        stream,
    )?;
    KernelLaunch::new(gpu, kernel)
        .grid([
            n_tile_blocks(gpu, kernel, 2 * n_out)?,
            max_m_tiles,
            num_experts,
        ])
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(gate_packed_ptrs)
        .arg_ptr(gate_scale_ptrs)
        .arg_ptr(gate_scale2_vals)
        .arg_ptr(up_packed_ptrs)
        .arg_ptr(up_scale_ptrs)
        .arg_ptr(up_scale2_vals)
        .arg_ptr(c_gate)
        .arg_ptr(c_up)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n_out)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: [`moe_w4a16_fused_gate_up_k64_n128`] with a 128-row M tile and a 256-thread
/// block. `max_m_tiles_m128` counts 128-row tiles: the call site halves the 64-row count
/// (`forward_prefill_routed.rs`).
#[allow(clippy::too_many_arguments)]
pub fn moe_w4a16_fused_gate_up_k64_m128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    gate_packed_ptrs: DevicePtr,
    gate_scale_ptrs: DevicePtr,
    gate_scale2_vals: DevicePtr,
    up_packed_ptrs: DevicePtr,
    up_scale_ptrs: DevicePtr,
    up_scale2_vals: DevicePtr,
    c_gate: DevicePtr,
    c_up: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n_out: u32,
    k: u32,
    max_m_tiles_m128: u32,
    stream: u64,
) -> Result<()> {
    check_e4m3_grouped(
        gpu,
        kernel,
        a,
        expert_offsets,
        sorted_token_ids,
        num_experts,
        k,
        stream,
    )?;
    KernelLaunch::new(gpu, kernel)
        .grid([
            n_tile_blocks(gpu, kernel, 2 * n_out)?,
            max_m_tiles_m128,
            num_experts,
        ])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(gate_packed_ptrs)
        .arg_ptr(gate_scale_ptrs)
        .arg_ptr(gate_scale2_vals)
        .arg_ptr(up_packed_ptrs)
        .arg_ptr(up_scale_ptrs)
        .arg_ptr(up_scale2_vals)
        .arg_ptr(c_gate)
        .arg_ptr(c_up)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n_out)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: Gate and up grouped GEMMs in one launch: the grid spans `2 * n_out` columns, the
/// first `n_out` for gate and the rest for up.
#[allow(clippy::too_many_arguments)]
pub fn moe_w4a16_fused_gate_up_n128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    gate_packed_ptrs: DevicePtr,
    gate_scale_ptrs: DevicePtr,
    gate_scale2_vals: DevicePtr,
    up_packed_ptrs: DevicePtr,
    up_scale_ptrs: DevicePtr,
    up_scale2_vals: DevicePtr,
    c_gate: DevicePtr,
    c_up: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n_out: u32,
    k: u32,
    max_m_tiles: u32,
    stream: u64,
) -> Result<()> {
    check_e4m3_grouped(
        gpu,
        kernel,
        a,
        expert_offsets,
        sorted_token_ids,
        num_experts,
        k,
        stream,
    )?;
    KernelLaunch::new(gpu, kernel)
        .grid([
            n_tile_blocks(gpu, kernel, 2 * n_out)?,
            max_m_tiles,
            num_experts,
        ])
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(gate_packed_ptrs)
        .arg_ptr(gate_scale_ptrs)
        .arg_ptr(gate_scale2_vals)
        .arg_ptr(up_packed_ptrs)
        .arg_ptr(up_scale_ptrs)
        .arg_ptr(up_scale2_vals)
        .arg_ptr(c_gate)
        .arg_ptr(c_up)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n_out)
        .arg_u32(k)
        .launch(stream)
}
