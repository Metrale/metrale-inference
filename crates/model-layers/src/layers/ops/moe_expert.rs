// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the single-token routed MoE expert GEMVs over NVFP4 pointer tables (`moe_expert_gemv.cu`); the shared-expert variants are in `moe_expert_shared.rs`.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.
//!
//! Every kernel here reads `expert_indices[slot]` on the device (`blockIdx.y` is the slot) and
//! looks the expert's weights up in device-side pointer tables, so no routing reaches the host.

#![allow(unused_imports)]

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;
#[path = "moe_expert_shared.rs"]
mod shared;
pub use shared::{
    moe_expert_gate_up_shared, moe_expert_gate_up_shared_bf16,
    moe_expert_gate_up_shared_bf16_batch2, moe_expert_gate_up_shared_fp8,
    moe_expert_silu_down_shared, moe_expert_silu_down_shared_bf16,
    moe_expert_silu_down_shared_bf16_batch2, moe_expert_silu_down_shared_fp8,
};

/// 2026-09-25: All top_k expert W4A16 GEMVs in one launch, output `[top_k, N]`. `input_stride`
/// is 0 when every expert reads the same input row (gate/up) and K when slot e reads row e
/// (down). A null weight pointer (a remote expert under EP) writes zeros.
pub fn moe_expert_gemv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    packed_ptrs: DevicePtr,
    scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    output: DevicePtr,
    expert_indices: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    input_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), top_k, 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(packed_ptrs)
        .arg_ptr(scale_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(output)
        .arg_ptr(expert_indices)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .arg_u32(input_stride)
        .launch(stream)
}
