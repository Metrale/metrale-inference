// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: Launchers for the FP16 h-state exact MTP verify: `gdn_exact_chain_f16_{2,3,4}`
//! (`gdn_exact_carry.cu` in the qwen3.6-27b directory) and the batched FP32 conv chain
//! `gdn_conv_chain_f32_batched` (`kernels/gb10/common/gated_delta_rule_carry.cu`).
//!
//! Owner: model-layers ops (GDN).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

/// 2026-10-01: Where the FP16 exact chain finds its states: per-sequence pointer tables (the
/// verify's WY tables: h, Hi0, Hi1, Hi2), or the bases of a single sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum F16ChainStates {
    /// 2026-10-01: Device tables of `batch` per-sequence base pointers.
    Tables([DevicePtr; 4]),
    /// 2026-10-01: One sequence's h slot and its intermediates 0..2; batch must be 1.
    Single([DevicePtr; 4]),
}

/// 2026-10-01: The FP16 h-state exact verify, K = 2..4 rows per sequence (`kernel` is the
/// `gdn_exact_chain_f16_{K}` entry): the decode's FP16 fused-norm chain per row with the state
/// in registers, the FP16 intermediates of rows 0..K-2 and the final state written, and the BF16
/// normed rows. Rows are `b * K + t`; `strides` is `[qk, v, gb, z, out]` per row.
#[allow(clippy::too_many_arguments)]
pub fn gdn_exact_chain_f16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    states: F16ChainStates,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    z_gate: DevicePtr,
    norm_weight: DevicePtr,
    output: DevicePtr,
    batch: u32,
    heads: [u32; 3],
    strides: [u32; 5],
    eps: f32,
    stream: u64,
) -> Result<()> {
    let (ptrs, table) = match states {
        F16ChainStates::Tables(p) => (p, 1u32),
        F16ChainStates::Single(p) => {
            anyhow::ensure!(
                batch == 1,
                "gdn_exact_chain_f16: single-sequence bases with batch {batch}"
            );
            (p, 0u32)
        }
    };
    let [num_k_heads, num_v_heads, k_dim] = heads;
    let [qk_stride, v_stride, gb_stride, z_stride, out_stride] = strides;
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, batch, 1])
        .block([128, 1, 1])
        .arg_ptr(ptrs[0])
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(z_gate)
        .arg_ptr(norm_weight)
        .arg_ptr(output)
        .arg_ptr(ptrs[1])
        .arg_ptr(ptrs[2])
        .arg_ptr(ptrs[3])
        .arg_u32(batch)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .arg_u32(z_stride)
        .arg_u32(out_stride)
        .arg_f32(eps)
        .arg_u32(table)
        .launch(stream)
}

/// 2026-10-01: [`super::gdn_conv_chain_f32`] for `batch` sequences on consecutive conv slots
/// (`dim * d_conv` floats apart): row `b * num_tokens + t` reads `new_input + row *
/// input_stride` and writes `output + row * output_stride`; the window after position t <
/// num_tokens - 1 goes to `conv_inter + b * inter_strides[1] + t * inter_strides[0]` floats.
#[allow(clippy::too_many_arguments)]
pub fn gdn_conv_chain_f32_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state: DevicePtr,
    new_input: DevicePtr,
    weight: &crate::weight_map::DenseWeight,
    output: DevicePtr,
    conv_inter: DevicePtr,
    batch: u32,
    num_tokens: u32,
    dims: [u32; 4],
    l2_eps: f32,
    row_strides: [u32; 2],
    inter_strides: [u64; 2],
    stream: u64,
) -> Result<()> {
    let [dim, d_conv, qk_channels, head_dim] = dims;
    KernelLaunch::new(gpu, kernel)
        .grid([dim.div_ceil(256), batch, 1])
        .block([256, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(new_input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_ptr(conv_inter)
        .arg_u32(num_tokens)
        .arg_u32(dim)
        .arg_u32(d_conv)
        .arg_u32(qk_channels)
        .arg_u32(head_dim)
        .arg_f32(l2_eps)
        .arg_u32(row_strides[0])
        .arg_u32(row_strides[1])
        .arg_u64(inter_strides[0])
        .arg_u64(inter_strides[1])
        .launch(stream)
}
