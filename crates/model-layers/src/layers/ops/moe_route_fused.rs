// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The Hopper fused exact router (`kernels/hopper/common/moe_route_fused_hopper.cu`):
//! router logits, softmax top-k and the slot sort of `GroupedRouting::PerRow` in one launch.
//!
//! Owner: model-layers (ops).
//! Invariants:
//! - [`moe_route_fused`] writes the bytes of `dense_gemv_batchm_split` ->
//!   `moe_topk_softmax_rows_warp` -> `moe_fp8_grouped_sort` on the same inputs, except the order
//!   of slots inside one expert's run of `sorted_token_ids`, which the sort leaves to its atomics
//!   in either form (`ops/moe_route_fused_tests.rs`).
//! - The launch refuses any shape outside [`moe_route_fused_shape_ok`].

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::Fp8GroupedSortOut;

/// 2026-10-14: Rows one launch routes (the kernel's RF_MAX_ROWS).
pub const MOE_ROUTE_FUSED_MAX_ROWS: u32 = 16;
/// 2026-10-14: Router outputs per block (RF_OUTS).
const OUTS_PER_BLOCK: u32 = 4;
/// 2026-10-14: Hidden width the per-lane weight registers hold: 64 lanes x RF_KV_MAX x 8.
const MAX_HIDDEN: u32 = 64 * 8 * 8;
/// 2026-10-14: Experts the warp top-k reads (TOPK_WARP_MAX_EXPERTS).
const MAX_EXPERTS: u32 = 512;
/// 2026-10-14: Picks per row the shared pick buffers hold (RF_MAX_TOP_K).
const MAX_TOP_K: u32 = 32;

/// 2026-10-14: Shape admission without a GPU: 1..=16 rows, a hidden width the weight registers
/// hold in 16-byte vectors, at most 512 experts, and 1 <= top_k <= min(32, experts).
pub fn moe_route_fused_shape_ok(m: u32, hidden: u32, num_experts: u32, top_k: u32) -> bool {
    (1..=MOE_ROUTE_FUSED_MAX_ROWS).contains(&m)
        && hidden >= 8
        && hidden.is_multiple_of(8)
        && hidden <= MAX_HIDDEN
        && (1..=MAX_EXPERTS).contains(&num_experts)
        && (1..=MAX_TOP_K.min(num_experts)).contains(&top_k)
}

/// 2026-10-14: Routes `m` rows of `input` `[m, hidden]` BF16 through the router weight `gate`
/// `[num_experts, hidden]` BF16: `logits` `[m, num_experts]` BF16, `indices` and `weights`
/// `[m, top_k]`, and the sort outputs `out`, which may overlap `logits`. `counter` is one u32,
/// zero on entry and zero again when the launch completes; launches that share it must not
/// overlap.
#[allow(clippy::too_many_arguments)]
pub fn moe_route_fused(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate: DevicePtr,
    logits: DevicePtr,
    indices: DevicePtr,
    weights: DevicePtr,
    out: Fp8GroupedSortOut,
    counter: DevicePtr,
    m: u32,
    num_experts: u32,
    hidden: u32,
    top_k: u32,
    normalize: bool,
    stream: u64,
) -> Result<()> {
    ensure!(
        moe_route_fused_shape_ok(m, hidden, num_experts, top_k),
        "moe_route_fused: m={m} hidden={hidden} experts={num_experts} top_k={top_k} outside the kernel's shape"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([num_experts.div_ceil(OUTS_PER_BLOCK), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate)
        .arg_ptr(logits)
        .arg_ptr(indices)
        .arg_ptr(weights)
        .arg_ptr(out.sorted_token_ids)
        .arg_ptr(out.sorted_expert_ids)
        .arg_ptr(out.expert_offsets)
        .arg_ptr(out.token_to_perm)
        .arg_ptr(out.active_experts)
        .arg_ptr(out.active_count)
        .arg_ptr(counter)
        .arg_u32(m)
        .arg_u32(num_experts)
        .arg_u32(hidden)
        .arg_u32(top_k)
        .arg_u32(u32::from(normalize))
        .launch(stream)
}
