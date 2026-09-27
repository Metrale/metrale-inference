// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Launchers for the carried-state GDN verify
//! (`kernels/gb10/common/gated_delta_rule_carry.cu`): the K = 2..4 verify
//! kernels, the conv twin, and their standalone folds.
//!
//! Owner: model-layers ops (GDN).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

/// 2026-09-26: Carried-state verify (`gdn_carry_wy{2,3,4}` or its `_lazy` form, chosen
/// by the caller's `kernel`). It applies each sequence's pending rows
/// (`pend[slot_tab[b]]`) to its copy of H and writes H back (the lazy form only when
/// this verify's rows might not fit behind them in the stash,
/// `pending + K > GDN_CARRY_CAP`), writes the output, and stashes the rows it verified
/// at `carry_base + slot * seq_floats`. Engaged word b becomes 2 after a write-back,
/// else 1.
#[allow(clippy::too_many_arguments)]
pub fn gdn_carry_wy(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_table: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    carry_base: DevicePtr,
    slot_tab: DevicePtr,
    pend: DevicePtr,
    seq_floats: u32,
    batch_size: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    qk_stride: u32,
    v_stride: u32,
    gb_stride: u32,
    k_dim: u32,
    engaged_flag: DevicePtr,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        !h_table.is_null()
            && !carry_base.is_null()
            && !slot_tab.is_null()
            && !pend.is_null()
            && !engaged_flag.is_null(),
        "gdn_carry_wy: null table/stash/slot/pend/flag"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, batch_size, 1])
        .block([128, 1, 1])
        .arg_ptr(h_table)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(output)
        .arg_ptr(carry_base)
        .arg_ptr(slot_tab)
        .arg_ptr(pend)
        .arg_u32(seq_floats)
        .arg_u32(batch_size)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .arg_u32(k_dim)
        .arg_ptr(engaged_flag)
        .launch(stream)
}

/// 2026-09-26: Standalone fold (`gdn_carry_flush`) over `layers` GDN layers:
/// layer `l` reads its h pointers at entry `l * table_layer_entries` of
/// `h_table`, its stash at `carry_base + l * carry_layer_floats` floats and
/// its counts at `pend + l * pend_layer_entries`. A count of 0 leaves H alone;
/// `pend` is not cleared.
#[allow(clippy::too_many_arguments)]
pub fn gdn_carry_flush(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_table: DevicePtr,
    table_layer_entries: u64,
    carry_base: DevicePtr,
    carry_layer_floats: u64,
    slot_tab: DevicePtr,
    pend: DevicePtr,
    pend_layer_entries: u32,
    seq_floats: u32,
    batch_size: u32,
    num_v_heads: u32,
    layers: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, batch_size, layers])
        .block([128, 1, 1])
        .arg_ptr(h_table)
        .arg_u64(table_layer_entries)
        .arg_ptr(carry_base)
        .arg_u64(carry_layer_floats)
        .arg_ptr(slot_tab)
        .arg_ptr(pend)
        .arg_u32(pend_layer_entries)
        .arg_u32(seq_floats)
        .arg_u32(batch_size)
        .arg_u32(num_v_heads)
        .launch(stream)
}

/// 2026-09-26: Carried-state conv verify (`gdn_carry_conv`): the twin of
/// `gdn_verify_fused_conv_kn_batched` that first shifts each sequence's pending input rows
/// into its window (writing it back when there were any), writes no snapshot and no final
/// window, and stashes the position inputs at `conv_stash + slot * stash_seq_elems`. `lazy`
/// must match the verify kernel of the same run (`gdn_carry_wy{K}_lazy`): the window is then
/// written back only when the stash is full, else on every launch.
#[allow(clippy::too_many_arguments)]
pub fn gdn_carry_conv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state: DevicePtr,
    new_input: DevicePtr,
    weight: &crate::weight_map::DenseWeight,
    output: DevicePtr,
    conv_stash: DevicePtr,
    slot_tab: DevicePtr,
    pend: DevicePtr,
    stash_seq_elems: u32,
    num_tokens: u32,
    dim: u32,
    d_conv: u32,
    qk_channels: u32,
    head_dim: u32,
    input_stride: u32,
    output_stride: u32,
    l2_eps: f32,
    n_seq: u32,
    conv_state_seq_stride: u32,
    input_seq_stride: u32,
    output_seq_stride: u32,
    lazy: bool,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([dim.div_ceil(256), n_seq, 1])
        .block([256, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(new_input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_ptr(conv_stash)
        .arg_ptr(slot_tab)
        .arg_ptr(pend)
        .arg_u32(stash_seq_elems)
        .arg_u32(num_tokens)
        .arg_u32(dim)
        .arg_u32(d_conv)
        .arg_u32(qk_channels)
        .arg_u32(head_dim)
        .arg_u32(input_stride)
        .arg_u32(output_stride)
        .arg_f32(l2_eps)
        .arg_u32(conv_state_seq_stride)
        .arg_u32(input_seq_stride)
        .arg_u32(output_seq_stride)
        .arg_u32(u32::from(lazy))
        .launch(stream)
}

/// 2026-09-26: Standalone conv fold (`gdn_carry_conv_flush`) over `layers` GDN layers, laid
/// out as in [`gdn_carry_flush`] with conv-state pointers and a BF16 stash.
#[allow(clippy::too_many_arguments)]
pub fn gdn_carry_conv_flush(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    state_table: DevicePtr,
    table_layer_entries: u64,
    conv_stash: DevicePtr,
    stash_layer_elems: u64,
    slot_tab: DevicePtr,
    pend: DevicePtr,
    pend_layer_entries: u32,
    stash_seq_elems: u32,
    batch_size: u32,
    dim: u32,
    d_conv: u32,
    layers: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([dim.div_ceil(256), batch_size, layers])
        .block([256, 1, 1])
        .arg_ptr(state_table)
        .arg_u64(table_layer_entries)
        .arg_ptr(conv_stash)
        .arg_u64(stash_layer_elems)
        .arg_ptr(slot_tab)
        .arg_ptr(pend)
        .arg_u32(pend_layer_entries)
        .arg_u32(stash_seq_elems)
        .arg_u32(batch_size)
        .arg_u32(dim)
        .arg_u32(d_conv)
        .launch(stream)
}

/// 2026-09-26: Rows the carry stash holds per (layer, slot): `CARRY_CAP` in
/// gated_delta_rule_carry.cu. A verify whose K rows might not fit behind the pending ones
/// (`pending + K > GDN_CARRY_CAP`) writes the state back and restarts the stash.
pub const GDN_CARRY_CAP: usize = 8;

/// 2026-09-26: Launch width (sequences) from which a run takes the lazy verify kernels,
/// which write the state back only when the stash is full. Measured on dgx3 (k=1,
/// --mtp-gate force): from 16 the lazy form is 2.5 % faster end to end at C=16; a threshold
/// of 8 instead of 16 gives C=8 214.4 -> 216.2 tok/s (three reps each) with C=4 unchanged;
/// at 2 and 4 sequences the eager form is ahead.
pub const GDN_CARRY_LAZY_MIN_SEQS: usize = 8;

/// 2026-09-26: Most pending rows an eager verify kernel folds; a slot holding more (only
/// after lazy verifies) is folded by `gdn_carry_flush` first.
pub const GDN_CARRY_EAGER_MAX_PENDING: usize = 4;

/// 2026-09-26: Per-(layer, slot) conv stash length in BF16 elements: `GDN_CARRY_CAP` input
/// rows.
pub const fn gdn_carry_conv_seq_elems(conv_dim: usize) -> usize {
    GDN_CARRY_CAP * conv_dim
}

/// 2026-09-26: Per-(layer, slot) stash width in floats:
/// `vn[CAP][nv][vd] | g[CAP][nv] | sk[CAP][nv][kd]`, the layout of the kernel's
/// `CARRY_VN`/`CARRY_G`/`CARRY_SK` macros.
pub const fn gdn_carry_seq_floats(nv: usize, kd: usize, vd: usize) -> usize {
    GDN_CARRY_CAP * (nv * vd + nv + nv * kd)
}
