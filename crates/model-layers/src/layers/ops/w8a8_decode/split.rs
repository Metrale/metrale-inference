// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: One launch of up to four per-row-scaled W8A8 projections of the same activation,
//! each into its own output (`w8a8_gemv_rowscale_mb1_ku8_split4`, `w8a8_gemv.cu`). A model whose
//! decode runs several small projections of one input (GLM-5.3's DSA q_a, kv_a, indexer wk and
//! compress gate) replaces their launches with this one. Each output row reads only its own
//! segment's weight rows and scales with the `mb1_ku8` entry's tile code, so every output holds
//! the bytes its own [`super::w8a8_gemv`] launch writes.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - 1..=[`SPLIT4_MAX_ROWS`] rows: the range [`super::w8a8_gemv`] serves with the `mb1_ku8`
//!   entry, the only one this kernel instantiates.
//! - Per-row scales only; every segment shares K; every segment but the last is a whole number
//!   of 16-row tiles, so no tile straddles two outputs.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::{GEMV_ROWS, GEMV_THREADS, W8A8_K_UNIT, W8a8Scratch, entry_index};
use crate::weight_map::{Fp8Weight, WeightQuantFormat};

/// 2026-10-21: The module and entry of the four-output launch.
pub const W8A8_SPLIT4_MODULE: &str = "w8a8_gemv";
pub const W8A8_SPLIT4_ENTRY: &str = "w8a8_gemv_rowscale_mb1_ku8_split4";
/// 2026-10-21: The most rows one split launch serves (one token tile of 8, the `mb1_ku8` range).
pub const SPLIT4_MAX_ROWS: usize = 8;

/// 2026-10-21: One projection of a split launch: its FP8 weight and its output (`[rows, ldc]`
/// BF16, the first `w.n` columns written).
#[derive(Clone, Copy, Debug)]
pub struct SplitOut {
    pub w: Fp8Weight,
    pub out: DevicePtr,
    pub ldc: u32,
}

/// 2026-10-21: Why `segs` at `rows` rows cannot run as one split launch, or `None` when it can.
pub fn split4_refusal(segs: &[SplitOut], rows: usize) -> Option<String> {
    if !(1..=4).contains(&segs.len()) {
        return Some(format!("{} projections, want 1..=4", segs.len()));
    }
    if !(1..=SPLIT4_MAX_ROWS).contains(&rows) || entry_index(rows) != 0 {
        return Some(format!("{rows} rows, want 1..={SPLIT4_MAX_ROWS}"));
    }
    let k = segs[0].w.k;
    if k == 0 || !k.is_multiple_of(W8A8_K_UNIT) {
        return Some(format!("K={k} is not a positive multiple of {W8A8_K_UNIT}"));
    }
    for (i, s) in segs.iter().enumerate() {
        if s.w.scale_format != WeightQuantFormat::Fp8PerRow || s.w.k != k {
            return Some(format!(
                "projection {i} is [{}, {}] {:?}; want per-row scales at K={k}",
                s.w.n, s.w.k, s.w.scale_format
            ));
        }
        if s.w.n == 0 || s.w.weight == DevicePtr::NULL || s.w.row_scale == DevicePtr::NULL {
            return Some(format!("projection {i} is empty"));
        }
        if s.out == DevicePtr::NULL || s.ldc < s.w.n {
            return Some(format!(
                "projection {i}: output ldc={} for {} rows",
                s.ldc, s.w.n
            ));
        }
        if i + 1 < segs.len() && !s.w.n.is_multiple_of(GEMV_ROWS) {
            return Some(format!(
                "projection {i} has {} rows, not a multiple of {GEMV_ROWS}",
                s.w.n
            ));
        }
    }
    None
}

/// 2026-10-21: The projections `segs` of the activation [`super::w8a8_act_quant`] left in
/// `scratch` (per-row layout, `rows` rows of K), as one launch of `kernel`
/// ([`W8A8_SPLIT4_ENTRY`]). Errors, before launching, on what [`split4_refusal`] names, on an
/// unresolved kernel, or on a scratch that does not hold the activation.
pub fn w8a8_gemv_split4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    segs: &[SplitOut],
    scratch: &W8a8Scratch,
    rows: usize,
    stream: u64,
) -> Result<()> {
    if let Some(why) = split4_refusal(segs, rows) {
        anyhow::bail!("w8a8_gemv_split4: {why}");
    }
    ensure!(
        kernel.0 != 0,
        "w8a8_gemv_split4: {W8A8_SPLIT4_ENTRY} is not in this target's kernels"
    );
    let k = segs[0].w.k;
    ensure!(
        scratch.q != DevicePtr::NULL
            && scratch.q_bytes >= rows * k as usize
            && scratch.scale_bytes >= rows * 4,
        "w8a8_gemv_split4: the scratch does not hold {rows} rows of K={k}"
    );
    let n: u32 = segs.iter().map(|s| s.w.n).sum();
    // 2026-10-21: An absent segment repeats segment 0 and starts at N, so no tile selects it.
    let mut nb = [n; 3];
    let mut start = 0;
    for (i, s) in segs.iter().enumerate().take(segs.len() - 1) {
        start += s.w.n;
        nb[i] = start;
    }
    let mut l = KernelLaunch::new(gpu, kernel)
        .grid([n.div_ceil(GEMV_ROWS), 1, 1])
        .block([GEMV_THREADS, 1, 1])
        .arg_ptr(scratch.q)
        .arg_ptr(scratch.scale);
    for i in 0..4 {
        let s = segs.get(i).unwrap_or(&segs[0]);
        l = l
            .arg_ptr(s.w.weight)
            .arg_ptr(s.w.row_scale)
            .arg_ptr(s.out)
            .arg_u32(s.ldc);
    }
    l.arg_u32(rows as u32)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(k)
        .arg_u32(nb[0])
        .arg_u32(nb[1])
        .arg_u32(nb[2])
        .launch(stream)
}

#[cfg(test)]
#[path = "split_tests.rs"]
mod tests;
