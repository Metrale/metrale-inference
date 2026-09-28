// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: W8A8 decode projections for 1..=64 rows (`w8a8_gemv.cu`,
//! `w8a8_act_quant.cu`): the BF16 activation is quantized to E4M3 with a
//! dynamic FP32 scale per token (per-row weights) or per token and 128-wide K
//! group (128x128 block-scaled weights), then multiplied against the
//! checkpoint's E4M3 weight on E4M3 tensor-core MMAs with FP32 accumulation.
//! This is the precision an FP8 W8A8 checkpoint declares for these layers.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - A row's output bits do not depend on how many rows share the launch: the
//!   WxAy engine (`wxay_engine.cuh`) gives every entry the same K split (8
//!   warps, 128-wide chunks, warp-order reduction), the entries differ only in
//!   token tiles and load depth, and the quantizer's scales are per row.
//! - [`w8a8_decode_available`] is true only when every launch [`w8a8_proj`]
//!   would make is in range of its kernel and scratch; [`w8a8_proj`] refuses
//!   (errors, launches nothing) whenever it is false.

use anyhow::{Result, bail, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use crate::layers::try_kernel;
use crate::weight_map::{Fp8Weight, WeightQuantFormat};

/// 2026-09-28: The most rows one W8A8 projection serves: 128-row launches, the weight read
/// once per launch (a C128 verify step has 256 rows).
pub const W8A8_MAX_ROWS: usize = 256;
/// 2026-09-28: The most rows one GEMV launch serves (16 token tiles of 8).
pub const W8A8_LAUNCH_ROWS: usize = 128;
/// 2026-09-28: The K width of one scale group and of one kernel K unit.
pub const W8A8_K_UNIT: u32 = 128;
const GEMV_MODULE: &str = "w8a8_gemv";
const QUANT_MODULE: &str = "w8a8_act_quant";
const QUANT_THREADS: u32 = 256;
/// 2026-09-28: The engine's block: 8 warps.
const GEMV_THREADS: u32 = 256;
/// 2026-09-28: Weight rows per CTA.
const GEMV_ROWS: u32 = 16;
/// 2026-09-28: Entry name suffixes, indexed by [`entry_index`].
const ENTRIES: [&str; 5] = ["mb1_ku8", "mb2", "mb4", "mb8", "mb16"];

/// 2026-09-28: How a W8A8 weight carries its scales, and so how the
/// activation is quantized: one scale per output row with one per token, or
/// one per 128x128 weight block with one per (token, 128-wide K group).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum W8a8Scale {
    PerRow,
    Block128,
}

impl W8a8Scale {
    /// 2026-09-28: The layout of `format`, or `None` for a format this family
    /// does not read.
    pub fn of(format: WeightQuantFormat) -> Option<Self> {
        match format {
            WeightQuantFormat::Fp8PerRow => Some(Self::PerRow),
            WeightQuantFormat::Fp8BlockScaled => Some(Self::Block128),
            _ => None,
        }
    }

    /// 2026-09-28: Output rows a segment boundary must be a multiple of: one
    /// kernel row tile (16), or one scale block (128).
    fn segment_align(self) -> u32 {
        match self {
            Self::PerRow => 16,
            Self::Block128 => 128,
        }
    }

    /// 2026-09-28: FP32 activation scales per row for a K-wide activation.
    pub fn act_scales_per_row(self, k: u32) -> usize {
        match self {
            Self::PerRow => 1,
            Self::Block128 => (k / W8A8_K_UNIT) as usize,
        }
    }
}

/// 2026-09-28: One W8A8 projection: up to three E4M3 weights with the same K
/// and scale layout, stacked by rows into one output (attention Q|K|V, GDN
/// QKV|Z, FFN gate|up). The checkpoint's tensors are read in place.
#[derive(Clone, Copy, Debug)]
pub struct W8a8Weight {
    segs: [Fp8Weight; 3],
    count: usize,
    n: u32,
    k: u32,
    scale: W8a8Scale,
}

impl W8a8Weight {
    /// 2026-09-28: Stack `segs` (1..=3). Errors unless they share K and a scale
    /// layout this family reads, K is a multiple of 128, and every segment but
    /// the last is a multiple of the layout's segment alignment.
    pub fn new(segs: &[Fp8Weight]) -> Result<Self> {
        ensure!(
            (1..=3).contains(&segs.len()),
            "W8a8Weight: {} segments, want 1..=3",
            segs.len()
        );
        let Some(scale) = W8a8Scale::of(segs[0].scale_format) else {
            bail!(
                "W8a8Weight: scale format {:?} is not per-row or 128x128 block",
                segs[0].scale_format
            );
        };
        let k = segs[0].k;
        ensure!(
            k > 0 && k.is_multiple_of(W8A8_K_UNIT),
            "W8a8Weight: K={k} is not a positive multiple of {W8A8_K_UNIT}"
        );
        for (i, s) in segs.iter().enumerate() {
            ensure!(
                s.k == k && s.scale_format == segs[0].scale_format,
                "W8a8Weight: segment {i} is [{}, {}] {:?}, segment 0 is [.., {k}] {:?}",
                s.n,
                s.k,
                s.scale_format,
                segs[0].scale_format
            );
            ensure!(
                s.n > 0 && s.weight != DevicePtr::NULL && s.row_scale != DevicePtr::NULL,
                "W8a8Weight: segment {i} is empty"
            );
            if i + 1 < segs.len() {
                ensure!(
                    s.n.is_multiple_of(scale.segment_align()),
                    "W8a8Weight: segment {i} has {} rows, not a multiple of {}",
                    s.n,
                    scale.segment_align()
                );
            }
        }
        let mut all = [segs[0]; 3];
        all[..segs.len()].copy_from_slice(segs);
        Ok(Self {
            segs: all,
            count: segs.len(),
            n: segs.iter().map(|s| s.n).sum(),
            k,
            scale,
        })
    }

    /// 2026-09-28: Output rows (all segments).
    pub fn n(&self) -> u32 {
        self.n
    }
    /// 2026-09-28: Contract width.
    pub fn k(&self) -> u32 {
        self.k
    }
    pub fn scale(&self) -> W8a8Scale {
        self.scale
    }

    /// 2026-09-28: `(n1, n2)`, the first output rows of segments 1 and 2 (`n` for
    /// an absent segment).
    fn boundaries(&self) -> (u32, u32) {
        let n0 = self.segs[0].n;
        match self.count {
            1 => (self.n, self.n),
            2 => (n0, self.n),
            _ => (n0, n0 + self.segs[1].n),
        }
    }
}

/// 2026-09-28: The quantizers and the ten GEMV entry points (2 scale layouts
/// x [`ENTRIES`]). A handle is 0 when its module is not compiled.
#[derive(Clone, Copy, Debug)]
pub struct W8a8Kernels {
    quant_row: KernelHandle,
    quant_g128: KernelHandle,
    quant_silu_row: KernelHandle,
    quant_silu_g128: KernelHandle,
    rowscale: [KernelHandle; 5],
    blk128: [KernelHandle; 5],
}

impl W8a8Kernels {
    pub fn load(gpu: &dyn GpuBackend) -> Self {
        let tiles = |layout: &str| {
            ENTRIES.map(|e| try_kernel(gpu, GEMV_MODULE, &format!("w8a8_gemv_{layout}_{e}")))
        };
        Self {
            quant_row: try_kernel(gpu, QUANT_MODULE, "w8a8_act_quant_row"),
            quant_g128: try_kernel(gpu, QUANT_MODULE, "w8a8_act_quant_g128"),
            quant_silu_row: try_kernel(gpu, QUANT_MODULE, "w8a8_act_quant_silu_row"),
            quant_silu_g128: try_kernel(gpu, QUANT_MODULE, "w8a8_act_quant_silu_g128"),
            rowscale: tiles("rowscale"),
            blk128: tiles("blk128"),
        }
    }

    /// 2026-09-28: No kernel resolved (a build without the modules).
    pub fn none() -> Self {
        let z = KernelHandle(0);
        Self {
            quant_row: z,
            quant_g128: z,
            quant_silu_row: z,
            quant_silu_g128: z,
            rowscale: [z; 5],
            blk128: [z; 5],
        }
    }

    /// 2026-09-28: Whether every kernel of `scale`'s layout resolved (both
    /// quantizers and every GEMV entry).
    pub fn resolved(&self, scale: W8a8Scale) -> bool {
        let (q, qs, g) = match scale {
            W8a8Scale::PerRow => (self.quant_row, self.quant_silu_row, &self.rowscale),
            W8a8Scale::Block128 => (self.quant_g128, self.quant_silu_g128, &self.blk128),
        };
        q.0 != 0 && qs.0 != 0 && g.iter().all(|h| h.0 != 0)
    }
}

/// 2026-09-28: Device scratch for one quantized activation: `[rows, k]` E4M3
/// and its FP32 scales. One per model; every W8A8 projection on the stream
/// quantizes into it immediately before its GEMV.
#[derive(Clone, Copy, Debug)]
pub struct W8a8Scratch {
    pub q: DevicePtr,
    pub q_bytes: usize,
    pub scale: DevicePtr,
    pub scale_bytes: usize,
}

impl W8a8Scratch {
    /// 2026-09-28: Room for [`W8A8_MAX_ROWS`] rows of a `max_k`-wide
    /// activation in either scale layout.
    pub fn alloc(gpu: &dyn GpuBackend, max_k: u32) -> Result<Self> {
        let rows = W8A8_MAX_ROWS;
        let q_bytes = rows * max_k as usize;
        let scale_bytes = rows * (max_k.div_ceil(W8A8_K_UNIT) as usize).max(1) * 4;
        Ok(Self {
            q: gpu.alloc(q_bytes)?,
            q_bytes,
            scale: gpu.alloc(scale_bytes)?,
            scale_bytes,
        })
    }

    fn fits(&self, rows: usize, k: u32, scale: W8a8Scale) -> bool {
        self.q != DevicePtr::NULL
            && self.q_bytes >= rows * k as usize
            && self.scale_bytes >= rows * scale.act_scales_per_row(k) * 4
    }
}

/// 2026-09-28: The GEMV entry (index into [`ENTRIES`]) for a launch of `rows` tokens: 1, 2,
/// 4, 8 and 16 token tiles up to 8, 16, 32, 64 and 128 rows. Every entry sums in the same
/// order, so the choice changes speed only.
fn entry_index(rows: usize) -> usize {
    match rows {
        0..=8 => 0,
        9..=16 => 1,
        17..=32 => 2,
        33..=64 => 3,
        _ => 4,
    }
}

/// 2026-09-28: The "W8A8 available for this layer and shape" hook: the kernels
/// of `w`'s layout resolved, `rows` is in `1..=W8A8_MAX_ROWS`, and the scratch
/// holds the quantized activation. The declared-precision policy decides
/// whether W8A8 is wanted; this says whether it can run.
pub fn w8a8_decode_available(
    kernels: &W8a8Kernels,
    w: &W8a8Weight,
    rows: usize,
    scratch: &W8a8Scratch,
) -> bool {
    (1..=W8A8_MAX_ROWS).contains(&rows)
        && kernels.resolved(w.scale)
        && scratch.fits(rows, w.k, w.scale)
}

/// 2026-09-28: Quantize `rows` rows of the BF16 activation `x` (`[rows, ldx]`,
/// the first `k` columns) into `scratch` for `scale`'s layout.
#[allow(clippy::too_many_arguments)]
pub fn w8a8_act_quant(
    gpu: &dyn GpuBackend,
    kernels: &W8a8Kernels,
    scale: W8a8Scale,
    x: DevicePtr,
    ldx: u32,
    rows: usize,
    k: u32,
    scratch: &W8a8Scratch,
    stream: u64,
) -> Result<()> {
    ensure!(
        (1..=W8A8_MAX_ROWS).contains(&rows) && scratch.fits(rows, k, scale),
        "w8a8_act_quant: {rows} rows of K={k} do not fit the scratch"
    );
    ensure!(
        k.is_multiple_of(W8A8_K_UNIT) && ldx >= k && ldx.is_multiple_of(8),
        "w8a8_act_quant: K={k}, ldx={ldx} (want K % 128 == 0, ldx >= K, ldx % 8 == 0)"
    );
    let kernel = match scale {
        W8a8Scale::PerRow => kernels.quant_row,
        W8a8Scale::Block128 => kernels.quant_g128,
    };
    ensure!(
        kernel.0 != 0,
        "w8a8_act_quant: {scale:?} quantizer not compiled"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([rows as u32, 1, 1])
        .block([QUANT_THREADS, 1, 1])
        .arg_ptr(x)
        .arg_ptr(scratch.q)
        .arg_ptr(scratch.scale)
        .arg_u32(k)
        .arg_u32(ldx)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-28: [`w8a8_act_quant`] of `bf16(silu(gate) * up)`, computed in the
/// same launch with the arithmetic of `moe_silu_mul`: gate and up are
/// `[rows, ld]` BF16, the first `k` columns read.
#[allow(clippy::too_many_arguments)]
pub fn w8a8_act_quant_silu(
    gpu: &dyn GpuBackend,
    kernels: &W8a8Kernels,
    scale: W8a8Scale,
    gate: DevicePtr,
    up: DevicePtr,
    ld: u32,
    rows: usize,
    k: u32,
    scratch: &W8a8Scratch,
    stream: u64,
) -> Result<()> {
    ensure!(
        (1..=W8A8_MAX_ROWS).contains(&rows) && scratch.fits(rows, k, scale),
        "w8a8_act_quant_silu: {rows} rows of K={k} do not fit the scratch"
    );
    ensure!(
        k.is_multiple_of(W8A8_K_UNIT) && ld >= k && ld.is_multiple_of(8),
        "w8a8_act_quant_silu: K={k}, ld={ld} (want K % 128 == 0, ld >= K, ld % 8 == 0)"
    );
    let kernel = match scale {
        W8a8Scale::PerRow => kernels.quant_silu_row,
        W8a8Scale::Block128 => kernels.quant_silu_g128,
    };
    ensure!(
        kernel.0 != 0,
        "w8a8_act_quant_silu: {scale:?} quantizer not compiled"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([rows as u32, 1, 1])
        .block([QUANT_THREADS, 1, 1])
        .arg_ptr(gate)
        .arg_ptr(up)
        .arg_ptr(scratch.q)
        .arg_ptr(scratch.scale)
        .arg_u32(k)
        .arg_u32(ld)
        .arg_u32(ld)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-28: The GEMV over the activation [`w8a8_act_quant`] left in
/// `scratch`: `out[m, 0..n]` (row pitch `ldc` BF16 elements) for `m < rows`.
pub fn w8a8_gemv(
    gpu: &dyn GpuBackend,
    kernels: &W8a8Kernels,
    w: &W8a8Weight,
    scratch: &W8a8Scratch,
    rows: usize,
    out: DevicePtr,
    ldc: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        w8a8_decode_available(kernels, w, rows, scratch),
        "w8a8_gemv: not available for {rows} rows of [{}, {}] {:?}",
        w.n,
        w.k,
        w.scale
    );
    ensure!(ldc >= w.n, "w8a8_gemv: ldc={ldc} < N={}", w.n);
    let (n1, n2) = w.boundaries();
    let s = &w.segs;
    let a_scales = w.scale.act_scales_per_row(w.k) * 4;
    // 2026-09-28: 128-row launches; a row's arithmetic does not depend on its launch.
    let mut done = 0;
    while done < rows {
        let m = (rows - done).min(W8A8_LAUNCH_ROWS);
        let e = entry_index(m);
        let kernel = match w.scale {
            W8a8Scale::PerRow => kernels.rowscale[e],
            W8a8Scale::Block128 => kernels.blk128[e],
        };
        KernelLaunch::new(gpu, kernel)
            .grid([w.n.div_ceil(GEMV_ROWS), 1, 1])
            .block([GEMV_THREADS, 1, 1])
            .arg_ptr(scratch.q.offset(done * w.k as usize))
            .arg_ptr(scratch.scale.offset(done * a_scales))
            .arg_ptr(s[0].weight)
            .arg_ptr(s[0].row_scale)
            .arg_ptr(s[1].weight)
            .arg_ptr(s[1].row_scale)
            .arg_ptr(s[2].weight)
            .arg_ptr(s[2].row_scale)
            .arg_ptr(out.offset(done * ldc as usize * 2))
            .arg_u32(m as u32)
            .arg_u32(w.n)
            .arg_u32(w.k)
            .arg_u32(w.k)
            .arg_u32(ldc)
            .arg_u32(n1)
            .arg_u32(n2)
            .launch(stream)?;
        done += m;
    }
    Ok(())
}

/// 2026-09-28: One W8A8 projection: quantize `x` (`[rows, ldx]` BF16) and
/// multiply, writing `out` (`[rows, ldc]` BF16, the first `w.n()` columns).
#[allow(clippy::too_many_arguments)]
pub fn w8a8_proj(
    gpu: &dyn GpuBackend,
    kernels: &W8a8Kernels,
    w: &W8a8Weight,
    x: DevicePtr,
    ldx: u32,
    rows: usize,
    out: DevicePtr,
    ldc: u32,
    scratch: &W8a8Scratch,
    stream: u64,
) -> Result<()> {
    ensure!(
        w8a8_decode_available(kernels, w, rows, scratch),
        "w8a8_proj: not available for {rows} rows of [{}, {}] {:?}",
        w.n,
        w.k,
        w.scale
    );
    w8a8_act_quant(gpu, kernels, w.scale, x, ldx, rows, w.k, scratch, stream)?;
    w8a8_gemv(gpu, kernels, w, scratch, rows, out, ldc, stream)
}

#[cfg(test)]
#[path = "w8a8_decode_tests.rs"]
mod tests;
