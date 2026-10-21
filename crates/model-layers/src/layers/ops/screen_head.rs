// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: The screened LM head (`kernels/gb10/common/screen_head.cu`): launchers, and the
//! bound the screen keeps every possible argmax under.
//!
//! ## The bound
//!
//! Row `i` of the BF16 head is `w`, an activation row is `x` (BF16), and the full head's logit
//! is `bf16(R)` with `R` the FP32 sum `dense_gemv_bf16_batchm` forms. The screen copy stores per
//! 128-weight block `b` a scale `s_b = max|w_b| / 127` and `q = rint(w / s_b)`; the load-time
//! build checks, in exact arithmetic, `|w - s_b q| <= s_b / 2 (1 + 2^-20)` for every weight. The
//! scan computes `S` from `q` and `s` in FP32 and `E = sum_b s_b |x_b|_1` rounded up. With
//! `T = sum x w` exactly, `H_b = |x_b|_1` and `t = 1 + 2^-20`:
//!
//! - `|T - sum_b s_b sum x q| <= sum |x| |w - s q| <= t/2 E` (the quantization term);
//! - `|R - T| <= gamma(d_ref) sum |x w| <= gamma(d_ref) 127.5 t E` (the reference's FP32 sum:
//!   products of two BF16 values are exact, and every term passes at most `d_ref` additions;
//!   `|w| <= s (127 + t/2)`);
//! - `|S - sum_b s_b sum x q| <= gamma(d_scr) 127 E` (the screen's FP32 sum, `|q| <= 127`).
//!
//! So `|R - S| <= kappa E` with `kappa` = [`screen_kappa`]; the kernel adds `2^-100` for
//! underflow (at most a few hundred operations of at most `2^-149` each) and refuses (marks
//! every row a candidate) past `2^100`, far from FP32 overflow.
//!
//! **Claim.** Let `up_i >= R_i >= lo_i` (rounded outward), `L = max_j lo_j`, attained at `j0`,
//! and `pad = 2^-7 (1 + 2^-7) |L| + 2^-120`. A row with `up_i < L - pad` is not the argmax of
//! the BF16 logits, nor tied with it.
//! *Proof.* `R_i <= up_i < L - pad <= R_j0 - pad`, and `R_j0 - R_i > (R_j0 - L) + pad`. Equal
//! BF16 roundings `m` of `R_i` and `R_j0` would put both in `m`'s rounding interval, of width at
//! most `ulp(m) <= 2^-7 |m| <= 2^-7 (1 + 2^-8) |R_j0| + 2^-133`, and `|R_j0| <= |L| + (R_j0 - L)`,
//! so `R_j0 - R_i <= 2^-7 (1 + 2^-8) (|L| + (R_j0 - L)) + 2^-133 < pad + (R_j0 - L)`, a
//! contradiction. Rounding is monotone, so `bf16(R_i) < bf16(R_j0)`: the row's key is below
//! row `j0`'s (or 0), and `j0` itself is a candidate (`up_j0 >= lo_j0 = L`). ∎
//!
//! The candidates are recomputed with the reference's own arithmetic and keyed as
//! `argmax_bf16` breaks ties (`argmax_key.cuh`), so the max key over every rank's shard names
//! the token `argmax_bf16` returns for the whole row.
//!
//! Owner: model-layers ops.
//! Invariants: [`screen_shape_ok`] holds for every launch.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

/// 2026-10-21: Weights per screen scale.
pub const SCREEN_BLOCK: u32 = 128;
/// 2026-10-21: Most activation rows per screened call (`SCREEN_MAX_M` in the kernel).
pub const SCREEN_MAX_M: u32 = 4;
/// 2026-10-21: Most `m * k` the scan stages in shared memory (`SCREEN_MAX_MK`).
pub const SCREEN_MAX_MK: u32 = 16_384;
/// 2026-10-21: Bytes of the state block (`ScreenState`, padded), zeroed before each call.
pub const SCREEN_STATE_BYTES: usize = 128;
/// 2026-10-21: Offset of the u32 candidate counts in the state block.
pub const SCREEN_STATE_COUNTS: usize = 64;
/// 2026-10-21: Rows of the head each scan warp walks (grid sizing only).
const SCAN_ROWS_PER_WARP: u32 = 8;
/// 2026-10-21: Warps per block of the scan and rescore (`SCREEN_WARPS`).
const WARPS: u32 = 8;

/// 2026-10-21: Whether the kernels take `m` activation rows of width `k`.
pub fn screen_shape_ok(m: u32, k: u32) -> bool {
    (1..=SCREEN_MAX_M).contains(&m) && k > 0 && k.is_multiple_of(512) && m * k <= SCREEN_MAX_MK
}

/// 2026-10-21: `n u / (1 - n u)`, `u = 2^-24`: the relative error bound of `n` FP32 roundings.
fn gamma(n: u32) -> f64 {
    let nu = f64::from(n) * 2f64.powi(-24);
    nu / (1.0 - nu)
}

/// 2026-10-21: The two parts of the bound at width `k`: the quantization term `t/2` and the FP32
/// rounding terms of the reference and the screen (module docs).
pub fn screen_kappa_parts(k: u32) -> (f64, f64) {
    let t = 1.0 + 2f64.powi(-20);
    // 2026-10-21: The reference: each of 64 lanes adds 8 products per pass over `k / 512` passes
    // (a 64-vector slab per pass), then 5 shuffle levels and the cross-warp add.
    let d_ref = 8 * k.div_ceil(512) + 6;
    // 2026-10-21: The screen: 16 products per chunk, one fold per chunk a lane walks
    // (`k / 512`), 5 shuffle levels, and one for the scale product.
    let d_scr = 16 + k.div_ceil(512) + 5 + 1;
    let rounding = 127.5 * t * gamma(d_ref) + 127.0 * gamma(d_scr);
    (0.5 * t, rounding)
}

/// 2026-10-21: `kappa` of the module docs, rounded up to FP32.
pub fn screen_kappa(k: u32) -> f32 {
    let (quant, rounding) = screen_kappa_parts(k);
    f32_up(quant + rounding)
}

/// 2026-10-21: The smallest FP32 at or above `v` (finite, positive `v`).
pub fn f32_up(v: f64) -> f32 {
    let f = v as f32;
    if f64::from(f) >= v {
        f
    } else {
        f32::from_bits(f.to_bits() + 1)
    }
}

/// 2026-10-21: Build the screen copy of `rows` head rows of width `k` at `w` into `q` (`rows *
/// k` bytes) and `s` (`rows * k / 128` FP32); `bad` (one u32, zeroed by the caller) gets bit 0
/// for a non-finite weight, bit 1 for a block past the residual bound.
#[allow(clippy::too_many_arguments)]
pub fn screen_head_quant_int8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    w: DevicePtr,
    q: DevicePtr,
    s: DevicePtr,
    bad: DevicePtr,
    rows: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    ensure!(screen_shape_ok(1, k), "screen_head_quant_int8: k={k}");
    let blocks = u64::from(rows) * u64::from(k / SCREEN_BLOCK);
    let grid = u32::try_from(blocks.div_ceil(u64::from(WARPS)))?;
    KernelLaunch::new(gpu, kernel)
        .grid([grid, 1, 1])
        .block([32 * WARPS, 1, 1])
        .arg_ptr(w)
        .arg_ptr(q)
        .arg_ptr(s)
        .arg_ptr(bad)
        .arg_u32(rows)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-10-21: The screened head's buffers for one shard.
#[derive(Clone, Copy, Debug)]
pub struct ScreenShard {
    /// 2026-10-21: The BF16 rows of this shard (`rows * k`).
    pub w: DevicePtr,
    /// 2026-10-21: Their screen copy and scales.
    pub q: DevicePtr,
    pub s: DevicePtr,
    /// 2026-10-21: FP32 `[SCREEN_MAX_M, rows]` scratch for the upper bounds.
    pub upper: DevicePtr,
    /// 2026-10-21: The state block (`SCREEN_STATE_BYTES`).
    pub state: DevicePtr,
    pub rows: u32,
    pub k: u32,
    /// 2026-10-21: Global index of the shard's first row.
    pub base: u32,
}

/// 2026-10-21: The screen kernels.
#[derive(Clone, Copy, Debug)]
pub struct ScreenKernels {
    pub scan: KernelHandle,
    pub rescore: KernelHandle,
    pub resolve: KernelHandle,
}

impl ScreenKernels {
    /// 2026-10-21: All three loaded.
    pub fn loaded(&self) -> bool {
        self.scan.0 != 0 && self.rescore.0 != 0 && self.resolve.0 != 0
    }
}

/// 2026-10-21: Grid of the scan over `rows` head rows.
pub fn scan_grid(rows: u32) -> u32 {
    div_ceil(rows, WARPS * SCAN_ROWS_PER_WARP).max(1)
}

/// 2026-10-21: Grid of the rescore over `rows` head rows (one warp per 32 rows).
pub fn rescore_grid(rows: u32) -> u32 {
    div_ceil(div_ceil(rows, 32), WARPS).max(1)
}

/// 2026-10-21: This shard's argmax keys of the `m` activation rows at `x` into the state block:
/// zero the state, scan, rescore. `kappa` is [`screen_kappa`] in production.
#[allow(clippy::too_many_arguments)]
pub fn screen_head_keys(
    gpu: &dyn GpuBackend,
    k: &ScreenKernels,
    sh: &ScreenShard,
    x: DevicePtr,
    m: u32,
    kappa: f32,
    stream: u64,
) -> Result<()> {
    ensure!(
        screen_shape_ok(m, sh.k) && sh.rows > 0,
        "screen_head_keys: m={m} k={} rows={}",
        sh.k,
        sh.rows
    );
    gpu.memset_async(sh.state, 0, SCREEN_STATE_BYTES, stream)?;
    KernelLaunch::new(gpu, k.scan)
        .grid([scan_grid(sh.rows), 1, 1])
        .block([32 * WARPS, 1, 1])
        .arg_ptr(sh.q)
        .arg_ptr(sh.s)
        .arg_ptr(x)
        .arg_u32(m)
        .arg_u32(sh.rows)
        .arg_u32(sh.k)
        .arg_f32(kappa)
        .arg_ptr(sh.upper)
        .arg_ptr(sh.state)
        .launch(stream)?;
    KernelLaunch::new(gpu, k.rescore)
        .grid([rescore_grid(sh.rows), 1, 1])
        .block([32 * WARPS, 1, 1])
        .arg_ptr(sh.w)
        .arg_ptr(x)
        .arg_u32(m)
        .arg_u32(sh.rows)
        .arg_u32(sh.k)
        .arg_u32(sh.base)
        .arg_ptr(sh.upper)
        .arg_ptr(sh.state)
        .launch(stream)
}

/// 2026-10-21: `out[0..m]` (u32 tokens) from the state block's keys and `n_peers` peer state
/// blocks at `peers + s * slot_bytes` (a one-shot exchange's slots; 0 peers on one rank).
#[allow(clippy::too_many_arguments)]
pub fn screen_head_resolve(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    state: DevicePtr,
    peers: DevicePtr,
    slot_bytes: u64,
    n_peers: u32,
    m: u32,
    out: DevicePtr,
    stream: u64,
) -> Result<()> {
    ensure!(
        (1..=SCREEN_MAX_M).contains(&m),
        "screen_head_resolve: m={m}"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([32, 1, 1])
        .arg_ptr(state)
        .arg_ptr(peers)
        .arg_u64(slot_bytes)
        .arg_u32(n_peers)
        .arg_u32(m)
        .arg_ptr(out)
        .launch(stream)
}

/// 2026-10-21: Bytes of the stamp a screened verify writes (0xFF) at the head of the logits
/// buffer, where every head call writes logits (`SCREEN_STAMP_WORDS` in the kernel).
pub const SCREEN_STAMP_BYTES: usize = 16;

/// 2026-10-21: `out` (one BF16) = the full head's logit at row `row`, id `id`: recomputed from the
/// whole head `w` (`[vocab, k]`) and activation row `row` of `x` while `stamp` holds the stamp (or
/// is NULL), else copied from `logits`.
#[allow(clippy::too_many_arguments)]
pub fn screen_head_exact_id(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    w: DevicePtr,
    x: DevicePtr,
    k: u32,
    (row, id, vocab): (u32, u32, u32),
    logits: DevicePtr,
    stamp: DevicePtr,
    out: DevicePtr,
    stream: u64,
) -> Result<()> {
    ensure!(
        id < vocab && k.is_multiple_of(8),
        "screen_head_exact_id: id={id} vocab={vocab} k={k}"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([32, 1, 1])
        .arg_ptr(w)
        .arg_ptr(x)
        .arg_u32(k)
        .arg_u32(row)
        .arg_u32(id)
        .arg_u32(vocab)
        .arg_ptr(logits)
        .arg_ptr(stamp)
        .arg_ptr(out)
        .launch(stream)
}

#[cfg(test)]
#[path = "screen_head_tests.rs"]
mod tests;
