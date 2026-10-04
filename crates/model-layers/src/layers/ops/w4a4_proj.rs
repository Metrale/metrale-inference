// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: W4A4 small-M projection on FP4 tensor cores (`w4a4_gemv_mx.cu`, module
//! `w4a4_gemv_mx`).
//!
//! The block-scale FP4 MMA (`kind::mxf4nvf4`) takes the checkpoint's NVFP4
//! weight bytes and E4M3 group scales as operands, with no dequant. The
//! activations are first quantised per row to NVFP4 with a per-row FP32 global
//! scale (`w4a4_quant_rows`), so the numerics are W4A4. Which projections take it
//! is the `--weight-quantization` tier's answer (`WeightQuantTier::w4a4_rows`):
//! under `declared`, a weight whose checkpoint declares FP4 activations (stamped
//! `Nvfp4Act::A4`); under `nvfp4`, every NVFP4 weight when `--w4a4-downcast` is on.
//! Every other NVFP4 weight stays W4A16.
//!
//! Scope: the projection sites that call [`nvfp4_proj_small_m`]: GDN
//! qkvz/out_proj, attention q/k/v/o and dense-FFN gate/up/down, up to
//! [`max_rows`] rows. The lm_head does not call it.
//!
//! Scratch: the quantised activations need `[M, K/2] + [M, K/16] + [M] f32` of
//! device memory. [`prepare`] allocates it once per backend at model build
//! (from `W4a16BatchmTiers::resolve`). Launches run on the caller's stream,
//! and the scratch and the last-quantisation record are shared, so two
//! projections on different streams at once would race on them.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - Device scratch is allocated only in [`prepare`], never by a launch.
//! - A projection reaches the FP4 kernels only when `WeightQuantTier::w4a4_rows` admits it.

use std::sync::{Mutex, OnceLock};

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::weight_map::QuantizedWeight;

/// 2026-09-25: Rows the narrow entries (`w4a4_gemv_mx32`) cover, and the widest
/// W4A16 fallback (`w4a16_gemv_batch32`).
pub const W4A4_MAX_M: u32 = 32;
/// 2026-09-25: Rows the wide entries (`w4a4_gemv_mx64*`) cover.
pub const W4A4_WIDE_MAX_M: u32 = 64;
/// 2026-09-25: Largest K the scratch is sized for.
pub const W4A4_MAX_K: u32 = 32768;

#[derive(Clone, Copy)]
struct W4a4State {
    quant: KernelHandle,
    mx8: KernelHandle,
    mx16: KernelHandle,
    mx32: KernelHandle,
    /// 2026-09-25: Activation-reuse twins (`METRALE_W4A4_MX_NT`). The
    /// `w4a4_gemv_nt_oracle` example compares them bit for bit with mx16/mx32.
    mx16_nt2: KernelHandle,
    mx32_nt4: KernelHandle,
    /// 2026-09-25: Persistent activation-staged entries (`METRALE_W4A4_MX_PS`),
    /// compared bit for bit with mx16/mx32 by the same example.
    mx16_ps: KernelHandle,
    mx32_ps: KernelHandle,
    /// 2026-09-25: Streaming multiprocessors: the persistent entries' grid.
    sms: u32,
    /// 2026-09-28: 33..=64 rows (zero handles when this target lacks them).
    mx64: KernelHandle,
    mx64_nt2: KernelHandle,
    /// 2026-09-28: [`W4A4_WIDE_MAX_M`] when `mx64`/`mx64_nt2` resolved, else
    /// [`W4A4_MAX_M`]; the scratch is sized for it.
    max_m: u32,
    aq: DevicePtr,
    a_scale: DevicePtr,
    a_gs: DevicePtr,
    /// 2026-09-25: W4A16 reference output for [`audit_enabled`] (null
    /// otherwise).
    audit_ref: DevicePtr,
}

/// 2026-09-25: Largest projection N the audit reference buffer holds; a wider
/// launch is not audited.
const AUDIT_MAX_N: usize = 65536;

/// 2026-09-25: `METRALE_W4A4_PROJ_AUDIT` (non-empty), a diagnostic. Every W4A4
/// projection launched outside a graph capture also runs the W4A16 path,
/// synchronises, and accumulates `||y_w4a4 - y_w4a16|| / ||y_w4a16||` per call
/// site, logged every 64 samples. Captured launches are not audited, so it
/// needs graphs off (`METRALE_NO_MTP_VERIFY_GRAPHS=1
/// METRALE_NO_DECODE_GRAPHS_MULTISEQ=1`) to see the decode and verify steps.
fn audit_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("METRALE_W4A4_PROJ_AUDIT").is_some_and(|v| !v.is_empty()))
}

// 2026-09-25: `w4a4_proj.rs` is loaded via `#[path = "ops/w4a4_proj.rs"]`, so
// the explicit `#[path]` is required to nest the submodule under this file.
#[path = "w4a4_proj/mx_plan.rs"]
mod mx_plan;
pub use mx_plan::*;
#[path = "w4a4_proj/fixed.rs"]
mod fixed;
pub use fixed::{fixed_nvfp4_proj, nvfp4_proj_mx};
#[path = "w4a4_proj/steps.rs"]
mod steps;
pub use steps::{Nvfp4ActBuf, W4a4Proj};

fn cache() -> &'static Mutex<Vec<(usize, Option<W4a4State>)>> {
    static CACHE: OnceLock<Mutex<Vec<(usize, Option<W4a4State>)>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Vec::new()))
}

fn key(gpu: &dyn GpuBackend) -> usize {
    gpu as *const dyn GpuBackend as *const () as usize
}

/// 2026-09-28: Resolve the kernels and allocate the scratch once per backend, when the
/// published tier can take the W4A4 path (`WeightQuantTier::uses_w4a4_decode`). Call at
/// model build, never inside a graph capture. When a kernel is missing it records `None`
/// for the backend and logs, and every projection stays W4A16.
pub fn prepare(gpu: &dyn GpuBackend) -> Result<()> {
    let tier = crate::layers::weight_quantization();
    // 2026-09-30: A fixed `--activation-quantization` may run NVFP4 activations under any tier
    // (`nvfp4_proj_mx`).
    let fixed = crate::layers::any_fixed();
    if !tier.uses_w4a4_decode() && !fixed {
        return Ok(());
    }
    let mut guard = cache().lock().unwrap_or_else(|p| p.into_inner());
    if guard.iter().any(|(k, _)| *k == key(gpu)) {
        return Ok(());
    }
    let h = |f: &str| crate::layers::try_kernel(gpu, "w4a4_gemv_mx", f);
    let (quant, mx8, mx16, mx32) = (
        h("w4a4_quant_rows"),
        h("w4a4_gemv_mx8"),
        h("w4a4_gemv_mx16"),
        h("w4a4_gemv_mx32"),
    );
    let (mx16_nt2, mx32_nt4) = (h("w4a4_gemv_mx16_nt2"), h("w4a4_gemv_mx32_nt4"));
    let (mx16_ps, mx32_ps) = (h("w4a4_gemv_mx16_ps"), h("w4a4_gemv_mx32_ps"));
    let state = if [quant, mx8, mx16, mx32, mx16_nt2, mx32_nt4, mx16_ps, mx32_ps]
        .iter()
        .all(|k| k.0 != 0)
    {
        let sms = gpu.sm_count()?;
        let (mx64, mx64_nt2, max_m) =
            wide_entries(tier, fixed, h("w4a4_gemv_mx64"), h("w4a4_gemv_mx64_nt2"));
        tracing::info!(
            "w4a4 projection ({tier:?}): METRALE_W4A4_MX_NT={} METRALE_W4A4_MX_PS={} ({sms} SMs), max rows {max_m}",
            mx_nt(),
            u8::from(mx_ps()),
        );
        let (m, k) = (max_m as usize, W4A4_MAX_K as usize);
        Some(W4a4State {
            quant,
            mx8,
            mx16,
            mx32,
            mx16_nt2,
            mx32_nt4,
            mx16_ps,
            mx32_ps,
            sms,
            mx64,
            mx64_nt2,
            max_m,
            aq: gpu.alloc(m * k / 2)?,
            a_scale: gpu.alloc(m * k / 16)?,
            a_gs: gpu.alloc(m * 4)?,
            audit_ref: if audit_enabled() {
                gpu.alloc(m * AUDIT_MAX_N * 2)?
            } else {
                DevicePtr::NULL
            },
        })
    } else {
        tracing::info!("w4a4_gemv_mx kernels are not in this target; NVFP4 projections run W4A16");
        None
    };
    guard.push((key(gpu), state));
    Ok(())
}

/// 2026-09-28: The 33..=64-row entries and the scratch's row count. Under `declared` they are
/// used when this target has them. Under `nvfp4` they belong to `--w4a4-downcast-wide`, as
/// before the tiers existed: without it they are not resolved and the scratch holds 32 rows.
/// 2026-09-30: Also under a fixed `--activation-quantization` (`fixed`), whose NVFP4 projections
/// run in chunks of the widest entry.
fn wide_entries(
    tier: metrale_config::WeightQuantTier,
    fixed: bool,
    mx64: KernelHandle,
    mx64_nt2: KernelHandle,
) -> (KernelHandle, KernelHandle, u32) {
    use metrale_config::{W4a4Downcast, WeightQuantization};
    if fixed && mx64.0 != 0 && mx64_nt2.0 != 0 {
        return (mx64, mx64_nt2, W4A4_WIDE_MAX_M);
    }
    match (tier.tier(), tier.downcast()) {
        (WeightQuantization::Declared, _) if mx64.0 != 0 && mx64_nt2.0 != 0 => {
            (mx64, mx64_nt2, W4A4_WIDE_MAX_M)
        }
        (WeightQuantization::Nvfp4, W4a4Downcast::Wide) => (mx64, mx64_nt2, W4A4_WIDE_MAX_M),
        _ => (KernelHandle(0), KernelHandle(0), W4A4_MAX_M),
    }
}

/// 2026-09-28: The widest row count the W4A4 path serves for `weight` on `gpu` under the
/// published tier (`WeightQuantTier::w4a4_rows`); 0 means W4A16.
pub fn weight_rows(gpu: &dyn GpuBackend, weight: &QuantizedWeight) -> u32 {
    rows_for(weight, max_rows(gpu))
}

fn rows_for(weight: &QuantizedWeight, kernel_rows: u32) -> u32 {
    crate::layers::weight_quantization().w4a4_rows(
        weight.act,
        W4A4_MAX_M,
        W4A4_WIDE_MAX_M,
        kernel_rows,
    )
}

/// 2026-09-28: Widest row count the W4A4 kernels serve on `gpu`: 64 or 32 once
/// [`prepare`] found them, else 0.
pub fn max_rows(gpu: &dyn GpuBackend) -> u32 {
    state(gpu).map_or(0, |s| s.max_m)
}

fn state(gpu: &dyn GpuBackend) -> Option<W4a4State> {
    let guard = cache().lock().unwrap_or_else(|p| p.into_inner());
    guard
        .iter()
        .find(|(k, _)| *k == key(gpu))
        .and_then(|(_, s)| *s)
}

/// 2026-09-28: Pure: may the W4A4 path serve this launch? `max_m` is [`max_rows`].
pub fn w4a4_route(m: u32, n: u32, k: u32, max_m: u32) -> bool {
    (1..=max_m).contains(&m) && n > 0 && k > 0 && k.is_multiple_of(64) && k <= W4A4_MAX_K
}

/// 2026-09-28: The projection launcher: the W4A4 FP4 MMA when the tier admits the
/// weight ([`weight_rows`]), the kernels are prepared and [`w4a4_route`] admits the
/// shape, else the W4A16
/// `w4a16_gemv_batchm` (which tries the tensor-core GEMV first). The W4A16
/// fallback returns an error above [`W4A4_MAX_M`] rows.
#[allow(clippy::too_many_arguments)]
#[track_caller]
pub fn nvfp4_proj_small_m(
    gpu: &dyn GpuBackend,
    batch_kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    proj(
        gpu,
        batch_kernel,
        input,
        weight,
        output,
        m,
        n,
        k,
        stream,
        false,
    )
}

/// 2026-09-25: [`nvfp4_proj_small_m`] for a projection whose `input` is byte for
/// byte the input of the preceding projection on this stream (attention k/v
/// after q, FFN up after gate). It skips re-quantising when the previous W4A4
/// quantisation was of exactly `(backend, input, m, k, stream)`, and otherwise
/// quantises as usual, so a wrong claim about the address cannot read a stale
/// quantisation. The caller guarantees the contents did not change in between.
#[allow(clippy::too_many_arguments)]
#[track_caller]
pub fn nvfp4_proj_small_m_same_input(
    gpu: &dyn GpuBackend,
    batch_kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    proj(
        gpu,
        batch_kernel,
        input,
        weight,
        output,
        m,
        n,
        k,
        stream,
        true,
    )
}

/// 2026-09-28: The prepared W4A4 kernels when [`nvfp4_proj_small_m`] takes the W4A4 path for
/// this launch: the tier uses W4A4 decode, the kernels are prepared, and [`w4a4_route`] admits
/// the shape at the weight's row edge.
fn w4a4_state_for(
    gpu: &dyn GpuBackend,
    weight: &QuantizedWeight,
    m: u32,
    n: u32,
    k: u32,
) -> Option<W4a4State> {
    if !crate::layers::weight_quantization().uses_w4a4_decode() {
        return None;
    }
    state(gpu).filter(|s| w4a4_route(m, n, k, rows_for(weight, s.max_m)))
}

/// 2026-09-28: Whether [`nvfp4_proj_small_m`] launches the W4A4 kernels for this shape (and
/// otherwise the W4A16 `w4a16_gemv_batchm`).
pub fn routes_w4a4(gpu: &dyn GpuBackend, weight: &QuantizedWeight, m: u32, n: u32, k: u32) -> bool {
    w4a4_state_for(gpu, weight, m, n, k).is_some()
}

/// 2026-09-25: What the scratch holds: (backend, input address, m, k, stream).
type QuantKey = (usize, u64, u32, u32, u64);

fn last_quant() -> &'static Mutex<Option<QuantKey>> {
    static LAST: OnceLock<Mutex<Option<QuantKey>>> = OnceLock::new();
    LAST.get_or_init(|| Mutex::new(None))
}

#[allow(clippy::too_many_arguments)]
#[track_caller]
fn proj(
    gpu: &dyn GpuBackend,
    batch_kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
    same_input: bool,
) -> Result<()> {
    let tier = crate::layers::weight_quantization();
    if let Some(s) = w4a4_state_for(gpu, weight, m, n, k) {
        let p = W4a4Proj(s);
        let want: QuantKey = (key(gpu), input.0, m, k, stream);
        let mut last = last_quant().lock().unwrap_or_else(|p| p.into_inner());
        if !(same_input && *last == Some(want)) {
            p.quantize(gpu, input, p.scratch(), m, k, stream)?;
            *last = Some(want);
        }
        drop(last);
        p.gemv(gpu, p.scratch(), weight, output, m, n, k, stream)?;
        if audit_enabled()
            && !s.audit_ref.is_null()
            && (n as usize) <= AUDIT_MAX_N
            && !gpu.stream_is_capturing(stream)
        {
            let site = std::panic::Location::caller();
            super::w4a16_gemv_batchm(
                gpu,
                batch_kernel,
                input,
                weight,
                s.audit_ref,
                m,
                n,
                k,
                stream,
            )?;
            audit(
                gpu,
                output,
                s.audit_ref,
                (m * n) as usize,
                stream,
                site,
                n,
                k,
            )?;
        }
        return Ok(());
    }
    // 2026-09-28: Under `declared` the projections of one group (q/k/v, gate/up) may route
    // differently. When the first takes W4A16 it quantizes nothing, so a later W4A4 member
    // passing `same_input` must not match a key an earlier layer left: forget it.
    if !same_input && tier.tier() == metrale_config::WeightQuantization::Declared {
        *last_quant().lock().unwrap_or_else(|p| p.into_inner()) = None;
    }
    anyhow::ensure!(
        m <= W4A4_MAX_M,
        "w4a4: {m} rows reached the W4A16 fallback, which covers at most {W4A4_MAX_M}"
    );
    super::w4a16_gemv_batchm(gpu, batch_kernel, input, weight, output, m, n, k, stream)
}

/// 2026-09-25: Per-site relative-error accumulator for [`audit_enabled`].
#[allow(clippy::too_many_arguments)]
fn audit(
    gpu: &dyn GpuBackend,
    got: DevicePtr,
    reference: DevicePtr,
    elems: usize,
    stream: u64,
    site: &'static std::panic::Location<'static>,
    n: u32,
    k: u32,
) -> Result<()> {
    type Acc = std::collections::HashMap<String, (u64, f64, f64)>;
    static ACC: OnceLock<Mutex<Acc>> = OnceLock::new();
    gpu.synchronize(stream)?;
    let mut a = vec![0u8; elems * 2];
    let mut b = vec![0u8; elems * 2];
    gpu.copy_d2h(got, &mut a)?;
    gpu.copy_d2h(reference, &mut b)?;
    let bf = |x: &[u8]| f32::from_bits((u16::from_le_bytes([x[0], x[1]]) as u32) << 16) as f64;
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.chunks_exact(2).zip(b.chunks_exact(2)) {
        let (x, y) = (bf(x), bf(y));
        num += (x - y) * (x - y);
        den += y * y;
    }
    let rel = if den > 0.0 { (num / den).sqrt() } else { 0.0 };
    let key = format!("{}:{} N={n} K={k}", site.file(), site.line());
    let mut acc = ACC
        .get_or_init(|| Mutex::new(Acc::new()))
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let e = acc.entry(key.clone()).or_insert((0, 0.0, 0.0));
    e.0 += 1;
    e.1 += rel;
    e.2 = e.2.max(rel);
    if e.0.is_multiple_of(64) {
        tracing::info!(
            "W4A4_AUDIT site={key} samples={} mean_rel={:.5} max_rel={:.5}",
            e.0,
            e.1 / e.0 as f64,
            e.2
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "w4a4_proj_tests.rs"]
mod w4a4_proj_tests;
