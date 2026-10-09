// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: `--dense-quantization fp8` for GLM-5.3: the BF16 dense projections (KDA q/k/v,
//! f_a/f_b, b, g_a/g_b, o; DSA q_a, absorbed q, kv_a, absorbed o; the indexer's wk and
//! compress gate; the shared expert's gate/up/down) served W8A8 at a precision BELOW the
//! checkpoint's declared BF16: FP8 E4M3 weights with one F32 scale per output channel,
//! quantized at load from the bound BF16 weights (`quantize_bf16_to_fp8`, max|row| / 448), and
//! dynamic per-token FP8 activations on the engine's W8A8 decode family (`ops::w8a8_decode`,
//! per-row layout). The FP32-output projections (the router, the indexer's wq_b and
//! weights_proj) stay BF16.
//!
//! The loader registers each projection's FP8 copy under its BF16 weight's device address; the
//! three GLM projection launchers (`glm5next_kda`, `glm5next_dsa::layer::proj_gemm`,
//! `glm5next_mlp::forward::launch`) ask [`proj`] first and run BF16 only when it declines.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - [`proj`] declines (`Ok(false)`) unless the tier is `fp8` and the weight was registered;
//!   with the tier at `declared` nothing is registered and every launch is unchanged.
//! - A registered weight is used only at the shape it was registered with; any other shape is
//!   an error, never a silent BF16 run.
//! - A row's output does not depend on the row count: the W8A8 family is row-invariant and
//!   launches here are chunked by whole rows.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use anyhow::{Result, bail, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::ops::{
    W8A8_MAX_ROWS, W8a8Kernels, W8a8Scratch, W8a8Weight, w8a8_gemv, w8a8_proj,
};
use metrale_model_layers::weight_map::{DenseWeight, Fp8Weight, WeightQuantFormat};

/// 2026-10-09: One backend's W8A8 kernels and activation scratch, and the registered weights by
/// BF16 address.
struct Fp8Dense {
    kernels: W8a8Kernels,
    scratch: W8a8Scratch,
    max_k: u32,
    weights: HashMap<u64, W8a8Weight>,
    /// 2026-10-09: What the scratch holds: `(input, rows, k, stream)` of the last quantization.
    quantized: Option<(u64, usize, usize, u64)>,
    /// 2026-10-09: The input a [`StableInput`] guard declares unchanged while it lives.
    stable: Option<u64>,
}

/// 2026-10-09: While alive, the caller promises that the BF16 rows at the guarded address do
/// not change, so consecutive projections of it reuse one FP8 quantization (KDA's q/k/v and gates
/// read the same hidden state; the shared expert's gate and up the same input). A projection of
/// another input re-quantizes into the scratch, after which the guarded input is quantized again
/// on its next use. Dropping the guard forgets the scratch's content. Under `declared` it does
/// nothing.
pub struct StableInput {
    active: bool,
}

/// 2026-10-09: Declare `a` unchanged until the guard drops (see [`StableInput`]).
pub fn stable_input(a: DevicePtr) -> StableInput {
    if !enabled() {
        return StableInput { active: false };
    }
    let mut g = state().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(s) = g.as_mut() {
        s.stable = Some(a.0);
    }
    StableInput { active: true }
}

impl Drop for StableInput {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let mut g = state().lock().unwrap_or_else(|e| e.into_inner());
        if let Some(s) = g.as_mut() {
            s.stable = None;
            s.quantized = None;
        }
    }
}

fn state() -> &'static Mutex<Option<Fp8Dense>> {
    static S: OnceLock<Mutex<Option<Fp8Dense>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(None))
}

/// 2026-10-09: Whether `--dense-quantization fp8` is in force.
pub fn enabled() -> bool {
    metrale_model_layers::layers::dense_quantization()
        == metrale_model_layers::layers::DenseQuantization::Fp8
}

/// 2026-10-09: The W8A8 family's K unit: the activation quantizer and the GEMV read K in
/// 128-wide chunks.
pub const FP8_K_UNIT: usize = 128;

/// 2026-10-09: The unit the shared expert's width splits over TP in: [`FP8_K_UNIT`] under the
/// fp8 tier, so each rank's down projection has a whole number of chunks (2048 over three ranks:
/// 768/640/640), else the BF16 kernels' `bf16_unit` (688/680/680).
pub fn shared_split_unit(bf16_unit: usize) -> usize {
    shared_split_unit_for(enabled(), bf16_unit)
}

fn shared_split_unit_for(fp8: bool, bf16_unit: usize) -> usize {
    if fp8 { FP8_K_UNIT } else { bf16_unit }
}

/// 2026-10-09: Resolve the W8A8 kernels and allocate the activation scratch for K up to
/// `max_k`, once, before the first [`register`]. Errors when the per-row W8A8 family is not in
/// this target's PTX: the operator asked for the tier, so it is never quietly dropped.
pub fn prepare(gpu: &dyn GpuBackend, max_k: usize) -> Result<()> {
    let mut g = state().lock().unwrap_or_else(|e| e.into_inner());
    if g.as_ref().is_some_and(|s| s.max_k as usize >= max_k) {
        return Ok(());
    }
    let kernels = W8a8Kernels::load(gpu);
    ensure!(
        kernels.resolved(metrale_model_layers::layers::ops::W8a8Scale::PerRow),
        "--dense-quantization fp8: the per-row W8A8 decode kernels (w8a8_gemv, w8a8_act_quant) \
         are not in this target's kernels"
    );
    let max_k = u32::try_from(max_k)?;
    let weights = g.take().map(|s| s.weights).unwrap_or_default();
    *g = Some(Fp8Dense {
        kernels,
        scratch: W8a8Scratch::alloc(gpu, max_k)?,
        max_k,
        weights,
        quantized: None,
        stable: None,
    });
    Ok(())
}

/// 2026-10-09: Quantize the BF16 `[n, k]` weight at `bf16` to FP8 per output channel and
/// register it. `quantize` is `gemv_fp8w::quantize_bf16_to_fp8`. Errors when K is not a
/// multiple of 128 (the W8A8 activation quantizer's unit) or exceeds the prepared scratch, or
/// when the address is registered at another shape.
pub fn register(
    gpu: &dyn GpuBackend,
    quantize: KernelHandle,
    bf16: DevicePtr,
    n: usize,
    k: usize,
    what: &str,
    stream: u64,
) -> Result<()> {
    let mut g = state().lock().unwrap_or_else(|e| e.into_inner());
    let Some(s) = g.as_mut() else {
        bail!("--dense-quantization fp8: {what} registered before prepare");
    };
    ensure!(
        k > 0 && k.is_multiple_of(FP8_K_UNIT) && k <= s.max_k as usize && n > 0,
        "--dense-quantization fp8: {what} is [{n}, {k}]; K must be a positive multiple of 128 \
         and at most {}",
        s.max_k
    );
    if let Some(w) = s.weights.get(&bf16.0) {
        ensure!(
            (w.n() as usize, w.k() as usize) == (n, k),
            "--dense-quantization fp8: {what} at {:#x} was registered as [{}, {}]",
            bf16.0,
            w.n(),
            w.k()
        );
        return Ok(());
    }
    let q = metrale_model_layers::weight_map::quantize_to_fp8(
        &DenseWeight { weight: bf16 },
        n,
        k,
        gpu,
        quantize,
        stream,
    )?;
    let w = W8a8Weight::new(&[Fp8Weight {
        weight: q.weight,
        row_scale: q.row_scale,
        n: n as u32,
        k: k as u32,
        scale_format: WeightQuantFormat::Fp8PerRow,
    }])?;
    s.weights.insert(bf16.0, w);
    Ok(())
}

/// 2026-10-09: Registered projections and their FP8 bytes (weights plus scales), for the log.
pub fn registered() -> (usize, usize) {
    let g = state().lock().unwrap_or_else(|e| e.into_inner());
    g.as_ref().map_or((0, 0), |s| {
        (
            s.weights.len(),
            s.weights
                .values()
                .map(|w| w.n() as usize * (w.k() as usize + 4))
                .sum(),
        )
    })
}

/// 2026-10-09: `c[m, n] = a[m, k] @ w[n, k]^T` (contiguous BF16 rows) on the FP8 copy of the
/// weight at `b`, in launches of at most `W8A8_MAX_ROWS` rows. `Ok(false)` when the tier is off
/// or `b` is not registered: the caller runs its BF16 path.
#[allow(clippy::too_many_arguments)]
pub fn proj(
    gpu: &dyn GpuBackend,
    b: DevicePtr,
    a: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<bool> {
    if !enabled() {
        return Ok(false);
    }
    proj_registered(gpu, b, a, c, m, n, k, stream)
}

/// 2026-10-09: [`proj`] whatever the published tier: run the registered FP8 copy of `b`, or
/// decline when there is none.
#[allow(clippy::too_many_arguments)]
fn proj_registered(
    gpu: &dyn GpuBackend,
    b: DevicePtr,
    a: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<bool> {
    let mut g = state().lock().unwrap_or_else(|e| e.into_inner());
    let Some(s) = g.as_mut() else {
        return Ok(false);
    };
    let Some(&w) = s.weights.get(&b.0) else {
        return Ok(false);
    };
    ensure!(
        (w.n() as usize, w.k() as usize) == (n, k),
        "--dense-quantization fp8: weight {:#x} registered as [{}, {}], launched as [{n}, {k}]",
        b.0,
        w.n(),
        w.k()
    );
    let key = (a.0, m, k, stream);
    if m <= W8A8_MAX_ROWS && s.stable == Some(a.0) && s.quantized == Some(key) {
        // 2026-10-09: The scratch already holds this stable input's quantization.
        w8a8_gemv(gpu, &s.kernels, &w, &s.scratch, m, c, n as u32, stream)?;
        return Ok(true);
    }
    let mut done = 0;
    while done < m {
        let rows = (m - done).min(W8A8_MAX_ROWS);
        w8a8_proj(
            gpu,
            &s.kernels,
            &w,
            a.offset(done * k * 2),
            k as u32,
            rows,
            c.offset(done * n * 2),
            n as u32,
            &s.scratch,
            stream,
        )?;
        done += rows;
    }
    // 2026-10-09: The scratch holds the last chunk only, which is all of it up to 256 rows.
    s.quantized = (m <= W8A8_MAX_ROWS).then_some(key);
    Ok(true)
}

#[cfg(test)]
#[path = "glm5next_fp8_dense_tests.rs"]
mod tests;
