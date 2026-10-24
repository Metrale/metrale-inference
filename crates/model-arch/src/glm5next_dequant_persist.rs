// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `METRALE_GLM_DEQUANT_PERSIST=kda|w4a16|all`: keep a persistent BF16 copy of the
//! GLM-5.3 dense weights the dequantize + cuBLASLt prefill levers serve, made once at load by
//! the same dequantization kernel, so a prefill launch runs only the GEMM.
//! `METRALE_GLM_W4A16_DEQUANT_GEMM_MIN_ROWS` (`glm5next_w4a16_wide`) and
//! `METRALE_GLM_FP8_DEQUANT_GEMM_MIN_ROWS` (`glm5next_fp8_wide`) otherwise dequantize the whole
//! weight into the shared scratch on every launch of their rows.
//!
//! `kda`: the KDA projections of the W4A16 tier (q, k, v, o, f_a, g_a, b); `w4a16`: every
//! projection of the W4A16 tier (also the shared expert's); `all`: also every projection of the
//! FP8 registry. Unset or `0` keeps no copy.
//!
//! Numerics: none change. The copy holds the bytes the per-launch dequantization writes into the
//! scratch (same kernel, same inputs, a pure element-wise map), and the GEMM is the same cuBLASLt
//! call on the same shapes, so every output byte equals the per-launch path's (byte-gated by the
//! `glm5next_dequant_persist_gate` example at every site shape).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Copies are made at load only (`dense_tiers::register_projections`), each its own
//!   allocation, finished (stream synchronized) before the next load step; never during a
//!   forward. A launch whose weight has no copy keeps the scratch path.
//! - A value is refused at load unless the lever that reads the copies is on:
//!   `METRALE_GLM_W4A16_DEQUANT_GEMM_MIN_ROWS` for every value, also
//!   `METRALE_GLM_FP8_DEQUANT_GEMM_MIN_ROWS` for `all`; and refused with
//!   `METRALE_GLM_W4A16_TC_GEMM_MIN_ROWS`, which takes those rows first.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::weight_map::{Fp8Weight, QuantizedWeight};

/// 2026-10-15: The lever's name.
pub const ENV: &str = "METRALE_GLM_DEQUANT_PERSIST";

/// 2026-10-15: Which projections keep a persistent BF16 copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Persist {
    Off,
    Kda,
    W4a16,
    All,
}

impl Persist {
    /// 2026-10-15: The lever value.
    pub fn name(self) -> &'static str {
        match self {
            Persist::Off => "off",
            Persist::Kda => "kda",
            Persist::W4a16 => "w4a16",
            Persist::All => "all",
        }
    }

    /// 2026-10-15: Whether the projection the loader names `name` keeps a copy; `fp8` when the
    /// FP8 registry serves it, else the W4A16 tier does.
    pub fn keeps(self, fp8: bool, name: &str) -> bool {
        match self {
            Persist::Off => false,
            Persist::Kda => !fp8 && name.starts_with("kda."),
            Persist::W4a16 => !fp8,
            Persist::All => true,
        }
    }
}

/// 2026-10-15: The lever's parse, pure (see the module invariants for what is refused): the value
/// and whether the W4A16 dequant GEMM, the W4A16 full-M tensor-core GEMM and the FP8 dequant GEMM
/// levers are on. 2026-10-17: resolved once into `DenseLevers::persist`.
pub(crate) fn parse(
    v: Option<&str>,
    w4a16_wide: bool,
    w4a16_tc: bool,
    fp8_wide: bool,
) -> Result<Persist, String> {
    let p = match v {
        None | Some("0") => return Ok(Persist::Off),
        Some("kda") => Persist::Kda,
        Some("w4a16") => Persist::W4a16,
        Some("all") => Persist::All,
        Some(s) => return Err(format!("{ENV}={s:?}: expected 0, kda, w4a16 or all")),
    };
    let n = p.name();
    if !w4a16_wide {
        return Err(format!(
            "{ENV}={n} needs METRALE_GLM_W4A16_DEQUANT_GEMM_MIN_ROWS, whose launches read the copies"
        ));
    }
    if w4a16_tc {
        return Err(format!(
            "{ENV}={n} with METRALE_GLM_W4A16_TC_GEMM_MIN_ROWS: that path takes the rows first, \
             so the copies would never be read"
        ));
    }
    if p == Persist::All && !fp8_wide {
        return Err(format!(
            "{ENV}=all needs METRALE_GLM_FP8_DEQUANT_GEMM_MIN_ROWS, whose launches read the FP8 \
             copies"
        ));
    }
    Ok(p)
}

/// 2026-10-15: One copy: its address and the `[n, k]` it holds.
#[derive(Clone, Copy, Debug)]
struct Kept {
    ptr: DevicePtr,
    n: usize,
    k: usize,
}

/// 2026-10-15: The copies, keyed by the quantized weight's address (the NVFP4 packed codes or
/// the E4M3 bytes; distinct allocations, so the two tiers never share a key).
fn copies() -> &'static Mutex<HashMap<u64, Kept>> {
    static C: OnceLock<Mutex<HashMap<u64, Kept>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

/// 2026-10-15: Allocate an `[n, k]` BF16 copy, fill it with `fill`, wait for it, and record it
/// under `key`. A key recorded twice is an error.
fn insert(
    gpu: &dyn GpuBackend,
    key: DevicePtr,
    n: usize,
    k: usize,
    what: &str,
    fill: impl FnOnce(DevicePtr) -> Result<()>,
) -> Result<()> {
    ensure!(
        !copies()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains_key(&key.0),
        "{ENV}: {what}'s weight {:#x} already has a copy",
        key.0
    );
    let ptr = gpu
        .alloc(n * k * 2)
        .with_context(|| format!("{ENV}: BF16 copy of {what} ({n} x {k})"))?;
    if let Err(e) = fill(ptr).and_then(|()| gpu.synchronize(0)) {
        gpu.free(ptr)?;
        return Err(e.context(format!("{ENV}: dequantizing {what}")));
    }
    copies()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(key.0, Kept { ptr, n, k });
    Ok(())
}

/// 2026-10-15: Keep a BF16 copy of the NVFP4 `[n, k]` weight `w` (packed codes and group
/// scales; the tensor scale stays the GEMM's alpha), dequantized as
/// `glm5next_w4a16_wide::dequant_gemm` does per launch.
pub fn keep_w4a16(
    gpu: &dyn GpuBackend,
    w: &QuantizedWeight,
    n: usize,
    k: usize,
    what: &str,
) -> Result<()> {
    insert(gpu, w.weight, n, k, what, |dst| {
        crate::glm5next_w4a16_wide::dequantize(gpu, w, dst, n * k, n, k, 0)
    })
}

/// 2026-10-15: Keep a BF16 copy of the per-row-scaled E4M3 weight `w`, dequantized as
/// `glm5next_fp8_wide::dequant_gemm` does per launch.
pub fn keep_fp8(gpu: &dyn GpuBackend, w: &Fp8Weight, what: &str) -> Result<()> {
    let (n, k) = (w.n as usize, w.k as usize);
    insert(gpu, w.weight, n, k, what, |dst| {
        crate::glm5next_fp8_wide::dequantize(gpu, w, dst, n * k, n, k, 0)
    })
}

/// 2026-10-15: The copy kept for the quantized weight at `weight`, launched as `[n, k]`; `None`
/// when it has none. A copy of another shape is an error.
pub fn copy_of(weight: DevicePtr, n: usize, k: usize) -> Result<Option<DevicePtr>> {
    let g = copies().lock().unwrap_or_else(|e| e.into_inner());
    let Some(c) = g.get(&weight.0) else {
        return Ok(None);
    };
    ensure!(
        (c.n, c.k) == (n, k),
        "{ENV}: the copy of {:#x} is [{}, {}], launched as [{n}, {k}]",
        weight.0,
        c.n,
        c.k
    );
    Ok(Some(c.ptr))
}

/// 2026-10-15: `(copies, bytes)` kept so far.
pub fn kept() -> (usize, usize) {
    let g = copies().lock().unwrap_or_else(|e| e.into_inner());
    (g.len(), g.values().map(|c| c.n * c.k * 2).sum())
}

/// 2026-10-15: The one-time load readback (nothing when the lever is off): the value, the
/// copies, and the device bytes they hold, under the resolved value `m`.
pub fn log_readback(m: Persist) {
    if m == Persist::Off {
        return;
    }
    let (count, bytes) = kept();
    tracing::info!(
        "glm dequant persist: {} copies={count} bytes={bytes} ({:.2} GB of persistent BF16; the \
         prefill GEMM is unchanged)",
        m.name(),
        bytes as f64 / 1e9
    );
}

#[cfg(test)]
#[path = "glm5next_dequant_persist_tests.rs"]
mod tests;
