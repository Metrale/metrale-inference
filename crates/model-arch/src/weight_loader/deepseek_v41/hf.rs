// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: [`V41Source`] over the safetensors checkpoint. Every tensor was deferred by the
//! checkpoint loader; each read takes only this rank's part ([`cut_of`], [`read_plan`]) by
//! positional read past the page cache, and uploads it in its stored precision: bf16 as bf16,
//! E4M3 with its E8M0 tile scales as [`ResidentMat::Fp8Block`], f32 widened on the host. The
//! engram projection is the one exception: no GEMM reads it in E4M3, so it is decoded to bf16
//! on the host, exactly (an E4M3 value times a power of two is a bf16 value).
//!
//! Owner: model-arch weight loader.
//! Invariants:
//! - An FP8 weight is admitted only with an E8M0 scale of one code per 32 x 32 tile.
//! - A bf16 tensor is uploaded once however often it is asked for (a router is read by its own
//!   layer and by the one before it).

use std::collections::HashMap;
use std::sync::Mutex;

use anyhow::{Context, Result, bail, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops::{FP8_BLOCK_TILE, ResidentMat};
use metrale_model_weights::weights::{DeferredTensor, WeightDtype, WeightStore};
use rayon::prelude::*;

use super::source::{EngramWkv, Format, T, V41Source, name, scale_of};
use super::tp_cut::{Cut, Local, cut_of, finish, read_plan};
use crate::deepseek_v41_ref::mat::{InputQuant, Mat, Store, Weight};

pub struct HfSource<'a> {
    store: &'a WeightStore,
    local: Local,
    bf16: Mutex<HashMap<String, DevicePtr>>,
}

/// 2026-10-19: `cut` of the deferred tensor `d`, read past the page cache.
pub fn read_cut(d: &DeferredTensor, cut: &Cut) -> Result<Vec<u8>> {
    let plan = read_plan(&d.shape, d.byte_size(), cut)?;
    finish(&plan, d.read_host_range_uncached(plan.start, plan.len)?)
}

/// 2026-10-19: A new device copy of `bytes`, named on the allocation ledger as the tensor `what`
/// held as `resident`, declared by the checkpoint as `declared`.
pub fn upload(
    gpu: &dyn GpuBackend,
    bytes: &[u8],
    what: &str,
    resident: &str,
    declared: &str,
) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len().max(16))?;
    gpu.copy_h2d(bytes, p)?;
    gpu.label_alloc(
        p,
        &format!("{what} resident={resident} declared={declared}"),
    );
    Ok(p)
}

/// 2026-10-19: A new device copy of the whole bf16 tensor `name`, deferred by the loader.
pub fn whole_bf16(gpu: &dyn GpuBackend, store: &WeightStore, name: &str) -> Result<DevicePtr> {
    let d = store
        .deferred(name)
        .with_context(|| format!("deepseek-v4.1: {name} was not deferred by the loader"))?;
    ensure!(
        d.dtype == WeightDtype::BF16,
        "{name}: expected bf16, got {:?}",
        d.dtype
    );
    upload(gpu, &read_cut(d, &Cut::Whole)?, name, "bf16", "bf16")
}

/// 2026-10-19: The E4M3 `codes` of a `[rows, cols]` weight times its E8M0 `scales` (one per
/// 32 x 32 tile, row-major over tiles) as little-endian bf16. Refuses the E8M0 NaN code.
pub fn decode_fp8_tiles(codes: &[u8], scales: &[u8], rows: usize, cols: usize) -> Result<Vec<u8>> {
    ensure!(
        !scales.contains(&0xFF),
        "an E8M0 scale holds the NaN code 0xFF"
    );
    let m = Mat::new(
        Store::E4m3 {
            w: codes,
            scale: scales,
            block_rows: FP8_BLOCK_TILE,
        },
        rows,
        cols,
        InputQuant::Exact,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut out = vec![0u8; rows * cols * 2];
    out.par_chunks_mut(cols * 2).enumerate().for_each(|(r, o)| {
        let mut v = vec![0f32; cols];
        m.decode_row(r, cols, &mut v);
        for (b, x) in o.chunks_exact_mut(2).zip(&v) {
            b.copy_from_slice(&half::bf16::from_f32(*x).to_le_bytes());
        }
    });
    Ok(out)
}

fn widen(dtype: WeightDtype, bytes: &[u8], what: &str) -> Result<Vec<f32>> {
    Ok(match dtype {
        WeightDtype::BF16 => bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect(),
        WeightDtype::FP32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        d => bail!("{what}: cannot widen {d:?} to f32"),
    })
}

impl<'a> HfSource<'a> {
    pub fn new(store: &'a WeightStore, local: Local) -> Self {
        HfSource {
            store,
            local,
            bf16: Mutex::new(HashMap::new()),
        }
    }

    fn tensor(&self, name: &str) -> Result<&DeferredTensor> {
        self.store
            .deferred(name)
            .with_context(|| format!("deepseek-v4.1: {name} was not deferred by the loader"))
    }

    /// 2026-10-19: The FP8 weight `n` cut by `cut`, its scale plane cut alike, and the kept
    /// `[rows, cols]`.
    fn fp8(&self, n: &str, cut: &Cut) -> Result<(Vec<u8>, Vec<u8>, usize, usize)> {
        let w = self.tensor(n)?;
        let sn = scale_of(n)?;
        let s = self.tensor(&sn)?;
        let tile = FP8_BLOCK_TILE;
        let [rows, cols] = w.shape[..] else {
            bail!("{n}: an FP8 weight is two-dimensional, got {:?}", w.shape)
        };
        ensure!(
            s.dtype == WeightDtype::FP8E8M0
                && rows.is_multiple_of(tile)
                && cols.is_multiple_of(tile)
                && s.shape == [rows / tile, cols / tile],
            "{n}: [{rows}, {cols}] needs an E8M0 [{}, {}] scale, {sn} is {:?} {:?}",
            rows / tile,
            cols / tile,
            s.dtype,
            s.shape
        );
        let (kr, kc) = match cut {
            Cut::Whole => (rows, cols),
            Cut::Rows(r) => (r.len(), cols),
            Cut::Cols(c) => (rows, c.len()),
        };
        Ok((read_cut(w, cut)?, read_cut(s, &cut.div(tile)?)?, kr, kc))
    }
}

impl V41Source for HfSource<'_> {
    fn has(&self, l: usize, t: T) -> bool {
        self.store
            .deferred(&name(Format::Safetensors, l, t))
            .is_some()
    }

    fn mat(&self, gpu: &dyn GpuBackend, l: usize, t: T) -> Result<ResidentMat> {
        let n = name(Format::Safetensors, l, t);
        let cut = cut_of(t, &self.local);
        match self.tensor(&n)?.dtype {
            WeightDtype::BF16 => Ok(ResidentMat::Bf16(upload(
                gpu,
                &read_cut(self.tensor(&n)?, &cut)?,
                &n,
                "bf16",
                "bf16",
            )?)),
            WeightDtype::FP8E4M3 => {
                let (w, s, _, _) = self.fp8(&n, &cut)?;
                Ok(ResidentMat::Fp8Block {
                    w: upload(gpu, &w, &n, "e4m3", "e4m3")?,
                    scale: upload(gpu, &s, &scale_of(&n)?, "e8m0", "e8m0")?,
                    // 2026-10-19: The reference rounds every FP8 projection's input to E4M3
                    // blocks except `wo_a`'s, which it reads widened.
                    qdq: t != T::WoA,
                })
            }
            d => bail!("{n}: expected bf16 or E4M3, got {d:?}"),
        }
    }

    fn bf16(&self, gpu: &dyn GpuBackend, l: usize, t: T) -> Result<DevicePtr> {
        let n = name(Format::Safetensors, l, t);
        let mut held = self.bf16.lock().unwrap();
        if let Some(&p) = held.get(&n) {
            return Ok(p);
        }
        let d = self.tensor(&n)?;
        ensure!(
            d.dtype == WeightDtype::BF16,
            "{n}: expected bf16, got {:?}",
            d.dtype
        );
        let p = upload(
            gpu,
            &read_cut(d, &cut_of(t, &self.local))?,
            &n,
            "bf16",
            "bf16",
        )?;
        held.insert(n, p);
        Ok(p)
    }

    fn host_f32(&self, _gpu: &dyn GpuBackend, l: usize, t: T) -> Result<Vec<f32>> {
        let n = name(Format::Safetensors, l, t);
        let d = self.tensor(&n)?;
        widen(d.dtype, &read_cut(d, &cut_of(t, &self.local))?, &n)
    }

    fn engram_wkv(&self, gpu: &dyn GpuBackend, l: usize) -> Result<EngramWkv> {
        let n = name(Format::Safetensors, l, T::EngramWkv);
        let (codes, scales, rows, cols) = self.fp8(&n, &Cut::Whole)?;
        let bits = decode_fp8_tiles(&codes, &scales, rows, cols).with_context(|| n.clone())?;
        Ok(EngramWkv {
            wkv: upload(gpu, &bits, &n, "bf16", "e4m3")?,
            wkv_q2k: DevicePtr(0),
            qdq_rows: true,
        })
    }
}

#[cfg(test)]
#[path = "hf_tests.rs"]
pub(super) mod tests;
