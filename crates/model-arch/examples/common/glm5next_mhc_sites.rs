// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: The mHC sites the GLM-5.3 mHC byte gates run on: two synthetic ones and the
//! checkpoint's `hc_{attn,ffn}_{fn,base,scale}` from the `mhc_layer{L}.safetensors` packets, at
//! GLM-5.3's shape (hidden 4096, hc 4), plus the gates' small device helpers.
//!
//! Owner: model-arch examples (GLM-5.3 mHC kernels).
//! Invariants:
//! - A packet tensor of the wrong dtype or shape is an error, never a silent cast.

use anyhow::{Context, Result, ensure};
use half::bf16;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_mhc::Glm5NextMhcSiteWeights;

pub const H: usize = 4096;
pub const HC: usize = 4;
pub const MIX: usize = (2 + HC) * HC;
/// 2026-10-15: The packet layers (first, a dense-to-MoE edge, middle, last).
pub const LAYERS: [usize; 4] = [0, 3, 22, 44];

/// 2026-10-15: xorshift64, uniform in [-1, 1).
pub struct Rng(pub u64);
impl Rng {
    pub fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

pub fn up(gpu: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(b.len().max(16))?;
    gpu.copy_h2d(b, p)?;
    Ok(p)
}

pub fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn bf16b(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| bf16::from_f32(*x).to_le_bytes())
        .collect()
}

pub fn read(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    gpu.synchronize(0)?;
    let mut b = vec![0u8; n];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// 2026-10-15: One site's weights on the device: `hc_fn` (BF16 bytes), `hc_base`, `hc_scale`.
pub struct Site {
    pub name: String,
    pub hc_fn: Vec<u8>,
    pub hc_fn_d: DevicePtr,
    pub hc_scale: DevicePtr,
    pub hc_base: DevicePtr,
}

impl Site {
    pub fn new(
        gpu: &dyn GpuBackend,
        name: String,
        hc_fn: Vec<u8>,
        base: &[f32],
        scale: &[f32],
    ) -> Result<Self> {
        ensure!(hc_fn.len() == MIX * HC * H * 2 && base.len() == MIX && scale.len() == 3);
        Ok(Self {
            name,
            hc_fn_d: up(gpu, &hc_fn)?,
            hc_fn,
            hc_scale: up(gpu, &f32b(scale))?,
            hc_base: up(gpu, &f32b(base))?,
        })
    }

    /// 2026-10-15: The site with `mix` as its per-token scratch.
    pub fn weights(&self, mix: DevicePtr) -> Glm5NextMhcSiteWeights {
        Glm5NextMhcSiteWeights {
            hc_fn: self.hc_fn_d,
            hc_fn_bf16: true,
            hc_scale: self.hc_scale,
            hc_base: self.hc_base,
            mix,
            done: DevicePtr(0),
        }
    }
}

/// 2026-10-15: The checkpoint's two sites of `mhc_layer{L}.safetensors` (BF16 `fn`, F32 `base`
/// and `scale`); a wrong dtype or shape is an error.
pub fn packet_sites(gpu: &dyn GpuBackend, dir: &str, layer: usize) -> Result<Vec<Site>> {
    let path = format!("{dir}/mhc_layer{layer}.safetensors");
    let raw = std::fs::read(&path).with_context(|| format!("reading {path}"))?;
    let n = u64::from_le_bytes(raw[..8].try_into()?) as usize;
    let head: serde_json::Value = serde_json::from_slice(&raw[8..8 + n])?;
    let tensor = |name: &str, dtype: &str, shape: &[u64]| -> Result<&[u8]> {
        let t = &head[name];
        ensure!(t["dtype"] == dtype, "{path} {name}: dtype {}", t["dtype"]);
        ensure!(
            t["shape"] == serde_json::json!(shape),
            "{path} {name}: shape {}",
            t["shape"]
        );
        let (a, b) = (
            t["data_offsets"][0].as_u64().context("offset")? as usize,
            t["data_offsets"][1].as_u64().context("offset")? as usize,
        );
        Ok(&raw[8 + n + a..8 + n + b])
    };
    let f32s = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let mut out = Vec::new();
    for s in ["attn", "ffn"] {
        let hc_fn = tensor(
            &format!("hc_{s}_fn"),
            "BF16",
            &[MIX as u64, (HC * H) as u64],
        )?;
        let base = f32s(tensor(&format!("hc_{s}_base"), "F32", &[MIX as u64])?);
        let scale = f32s(tensor(&format!("hc_{s}_scale"), "F32", &[3])?);
        out.push(Site::new(
            gpu,
            format!("layer {layer} {s}"),
            hc_fn.to_vec(),
            &base,
            &scale,
        )?);
    }
    Ok(out)
}

/// 2026-10-15: A synthetic site: `hc_fn` uniform in `[-w, w)`, `hc_base` in `[-0.5, 0.5)`.
pub fn synthetic(gpu: &dyn GpuBackend, rng: &mut Rng, name: &str, w: f32) -> Result<Site> {
    let hc_fn: Vec<u8> = (0..MIX * HC * H)
        .flat_map(|_| bf16::from_f32(rng.unit() * w).to_le_bytes())
        .collect();
    let base: Vec<f32> = (0..MIX).map(|_| rng.unit() * 0.5).collect();
    Site::new(gpu, name.to_string(), hc_fn, &base, &[0.7, 1.3, 0.9])
}

/// 2026-10-15: The two synthetic sites, then both sites of every packet layer.
pub fn all_sites(gpu: &dyn GpuBackend, rng: &mut Rng, dir: &str) -> Result<Vec<Site>> {
    let mut sites = vec![
        synthetic(gpu, rng, "synthetic w 0.05", 0.05)?,
        synthetic(gpu, rng, "synthetic w 0.5", 0.5)?,
    ];
    for l in LAYERS {
        sites.extend(packet_sites(gpu, dir, l)?);
    }
    Ok(sites)
}
