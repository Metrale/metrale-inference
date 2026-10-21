// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: Routed experts of one GLM-5.3 MoE layer from the real checkpoint, rank 0's TP=3
//! slice (gate/up rows 0..704 over K 4096; down columns 0..704, a `_k64` K), as global-id pointer
//! tables over both block-scale forms: the plain E4M3 scales, and the compact ones
//! (`glm5next_mlp::scale_dict`) with a per-expert base-code table (`PLAIN` for an expert that keeps
//! its plain scales). For the `METRALE_GLM_MOE_SCALE_DICT` gates.
//!
//! Owner: gpu examples (GLM-5.3 MoE).
//! Invariants: an expert not loaded has null pointers (the kernels skip it).
#![allow(dead_code)]

use anyhow::Result;
use metrale_config::TpSlice;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_mlp::build_w4a4::{slice_nvfp4_cols, slice_nvfp4_rows};
use metrale_model_arch::glm5next_mlp::scale_dict::compact;

use crate::ckpt;
use crate::fixture::{up, up_f32};

pub const HID: usize = 4096;
pub const SLICE: usize = 704;
pub const EXPERTS: usize = 288;
pub const PLAIN: u32 = 0xFFFF_FFFF;

/// 2026-10-21: One projection's device tables over all `EXPERTS` ids.
pub struct Proj {
    pub packed: DevicePtr,
    pub plain: DevicePtr,
    /// 2026-10-21: Each expert's compact scales, or its plain ones where `base` is `PLAIN`.
    pub cs: DevicePtr,
    pub base: DevicePtr,
    pub s2: DevicePtr,
    /// 2026-10-21: Plain bytes (codes + scales) of the loaded experts.
    pub bytes: usize,
    /// 2026-10-21: Loaded tensors whose codes span more than 32 (they keep plain scales).
    pub fallback: usize,
}

/// 2026-10-21: A scalar F32 tensor.
pub fn f32_of(dir: &str, name: &str) -> Result<f32> {
    let t = ckpt::tensor_raw(dir, name)?;
    Ok(f32::from_le_bytes(t.bytes[..4].try_into()?))
}

fn table(g: &dyn GpuBackend, v: &[u64]) -> Result<DevicePtr> {
    up(
        g,
        &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
    )
}

/// 2026-10-21: Projection `name` of layer `l`, `experts` loaded; `forced` keeps its plain scales
/// in the compact tables (the per-expert fallback path on a tensor that would compact). `alias`:
/// every id `e` of the tables points at loaded expert `experts[e % experts.len()]`, so a routing
/// over all `EXPERTS` ids has weights everywhere (a prefill's tile structure on fewer loads).
pub fn proj(
    g: &dyn GpuBackend,
    dir: &str,
    l: usize,
    name: &str,
    experts: &[usize],
    forced: usize,
    alias: bool,
) -> Result<Proj> {
    let (mut packed, mut plain, mut cs) = (
        vec![0u64; EXPERTS],
        vec![0u64; EXPERTS],
        vec![0u64; EXPERTS],
    );
    let (mut base, mut s2) = (vec![PLAIN; EXPERTS], vec![0f32; EXPERTS]);
    let (mut bytes, mut fallback) = (0, 0);
    for &e in experts {
        let p = format!("model.language_model.layers.{l}.mlp.experts.{e}.{name}");
        let w = ckpt::tensor_raw(dir, &format!("{p}.weight"))?;
        let s = ckpt::tensor_raw(dir, &format!("{p}.weight_scale"))?;
        let (n, k) = (w.shape[0], w.shape[1] * 2);
        let cut = TpSlice {
            start: 0,
            len: SLICE,
        };
        let (q, sc, rows, groups) = if name == "down_proj" {
            let (q, sc) = slice_nvfp4_cols(&w.bytes, &s.bytes, n, k, cut)?;
            (q, sc, n, SLICE / 16)
        } else {
            let (q, sc) = slice_nvfp4_rows(&w.bytes, &s.bytes, n, k, cut)?;
            (q, sc, SLICE, k / 16)
        };
        bytes += q.len() + sc.len();
        packed[e] = up(g, &q)?.0;
        plain[e] = up(g, &sc)?.0;
        s2[e] = f32_of(dir, &format!("{p}.weight_scale_2"))?;
        match compact(&sc, rows, groups) {
            Some(c) if e != forced => {
                cs[e] = up(g, &c.bytes)?.0;
                base[e] = c.base4();
            }
            Some(_) => cs[e] = plain[e],
            None => {
                cs[e] = plain[e];
                fallback += 1;
            }
        }
    }
    if alias {
        for e in 0..EXPERTS {
            let src = experts[e % experts.len()];
            (packed[e], plain[e], cs[e], base[e], s2[e]) =
                (packed[src], plain[src], cs[src], base[src], s2[src]);
        }
    }
    Ok(Proj {
        packed: table(g, &packed)?,
        plain: table(g, &plain)?,
        cs: table(g, &cs)?,
        base: up(
            g,
            &base
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )?,
        s2: up_f32(g, &s2)?,
        bytes,
        fallback,
    })
}

/// 2026-10-21: Gate, up and down of layer `l` and its activation scales (gate/up, down).
pub struct Layer {
    pub gate: Proj,
    pub up: Proj,
    pub down: Proj,
    pub gs_gu: f32,
    pub gs_down: f32,
}

impl Layer {
    pub fn load(
        g: &dyn GpuBackend,
        dir: &str,
        l: usize,
        experts: &[usize],
        forced: usize,
        alias: bool,
    ) -> Result<Self> {
        let p = |name| proj(g, dir, l, name, experts, forced, alias);
        let pre = format!("model.language_model.layers.{l}.mlp.experts.{}", experts[0]);
        Ok(Self {
            gate: p("gate_proj")?,
            up: p("up_proj")?,
            down: p("down_proj")?,
            gs_gu: f32_of(dir, &format!("{pre}.gate_proj.input_scale"))?,
            gs_down: f32_of(dir, &format!("{pre}.down_proj.input_scale"))?,
        })
    }

    pub fn fallback(&self) -> usize {
        self.gate.fallback + self.up.fallback + self.down.fallback
    }
}
