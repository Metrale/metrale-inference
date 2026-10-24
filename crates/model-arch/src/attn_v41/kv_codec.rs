// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: DeepSeek-V4.1's attention caches in the formats the checkpoint's own inference
//! code declares, as kv-codec registry rows: every layer's window ring holds `mxfp8` rows (FP8
//! E4M3 values, an E8M0 scale per 32), the kv sources' latent cache `nvfp4` rows (E2M1, an
//! E4M3 scale per 16) and their index-key cache `mxfp4` rows (E2M1, an E8M0 scale per 32).
//! The registry's `LatentKvPlan` holds the choice and sizes the rows; this module binds the
//! declared plan, resolves the kernel that packs each cache's rows, and decodes rows on the
//! host.
//!
//! The reference quantizes these rows in place (quantize, then dequantize to bf16), so a
//! stored value is always `code * scale`. Each such product (e4m3 times a power of two, e2m1
//! times e4m3, e2m1 times a power of two) is exact in bf16 and f32: keeping the codes and
//! scales and decoding them where the rows are read gives the consumers the floats the bf16
//! storage gave them, at 288 + 68 bytes a compressed row instead of 1024 + 256, and 528 bytes
//! a window row instead of 1024. Two E2M1 codes share a byte, the lower-indexed value in the
//! low nibble (the checkpoint's FP4 packing).
//!
//! Owner: model-arch, DeepSeek-V4.1.
//! Invariants:
//! - Every size comes from the registry spec of the cache's codec.
//! - `decode_rows` is the host form of the device decode, used as its test oracle.

use anyhow::{Result, anyhow, bail, ensure};
use metrale_kv_codec::{HostLutId, KvCodecId, LatentCache, LatentKvPlan, by_name, spec};

use crate::deepseek_v41_ref::attn::FP8_BLOCK;
use crate::deepseek_v41_ref::mat::{e2m1, e4m3, e8m0};

/// 2026-10-19: The registry codec named `name`, bound at compile time.
const fn codec(name: &str) -> KvCodecId {
    match by_name(name) {
        Some(id) => id,
        None => panic!("the kv-codec registry lacks a DeepSeek-V4.1 cache codec"),
    }
}

/// 2026-10-19: The plan the checkpoint's inference code declares (`act_quant` on the window
/// rows, `fp4_act_quant(…, 16, …, e4m3)` on the latents, `fp4_act_quant(…, 32, …)` on the index
/// keys).
pub fn declared() -> Result<LatentKvPlan> {
    LatentKvPlan::new(codec("mxfp8"), codec("nvfp4"), codec("mxfp4")).map_err(|e| anyhow!("{e}"))
}

/// 2026-10-19: A stored element of a row: a value code or a group scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Elem {
    E2m1,
    E4m3,
    E8m0,
}

impl Elem {
    fn read(self, bytes: &[u8], i: usize) -> f32 {
        match self {
            Elem::E2m1 => e2m1(bytes[i / 2] >> (4 * (i % 2))),
            Elem::E4m3 => e4m3(bytes[i]),
            Elem::E8m0 => e8m0(bytes[i]),
        }
    }
}

/// 2026-10-19: The (value, scale) elements a host LUT reads; `None` for one these caches never
/// hold.
fn lut_elems(lut: Option<HostLutId>) -> Option<(Elem, Elem)> {
    match lut? {
        HostLutId::E4m3E8m0 => Some((Elem::E4m3, Elem::E8m0)),
        HostLutId::E2m1E4m3 => Some((Elem::E2m1, Elem::E4m3)),
        HostLutId::E2m1E8m0 => Some((Elem::E2m1, Elem::E8m0)),
        HostLutId::Copy
        | HostLutId::E4m3Tensor
        | HostLutId::Turbo4Lut
        | HostLutId::Turbo3Lut
        | HostLutId::E4m3Bf16Norm => None,
    }
}

/// 2026-10-19: The kernel that packs bf16 rows into one cache's stored rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowPack {
    /// 2026-10-19: `attn_v41_kv_pack_fp8`: E4M3 values, an E8M0 scale per `FP8_BLOCK`.
    Fp8E8m0,
    /// 2026-10-19: `attn_v41_kv_pack_fp4`: E2M1 values, one scale per `group`, E4M3 when
    /// `e4m3_scale`, else E8M0.
    Fp4 { group: u32, e4m3_scale: bool },
}

/// 2026-10-19: One cache's rows, resolved once at init: the packing kernel and the stored bytes
/// of a row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct V41Row {
    pub codec: KvCodecId,
    pub pack: RowPack,
    pub bytes: usize,
}

/// 2026-10-19: The three caches' rows at one head geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct V41Rows {
    pub window: V41Row,
    pub latent: V41Row,
    pub index_key: V41Row,
}

impl V41Rows {
    /// 2026-10-19: Resolves each cache's packing kernel and row bytes; fails for a codec no
    /// kernel here packs or a row the codec cannot lay out.
    pub fn resolve(plan: &LatentKvPlan, head_dim: usize, index_hd: usize) -> Result<Self> {
        let row = |cache: LatentCache, values: usize| -> Result<V41Row> {
            let id = plan.codec(cache);
            let s = spec(id);
            let pack = match lut_elems(s.host_dequant) {
                Some((Elem::E4m3, Elem::E8m0)) if s.group as usize == FP8_BLOCK => RowPack::Fp8E8m0,
                Some((Elem::E2m1, scale @ (Elem::E4m3 | Elem::E8m0))) => RowPack::Fp4 {
                    group: s.group,
                    e4m3_scale: scale == Elem::E4m3,
                },
                Some(_) | None => {
                    bail!("no packing kernel writes {} {} rows", cache.name(), s.name)
                }
            };
            let bytes = plan
                .row_bytes(cache, u32::try_from(values)?)
                .map_err(|e| anyhow!("{e}"))?;
            Ok(V41Row {
                codec: id,
                pack,
                bytes: usize::try_from(bytes)?,
            })
        };
        Ok(Self {
            window: row(LatentCache::Window, head_dim)?,
            latent: row(LatentCache::Latent, head_dim)?,
            index_key: row(LatentCache::IndexKey, index_hd)?,
        })
    }
}

/// 2026-10-19: The values of `rows` stored `codec` rows of `values` each, in f32.
pub fn decode_rows(codec: KvCodecId, bytes: &[u8], rows: usize, values: usize) -> Result<Vec<f32>> {
    let s = spec(codec);
    let elems = lut_elems(s.host_dequant)
        .ok_or_else(|| anyhow!("{} rows have no host decode here", s.name))?;
    let v = u64::try_from(values)?;
    let data = usize::try_from(s.data_bytes(v))?;
    let rb = data + usize::try_from(s.side_bytes(v))?;
    decode(
        bytes,
        rows,
        values,
        usize::try_from(s.group)?,
        data,
        rb,
        elems,
    )
}

/// 2026-10-19: [`decode_rows`] with the layout spelled out: `data` code bytes then the scales,
/// `rb` bytes a row, one scale per `group` values.
fn decode(
    bytes: &[u8],
    rows: usize,
    values: usize,
    group: usize,
    data: usize,
    rb: usize,
    (value, scale): (Elem, Elem),
) -> Result<Vec<f32>> {
    ensure!(
        group > 0 && values.is_multiple_of(group),
        "{values} values are not whole groups of {group}"
    );
    ensure!(
        bytes.len() >= rows * rb,
        "{} bytes for {rows} rows of {rb}",
        bytes.len()
    );
    let mut out = Vec::with_capacity(rows * values);
    for row in bytes.chunks_exact(rb).take(rows) {
        let (codes, scales) = row.split_at(data);
        out.extend((0..values).map(|d| value.read(codes, d) * scale.read(scales, d / group)));
    }
    Ok(out)
}

#[cfg(test)]
#[path = "kv_codec_tests.rs"]
mod tests;
