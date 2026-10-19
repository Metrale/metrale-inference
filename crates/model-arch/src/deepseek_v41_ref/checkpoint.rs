// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: The reference's weights from named checkpoint tensors, in their stored
//! precision. The tensors arrive through [`TensorSource`] as borrowed bytes; whoever opens the
//! files (an mmap host, or the golden fixture in tests) implements it.
//!
//! Owner: model-arch, DeepSeek-V4.1 reference.
//! Invariants:
//! - Names are the publisher's (`layers.{l}.attn.wq_a.weight`, its `.scale` beside it).
//! - A projection's input treatment follows the publisher's `linear()`: an fp8 or fp4 weight
//!   quantizes its input, a bf16 or f32 one does not. `wo_a` is the one exception: its
//!   converter widens it to bf16, so it is declared `Exact` whatever its storage, and the
//!   attention's own `o_act_quant` does the QDQ.
//! - Every shape is checked against the config; a mismatch is an error, never a reshape.

use metrale_error::{Error, ErrorCode, error};

use super::engram::EngramTables;
use super::mat::{InputQuant, MX_BLOCK, Mat, Store};
use super::model::{EngramWeights, LayerWeights, ModelCfg, ModelWeights};

/// 2026-10-19: The safetensors dtypes the checkpoint uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RawDtype {
    Bf16,
    F32,
    F8E4m3,
    F8E8m0,
    /// fp4 e2m1, two codes per byte, low nibble first.
    I8,
}

pub struct RawTensor<'a> {
    pub dtype: RawDtype,
    pub shape: Vec<u64>,
    pub bytes: &'a [u8],
}

/// 2026-10-19: Named tensors as borrowed bytes; an absent name is an error.
pub trait TensorSource {
    fn tensor(&self, name: &str) -> Result<RawTensor<'_>, Error>;
}

fn refuse(name: &str, why: String) -> Error {
    error!(
        ErrorCode::InvalidArgument,
        "checkpoint tensor",
        format!("{name}: {why}")
    )
}

fn expect_shape(name: &str, t: &RawTensor<'_>, want: &[usize]) -> Result<(), Error> {
    let want: Vec<u64> = want.iter().map(|&d| d as u64).collect();
    if t.shape == want {
        Ok(())
    } else {
        Err(refuse(
            name,
            format!("shape {:?}, expected {want:?}", t.shape),
        ))
    }
}

/// 2026-10-19: A bf16 or f32 tensor of `len` elements (any shape), as f32 values.
fn vector<S: TensorSource>(src: &S, name: &str, len: usize) -> Result<Vec<f32>, Error> {
    let t = src.tensor(name)?;
    let n: u64 = t.shape.iter().product();
    if n != len as u64 {
        return Err(refuse(name, format!("{n} elements, expected {len}")));
    }
    let width = match t.dtype {
        RawDtype::Bf16 => 2,
        RawDtype::F32 => 4,
        other => {
            return Err(refuse(
                name,
                format!("{other:?} where bf16 or f32 is expected"),
            ));
        }
    };
    if t.bytes.len() != len * width {
        return Err(refuse(
            name,
            format!("{} bytes for {len} elements", t.bytes.len()),
        ));
    }
    Ok(match t.dtype {
        RawDtype::Bf16 => t
            .bytes
            .chunks_exact(2)
            .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
            .collect(),
        _ => t
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
    })
}

/// 2026-10-19: How a weight's input is treated: by the publisher's `linear()` rule, or
/// declared exact (the widened `wo_a`, and tables that are only gathered).
#[derive(Clone, Copy)]
enum Site {
    Linear,
    Widened,
}

/// 2026-10-19: The `[rows, cols]` weight `{base}.weight` (with `{base}.scale` when quantized).
/// `scale_rows` is the row extent of one e4m3 scale (32 for the 32x32 tiles, 1 for the
/// Engram table's per-row blocks).
fn matrix<'s, S: TensorSource>(
    src: &'s S,
    base: &str,
    rows: usize,
    cols: usize,
    scale_rows: usize,
    site: Site,
) -> Result<Mat<'s>, Error> {
    let name = format!("{base}.weight");
    let w = src.tensor(&name)?;
    let scale = |want: [usize; 2]| -> Result<&'s [u8], Error> {
        let sname = format!("{base}.scale");
        let s = src.tensor(&sname)?;
        if s.dtype != RawDtype::F8E8m0 {
            return Err(refuse(
                &sname,
                format!("{:?} where an E8M0 scale is expected", s.dtype),
            ));
        }
        expect_shape(&sname, &s, &want)?;
        if let Some(&b) = s.bytes.iter().find(|&&b| b == 0xFF) {
            // 2026-10-19: Decoding panics on the NaN code; a checkpoint holding it is refused here.
            return Err(refuse(&sname, format!("holds the E8M0 NaN code {b:#x}")));
        }
        Ok(s.bytes)
    };
    let (store, quantized) = match w.dtype {
        RawDtype::Bf16 => {
            expect_shape(&name, &w, &[rows, cols])?;
            (Store::Bf16(w.bytes), false)
        }
        RawDtype::F32 => {
            expect_shape(&name, &w, &[rows, cols])?;
            (Store::F32(w.bytes), false)
        }
        RawDtype::F8E4m3 => {
            expect_shape(&name, &w, &[rows, cols])?;
            let tiles = [rows.div_ceil(scale_rows), cols.div_ceil(MX_BLOCK)];
            let store = Store::E4m3 {
                w: w.bytes,
                scale: scale(tiles)?,
                block_rows: scale_rows,
            };
            (store, true)
        }
        RawDtype::I8 => {
            expect_shape(&name, &w, &[rows, cols / 2])?;
            let store = Store::E2m1 {
                w: w.bytes,
                scale: scale([rows, cols / MX_BLOCK])?,
            };
            (store, true)
        }
        other => return Err(refuse(&name, format!("{other:?} is not a weight dtype"))),
    };
    let input = match (site, quantized) {
        (Site::Linear, true) => InputQuant::Fp8Block,
        _ => InputQuant::Exact,
    };
    Mat::new(store, rows, cols, input)
}

/// 2026-10-19: Every backbone weight of `c` from `src`. `t` names the Engram layers and their
/// table geometry.
pub fn model_weights<'s, S: TensorSource>(
    src: &'s S,
    c: &ModelCfg,
    t: &EngramTables,
) -> Result<ModelWeights<Mat<'s>>, Error> {
    let tile = MX_BLOCK;
    let lin = |base: &str, rows, cols| matrix(src, base, rows, cols, tile, Site::Linear);
    let (dim, hd, hc) = (c.dim, c.head_dim, c.hc);
    let mix = (2 + hc) * hc;
    let layers = (0..c.n_layers)
        .map(|l| -> Result<LayerWeights<Mat<'s>>, Error> {
            let n = |s: &str| format!("layers.{l}.{s}");
            let ac = c.attn_cfg(l);
            let (comp, idx) = (ac.is_kv_source, ac.is_index_source);
            let opt_lin =
                |on: bool, s: &str, rows, cols| on.then(|| lin(&n(s), rows, cols)).transpose();
            let opt_vec = |on: bool, s: &str, len| on.then(|| vector(src, &n(s), len)).transpose();
            let ffn = |s: &str, inter: usize| -> Result<(Mat<'s>, Mat<'s>, Mat<'s>), Error> {
                Ok((
                    lin(&n(&format!("{s}.w1")), inter, dim)?,
                    lin(&n(&format!("{s}.w2")), dim, inter)?,
                    lin(&n(&format!("{s}.w3")), inter, dim)?,
                ))
            };
            let engram = match t.layer_ids.iter().position(|&x| x == l) {
                None => None,
                Some(hi) => {
                    let in_f = t.n_hash_cols() * t.head_dim;
                    Some(EngramWeights {
                        table: matrix(
                            src,
                            &n("engram.embed"),
                            t.num_embeddings[hi],
                            t.head_dim,
                            1,
                            Site::Widened,
                        )?,
                        wkv: lin(&n("engram.wkv"), dim * (hc + 1), in_f)?,
                        q: vector(src, &n("engram.q_weight"), hc * dim)?,
                        k: vector(src, &n("engram.k_weight"), hc * dim)?,
                        hash_index: hi,
                    })
                }
            };
            let width = c.n_heads * hd / c.groups;
            Ok(LayerWeights {
                hc_attn_fn: vector(src, &n("hc_attn_fn"), mix * hc * dim)?,
                hc_attn_scale: vector(src, &n("hc_attn_scale"), 3)?,
                hc_attn_base: vector(src, &n("hc_attn_base"), mix)?,
                hc_ffn_fn: vector(src, &n("hc_ffn_fn"), mix * hc * dim)?,
                hc_ffn_scale: vector(src, &n("hc_ffn_scale"), 3)?,
                hc_ffn_base: vector(src, &n("hc_ffn_base"), mix)?,
                attn_norm: vector(src, &n("attn_norm.weight"), dim)?,
                ffn_norm: vector(src, &n("ffn_norm.weight"), dim)?,
                sink: vector(src, &n("attn.attn_sink"), c.n_heads)?,
                wq_a: lin(&n("attn.wq_a"), c.q_rank, dim)?,
                q_norm: vector(src, &n("attn.q_norm.weight"), c.q_rank)?,
                wq_b: lin(&n("attn.wq_b"), c.n_heads * hd, c.q_rank)?,
                wkv: lin(&n("attn.wkv"), hd, dim)?,
                kv_norm: vector(src, &n("attn.kv_norm.weight"), hd)?,
                wo_a: matrix(
                    src,
                    &n("attn.wo_a"),
                    c.groups * c.o_rank,
                    width,
                    tile,
                    Site::Widened,
                )?,
                wo_b: lin(&n("attn.wo_b"), dim, c.groups * c.o_rank)?,
                comp_wkv: opt_lin(comp, "attn.compressor.wkv", hd, dim)?,
                comp_wgate: opt_lin(comp && ac.ratio > 1, "attn.compressor.wgate", hd, dim)?,
                comp_norm: opt_vec(comp, "attn.compressor.norm.weight", hd)?,
                idx_wq_b: opt_lin(
                    idx,
                    "attn.indexer.wq_b",
                    c.index_heads * c.index_hd,
                    c.q_rank,
                )?,
                idx_weights_proj: opt_lin(idx, "attn.indexer.weights_proj", c.index_heads, dim)?,
                idx_wk: opt_lin(idx && comp, "attn.indexer.wk", c.index_hd, hd)?,
                idx_k_norm: opt_vec(idx && comp, "attn.indexer.k_norm.weight", c.index_hd)?,
                gate_w: lin(&n("ffn.gate"), c.moe.n_routed, dim)?,
                gate_bias: vector(src, &n("ffn.gate.bias"), c.moe.n_routed)?,
                experts: (0..c.moe.n_routed)
                    .map(|e| ffn(&format!("ffn.experts.{e}"), c.moe.inter))
                    .collect::<Result<_, _>>()?,
                shared: ffn("ffn.shared_experts", c.moe.inter)?,
                engram,
            })
        })
        .collect::<Result<_, _>>()?;
    Ok(ModelWeights {
        embed: matrix(src, "embed", c.vocab, dim, tile, Site::Widened)?,
        norm: vector(src, "norm.weight", dim)?,
        head: lin("head", c.vocab, dim)?,
        layers,
    })
}

#[cfg(test)]
#[path = "checkpoint_tests.rs"]
mod tests;
