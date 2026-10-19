// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The chunked-prefill leg of `tq128_microtest`: `paged_prefill_attn_*` over the
//! last sequence's cache, for a chunk of `CHUNK` tokens that starts after `SEQ_LENS[2] - CHUNK`
//! cached ones, against f64 causal attention over the reference's f32 dequant. Token t of the
//! chunk attends positions `[0, start + t]`.
//!
//! Negative controls, each of which must miss `TOL`: a mask that drops each token's own position
//! (an off-by-one causal bound), a mask that lets every token see the whole sequence (no causal
//! bound), and a wrong GQA mapping.
//!
//! Owner: kv-tq128 examples.
//! Invariants: the chunk's own rows are already in the cache, as the serve's write-before-attend
//! order guarantees.

use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_kv_codec::KvOp;
use metrale_kv_tq128::KvSide;

use super::fixture::{SENTINEL, bytes as bytes_bf16, download, upload};
use super::reference::{
    BLOCK_SIZE, Geometry, KV_HEADS, MAX_BLOCKS_PER_SEQ, Q_HEADS, Rng, SEQ_LENS, TOL, bf16_at,
    block_tables,
};
use super::{bytes_f32, kernel};

/// 2026-10-18: Tokens in the prefill chunk; spans a block boundary of the last sequence.
const CHUNK: usize = 13;
/// 2026-10-18: The sequence the chunk belongs to (the longest).
const SEQ: usize = 2;

/// 2026-10-18: f64 causal attention for the chunk: token t, head h attends the first
/// `len_of(t)` positions of `SEQ` through KV head `kv_of(h)`.
fn reference(
    geo: &Geometry,
    q: &[bf16],
    kd: &[f64],
    vd: &[f64],
    len_of: impl Fn(usize) -> usize,
    kv_of: fn(usize) -> usize,
) -> Vec<f64> {
    let d = geo.head_dim;
    let first: usize = SEQ_LENS[..SEQ].iter().sum();
    let mut out = vec![0f64; CHUNK * Q_HEADS * d];
    for t in 0..CHUNK {
        for h in 0..Q_HEADS {
            let kvh = kv_of(h);
            let qr: Vec<f64> = (0..d)
                .map(|i| f64::from(q[(t * Q_HEADS + h) * d + i].to_f32()))
                .collect();
            let at = |p: usize, i: usize| (first + p) * geo.width() + kvh * d + i;
            let scores: Vec<f64> = (0..len_of(t))
                .map(|p| (0..d).map(|i| qr[i] * kd[at(p, i)]).sum::<f64>() / (d as f64).sqrt())
                .collect();
            let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let w: Vec<f64> = scores.iter().map(|x| (x - m).exp()).collect();
            let z: f64 = w.iter().sum();
            for i in 0..d {
                let acc: f64 = w.iter().enumerate().map(|(p, w)| w * vd[at(p, i)]).sum();
                out[(t * Q_HEADS + h) * d + i] = acc / z;
            }
        }
    }
    out
}

/// 2026-10-18: The worst row's max |gpu - ref| / max |ref| over the chunk's rows.
fn worst_row_error(geo: &Geometry, gpu: &[u8], r: &[f64]) -> f64 {
    let d = geo.head_dim;
    (0..CHUNK * Q_HEADS)
        .map(|row| {
            let scale = (0..d)
                .map(|i| r[row * d + i].abs())
                .fold(0.0, f64::max)
                .max(1e-30);
            (0..d)
                .map(|i| (bf16_at(gpu, row * d + i) - r[row * d + i]).abs())
                .fold(0.0, f64::max)
                / scale
        })
        .fold(0.0, f64::max)
}

fn gpu_prefill(
    g: &dyn GpuBackend,
    geo: &Geometry,
    q: &[bf16],
    k: &[u8],
    v: &[u8],
    start: usize,
) -> Result<Vec<u8>> {
    let n = CHUNK * Q_HEADS * geo.head_dim * 2;
    let o = upload(g, &vec![SENTINEL; n])?;
    let table: Vec<u8> = block_tables()[SEQ * MAX_BLOCKS_PER_SEQ..(SEQ + 1) * MAX_BLOCKS_PER_SEQ]
        .iter()
        .flat_map(|b| b.to_le_bytes())
        .collect();
    KernelLaunch::new(g, kernel(g, geo.codec, KvOp::PrefillAttn)?)
        .grid([Q_HEADS as u32, CHUNK as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(upload(g, &bytes_bf16(q))?)
        .arg_ptr(upload(g, k)?)
        .arg_ptr(upload(g, v)?)
        .arg_ptr(o)
        .arg_ptr(upload(g, &table)?)
        .arg_ptr(upload(g, &bytes_f32(geo.codec.codebook(KvSide::Key)))?)
        .arg_ptr(upload(g, &bytes_f32(geo.codec.codebook(KvSide::Value)))?)
        .arg_u32(start as u32)
        .arg_u32(Q_HEADS as u32)
        .arg_u32(KV_HEADS as u32)
        .arg_u32(geo.head_dim as u32)
        .arg_u32(BLOCK_SIZE as u32)
        .arg_f32(1.0 / (geo.head_dim as f32).sqrt())
        .arg_f32(geo.codec.decode_scale(KvSide::Key))
        .arg_f32(geo.codec.decode_scale(KvSide::Value))
        .arg_u32((Q_HEADS * geo.head_dim) as u32)
        .arg_u64(geo.block_stride() as u64)
        .arg_u64(geo.block_stride() as u64)
        .launch(0)?;
    g.synchronize(0)?;
    download(g, o, n)
}

/// 2026-10-18: Runs the prefill leg over the encoded `caches` (K, V) and their f32 dequant
/// `deq`; prints one line and returns whether it passed.
pub fn check(
    g: &dyn GpuBackend,
    geo: &Geometry,
    rng: &mut Rng,
    caches: &[Vec<u8>],
    deq: &[Vec<f64>],
    tag: &str,
) -> Result<bool> {
    let start = SEQ_LENS[SEQ] - CHUNK;
    let q: Vec<bf16> = (0..CHUNK * Q_HEADS * geo.head_dim)
        .map(|_| bf16::from_f64(rng.normal()))
        .collect();
    let out = gpu_prefill(g, geo, &q, &caches[0], &caches[1], start)?;
    let gqa = |h: usize| h / (Q_HEADS / KV_HEADS);
    let causal = |t: usize| start + t + 1;
    let err = worst_row_error(
        geo,
        &out,
        &reference(geo, &q, &deq[0], &deq[1], causal, gqa),
    );
    let controls = [
        (
            "own position dropped",
            reference(geo, &q, &deq[0], &deq[1], |t| start + t, gqa),
        ),
        (
            "no causal bound",
            reference(geo, &q, &deq[0], &deq[1], |_| SEQ_LENS[SEQ], gqa),
        ),
        (
            "gqa",
            reference(geo, &q, &deq[0], &deq[1], causal, |h| h % KV_HEADS),
        ),
    ]
    .map(|(name, r)| (name, worst_row_error(geo, &out, &r)));
    let shown: Vec<String> = controls
        .iter()
        .map(|(n, e)| format!("{n} {e:.2e}"))
        .collect();
    println!(
        "{tag} prefill: worst row err {err:.2e} (tol {TOL:.0e}; chunk {CHUNK} from {start})  [controls {}]",
        shown.join(", ")
    );
    let mut ok = true;
    if err > TOL {
        println!("  FAIL prefill: error exceeds the tolerance");
        ok = false;
    }
    if controls.iter().any(|(_, e)| *e <= TOL) {
        println!("  FAIL prefill control: a known-wrong reference passed");
        ok = false;
    }
    Ok(ok)
}
