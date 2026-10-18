// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: The kernel-level half of `dsa_fp8_indexer_gpu`: direct launches of the BF16
//! pool compression and the FP8 store / compression, the checks of an FP8 V pool against the
//! host twin, and the long-context gate (store segments against per-row stores, exact
//! quantization, top-k overlap of host scores over BF16 against FP8 pool keys).
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants:
//! - Every launch is on the backend's stream ([`stream`]); `read` synchronises it.
//! - A store launch of more than one row is one segment of one sequence (the kernel's store
//!   invariant): phase 1, then phase 2. A one-row launch is phase 3.

use anyhow::{Context, Result, bail, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_dsa::Glm5NextDsaKernels;
use metrale_model_arch::glm5next_dsa::fp8_rows::Fp8IndexerLayout;
use metrale_model_arch::glm5next_dsa::fp8_twin::{self, D};
use metrale_model_arch::glm5next_dsa::paged::{IndexerRowsDev, PagedIndexerLayout};

use super::rig::{Lcg, read, stream, up, up_i32};

const KP: usize = 4;
const BS: usize = 16;
/// 2026-10-16: The long-context gate's tokens: 8192 pools, four times the selection width.
pub(crate) const TOKENS: usize = 32_768;
const TOPK: usize = 2048;
const HEADS: usize = 32;
const QUERIES: usize = 4;
/// 2026-10-16: A prefill sub-chunk's rows; the first chunk is [`FIRST_CHUNK`] rows so every
/// later chunk starts inside a pool and its first completion reads ring rows.
const CHUNK: usize = 4096;
const FIRST_CHUNK: usize = 6;
/// 2026-10-16: Provisional: the synthetic-key overlap floor. The recorded-serve overlap is
/// measured by the GPU recipe, not here.
const OVERLAP_FLOOR: f64 = 0.90;

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// 2026-10-16: `dsa_kpool_compress` over the BF16 rows `rows` of a `len`-row sequence; the
/// keys of its `len / KP` complete pools, `[pool][D]`.
pub(crate) fn compress_bf16(
    gpu: &dyn GpuBackend,
    k: &Glm5NextDsaKernels,
    rows: &IndexerRowsDev,
    ape: DevicePtr,
    len: usize,
) -> Result<Vec<f32>> {
    ensure!(rows.fp8.is_none(), "compress_bf16 over FP8 rows");
    let p = len / KP;
    let keys = gpu.alloc(p.max(1) * D * 4)?;
    let idx = gpu.alloc(p.max(1) * KP * 4)?;
    let valid = gpu.alloc(p.max(1))?;
    if p > 0 {
        KernelLaunch::new(gpu, k.kpool_compress)
            .grid([p as u32, 1, 1])
            .block([D as u32, 1, 1])
            .arg_ptr(rows.k)
            .arg_ptr(rows.gate)
            .arg_ptr(rows.valid)
            .arg_ptr(ape)
            .arg_ptr(keys)
            .arg_ptr(idx)
            .arg_ptr(valid)
            .arg_u32(len as u32)
            .arg_u32(D as u32)
            .arg_u32(KP as u32)
            .arg_i32(0)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(rows.bt)
            .arg_u32(rows.block_size)
            .arg_u32(rows.blk_elems)
            .launch(stream(gpu))?;
    }
    let out = f32s(&read(gpu, keys, p * D * 4)?);
    for b in [keys, idx, valid] {
        gpu.free(b)?;
    }
    Ok(out)
}

/// 2026-10-16: `dsa_kpool_compress_fp8` over the FP8 rows `rows` of a `len`-row sequence.
pub(crate) fn compress_fp8(
    gpu: &dyn GpuBackend,
    k: &Glm5NextDsaKernels,
    rows: &IndexerRowsDev,
    len: usize,
) -> Result<Vec<f32>> {
    let f = rows.fp8.context("compress_fp8 over BF16 rows")?;
    let p = len / KP;
    let keys = gpu.alloc(p.max(1) * D * 4)?;
    let idx = gpu.alloc(p.max(1) * KP * 4)?;
    let valid = gpu.alloc(p.max(1))?;
    if p > 0 {
        KernelLaunch::new(gpu, k.fp8.compress)
            .grid([p as u32, 1, 1])
            .block([D as u32, 1, 1])
            .arg_ptr(rows.k)
            .arg_ptr(keys)
            .arg_ptr(idx)
            .arg_ptr(valid)
            .arg_u32(len as u32)
            .arg_u32(f.kp)
            .arg_u32(f.tail_sets)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(rows.bt)
            .arg_u32(rows.block_size)
            .arg_u32(f.blk_bytes)
            .launch(stream(gpu))?;
    }
    let out = f32s(&read(gpu, keys, p * D * 4)?);
    for b in [keys, idx, valid] {
        gpu.free(b)?;
    }
    Ok(out)
}

/// 2026-10-16: Checks an FP8 V pool (`pool`, host bytes) of a `len`-row sequence against the
/// BF16 cache of the same rows: every complete pool's entry and scale is the twin's
/// quantization of the BF16 compression's key (`bf16_keys`), bit for bit, and every open row
/// in the ring is the BF16 row (`bf16_row(x)` gives its key and gate bytes). Returns the
/// number of pools checked.
pub(crate) fn check_pool(
    what: &str,
    lay: &Fp8IndexerLayout,
    pool: &[u8],
    bt: &[u32],
    len: usize,
    bf16_keys: &[f32],
    bf16_row: &dyn Fn(usize) -> Result<(Vec<u8>, Vec<u8>)>,
) -> Result<usize> {
    let blk = |x: usize| bt[x / BS] as usize * lay.block_stride_bytes;
    let n = len / KP;
    for p in 0..n {
        let x = p * KP;
        let key: [f32; D] = bf16_keys[p * D..(p + 1) * D].try_into()?;
        let (q, sc) = fp8_twin::quantize_pool(&key);
        let e = blk(x) + lay.entry_off(x);
        let s = blk(x) + lay.scale_off(x);
        let got_sc = f32::from_le_bytes(pool[s..s + 4].try_into()?);
        if pool[e..e + D] != q || got_sc.to_bits() != sc.to_bits() {
            let at = (0..D).find(|&d| pool[e + d] != q[d]);
            bail!(
                "{what}: pool {p} entry differs from the twin (channel {at:?}, scale {got_sc} \
                 vs {sc})"
            );
        }
    }
    for x in n * KP..len {
        let (k, g) = bf16_row(x)?;
        let (ko, go) = (blk(x) + lay.ring_key_off(x), blk(x) + lay.ring_gate_off(x));
        if pool[ko..ko + D * 2] != k[..] || pool[go..go + D * 2] != g[..] {
            bail!("{what}: open row {x} in the ring differs from the BF16 row");
        }
    }
    Ok(n)
}

/// 2026-10-16: The twin's read-back of every complete pool of an FP8 V pool, `[pool][D]`.
fn dequantize_all(lay: &Fp8IndexerLayout, pool: &[u8], bt: &[u32], len: usize) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(len / KP * D);
    for p in 0..len / KP {
        let x = p * KP;
        let b = bt[x / BS] as usize * lay.block_stride_bytes;
        let (e, s) = (b + lay.entry_off(x), b + lay.scale_off(x));
        let q: [u8; D] = pool[e..e + D].try_into()?;
        out.extend(fp8_twin::dequantize_pool(
            &q,
            f32::from_le_bytes(pool[s..s + 4].try_into()?),
        ));
    }
    Ok(out)
}

/// 2026-10-16: `dsa_indexer_store_fp8` of staged rows `r0..r0 + n` (positions `r0..`) of one
/// sequence, one segment: phases 1 and 2, or phase 3 for one row.
#[allow(clippy::too_many_arguments)]
fn store(
    gpu: &dyn GpuBackend,
    k: &Glm5NextDsaKernels,
    lay: &Fp8IndexerLayout,
    v_pool: DevicePtr,
    stage: (DevicePtr, DevicePtr),
    pos: DevicePtr,
    bt: DevicePtr,
    ape: DevicePtr,
    r0: usize,
    n: usize,
) -> Result<()> {
    let phases: &[u32] = if n == 1 { &[3] } else { &[1, 2] };
    for &phase in phases {
        KernelLaunch::new(gpu, k.fp8.store)
            .grid([1, n as u32, 1])
            .block([D as u32, 1, 1])
            .arg_ptr(stage.0.offset(r0 * D * 2))
            .arg_ptr(stage.1.offset(r0 * D * 2))
            .arg_ptr(pos.offset(r0 * 4))
            .arg_ptr(v_pool)
            .arg_ptr(ape)
            .arg_u32(lay.kp as u32)
            .arg_u32(lay.tail_sets as u32)
            .arg_ptr(bt)
            .arg_u32(0)
            .arg_u32(lay.block_size as u32)
            .arg_u32(lay.block_stride_bytes as u32)
            .arg_u32(n as u32)
            .arg_u32(n as u32)
            .arg_u32(phase)
            .launch(stream(gpu))?;
    }
    Ok(())
}

/// 2026-10-16: Host scores of `keys` (`[pool][D]`) for query `qi`: the selector's
/// `sum_h w_h * relu(scale * q_h . key)`.
fn scores(q: &[f32], w: &[f32], keys: &[f32], qi: usize) -> Vec<f32> {
    let scale = 1.0 / (D as f32).sqrt();
    keys.chunks_exact(D)
        .map(|key| {
            (0..HEADS)
                .map(|h| {
                    let qh = &q[(qi * HEADS + h) * D..(qi * HEADS + h + 1) * D];
                    let dot: f32 = qh.iter().zip(key).map(|(a, b)| a * b).sum();
                    w[qi * HEADS + h] * (scale * dot).max(0.0)
                })
                .sum()
        })
        .collect()
}

/// 2026-10-16: The selector's top-k set: score descending, then pool ascending.
fn topk(s: &[f32]) -> std::collections::BTreeSet<usize> {
    let mut i: Vec<usize> = (0..s.len()).collect();
    i.sort_by(|&a, &b| s[b].total_cmp(&s[a]).then(a.cmp(&b)));
    i.into_iter().take(TOPK).collect()
}

fn overlap(a: &[f32], b: &[f32], q: &[f32], w: &[f32]) -> f64 {
    let mut sum = 0.0;
    for qi in 0..QUERIES {
        let (ta, tb) = (topk(&scores(q, w, a, qi)), topk(&scores(q, w, b, qi)));
        sum += ta.intersection(&tb).count() as f64 / TOPK as f64;
    }
    sum / QUERIES as f64
}

/// 2026-10-16: Random BF16 rows of `n` tokens in a BF16 paged V pool through `bt`; returns the
/// pool, its rows' addressing and the host key / gate bytes.
fn bf16_cache(
    gpu: &dyn GpuBackend,
    rng: &mut Lcg,
    bt: &[u32],
    bt_dev: DevicePtr,
    n: usize,
) -> Result<(DevicePtr, IndexerRowsDev, Vec<u8>, Vec<u8>)> {
    let lay = PagedIndexerLayout::new(BS, BS * D * 4, D)?;
    let (k, g) = (rng.bf16_bytes(n * D, 1.0), rng.bf16_bytes(n * D, 2.0));
    let blocks = *bt.iter().max().context("an empty block table")? as usize + 1;
    let mut host = vec![0u8; blocks * lay.block_stride_bytes];
    for x in 0..n {
        let o = lay.key_row_bytes(bt, x)?;
        host[o..o + D * 2].copy_from_slice(&k[x * D * 2..(x + 1) * D * 2]);
        let o = o + lay.gate_offset_bytes();
        host[o..o + D * 2].copy_from_slice(&g[x * D * 2..(x + 1) * D * 2]);
    }
    let pool = up(gpu, &host)?;
    Ok((pool, IndexerRowsDev::paged(&lay, pool, bt_dev)?, k, g))
}

/// 2026-10-16: Gate E: [`TOKENS`] random rows stored into an FP8 cache in prefill sub-chunks
/// and row by row; the two pools must be byte-identical, every entry the twin's quantization
/// of the BF16 compression, every FP8 compression key the twin's read-back; then the top-k
/// overlap of host scores, with a self control (1.0) and an unrelated-keys control.
pub(crate) fn long_context_gate(
    gpu: &dyn GpuBackend,
    k: &Glm5NextDsaKernels,
    rng: &mut Lcg,
) -> Result<()> {
    let n = TOKENS;
    let blocks = n / BS;
    // 2026-10-16: A scrambled table over twice the blocks (`i * 7 + 3 mod 2B` is a bijection).
    let bt: Vec<u32> = (0..blocks)
        .map(|b| ((b * 7 + 3) % (2 * blocks)) as u32)
        .collect();
    let bt_dev = up_i32(gpu, &bt.iter().map(|&b| b as i32).collect::<Vec<_>>())?;
    let ape: Vec<u8> = (0..KP * D)
        .flat_map(|_| (rng.next() * 0.1).to_le_bytes())
        .collect();
    let ape = up(gpu, &ape)?;
    let (bpool, brows, kh, gh) = bf16_cache(gpu, rng, &bt, bt_dev, n)?;
    let bkeys = compress_bf16(gpu, k, &brows, ape, n)?;

    let lay = Fp8IndexerLayout::new(BS, Fp8IndexerLayout::block_bytes(BS, D, KP, 1)?, D, KP, 1)?;
    let stage = (up(gpu, &kh)?, up(gpu, &gh)?);
    let pos = up_i32(gpu, &(0..n as i32).collect::<Vec<_>>())?;
    let bytes = 2 * blocks * lay.block_stride_bytes;
    let mut pools = Vec::new();
    for chunked in [true, false] {
        let v = gpu.alloc(bytes)?;
        gpu.memset(v, 0, bytes)?;
        let mut r0 = 0;
        while r0 < n {
            let m = match (chunked, r0) {
                (false, _) => 1,
                (true, 0) => FIRST_CHUNK,
                (true, _) => CHUNK.min(n - r0),
            };
            store(gpu, k, &lay, v, stage, pos, bt_dev, ape, r0, m)?;
            r0 += m;
        }
        pools.push((v, read(gpu, v, bytes)?));
    }
    if pools[0].1 != pools[1].1 {
        let at = pools[0].1.iter().zip(&pools[1].1).position(|(a, b)| a != b);
        bail!("E: the chunked store's FP8 pool differs from the per-row store's at byte {at:?}");
    }
    let row = |x: usize| -> Result<(Vec<u8>, Vec<u8>)> {
        Ok((
            kh[x * D * 2..(x + 1) * D * 2].to_vec(),
            gh[x * D * 2..(x + 1) * D * 2].to_vec(),
        ))
    };
    let checked = check_pool("E", &lay, &pools[0].1, &bt, n, &bkeys, &row)?;
    let frows = IndexerRowsDev::paged_fp8(&lay, pools[0].0, bt_dev)?;
    let fkeys = compress_fp8(gpu, k, &frows, n)?;
    let twin = dequantize_all(&lay, &pools[0].1, &bt, n)?;
    if fkeys
        .iter()
        .zip(&twin)
        .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        bail!("E: dsa_kpool_compress_fp8 differs from the twin's read-back");
    }
    let (mut worst, mut mean) = (0.0f64, 0.0f64);
    for (a, b) in bkeys.chunks_exact(D).zip(fkeys.chunks_exact(D)) {
        let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
        let den: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum();
        let rel = (num / den.max(f64::MIN_POSITIVE)).sqrt();
        worst = worst.max(rel);
        mean += rel / checked as f64;
    }
    let q: Vec<f32> = (0..QUERIES * HEADS * D).map(|_| rng.next()).collect();
    let w: Vec<f32> = (0..QUERIES * HEADS).map(|_| rng.next()).collect();
    let (_, other_rows, _, _) = bf16_cache(gpu, rng, &bt, bt_dev, n)?;
    let other = compress_bf16(gpu, k, &other_rows, ape, n)?;
    let control_self = overlap(&bkeys, &bkeys, &q, &w);
    let control_other = overlap(&bkeys, &other, &q, &w);
    let got = overlap(&bkeys, &fkeys, &q, &w);
    println!(
        "  E: {n} tokens, {checked} pools: chunked store == per-row store; entries == twin; \
         compress_fp8 == twin read-back"
    );
    println!("  E: pool key rel-L2 error mean {mean:.5} worst {worst:.5}");
    println!(
        "  E: top-{TOPK} of {checked} overlap bf16 vs fp8 {got:.4} (controls: self \
         {control_self:.4}, unrelated keys {control_other:.4})"
    );
    ensure!(
        control_self == 1.0 && control_other < 0.5,
        "E: the overlap controls are wrong (self {control_self}, unrelated {control_other})"
    );
    ensure!(
        got >= OVERLAP_FLOOR,
        "E: top-{TOPK} overlap {got:.4} is under the provisional floor {OVERLAP_FLOOR}"
    );
    for b in [
        bpool,
        other_rows.k,
        stage.0,
        stage.1,
        pos,
        ape,
        bt_dev,
        pools[0].0,
        pools[1].0,
    ] {
        gpu.free(b)?;
    }
    Ok(())
}
