// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: GPU parity gate for the 128-group rotated KV codecs (turbo3_tcq, turbo2_tcq,
//! turbo1_tcq, turbo4_n128) against the bit-exact CPU reference `metrale-kv-tq128`.
//!
//! For each codec and head_dim 128, 256 and 512, over three paged sequences plus one token
//! with a negative slot:
//! - encode: the cache bytes `reshape_and_cache_*` writes, K and V, equal the reference's
//!   `encode_head` byte for byte over the whole cache. Slots never written keep a sentinel.
//! - dequant: `tq128_dequant_bf16_*` output equals `dequant_head_bf16` byte for byte.
//! - decode: `paged_decode_attn_*` is within `TOL` (max abs error over a row, relative to the
//!   row's max |ref|) of f64 attention over the reference's f32 dequant (`dequant_group`), which
//!   is what the kernel attends over. Negative controls: a wrong GQA mapping, and (codecs with a
//!   V factor) a V decode scale of 1.0, must both miss `TOL`. The error against attention over
//!   the bf16 dequant is printed but not gated: bf16-rounding a large-norm K row moves a
//!   cancelling score by O(1), so that reference is not accurate enough to gate on.
//! - prefill: `paged_prefill_attn_*` over a chunk at the end of the longest sequence, against
//!   causal f64 attention, with its own negative controls (`prefill.rs`).
//!
//! The rows include a zero row and a row of 1e-20 values (both code an all-zero group, which
//! makes every Viterbi step a tie), an outlier, a large-norm row and a constant row.
//!
//! Owner: kv-tq128 examples.
//! Invariants:
//! - Each compiled circuit target (`common/norm_fusion_fixture.rs` TARGETS) is run, since a
//!   target may shadow a `common/` kernel.
//!
//!   cargo run -p metrale-gpu-examples --release --example tq128_microtest \
//!       --features cuda,gpu-examples

#[path = "../common/norm_fusion_fixture.rs"]
mod fixture;
mod prefill;
mod reference;

use anyhow::{Context, Result, ensure};
use fixture::{SENTINEL, backends, bytes as bytes_bf16, download, upload};
use half::bf16;
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_kv_codec::{HeadDimClass, KvKernelKey, KvOp, KvPair, by_name};
use metrale_kv_tq128::codebooks::TURBO4_MID;
use metrale_kv_tq128::{GROUP, KvSide, Tq128Codec};
use reference::{
    BLOCK_SIZE, Geometry, KV_HEADS, MAX_BLOCKS_PER_SEQ, NUM_BLOCKS, Q_HEADS, Rng, SEQ_LENS, TOL,
    bf16_at, block_tables, cpu_dequant, cpu_dequant_f32, cpu_encode, first_diff, reference, rows,
    slots, worst_row_error,
};

/// 2026-10-18: The registry kernel of `op` for this codec, so the example names no kernel itself.
fn kernel(g: &dyn GpuBackend, codec: Tq128Codec, op: KvOp) -> Result<KernelHandle> {
    let id =
        by_name(codec.name()).with_context(|| format!("{} is not registered", codec.name()))?;
    let name = KvKernelKey::new(op, KvPair::symmetric(id), HeadDimClass::Model)
        .kernel_name()
        .with_context(|| format!("{} has no {op:?} kernel", codec.name()))?;
    g.kernel(name.module, name.func)
}

fn bytes_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn gpu_encode(
    g: &dyn GpuBackend,
    geo: &Geometry,
    side: KvSide,
    src: &[bf16],
    slots: &[i64],
) -> Result<Vec<u8>> {
    let n = NUM_BLOCKS * geo.block_stride();
    let cache = upload(g, &vec![SENTINEL; n])?;
    let src_p = upload(g, &bytes_bf16(src))?;
    let slot_p = upload(
        g,
        &slots
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;
    let book = upload(g, &bytes_f32(geo.codec.codebook(side)))?;
    let items = slots.len() * KV_HEADS * geo.head_dim / GROUP;
    let mut l = KernelLaunch::new(g, kernel(g, geo.codec, KvOp::Write)?)
        .grid([items.div_ceil(4) as u32, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(src_p)
        .arg_ptr(cache)
        .arg_ptr(slot_p)
        .arg_ptr(book);
    if geo.codec.trellis().is_none() {
        l = l.arg_ptr(upload(g, &bytes_f32(&TURBO4_MID))?);
    }
    l.arg_u32(slots.len() as u32)
        .arg_u32(KV_HEADS as u32)
        .arg_u32(geo.head_dim as u32)
        .arg_u32(BLOCK_SIZE as u32)
        .arg_u32(geo.width() as u32)
        .arg_u64(geo.block_stride() as u64)
        .launch(0)?;
    g.synchronize(0)?;
    download(g, cache, n)
}

fn gpu_dequant(
    g: &dyn GpuBackend,
    geo: &Geometry,
    side: KvSide,
    cache: &[u8],
    slots: &[i64],
) -> Result<Vec<u8>> {
    let n = slots.len() * geo.width() * 2;
    let out = upload(g, &vec![SENTINEL; n])?;
    let cache_p = upload(g, cache)?;
    let slot_p = upload(
        g,
        &slots
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;
    let book = upload(g, &bytes_f32(geo.codec.codebook(side)))?;
    let items = slots.len() * KV_HEADS * geo.head_dim / GROUP;
    KernelLaunch::new(g, kernel(g, geo.codec, KvOp::DequantBf16)?)
        .grid([items.div_ceil(4) as u32, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(cache_p)
        .arg_ptr(out)
        .arg_ptr(slot_p)
        .arg_ptr(book)
        .arg_u32(slots.len() as u32)
        .arg_u32(KV_HEADS as u32)
        .arg_u32(geo.head_dim as u32)
        .arg_u32(BLOCK_SIZE as u32)
        .arg_u32(geo.width() as u32)
        .arg_u64(geo.block_stride() as u64)
        .arg_f32(geo.codec.decode_scale(side))
        .launch(0)?;
    g.synchronize(0)?;
    download(g, out, n)
}

fn gpu_decode(
    g: &dyn GpuBackend,
    geo: &Geometry,
    q: &[bf16],
    k: &[u8],
    v: &[u8],
) -> Result<Vec<u8>> {
    let n = SEQ_LENS.len() * Q_HEADS * geo.head_dim * 2;
    let o = upload(g, &vec![SENTINEL; n])?;
    let lens: Vec<u8> = SEQ_LENS
        .iter()
        .flat_map(|&l| (l as i32).to_le_bytes())
        .collect();
    let tables: Vec<u8> = block_tables()
        .iter()
        .flat_map(|b| b.to_le_bytes())
        .collect();
    KernelLaunch::new(g, kernel(g, geo.codec, KvOp::DecodeAttn)?)
        .grid([Q_HEADS as u32, SEQ_LENS.len() as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(upload(g, &bytes_bf16(q))?)
        .arg_ptr(upload(g, k)?)
        .arg_ptr(upload(g, v)?)
        .arg_ptr(o)
        .arg_ptr(upload(g, &tables)?)
        .arg_ptr(upload(g, &lens)?)
        .arg_ptr(upload(g, &bytes_f32(geo.codec.codebook(KvSide::Key)))?)
        .arg_ptr(upload(g, &bytes_f32(geo.codec.codebook(KvSide::Value)))?)
        .arg_u32(MAX_BLOCKS_PER_SEQ as u32)
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

fn check(g: &dyn GpuBackend, geo: &Geometry, seed: u64) -> Result<bool> {
    let slots = slots();
    let tokens = slots.len();
    let mut rng = Rng(seed);
    let (k_src, v_src) = (
        rows(&mut rng, tokens, geo.width()),
        rows(&mut rng, tokens, geo.width()),
    );
    let q: Vec<bf16> = (0..SEQ_LENS.len() * Q_HEADS * geo.head_dim)
        .map(|_| bf16::from_f64(rng.normal()))
        .collect();
    let tag = format!("{:<11} hd={:<3}", geo.codec.name(), geo.head_dim);
    let mut ok = true;
    let mut caches = Vec::new();
    let mut deq = Vec::new();
    let mut deq_bf16 = Vec::new();
    for (side, src) in [(KvSide::Key, &k_src), (KvSide::Value, &v_src)] {
        let cpu = cpu_encode(geo, side, src, &slots)?;
        let gpu = gpu_encode(g, geo, side, src, &slots)?;
        let enc_diff = first_diff(&cpu, &gpu);
        let cpu_d = cpu_dequant(geo, side, &cpu, &slots)?;
        let gpu_d = gpu_dequant(g, geo, side, &cpu, &slots)?;
        let deq_diff = first_diff(&cpu_d, &gpu_d);
        println!(
            "{tag} {side:?}: encode identical={} ({} B)  dequant identical={} ({} B)",
            enc_diff.is_none(),
            cpu.len(),
            deq_diff.is_none(),
            cpu_d.len()
        );
        if let Some(i) = enc_diff {
            let bb = geo.codec.block_bytes();
            let (b0, n) = (
                i / bb * bb,
                cpu.iter().zip(&gpu).filter(|(x, y)| x != y).count(),
            );
            println!(
                "  FAIL encode: {n} bytes differ; first in block at byte {b0}: cpu {:02x?} gpu {:02x?}",
                &cpu[b0..b0 + 8],
                &gpu[b0..b0 + 8]
            );
            ok = false;
        }
        if let Some(i) = deq_diff {
            println!(
                "  FAIL dequant: first differing byte {i}: cpu {:#04x} gpu {:#04x}",
                cpu_d[i], gpu_d[i]
            );
            ok = false;
        }
        deq.push(cpu_dequant_f32(geo, side, &cpu, &slots)?);
        deq_bf16.push(
            cpu_d
                .chunks_exact(2)
                .map(|b| bf16_at(b, 0))
                .collect::<Vec<f64>>(),
        );
        caches.push(cpu);
    }
    let out = gpu_decode(g, geo, &q, &caches[0], &caches[1])?;
    let gqa = |h: usize| h / (Q_HEADS / KV_HEADS);
    let err = worst_row_error(geo, &out, &reference(geo, &q, &deq[0], &deq[1], gqa, 1.0));
    let err_bf16 = worst_row_error(
        geo,
        &out,
        &reference(geo, &q, &deq_bf16[0], &deq_bf16[1], gqa, 1.0),
    );
    let ctrl_gqa = worst_row_error(
        geo,
        &out,
        &reference(geo, &q, &deq[0], &deq[1], |h| h % KV_HEADS, 1.0),
    );
    let vs = f64::from(geo.codec.decode_scale(KvSide::Value));
    let ctrl_scale = (vs != 1.0).then(|| {
        worst_row_error(
            geo,
            &out,
            &reference(geo, &q, &deq[0], &deq[1], gqa, 1.0 / vs),
        )
    });
    println!(
        "{tag} decode: worst row err {err:.2e} (tol {TOL:.0e}; vs bf16 dequant {err_bf16:.2e})  [control gqa {ctrl_gqa:.2e}{}]",
        ctrl_scale.map_or(String::new(), |c| format!(", control v_scale=1 {c:.2e}"))
    );
    if err > TOL {
        println!("  FAIL decode: error exceeds the tolerance");
        ok = false;
    }
    ok &= prefill::check(g, geo, &mut rng, &caches, &deq, &tag)?;
    if ctrl_gqa <= TOL || ctrl_scale.is_some_and(|c| c <= TOL) {
        println!(
            "  FAIL control: a known-wrong reference passed, so the tolerance does not discriminate"
        );
        ok = false;
    }
    Ok(ok)
}

fn main() -> Result<()> {
    ensure!(
        TURBO4_MID.len() == 15,
        "turbo4_n128 kernels take 15 midpoints"
    );
    let mut all_ok = true;
    for (target, gpu) in backends()? {
        println!("== target {target}");
        let g: &dyn GpuBackend = &gpu;
        for codec in Tq128Codec::ALL {
            for head_dim in [128, 256, 512] {
                for seed in [0x5EED_0001u64, 0x5EED_0002] {
                    all_ok &= check(g, &Geometry { codec, head_dim }, seed)?;
                }
            }
        }
    }
    println!("\n{}", if all_ok { "PASS" } else { "FAIL" });
    ensure!(all_ok, "tq128 GPU parity failed");
    Ok(())
}
