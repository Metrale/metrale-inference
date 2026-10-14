// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The selective prefill attention kernels against a CPU reference on synthetic data:
//! `qsa_prefill_attn` (CUDA cores) and, where the target compiles it, `qsa_prefill_attn_tc`
//! (Hopper tensor cores). Covered: the GQA head mapping, paged addressing, duplicate and
//! out-of-order block lists, the tail tokens, key counts below one 64-key tile, across several
//! tiles with a partial last one, and at the model's topk (2048 keys + tail). The tensor-core
//! kernel must also write every output element, give the same bytes on a second run, and give a
//! row the same bytes whether it is launched alone or inside a slab.
//!
//! Owner: model-layers (QSA).
//! Invariants: none beyond the types.
//!
//! GPU tests: run with
//! ```text
//! cargo test -p metrale-model-layers --release --lib qsa_prefill_attn -- --ignored --nocapture
//! ```

#![allow(unused_imports)]

use super::*;

const NQ: usize = 24;
const NKV: usize = 2;
const HD: usize = 256;
const RATIO: usize = 4;
const BS: usize = 16;
const SENTINEL: u8 = 0xAB;

/// 2026-10-14: One synthetic launch: `rows` query rows from `first_pos`, `topk` blocks each.
struct Case {
    rows: usize,
    first_pos: usize,
    topk: usize,
}

/// 2026-10-14: Small: 32 keys + tail (< one tile), duplicate blocks. Mid: 160 keys + tail
/// (three tiles, the last partial). Model: topk 512 at the inert bound (2048 keys + tail).
const CASES: [Case; 3] = [
    Case {
        rows: 3,
        first_pos: 41,
        topk: 8,
    },
    Case {
        rows: 5,
        first_pos: 700,
        topk: 40,
    },
    Case {
        rows: 4,
        first_pos: 2052,
        topk: 512,
    },
];

/// 2026-10-14: Host copies and device buffers of one case.
struct Fixture {
    q: Vec<u16>,
    k: Vec<u16>,
    v: Vec<u16>,
    lists: Vec<i32>,
    d_q: DevicePtr,
    d_k: DevicePtr,
    d_v: DevicePtr,
    d_lists: DevicePtr,
    d_table: DevicePtr,
}

fn bytes16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes32(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn unbf(u: u16) -> f32 {
    f32::from_bits((u as u32) << 16)
}

fn fixture(g: &dyn GpuBackend, c: &Case) -> Fixture {
    let pages = (c.first_pos + c.rows).div_ceil(BS);
    let mut seed = 0x12345u32 ^ c.topk as u32;
    let mut nextbf = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let f = ((seed >> 8) as f32 / (1 << 24) as f32) - 0.5;
        (f.to_bits() >> 16) as u16
    };
    let q: Vec<u16> = (0..c.rows * NQ * HD).map(|_| nextbf()).collect();
    let kv = pages * BS * NKV * HD;
    let k: Vec<u16> = (0..kv).map(|_| nextbf()).collect();
    let v: Vec<u16> = (0..kv).map(|_| nextbf()).collect();
    // 2026-10-14: Block ids in a scrambled order (stride 7 mod n): a permutation of 0..topk, or
    // ids below 5 with repeats when topk < 16. All are below the first row's complete blocks.
    let modulus = if c.topk < 16 { 5 } else { c.topk };
    assert!(modulus < (c.first_pos + 1) / RATIO);
    let lists: Vec<i32> = (0..c.rows)
        .flat_map(|r| (0..c.topk).map(move |i| ((i * 7 + r * 3) % modulus) as i32))
        .collect();
    let table: Vec<i32> = (0..pages as i32).collect();
    Fixture {
        d_q: upload(g, &bytes16(&q)),
        d_k: upload(g, &bytes16(&k)),
        d_v: upload(g, &bytes16(&v)),
        d_lists: upload(g, &bytes32(&lists)),
        d_table: upload(g, &bytes32(&table)),
        q,
        k,
        v,
        lists,
    }
}

/// 2026-10-14: Rows [r0, r0 + n) of the case into a sentinel-filled buffer; its raw bytes.
fn launch(
    g: &dyn GpuBackend,
    kern: ops::QsaPrefillAttn,
    c: &Case,
    f: &Fixture,
    r0: usize,
    n: usize,
) -> Vec<u8> {
    let bytes = n * NQ * HD * 2;
    let out = g.alloc(bytes).unwrap();
    g.memset(out, SENTINEL, bytes).unwrap();
    let stream = g.default_stream();
    ops::qsa_prefill_attn(
        g,
        kern,
        f.d_q.offset(r0 * NQ * HD * 2),
        f.d_k,
        f.d_v,
        f.d_table,
        f.d_lists.offset(r0 * c.topk * 4),
        out,
        n as u32,
        (c.first_pos + r0) as u32,
        c.topk as u32,
        RATIO as u32,
        BS as u32,
        NQ as u32,
        NKV as u32,
        HD as u32,
        1.0 / (HD as f32).sqrt(),
        stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();
    let mut raw = vec![0u8; bytes];
    g.copy_d2h(out, &mut raw).unwrap();
    raw
}

/// 2026-10-14: The worst per-(row, head) cosine of `got` against an FP32 softmax attention over
/// the row's listed blocks (list order) and its incomplete tail block.
fn worst_cos(c: &Case, f: &Fixture, got: &[u8]) -> f64 {
    let got: Vec<f32> = got
        .chunks_exact(2)
        .map(|b| unbf(u16::from_le_bytes([b[0], b[1]])))
        .collect();
    let scale = 1.0 / (HD as f32).sqrt();
    let group = NQ / NKV;
    let mut worst = 1.0f64;
    for r in 0..c.rows {
        let pos = c.first_pos + r;
        let complete = (pos + 1) / RATIO;
        let tail = (pos + 1) - complete * RATIO;
        let mut toks: Vec<usize> = f.lists[r * c.topk..(r + 1) * c.topk]
            .iter()
            .flat_map(|&b| (0..RATIO).map(move |i| b as usize * RATIO + i))
            .collect();
        toks.extend(complete * RATIO..complete * RATIO + tail);
        for h in 0..NQ {
            let kvh = h / group;
            let qv = &f.q[(r * NQ + h) * HD..][..HD];
            let scores: Vec<f32> = toks
                .iter()
                .map(|&t| {
                    let kr = &f.k[(t * NKV + kvh) * HD..][..HD];
                    qv.iter()
                        .zip(kr)
                        .map(|(a, b)| unbf(*a) * unbf(*b))
                        .sum::<f32>()
                        * scale
                })
                .collect();
            let m = scores.iter().cloned().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
            let l: f32 = exps.iter().sum();
            let mut want = vec![0.0f32; HD];
            for (i, &t) in toks.iter().enumerate() {
                let vr = &f.v[(t * NKV + kvh) * HD..][..HD];
                for (w, x) in want.iter_mut().zip(vr) {
                    *w += exps[i] / l * unbf(*x);
                }
            }
            let gv = &got[(r * NQ + h) * HD..][..HD];
            let dot: f64 = gv
                .iter()
                .zip(&want)
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum();
            let ng = gv.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            let nr = want.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            worst = worst.min(dot / (ng * nr).max(1e-30));
        }
    }
    worst
}

/// 2026-10-14: Both kernels against the CPU reference; the tensor-core kernel's coverage,
/// rerun determinism and slab invariance. Fails when the target has no tensor-core kernel only
/// if it is built for Hopper (`METRALE_EXPECT_HOPPER=1`), so the base check still runs elsewhere.
#[test]
#[ignore]
fn qsa_prefill_attn_kernels_match_cpu() {
    let set = metrale_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with METRALE_TARGET_MODEL='*'");
    let gpu = metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
        .expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let base = g.kernel("qsa_indexer", "qsa_prefill_attn").unwrap();
    let tc = ops::QsaPrefillAttn::resolve_tc(g);
    if std::env::var("METRALE_EXPECT_HOPPER").as_deref() == Ok("1") {
        assert_ne!(
            tc.0, 0,
            "qsa_prefill_attn_tc not resolved on a Hopper build"
        );
    }
    let tc_kern = ops::QsaPrefillAttn::select(base, tc, NQ as u32, NKV as u32, HD as u32);
    for c in &CASES {
        let f = fixture(g, c);
        let base_out = launch(g, ops::QsaPrefillAttn::Base(base), c, &f, 0, c.rows);
        let cos = worst_cos(c, &f, &base_out);
        println!("topk {}: base worst cos = {cos:.9}", c.topk);
        assert!(
            cos > 0.999,
            "base kernel diverges at topk {}: {cos}",
            c.topk
        );
        let ops::QsaPrefillAttn::Tc(_) = tc_kern else {
            println!("topk {}: no tensor-core kernel on this target", c.topk);
            continue;
        };
        let out = launch(g, tc_kern, c, &f, 0, c.rows);
        let unwritten = out
            .chunks_exact(2)
            .filter(|b| b.iter().all(|&x| x == SENTINEL))
            .count();
        assert_eq!(unwritten, 0, "topk {}: unwritten elements", c.topk);
        let cos = worst_cos(c, &f, &out);
        println!("topk {}: tc worst cos = {cos:.9}", c.topk);
        assert!(
            cos > 0.999,
            "tensor-core kernel diverges at topk {}: {cos}",
            c.topk
        );
        assert_eq!(
            launch(g, tc_kern, c, &f, 0, c.rows),
            out,
            "topk {}: rerun",
            c.topk
        );
        let row = NQ * HD * 2;
        for r in 0..c.rows {
            let alone = launch(g, tc_kern, c, &f, r, 1);
            assert_eq!(
                alone,
                out[r * row..][..row],
                "topk {}: row {r} alone",
                c.topk
            );
        }
    }
}

/// 2026-10-14: The admission of the tensor-core kernel: only the shapes it does not trap on.
#[test]
fn qsa_prefill_attn_tc_admission() {
    use metrale_gpu_runtime::gpu::KernelHandle;
    let (b, t) = (KernelHandle(1), KernelHandle(2));
    let picks =
        |nq, nkv, hd, tc: KernelHandle| match ops::QsaPrefillAttn::select(b, tc, nq, nkv, hd) {
            ops::QsaPrefillAttn::Tc(h) => h.0,
            ops::QsaPrefillAttn::Base(h) => h.0,
        };
    assert_eq!(picks(24, 2, 256, t), 2, "model shape");
    assert_eq!(picks(32, 2, 256, t), 2, "group 16");
    assert_eq!(picks(2, 2, 256, t), 2, "group 1");
    assert_eq!(picks(24, 2, 256, KernelHandle(0)), 1, "unresolved");
    assert_eq!(picks(24, 2, 128, t), 1, "hd");
    assert_eq!(picks(34, 2, 256, t), 1, "group 17");
    assert_eq!(picks(25, 2, 256, t), 1, "nq % nkv");
    assert_eq!(picks(24, 0, 256, t), 1, "nkv 0");
}

/// 2026-10-14: Dense prefill keeps exactly the rows before the inert bound: a chunk ending at or
/// before it keeps all, one straddling it keeps the prefix, one starting at or past it keeps none.
#[test]
fn qsa_dense_rows_stop_at_the_inert_bound() {
    use super::dense_rows_before as rows;
    let b = 2051;
    assert_eq!(rows(b, 0, 2051, false), 2051, "ends at the bound");
    assert_eq!(rows(b, 0, 4096, false), 2051, "first 32k chunk");
    assert_eq!(rows(b, 2000, 100, false), 51, "straddles");
    assert_eq!(rows(b, 2051, 4096, false), 0, "starts at the bound");
    assert_eq!(rows(b, 4096, 4096, false), 0, "later chunk");
    assert_eq!(rows(b, 0, 0, false), 0, "empty chunk");
    assert_eq!(rows(b, 4096, 4096, true), 4096, "lever off");
}
