// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: Bit-identity gate and microbench of `METRALE_GLM_DSA_PROJ_IN_MERGED`: a DSA
//! layer's input projections at the TP=3 rank-0 serving shape under `--dsa-quantization w8a8`
//! (hidden 4096; q_a 1536, kv_a 512, indexer wk 128, compress gate 128; q_absorb 22 x 512 over
//! 1536), M = 1..8 rows. The ORACLE is today's sequence (`project_in` then
//! `indexer_project_rows` under one `stable_input` guard): quantize hidden, q_a, RMSNorm,
//! quantize q_resid, q_absorb, quantize hidden again, kv_a, wk, gate (the last two reuse kv_a's
//! quantization). The new sequence: quantize hidden, one `w8a8_gemv_rowscale_mb1_ku8_split4`
//! launch writing q_a, kv_a, wk and gate, RMSNorm, quantize q_resid, q_absorb.
//! At 2..=8 rows both then run the selector query (wq_b over q_resid, 4096 x 1536) and head
//! weights (weights_proj over hidden, 32 x 4096) at FP32 out: today two
//! `dense_gemv_bf16_batchm_few_fp32out` launches, new one `..._fp32out_pair`
//! (`METRALE_GLM_DSA_SELECTOR_PAIR`).
//!
//! Owner: gpu examples (GLM-5.3 DSA).
//! Invariants:
//! - Fails unless q_a, q_resid, q_abs, kv_a, wk, gate, q_idx and head weights are byte-identical for every M in
//!   `ROWS_TESTED`, every output poisoned before each run, the rows drawn from four families
//!   (drawn values, one large value, a zero row, a NaN).
//! - A negative control must DIFFER: the split launch with the wk and gate weights swapped.
//! - `DSA_PROJ_BASE_PTX=<dir>` (the `w8a8_gemv.ptx` and `dense_gemv_bf16_batchm_few.ptx` before
//!   the change): the oracle on those
//!   kernels must also match, so the policy refactor kept the existing entries' bytes.
//!
//!   cargo build -p metrale-gpu-examples --release --example glm5next_dsa_proj_in_gate \
//!       --features cuda,gpu-examples
//!
//! Timing: `DSA_LAYERS` layers with distinct weights per graph, PDL on (the serve's default),
//! each arm per M in `ROWS_TIMED` (graph_arms). `--no-timing` skips it.

#[path = "../common/glm5next_rows_fixture.rs"]
mod fixture;

use anyhow::{Result, bail};
use fixture::{Lcg, read, same, up, up_bf16, up_f32};
use metrale_gpu_examples::graph_arms::{Plan, time_arms};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_layers::layers::ops::{
    SplitOut, W8A8_SPLIT4_ENTRY, W8A8_SPLIT4_MODULE, W8a8Kernels, W8a8Scale, W8a8Scratch,
    W8a8Weight, w8a8_act_quant, w8a8_gemv, w8a8_gemv_split4, w8a8_proj,
};
use metrale_model_layers::weight_map::{Fp8Weight, WeightQuantFormat};

mod oracle;
use oracle::{kernels, oracle, outs};

const HID: usize = 4096;
const QA: usize = 1536;
const KV: usize = 512;
const IDX: usize = 128;
const QABS: usize = 22 * 512;
const DSA_LAYERS: usize = 11;
const EPS: f32 = 1e-5;
const ROWS_TESTED: [usize; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
const ROWS_TIMED: [usize; 3] = [1, 2, 4];
const QIDX: usize = 32 * 128;
const HEADS: usize = 32;
const NAMES: [&str; 8] = [
    "q_a",
    "q_resid",
    "q_abs",
    "kv_a",
    "wk",
    "gate",
    "q_idx",
    "head weights",
];
/// 2026-10-21: Output bytes per row.
const ROW_BYTES: [usize; 8] = [
    QA * 2,
    QA * 2,
    QABS * 2,
    KV * 2,
    IDX * 2,
    IDX * 2,
    QIDX * 4,
    HEADS * 4,
];

struct Kernels {
    w8a8: W8a8Kernels,
    split4: KernelHandle,
    rms: KernelHandle,
    few_f32: KernelHandle,
    pair: KernelHandle,
}

/// 2026-10-21: One layer's FP8 weights (q_a, kv_a, wk, gate, q_absorb), the q_a norm, and the
/// BF16 wq_b and weights_proj.
struct Layer {
    w: [Fp8Weight; 5],
    norm: DevicePtr,
    wq_b: DevicePtr,
    wproj: DevicePtr,
}

/// 2026-10-21: One arm's input, outputs (`NAMES` order) and W8A8 scratch at M rows.
struct Bufs {
    m: usize,
    hidden: DevicePtr,
    out: [DevicePtr; 8],
    scratch: W8a8Scratch,
}

/// 2026-10-10: FP8 E4M3 bytes with the two NaN codes (0x7F, 0xFF) moved to their neighbours.
fn e4m3(rng: &mut Lcg, n: usize) -> Vec<u8> {
    (0..n)
        .map(|_| {
            let b = ((rng.next() + 1.0) * 127.99) as u8;
            if b & 0x7F == 0x7F { b ^ 0x01 } else { b }
        })
        .collect()
}

fn fp8(g: &dyn GpuBackend, rng: &mut Lcg, n: usize, k: usize) -> Result<Fp8Weight> {
    let s: Vec<f32> = (0..n).map(|_| 0.001 + rng.next().abs() * 0.01).collect();
    Ok(Fp8Weight {
        weight: up(g, &e4m3(rng, n * k))?,
        row_scale: up_f32(g, &s)?,
        n: n as u32,
        k: k as u32,
        scale_format: WeightQuantFormat::Fp8PerRow,
    })
}

fn layer(g: &dyn GpuBackend, rng: &mut Lcg) -> Result<Layer> {
    Ok(Layer {
        w: [
            fp8(g, rng, QA, HID)?,
            fp8(g, rng, KV, HID)?,
            fp8(g, rng, IDX, HID)?,
            fp8(g, rng, IDX, HID)?,
            fp8(g, rng, QABS, QA)?,
        ],
        norm: up_bf16(g, &rng.vec(QA, 1.0))?,
        wq_b: up_bf16(g, &rng.vec(QIDX * QA, 0.05))?,
        wproj: up_bf16(g, &rng.vec(HEADS * HID, 0.05))?,
    })
}

/// 2026-10-21: Row r of hidden, by family r % 4: drawn; one large value; zeros; a NaN.
fn hidden_rows(rng: &mut Lcg, m: usize) -> Vec<f32> {
    (0..m)
        .flat_map(|r| match r % 4 {
            0 => rng.vec(HID, 2.0),
            1 => {
                let mut v = rng.vec(HID, 0.01);
                v[77] = 300.0;
                v
            }
            2 => vec![0.0; HID],
            _ => {
                let mut v = rng.vec(HID, 1.0);
                v[5] = f32::NAN;
                v
            }
        })
        .collect()
}

fn bufs(g: &dyn GpuBackend, m: usize, h: &[f32]) -> Result<Bufs> {
    let out = ROW_BYTES.map(|w| g.alloc(m * w));
    let [a, b, c, d, e, f, q, w] = out;
    Ok(Bufs {
        m,
        hidden: up_bf16(g, h)?,
        out: [a?, b?, c?, d?, e?, f?, q?, w?],
        scratch: W8a8Scratch::alloc(g, HID as u32)?,
    })
}

fn poison(g: &dyn GpuBackend, b: &Bufs) -> Result<()> {
    for (p, w) in b.out.iter().zip(ROW_BYTES) {
        g.memset(*p, 0x5A, b.m * w)?;
    }
    Ok(())
}

fn one(w: &Fp8Weight) -> Result<W8a8Weight> {
    W8a8Weight::new(&[*w])
}

/// 2026-10-21: The selector query and head weights at 2..=8 rows: two few-row FP32-out launches
/// (today) or the pair launch; nothing at one row (that path is the M = 1 GEMV, unchanged).
fn selector(
    g: &dyn GpuBackend,
    k: &Kernels,
    l: &Layer,
    b: &Bufs,
    pair: bool,
    s: u64,
) -> Result<()> {
    if b.m < 2 {
        return Ok(());
    }
    let sets = [
        (b.out[1], l.wq_b, b.out[6], QIDX, QA),
        (b.hidden, l.wproj, b.out[7], HEADS, HID),
    ];
    if pair {
        let mut x = KernelLaunch::new(g, k.pair)
            .grid([(QIDX.div_ceil(4) + HEADS.div_ceil(4)) as u32, 1, 1])
            .block([256, 1, 1]);
        for (a, w, c, n, kk) in sets {
            x = x
                .arg_ptr(a)
                .arg_ptr(w)
                .arg_ptr(c)
                .arg_u32(n as u32)
                .arg_u32(kk as u32)
                .arg_u32(n as u32);
        }
        return x.arg_u32(b.m as u32).launch(s);
    }
    for (a, w, c, n, kk) in sets {
        KernelLaunch::new(g, k.few_f32)
            .grid([n.div_ceil(4) as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a)
            .arg_ptr(w)
            .arg_ptr(c)
            .arg_u32(b.m as u32)
            .arg_u32(n as u32)
            .arg_u32(kk as u32)
            .arg_u32(n as u32)
            .launch(s)?;
    }
    Ok(())
}

/// 2026-10-21: RMSNorm of q_a into q_resid, quantize q_resid, q_absorb (both arms' tail).
fn tail(g: &dyn GpuBackend, k: &Kernels, l: &Layer, b: &Bufs, s: u64) -> Result<()> {
    KernelLaunch::new(g, k.rms)
        .grid([b.m as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(b.out[0])
        .arg_ptr(l.norm)
        .arg_ptr(b.out[1])
        .arg_u32(QA as u32)
        .arg_f32(EPS)
        .launch(s)?;
    let qabs = one(&l.w[4])?;
    w8a8_proj(
        g,
        &k.w8a8,
        &qabs,
        b.out[1],
        QA as u32,
        b.m,
        b.out[2],
        QABS as u32,
        &b.scratch,
        s,
    )
}

/// 2026-10-21: Today's launches (`project_in`, then the indexer's wk and gate).
fn old(g: &dyn GpuBackend, k: &Kernels, l: &Layer, b: &Bufs, s: u64) -> Result<()> {
    let (m, h) = (b.m, HID as u32);
    w8a8_proj(
        g,
        &k.w8a8,
        &one(&l.w[0])?,
        b.hidden,
        h,
        m,
        b.out[0],
        QA as u32,
        &b.scratch,
        s,
    )?;
    tail(g, k, l, b, s)?;
    w8a8_proj(
        g,
        &k.w8a8,
        &one(&l.w[1])?,
        b.hidden,
        h,
        m,
        b.out[3],
        KV as u32,
        &b.scratch,
        s,
    )?;
    w8a8_gemv(
        g,
        &k.w8a8,
        &one(&l.w[2])?,
        &b.scratch,
        m,
        b.out[4],
        IDX as u32,
        s,
    )?;
    w8a8_gemv(
        g,
        &k.w8a8,
        &one(&l.w[3])?,
        &b.scratch,
        m,
        b.out[5],
        IDX as u32,
        s,
    )?;
    selector(g, k, l, b, false, s)
}

/// 2026-10-21: `DSA_PROJ_OLD=merge` / `=pair` keeps that lever on today's launches in the new
/// arm (to time each lever alone; the byte gate is unchanged either way).
fn old_lever(name: &str) -> Result<bool> {
    Ok(metrale_gpu_examples::env_var("DSA_PROJ_OLD")?
        .is_some_and(|v| v.split(',').any(|x| x == name)))
}

/// 2026-10-21: The merged launches; `swap` exchanges the wk and gate weights (the control).
fn new(g: &dyn GpuBackend, k: &Kernels, l: &Layer, b: &Bufs, swap: bool, s: u64) -> Result<()> {
    if old_lever("merge")? {
        let (m, h) = (b.m, HID as u32);
        w8a8_proj(
            g,
            &k.w8a8,
            &one(&l.w[0])?,
            b.hidden,
            h,
            m,
            b.out[0],
            QA as u32,
            &b.scratch,
            s,
        )?;
        tail(g, k, l, b, s)?;
        w8a8_proj(
            g,
            &k.w8a8,
            &one(&l.w[1])?,
            b.hidden,
            h,
            m,
            b.out[3],
            KV as u32,
            &b.scratch,
            s,
        )?;
        w8a8_gemv(
            g,
            &k.w8a8,
            &one(&l.w[2])?,
            &b.scratch,
            m,
            b.out[4],
            IDX as u32,
            s,
        )?;
        w8a8_gemv(
            g,
            &k.w8a8,
            &one(&l.w[3])?,
            &b.scratch,
            m,
            b.out[5],
            IDX as u32,
            s,
        )?;
        return selector(g, k, l, b, !old_lever("pair")?, s);
    }
    let (m, h) = (b.m, HID as u32);
    w8a8_act_quant(
        g,
        &k.w8a8,
        W8a8Scale::PerRow,
        b.hidden,
        h,
        m,
        h,
        &b.scratch,
        s,
    )?;
    let (wk, gate) = if swap { (3, 2) } else { (2, 3) };
    let segs = [(0, 0, QA), (1, 3, KV), (wk, 4, IDX), (gate, 5, IDX)].map(|(w, o, n)| SplitOut {
        w: l.w[w],
        out: b.out[o],
        ldc: n as u32,
    });
    w8a8_gemv_split4(g, k.split4, &segs, &b.scratch, m, s)?;
    tail(g, k, l, b, s)?;
    selector(g, k, l, b, !old_lever("pair")?, s)
}

fn main() -> Result<()> {
    let timing = !std::env::args().any(|a| a == "--no-timing");
    let sets = metrale_kernels::all_ptx_sets();
    let glm = sets
        .iter()
        .find(|s| s.target.model == "glm-5.3-flash")
        .ok_or_else(|| anyhow::anyhow!("glm-5.3-flash kernel target not built"))?;
    let gpu = MetraleCudaBackend::new(0, &glm.modules)?;
    let g: &dyn GpuBackend = &gpu;
    let k = kernels(g, true)?;
    println!(
        "DSA proj-in gate: today's launches are the ORACLE (hidden {HID}, q_a {QA}, kv_a {KV}, wk/gate {IDX}, q_abs {QABS})\n"
    );
    let want = oracle(g, &k)?;
    if let Some(dir) = metrale_gpu_examples::env_var("DSA_PROJ_BASE_PTX")? {
        let base = MetraleCudaBackend::new(
            0,
            &metrale_gpu_examples::base_ptx::with_base(&glm.modules, &dir)?,
        )?;
        let kb = kernels(&base, false)?;
        for ((m, got), w) in ROWS_TESTED.iter().zip(oracle(&base, &kb)?).zip(&want) {
            for ((what, x), y) in NAMES.iter().zip(&got).zip(w) {
                same(&format!("M={m} {what}: base PTX vs this build"), x, y)?;
            }
        }
        println!("  oracle on the base w8a8_gemv PTX: BYTE-IDENTICAL to this build's, M = 1..8");
    }
    let mut rng = Lcg(0x6473_6170);
    let l = layer(g, &mut rng)?;
    for (i, &m) in ROWS_TESTED.iter().enumerate() {
        let h = hidden_rows(&mut rng, m);
        let b = bufs(g, m, &h)?;
        poison(g, &b)?;
        new(g, &k, &l, &b, false, 0)?;
        for ((what, x), y) in NAMES.iter().zip(&want[i]).zip(&outs(g, &b)?) {
            same(&format!("M={m} {what}"), x, y)?;
        }
        poison(g, &b)?;
        new(g, &k, &l, &b, true, 0)?;
        if !old_lever("merge")? && outs(g, &b)?[4] == want[i][4] {
            bail!("M={m}: control (wk and gate weights swapped) did not move wk");
        }
        println!("  M={m}  every output BYTE-IDENTICAL; control differs");
    }

    if timing {
        anyhow::ensure!(
            gpu.enable_programmatic_launches()?,
            "programmatic launches unavailable"
        );
        let s = g.create_stream()?;
        let layers: Vec<Layer> = (0..DSA_LAYERS)
            .map(|_| layer(g, &mut rng))
            .collect::<Result<_>>()?;
        let plan = Plan {
            calls: DSA_LAYERS,
            rounds: 7,
            trials: 15,
            warm: std::time::Duration::from_millis(300),
        };
        println!(
            "\n  TIMING (PDL on, both levers): {DSA_LAYERS} layers (distinct weights) per graph, us per layer median [min, max]"
        );
        for m in ROWS_TIMED {
            let h = hidden_rows(&mut rng, m);
            let (a, b) = (bufs(g, m, &h)?, bufs(g, m, &h)?);
            let arm_old = |st: u64, i: usize| old(g, &k, &layers[i], &a, st);
            let arm_new = |st: u64, i: usize| new(g, &k, &layers[i], &b, false, st);
            let r = time_arms(g, s, plan, &[&arm_old, &arm_new])?;
            println!(
                "    M={m}  today {:6.2} [{:6.2}, {:6.2}]  merged {:6.2} [{:6.2}, {:6.2}]  saves {:.2} us/layer = {:.3} ms/step at {DSA_LAYERS} layers",
                r[0].0,
                r[0].1,
                r[0].2,
                r[1].0,
                r[1].1,
                r[1].2,
                r[0].0 - r[1].0,
                (r[0].0 - r[1].0) * DSA_LAYERS as f64 / 1000.0
            );
        }
    }
    println!("\nDSA proj-in gate PASS");
    Ok(())
}
