// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Byte gate and microbench for the per-row launch families of GLM-5.3's batched
//! decode at the per-rank shapes of the TP=3/EP=3 layout, each at R = 1, 4 and 16 rows: the
//! per-row launches the decode issued (old) against the one-launch-per-group form (new).
//!
//! 1. KDA conv: R x `causal_conv1d_update_l2norm` against `causal_conv1d_update_l2norm_rows`.
//! 2. KDA recurrence: R x `kda_recurrent_decode_bf16_smem` against
//!    `kda_recurrent_decode_bf16_rows_reg` (and `_smem_rows` for reference).
//! 3. Router logits (N 288, K 4096): R x `dense_gemv_bf16_fp32out` against the batched FP32-out
//!    GEMV the router takes under `METRALE_GLM_ROUTER_ROWS=1` (runtime-M entry at 2..=8 rows,
//!    register-resident entry at 9..=16).
//! 4. DSA indexer `wk` + compress gate (N 128, K 4096), under `declared` (BF16 GEMV against the
//!    batched GEMV) and under `fp8`/`w4a16` (two per-row W8A8 projections against one
//!    quantization and two R-row W8A8 GEMVs), plus the key LayerNorm, as
//!    `METRALE_GLM_DSA_INDEXER_ROWS=1` runs them.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants:
//! - Exits with an error on the first byte mismatch (outputs, and KDA states); prints the GPU
//!   time per group otherwise, measured on a captured CUDA graph (the serve replays graphs),
//!   median and spread over repeats (`common/graph_timing.rs`).
//!
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-model-arch --release --example glm5next_rowbatch_microtest \
//!       --features cuda,gpu-examples

#[path = "common/glm5next_rows_fixture.rs"]
mod fixture;
#[path = "common/glm5next_rowbatch_gemv.rs"]
mod gemv_parts;
#[path = "common/graph_timing.rs"]
mod timing;

use anyhow::{Context, Result};
use fixture::{Lcg, same, up_bf16, up_f32};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use timing::report;

const ROWS: [usize; 3] = [1, 4, 16];
/// 2026-10-09: Copies of a group captured into one graph: the decode's 34 KDA layers.
const COPIES: usize = 34;
/// 2026-10-09: Per-rank KDA heads at TP=3 (64 heads split 22/21/21), head_dim 128.
const H: usize = 22;
const D: usize = 128;
const HIDDEN: usize = 4096;

fn zeros(g: &dyn GpuBackend, n: usize) -> Result<DevicePtr> {
    let p = g.alloc(n.max(1))?;
    g.memset(p, 0, n)?;
    Ok(p)
}

fn read(g: &dyn GpuBackend, s: u64, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    g.synchronize(s)?;
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// 2026-10-09: `timing::time_graph` of `COPIES` copies of `f`.
fn tg(g: &dyn GpuBackend, s: u64, f: &dyn Fn(u64) -> Result<()>) -> Result<(f64, f64, f64)> {
    timing::time_graph(g, s, COPIES, &mut |s| f(s))
}

/// 2026-10-09: Sixteen state pointers and sixteen workspace rows, the rows kernels' tail.
fn rows_tail<'a>(mut l: KernelLaunch<'a>, states: &[DevicePtr]) -> KernelLaunch<'a> {
    for r in 0..16 {
        l = l.arg_u64(states.get(r).map_or(0, |p| p.0));
    }
    for r in 0..16u32 {
        l = l.arg_u32(r);
    }
    l
}

/// 2026-10-09: Parts 1 and 2: the KDA conv and recurrence.
fn kda(g: &dyn GpuBackend, s: u64, rng: &mut Lcg) -> Result<()> {
    let (qkv, kc) = (H * D, 4usize);
    let cd = 3 * qkv;
    let qk = 2 * qkv;
    let conv1 = g.kernel("causal_conv1d", "causal_conv1d_update_l2norm")?;
    let conv_rows = g.kernel("causal_conv1d", "causal_conv1d_update_l2norm_rows")?;
    let rec1 = g.kernel("kda_recurrent", "kda_recurrent_decode_bf16_smem")?;
    let rec_smem_rows = g.kernel("kda_recurrent", "kda_recurrent_decode_bf16_smem_rows")?;
    let rec_reg = g.kernel("kda_recurrent", "kda_recurrent_decode_bf16_rows_reg")?;
    let (vpb, scale) = (32usize, 1.0 / (D as f32).sqrt());
    let smem = ((3 * D + vpb * (D + 1)) * 4) as u32;
    let conv_w = up_bf16(g, &rng.vec(cd * kc, 0.5))?;
    for rows in ROWS {
        let qkv_in = up_bf16(g, &rng.vec(rows * cd, 1.0))?;
        let gate = up_f32(
            g,
            &(0..rows * qkv)
                .map(|_| -0.05 - rng.next().abs() * 0.1)
                .collect::<Vec<_>>(),
        )?;
        let beta = up_f32(
            g,
            &(0..rows * H)
                .map(|_| 0.5 + rng.next() * 0.2)
                .collect::<Vec<_>>(),
        )?;
        let (cb, hb) = (cd * kc * 4, H * D * D * 4);
        let conv_init: Vec<Vec<f32>> = (0..rows).map(|_| rng.vec(cd * kc, 0.5)).collect();
        let rec_init: Vec<Vec<f32>> = (0..rows).map(|_| rng.vec(H * D * D, 0.1)).collect();
        let mk =
            |v: &[Vec<f32>]| -> Result<Vec<DevicePtr>> { v.iter().map(|x| up_f32(g, x)).collect() };
        let (ca, cbat) = (mk(&conv_init)?, mk(&conv_init)?);
        let (ra, rbat, rsm) = (mk(&rec_init)?, mk(&rec_init)?, mk(&rec_init)?);
        let (oa, ob) = (zeros(g, rows * cd * 2)?, zeros(g, rows * cd * 2)?);
        let conv_old = |st: &[DevicePtr], out: DevicePtr, s: u64| -> Result<()> {
            for (r, p) in st.iter().enumerate() {
                KernelLaunch::new(g, conv1)
                    .grid([cd.div_ceil(256) as u32, 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(*p)
                    .arg_ptr(qkv_in.offset(r * cd * 2))
                    .arg_ptr(conv_w)
                    .arg_ptr(DevicePtr::NULL)
                    .arg_ptr(out.offset(r * cd * 2))
                    .arg_u32(1)
                    .arg_u32(cd as u32)
                    .arg_u32(kc as u32)
                    .arg_u32(qk as u32)
                    .arg_u32(D as u32)
                    .arg_f32(1e-6)
                    .launch(s)?;
            }
            Ok(())
        };
        let conv_new = |st: &[DevicePtr], out: DevicePtr, s: u64| -> Result<()> {
            let l = KernelLaunch::new(g, conv_rows)
                .grid([cd.div_ceil(256) as u32, rows as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(qkv_in)
                .arg_ptr(conv_w)
                .arg_ptr(DevicePtr::NULL)
                .arg_ptr(out)
                .arg_u32(cd as u32)
                .arg_u32(kc as u32)
                .arg_u32(qk as u32)
                .arg_u32(D as u32)
                .arg_f32(1e-6);
            rows_tail(l, st).launch(s)
        };
        conv_old(&ca, oa, s)?;
        conv_new(&cbat, ob, s)?;
        same(
            "KDA conv out",
            &read(g, s, oa, rows * cd * 2)?,
            &read(g, s, ob, rows * cd * 2)?,
        )?;
        for r in 0..rows {
            same(
                &format!("KDA conv state {r}"),
                &read(g, s, ca[r], cb)?,
                &read(g, s, cbat[r], cb)?,
            )?;
        }
        // 2026-10-09: The recurrence reads the conv output the old arm wrote (q|k|v per row).
        let (ya, yb, ysm) = (
            zeros(g, rows * qkv * 4)?,
            zeros(g, rows * qkv * 4)?,
            zeros(g, rows * qkv * 4)?,
        );
        let rec_old = |st: &[DevicePtr], out: DevicePtr, s: u64| -> Result<()> {
            for (r, p) in st.iter().enumerate() {
                let q = oa.offset(r * cd * 2);
                KernelLaunch::new(g, rec1)
                    .grid([H as u32, (D / vpb) as u32, 1])
                    .block([vpb as u32, 1, 1])
                    .shared_mem(smem)
                    .arg_ptr(q)
                    .arg_ptr(q.offset(qkv * 2))
                    .arg_ptr(q.offset(qkv * 4))
                    .arg_ptr(gate.offset(r * qkv * 4))
                    .arg_ptr(beta.offset(r * H * 4))
                    .arg_ptr(*p)
                    .arg_ptr(out.offset(r * qkv * 4))
                    .arg_u32(H as u32)
                    .arg_u32(D as u32)
                    .arg_f32(scale)
                    .arg_u32(vpb as u32)
                    .launch(s)?;
            }
            Ok(())
        };
        let rec_new =
            |k: KernelHandle, reg: bool, st: &[DevicePtr], out: DevicePtr, s: u64| -> Result<()> {
                let mut l = if reg {
                    KernelLaunch::new(g, k)
                        .grid([H as u32, 1, rows as u32])
                        .block([D as u32, 1, 1])
                        .shared_mem((3 * D * 4) as u32)
                } else {
                    KernelLaunch::new(g, k)
                        .grid([H as u32, (D / vpb) as u32, rows as u32])
                        .block([vpb as u32, 1, 1])
                        .shared_mem(smem)
                }
                .arg_ptr(oa)
                .arg_ptr(oa.offset(qkv * 2))
                .arg_ptr(oa.offset(qkv * 4))
                .arg_ptr(gate)
                .arg_ptr(beta)
                .arg_ptr(out)
                .arg_u32(H as u32);
                l = if reg {
                    l.arg_f32(scale)
                } else {
                    l.arg_u32(D as u32).arg_f32(scale).arg_u32(vpb as u32)
                };
                l = l
                    .arg_u32(cd as u32)
                    .arg_u32(qkv as u32)
                    .arg_u32(H as u32)
                    .arg_u32(qkv as u32);
                rows_tail(l, st).launch(s)
            };
        rec_old(&ra, ya, s)?;
        rec_new(rec_reg, true, &rbat, yb, s)?;
        rec_new(rec_smem_rows, false, &rsm, ysm, s)?;
        let want = read(g, s, ya, rows * qkv * 4)?;
        same(
            "KDA out (rows_reg)",
            &want,
            &read(g, s, yb, rows * qkv * 4)?,
        )?;
        same(
            "KDA out (smem_rows)",
            &want,
            &read(g, s, ysm, rows * qkv * 4)?,
        )?;
        for r in 0..rows {
            let st = read(g, s, ra[r], hb)?;
            same(
                &format!("KDA state {r} (rows_reg)"),
                &st,
                &read(g, s, rbat[r], hb)?,
            )?;
            same(
                &format!("KDA state {r} (smem_rows)"),
                &st,
                &read(g, s, rsm[r], hb)?,
            )?;
        }
        // 2026-10-09: Timing advances the states further, which does not change the work.
        report(
            "KDA conv",
            rows,
            tg(g, s, &|s| conv_old(&ca, oa, s))?,
            tg(g, s, &|s| conv_new(&cbat, ob, s))?,
        );
        let old = tg(g, s, &|s| rec_old(&ra, ya, s))?;
        report(
            "KDA recurrence (rows_reg)",
            rows,
            old,
            tg(g, s, &|s| rec_new(rec_reg, true, &rbat, yb, s))?,
        );
        report(
            "KDA recurrence (smem_rows)",
            rows,
            old,
            tg(g, s, &|s| rec_new(rec_smem_rows, false, &rsm, ysm, s))?,
        );
        for p in ca
            .iter()
            .chain(&cbat)
            .chain(&ra)
            .chain(&rbat)
            .chain(&rsm)
            .chain([&qkv_in, &gate, &beta, &oa, &ob, &ya, &yb, &ysm])
        {
            g.free(*p)?;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let sets = metrale_kernels::all_ptx_sets();
    let glm = sets
        .iter()
        .find(|s| s.target.model == "glm-5.3-flash")
        .context("glm-5.3-flash kernel target not built")?;
    let gpu = MetraleCudaBackend::new(0, &glm.modules)?;
    let g: &dyn GpuBackend = &gpu;
    let s = g.create_stream()?;
    let mut rng = Lcg(0x6c6d_7262_0001);
    kda(g, s, &mut rng)?;
    let f32_out = [
        g.kernel("gemv", "dense_gemv_bf16_fp32out")?,
        g.kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm_fp32out")?,
        g.kernel(
            "dense_gemv_bf16_batchm",
            "dense_gemv_bf16_batchm_wide_fp32out",
        )?,
    ];
    gemv_parts::gemv_family(
        g,
        s,
        &mut rng,
        "router logits (FP32 out)",
        f32_out,
        288,
        1,
        4,
    )?;
    let bf16_out = [
        g.kernel("gemv", "dense_gemv_bf16")?,
        g.kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")?,
        g.kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm_wide")?,
    ];
    gemv_parts::gemv_family(g, s, &mut rng, "indexer wk+gate BF16", bf16_out, D, 2, 2)?;
    gemv_parts::indexer_w8a8(g, s, &mut rng)?;
    println!("PASS: every batched result is byte-identical to its per-row reference");
    Ok(())
}
