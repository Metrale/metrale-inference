// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Part 4 of `glm5next_c1_fuse_microtest`: the KDA gate chain of one token at the TP=3
//! per-rank shape (22 heads of 128, low rank 128), unfused (act-quant + f_b W8A8 + kda_gate +
//! beta sigmoid + act-quant + g_b W8A8, six launches) against `kda_gates_w8a8_k128`
//! (`METRALE_GLM_KDA_GATES_FUSED=1`). g_raw, gate, out_gate and beta must be byte-identical, on
//! drawn rows, on rows whose amax is set by one large value, on a zero row (scale floor 1e-12)
//! and on a row with a NaN (dropped by the amax, clamped by the encoder).
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: as `glm5next_c1_fuse_microtest`'s.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::fixture::{Lcg, same, up, up_bf16, up_f32};
use super::timing::report;
use super::{read, tg, zeros};

const H: usize = 22;
const D: usize = 128;
const N: usize = H * D;
const K: usize = 128;
const KDA_LAYERS: usize = 34;

/// 2026-10-10: FP8 E4M3 bytes with the two NaN codes (0x7F, 0xFF) moved to their neighbours.
fn e4m3(rng: &mut Lcg, n: usize) -> Vec<u8> {
    (0..n)
        .map(|_| {
            let b = ((rng.next() + 1.0) * 127.99) as u8;
            if b & 0x7F == 0x7F { b ^ 0x01 } else { b }
        })
        .collect()
}

struct Bufs {
    f_low: DevicePtr,
    g_low: DevicePtr,
    beta_bf16: DevicePtr,
    out: [DevicePtr; 4],
}

pub(crate) fn kda_gates(g: &dyn GpuBackend, s: u64, rng: &mut Lcg) -> Result<()> {
    let quant = g.kernel("w8a8_act_quant", "w8a8_act_quant_row")?;
    let gemv = g.kernel("w8a8_gemv", "w8a8_gemv_rowscale_mb1_ku8")?;
    let gate_k = g.kernel("kda_gate", "kda_gate_bf16")?;
    let sigmoid = g.kernel("kda_layer_ops", "kda_sigmoid_bf16_f32")?;
    let fused = g.kernel("kda_gates_w8a8", "kda_gates_w8a8_k128")?;
    let (wf, wg) = (up(g, &e4m3(rng, N * K))?, up(g, &e4m3(rng, N * K))?);
    let scales =
        |rng: &mut Lcg| -> Vec<f32> { (0..N).map(|_| 0.001 + rng.next().abs() * 0.01).collect() };
    let (sf, sg) = (up_f32(g, &scales(rng))?, up_f32(g, &scales(rng))?);
    let dt_bias = up_f32(g, &rng.vec(N, 0.5))?;
    let a_log = up_f32(g, &rng.vec(H, 1.0))?;
    let (aq, as_) = (zeros(g, K)?, zeros(g, 4)?);
    let lb = -5.0f32;
    // 2026-10-10: (f_low, g_low) rows: drawn; one large value; zeros; a NaN.
    let mut rows: Vec<(Vec<f32>, Vec<f32>)> =
        (0..3).map(|_| (rng.vec(K, 2.0), rng.vec(K, 2.0))).collect();
    let mut spike = rng.vec(K, 0.01);
    spike[77] = 300.0;
    rows.push((spike.clone(), spike));
    rows.push((vec![0.0; K], rng.vec(K, 1.0)));
    let mut nan = rng.vec(K, 1.0);
    nan[5] = f32::NAN;
    rows.push((rng.vec(K, 1.0), nan));
    let sizes = [N * 2, N * 4, N * 2, H * 4];
    let arm = |f: &[f32], gl: &[f32]| -> Result<Bufs> {
        Ok(Bufs {
            f_low: up_bf16(g, f)?,
            g_low: up_bf16(g, gl)?,
            beta_bf16: up_bf16(g, &beta_logits())?,
            out: [
                zeros(g, sizes[0])?,
                zeros(g, sizes[1])?,
                zeros(g, sizes[2])?,
                zeros(g, sizes[3])?,
            ],
        })
    };
    // 2026-10-10: out = [g_raw (BF16), gate (F32), out_gate (BF16), beta (F32)].
    let old = |b: &Bufs, s: u64| -> Result<()> {
        for (x, w, sc, o) in [(b.f_low, wf, sf, b.out[0]), (b.g_low, wg, sg, b.out[2])] {
            KernelLaunch::new(g, quant)
                .grid([1, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(x)
                .arg_ptr(aq)
                .arg_ptr(as_)
                .arg_u32(K as u32)
                .arg_u32(K as u32)
                .arg_u32(K as u32)
                .launch(s)?;
            KernelLaunch::new(g, gemv)
                .grid([N.div_ceil(16) as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(aq)
                .arg_ptr(as_)
                .arg_ptr(w)
                .arg_ptr(sc)
                .arg_ptr(w)
                .arg_ptr(sc)
                .arg_ptr(w)
                .arg_ptr(sc)
                .arg_ptr(o)
                .arg_u32(1)
                .arg_u32(N as u32)
                .arg_u32(K as u32)
                .arg_u32(K as u32)
                .arg_u32(N as u32)
                .arg_u32(N as u32)
                .arg_u32(N as u32)
                .launch(s)?;
            if o == b.out[0] {
                KernelLaunch::new(g, gate_k)
                    .grid([H as u32, 1, 1])
                    .block([128, 1, 1])
                    .arg_ptr(b.out[0])
                    .arg_ptr(dt_bias)
                    .arg_ptr(a_log)
                    .arg_ptr(b.out[1])
                    .arg_u32(1)
                    .arg_u32(H as u32)
                    .arg_u32(D as u32)
                    .arg_f32(lb)
                    .launch(s)?;
                KernelLaunch::new(g, sigmoid)
                    .grid([1, 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(b.beta_bf16)
                    .arg_ptr(b.out[3])
                    .arg_u32(H as u32)
                    .launch(s)?;
            }
        }
        Ok(())
    };
    let new = |b: &Bufs, s: u64| -> Result<()> {
        KernelLaunch::new(g, fused)
            .grid([(2 * N / 16).div_ceil(8) as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(b.f_low)
            .arg_ptr(b.g_low)
            .arg_ptr(wf)
            .arg_ptr(sf)
            .arg_ptr(wg)
            .arg_ptr(sg)
            .arg_ptr(b.out[0])
            .arg_ptr(b.out[2])
            .arg_ptr(dt_bias)
            .arg_ptr(a_log)
            .arg_ptr(b.out[1])
            .arg_ptr(b.beta_bf16)
            .arg_ptr(b.out[3])
            .arg_u32(N as u32)
            .arg_u32(D as u32)
            .arg_u32(H as u32)
            .arg_f32(lb)
            .launch(s)
    };
    for (i, (f, gl)) in rows.iter().enumerate() {
        let (a, b) = (arm(f, gl)?, arm(f, gl)?);
        old(&a, s)?;
        new(&b, s)?;
        for (j, what) in ["g_raw", "gate", "out_gate", "beta"].iter().enumerate() {
            same(
                &format!("KDA gates row {i} {what}"),
                &read(g, s, a.out[j], sizes[j])?,
                &read(g, s, b.out[j], sizes[j])?,
            )?;
        }
        if i == 0 {
            report(
                "KDA gate chain (6 -> 1)",
                1,
                tg(g, s, KDA_LAYERS, &|s| old(&a, s))?,
                tg(g, s, KDA_LAYERS, &|s| new(&b, s))?,
            );
        }
        for p in [a, b]
            .iter()
            .flat_map(|x| [x.f_low, x.g_low, x.beta_bf16].into_iter().chain(x.out))
        {
            g.free(p)?;
        }
    }
    for p in [wf, wg, sf, sg, dt_bias, a_log, aq, as_] {
        g.free(p)?;
    }
    Ok(())
}

/// 2026-10-10: beta logits spanning the sigmoid's range, the same in both arms.
fn beta_logits() -> Vec<f32> {
    (0..H).map(|h| (h as f32 - 11.0) * 0.75).collect()
}
