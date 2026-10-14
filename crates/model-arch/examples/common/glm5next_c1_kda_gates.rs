// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Part 4 of `glm5next_c1_fuse_microtest`: the KDA gate chain at the TP=3 per-rank
//! shape (22 heads of 128, low rank 128), unfused (act-quant + f_b W8A8 + kda_gate + beta sigmoid
//! + act-quant + g_b W8A8, six launches) against `kda_gates_w8a8_k128`
//! (`METRALE_GLM_KDA_GATES_FUSED=1`). g_raw, gate, out_gate and beta must be byte-identical, on
//! drawn rows, on rows whose amax is set by one large value, on a zero row (scale floor 1e-12)
//! and on a row with a NaN (dropped by the amax, clamped by the encoder).
//! 2026-10-14: At M = 1..=16, 24 and 32 rows (the MTP verify groups), each M a run of the edge
//! rows and drawn ones, with the unfused projections on the production W8A8 path
//! (`ops::w8a8_proj`: the row quantizer, then the GEMV entry for M rows). That is the leaf
//! production's unfused chain reaches: `Glm5NextKdaLayer::gemm` -> `glm_mm` -> (f_b / g_b are
//! FP8-tier, so not W4A16) `glm5next_fp8_dense::proj` -> `w8a8_proj` in one chunk at these widths.
//! 2026-10-14: Each arm's outputs start from a different byte (0xA5, 0x5A), so a launch that
//! skips a row cannot compare equal; at M = 8 a negative control flips the top mantissa bit of
//! row 0's amax element of f_low and g_low and of beta logit 0, reruns the fused kernel and
//! requires every output to differ from the unfused one.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: as `glm5next_c1_fuse_microtest`'s.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_layers::layers::ops::{W8a8Kernels, W8a8Scratch, W8a8Weight, w8a8_proj};
use metrale_model_layers::weight_map::{Fp8Weight, WeightQuantFormat};

use super::fixture::{Lcg, same, up, up_bf16, up_f32};
use super::timing::report;
use super::{read, tg};

const H: usize = 22;
const D: usize = 128;
const N: usize = H * D;
const K: usize = 128;
const KDA_LAYERS: usize = 34;
/// 2026-10-14: The row counts checked; the timed ones are the C1/C2/C4/C8 verify groups.
const ROWS: [usize; 18] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 32,
];
const TIMED: [usize; 4] = [2, 4, 8, 16];
/// 2026-10-14: The row count of the negative control, and the outputs in `Bufs::out` order.
const NEG_M: usize = 8;
const OUTS: [&str; 4] = ["g_raw", "gate", "out_gate", "beta"];

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
    m: usize,
    f_low: DevicePtr,
    g_low: DevicePtr,
    beta_bf16: DevicePtr,
    out: [DevicePtr; 4],
}

/// 2026-10-14: `m` (f_low, g_low) rows: the edge rows (one large value; zeros in f_low; a NaN in
/// g_low) at positions that move with `m`, drawn rows elsewhere.
fn rows(rng: &mut Lcg, m: usize) -> (Vec<f32>, Vec<f32>) {
    let (mut f, mut g) = (Vec::with_capacity(m * K), Vec::with_capacity(m * K));
    for i in 0..m {
        let (mut a, mut b) = (rng.vec(K, 2.0), rng.vec(K, 2.0));
        match (i + m) % 5 {
            0 => {
                a = rng.vec(K, 0.01);
                a[77] = 300.0;
                b.clone_from(&a);
            }
            1 => a = vec![0.0; K],
            2 => b[5] = f32::NAN,
            _ => {}
        }
        f.extend(a);
        g.extend(b);
    }
    (f, g)
}

pub(crate) fn kda_gates(g: &dyn GpuBackend, s: u64, rng: &mut Lcg) -> Result<()> {
    let w8 = W8a8Kernels::load(g);
    let scratch = W8a8Scratch::alloc(g, K as u32)?;
    let gate_k = g.kernel("kda_gate", "kda_gate_bf16")?;
    let sigmoid = g.kernel("kda_layer_ops", "kda_sigmoid_bf16_f32")?;
    let fused = g.kernel("kda_gates_w8a8", "kda_gates_w8a8_k128")?;
    let (wf, wg) = (up(g, &e4m3(rng, N * K))?, up(g, &e4m3(rng, N * K))?);
    let scales =
        |rng: &mut Lcg| -> Vec<f32> { (0..N).map(|_| 0.001 + rng.next().abs() * 0.01).collect() };
    let (sf, sg) = (up_f32(g, &scales(rng))?, up_f32(g, &scales(rng))?);
    let fp8 = |weight, row_scale| Fp8Weight {
        weight,
        row_scale,
        n: N as u32,
        k: K as u32,
        scale_format: WeightQuantFormat::Fp8PerRow,
    };
    let (w8f, w8g) = (
        W8a8Weight::new(&[fp8(wf, sf)])?,
        W8a8Weight::new(&[fp8(wg, sg)])?,
    );
    let dt_bias = up_f32(g, &rng.vec(N, 0.5))?;
    let a_log = up_f32(g, &rng.vec(H, 1.0))?;
    let lb = -5.0f32;
    let sizes = |m: usize| [m * N * 2, m * N * 4, m * N * 2, m * H * 4];
    let outs = |m: usize, fill: u8| -> Result<[DevicePtr; 4]> {
        let z = sizes(m);
        let mut o = [DevicePtr::NULL; 4];
        for (p, n) in o.iter_mut().zip(z) {
            *p = g.alloc(n)?;
            g.memset(*p, fill, n)?;
        }
        Ok(o)
    };
    let arm = |m: usize, f: &[f32], gl: &[f32], fill: u8| -> Result<Bufs> {
        Ok(Bufs {
            m,
            f_low: up_bf16(g, f)?,
            g_low: up_bf16(g, gl)?,
            beta_bf16: up_bf16(g, &beta_logits(m))?,
            out: outs(m, fill)?,
        })
    };
    // 2026-10-10: out = [g_raw (BF16), gate (F32), out_gate (BF16), beta (F32)].
    let old = |b: &Bufs, s: u64| -> Result<()> {
        let m = b.m;
        w8a8_proj(
            g, &w8, &w8f, b.f_low, K as u32, m, b.out[0], N as u32, &scratch, s,
        )?;
        KernelLaunch::new(g, gate_k)
            .grid([(m * H) as u32, 1, 1])
            .block([128, 1, 1])
            .arg_ptr(b.out[0])
            .arg_ptr(dt_bias)
            .arg_ptr(a_log)
            .arg_ptr(b.out[1])
            .arg_u32(m as u32)
            .arg_u32(H as u32)
            .arg_u32(D as u32)
            .arg_f32(lb)
            .launch(s)?;
        KernelLaunch::new(g, sigmoid)
            .grid([((m * H) as u32).div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(b.beta_bf16)
            .arg_ptr(b.out[3])
            .arg_u32((m * H) as u32)
            .launch(s)?;
        w8a8_proj(
            g, &w8, &w8g, b.g_low, K as u32, m, b.out[2], N as u32, &scratch, s,
        )
    };
    let new = |b: &Bufs, s: u64| -> Result<()> {
        KernelLaunch::new(g, fused)
            .grid([(2 * N / 16).div_ceil(8) as u32, b.m.div_ceil(8) as u32, 1])
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
            .arg_u32(b.m as u32)
            .arg_u32(N as u32)
            .arg_u32(D as u32)
            .arg_u32(H as u32)
            .arg_f32(lb)
            .launch(s)
    };
    for m in ROWS {
        let (f, gl) = rows(rng, m);
        let (a, b) = (arm(m, &f, &gl, 0xA5)?, arm(m, &f, &gl, 0x5A)?);
        old(&a, s)?;
        new(&b, s)?;
        let z = sizes(m);
        for (j, what) in OUTS.iter().enumerate() {
            same(
                &format!("KDA gates M={m} {what}"),
                &read(g, s, a.out[j], z[j])?,
                &read(g, s, b.out[j], z[j])?,
            )?;
        }
        if m == NEG_M {
            let c = Bufs {
                m,
                f_low: flipped(g, s, b.f_low, m * K, amax_at(&f[..K]))?,
                g_low: flipped(g, s, b.g_low, m * K, amax_at(&gl[..K]))?,
                beta_bf16: flipped(g, s, b.beta_bf16, m * H, 0)?,
                out: outs(m, 0x5A)?,
            };
            new(&c, s)?;
            for (j, what) in OUTS.iter().enumerate() {
                let (x, y) = (read(g, s, a.out[j], z[j])?, read(g, s, c.out[j], z[j])?);
                if same(what, &x, &y).is_ok() {
                    anyhow::bail!("KDA gates M={m} {what}: negative control did not differ");
                }
            }
            println!("NEG CONTROL: differs (g_raw, gate, out_gate, beta) at M={m}");
            for p in [c.f_low, c.g_low, c.beta_bf16].into_iter().chain(c.out) {
                g.free(p)?;
            }
        }
        if m == 1 || TIMED.contains(&m) {
            report(
                "KDA gate chain (6 -> 1)",
                m,
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
    println!("KDA gates: byte-identical at M = {ROWS:?}");
    for p in [wf, wg, sf, sg, dt_bias, a_log, scratch.q, scratch.scale] {
        g.free(p)?;
    }
    Ok(())
}

/// 2026-10-14: A copy of the `n` BF16 values at `p` with value `at`'s top mantissa bit flipped
/// (not its lowest: a one-ulp change mostly rounds to the same E4M3 code in the row quantizer).
fn flipped(g: &dyn GpuBackend, s: u64, p: DevicePtr, n: usize, at: usize) -> Result<DevicePtr> {
    let mut v = read(g, s, p, n * 2)?;
    v[2 * at] ^= 0x40;
    up(g, &v)
}

/// 2026-10-14: The index of the largest magnitude in `row`, so a flip there moves the row scale.
fn amax_at(row: &[f32]) -> usize {
    (0..row.len()).fold(0, |b, i| if row[i].abs() > row[b].abs() { i } else { b })
}

/// 2026-10-10: beta logits spanning the sigmoid's range, the same in both arms.
/// 2026-10-14: `m` rows of them, each row shifted so rows differ.
fn beta_logits(m: usize) -> Vec<f32> {
    (0..m * H)
        .map(|i| ((i % H) as f32 - 11.0) * 0.75 + (i / H) as f32 * 0.125)
        .collect()
}
