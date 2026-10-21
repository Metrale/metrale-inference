// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: Bit-identity gate and microbench: the KDA gate chain of M = 1..8 tokens at the
//! TP=3 per-rank shape (22 heads of 128, low rank 128), unfused (act-quant + f_b W8A8 + kda_gate +
//! beta sigmoid + act-quant + g_b W8A8, six launches, the path a verify step takes without
//! `METRALE_GLM_KDA_GATES_ROWS`) against `kda_gates_w8a8_k128_rows`.
//!
//! Owner: gpu examples (GLM-5.3 KDA).
//! Invariants:
//! - Fails unless g_raw, gate, out_gate and beta are byte-identical for every M in `ROWS_TESTED`,
//!   with the M rows drawn from four families in one launch (drawn values, one large value per
//!   row, a zero row (scale floor 1e-12), a NaN (dropped by the amax, clamped by the encoder)),
//!   so rows with different scales share every MMA tile; and unless the rows kernel at M = 1 is
//!   byte-identical to `kda_gates_w8a8_k128` (the one-token kernel).
//! - A negative control must DIFFER: one g_low value of the last row set to half the row's amax.
//!
//!   cargo build -p metrale-gpu-examples --release --example glm5next_kda_gates_rows_gate \
//!       --features cuda,gpu-examples
//!
//! Timing: `KDA_LAYERS` layers with distinct weights in one graph, each arm per M in
//! `ROWS_TIMED` (graph_arms: clock-warmed, interleaved rounds). `--no-timing` skips it.

#[path = "common/glm5next_rows_fixture.rs"]
mod fixture;

use anyhow::{Result, bail};
use fixture::{Lcg, same, up, up_bf16, up_f32};
use metrale_gpu_examples::graph_arms::{Plan, time_arms};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

const H: usize = 22;
const D: usize = 128;
const N: usize = H * D;
const K: usize = 128;
const KDA_LAYERS: usize = 34;
const LB: f32 = -5.0;
const ROWS_TESTED: [usize; 7] = [1, 2, 3, 4, 5, 7, 8];
const ROWS_TIMED: [usize; 3] = [1, 2, 4];
const NAMES: [&str; 4] = ["g_raw", "gate", "out_gate", "beta"];

struct Kernels {
    quant: KernelHandle,
    gemv: KernelHandle,
    gate: KernelHandle,
    sigmoid: KernelHandle,
    one: KernelHandle,
    rows: KernelHandle,
}

/// 2026-10-21: One layer's weights.
struct Layer {
    wf: DevicePtr,
    wg: DevicePtr,
    sf: DevicePtr,
    sg: DevicePtr,
    dt_bias: DevicePtr,
    a_log: DevicePtr,
}

/// 2026-10-21: One arm's inputs and outputs at M rows, and the unfused path's W8A8 scratch.
struct Bufs {
    m: usize,
    f_low: DevicePtr,
    g_low: DevicePtr,
    beta_bf16: DevicePtr,
    out: [DevicePtr; 4],
    aq: DevicePtr,
    a_s: DevicePtr,
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

fn zeros(g: &dyn GpuBackend, n: usize) -> Result<DevicePtr> {
    let p = g.alloc(n.max(1))?;
    g.memset(p, 0, n)?;
    Ok(p)
}

fn read(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    g.synchronize(0)?;
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

fn sizes(m: usize) -> [usize; 4] {
    [m * N * 2, m * N * 4, m * N * 2, m * H * 4]
}

fn layer(g: &dyn GpuBackend, rng: &mut Lcg) -> Result<Layer> {
    let mut scales = || -> Vec<f32> { (0..N).map(|_| 0.001 + rng.next().abs() * 0.01).collect() };
    let (sf, sg) = (scales(), scales());
    Ok(Layer {
        wf: up(g, &e4m3(rng, N * K))?,
        wg: up(g, &e4m3(rng, N * K))?,
        sf: up_f32(g, &sf)?,
        sg: up_f32(g, &sg)?,
        dt_bias: up_f32(g, &rng.vec(N, 0.5))?,
        a_log: up_f32(g, &rng.vec(H, 1.0))?,
    })
}

/// 2026-10-21: Row r of the f_low / g_low inputs, by family r % 4: drawn; one large value; zeros
/// (f) or drawn (g); drawn (f) or a NaN (g).
fn low_rows(rng: &mut Lcg, m: usize) -> (Vec<f32>, Vec<f32>) {
    let (mut f, mut gl) = (Vec::new(), Vec::new());
    for r in 0..m {
        let (a, b) = match r % 4 {
            0 => (rng.vec(K, 2.0), rng.vec(K, 2.0)),
            1 => {
                let mut v = rng.vec(K, 0.01);
                v[77] = 300.0;
                (v.clone(), v)
            }
            2 => (vec![0.0; K], rng.vec(K, 1.0)),
            _ => {
                let mut v = rng.vec(K, 1.0);
                v[5] = f32::NAN;
                (rng.vec(K, 1.0), v)
            }
        };
        f.extend(a);
        gl.extend(b);
    }
    (f, gl)
}

fn bufs(g: &dyn GpuBackend, m: usize, f: &[f32], gl: &[f32]) -> Result<Bufs> {
    let beta: Vec<f32> = (0..m * H)
        .map(|i| ((i % H) as f32 - 11.0) * 0.75 + (i / H) as f32 * 0.1)
        .collect();
    let s = sizes(m);
    Ok(Bufs {
        m,
        f_low: up_bf16(g, f)?,
        g_low: up_bf16(g, gl)?,
        beta_bf16: up_bf16(g, &beta)?,
        out: [
            zeros(g, s[0])?,
            zeros(g, s[1])?,
            zeros(g, s[2])?,
            zeros(g, s[3])?,
        ],
        aq: zeros(g, m * K)?,
        a_s: zeros(g, m * 4)?,
    })
}

/// 2026-10-21: The unfused chain at `b.m` rows, the launches `front_end` makes.
fn old(g: &dyn GpuBackend, k: &Kernels, l: &Layer, b: &Bufs, s: u64) -> Result<()> {
    let m = b.m as u32;
    for (x, w, sc, o) in [
        (b.f_low, l.wf, l.sf, b.out[0]),
        (b.g_low, l.wg, l.sg, b.out[2]),
    ] {
        KernelLaunch::new(g, k.quant)
            .grid([m, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(b.aq)
            .arg_ptr(b.a_s)
            .arg_u32(K as u32)
            .arg_u32(K as u32)
            .arg_u32(K as u32)
            .launch(s)?;
        let mut gl = KernelLaunch::new(g, k.gemv)
            .grid([N.div_ceil(16) as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(b.aq)
            .arg_ptr(b.a_s);
        for _ in 0..3 {
            gl = gl.arg_ptr(w).arg_ptr(sc);
        }
        gl.arg_ptr(o)
            .arg_u32(m)
            .arg_u32(N as u32)
            .arg_u32(K as u32)
            .arg_u32(K as u32)
            .arg_u32(N as u32)
            .arg_u32(N as u32)
            .arg_u32(N as u32)
            .launch(s)?;
        if o == b.out[0] {
            KernelLaunch::new(g, k.gate)
                .grid([m * H as u32, 1, 1])
                .block([128, 1, 1])
                .arg_ptr(b.out[0])
                .arg_ptr(l.dt_bias)
                .arg_ptr(l.a_log)
                .arg_ptr(b.out[1])
                .arg_u32(m)
                .arg_u32(H as u32)
                .arg_u32(D as u32)
                .arg_f32(LB)
                .launch(s)?;
            KernelLaunch::new(g, k.sigmoid)
                .grid([(m * H as u32).div_ceil(256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(b.beta_bf16)
                .arg_ptr(b.out[3])
                .arg_u32(m * H as u32)
                .launch(s)?;
        }
    }
    Ok(())
}

/// 2026-10-21: One fused launch: the rows kernel with M = `b.m`, or (`one`) the one-token kernel.
fn fused(g: &dyn GpuBackend, k: &Kernels, l: &Layer, b: &Bufs, one: bool, s: u64) -> Result<()> {
    let mut f = KernelLaunch::new(g, if one { k.one } else { k.rows })
        .grid([(2 * N / 16).div_ceil(8) as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(b.f_low)
        .arg_ptr(b.g_low)
        .arg_ptr(l.wf)
        .arg_ptr(l.sf)
        .arg_ptr(l.wg)
        .arg_ptr(l.sg)
        .arg_ptr(b.out[0])
        .arg_ptr(b.out[2])
        .arg_ptr(l.dt_bias)
        .arg_ptr(l.a_log)
        .arg_ptr(b.out[1])
        .arg_ptr(b.beta_bf16)
        .arg_ptr(b.out[3]);
    if !one {
        f = f.arg_u32(b.m as u32);
    }
    f.arg_u32(N as u32)
        .arg_u32(D as u32)
        .arg_u32(H as u32)
        .arg_f32(LB)
        .launch(s)
}

fn outs(g: &dyn GpuBackend, b: &Bufs) -> Result<Vec<Vec<u8>>> {
    b.out
        .iter()
        .zip(sizes(b.m))
        .map(|(&p, n)| read(g, p, n))
        .collect()
}

fn main() -> Result<()> {
    let timing = !std::env::args().any(|a| a == "--no-timing");
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;
    let k = Kernels {
        quant: g.kernel("w8a8_act_quant", "w8a8_act_quant_row")?,
        gemv: g.kernel("w8a8_gemv", "w8a8_gemv_rowscale_mb1_ku8")?,
        gate: g.kernel("kda_gate", "kda_gate_bf16")?,
        sigmoid: g.kernel("kda_layer_ops", "kda_sigmoid_bf16_f32")?,
        one: g.kernel("kda_gates_w8a8", "kda_gates_w8a8_k128")?,
        rows: g.kernel("kda_gates_w8a8", "kda_gates_w8a8_k128_rows")?,
    };
    let mut rng = Lcg(0x6b64_6167);
    let l = layer(g, &mut rng)?;
    println!("KDA gates rows gate: the unfused six-launch chain is the ORACLE (N {N}, K {K})\n");
    for m in ROWS_TESTED {
        let (f, gl) = low_rows(&mut rng, m);
        let (a, b) = (bufs(g, m, &f, &gl)?, bufs(g, m, &f, &gl)?);
        old(g, &k, &l, &a, 0)?;
        fused(g, &k, &l, &b, false, 0)?;
        let (oa, ob) = (outs(g, &a)?, outs(g, &b)?);
        for ((what, x), y) in NAMES.iter().zip(&oa).zip(&ob) {
            same(&format!("M={m} {what}"), x, y)?;
        }
        if m == 1 {
            let c = bufs(g, m, &f, &gl)?;
            fused(g, &k, &l, &c, true, 0)?;
            for ((what, x), y) in NAMES.iter().zip(&ob).zip(&outs(g, &c)?) {
                same(&format!("M=1 rows kernel vs one-token kernel {what}"), x, y)?;
            }
        }
        // 2026-10-21: One value of the last row set to half its row's amax: a different E4M3 code
        // (a one-ulp BF16 move usually encodes to the same one) with the row's scale unchanged.
        let mut gl2 = gl.clone();
        let row = &gl[(m - 1) * K..m * K];
        let amax = row
            .iter()
            .filter(|v| !v.is_nan())
            .fold(0f32, |a, v| a.max(v.abs()));
        let last = (m - 1) * K + 3;
        gl2[last] = if gl2[last] == amax * 0.5 {
            amax * 0.25
        } else {
            amax * 0.5
        };
        let c = bufs(g, m, &f, &gl2)?;
        fused(g, &k, &l, &c, false, 0)?;
        if outs(g, &c)?[2] == oa[2] {
            bail!("M={m}: control (one g_low value of the last row) did not move out_gate");
        }
        println!("  M={m}  g_raw/gate/out_gate/beta BYTE-IDENTICAL; control differs");
    }

    if timing {
        let s = g.create_stream()?;
        let layers: Vec<Layer> = (0..KDA_LAYERS)
            .map(|_| layer(g, &mut rng))
            .collect::<Result<_>>()?;
        let plan = Plan {
            calls: KDA_LAYERS,
            rounds: 5,
            trials: 9,
            warm: std::time::Duration::from_millis(300),
        };
        println!(
            "\n  TIMING: {KDA_LAYERS} layers (distinct weights) per graph, us per layer median [min, max]"
        );
        for m in ROWS_TIMED {
            let (f, gl) = low_rows(&mut rng, m);
            let (a, b) = (bufs(g, m, &f, &gl)?, bufs(g, m, &f, &gl)?);
            let arm_old = |st: u64, i: usize| old(g, &k, &layers[i], &a, st);
            let arm_new = |st: u64, i: usize| fused(g, &k, &layers[i], &b, false, st);
            let r = time_arms(g, s, plan, &[&arm_old, &arm_new])?;
            println!(
                "    M={m}  unfused {:6.2} [{:6.2}, {:6.2}]  rows {:6.2} [{:6.2}, {:6.2}]  saves {:.3} ms/step at {KDA_LAYERS} layers",
                r[0].0,
                r[0].1,
                r[0].2,
                r[1].0,
                r[1].1,
                r[1].2,
                (r[0].0 - r[1].0) * KDA_LAYERS as f64 / 1000.0
            );
        }
    }
    println!("\nKDA gates rows gate PASS");
    Ok(())
}
