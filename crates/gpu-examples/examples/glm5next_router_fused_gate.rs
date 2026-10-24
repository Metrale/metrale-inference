// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: Byte gate and microbench of `--glm-router-fused` (R2): `glm5next_router_gemv_topk_few`
//! (one launch) against the pair it replaces, `dense_gemv_bf16_batchm_few_fp32out` then
//! `glm5next_router_topk_rank`, at GLM-5.3's router (288 experts x 4096, top 8, routed scale 2.5,
//! renormalised).
//!
//! Owner: gpu examples (GLM-5.3 MoE router).
//! Invariants:
//! - Fails unless the logits, ids and weights match byte for byte at M = 2..=8 on four input
//!   families: drawn values; tie-forcing (duplicated router rows and equal biases, so experts tie
//!   on their choice and the lower id must win); saturated sigmoids (inputs x100); a NaN logit
//!   (one router row NaN: never selected).
//! - Fails unless a CUDA graph of 1000 fused launches leaves every replay's bytes equal and the
//!   ticket back at 0.
//! - Negative control must DIFFER: the bias of the first chosen expert lowered by 10 changes the ids.
//! - `sabotage` as the first argument starts the ticket at 1 (not 0): the gate must then fail.
//! - Times both arms as graphs of 42 calls (one per MoE layer) with distinct router weights.
//!
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-gpu-examples --release --example glm5next_router_fused_gate \
//!       --features cuda,gpu-examples [-- sabotage]

#[path = "common/glm5next_rows_fixture.rs"]
mod fixture;
#[path = "common/graph_timing.rs"]
mod timing;

use anyhow::{Result, bail, ensure};
use fixture::{Lcg, read, up, up_bf16, up_f32};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

const N: usize = 288;
const K: usize = 4096;
const TOP_K: usize = 8;
const SCALE: f32 = 2.5;
const LAYERS: usize = 42;
const MMAX: usize = 8;

struct Kernels {
    few: KernelHandle,
    rank: KernelHandle,
    fused: KernelHandle,
}

/// 2026-10-24: Device buffers of one arm.
struct Bufs {
    logits: DevicePtr,
    ids: DevicePtr,
    wts: DevicePtr,
    ctr: DevicePtr,
}

impl Bufs {
    fn alloc(g: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            logits: g.alloc(MMAX * N * 4)?,
            ids: g.alloc(MMAX * TOP_K * 4)?,
            wts: g.alloc(MMAX * TOP_K * 4)?,
            ctr: up(g, &[0u8; 4])?,
        })
    }
    fn poison(&self, g: &dyn GpuBackend) -> Result<()> {
        for (p, n) in [
            (self.logits, MMAX * N * 4),
            (self.ids, MMAX * TOP_K * 4),
            (self.wts, MMAX * TOP_K * 4),
        ] {
            g.copy_h2d(&vec![0xABu8; n], p)?;
        }
        Ok(())
    }
    fn outs(&self, g: &dyn GpuBackend, m: usize) -> Result<[Vec<u8>; 3]> {
        Ok([
            read(g, self.logits, m * N * 4)?,
            read(g, self.ids, m * TOP_K * 4)?,
            read(g, self.wts, m * TOP_K * 4)?,
        ])
    }
}

fn pair(
    g: &dyn GpuBackend,
    k: &Kernels,
    x: DevicePtr,
    w: DevicePtr,
    bias: DevicePtr,
    b: &Bufs,
    m: usize,
    s: u64,
) -> Result<()> {
    KernelLaunch::new(g, k.few)
        .grid([N.div_ceil(4) as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_ptr(w)
        .arg_ptr(b.logits)
        .arg_u32(m as u32)
        .arg_u32(N as u32)
        .arg_u32(K as u32)
        .arg_u32(N as u32)
        .launch(s)?;
    KernelLaunch::new(g, k.rank)
        .grid([m as u32, 1, 1])
        .block([N.div_ceil(32) as u32 * 32, 1, 1])
        .arg_ptr(b.logits)
        .arg_ptr(bias)
        .arg_ptr(b.ids)
        .arg_ptr(b.wts)
        .arg_u32(N as u32)
        .arg_u32(TOP_K as u32)
        .arg_u32(1)
        .arg_f32(SCALE)
        .arg_u32(1)
        .arg_u32(0)
        .launch(s)
}

/// 2026-10-24: The launch `glm5next_mlp::forward::router_fused::try_launch` makes.
fn fused(
    g: &dyn GpuBackend,
    k: &Kernels,
    x: DevicePtr,
    w: DevicePtr,
    bias: DevicePtr,
    b: &Bufs,
    m: usize,
    s: u64,
) -> Result<()> {
    KernelLaunch::new(g, k.fused)
        .grid([N.div_ceil(4) as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_ptr(w)
        .arg_ptr(b.logits)
        .arg_ptr(bias)
        .arg_ptr(b.ids)
        .arg_ptr(b.wts)
        .arg_ptr(b.ctr)
        .arg_u32(m as u32)
        .arg_u32(N as u32)
        .arg_u32(K as u32)
        .arg_u32(TOP_K as u32)
        .arg_f32(SCALE)
        .arg_u32(1)
        .arg_u32(0)
        .launch(s)
}

/// 2026-10-24: Router weights, bias and inputs of one family.
fn family(rng: &mut Lcg, f: &str) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut w = rng.vec(N * K, 0.02);
    let mut bias = rng.vec(N, 0.1);
    let mut x = rng.vec(MMAX * K, 1.0);
    match f {
        "ties" => {
            for e in (1..N).step_by(3) {
                let (src, dst) = ((e - 1) * K, e * K);
                w.copy_within(src..src + K, dst);
                bias[e] = bias[e - 1];
            }
        }
        "saturated" => x.iter_mut().for_each(|v| *v *= 100.0),
        "nan" => w[17 * K..18 * K].iter_mut().for_each(|v| *v = f32::NAN),
        _ => {}
    }
    (w, bias, x)
}

fn main() -> Result<()> {
    let sabotage = std::env::args().nth(1).as_deref() == Some("sabotage");
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;
    let k = Kernels {
        few: g.kernel(
            "dense_gemv_bf16_batchm_few",
            "dense_gemv_bf16_batchm_few_fp32out",
        )?,
        rank: g.kernel("glm5next_ffn", "glm5next_router_topk_rank")?,
        fused: g.kernel("glm5next_router_fused", "glm5next_router_gemv_topk_few")?,
    };
    let (a, b) = (Bufs::alloc(g)?, Bufs::alloc(g)?);
    if sabotage {
        g.copy_h2d(&1u32.to_le_bytes(), b.ctr)?;
    }
    let mut rng = Lcg(0x726f_7574_6566_7573);
    let mut failed = false;
    println!(
        "glm5next router GEMV + top-k fused gate: N={N} K={K} top_k={TOP_K}{}",
        if sabotage { " (SABOTAGE)" } else { "" }
    );
    for f in ["drawn", "ties", "saturated", "nan"] {
        let (w, bias, x) = family(&mut rng, f);
        let (wd, bd, xd) = (up_bf16(g, &w)?, up_f32(g, &bias)?, up_bf16(g, &x)?);
        for m in 2..=MMAX {
            a.poison(g)?;
            b.poison(g)?;
            pair(g, &k, xd, wd, bd, &a, m, 0)?;
            fused(g, &k, xd, wd, bd, &b, m, 0)?;
            g.synchronize(0)?;
            let (want, got) = (a.outs(g, m)?, b.outs(g, m)?);
            let bad: Vec<&str> = ["logits", "ids", "wts"]
                .iter()
                .zip(want.iter().zip(&got))
                .filter(|(_, (w, g))| w != g)
                .map(|(n, _)| *n)
                .collect();
            if !bad.is_empty() || m == 2 {
                println!(
                    "  {f:<9} M={m}: {}",
                    if bad.is_empty() {
                        "SAME".to_string()
                    } else {
                        format!("DIFF {bad:?}")
                    }
                );
            }
            failed |= !bad.is_empty();
        }
        // 2026-10-24: Control: bias[want's first id of row 0] lowered by 10 moves the ids.
        let first = i32::from_le_bytes(a.outs(g, 2)?[1][..4].try_into()?);
        let mut bias2 = bias.clone();
        bias2[first.max(0) as usize] -= 10.0;
        let bd2 = up_f32(g, &bias2)?;
        b.poison(g)?;
        fused(g, &k, xd, wd, bd2, &b, 2, 0)?;
        let moved = b.outs(g, 2)?[1] != a.outs(g, 2)?[1];
        println!(
            "  {f:<9} control (bias of id {first} - 10): {}",
            if moved {
                "ids differ (ok)"
            } else {
                "SAME (BAD)"
            }
        );
        failed |= !moved;
    }
    // 2026-10-24: Graph replay: 1000 fused launches in one graph; every replay's bytes and the ticket.
    let (w, bias, x) = family(&mut rng, "drawn");
    let (wd, bd, xd) = (up_bf16(g, &w)?, up_f32(g, &bias)?, up_bf16(g, &x)?);
    pair(g, &k, xd, wd, bd, &a, 2, 0)?;
    let want = a.outs(g, 2)?;
    let s = g.create_stream()?;
    g.begin_capture(s)?;
    for _ in 0..1000 {
        fused(g, &k, xd, wd, bd, &b, 2, s)?;
    }
    let graph = g.end_capture(s)?;
    for _ in 0..3 {
        g.launch_graph(graph, s)?;
    }
    g.synchronize(s)?;
    g.destroy_graph(graph)?;
    let ctr = u32::from_le_bytes(read(g, b.ctr, 4)?[..4].try_into()?);
    let replay_same = b.outs(g, 2)? == want;
    println!(
        "  graph x3000: bytes {} | ticket {ctr}",
        if replay_same { "SAME" } else { "DIFF" }
    );
    failed |= !replay_same || ctr != 0;
    if failed {
        bail!("glm5next_router_gemv_topk_few gate FAILED");
    }
    // 2026-10-24: Timing: 42 calls with distinct router weights per graph.
    let ws: Vec<DevicePtr> = (0..LAYERS)
        .map(|_| up_bf16(g, &rng.vec(N * K, 0.02)))
        .collect::<Result<_>>()?;
    for m in [2usize] {
        let mut i = 0usize;
        let old = timing::time_graph(g, s, LAYERS, &mut |st| {
            i = (i + 1) % LAYERS;
            pair(g, &k, xd, ws[i], bd, &a, m, st)
        })?;
        let new = timing::time_graph(g, s, LAYERS, &mut |st| {
            i = (i + 1) % LAYERS;
            fused(g, &k, xd, ws[i], bd, &b, m, st)
        })?;
        timing::report("router + top-k", m, old, new);
    }
    ensure!(
        u32::from_le_bytes(read(g, b.ctr, 4)?[..4].try_into()?) == 0,
        "ticket not 0 after timing"
    );
    println!(
        "\nPASS: fused logits, ids and weights equal the pair's; replays and the control as required"
    );
    Ok(())
}
