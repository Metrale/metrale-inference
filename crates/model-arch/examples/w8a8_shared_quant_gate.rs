// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Bit gate and timing of the shared activation quantization in the W8A8 prefill GEMM
//! (`glm5next_w8a8_gemm::proj` with `stable`): the projections of one input in a GLM-5.3 layer
//! (KDA q/k/v/f_a/b_proj/g_a, DSA q_a/kv_a/indexer) run through the production entry point, and
//! every consumer's output must equal, bit for bit, a fresh quantize + GEMM of the same input
//! (`w8a8_proj_wide` on a scratch of its own).
//!
//! `--control stale` overwrites the input after the first consumer while still vouching for it:
//! the later consumers then read the first input's quantization, so they must differ from the
//! fresh reference over the new input, and the gate must exit nonzero. It proves that the reuse
//! is armed, not only that it is harmless.
//!
//! Timing (`--control none`): each group's projections at M=8192 and 8208, quantized per
//! consumer, quantized once, and dequantize + BF16 GEMM (today's FP8 prefill path).
//!
//! Owner: model-arch examples.
//! Invariants: exits 0 only when every compared value matched ("GATE PASS").
//!
//! Run (GB10, < 3 GB of device memory; the lever must be on and the scratch one sub-chunk wide):
//!   METRALE_GLM_W8A8_GEMM_MIN_ROWS=17 METRALE_GLM_PREFILL_ROWS=8208 \
//!   cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!     --example w8a8_shared_quant_gate -- --control none

use anyhow::{Result, bail, ensure};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_fp8_wide::dequant_gemm;
use metrale_model_arch::glm5next_layer::prefill_rows;
use metrale_model_arch::glm5next_w8a8_gemm::{self as wide, TILE};
use metrale_model_layers::layers::ops::{
    W8a8GemmKernels, W8a8Kernels, W8a8Scale, W8a8Scratch, W8a8Weight, w8a8_proj_wide,
};
use metrale_model_layers::layers::{DenseQuantization, set_dense_quantization_from_cli};
use metrale_model_layers::weight_map::Fp8Weight;

#[path = "common/w8a8_fixture.rs"]
mod fixture;
use fixture::{Rng, download, median_us, seg, upload};

const K: usize = 4096;
/// 2026-10-14: The projections of one layer input on one TP=3 rank, `(name, [consumer rows])`.
const GROUPS: [(&str, &[usize]); 2] = [
    (
        "kda q|k|v|f_a|b_proj|g_a",
        &[2816, 2816, 2816, 128, 22, 128],
    ),
    ("dsa q_a|kv_a|indexer", &[1536, 512, 144]),
];
/// 2026-10-14: 17 is the lever's floor, 8208 the recipe's sub-chunk; 10000 needs two launches,
/// so it must quantize per consumer and still match.
const ROWS: [usize; 5] = [17, 1000, 8192, 8208, 10000];
const TIME_ROWS: [usize; 2] = [8192, 8208];
/// 2026-10-14: FP8 dense MAMF of GB10 (kernels/DEVICES.toml, measured).
const FP8_MAMF_TFLOPS: f64 = 207.7;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Control {
    None,
    Stale,
}

fn parse_args() -> Result<Control> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["--control", "none"] => Ok(Control::None),
        ["--control", "stale"] => Ok(Control::Stale),
        other => bail!("usage: --control {{none,stale}} (got {other:?})"),
    }
}

fn main() -> Result<()> {
    let control = parse_args()?;
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    set_dense_quantization_from_cli(DenseQuantization::Fp8);
    ensure!(
        wide::min_rows()?.is_some() && prefill_rows() >= 8208,
        "set {}=17 and METRALE_GLM_PREFILL_ROWS=8208",
        wide::MIN_ROWS_ENV
    );
    wide::reserve(&gpu, K)?;
    let k8 = W8a8Kernels::load(&gpu);
    let gk = W8a8GemmKernels::load(&gpu);
    ensure!(
        k8.resolved(W8a8Scale::PerRow) && gk.resolved(TILE),
        "the per-row W8A8 kernels or {} are not in this target",
        TILE.entry()
    );
    let quant = gpu.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?;
    let mmax = ROWS[ROWS.len() - 1];
    let mut rng = Rng(0x5a4e_d0f0_9a7e_0002);
    let mut bits =
        |n: usize| -> Vec<u8> { (0..n).flat_map(|_| rng.bf16_bits().to_le_bytes()).collect() };
    let (x_host, x_new_host) = (bits(mmax * K), bits(mmax * K));
    let x = upload(&gpu, &x_host)?;
    let x_new = upload(&gpu, &x_new_host)?;
    let q_ref = W8a8Scratch::alloc_rows(&gpu, mmax, K as u32)?;
    let nmax = GROUPS.iter().flat_map(|g| g.1).max().copied().unwrap_or(K);
    let (o_ref, o_got) = (gpu.alloc(mmax * nmax * 2)?, gpu.alloc(mmax * nmax * 2)?);
    let mut wrng = Rng(0x5a4e_0000_0000_0003);
    let (mut compared, mut mismatched) = (0usize, 0usize);
    for (name, rows) in GROUPS {
        let segs: Vec<Fp8Weight> = rows
            .iter()
            .map(|&n| seg(&gpu, &mut wrng, quant, n, K))
            .collect::<Result<_>>()?;
        let ws: Vec<W8a8Weight> = segs
            .iter()
            .map(|s| W8a8Weight::new(std::slice::from_ref(s)))
            .collect::<Result<_>>()?;
        for m in ROWS {
            // 2026-10-14: The reference input: `x`, or under the control the input the later
            // consumers are launched on after the overwrite.
            let x_ref = if control == Control::Stale { x_new } else { x };
            let mut per = Vec::new();
            wide::forget();
            for (i, w) in ws.iter().enumerate() {
                let n = rows[i];
                if control == Control::Stale && i == 1 {
                    gpu.synchronize(0)?;
                    gpu.copy_h2d(&x_new_host[..m * K * 2], x)?;
                }
                wide::proj(&gpu, &k8, w, x, o_got, m, n, K, 0, true)?;
                let got = download(&gpu, o_got, m * n * 2)?;
                w8a8_proj_wide(
                    &gpu, &k8, &gk, TILE, w, x_ref, K as u32, m, o_ref, n as u32, &q_ref, 0,
                )?;
                let want = download(&gpu, o_ref, m * n * 2)?;
                let bad = got
                    .chunks_exact(2)
                    .zip(want.chunks_exact(2))
                    .filter(|(a, b)| a != b)
                    .count();
                compared += m * n;
                mismatched += bad;
                per.push(bad);
            }
            wide::forget();
            gpu.copy_h2d(&x_host[..m * K * 2], x)?;
            println!("bits {name:<26} M={m:5}: mismatches per consumer {per:?}");
        }
        if control == Control::None {
            time_group(&gpu, &k8, &segs, &ws, x, o_got, name)?;
        }
        for s in &segs {
            gpu.free(s.weight)?;
            gpu.free(s.row_scale)?;
        }
    }
    println!("control={control:?}: {mismatched} of {compared} compared values differ");
    if mismatched > 0 {
        println!("GATE FAIL");
        bail!("{mismatched} mismatches");
    }
    println!("GATE PASS");
    Ok(())
}

/// 2026-10-14: One group's projections per launch of the group: quantized per consumer (unvouched),
/// quantized once (vouched), and dequantize + BF16 GEMM per consumer.
fn time_group(
    gpu: &dyn GpuBackend,
    k8: &W8a8Kernels,
    segs: &[Fp8Weight],
    ws: &[W8a8Weight],
    x: DevicePtr,
    out: DevicePtr,
    name: &str,
) -> Result<()> {
    let nsum: usize = segs.iter().map(|s| s.n as usize).sum();
    let nmax = segs.iter().map(|s| s.n as usize).max().unwrap_or(0);
    let dq = gpu.alloc(nmax * K * 2)?;
    for m in TIME_ROWS {
        let group = |stable: bool| {
            wide::forget();
            for (w, s) in ws.iter().zip(segs) {
                wide::proj(gpu, k8, w, x, out, m, s.n as usize, K, 0, stable)?;
            }
            Ok(())
        };
        let separate = median_us(gpu, 4, |_| group(false))?;
        let shared = median_us(gpu, 4, |_| group(true))?;
        let dequant = median_us(gpu, 4, |_| {
            for s in segs {
                let n = s.n as usize;
                dequant_gemm(gpu, s, dq, n * K, x, out, m, n, K, 0)?;
            }
            Ok(())
        })?;
        let tf = |us: f64| 2.0 * (m * nsum * K) as f64 / us / 1e6;
        let pct = |us: f64| 100.0 * tf(us) / FP8_MAMF_TFLOPS;
        println!(
            "time {name:<26} M={m:5} N={nsum}: per-consumer quant {separate:8.1} us ({:5.1} TFLOP/s, {:4.1}% MAMF) | shared quant {shared:8.1} us ({:5.1}, {:4.1}%) | dequant+gemm {dequant:8.1} us ({:5.1}, {:4.1}%) | shared/dequant {:.3}",
            tf(separate),
            pct(separate),
            tf(shared),
            pct(shared),
            tf(dequant),
            pct(dequant),
            shared / dequant
        );
    }
    wide::forget();
    gpu.free(dq)?;
    Ok(())
}
