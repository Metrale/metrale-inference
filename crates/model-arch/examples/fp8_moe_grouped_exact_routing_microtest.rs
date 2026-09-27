// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Byte parity of the grouped FP8 MoE decode's exact routings against the
//! routers they replace, at Qwen3.6-35B-A3B shapes (hidden 2048, 256 experts, top-8):
//! - `GroupedRouting::PerRow`: `dense_gemv_bf16_batchm` in one launch with 4, 16 or 1
//!   rows per block row (2026-09-27), then `moe_topk_softmax_rows`, against
//!   `MoeLayer::forward`'s router, `dense_gemv_bf16` + `moe_topk_softmax` once per row.
//! - `GroupedRouting::PerToken`: `dense_gemm_bf16` over all rows, then
//!   `moe_topk_softmax_rows`, against `MoeLayer::forward_batched`'s router, the same GEMM
//!   then `moe_topk_softmax` once per row.
//!
//! 2026-09-26: Also `moe_router_gemm_bf16` (`router_gemm_bf16`, used by every
//! BF16 decode router) against `dense_gemm_bf16` on the same operands, byte for
//! byte at M = 1..64, with both legs timed.
//!
//! The expert and blend half of the grouped decode is checked against the per-row
//! kernels by `fp8_moe_grouped_decode_microtest`; with this test, the whole grouped MoE
//! output is shown equal to the per-row path's bytes.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Returns an error unless, for every M and both `normalize` settings, the logits,
//!   expert indices and weights of each exact routing equal their reference's bytes.
//! - Rows are built with exact logit ties (gate rows duplicated), where the tie-break
//!   decides the slot order. It runs at 256 experts (the model) and at 512, where some
//!   ties pair a lower lane with a higher expert id. There the control leg,
//!   `moe_topk_softmax_batched` on the same logits, must differ from the reference on
//!   at least one row, or the ties were not exercised and the run fails. At 256
//!   experts each thread holds one expert, so the batched kernel's lower-lane
//!   tie-break is the lower index too and the control is not expected to differ.
//!
//! Run (GB10):
//!   cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!     --example fp8_moe_grouped_exact_routing_microtest

use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::DenseWeight;

const H: usize = 2048;
const E_MAX: usize = 512;
const TOP_K: usize = 8;
const MAX_M: usize = 64;
const MS: [usize; 13] = [2, 3, 4, 5, 8, 15, 16, 17, 31, 32, 33, 48, 64];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn bf16(&mut self, scale: f32) -> [u8; 2] {
        bf16::from_f32(((self.next() % 2049) as f32 - 1024.0) / 1024.0 * scale)
            .to_bits()
            .to_le_bytes()
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len().max(16))?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

fn read(gpu: &dyn GpuBackend, ptr: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; n];
    gpu.copy_d2h(ptr, &mut buf)?;
    Ok(buf)
}

/// 2026-09-26: Gate weight `[e, H]` with repeated rows: expert `2i + 1` copies expert
/// `2i` for `i < 64`, and with 512 experts expert `256 + j` copies expert `j + 5` for
/// `j < 128` (lane `j` of the top-k block against the higher lane `j + 5`), so rows have
/// exact logit ties among the experts that compete for the top-k slots.
fn tied_gate(rng: &mut Rng, e: usize) -> Vec<u8> {
    let mut rows: Vec<Vec<u8>> = (0..e)
        .map(|_| (0..H).flat_map(|_| rng.bf16(0.05)).collect())
        .collect();
    for i in 0..64 {
        rows[2 * i + 1] = rows[2 * i].clone();
    }
    if e > 256 {
        for j in 0..128 {
            rows[256 + j] = rows[j + 5].clone();
        }
    }
    rows.concat()
}

struct Kernels {
    gemv: KernelHandle,
    gemv_m: KernelHandle,
    gemm: KernelHandle,
    topk: KernelHandle,
    topk_rows: KernelHandle,
    topk_batched: KernelHandle,
}

struct Bufs {
    e: usize,
    logits_ref: DevicePtr,
    logits_new: DevicePtr,
    idx_ref: DevicePtr,
    w_ref: DevicePtr,
    idx_new: DevicePtr,
    w_new: DevicePtr,
}

/// 2026-09-26: `topk` once per row of `logits`, as `MoeLayer::forward` and
/// `forward_batched` run it.
fn topk_per_row(gpu: &dyn GpuBackend, k: &Kernels, b: &Bufs, m: usize, norm: bool) -> Result<()> {
    for t in 0..m {
        ops::moe_topk_softmax(
            gpu,
            k.topk,
            b.logits_ref.offset(t * b.e * 2),
            b.idx_ref.offset(t * TOP_K * 4),
            b.w_ref.offset(t * TOP_K * 4),
            b.e as u32,
            TOP_K as u32,
            norm,
            0,
        )?;
    }
    Ok(())
}

fn topk_all(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    b: &Bufs,
    m: usize,
    norm: bool,
) -> Result<()> {
    ops::moe_topk_softmax_batched(
        gpu,
        kernel,
        b.logits_new,
        b.idx_new,
        b.w_new,
        b.e as u32,
        TOP_K as u32,
        norm,
        m as u32,
        0,
    )
}

/// 2026-09-26: Whether the new leg's logits, indices and weights equal the reference's bytes.
fn same(gpu: &dyn GpuBackend, b: &Bufs, m: usize) -> Result<(bool, bool)> {
    gpu.synchronize(0)?;
    let n = m * b.e * 2;
    let logits = read(gpu, b.logits_ref, n)? == read(gpu, b.logits_new, n)?;
    let routing = read(gpu, b.idx_ref, m * TOP_K * 4)? == read(gpu, b.idx_new, m * TOP_K * 4)?
        && read(gpu, b.w_ref, m * TOP_K * 4)? == read(gpu, b.w_new, m * TOP_K * 4)?;
    Ok((logits, routing))
}

/// 2026-09-26: `moe_router_gemm_bf16` against `dense_gemm_bf16` at E = 256: the
/// number of M values whose output bytes differ, with the mean time of each leg.
fn router_gemm_parity(gpu: &dyn GpuBackend, gate: &DenseWeight, input: DevicePtr) -> Result<usize> {
    let gemm = gpu.kernel("gemm", "dense_gemm_bf16")?;
    let fast = gpu.kernel("moe_router_gemm", "moe_router_gemm_bf16")?;
    let (e, h) = (256u32, H as u32);
    let out_ref = gpu.alloc(MAX_M * 256 * 2)?;
    let out_new = gpu.alloc(MAX_M * 256 * 2)?;
    let mut failures = 0usize;
    for m in [1u32, 2, 3, 4, 8, 16, 17, 32, 33, 64] {
        let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
            gpu.synchronize(0)?;
            let t = std::time::Instant::now();
            for _ in 0..50 {
                f()?;
            }
            gpu.synchronize(0)?;
            Ok(t.elapsed().as_secs_f64() * 1e6 / 50.0)
        };
        let us_ref = time(&|| ops::dense_gemm(gpu, gemm, input, gate, out_ref, m, e, h, 0))?;
        let us_new = time(&|| ops::moe_router_gemm(gpu, fast, input, gate, out_new, m, e, h, 0))?;
        let n = m as usize * 256 * 2;
        let same = read(gpu, out_ref, n)? == read(gpu, out_new, n)?;
        println!(
            "router GEMM M={m:2}: dense_gemm {us_ref:7.1}us, moe_router_gemm {us_new:7.1}us, bytes equal {same}"
        );
        failures += usize::from(!same);
    }
    Ok(failures)
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let k = Kernels {
        gemv: gpu.kernel("gemv", "dense_gemv_bf16")?,
        gemv_m: gpu.kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")?,
        gemm: gpu.kernel("gemm", "dense_gemm_bf16")?,
        topk: gpu.kernel("moe_topk", "moe_topk_softmax")?,
        topk_rows: gpu.kernel("moe_topk", "moe_topk_softmax_rows")?,
        topk_batched: gpu.kernel("moe_topk", "moe_topk_softmax_batched")?,
    };
    let mut rng = Rng(0x6578_6163_7432_0926);
    let input: Vec<u8> = (0..MAX_M * H).flat_map(|_| rng.bf16(1.0)).collect();
    let input = upload(&gpu, &input)?;
    let h = H as u32;
    let mut failures = 0usize;
    let mut control_differs = 0usize;
    for e_count in [256usize, E_MAX] {
        let gate = DenseWeight {
            weight: upload(&gpu, &tied_gate(&mut rng, e_count))?,
        };
        if e_count == 256 {
            failures += router_gemm_parity(&gpu, &gate, input)?;
        }
        let b = Bufs {
            e: e_count,
            logits_ref: gpu.alloc(MAX_M * e_count * 2)?,
            logits_new: gpu.alloc(MAX_M * e_count * 2)?,
            idx_ref: gpu.alloc(MAX_M * TOP_K * 4)?,
            w_ref: gpu.alloc(MAX_M * TOP_K * 4)?,
            idx_new: gpu.alloc(MAX_M * TOP_K * 4)?,
            w_new: gpu.alloc(MAX_M * TOP_K * 4)?,
        };
        let e = e_count as u32;
        for norm in [true, false] {
            for m in MS {
                // 2026-09-26: PerRow: reference `dense_gemv_bf16` + `moe_topk_softmax` per row.
                for t in 0..m {
                    let (inp, out) = (input.offset(t * H * 2), b.logits_ref.offset(t * b.e * 2));
                    ops::dense_gemv(&gpu, k.gemv, inp, &gate, out, e, h, 0)?;
                }
                topk_per_row(&gpu, &k, &b, m, norm)?;
                // 2026-09-27: One `dense_gemv_batchm_split` launch, at the router's 4 rows
                // per block row (`forward_fp8_grouped_router.rs`), 16 and 1.
                for per_block in [4usize, 16, 1] {
                    ops::dense_gemv_batchm_split(
                        &gpu,
                        k.gemv_m,
                        input,
                        &gate,
                        b.logits_new,
                        m as u32,
                        m.div_ceil(per_block) as u32,
                        e,
                        h,
                        e,
                        0,
                    )?;
                    topk_all(&gpu, k.topk_rows, &b, m, norm)?;
                    let (lg, rt) = same(&gpu, &b, m)?;
                    println!(
                        "E={e_count} PerRow/{per_block:2} norm={norm:5} M={m:2}: logits {lg}, routing {rt}"
                    );
                    failures += usize::from(!(lg && rt));
                }

                // 2026-09-26: Control: the batched tie-break on the same logits.
                topk_all(&gpu, k.topk_batched, &b, m, norm)?;
                let (_, rt_control) = same(&gpu, &b, m)?;
                control_differs += usize::from(!rt_control);

                // 2026-09-26: PerToken: one `dense_gemm_bf16` feeds both legs; the
                // reference runs `moe_topk_softmax` per row, the new leg
                // `moe_topk_softmax_rows`.
                ops::dense_gemm(&gpu, k.gemm, input, &gate, b.logits_ref, m as u32, e, h, 0)?;
                ops::dense_gemm(&gpu, k.gemm, input, &gate, b.logits_new, m as u32, e, h, 0)?;
                topk_per_row(&gpu, &k, &b, m, norm)?;
                topk_all(&gpu, k.topk_rows, &b, m, norm)?;
                let (lg, rt) = same(&gpu, &b, m)?;
                println!("E={e_count} PerToken norm={norm:5} M={m:2}: logits {lg}, routing {rt}");
                failures += usize::from(!(lg && rt));
            }
        }
    }
    println!(
        "control (moe_topk_softmax_batched) differed from the reference in {control_differs} case(s)"
    );
    ensure!(
        control_differs > 0,
        "the control never differed: the tied rows did not exercise the tie-break"
    );
    ensure!(failures == 0, "{failures} case(s) not byte-identical");
    println!("ALL PASS: exact grouped routings are byte-identical to the routers they replace");
    Ok(())
}
