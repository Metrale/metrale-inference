// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: Nemotron-H's FP32 router (`launch_route_f32`: `nemotron_router_f32`, then
//! `nemotron_moe_topk_sigmoid_batched_f32`) against an FP64 host reference of HF's
//! `NemotronHTopkRouter` (FP32 logits of the BF16 input and the FP32 weight, sigmoid, plus the
//! correction bias for selection, top-k, normalized and scaled weights).
//!
//! With `NEMOTRON_ROUTER_CASES=<dir>` it also routes recorded HF router inputs (files written by a
//! capture script: `L<l>.gate` = W f32 [E, H] then bias f32 [E]; `L<l>_<name>.case` = u32 rows,
//! x f32 [rows, H], HF's top-k u32 [rows, k]) through the FP32 router and through the BF16 router
//! it replaces (`dense_gemv_bf16` on the weight cast to BF16, BF16 logits, `moe_topk_sigmoid`),
//! and counts the tokens whose expert set differs from HF's.
//!
//! Owner: model-engine tests.
//! Invariants: none beyond the types.
//!
//! Run on a GB10 with an external timeout:
//! ```text
//! METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=nemotron-3-nano-30b-a3b METRALE_TARGET_QUANT=nvfp4 \
//! cargo test -p metrale-model-engine --test nemotron_router_f32_cuda_oracle --no-run
//! NEMOTRON_ROUTER_GPU_ORDINAL=0 timeout 180s cargo test -p metrale-model-engine \
//! --test nemotron_router_f32_cuda_oracle -- --ignored --nocapture --test-threads=1
//! ```
//! Keep the same METRALE_TARGET_* values and target directory for both commands.

#![cfg(feature = "cuda")]

use anyhow::{Context, Result, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::nemotron_moe::{RouteF32Io, RouteF32Shape, RouterF32, launch_route_f32};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::DenseWeight;

const H: usize = 2688;
const E: usize = 128;
const K: usize = 6;
const SCALE: f32 = 2.5;

struct Mem<'a> {
    gpu: &'a dyn GpuBackend,
    pointers: Vec<DevicePtr>,
}

impl Mem<'_> {
    fn upload(&mut self, bytes: &[u8]) -> Result<DevicePtr> {
        let p = self.gpu.alloc(bytes.len())?;
        self.pointers.push(p);
        self.gpu.copy_h2d(bytes, p)?;
        Ok(p)
    }
    fn zeroed(&mut self, bytes: usize) -> Result<DevicePtr> {
        self.upload(&vec![0; bytes])
    }
}

impl Drop for Mem<'_> {
    fn drop(&mut self) {
        for p in self.pointers.drain(..).rev() {
            let _ = self.gpu.free(p);
        }
    }
}

fn le_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn le_u32(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read(gpu: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut raw = vec![0; bytes];
    gpu.copy_d2h(p, &mut raw)?;
    Ok(raw)
}

/// 2026-10-02: HF's routing of one token in FP64: logits, the sorted top-k ids, the weight of
/// each id, and the selection-score gap between the k-th and the (k+1)-th expert.
struct Reference {
    logits: Vec<f64>,
    ids: Vec<u32>,
    weights: Vec<(u32, f64)>,
    gap: f64,
    abs_dot: Vec<f64>,
}

fn reference(x: &[f32], w: &[f32], bias: &[f32]) -> Reference {
    let logits: Vec<f64> = (0..E)
        .map(|e| (0..H).map(|i| x[i] as f64 * w[e * H + i] as f64).sum())
        .collect();
    let abs_dot = (0..E)
        .map(|e| {
            (0..H)
                .map(|i| (x[i] as f64 * w[e * H + i] as f64).abs())
                .sum()
        })
        .collect();
    let sig: Vec<f64> = logits.iter().map(|l| 1.0 / (1.0 + (-l).exp())).collect();
    let mut order: Vec<usize> = (0..E).collect();
    order.sort_by(|&a, &b| (sig[b] + bias[b] as f64).total_cmp(&(sig[a] + bias[a] as f64)));
    let sel = |r: usize| sig[order[r]] + bias[order[r]] as f64;
    let sum: f64 = order[..K].iter().map(|&e| sig[e]).sum::<f64>() + 1e-20;
    let mut ids: Vec<u32> = order[..K].iter().map(|&e| e as u32).collect();
    ids.sort_unstable();
    let weights = order[..K]
        .iter()
        .map(|&e| (e as u32, sig[e] / sum * SCALE as f64))
        .collect();
    Reference {
        logits,
        ids,
        weights,
        gap: sel(K - 1) - sel(K),
        abs_dot,
    }
}

/// 2026-10-02: The routers under test, resolved once.
struct Routers {
    f32: RouterF32,
    gemv: metrale_gpu_runtime::gpu::KernelHandle,
    topk: metrale_gpu_runtime::gpu::KernelHandle,
}

/// 2026-10-02: One layer's router on the device: the FP32 weight, its BF16 cast and the bias.
struct Gate {
    w: Vec<f32>,
    bias: Vec<f32>,
    w_dev: DevicePtr,
    w_bf16_dev: DevicePtr,
    bias_dev: DevicePtr,
}

fn gate(mem: &mut Mem, w: Vec<f32>, bias: Vec<f32>) -> Result<Gate> {
    let w_dev = mem.upload(&w.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
    let w_bf16_dev = mem.upload(
        &w.iter()
            .flat_map(|&v| bf16::from_f32(v).to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let bias_dev = mem.upload(
        &bias
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    Ok(Gate {
        w,
        bias,
        w_dev,
        w_bf16_dev,
        bias_dev,
    })
}

/// 2026-10-02: Route `x` [rows, H] through the FP32 router: (logits, sorted ids per row, weights
/// by id per row).
#[allow(clippy::type_complexity)]
fn route_f32(
    gpu: &dyn GpuBackend,
    r: &Routers,
    g: &Gate,
    x: &[f32],
    stream: u64,
) -> Result<(Vec<f32>, Vec<Vec<u32>>, Vec<Vec<(u32, f32)>>)> {
    let rows = x.len() / H;
    let mut mem = Mem {
        gpu,
        pointers: Vec::new(),
    };
    let io = RouteF32Io {
        normed: mem.upload(
            &x.iter()
                .flat_map(|&v| bf16::from_f32(v).to_le_bytes())
                .collect::<Vec<_>>(),
        )?,
        gate: g.w_dev,
        bias: g.bias_dev,
        logits: mem.zeroed(rows * E * 4)?,
        indices: mem.zeroed(rows * K * 4)?,
        weights: mem.zeroed(rows * K * 4)?,
    };
    let shape = RouteF32Shape {
        n: rows as u32,
        num_experts: E as u32,
        hidden: H as u32,
        top_k: K as u32,
        normalize: true,
        scale: SCALE,
    };
    launch_route_f32(gpu, &r.f32, &io, shape, stream)?;
    gpu.synchronize(stream)?;
    let logits = le_f32(&read(gpu, io.logits, rows * E * 4)?);
    let ids = le_u32(&read(gpu, io.indices, rows * K * 4)?);
    let wts = le_f32(&read(gpu, io.weights, rows * K * 4)?);
    let mut sorted = Vec::new();
    let mut by_id = Vec::new();
    for t in 0..rows {
        let row: Vec<(u32, f32)> = (0..K).map(|j| (ids[t * K + j], wts[t * K + j])).collect();
        let mut s: Vec<u32> = row.iter().map(|p| p.0).collect();
        s.sort_unstable();
        sorted.push(s);
        by_id.push(row);
    }
    Ok((logits, sorted, by_id))
}

/// 2026-10-02: Route one token through the BF16 router the FP32 one replaces; sorted ids.
fn route_bf16(
    gpu: &dyn GpuBackend,
    r: &Routers,
    g: &Gate,
    x: &[f32],
    stream: u64,
) -> Result<Vec<u32>> {
    let mut mem = Mem {
        gpu,
        pointers: Vec::new(),
    };
    let input = mem.upload(
        &x.iter()
            .flat_map(|&v| bf16::from_f32(v).to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let logits = mem.zeroed(E * 2)?;
    let (ids, wts) = (mem.zeroed(K * 4)?, mem.zeroed(K * 4)?);
    let w = DenseWeight {
        weight: g.w_bf16_dev,
    };
    ops::dense_gemv(gpu, r.gemv, input, &w, logits, E as u32, H as u32, stream)?;
    ops::moe_topk_sigmoid(
        gpu, r.topk, logits, g.bias_dev, ids, wts, E as u32, K as u32, true, SCALE, stream,
    )?;
    gpu.synchronize(stream)?;
    let mut s = le_u32(&read(gpu, ids, K * 4)?);
    s.sort_unstable();
    Ok(s)
}

fn xorshift(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    ((*state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

#[test]
#[ignore = "requires an explicitly selected idle CUDA device and compiled Nemotron kernels"]
fn the_fp32_router_routes_as_hf() -> Result<()> {
    let ordinal = std::env::var("NEMOTRON_ROUTER_GPU_ORDINAL")?.parse()?;
    let target = metrale_kernels::ptx_for_exact_target("nemotron-3-nano-30b-a3b", "nvfp4")
        .context("Nemotron-3-Nano target")?;
    let gpu = MetraleCudaBackend::new(ordinal, &target.modules)?;
    let stream = gpu.create_stream()?;
    let r = Routers {
        f32: RouterF32::load(&gpu, true)?.context("an FP32 router loads its kernels")?,
        gemv: gpu.kernel("gemv", "dense_gemv_bf16")?,
        topk: gpu.kernel("moe_topk_sig", "moe_topk_sigmoid")?,
    };
    let mut mem = Mem {
        gpu: &gpu,
        pointers: Vec::new(),
    };

    // 2026-10-02: Synthetic: BF16 inputs of RMS-norm scale, FP32 weights of the checkpoint's
    // scale (|w| ~ 0.02) with low mantissa bits a BF16 cast drops, a small bias.
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let rows = 9;
    let x: Vec<f32> = (0..rows * H)
        .map(|_| bf16::from_f32(xorshift(&mut s) * 2.0).to_f32())
        .collect();
    let w: Vec<f32> = (0..E * H).map(|_| xorshift(&mut s) * 0.02).collect();
    let bias: Vec<f32> = (0..E).map(|_| xorshift(&mut s) * 1e-3).collect();
    let g = gate(&mut mem, w, bias)?;
    let (logits, ids, weights) = route_f32(&gpu, &r, &g, &x, stream)?;
    for t in 0..rows {
        let rf = reference(&x[t * H..(t + 1) * H], &g.w, &g.bias);
        // 2026-10-02: FP32 sums of H products in 32 lanes then a 5-step tree: each logit is
        // within (H / 32 + 5) * 2^-24 of the sum of |products|.
        let bound =
            |e: usize| (H as f64 / 32.0 + 5.0) * f64::from(f32::EPSILON) / 2.0 * rf.abs_dot[e];
        for e in 0..E {
            let d = (logits[t * E + e] as f64 - rf.logits[e]).abs();
            ensure!(
                d <= bound(e),
                "row {t} expert {e}: |d| {d:e} > {:e}",
                bound(e)
            );
        }
        let worst = (0..E).map(bound).fold(0.0, f64::max);
        if rf.gap > worst {
            ensure!(
                ids[t] == rf.ids,
                "row {t}: {:?} != {:?} (gap {:e})",
                ids[t],
                rf.ids,
                rf.gap
            );
            for &(e, wt) in &weights[t] {
                let want = rf.weights.iter().find(|p| p.0 == e).context("selected")?.1;
                ensure!(
                    ((wt as f64) - want).abs() <= 1e-5 * want,
                    "row {t} expert {e}: {wt} != {want}"
                );
            }
        }
    }
    println!("PASS synthetic: {rows} rows, logits within the FP32 bound, ids and weights as FP64");

    let Ok(dir) = std::env::var("NEMOTRON_ROUTER_CASES") else {
        return Ok(());
    };
    let (mut n, mut miss_f32, mut miss_bf16) = (0usize, 0usize, 0usize);
    let mut gates = std::collections::BTreeMap::new();
    let mut cases: Vec<_> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "case"))
        .collect();
    cases.sort();
    for case in cases {
        let name = case
            .file_name()
            .context("name")?
            .to_string_lossy()
            .into_owned();
        let layer = name[..3].to_owned();
        if !gates.contains_key(&layer) {
            let raw = le_f32(&std::fs::read(format!("{dir}/{layer}.gate"))?);
            ensure!(raw.len() == E * H + E, "{layer}.gate size");
            let g = gate(&mut mem, raw[..E * H].to_vec(), raw[E * H..].to_vec())?;
            gates.insert(layer.clone(), g);
        }
        let g = &gates[&layer];
        let raw = std::fs::read(&case)?;
        let rows = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as usize;
        let x = le_f32(&raw[4..4 + rows * H * 4]);
        let hf = le_u32(&raw[4 + rows * H * 4..]);
        let (_, ids, _) = route_f32(&gpu, &r, g, &x, stream)?;
        for t in 0..rows {
            let mut want = hf[t * K..(t + 1) * K].to_vec();
            want.sort_unstable();
            let old = route_bf16(&gpu, &r, g, &x[t * H..(t + 1) * H], stream)?;
            miss_f32 += usize::from(ids[t] != want);
            miss_bf16 += usize::from(old != want);
            n += 1;
        }
    }
    println!(
        "HF cases: {n} token-layer routings; expert set differs from HF's: FP32 router {miss_f32}, BF16 router {miss_bf16}"
    );
    ensure!(
        miss_f32 <= miss_bf16,
        "the FP32 router routes further from HF than the BF16 one"
    );
    ensure!(
        miss_f32 == 0,
        "the FP32 router differs from HF on {miss_f32} routings"
    );
    Ok(())
}
