// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Gate and microbench for the routed-MoE site of the GLM-5.3 multi-sequence prefill
//! (`Glm5NextLayer::forward_prefill_spans`): sequences whose prompts are prefilled together
//! share one `forward_moe` call over all their rows. For several sets of prompt lengths it runs
//! `forward_moe` on each sequence's rows alone (what the one-request-at-a-time prefill runs) and
//! on all of them packed sequence-major, and compares every sequence's output rows byte for
//! byte; then it times each burst of `TIMED` apart (one call per prompt) and together.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants:
//! - Exits with an error unless every sequence's rows of the packed call are byte-identical to
//!   its own call, for every case.
//!
//! The site is one rank of the three-box TP=3 / EP=3 serve (hidden 4096, 96 of 288 experts
//! bound, expert width 2048, top 8, shared slice 688) on synthetic weights, with the routed
//! experts stamped W4A4 as the checkpoint declares, so the row-count ladder is the serve's
//! (W4A4 up to 16 rows, W4A16 above). Every case keeps every sequence above 16 rows and at or
//! above `METRALE_GLM_MOE_PREFILL_GEMM_MIN_ROWS`, so the single and the packed call run the
//! same kernels; a prompt of 16 tokens or fewer runs the W4A4 tier alone and the W4A16 tier
//! when batched, which this gate does not claim to be identical.
//!
//! The mixers run per sequence in the pass, with the same rows and state as alone, and the
//! mHC and norm launches compute each token on its own (their own gates cover them).
//!
//! 2026-10-14: Under `METRALE_GLM_MOE_W4A4_GROUPED=1` (the declared serve) the routed experts run
//! W4A4 at every row count, the grouped W4A4 GEMM above 16 rows, as the serve's loader stamps
//! them; a second argument `tp` builds rank 0 of the `--moe-expert-layout tp` serve (all 288
//! experts, each sliced to 704 columns) instead of the `ep` layout. `INFO_CASES` (pieces of 16
//! rows or fewer, which the prefill grouping runs alone) are reported but never fail the gate.
//!
//! ```text
//! METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//! METRALE_GLM_MOE_PREFILL_GEMM_MIN_ROWS=17 GLM_BENCH_GPU_ORDINAL=0 \
//! cargo run -p metrale-model-arch --release --example glm5next_prefill_spans_mlp_gate \
//!     --features cuda,gpu-examples -- w4a16 [ep|tp]
//! ```
//! The first argument is the serve's `--dense-quantization` (`declared`, `fp8` or `w4a16`), the
//! second its `--moe-expert-layout` (`ep` when absent).

use anyhow::{Context, Result, bail};
use half::bf16;
use metrale_config::{ActivationQuantization, Nvfp4Act, ProjFamily};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_mlp::build::build_moe;
use metrale_model_arch::glm5next_mlp::expert_tp::expert_slice;
use metrale_model_arch::glm5next_mlp::forward::{
    Glm5NextMlpWorkspace, forward_moe, forward_moe_pieces, moe_w4a4_grouped,
};
use metrale_model_arch::glm5next_mlp::precision::{GroupPrecision, MlpGroup};
use metrale_model_arch::glm5next_mlp::weights::{Glm5NextExpertWeights, Nvfp4Proj};
use metrale_model_arch::glm5next_mlp::{Glm5NextMlpConfig, Glm5NextMlpKernels};
use metrale_model_layers::layers::{DenseQuantization, set_dense_quantization_from_cli};

/// 2026-10-09: The rows one pass may group (`METRALE_GLM_PREFILL_ROWS` in the serve recipe).
const MAX_ROWS: usize = 2048;
/// 2026-10-09: Prompt-length sets, each at most `MAX_ROWS` rows in all: the ladder's two
/// 198-token prompts, three uneven prompts, a short one before a long one, and a C16 burst of
/// 77-token prompts in one group.
/// 2026-10-14: Plus the C8 ladder burst of 196-token prompts and the 192-row pieces before a
/// block-boundary capture.
const CASES: [&[usize]; 6] = [
    &[198, 198],
    &[198, 197, 116],
    &[64, 300],
    &[77; 16],
    &[196; 8],
    &[192; 7],
];
/// 2026-10-14: The 4-row tails after a 192-row capture, packed: reported, never a failure.
const INFO_CASES: [&[usize]; 2] = [&[4; 7], &[16, 16]];
/// 2026-10-09: The bursts timed apart and together.
const TIMED: [&[usize]; 4] = [&[198, 198], &[77; 16], &[198; 8], &[4; 7]];
const TIMING_REPS: usize = 9;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

fn up(gpu: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(b.len().max(1))?;
    gpu.copy_h2d(b, p)?;
    Ok(p)
}

fn up_bf16(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    up(
        gpu,
        &v.iter()
            .flat_map(|x| bf16::from_f32(*x).to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

/// 2026-10-09: How many `row_bytes` rows of two BF16 buffers differ, and the largest absolute
/// difference between their elements.
fn diff_stats(a: &[u8], b: &[u8], row_bytes: usize) -> (usize, f32) {
    let val = |c: &[u8]| bf16::from_le_bytes([c[0], c[1]]).to_f32();
    let rows = a
        .chunks(row_bytes)
        .zip(b.chunks(row_bytes))
        .filter(|(x, y)| x != y)
        .count();
    let max = a
        .chunks(2)
        .zip(b.chunks(2))
        .map(|(x, y)| (val(x) - val(y)).abs())
        .fold(0.0f32, f32::max);
    (rows, max)
}

fn read(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// 2026-10-09: Rank 0 of GLM-5.3-Flash at TP=3 / EP=3. 2026-10-14: With `tp`, rank 0 of the
/// `--moe-expert-layout tp` serve: every expert, sliced to this rank's columns.
fn cfg(tp: bool) -> Result<Glm5NextMlpConfig> {
    let ep = Glm5NextMlpConfig {
        hidden: 4096,
        local_dense_intermediate: 4096,
        dense_start: 0,
        moe_intermediate: 2048,
        local_shared_intermediate: 688,
        shared_start: 0,
        num_experts: 288,
        local_experts: 96,
        ep_rank: 0,
        top_k: 8,
        routed_scale: 2.5,
        renormalize: true,
        swiglu_limit: 10.0,
        router_bf16_ladder: false,
        tp_world_size: 3,
        ep_world_size: 3,
        expert_shard: metrale_model_arch::glm5next_mlp::ExpertShard::Whole,
    };
    if !tp {
        return Ok(ep);
    }
    let slice = expert_slice(ep.moe_intermediate, 3, 0)?;
    Ok(Glm5NextMlpConfig {
        moe_intermediate: slice.len,
        local_experts: ep.num_experts,
        ep_world_size: 1,
        expert_shard: metrale_model_arch::glm5next_mlp::ExpertShard::Sliced(slice),
        ..ep
    })
}

fn expert_proj(
    gpu: &dyn GpuBackend,
    rng: &mut Rng,
    n: usize,
    k: usize,
    gs: f32,
) -> Result<Nvfp4Proj> {
    let packed: Vec<u8> = (0..n * k / 2).map(|_| rng.next() as u8).collect();
    let scales: Vec<u8> = (0..n * k / 16)
        .map(|_| 0x30 + (rng.next() % 9) as u8)
        .collect();
    Ok(Nvfp4Proj {
        packed: up(gpu, &packed)?,
        scale: up(gpu, &scales)?,
        scale_2: 0.01,
        input_scale: Some(gs),
    })
}

fn main() -> Result<()> {
    let dense = match std::env::args().nth(1).as_deref() {
        Some("declared") => DenseQuantization::Declared,
        Some("fp8") => DenseQuantization::Fp8,
        Some("w4a16") => DenseQuantization::W4a16,
        other => {
            bail!("usage: glm5next_prefill_spans_mlp_gate <declared|fp8|w4a16>, got {other:?}")
        }
    };
    set_dense_quantization_from_cli(dense);
    let tp = match std::env::args().nth(2).as_deref() {
        None | Some("ep") => false,
        Some("tp") => true,
        other => bail!("the second argument is the expert layout (ep or tp), got {other:?}"),
    };
    let shortest = CASES
        .iter()
        .flat_map(|c| c.iter())
        .min()
        .copied()
        .unwrap_or(0);
    let floor: usize = std::env::var("METRALE_GLM_MOE_PREFILL_GEMM_MIN_ROWS")
        .context("set METRALE_GLM_MOE_PREFILL_GEMM_MIN_ROWS to the serve recipe's value")?
        .parse()?;
    if floor > shortest {
        bail!(
            "METRALE_GLM_MOE_PREFILL_GEMM_MIN_ROWS={floor} is above the shortest case ({shortest} \
             rows): its single call would not run the grouped GEMM the packed call runs"
        );
    }
    let ordinal = std::env::var("GLM_BENCH_GPU_ORDINAL")
        .context("GLM_BENCH_GPU_ORDINAL names the GPU to run on")?
        .parse()?;
    let target = metrale_kernels::ptx_for_exact_target("glm-5.3-flash", "nvfp4")
        .context("glm-5.3-flash nvfp4 target")?;
    let gpu = MetraleCudaBackend::new(ordinal, &target.modules)?;
    // 2026-10-14: What the serve's loader runs under `--dense-quantization w4a16`: resolve the
    // W4A16 tiers and record the SM count the segmented launches (`METRALE_GLM_W4A16_SEG`) read.
    if matches!(dense, DenseQuantization::W4a16) {
        metrale_model_arch::glm5next_w4a16_dense::Nvfp4QuantKernels::load(&gpu)?;
    }
    let stream = gpu.create_stream()?;
    let (c, k) = (cfg(tp)?, Glm5NextMlpKernels::resolve(&gpu)?);
    let mi = c.moe_intermediate;
    // 2026-10-14: The routed W4A4 row cap the serve's loader stamps (`mlp_precision.rs`).
    let w4a4_rows = if moe_w4a4_grouped()? {
        k.w4a4_expert_rows_grouped()
    } else {
        k.w4a4_expert_rows()
    };
    println!(
        "layout {}, expert width {mi}, routed W4A4 up to {w4a4_rows} rows",
        if tp { "tp" } else { "ep" }
    );
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let experts: Vec<Glm5NextExpertWeights> = (0..c.local_experts)
        .map(|_| -> Result<_> {
            Ok(Glm5NextExpertWeights {
                gate_proj: expert_proj(&gpu, &mut rng, mi, 4096, 3.0 / 2688.0)?,
                up_proj: expert_proj(&gpu, &mut rng, mi, 4096, 3.0 / 2688.0)?,
                down_proj: expert_proj(&gpu, &mut rng, 4096, mi, 100.0 / 2688.0)?,
            })
        })
        .collect::<Result<_>>()?;
    let router: Vec<f32> = (0..c.num_experts * c.hidden)
        .map(|_| rng.unit() * 0.05)
        .collect();
    let full_shared = c.local_shared_intermediate * 3;
    let shared: Vec<f32> = (0..c.hidden * full_shared)
        .map(|_| rng.unit() * 0.02)
        .collect();
    let load = |n: &str| -> Result<Vec<f32>> {
        Ok(match n {
            "mlp.gate.weight" => router.clone(),
            "mlp.gate.e_score_correction_bias" => vec![0.0; c.num_experts],
            _ => shared.clone(),
        })
    };
    let expert = |id: usize| -> Result<Glm5NextExpertWeights> { Ok(experts[id]) };
    let site = build_moe(
        &gpu,
        &c,
        full_shared,
        &load,
        &expert,
        &|has| {
            GroupPrecision::resolve(
                MlpGroup::RoutedExperts,
                ActivationQuantization::default()
                    .ladder(ProjFamily::Moe)
                    .clone(),
                Nvfp4Act::A4,
                w4a4_rows,
                has,
            )
        },
        MAX_ROWS,
    )?;
    let ws = Glm5NextMlpWorkspace::new(&gpu, &c, MAX_ROWS)?;
    let row_bytes = c.hidden * 2;
    let out_one = gpu.alloc(MAX_ROWS * row_bytes)?;
    let out_all = gpu.alloc(MAX_ROWS * row_bytes)?;
    let moe = |x: DevicePtr, out: DevicePtr, rows: usize| {
        forward_moe(&gpu, &k, &c, &site, x, out, rows, &ws, false, stream)
    };

    let mut failures = 0usize;
    for (rows, counts) in CASES
        .iter()
        .map(|r| (*r, true))
        .chain(INFO_CASES.iter().map(|r| (*r, false)))
    {
        let total: usize = rows.iter().sum();
        let x = up_bf16(
            &gpu,
            &(0..total * c.hidden)
                .map(|_| rng.unit() * 3.0)
                .collect::<Vec<_>>(),
        )?;
        // 2026-10-09: The packed call as the pass makes it: one router launch per sequence.
        forward_moe_pieces(
            &gpu, &k, &c, &site, x, out_all, total, rows, &ws, false, stream,
        )?;
        gpu.synchronize(stream)?;
        let all = read(&gpu, out_all, total * row_bytes)?;
        let mut at = 0usize;
        for (s, &n) in rows.iter().enumerate() {
            moe(x.offset(at * row_bytes), out_one, n)?;
            gpu.synchronize(stream)?;
            let one = read(&gpu, out_one, n * row_bytes)?;
            let packed = &all[at * row_bytes..(at + n) * row_bytes];
            match one.iter().zip(packed).position(|(a, b)| a != b) {
                None => println!("  {rows:?} sequence {s} ({n} rows): byte-identical"),
                Some(i) => {
                    failures += usize::from(counts);
                    let (rows_off, max) = diff_stats(&one, packed, row_bytes);
                    println!(
                        "  {rows:?} sequence {s} ({n} rows): DIFFERS from row {} (byte {i}); \
                         {rows_off} rows differ, max |diff| {max:.3e}{}",
                        i / row_bytes,
                        if counts { "" } else { " (informational)" }
                    );
                }
            }
            at += n;
        }
    }

    let x = up_bf16(
        &gpu,
        &(0..MAX_ROWS * c.hidden)
            .map(|_| rng.unit() * 3.0)
            .collect::<Vec<_>>(),
    )?;
    let time = |f: &mut dyn FnMut() -> Result<()>| -> Result<Vec<f64>> {
        for _ in 0..3 {
            f()?;
        }
        gpu.synchronize(stream)?;
        let mut v = Vec::with_capacity(TIMING_REPS);
        for _ in 0..TIMING_REPS {
            let t = std::time::Instant::now();
            f()?;
            gpu.synchronize(stream)?;
            v.push(t.elapsed().as_secs_f64() * 1e3);
        }
        v.sort_by(|a, b| a.total_cmp(b));
        Ok(v)
    };
    let med = |v: &[f64]| v[v.len() / 2];
    for rows in TIMED {
        let total: usize = rows.iter().sum();
        let apart = time(&mut || {
            let mut at = 0usize;
            for &n in rows {
                moe(x.offset(at * row_bytes), out_one, n)?;
                at += n;
            }
            Ok(())
        })?;
        let together = time(&mut || {
            forward_moe_pieces(
                &gpu, &k, &c, &site, x, out_all, total, rows, &ws, false, stream,
            )
        })?;
        println!(
            "{} prompts of {:?} rows, one rank's routed site: apart {:.2} ms (min {:.2}, max {:.2}), \
             together {:.2} ms (min {:.2}, max {:.2}), median of {TIMING_REPS}",
            rows.len(),
            rows[0],
            med(&apart),
            apart[0],
            apart[TIMING_REPS - 1],
            med(&together),
            together[0],
            together[TIMING_REPS - 1]
        );
    }
    if failures > 0 {
        bail!("{failures} sequence(s) differ between their own call and the packed call");
    }
    Ok(())
}
