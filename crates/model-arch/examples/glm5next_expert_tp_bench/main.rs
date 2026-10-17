// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: GLM-5.3's routed experts under the two expert layouts of a TP=3 serve, one rank
//! at a time on one GPU, at the declared W4A4 tier (hidden 4096, 288 experts, top 8):
//!
//! 1. Byte gate of the `_k64` twins (`gate.rs`).
//! 2. `forward_moe` at a 704-wide slice equals `forward_moe` at 768 over the same experts
//!    zero-padded to 768 (bytes of the layer output, 1, 4, 8 and 16 rows): the 64-unit split
//!    computes what a padded 128-unit kernel would.
//! 3. The per-rank MoE time (`forward_moe`, CUDA-graph replay) on the same router, rows and
//!    weights: `ep` ranks 0, 1 and 2 (96 whole 2048-wide experts each; a step waits for the
//!    slowest, so the critical path is their maximum) against `tp` slices of every expert, 768
//!    (rank 0 of the 128-unit split), 704 (ranks 0 and 1 of the 64-unit split) and 640. Each
//!    draw of `x` is one routing; the candidates are interleaved per replay round so a load
//!    another process puts on the GPU falls on all alike.
//!
//! The weights are views into one pool of random NVFP4 bytes (timing does not depend on their
//! values), about 1.6 GB of device memory in all.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: exits with an error on the first byte mismatch.
//!
//! ```text
//! METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//! GLM_BENCH_GPU_ORDINAL=0 cargo run -p metrale-model-arch --release \
//!     --example glm5next_expert_tp_bench --features cuda,gpu-examples
//! ```

use anyhow::{Context, Result, ensure};
use metrale_config::{ActivationQuantization, Nvfp4Act, ProjFamily};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_mlp::build::build_moe;
use metrale_model_arch::glm5next_mlp::expert_tp::expert_slice;
use metrale_model_arch::glm5next_mlp::forward::{Glm5NextMlpWorkspace, forward_moe};
use metrale_model_arch::glm5next_mlp::precision::{GroupPrecision, MlpGroup};
use metrale_model_arch::glm5next_mlp::weights::{
    Glm5NextExpertWeights, Glm5NextMoeWeights, Nvfp4Proj,
};
use metrale_model_arch::glm5next_mlp::{ExpertShard, Glm5NextMlpConfig, Glm5NextMlpKernels};

#[allow(dead_code)]
#[path = "../glm5next_moe_wide_bench/device.rs"]
mod device;
mod gate;
use device::*;

const SHARED: usize = 688;
/// 2026-10-10: Routings (draws of `x`) per row count; `GLM_BENCH_DRAWS` overrides it.
const DRAWS: usize = 6;
/// 2026-10-10: The widest slice; every pool slot holds one expert projection this wide.
const POOL_W: usize = 768;

/// 2026-10-10: One projection of every expert slot, packed and scale bytes back to back.
struct Pool {
    packed: DevicePtr,
    scale: DevicePtr,
}

impl Pool {
    fn new(gpu: &dyn GpuBackend, rng: &mut Rng) -> Result<Self> {
        let (pb, sb) = (
            EXPERTS * POOL_W * HIDDEN / 2,
            EXPERTS * POOL_W * HIDDEN / 16,
        );
        let p: Vec<u8> = (0..pb).map(|_| rng.next() as u8).collect();
        let s: Vec<u8> = (0..sb).map(|_| 0x30 + (rng.next() % 9) as u8).collect();
        Ok(Self {
            packed: up(gpu, &p)?,
            scale: up(gpu, &s)?,
        })
    }
    /// 2026-10-10: An `[n, k]` view at byte slot `i` of `n * k` elements' worth.
    fn view(&self, i: usize, n: usize, k: usize, gs: f32) -> Nvfp4Proj {
        Nvfp4Proj {
            packed: self.packed.offset(i * n * k / 2),
            scale: self.scale.offset(i * n * k / 16),
            scale_2: 0.01,
            input_scale: Some(gs),
        }
    }
}

fn cfg(layout: &str, r: usize) -> Result<Glm5NextMlpConfig> {
    let base = Glm5NextMlpConfig {
        hidden: HIDDEN,
        local_dense_intermediate: 4096,
        dense_start: 0,
        moe_intermediate: 2048,
        local_shared_intermediate: SHARED,
        shared_start: 0,
        num_experts: EXPERTS,
        local_experts: LOCAL,
        ep_rank: r,
        top_k: TOP_K,
        routed_scale: 2.5,
        renormalize: true,
        swiglu_limit: 10.0,
        router_bf16_ladder: false,
        tp_world_size: 3,
        ep_world_size: 3,
        expert_shard: ExpertShard::Whole,
    };
    Ok(match layout {
        "ep" => base,
        _ => {
            let w: usize = layout.parse()?;
            // 2026-10-10: The slice geometry of a rank whose width is `w` (768 is not a 64-unit
            // slice of 2048, so it is stated directly).
            let s = if w == 768 {
                metrale_model_arch::glm5next_mlp::ExpertSlice {
                    start: 0,
                    real: 768,
                    len: 768,
                    full: 2048,
                }
            } else {
                expert_slice(2048, 3, if w == 704 { 0 } else { 2 })?
            };
            ensure!(s.len == w, "slice {s:?} is not {w} wide");
            Glm5NextMlpConfig {
                moe_intermediate: w,
                local_experts: EXPERTS,
                ep_rank: 0,
                ep_world_size: 1,
                expert_shard: ExpertShard::Sliced(s),
                ..base
            }
        }
    })
}

struct Site {
    name: String,
    c: Glm5NextMlpConfig,
    w: Glm5NextMoeWeights,
    ws: Glm5NextMlpWorkspace,
}

fn site(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    c: Glm5NextMlpConfig,
    name: &str,
    expert: &dyn Fn(usize) -> Result<Glm5NextExpertWeights>,
    host: &(Vec<f32>, Vec<f32>),
) -> Result<Site> {
    let load = |n: &str| -> Result<Vec<f32>> {
        Ok(match n {
            "mlp.gate.weight" => host.0.clone(),
            "mlp.gate.e_score_correction_bias" => vec![0.0; EXPERTS],
            _ => host.1.clone(),
        })
    };
    let prec = |has| {
        GroupPrecision::resolve(
            MlpGroup::RoutedExperts,
            ActivationQuantization::default()
                .ladder(ProjFamily::Moe)
                .clone(),
            Nvfp4Act::A4,
            k.w4a4_expert_rows(),
            has,
        )
    };
    Ok(Site {
        name: name.to_string(),
        w: build_moe(gpu, &c, SHARED * 3, &load, expert, &prec, MAX_ROWS)?,
        ws: Glm5NextMlpWorkspace::new(gpu, &c, MAX_ROWS)?,
        c,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_site(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    s: &Site,
    x: DevicePtr,
    out: DevicePtr,
    rows: usize,
    capturing: bool,
    stream: u64,
) -> Result<()> {
    forward_moe(gpu, k, &s.c, &s.w, x, out, rows, &s.ws, capturing, stream)
}

fn main() -> Result<()> {
    let ordinal = std::env::var("GLM_BENCH_GPU_ORDINAL")
        .context("GLM_BENCH_GPU_ORDINAL names the GPU to run on")?
        .parse()?;
    let target = metrale_kernels::ptx_for_exact_target("glm-5.3-flash", "nvfp4")
        .context("glm-5.3-flash nvfp4 target")?;
    let gpu = MetraleCudaBackend::new(ordinal, &target.modules)?;
    let g: &dyn GpuBackend = &gpu;
    let stream = g.create_stream()?;
    let k = Glm5NextMlpKernels::resolve(g)?;
    let kern = |quant, slots, sweep| Kern {
        quant,
        row_union: k.moe_row_union,
        slots,
        union: k.w4a4_moe_sweep,
        sweep,
        ctas: k.w4a4_sweep_ctas,
    };
    let t = &k.w4a4_moe_k64;
    let twins = gate::Twins {
        plain: kern(k.w4a4_quant_static, k.w4a4_moe_slots, k.w4a4_moe_sweep),
        k64: kern(t.quant, t.slots, t.sweep),
    };
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    gate::run(g, &twins, &mut rng)?;

    let pool = [
        Pool::new(g, &mut rng)?,
        Pool::new(g, &mut rng)?,
        Pool::new(g, &mut rng)?,
    ];
    let host = (
        (0..EXPERTS * HIDDEN)
            .map(|_| rng.unit() * 0.05)
            .collect::<Vec<f32>>(),
        (0..HIDDEN * SHARED * 3)
            .map(|_| rng.unit() * 0.02)
            .collect::<Vec<f32>>(),
    );
    let out = g.alloc(MAX_ROWS * HIDDEN * 2)?;
    let xs: Vec<DevicePtr> = (0..device::env_or("GLM_BENCH_DRAWS", DRAWS))
        .map(|_| {
            up_bf16(
                g,
                &(0..MAX_ROWS * HIDDEN)
                    .map(|_| rng.unit() * 3.0)
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Result<_>>()?;

    identity(g, &k, &mut rng, &host, &xs[0], out, stream)?;

    // 2026-10-10: ep: rank r's 96 experts (ids 96r..96r+96) on the same 96 pool slots; tp: every
    // expert on its own slot. Gate and up are `[w, 4096]`, down `[4096, w]`.
    let proj = |w: usize, slot: usize| Glm5NextExpertWeights {
        gate_proj: pool[0].view(slot, w, HIDDEN, GS_GATE_UP),
        up_proj: pool[1].view(slot, w, HIDDEN, GS_GATE_UP),
        down_proj: pool[2].view(slot, HIDDEN, w, GS_DOWN),
    };
    let mut sites = Vec::new();
    for r in 0..3 {
        let f = |id: usize| Ok(proj(2048, id % LOCAL));
        sites.push(site(
            g,
            &k,
            cfg("ep", r)?,
            &format!("ep rank {r}"),
            &f,
            &host,
        )?);
    }
    for w in [768usize, 704, 640] {
        let f = |id: usize| Ok(proj(w, id));
        sites.push(site(
            g,
            &k,
            cfg(&w.to_string(), 0)?,
            &format!("tp {w}"),
            &f,
            &host,
        )?);
    }
    for rows in [1usize, 4, 16] {
        timing(g, &k, &sites, &xs, out, rows, stream)?;
    }
    Ok(())
}

/// 2026-10-10: Part 2: 704 against zero-padded 768, output bytes, at W4A4.
fn identity(
    g: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    rng: &mut Rng,
    host: &(Vec<f32>, Vec<f32>),
    x: &DevicePtr,
    out: DevicePtr,
    stream: u64,
) -> Result<()> {
    const E: usize = 48;
    let mut nat = Vec::with_capacity(E);
    for _ in 0..E {
        let mut one = |n: usize, kk: usize| {
            let p: Vec<u8> = (0..n * kk / 2).map(|_| rng.next() as u8).collect();
            let s: Vec<u8> = (0..n * kk / 16)
                .map(|_| 0x30 + (rng.next() % 9) as u8)
                .collect();
            (p, s)
        };
        nat.push([one(704, HIDDEN), one(704, HIDDEN), one(HIDDEN, 704)]);
    }
    let upw = |(p, s): &(Vec<u8>, Vec<u8>), gs: f32| -> Result<Nvfp4Proj> {
        Ok(Nvfp4Proj {
            packed: up(g, p)?,
            scale: up(g, s)?,
            scale_2: 0.01,
            input_scale: Some(gs),
        })
    };
    let rows_pad = |(p, s): &(Vec<u8>, Vec<u8>)| {
        let (mut p, mut s) = (p.clone(), s.clone());
        p.resize(768 * HIDDEN / 2, 0);
        s.resize(768 * HIDDEN / 16, 0);
        (p, s)
    };
    let mut a = Vec::with_capacity(E);
    let mut b = Vec::with_capacity(E);
    for [gt, ut, dt] in &nat {
        a.push(Glm5NextExpertWeights {
            gate_proj: upw(gt, GS_GATE_UP)?,
            up_proj: upw(ut, GS_GATE_UP)?,
            down_proj: upw(dt, GS_DOWN)?,
        });
        let (dp, ds) = gate::pad_cols(&dt.0, &dt.1, HIDDEN, 704, 768);
        b.push(Glm5NextExpertWeights {
            gate_proj: upw(&rows_pad(gt), GS_GATE_UP)?,
            up_proj: upw(&rows_pad(ut), GS_GATE_UP)?,
            down_proj: upw(&(dp, ds), GS_DOWN)?,
        });
    }
    let ca = cfg("704", 0)?;
    let cb = Glm5NextMlpConfig {
        moe_intermediate: 768,
        expert_shard: ExpertShard::Sliced(metrale_model_arch::glm5next_mlp::ExpertSlice {
            start: 0,
            real: 768,
            len: 768,
            full: 2048,
        }),
        ..ca
    };
    let sa = site(g, k, ca, "704", &|id| Ok(a[id % E]), host)?;
    let sb = site(g, k, cb, "768 padded", &|id| Ok(b[id % E]), host)?;
    for rows in [1usize, 4, 8, 16] {
        let n = rows * HIDDEN * 2;
        run_site(g, k, &sa, *x, out, rows, false, stream)?;
        g.synchronize(stream)?;
        let got = read(g, out, n)?;
        run_site(g, k, &sb, *x, out, rows, false, stream)?;
        g.synchronize(stream)?;
        let want = read(g, out, n)?;
        ensure!(
            got == want,
            "rows={rows}: forward_moe at 704 differs from padded 768"
        );
        println!(
            "forward_moe W4A4 rows={rows:2}: 704 == zero-padded 768 (fnv1a {:016x})",
            fnv1a(&got)
        );
    }
    Ok(())
}

/// 2026-10-10: Part 3 at `rows` rows: per draw, every site's median and 10th percentile; the ep
/// critical path is the maximum over its three ranks of each. Means over the draws, with the
/// draws' range of the median.
fn timing(
    g: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    sites: &[Site],
    xs: &[DevicePtr],
    out: DevicePtr,
    rows: usize,
    stream: u64,
) -> Result<()> {
    let mut acc = vec![Vec::new(); sites.len() + 1];
    for &x in xs {
        let fs: Vec<Box<dyn Fn() -> Result<()> + '_>> = sites
            .iter()
            .map(|s| {
                Box::new(move || run_site(g, k, s, x, out, rows, true, stream))
                    as Box<dyn Fn() -> Result<()>>
            })
            .collect();
        let refs: Vec<&dyn Fn() -> Result<()>> = fs.iter().map(|f| f.as_ref()).collect();
        let t = time_set(g, stream, &refs)?;
        for (i, ti) in t.iter().enumerate() {
            acc[i].push([ti[0], ti[1]]);
        }
        let crit = |j: usize| t[0][j].max(t[1][j]).max(t[2][j]);
        acc[sites.len()].push([crit(0), crit(1)]);
    }
    println!(
        "rows={rows:2}: forward_moe over {} routings (us): mean median, mean p10, median range",
        xs.len()
    );
    let names: Vec<&str> = sites
        .iter()
        .map(|s| s.name.as_str())
        .chain(["ep critical path (max over ranks)"])
        .collect();
    for (name, v) in names.iter().zip(&acc) {
        let mean = |j: usize| v.iter().map(|x| x[j]).sum::<f64>() / v.len() as f64;
        let (lo, hi) = v
            .iter()
            .fold((f64::MAX, 0f64), |(a, b), x| (a.min(x[0]), b.max(x[0])));
        println!(
            "  {name:34} {:8.1} {:8.1}  [{lo:.1}, {hi:.1}]",
            mean(0),
            mean(1)
        );
    }
    Ok(())
}
