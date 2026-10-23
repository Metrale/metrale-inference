// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The CPU equivalence gate of the plan-executed mHC sites: on the recording device
//! and communicator, every text layer's mixer and FFN site (pre-mix + norm, the all-reduce
//! lowered to the one-shot exchange, the post-mix summing the partials) run from each rank's
//! plan put the same operations on the same streams, with the same collectives in the same
//! order, as the hand-written site under the served levers, at decode n=1 and verify n=2. One
//! changed field of the plan turns it red; a row count without a plan runs the layer's code; a
//! payload the communicator refuses is an error on the plan path, not an all-reduce.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use metrale_circuit::Circuit;
use metrale_gpu_runtime::gpu::recording::{Op, RecordingGpu};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::circuit_exec::family::RankPlans;

use super::*;
use crate::glm5next_layer::{Glm5NextMhc, GlmLevers, LayerIo, MhcSite};
use crate::glm5next_mhc::{Glm5NextMhcKernels, Glm5NextMhcSiteWeights};
use crate::glm5next_mhc_plan::InstalledMhc;
use crate::glm5next_plan::fixture::{
    RUNS, model_config_at, plan, ranks, ranks_at, served_levers, served_world, worlds,
};
use crate::glm5next_plan::fixture_site::{RecordingComm, SMS, STREAM, gate};

const HIDDEN: usize = 4096;
/// 2026-10-23: The site's buffers, distinct synthetic addresses.
const STREAMS: DevicePtr = DevicePtr(0x6000_0000);
const Y: DevicePtr = DevicePtr(0x6100_0000);
const POST: DevicePtr = DevicePtr(0x6200_0000);
const COMB: DevicePtr = DevicePtr(0x6300_0000);
const NORMED: DevicePtr = DevicePtr(0x6400_0000);
const OUT: DevicePtr = DevicePtr(0x6500_0000);
const NORMS: [DevicePtr; 2] = [DevicePtr(0x6600_0000), DevicePtr(0x6700_0000)];

struct Layer {
    gpu: RecordingGpu,
    world: usize,
    levers: GlmLevers,
    mhc: Glm5NextMhc,
    rms_eps: f32,
}

fn site_weights(tag: u64, bf16: bool) -> Glm5NextMhcSiteWeights {
    let p = |s: u64| DevicePtr(0x5_0000_0000 + (tag << 20) + (s << 12));
    Glm5NextMhcSiteWeights {
        hc_fn: p(1),
        hc_fn_bf16: bf16,
        hc_scale: p(2),
        hc_base: p(3),
        mix: p(4),
        done: p(5),
        mix_tokens: 64,
    }
}

fn layer(rank: usize, world: usize, levers: GlmLevers) -> Layer {
    let gpu = RecordingGpu::new(SMS);
    let c = model_config_at(rank, world);
    let mhc = Glm5NextMhc {
        kernels: Glm5NextMhcKernels::resolve(&gpu, levers.mhc).unwrap(),
        attn: site_weights(1, true),
        ffn: site_weights(2, true),
        hc_mult: c.hc_mult,
        sinkhorn_iters: c.hc_sinkhorn_iters,
        hc_eps: c.hc_eps,
        fold_streams: DevicePtr::NULL,
    };
    gpu.take();
    Layer {
        gpu,
        world,
        levers,
        mhc,
        rms_eps: c.rms_norm_eps as f32,
    }
}

/// 2026-10-23: The text layers of `c` (every layer with a mixer block).
fn text_layers(c: &Circuit) -> Vec<usize> {
    let mut v: Vec<usize> = c
        .nodes
        .iter()
        .filter(|n| n.block == "kda" || n.block == "dsa")
        .filter_map(|n| n.layer)
        .collect();
    v.dedup();
    v
}

/// 2026-10-23: Both sites of a layer at `rows` rows, from `plans` (`None`: the layer's own code),
/// over a communicator that takes the one-shot path when `oneshot`.
fn run(
    l: &Layer,
    rank: usize,
    rows: usize,
    plans: Option<&InstalledMhc>,
    oneshot: bool,
) -> anyhow::Result<Vec<Op>> {
    let comm = RecordingComm {
        gpu: &l.gpu,
        rank,
        world: l.world,
        oneshot,
    };
    let io = LayerIo {
        levers: &l.levers,
        rms_norm_k: l.gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
        hidden: HIDDEN,
        rms_eps: l.rms_eps,
        layer_idx: 4,
        comm: Some(&comm),
        graph_capture: false,
        plans,
    };
    let s = MhcSite { io, mhc: &l.mhc };
    for (w, norm) in [(&l.mhc.attn, NORMS[0]), (&l.mhc.ffn, NORMS[1])] {
        s.mhc_pre_norm_one(
            &l.gpu, w, norm, STREAMS, Y, POST, COMB, NORMED, rows, None, STREAM,
        )?;
        let fused = l.levers.ar_fused_post;
        let partials = s.io.reduce_or_exchange(OUT, &[rows], fused, STREAM)?;
        s.hc_post_reduced(
            &l.gpu, OUT, partials, STREAMS, STREAMS, POST, COMB, rows, STREAM,
        )?;
    }
    Ok(l.gpu.take())
}

fn installed(
    l: &Layer,
    c: &Circuit,
    plans: &[metrale_circuit::FusionPlan],
    layer: usize,
    policy: &metrale_circuit::Policy,
) -> InstalledMhc {
    let inst = InstalledMhc::default();
    let rp = RankPlans {
        circuit: c,
        plans,
        policy,
    };
    let sites = install(&l.gpu, layer, Some(rp), &l.levers, &l.mhc, HIDDEN, &inst).unwrap();
    // 2026-10-23: Per gated run: two pre-norms and two post-mixes.
    assert_eq!(sites.len(), 4 * RUNS.len(), "layer {layer}: {sites:#?}");
    inst
}

/// 2026-10-23: Every text layer of every rank, at decode n=1 and verify n=2, runs both mHC
/// sites from its plan with the hand-written site's launches, streams and collectives: per site
/// the c1 pre-norm, the exchange ordered on the compute stream, and the rank-sum post-mix.
#[test]
fn every_plan_executed_mhc_site_launches_what_the_layer_launches() {
    let (mut sites, mut want) = (0, 0);
    for (world, rank) in worlds()
        .into_iter()
        .flat_map(|w| (0..w).map(move |r| (w, r)))
    {
        let (c, rules, policy) = ranks_at(world).unwrap();
        let l = layer(rank, world, served_levers());
        let plans: Vec<_> = RUNS
            .iter()
            .map(|&(m, r)| plan(&c, &rules, &policy, rank, m, r))
            .collect();
        let layers = text_layers(&c[rank]);
        assert_eq!(layers.len(), 45, "rank {rank}");
        for &layer_idx in &layers {
            let inst = installed(&l, &c[rank], &plans, layer_idx, &policy);
            for (_, rows) in RUNS {
                let rows = rows as usize;
                let a = run(&l, rank, rows, None, true).unwrap();
                let b = run(&l, rank, rows, Some(&inst), true).unwrap();
                if let Some(d) = gate(&a, &b) {
                    panic!("rank {rank} layer {layer_idx} n={rows}: {d}");
                }
                assert_eq!(a.len(), 6, "rank {rank} layer {layer_idx}: {a:#?}");
                let kernels: Vec<_> = a.iter().filter_map(Op::kernel).collect();
                let (pre, post) = (
                    "glm5next_mhc_c1::glm5next_hc_pre_norm_c1",
                    "glm5next_hc_post_sum::glm5next_hc_post_rank_sum",
                );
                assert_eq!(
                    kernels,
                    [pre, post, pre, post],
                    "rank {rank} layer {layer_idx}"
                );
                let comms: Vec<_> = a
                    .iter()
                    .filter(|o| {
                        matches!(
                            o,
                            Op::Comm {
                                stream: Some(STREAM),
                                ..
                            }
                        )
                    })
                    .collect();
                assert_eq!(
                    comms.len(),
                    2,
                    "the exchange is ordered on the compute stream"
                );
                sites += 2;
            }
        }
        want += 45 * 2 * 2;
    }
    assert_eq!(sites, want);
    assert!(sites > 3 * 45 * 2 * 2, "more than the served world's sites");
}

/// 2026-10-23: One changed plan field turns the gate red at its launch: the pre-norm grid, the
/// pre-norm entry (the per-row `glm5next_hc_pre_norm`), the post kernel (the plain
/// `glm5next_hc_post`). The unchanged plan is green on the same site.
#[test]
fn a_one_field_sabotage_turns_the_mhc_gate_red() {
    let (c, rules, policy) = ranks();
    let l = layer(0, served_world(), served_levers());
    let plans = [plan(
        &c,
        &rules,
        &policy,
        0,
        metrale_circuit::Mode::Verify,
        2,
    )];
    let inst = installed_one(&l, &c[0], &plans, &policy);
    let a = run(&l, 0, 2, None, true).unwrap();
    assert_eq!(gate(&a, &run(&l, 0, 2, Some(&inst), true).unwrap()), None);
    let good = inst.get(2).unwrap();
    let k = |m: &str, f: &str| l.gpu.kernel(m, f).unwrap();
    let mut grid = good;
    grid.pre.grid_y += 1;
    let mut entry = good;
    entry.pre.kernel = k("glm5next_mhc_fused", "glm5next_hc_pre_norm");
    let mut post = good;
    post.post.kernel = k("glm5next_mhc", "glm5next_hc_post");
    for (bad, what) in [(grid, "grid"), (entry, "pre entry"), (post, "post kernel")] {
        let s = InstalledMhc::default();
        s.set(vec![(bad, Vec::new())]);
        let d = gate(&a, &run(&l, 0, 2, Some(&s), true).unwrap());
        assert!(d.is_some(), "a changed {what} is red");
    }
}

fn installed_one(
    l: &Layer,
    c: &Circuit,
    plans: &[metrale_circuit::FusionPlan],
    policy: &metrale_circuit::Policy,
) -> InstalledMhc {
    let inst = InstalledMhc::default();
    let rp = RankPlans {
        circuit: c,
        plans,
        policy,
    };
    install(&l.gpu, 4, Some(rp), &l.levers, &l.mhc, HIDDEN, &inst).unwrap();
    inst
}

/// 2026-10-23: A row count without a plan runs the layer's own code (here 3 rows with only the
/// verify n=2 plan installed), and the plan path refuses an exchange the communicator declines
/// where the layer falls back to the all-reduce.
#[test]
fn uninstalled_rows_run_the_layer_and_a_declined_exchange_is_an_error() {
    let (c, rules, policy) = ranks();
    let l = layer(0, served_world(), served_levers());
    let plans = [plan(
        &c,
        &rules,
        &policy,
        0,
        metrale_circuit::Mode::Verify,
        2,
    )];
    let inst = installed_one(&l, &c[0], &plans, &policy);
    let a = run(&l, 0, 3, None, true).unwrap();
    assert_eq!(gate(&a, &run(&l, 0, 3, Some(&inst), true).unwrap()), None);
    let legacy = run(&l, 0, 2, None, false).unwrap();
    assert!(
        legacy
            .iter()
            .any(|o| matches!(o, Op::Comm { call, .. } if call.starts_with("all_reduce")))
    );
    let e = run(&l, 0, 2, Some(&inst), false).err().unwrap();
    assert!(format!("{e:#}").contains("no one-shot exchange"), "{e:#}");
}

/// 2026-10-23: Installation refuses a lever the mHC rules do not model and a site whose weights
/// cannot run the fused pre-norm the plan names, installing nothing.
#[test]
fn installation_refuses_unmodelled_levers_and_unfusable_sites() {
    let (c, rules, policy) = ranks();
    let plans = [plan(
        &c,
        &rules,
        &policy,
        0,
        metrale_circuit::Mode::Decode,
        1,
    )];
    let mut levers = served_levers();
    levers.mhc.pre_c1 = false;
    let l = layer(0, served_world(), levers);
    let inst = InstalledMhc::default();
    let rp = RankPlans {
        circuit: &c[0],
        plans: &plans,
        policy: &policy,
    };
    let e = install(&l.gpu, 4, Some(rp), &l.levers, &l.mhc, HIDDEN, &inst)
        .err()
        .unwrap();
    assert!(format!("{e:#}").contains("PRE_C1=0"), "{e:#}");
    let mut l = layer(0, served_world(), served_levers());
    l.mhc.ffn.hc_fn_bf16 = false;
    let e = install(&l.gpu, 4, Some(rp), &l.levers, &l.mhc, HIDDEN, &inst)
        .err()
        .unwrap();
    assert!(format!("{e:#}").contains("FFN site cannot run"), "{e:#}");
    assert!(inst.sites().is_empty());
}
