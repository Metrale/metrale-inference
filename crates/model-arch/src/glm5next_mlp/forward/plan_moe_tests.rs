// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The CPU equivalence gate of the plan-executed routed MoE site: on the recording
//! device port, the hand-written `forward_moe` under the served recipe's levers and the site run
//! from each rank's served plan put the same operations on the stream (kernel by name, grid,
//! block, shared memory, every argument, every memset), on every MoE layer of every rank at
//! decode n=1 and verify n=2. A one-argument sabotage of the plan turns it red at that launch.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use metrale_circuit::{KernelId, Mode};
use metrale_config::{ActivationQuantization, Nvfp4Act, ProjFamily};
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_gpu_runtime::gpu::recording::{Op, RecordingGpu};

use super::super::Glm5NextMlpWorkspace;
use super::*;
use crate::glm5next_mlp::precision::{GroupPrecision, MlpGroup};
use crate::glm5next_mlp::weights::{
    Glm5NextDenseMlpWeights, Glm5NextExpertPtrTable, Glm5NextExpertWeights, Glm5NextMoePtrTables,
    Nvfp4Proj, ScaleLayout,
};
use crate::glm5next_mlp::{Glm5NextMlpConfig, Glm5NextMlpKernels};
use crate::glm5next_plan::compile_moe;
use crate::glm5next_plan::fixture::{RUNS, mlp_config, moe_layers, plan, ranks, served_levers};

/// 2026-10-23: GB10's SM count (the sweep's persistent grid).
const SMS: u32 = 48;
const STREAM: u64 = 7;
/// 2026-10-23: The checkpoint's routed experts, all local under the served `tp` layout (the
/// site check pins it to `cfg.num_experts`).
const EXPERTS: u64 = 288;
const X: DevicePtr = DevicePtr(0x9000_0000);
const OUT: DevicePtr = DevicePtr(0x9100_0000);
const SCALES: W4a4ActScales = W4a4ActScales {
    gate_up: 0.000642,
    down: 0.000115,
};

struct Site {
    gpu: RecordingGpu,
    k: Glm5NextMlpKernels,
    cfg: Glm5NextMlpConfig,
    ws: Glm5NextMlpWorkspace,
}

fn site(rank: usize) -> Site {
    let gpu = RecordingGpu::new(SMS);
    let k = Glm5NextMlpKernels::resolve(&gpu, served_levers()).unwrap();
    let cfg = mlp_config(rank);
    let ws = Glm5NextMlpWorkspace::new(&gpu, &cfg, 16, &k.levers.mlp).unwrap();
    gpu.take();
    Site { gpu, k, cfg, ws }
}

/// 2026-10-23: Layer `l`'s weights at distinct synthetic addresses, every expert table carrying
/// a base table (compact scales, as the served load packs them) unless `plain`. The per-expert
/// projections are bound (`forward_moe` checks their count) but no W4A4 launch reads them.
fn weights(l: usize, plain: bool) -> Glm5NextMoeWeights {
    let tag = |s: u64| DevicePtr(0x4_0000_0000 + ((l as u64) << 24) + (s << 12));
    let proj = |s: u64| Nvfp4Proj {
        packed: tag(s),
        scale: tag(s + 1),
        scale_2: 0.5,
        input_scale: None,
        scale_layout: ScaleLayout::Plain,
    };
    let experts = (0..EXPERTS)
        .map(|e| Glm5NextExpertWeights {
            gate_proj: proj(64 + 6 * e),
            up_proj: proj(66 + 6 * e),
            down_proj: proj(68 + 6 * e),
        })
        .collect();
    let table = |s: u64| Glm5NextExpertPtrTable {
        packed_ptrs: tag(s),
        scale_ptrs: tag(s + 1),
        scale2_vals: tag(s + 2),
        scale_bases: if plain { DevicePtr::NULL } else { tag(s + 3) },
    };
    Glm5NextMoeWeights {
        router: tag(1),
        router_bias: tag(2),
        shared: Glm5NextDenseMlpWeights {
            gate_proj: tag(3),
            up_proj: tag(4),
            down_proj: tag(5),
        },
        experts,
        ptrs: Glm5NextMoePtrTables {
            gate: table(16),
            up: table(32),
            down: table(48),
        },
        precision: GroupPrecision::resolve(
            MlpGroup::RoutedExperts,
            ActivationQuantization::default()
                .ladder(ProjFamily::Moe)
                .clone(),
            Nvfp4Act::A4,
            16,
            true,
        )
        .unwrap(),
        act_scales: Some(SCALES),
    }
}

fn legacy(s: &Site, w: &Glm5NextMoeWeights, rows: usize) -> Vec<Op> {
    super::super::forward_moe(&s.gpu, &s.k, &s.cfg, w, X, OUT, rows, &s.ws, true, STREAM).unwrap();
    s.gpu.take()
}

fn planned(s: &Site, w: &Glm5NextMoeWeights, steps: MoeSteps<KernelId>) -> Vec<Op> {
    let rows = steps.rows;
    let p = bind(&s.gpu, steps, &s.cfg, w, &s.ws).unwrap();
    run_with_legacy_shared(&p, &s.gpu, &s.k, &s.cfg, w, &s.ws, X, OUT, rows, STREAM).unwrap();
    s.gpu.take()
}

/// 2026-10-23: `None` when the two logs are equal, else the first differing operation of each.
fn first_difference(a: &[Op], b: &[Op]) -> Option<String> {
    if a == b {
        return None;
    }
    let i = a.iter().zip(b).take_while(|(x, y)| x == y).count();
    Some(format!(
        "op {i} of {} / {}:\n  legacy {:?}\n  plan   {:?}",
        a.len(),
        b.len(),
        a.get(i),
        b.get(i)
    ))
}

/// 2026-10-23: Every routed MoE site of every rank, at decode n=1 and verify n=2, launches from
/// its plan exactly what `forward_moe` launches under the served levers. The logs are not
/// trivially equal: each holds the site's whole launch sequence, which the counts pin.
#[test]
fn every_plan_executed_moe_site_launches_what_the_legacy_site_launches() {
    let (c, rules, policy) = ranks();
    let mut sites = 0;
    for rank in 0..3 {
        let s = site(rank);
        for (mode, rows) in RUNS {
            let p = plan(&c, &rules, &policy, rank, mode, rows);
            for l in moe_layers(&c[rank]) {
                let w = weights(l, false);
                let steps = compile_moe(&c[rank], &p, l).unwrap().steps;
                let (a, b) = (legacy(&s, &w, rows as usize), planned(&s, &w, steps));
                if let Some(d) = first_difference(&a, &b) {
                    panic!("rank {rank} {mode:?} n={rows} layer {l}: {d}");
                }
                // 2026-10-23: router (1 per row, or 1), top-k, quant, [union], gate/up sweep,
                // SwiGLU, quant, down sweep, the shared expert's 4 launches, combine.
                let want = if rows == 1 { 12 } else { 13 };
                assert_eq!(a.len(), want, "rank {rank} {mode:?} layer {l}: {a:#?}");
                sites += 1;
            }
        }
    }
    assert_eq!(sites, 3 * 2 * 42);
}

fn one_site(rank: usize, mode: Mode, rows: u64) -> (Site, Glm5NextMoeWeights, MoeSteps<KernelId>) {
    let (c, rules, policy) = ranks();
    let p = plan(&c, &rules, &policy, rank, mode, rows);
    let steps = compile_moe(&c[rank], &p, 4).unwrap().steps;
    (site(rank), weights(4, false), steps)
}

/// 2026-10-23: Changing one field of the plan turns the gate red at the launch that carries it:
/// the combine's grid (blocks per row), the top-k block width, and the down quantizer (the plain
/// entry where the 704-column slice needs the `_k64` twin). The unchanged plan is green on the
/// same site, so each red is the sabotage's.
#[test]
fn a_one_field_sabotage_turns_the_gate_red_at_its_launch() {
    let (s, w, steps) = one_site(0, Mode::Verify, 2);
    let a = legacy(&s, &w, 2);
    assert_eq!(first_difference(&a, &planned(&s, &w, steps.clone())), None);
    let plan_side = |d: &str| d.lines().nth(2).unwrap().to_string();
    let mut blend = steps.clone();
    blend.blend.blocks_per_row = 1;
    let d = first_difference(&a, &planned(&s, &w, blend)).expect("a changed combine grid is red");
    assert!(plan_side(&d).contains("glm5next_moe_combine"), "{d}");
    let mut topk = steps.clone();
    topk.topk.block = 256;
    let d = first_difference(&a, &planned(&s, &w, topk)).expect("a changed top-k block is red");
    assert!(plan_side(&d).contains("glm5next_router_topk_rank"), "{d}");
    let mut down = steps;
    down.down.quant.func = "w4a4_quant_rows_static".into();
    let d = first_difference(&a, &planned(&s, &w, down)).expect("the plain down quantizer is red");
    assert!(
        plan_side(&d).contains("w4a4_gemv_mx_moe::w4a4_quant_rows_static\""),
        "{d}"
    );
}

/// 2026-10-23: `bind` refuses a site the plan was not compiled for: rank 2's plan on rank 0's
/// site (another expert slice), and compact-scale entries over plain tables.
#[test]
fn bind_refuses_another_ranks_site_and_another_scale_form() {
    let (_, w, steps2) = one_site(2, Mode::Decode, 1);
    let s0 = site(0);
    let e = bind(&s0.gpu, steps2, &s0.cfg, &w, &s0.ws).err().unwrap();
    assert!(format!("{e:#}").contains("expert slice"), "{e:#}");
    let (s, _, steps) = one_site(0, Mode::Decode, 1);
    let e = bind(&s.gpu, steps, &s.cfg, &weights(4, true), &s.ws)
        .err()
        .unwrap();
    assert!(format!("{e:#}").contains("Compact"), "{e:#}");
}
