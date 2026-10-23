// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The CPU equivalence gate of the plan-executed DSA mixers: on the recording device,
//! every DSA layer of every rank, built at the served shape (the recipe's levers, settings and
//! context, a paged KV cache whose V blocks keep the persisted BF16 pool keys, row metadata),
//! runs a captured decode n=1 and a captured verify n=2 from its installed plan and puts the same
//! operations on the same streams as `decode_k`. One changed field of the plan turns it red; a
//! plan runs only a captured pass with row metadata; installation refuses a lever the rules do
//! not model and an attend whose split count is not the layer's.
//!
//! The DSA projections choose their launches from the process-wide tiers (`--dense-quantization
//! w4a16`, `--dsa-quantization w8a8`, published once per process), so the gates run in a child
//! process that publishes the served tiers first (`fixture::in_child_process`); nothing on either
//! path is mocked. The served-shape mixers are `dsa_fixture.rs`.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use metrale_circuit::Mode;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::gpu::recording::RecordingGpu;

use super::*;
use crate::glm5next_dsa::attend::Glm5NextDsaDecodeKernel;
use crate::glm5next_dsa::plan::{AttendStep, plan_rows};
use crate::glm5next_plan::dsa_fixture::*;
use crate::glm5next_plan::dsa_sabotage::sabotaged;
use crate::glm5next_plan::fixture::{RUNS, plan, ranks, served_levers};
use crate::glm5next_plan::fixture_site::{SMS, gate};
use crate::glm5next_plan::{compile_dsa, dsa};

/// 2026-10-23: This module's path in the test binary (`in_served_process`).
const TESTS: &str = "glm5next_plan::dsa::tests";

/// 2026-10-23: Every DSA mixer of every rank, at a captured decode n=1 and a captured verify n=2,
/// launches from its installed plan exactly what `decode_k` launches under the served levers:
/// the same stream graph and the same operations. The plan's own launches are the served forms,
/// in order; the projections are the declared legacy sites.
#[test]
fn every_dsa_mixer_launches_what_decode_k_launches() {
    in_served_process(
        TESTS,
        "every_dsa_mixer_launches_what_decode_k_launches",
        || {
            let (c, rules, policy) = ranks();
            let gpu = RecordingGpu::new(SMS);
            let (mut runs, mut sites_total) = (0, 0);
            for rank in 0..3 {
                let layers = dsa_layers(&c[rank]);
                assert!(
                    layers.len() >= 10,
                    "rank {rank}: {} DSA layers",
                    layers.len()
                );
                let st = stack(&gpu, rank, served_levers());
                let mut r = rig(&gpu, &st);
                let plans: Vec<_> = RUNS
                    .iter()
                    .map(|&(m, n)| plan(&c, &rules, &policy, rank, m, n))
                    .collect();
                for (attn, &l) in layers.iter().enumerate() {
                    let dl = layer(&gpu, &st, l, attn);
                    let (inst, sites) = installed(&gpu, &c[rank], &plans, &policy, &dl)
                        .unwrap_or_else(|e| panic!("rank {rank} layer {l}: {e:#}"));
                    assert_eq!(
                        sites.len(),
                        5 * RUNS.len(),
                        "rank {rank} layer {l}: {sites:#?}"
                    );
                    sites_total += sites.len();
                    for (mode, rows) in RUNS {
                        let rows = rows as usize;
                        let site =
                            compile_dsa(&c[rank], &plans[rows - 1], text(l), &policy).unwrap();
                        let want_legacy: &[&str] = if rows == 1 {
                            &[
                                "dsa [xn_quant]",
                                "dsa [q_a,kv_a]",
                                "dsa [q_a_norm]",
                                "dsa [qan_quant]",
                                "dsa [q_b]",
                                "dsa [idx_k]",
                                "dsa [idx_gate]",
                                "dsa [idx_w]",
                                "dsa [idx_q]",
                                "dsa [a_quant]",
                                "dsa [o]",
                            ]
                        } else {
                            &[
                                "dsa [xn_quant]",
                                "dsa [q_a,kv_a,idx_k,idx_gate]",
                                "dsa [q_a_norm]",
                                "dsa [qan_quant]",
                                "dsa [q_b]",
                                "dsa [idx_w,idx_q]",
                                "dsa [a_quant]",
                                "dsa [o]",
                            ]
                        };
                        assert_eq!(site.legacy, want_legacy, "rank {rank} {mode:?} layer {l}");
                        let p = inst.get(rows).unwrap();
                        let legacy = r.run(&gpu, &dl, rows, None).unwrap();
                        let planned = r.run(&gpu, &dl, rows, Some(&p)).unwrap();
                        if let Some(d) = gate(&legacy, &planned) {
                            panic!("rank {rank} n={rows} layer {l}: {d}");
                        }
                        let want = own_launches(rows);
                        let mine: Vec<String> = kernels(&planned)
                            .into_iter()
                            .filter(|k| want.contains(k))
                            .collect();
                        assert_eq!(mine, want, "rank {rank} n={rows}: the plan's own launches");
                        runs += 1;
                    }
                }
            }
            println!("{runs} DSA mixer runs gated, {sites_total} DSA sites installed");
        },
    );
}

/// 2026-10-23: One changed field of a bound plan turns the gate red: the latent write, the key
/// norm, the store, the pool key, the geometry, the scores, a top-k pass, the collect, the
/// expansion, the attend entry and the attend's split count; and the ceiling (pools, CTAs, SMs).
#[test]
fn a_changed_plan_field_turns_the_gate_red() {
    in_served_process(TESTS, "a_changed_plan_field_turns_the_gate_red", || {
        let (c, rules, policy) = ranks();
        let gpu = RecordingGpu::new(SMS);
        let st = stack(&gpu, 0, served_levers());
        let mut r = rig(&gpu, &st);
        let l = dsa_layers(&c[0])[0];
        let dl = layer(&gpu, &st, l, 0);
        let plans: Vec<_> = RUNS
            .iter()
            .map(|&(m, n)| plan(&c, &rules, &policy, 0, m, n))
            .collect();
        let (inst, _) = installed(&gpu, &c[0], &plans, &policy, &dl).unwrap();
        let other = gpu.kernel("dsa_indexer", "dsa_kpool_compress").unwrap();
        for (_, rows) in RUNS {
            let rows = rows as usize;
            let good = inst.get(rows).unwrap();
            let legacy = r.run(&gpu, &dl, rows, None).unwrap();
            assert_eq!(
                gate(&legacy, &r.run(&gpu, &dl, rows, Some(&good)).unwrap()),
                None
            );
            let bad = sabotaged(&good, other);
            for (what, p) in bad {
                let d = match r.run(&gpu, &dl, rows, Some(&p)) {
                    Ok(ops) => gate(&legacy, &ops),
                    Err(e) => Some(format!("refused: {e:#}")),
                };
                assert!(d.is_some(), "n={rows}: a changed {what} stayed green");
            }
        }
    });
}

/// 2026-10-23: A plan runs only a captured pass that is not a prefill piece, with metadata for
/// its rows; every other pass is the layer's own code. The plan path refuses an eager pass.
#[test]
fn a_plan_runs_only_a_captured_pass_with_row_metadata() {
    in_served_process(
        TESTS,
        "a_plan_runs_only_a_captured_pass_with_row_metadata",
        || {
            let gpu = RecordingGpu::new(SMS);
            let st = stack(&gpu, 0, served_levers());
            let r = rig(&gpu, &st);
            assert!(plan_rows(&r.ctx(&gpu, 1, true), 1, false).is_some());
            assert!(plan_rows(&r.ctx(&gpu, 2, true), 2, false).is_some());
            assert!(
                plan_rows(&r.ctx(&gpu, 1, false), 1, false).is_none(),
                "eager"
            );
            assert!(
                plan_rows(&r.ctx(&gpu, 1, true), 1, true).is_none(),
                "a prefill piece"
            );
            assert!(
                plan_rows(&r.ctx(&gpu, 1, true), 2, false).is_none(),
                "1-row metadata, 2 rows"
            );
            let mut c = r.ctx(&gpu, 1, true);
            c.decode_step = false;
            assert!(
                plan_rows(&c, 1, false).is_none(),
                "one row off a decode step"
            );
            c.attn_metadata = None;
            assert!(plan_rows(&c, 2, false).is_none(), "no metadata");
        },
    );
}

/// 2026-10-23: Installation refuses a DSA lever the rules do not model and the head-batched
/// diagnostics, naming each, and an attend whose split count is not the layer's decode kernel's.
#[test]
fn installation_refuses_unmodelled_levers_and_another_split_count() {
    in_served_process(
        TESTS,
        "installation_refuses_unmodelled_levers_and_another_split_count",
        || {
            let (c, rules, policy) = ranks();
            let gpu = RecordingGpu::new(SMS);
            let st = stack(&gpu, 0, served_levers());
            let l = dsa_layers(&c[0])[0];
            let plans = [plan(&c, &rules, &policy, 0, Mode::Decode, 1)];
            let mut dl = layer(&gpu, &st, l, 0);
            assert!(
                dsa::unmodelled(&dl).is_empty(),
                "the served levers are modelled"
            );
            dl.levers.dsa_verify_select_once = false;
            dl.levers.dsa.serve_ref = true;
            dl.levers.dsa.select.topk_mc = false;
            let e = installed(&gpu, &c[0], &plans, &policy, &dl).err().unwrap();
            let e = format!("{e:#}");
            for lever in ["SELECT_ONCE=0", "SERVE_REF", "TOPK_MC=0"] {
                assert!(e.contains(lever), "{lever}: {e}");
            }
            let mut dl = layer(&gpu, &st, l, 0);
            dl.decode_kernel = Glm5NextDsaDecodeKernel::resolve(&gpu, Some(4)).unwrap();
            let e = installed(&gpu, &c[0], &plans, &policy, &dl).err().unwrap();
            assert!(format!("{e:#}").contains("Some(8)"), "{e:#}");
        },
    );
}

/// 2026-10-23: The attend's split count is the policy's `glm_dsa_decode_hb`; a policy with the
/// head-batched attend off fuses the per-head kernel, and a plan naming the merge under it is
/// refused at compile.
#[test]
fn the_attend_splits_come_from_the_policy() {
    let (c, rules, policy) = ranks();
    let l = dsa_layers(&c[0])[0];
    let p = plan(&c, &rules, &policy, 0, Mode::Decode, 1);
    let s = compile_dsa(&c[0], &p, text(l), &policy).unwrap().steps;
    assert!(matches!(
        s.attend,
        AttendStep::HeadBatched { splits: 8, .. }
    ));
    let mut off = policy.clone();
    off.settings.insert(dsa::HB_SETTING.into(), "off".into());
    let e = compile_dsa(&c[0], &p, text(l), &off).err().unwrap();
    assert!(format!("{e:#}").contains("head-batched merge"), "{e:#}");
    let per_head = plan(&c, &rules, &off, 0, Mode::Decode, 1);
    let s = compile_dsa(&c[0], &per_head, text(l), &off).unwrap().steps;
    assert!(matches!(s.attend, AttendStep::PerHead(_)), "{:?}", s.attend);
    let mut bad = policy.clone();
    bad.settings.insert(dsa::HB_SETTING.into(), "eight".into());
    assert!(dsa::hb_splits(&bad).is_err());
}

/// 2026-10-23: A verify whose selection the plan runs per row (another rule, `glm_dsa_verify_rows`
/// off) is refused at compile rather than run as one pass.
#[test]
fn a_per_row_verify_selection_is_refused() {
    let (c, rules, policy) = ranks();
    let l = dsa_layers(&c[0])[0];
    let mut p = plan(&c, &rules, &policy, 0, Mode::Verify, 2);
    let i = PlanGroup::all(&c[0], &p)
        .position(|g| g.layer() == Some(l) && g.locals() == ["select"])
        .unwrap();
    p.groups[i].repeat = metrale_circuit::rules::Repeat::PerRow;
    let e = compile_dsa(&c[0], &p, text(l), &policy).err().unwrap();
    assert!(format!("{e:#}").contains("in one pass"), "{e:#}");
}
