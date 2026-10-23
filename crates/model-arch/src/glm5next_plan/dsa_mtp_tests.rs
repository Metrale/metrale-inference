// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The CPU equivalence gate of the draft head's plan-executed DSA mixer: on the
//! recording device, the MTP block's DSA layer of every rank, built as the MTP loader builds it
//! (the drafter's capped context, its flat indexer cache with persisted pool keys, its one-layer
//! KV pool, BF16 projections, the recipe's levers and settings), runs a captured draft row from
//! its installed draft plan (`Mode::Draft`, one row) and puts the same operations on the same
//! streams as `decode_k` on the drafter graph's pass (`graph::forward_graphed`: captured, a
//! decode step, the state's one-row metadata). One changed field of the plan turns it red; each
//! site installs only at its own shapes, and the MTP site refuses a plan of another mode.
//!
//! Runs in a child process that publishes the served tiers (`dsa_fixture::in_served_process`),
//! as the text-layer gate does: nothing on either path is mocked.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use std::sync::Arc;

use metrale_circuit::Mode;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::gpu::recording::RecordingGpu;

use crate::glm5next_dsa::fp8_rows::IndexerRowFormat;
use crate::glm5next_dsa::layer::{Glm5NextDsaLayer, Glm5NextDsaWorkspace};
use crate::glm5next_dsa::paged::IndexerCache;
use crate::glm5next_mtp_head::{MtpLevers, drafter_context_rows, drafter_dsa_config, kv_pool};
use crate::glm5next_plan::compile_dsa;
use crate::glm5next_plan::dsa::DsaSite;
use crate::glm5next_plan::dsa_fixture::*;
use crate::glm5next_plan::dsa_sabotage::sabotaged;
use crate::glm5next_plan::fixture::{
    plan, ranks, ranks_at, served_levers, served_settings, served_world, worlds,
};
use crate::glm5next_plan::fixture_site::{SMS, gate};
use metrale_model_layers::circuit_exec::family::PlanGroup;

/// 2026-10-23: This module's path in the test binary (`in_served_process`).
const TESTS: &str = "glm5next_plan::dsa_mtp_tests";

/// 2026-10-23: The draft head's DSA layer of rank `rank` as `load_glm5next_mtp_module` builds
/// it, its stack, and a rig over the drafter's KV pool whose passes run at half its row space.
fn drafter(gpu: &RecordingGpu, rank: usize, world: usize) -> (Stack, Glm5NextDsaLayer, Rig) {
    let levers = served_levers();
    assert!(
        !levers.dense.mtp_drafter_tiers,
        "the served drafter keeps BF16 projections (METRALE_GLM_MTP_DRAFTER_TIERS off)"
    );
    let s = served_settings();
    let mtp = MtpLevers::from_settings(&s).unwrap();
    let mut st = stack(gpu, rank, world, levers);
    st.cfg = drafter_dsa_config(&st.config, mtp.context_cap).unwrap();
    st.format = IndexerRowFormat::drafter(
        s.kernels.glm_dsa_pool_keys,
        st.kernels.rows.pool_keys_ready(),
    )
    .unwrap();
    assert_eq!(st.format, IndexerRowFormat::Bf16Pooled);
    st.cache = IndexerCache::Flat;
    st.ws = Arc::new(Glm5NextDsaWorkspace::new(gpu, &st.cfg, mtp.dsa_rows(), &levers).unwrap());
    let (l, _) = unregistered(gpu, &st, st.config.num_hidden_layers, 0);
    let rows = drafter_context_rows(st.config.serve_max_seq_len, &st.cfg);
    let kv = kv_pool::drafter_kv_config(st.cfg.kv_lora_rank);
    let r = rig_with(gpu, &st, kv, kv_pool::blocks_per_seq(rows), rows / 2 + 1);
    (st, l, r)
}

fn mtp(l: &Glm5NextDsaLayer) -> DsaSite {
    DsaSite::Mtp { layer: l.layer_idx }
}

/// 2026-10-23: On every rank the draft head's DSA mixer launches from its installed draft plan
/// exactly what `decode_k` launches on the drafter graph's pass: the same stream graph and the
/// same operations. The plan's own launches are the served forms in order (the decode row's:
/// latent write, key norm, placement, persisted selection, head-batched attend); the BF16
/// projections are the declared legacy sites, the block's residual steps outside the mixer.
#[test]
fn the_draft_head_dsa_mixer_launches_what_decode_k_launches() {
    in_served_process(
        TESTS,
        "the_draft_head_dsa_mixer_launches_what_decode_k_launches",
        || {
            let gpu = RecordingGpu::new(SMS);
            let mut runs = 0;
            for (world, rank) in worlds()
                .into_iter()
                .flat_map(|w| (0..w).map(move |r| (w, r)))
            {
                let (c, rules, policy) = ranks_at(world).unwrap();
                let (_st, dl, mut r) = drafter(&gpu, rank, world);
                let plans = [plan(&c, &rules, &policy, rank, Mode::Draft, 1)];
                let (inst, sites) = installed_at(&gpu, &c[rank], &plans, &policy, &dl, mtp(&dl))
                    .unwrap_or_else(|e| panic!("rank {rank}: {e:#}"));
                assert_eq!(sites.len(), 5, "rank {rank}: {sites:#?}");
                assert!(
                    sites
                        .iter()
                        .all(|s| s.starts_with(&format!("layer {} n=1: mtp_dsa [", dl.layer_idx))),
                    "{sites:#?}"
                );
                let site = compile_dsa(&c[rank], &plans[0], mtp(&dl), &policy).unwrap();
                assert_eq!(
                    site.legacy,
                    [
                        "mtp_dsa [q_a]",
                        "mtp_dsa [q_a_norm]",
                        "mtp_dsa [q_b]",
                        "mtp_dsa [kv_a]",
                        "mtp_dsa [idx_k]",
                        "mtp_dsa [idx_gate]",
                        "mtp_dsa [idx_w]",
                        "mtp_dsa [idx_q]",
                        "mtp_dsa [o]",
                    ],
                    "rank {rank}"
                );
                let p = inst.get(1).unwrap();
                let legacy = r.run(&gpu, &dl, 1, None).unwrap();
                let planned = r.run(&gpu, &dl, 1, Some(&p)).unwrap();
                if let Some(d) = gate(&legacy, &planned) {
                    panic!("rank {rank} draft n=1: {d}");
                }
                let want = own_launches(1);
                let mine: Vec<String> = kernels(&planned)
                    .into_iter()
                    .filter(|k| want.contains(k))
                    .collect();
                assert_eq!(mine, want, "rank {rank}: the plan's own launches");
                runs += 1;
            }
            println!(
                "{runs} draft-head DSA mixer runs gated, {} sites installed",
                5 * runs
            );
        },
    );
}

/// 2026-10-23: One changed field of the bound draft plan turns the gate red (the fields of
/// `dsa_sabotage::sabotaged`).
#[test]
fn a_changed_draft_plan_field_turns_the_gate_red() {
    in_served_process(
        TESTS,
        "a_changed_draft_plan_field_turns_the_gate_red",
        || {
            let (c, rules, policy) = ranks();
            let gpu = RecordingGpu::new(SMS);
            let (_st, dl, mut r) = drafter(&gpu, 0, served_world());
            let plans = [plan(&c, &rules, &policy, 0, Mode::Draft, 1)];
            let (inst, _) = installed_at(&gpu, &c[0], &plans, &policy, &dl, mtp(&dl)).unwrap();
            let good = inst.get(1).unwrap();
            let legacy = r.run(&gpu, &dl, 1, None).unwrap();
            assert_eq!(
                gate(&legacy, &r.run(&gpu, &dl, 1, Some(&good)).unwrap()),
                None
            );
            let other = gpu.kernel("dsa_indexer", "dsa_kpool_compress").unwrap();
            let bad = sabotaged(&good, other);
            assert_eq!(bad.len(), 15);
            for (what, p) in bad {
                let d = match r.run(&gpu, &dl, 1, Some(&p)) {
                    Ok(ops) => gate(&legacy, &ops),
                    Err(e) => Some(format!("refused: {e:#}")),
                };
                assert!(d.is_some(), "draft n=1: a changed {what} stayed green");
            }
        },
    );
}

/// 2026-10-23: Each site installs only at its own shapes: the MTP site nothing from the text
/// layers' decode and verify plans, a text site nothing from the draft plan.
#[test]
fn each_dsa_site_installs_only_at_its_own_shapes() {
    in_served_process(
        TESTS,
        "each_dsa_site_installs_only_at_its_own_shapes",
        || {
            let (c, rules, policy) = ranks();
            let gpu = RecordingGpu::new(SMS);
            let (_st, dl, _r) = drafter(&gpu, 0, served_world());
            let text_plans = [
                plan(&c, &rules, &policy, 0, Mode::Decode, 1),
                plan(&c, &rules, &policy, 0, Mode::Verify, 2),
            ];
            let (inst, sites) =
                installed_at(&gpu, &c[0], &text_plans, &policy, &dl, mtp(&dl)).unwrap();
            assert!(sites.is_empty() && inst.get(1).is_none(), "{sites:#?}");
            let draft = [plan(&c, &rules, &policy, 0, Mode::Draft, 1)];
            let st = stack(&gpu, 0, served_world(), served_levers());
            let l = dsa_layers(&c[0])[0];
            let text_layer = layer(&gpu, &st, l, 0);
            let (inst, sites) = installed(&gpu, &c[0], &draft, &policy, &text_layer).unwrap();
            assert!(sites.is_empty() && inst.get(1).is_none(), "{sites:#?}");
        },
    );
}

/// 2026-10-23: The MTP site's groups compile only in a draft: the draft plan relabelled a decode
/// is refused naming the mode, and a draft plan has no text-layer DSA mixer.
#[test]
fn the_mtp_site_compiles_only_a_draft() {
    let (c, rules, policy) = ranks();
    let mut p = plan(&c, &rules, &policy, 0, Mode::Draft, 1);
    let site = DsaSite::Mtp { layer: 0 };
    let g = PlanGroup::all(&c[0], &p)
        .find(|g| g.block() == "mtp_dsa")
        .unwrap();
    assert_eq!(g.layer(), None, "the draft blocks carry no layer");
    assert!(compile_dsa(&c[0], &p, site, &policy).is_ok());
    let l = dsa_layers(&c[0])[0];
    let e = compile_dsa(&c[0], &p, text(l), &policy).err().unwrap();
    assert!(format!("{e:#}").contains("no DSA mixer"), "{e:#}");
    p.mode = Mode::Decode;
    let e = compile_dsa(&c[0], &p, site, &policy).err().unwrap();
    assert!(format!("{e:#}").contains("[Draft]"), "{e:#}");
}
