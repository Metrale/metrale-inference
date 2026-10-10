// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for `Glm5NextMlpConfig` derivation and refusals, without a GPU.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use metrale_config::{Glm5NextRouterMode, ModelConfig, TpSlice, parse_config};

use super::*;

/// 2026-09-25: The GLM-5.3-Flash checkpoint config fixture, parsed by the `glm5_next` parser.
/// `model-engine/tests/glm5next_skeleton.rs` reads the same file.
const CONFIG: &str =
    include_str!("../../../model-engine/tests/fixtures/glm53-nvfp4-9e0d74e3-config.json");

fn glm_config() -> ModelConfig {
    parse_config(CONFIG).expect("the real checkpoint config parses")
}

#[test]
fn reads_the_checkpoint_geometry_at_world_one() {
    let c = Glm5NextMlpConfig::from_config(&glm_config()).unwrap();
    assert_eq!(c.hidden, 4096);
    assert_eq!(c.local_dense_intermediate, 12288);
    assert_eq!(c.moe_intermediate, 2048);
    assert_eq!(c.local_shared_intermediate, 2048);
    assert_eq!(c.num_experts, 288);
    assert_eq!(c.local_experts, 288);
    assert_eq!(c.top_k, 8);
    assert_eq!(c.routed_scale, 2.5);
    assert!(c.renormalize);
    assert_eq!(c.swiglu_limit, 10.0);
    assert!(!c.router_bf16_ladder);
    assert!(!c.needs_all_reduce());
}

/// 2026-09-25: TP 2 and EP 2 on the same two ranks: the dense/shared widths halve, the expert
/// set halves, and an expert's width is unchanged.
#[test]
fn tp2_ep2_halves_widths_and_the_expert_set() {
    let mut base = glm_config();
    base.tp_world_size = 2;
    base.ep_world_size = 2;

    for rank in 0..2 {
        let mut m = base.clone();
        m.tp_rank = rank;
        m.ep_rank = rank;
        let c = Glm5NextMlpConfig::from_config(&m).unwrap();
        assert_eq!(c.local_dense_intermediate, 6144);
        assert_eq!(c.local_shared_intermediate, 1024);
        assert_eq!(c.moe_intermediate, 2048, "rank {rank}");
        assert_eq!(c.local_experts, 144);
        assert!(c.needs_all_reduce());
    }
}

/// 2026-10-08: The even split's column offsets at TP 2: rank 1 starts half-way.
#[test]
fn tp2_rank1_starts_half_way() {
    let mut m = glm_config();
    m.tp_world_size = 2;
    m.tp_rank = 1;
    let c = Glm5NextMlpConfig::from_config(&m).unwrap();
    assert_eq!(
        c.dense_slice(),
        TpSlice {
            start: 6144,
            len: 6144
        }
    );
    assert_eq!(
        c.shared_slice(),
        TpSlice {
            start: 1024,
            len: 1024
        }
    );
}

/// 2026-10-08: World = TP = EP = 3: the dense width splits 4096 x 3, the shared expert 688/680/680
/// (8-column units, the BF16 GEMV's `K % 8` rule), and the experts 96 x 3, each id owned once.
#[test]
fn tp3_ep3_splits_widths_in_gemm_units_and_experts_evenly() {
    let mut base = glm_config();
    base.tp_world_size = 3;
    base.ep_world_size = 3;
    let cfgs: Vec<Glm5NextMlpConfig> = (0..3)
        .map(|r| {
            let mut m = base.clone();
            m.tp_rank = r;
            m.ep_rank = r;
            Glm5NextMlpConfig::from_config(&m).unwrap()
        })
        .collect();
    let dense: Vec<_> = cfgs.iter().map(|c| c.dense_slice()).collect();
    let shared: Vec<_> = cfgs.iter().map(|c| c.shared_slice()).collect();
    let s = |start, len| TpSlice { start, len };
    assert_eq!(dense, vec![s(0, 4096), s(4096, 4096), s(8192, 4096)]);
    assert_eq!(shared, vec![s(0, 688), s(688, 680), s(1368, 680)]);
    for c in &cfgs {
        assert_eq!(c.local_shared_intermediate % BF16_GEMM_K_ALIGN, 0);
        assert_eq!(c.moe_intermediate, 2048, "an expert is never TP-split");
        assert_eq!(c.local_experts, 96);
        assert!(c.needs_all_reduce());
    }
    assert_eq!(
        cfgs.iter()
            .map(|c| c.local_expert_range())
            .collect::<Vec<_>>(),
        vec![0..96, 96..192, 192..288]
    );
}

/// 2026-10-08: A width that is not a whole number of 8-column units cannot split over TP.
#[test]
fn a_width_off_the_gemm_unit_is_refused_above_one_rank() {
    let mut c = glm_config();
    c.shared_expert_intermediate_size = 2044;
    assert!(
        Glm5NextMlpConfig::from_config(&c).is_ok(),
        "one rank slices nothing"
    );
    c.tp_world_size = 3;
    let err = format!("{:#}", Glm5NextMlpConfig::from_config(&c).unwrap_err());
    assert!(
        err.contains("shared_expert_intermediate_size 2044"),
        "{err}"
    );
    assert!(err.contains("not a multiple"), "{err}");
}

/// 2026-09-25: At EP 2 the ranks own ids 0..144 and 144..288: every id is owned by exactly one
/// rank, and a remote id has no local slot.
#[test]
fn every_expert_id_is_owned_by_exactly_one_rank() {
    let mut base = glm_config();
    base.tp_world_size = 2;
    base.ep_world_size = 2;

    let cfgs: Vec<Glm5NextMlpConfig> = (0..2)
        .map(|r| {
            let mut m = base.clone();
            m.ep_rank = r;
            Glm5NextMlpConfig::from_config(&m).unwrap()
        })
        .collect();

    assert_eq!(cfgs[0].local_expert_range(), 0..144);
    assert_eq!(cfgs[1].local_expert_range(), 144..288);

    for id in 0..288usize {
        let owners: Vec<usize> = cfgs
            .iter()
            .enumerate()
            .filter(|(_, c)| c.local_slot(id).is_some())
            .map(|(r, _)| r)
            .collect();
        assert_eq!(owners.len(), 1, "expert {id} owned by {owners:?}");
    }
    // 2026-09-25: The local slot is the offset into the rank's range, not the global id.
    assert_eq!(cfgs[1].local_slot(200), Some(56));
    assert_eq!(cfgs[0].local_slot(200), None);
}

/// 2026-09-25: A zero clamp is refused: `min(gate, 0)` would force the gate non-positive.
#[test]
fn a_zero_swiglu_limit_is_refused() {
    let mut c = glm_config();
    c.swiglu_limit = 0.0;
    let err = Glm5NextMlpConfig::from_config(&c).unwrap_err();
    assert!(err.to_string().contains("swiglu_limit"), "{err}");
}

/// 2026-09-25: `glm5next_router_topk` holds its selection in `sel_id[16]` / `sel_w[16]`, so
/// top_k 17 is refused and 16 is accepted.
#[test]
fn a_top_k_past_the_kernels_register_budget_is_refused() {
    let mut c = glm_config();
    c.num_experts_per_tok = KERNEL_MAX_TOP_K + 1;
    let err = Glm5NextMlpConfig::from_config(&c).unwrap_err();
    assert!(err.to_string().contains("16-slot"), "{err}");

    let mut ok = glm_config();
    ok.num_experts_per_tok = KERNEL_MAX_TOP_K;
    assert!(Glm5NextMlpConfig::from_config(&ok).is_ok(), "16 is legal");
}

/// 2026-09-25: An expert count that does not divide over EP is refused, because some ids would
/// have no owner.
#[test]
fn an_expert_count_that_does_not_divide_over_ep_is_refused() {
    let mut c = glm_config();
    c.ep_world_size = 7;
    let err = Glm5NextMlpConfig::from_config(&c).unwrap_err();
    assert!(err.to_string().contains("owned by nobody"), "{err}");
}

/// 2026-09-25: `router_bf16_ladder` follows the config's `glm5next_router_mode`.
#[test]
fn the_router_ladder_comes_from_the_config() {
    let mut c = glm_config();
    c.glm5next_router_mode = Glm5NextRouterMode::VllmBf16;
    assert!(
        Glm5NextMlpConfig::from_config(&c)
            .unwrap()
            .router_bf16_ladder
    );
}

/// 2026-10-08: EP without TP is refused (the replicated dense and shared outputs would be summed
/// once per rank); EP beside TP of the same size is accepted.
#[test]
fn ep_without_tp_is_refused() {
    let mut c = glm_config();
    c.ep_world_size = 3;
    let err = format!("{:#}", Glm5NextMlpConfig::from_config(&c).unwrap_err());
    assert!(err.contains("without tensor parallelism"), "{err}");
    c.tp_world_size = 3;
    assert!(Glm5NextMlpConfig::from_config(&c).is_ok());
}
