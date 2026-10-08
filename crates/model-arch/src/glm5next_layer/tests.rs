// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Host-only tests of the GLM-5.3 layer wiring, on the skeleton built from the
//! checkpoint config fixture.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use metrale_config::parse_config;

use crate::glm5next_skeleton::{Glm5NextTextSkeleton, Mixer, Mlp, ResidualStep, Site};

const CONFIG: &str =
    include_str!("../../../model-engine/tests/fixtures/glm53-nvfp4-9e0d74e3-config.json");

fn skeleton() -> Glm5NextTextSkeleton {
    Glm5NextTextSkeleton::from_config(&parse_config(CONFIG).expect("real config parses"))
        .expect("skeleton builds")
}

/// 2026-09-25: The skeleton's residual plan for layer 0 is `hc_pre -> norm -> sublayer -> hc_post`
/// at the attention site and again at the FFN site, the order `forward_one` runs.
#[test]
fn the_layer_executes_the_skeletons_residual_plan() {
    let sk = skeleton();
    let l = sk.layers[0];
    assert_eq!(
        sk.residual_plan(&l),
        vec![
            ResidualStep::SaveResidual,
            ResidualStep::HcPre(Site::Attn),
            ResidualStep::Norm("input_layernorm.weight"),
            ResidualStep::Mixer,
            ResidualStep::HcPost(Site::Attn),
            ResidualStep::SaveResidual,
            ResidualStep::HcPre(Site::Ffn),
            ResidualStep::Norm("post_attention_layernorm.weight"),
            ResidualStep::Mlp,
            ResidualStep::HcPost(Site::Ffn),
        ],
        "forward_one runs hc_pre -> norm -> sublayer -> hc_post twice, in this order"
    );
}

/// 2026-09-25: The layer census of the config fixture: 45 text layers, 34 KDA and 11 DSA,
/// 3 dense and 42 routed, with a hyper-connection on every text layer and none on the MTP layer.
#[test]
fn the_stack_the_composite_must_cover() {
    let sk = skeleton();
    assert_eq!(sk.layers.len(), 45);
    assert_eq!(
        sk.layers.iter().filter(|l| l.mixer == Mixer::Kda).count(),
        34
    );
    assert_eq!(
        sk.layers.iter().filter(|l| l.mixer == Mixer::Dsa).count(),
        11
    );
    assert_eq!(sk.layers.iter().filter(|l| l.mlp == Mlp::Dense).count(), 3);
    assert_eq!(
        sk.layers.iter().filter(|l| l.mlp == Mlp::RoutedMoe).count(),
        42
    );
    assert!(sk.layers.iter().all(|l| l.hyper_connection));
    assert!(!sk.mtp.expect("layer 45 exists").hyper_connection);
}

/// 2026-09-25: The fixture has KDA+dense, KDA+routed and DSA+routed layers, and no DSA+dense
/// layer.
#[test]
fn every_dispatch_arm_is_reachable() {
    let sk = skeleton();
    let combos: std::collections::BTreeSet<(bool, bool)> = sk
        .layers
        .iter()
        .map(|l| (l.mixer == Mixer::Kda, l.mlp == Mlp::Dense))
        .collect();
    assert!(combos.contains(&(true, true)), "KDA + dense");
    assert!(combos.contains(&(true, false)), "KDA + routed");
    assert!(combos.contains(&(false, false)), "DSA + routed");
    assert!(!combos.contains(&(false, true)), "no DSA + dense today");
}

/// 2026-10-08: The batched decode's row groups cover every row exactly once, in order, none
/// empty and none wider than the cap; a cap of 0 acts as 1.
#[test]
fn multi_seq_groups_cover_every_row_once_in_order() {
    use super::multi_seq_chunks;
    for rows in 0..=40usize {
        for cap in [1usize, 3, 16] {
            let g = multi_seq_chunks(rows, cap);
            let flat: Vec<usize> = g.iter().flat_map(|&(s, w)| s..s + w).collect();
            assert_eq!(flat, (0..rows).collect::<Vec<_>>(), "rows={rows} cap={cap}");
            assert!(g.iter().all(|&(_, w)| (1..=cap).contains(&w)));
        }
    }
    assert_eq!(multi_seq_chunks(33, 16), vec![(0, 16), (16, 16), (32, 1)]);
    assert_eq!(multi_seq_chunks(2, 0), vec![(0, 1), (1, 1)]);
}

/// 2026-10-08: The group width is the batched GEMV's widest M, the width up to which a row's
/// projections carry the M = 1 bits, and every launcher the group passes through accepts it:
/// the mHC `mix` scratch and the workspaces the loader sizes for `PREFILL_ROWS` rows.
#[test]
fn the_group_width_is_the_bit_identical_gemv_width_and_fits_every_launcher() {
    use super::multi_seq_chunk_rows;
    let w = multi_seq_chunk_rows();
    assert_eq!(
        w,
        metrale_model_layers::layers::ops::DENSE_GEMV_BATCHM_MAX_M as usize
    );
    assert!(w <= crate::glm5next_mhc::mhc_mix_max_tokens());
    let loader = include_str!("../weight_loader/glm5_next_load/loader.rs");
    assert!(
        loader.contains(
            "let verify_k = (metrale_model_layers::layers::ops::DENSE_GEMV_BATCHM_MAX_M as usize)"
        ),
        "the loader's workspace rows must start from the batched GEMV width"
    );
}
