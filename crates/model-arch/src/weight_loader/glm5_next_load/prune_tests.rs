// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Which store tensors `Glm5NextWeightLoader::prune_after_load` frees.
//!
//! Owner: model-arch weight loader (GLM-5.3).
//! Invariants: none beyond the types.

use super::is_reuploaded;

#[test]
fn prunes_only_the_reuploaded_layer_tensors() {
    let n = 45;
    // 2026-09-25: Text-stack tensors the binders re-uploaded: freed.
    assert!(is_reuploaded(
        "model.language_model.layers.0.self_attn.q_proj.weight",
        n
    ));
    assert!(is_reuploaded(
        "model.language_model.layers.44.mlp.gate.weight",
        n
    ));
    assert!(is_reuploaded(
        "model.language_model.layers.3.hc_attn_fn.weight",
        n
    ));
    // 2026-09-25: Routed experts: kept, since a U8 expert is bound zero-copy.
    assert!(!is_reuploaded(
        "model.language_model.layers.7.mlp.experts.12.down_proj.weight",
        n
    ));
    // 2026-09-25: The MTP block `layers.{n}`: kept for
    // `glm5_next_mtp::load_glm5next_mtp_module`.
    assert!(!is_reuploaded(
        "model.language_model.layers.45.self_attn.q_proj.weight",
        n
    ));
    // 2026-09-25: Not a layer tensor.
    assert!(!is_reuploaded(
        "model.language_model.embed_tokens.weight",
        n
    ));
    assert!(!is_reuploaded("lm_head.weight", n));
    // 2026-09-25: A non-numeric layer index never matches.
    assert!(!is_reuploaded("model.language_model.layers.x.foo", n));
}

/// 2026-10-17: The MTP layer's E4M3 pairs are freed, weight and scale; its BF16 tensors (the
/// zero-copy `eh_proj` among them), its routed experts, an unpaired E4M3 weight and a text
/// layer's pair are not.
#[test]
fn prunes_the_mtp_layers_widened_pairs_only() {
    use metrale_model_weights::weights::WeightDtype::{BF16, FP8E4M3, UInt8};
    let n = 45;
    let l = |rel: &str| format!("model.language_model.layers.{n}.{rel}");
    let store = [
        (l("self_attn.o_proj.weight"), FP8E4M3),
        (l("self_attn.o_proj.weight_scale"), UInt8),
        (l("mlp.shared_experts.up_proj.weight"), FP8E4M3),
        (l("mlp.shared_experts.up_proj.weight_scale"), UInt8),
        (l("self_attn.q_a_proj.weight"), FP8E4M3),
        (l("eh_proj.weight"), BF16),
        (l("mlp.experts.3.up_proj.weight"), UInt8),
        (l("mlp.experts.3.up_proj.weight_scale"), FP8E4M3),
        (l("mlp.experts.4.up_proj.weight"), FP8E4M3),
        (l("mlp.experts.4.up_proj.weight_scale"), UInt8),
        (
            "model.language_model.layers.44.self_attn.o_proj.weight".to_string(),
            FP8E4M3,
        ),
        (
            "model.language_model.layers.44.self_attn.o_proj.weight_scale".to_string(),
            UInt8,
        ),
    ];
    let dtype = |x: &str| store.iter().find(|(k, _)| k == x).map(|(_, d)| *d);
    let freed: Vec<&str> = store
        .iter()
        .map(|(k, _)| k.as_str())
        .filter(|k| super::prune::is_widened_mtp_original(k, n, dtype))
        .collect();
    assert_eq!(
        freed,
        [
            l("self_attn.o_proj.weight"),
            l("self_attn.o_proj.weight_scale"),
            l("mlp.shared_experts.up_proj.weight"),
            l("mlp.shared_experts.up_proj.weight_scale"),
        ]
    );
}
