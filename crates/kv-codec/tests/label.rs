// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: Telemetry labels: one distinct label per codec, read from its spec row, and the
//! `KvPlanLabel` text the high-speed-swap summary line prints.

use metrale_kv_codec::{
    BlockGeometry, KvCachePlan, KvCodecId, KvPlanLabel, KvSidePlan, LayerKvPlan, RungIx, spec,
};

/// 2026-10-18: The label text each codec printed before the registry, except turbo2, which the
/// old summary match folded into "Turbo3" (the one declared label change); then, in wire order,
/// the 128-group codecs, the latent-cache codecs and the indexer codecs, which had no label
/// before the registry.
const GOLDEN: [&str; 14] = [
    "BF16",
    "FP8",
    "NVFP4",
    "Turbo4",
    "Turbo3",
    "Turbo2",
    "Turbo8",
    "Turbo3Tcq",
    "Turbo2Tcq",
    "Turbo1Tcq",
    "MXFP8",
    "MXFP4",
    "FP8Row",
    "Turbo4N128",
];

#[test]
fn every_codec_has_its_own_label() {
    assert_eq!(
        KvCodecId::ALL.len(),
        GOLDEN.len(),
        "a new codec needs its golden label"
    );
    for (id, want) in KvCodecId::ALL.into_iter().zip(GOLDEN) {
        assert_eq!(spec(id).label, want, "{}", id.name());
    }
    for (i, a) in KvCodecId::ALL.into_iter().enumerate() {
        for b in KvCodecId::ALL.into_iter().skip(i + 1) {
            assert_ne!(
                spec(a).label,
                spec(b).label,
                "{} and {}",
                a.name(),
                b.name()
            );
        }
    }
}

#[test]
fn turbo2_layers_are_counted_as_turbo2() {
    let label = KvPlanLabel::by_k_codec([KvCodecId::Turbo2, KvCodecId::Turbo3]);
    assert_eq!(label.to_string(), "1 Turbo2 + 1 Turbo3");
}

#[test]
fn a_plan_counts_layers_under_their_k_codec_in_label_order() {
    let side = |codec| KvSidePlan {
        codec,
        rung: RungIx(0),
    };
    let geometry = BlockGeometry {
        block_size: 16,
        num_kv_heads: 4,
        head_dim: 256,
    };
    let layer = |k, v| LayerKvPlan {
        k: side(k),
        v: side(v),
        geometry,
    };
    let mut layers = vec![layer(KvCodecId::Bf16, KvCodecId::Bf16)];
    layers.extend((0..3).map(|_| layer(KvCodecId::Turbo4, KvCodecId::Turbo3)));
    layers.push(layer(KvCodecId::Fp8, KvCodecId::Turbo2));
    let plan = KvCachePlan {
        layers: layers.into(),
        vbr: None,
    };
    let label = plan.label();
    assert_eq!(label.layers(), 5);
    assert_eq!(label.to_string(), "1 BF16 + 1 FP8 + 3 Turbo4");
}

#[test]
fn only_bf16_stores_plain_bf16() {
    for id in KvCodecId::ALL {
        assert_eq!(
            spec(id).stores_bf16(),
            id == KvCodecId::Bf16,
            "{}",
            id.name()
        );
    }
}
