// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Where the whole layout's decoded csf scales live: one slab per layer adopted by
//! the derived-weight ledger, sized as the load pre-flight charges it. A child of
//! `expert_scales_tests` for its encoder and fixtures (split at the file cap).
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

use super::*;

/// 2026-10-17: Experts `ID` and `ID + 1` of width `full`, csf-encoded or native.
fn two_experts(full: usize, csf: bool) -> Vec<Tensor> {
    let mut t = expert(full, csf);
    let (this, next) = (format!(".experts.{ID}."), format!(".experts.{}.", ID + 1));
    let twin: Vec<Tensor> = t
        .iter()
        .map(|(n, d, s, b)| (n.replace(&this, &next), *d, s.clone(), b.clone()))
        .collect();
    t.extend(twin);
    t
}

/// 2026-10-17: Whole layout: the decoded scales of a layer's experts are one allocation, adopted
/// by the derived-weight ledger under the csf label at the size `slot_layout` gives and the
/// pre-flight charge predicts; the native twin (negative control) allocates nothing.
#[test]
fn whole_layout_decoded_scales_are_one_ledgered_slab_per_layer() {
    let full = 128;
    let resident = |name: &str| !is_csf_component(name);
    let sizes = [full * HIDDEN / 16, full * HIDDEN / 16, HIDDEN * full / 16];
    let stride = slot_layout(&sizes, SLAB_ALIGN).unwrap().stride;
    for csf in [true, false] {
        let gpu = MockGpuBackend::new();
        let sh = shard(
            &format!("slab-{csf}"),
            &two_experts(full, csf),
            &gpu,
            resident,
        );
        let native = shard(
            &format!("slab-twin-{csf}"),
            &two_experts(full, false),
            &gpu,
            resident,
        );
        let (bytes0, allocs0) = (sh.store.derived().bytes(), gpu.alloc_count());
        let slab = ExpertSlab::new(ID..ID + 2, slab_label(ExpertShard::Whole));
        let bound: Vec<_> = (ID..ID + 2)
            .map(|id| {
                bind_routed_expert(&gpu, &sh.store, LAYER, id, ExpertShard::Whole, &slab, None)
            })
            .collect::<anyhow::Result<_>>()
            .unwrap();
        let (grew, allocs) = (
            sh.store.derived().bytes() - bytes0,
            gpu.alloc_count() - allocs0,
        );
        if !csf {
            assert_eq!((grew, allocs), (0, 0), "native scales stay the store's own");
            continue;
        }
        assert_eq!((grew, allocs), (2 * stride, 1), "one slab of two slots");
        assert_eq!(
            sh.store.derived().by_label(),
            vec![(CSF_SCALE_SLAB_LABEL, 2 * stride, 1)]
        );
        let charge: usize = two_experts(full, true)
            .iter()
            .map(|(n, _, shape, _)| super::super::whole_layout_charge(n, shape))
            .sum();
        assert_eq!(charge, 2 * stride, "the pre-flight charge is the slab");
        let slab_twin = ExpertSlab::new(ID..ID + 2, slab_label(ExpertShard::Whole));
        for (id, e) in (ID..ID + 2).zip(&bound) {
            let w = bind_routed_expert(
                &gpu,
                &native.store,
                LAYER,
                id,
                ExpertShard::Whole,
                &slab_twin,
                None,
            )
            .unwrap();
            for (got, want, n) in [
                (&e.gate_proj, &w.gate_proj, sizes[0]),
                (&e.up_proj, &w.up_proj, sizes[1]),
                (&e.down_proj, &w.down_proj, sizes[2]),
            ] {
                assert_eq!(
                    read(&gpu, got.scale, n),
                    read(&gpu, want.scale, n),
                    "expert {id}"
                );
            }
        }
    }
}

/// 2026-10-17: The charge is stated for the whole layout only and only on a codes component; an
/// overflowing shape charges `usize::MAX` so the pre-flight refuses it.
#[test]
fn the_preflight_charge_covers_whole_layout_csf_scales_only() {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    let charge = super::super::super::Glm5NextWeightLoader
        .defer_charge(&config)
        .expect("the whole layout states a charge");
    let leaf = |l: &str| qualified(&format!("mlp.experts.{ID}.gate_proj.weight_scale.{l}"));
    assert_eq!(
        charge(&leaf("csf32_codes"), WeightDtype::UInt8, &[2048]),
        4096
    );
    assert_eq!(
        charge(&leaf("csf32_codes"), WeightDtype::UInt8, &[2049]),
        8192
    );
    assert_eq!(charge(&leaf("csf32_bases"), WeightDtype::UInt8, &[2048]), 0);
    assert_eq!(
        charge(&leaf("csf32_exceptions"), WeightDtype::UInt16, &[9]),
        0
    );
    let head = "lm_head.weight";
    assert_eq!(charge(head, WeightDtype::FP8E4M3, &[4096, 4096]), 0);
    let huge = [usize::MAX / 2 + 1];
    assert_eq!(
        charge(&leaf("csf32_codes"), WeightDtype::UInt8, &huge),
        usize::MAX
    );
    config.moe_expert_layout = metrale_config::MoeExpertLayout::Tp;
    assert!(
        super::super::super::Glm5NextWeightLoader
            .defer_charge(&config)
            .is_none(),
        "the tp layout's slab is not charged here"
    );
}
