// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: The tp binder under a packing layer (`ScalePack`, `METRALE_GLM_MOE_SCALE_DICT`): a
//! sliced projection whose scales fit is placed packed in its own slot bytes (the slicer's cut,
//! packed), one that does not keeps its plain bytes, the counts say which; a plain plan packs
//! nothing, and a whole expert under a packing layer is refused.

use super::super::super::scale_pack::ScalePack;
use super::*;
use crate::glm5next_mlp::scale_dict::{ScaleDictPlan, pack};
use crate::glm5next_mlp::weights::ScaleLayout;

/// 2026-10-21: [`packed_expert`] with down's scales in 0x30..=0x4F (they pack); gate and up keep
/// HIDDEN / 16 = 2 groups per row, which no `_cs` entry reads, so they stay plain.
fn narrow_down(full: usize) -> Vec<(String, WeightDtype, Vec<usize>, Vec<u8>)> {
    let mut t = packed_expert(full);
    let name = qualified("mlp.experts.7.down_proj.weight_scale");
    let e = t.iter_mut().find(|x| x.0 == name).unwrap();
    e.3 = (0..e.3.len()).map(|i| 0x30 + (i * 7 % 32) as u8).collect();
    t
}

#[test]
fn a_packing_layer_places_packed_scales_where_they_fit() {
    let t = narrow_down(640);
    let sh = shard("pack-640", &t);
    let s = expert_slice(640, 3, 1).unwrap();
    let gpu = MockGpuBackend::new();
    let p = ScalePack::new(ScaleDictPlan::Compact);
    let e = bind_routed_expert(
        &gpu,
        &sh.store,
        LAYER,
        7,
        ExpertShard::Sliced(s),
        &one(7),
        Some(&p),
    )
    .unwrap();
    let (wp, ws) = (
        bytes_of(&t, "mlp.experts.7.down_proj.weight"),
        bytes_of(&t, "mlp.experts.7.down_proj.weight_scale"),
    );
    let (codes, plain) = slice_expert_cols(&wp, &ws, HIDDEN, &s).unwrap();
    let (want, layout) = pack(plain.clone(), HIDDEN, s.len / 16);
    assert!(
        matches!(layout, ScaleLayout::Compact { base: 0x30 }),
        "{layout:?}"
    );
    assert_eq!(e.down_proj.scale_layout, layout);
    assert_eq!(
        read(&gpu, e.down_proj.scale, want.len()),
        want,
        "packed bytes in the slot"
    );
    assert_ne!(want, plain);
    assert_eq!(
        read(&gpu, e.down_proj.packed, codes.len()),
        codes,
        "codes untouched"
    );
    for (proj, leaf) in [(&e.gate_proj, "gate_proj"), (&e.up_proj, "up_proj")] {
        assert_eq!(proj.scale_layout, ScaleLayout::Plain, "{leaf}");
        let ws = bytes_of(&t, &format!("mlp.experts.7.{leaf}.weight_scale"));
        let wp = bytes_of(&t, &format!("mlp.experts.7.{leaf}.weight"));
        let cut = slice_expert_rows(&wp, &ws, HIDDEN, &s).unwrap().1;
        assert_eq!(read(&gpu, proj.scale, cut.len()), cut, "{leaf} plain bytes");
    }
    let (packed, kept, _) = p.counts();
    assert_eq!((packed, kept), (1, 2));

    // 2026-10-21: A plain plan places every slicer cut as is.
    let gpu = MockGpuBackend::new();
    let off = ScalePack::new(ScaleDictPlan::NotW4a4EveryRow);
    let e = bind_routed_expert(
        &gpu,
        &sh.store,
        LAYER,
        7,
        ExpertShard::Sliced(s),
        &one(7),
        Some(&off),
    )
    .unwrap();
    assert_eq!(e.down_proj.scale_layout, ScaleLayout::Plain);
    assert_eq!(read(&gpu, e.down_proj.scale, plain.len()), plain);
    assert_eq!(off.counts().0 + off.counts().1, 0);
}

#[test]
fn a_whole_expert_under_a_packing_layer_is_refused() {
    let t = narrow_down(64);
    let sh = shard("pack-whole", &t);
    let gpu = MockGpuBackend::new();
    let p = ScalePack::new(ScaleDictPlan::Compact);
    let e = bind_routed_expert(
        &gpu,
        &sh.store,
        LAYER,
        7,
        ExpertShard::Whole,
        &one_whole(7),
        Some(&p),
    )
    .unwrap_err();
    assert!(e.to_string().contains("cannot pack"), "{e}");
}
