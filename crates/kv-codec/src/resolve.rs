// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: Resolving the KV-cache plan (ROTI): the requested pair, the boundary layer count
//! and the boundary pair become one immutable [`KvCachePlan`] with an entry for every attention
//! layer. Nothing downstream re-derives which layer holds which codec.
//!
//! Owner: kv-codec.
//! Invariants:
//! - The plan has exactly one entry per geometry passed in.
//! - Layer `i` holds the boundary pair iff `boundary_layers > 0`, the requested pair differs from
//!   the boundary pair, and `i < boundary_layers` or `i >= n - boundary_layers`.

use alloc::vec::Vec;

use crate::plan::{KvCachePlan, KvSidePlan, LayerKvPlan, RungIx};
use crate::spec::BlockGeometry;
use crate::{KvCodecId, KvPair};

/// 2026-10-18: What a plan is resolved from.
#[derive(Clone, Copy, Debug)]
pub struct KvPlanInputs<'a> {
    /// 2026-10-18: The pair `--kv-cache-dtype` names.
    pub requested: KvPair,
    /// 2026-10-18: Leading and trailing layers held at `boundary` (capped at the layer count).
    pub boundary_layers: u32,
    /// 2026-10-18: The pair the boundary layers hold.
    pub boundary: KvPair,
    /// 2026-10-18: The block geometry of every attention layer, in layer order.
    pub geometry: &'a [BlockGeometry],
}

impl KvCachePlan {
    /// 2026-10-18: The plan for `inputs`, with no VBR ladder; every side sits at rung 0.
    pub fn resolve(inputs: &KvPlanInputs<'_>) -> KvCachePlan {
        let n = inputs.geometry.len();
        let hp = usize::try_from(inputs.boundary_layers).map_or(n, |b| b.min(n));
        let overlay = hp > 0 && inputs.requested != inputs.boundary;
        let layers: Vec<LayerKvPlan> = inputs
            .geometry
            .iter()
            .enumerate()
            .map(|(i, &geometry)| {
                let at_boundary = overlay && (i < hp || i >= n - hp);
                let pair = if at_boundary {
                    inputs.boundary
                } else {
                    inputs.requested
                };
                LayerKvPlan {
                    k: rung0(pair.k()),
                    v: rung0(pair.v()),
                    geometry,
                }
            })
            .collect();
        KvCachePlan {
            layers: layers.into_boxed_slice(),
            vbr: None,
        }
    }

    /// 2026-10-18: The pair of layer `ix`, when the plan has that layer.
    pub fn pair(&self, ix: usize) -> Option<KvPair> {
        self.layers.get(ix).map(LayerKvPlan::pair)
    }

    /// 2026-10-18: The one pair every layer holds, or `None` when layers differ or there are
    /// none.
    pub fn uniform(&self) -> Option<KvPair> {
        let first = self.layers.first()?.pair();
        self.layers
            .iter()
            .all(|l| l.pair() == first)
            .then_some(first)
    }
}

impl LayerKvPlan {
    /// 2026-10-18: The layer's (K, V) pair. A plan only holds pairs `KvPair` accepted.
    pub fn pair(&self) -> KvPair {
        KvPair::from_parts(self.k.codec, self.v.codec)
    }
}

const fn rung0(codec: KvCodecId) -> KvSidePlan {
    KvSidePlan {
        codec,
        rung: RungIx(0),
    }
}

/// 2026-10-18: Entry `ix` of a per-attention-layer list. A missing entry is the coded
/// `KvPlanMissing` error (PCND), never a caller-chosen default.
pub fn layer_entry<T: Copy>(list: &[T], ix: usize) -> Result<T, metrale_error::Error> {
    list.get(ix).copied().ok_or_else(|| {
        metrale_error::error!(
            metrale_error::ErrorCode::KvPlanMissing,
            alloc::format!(
                "attention layer {ix} (the per-layer list has {})",
                list.len()
            )
        )
    })
}
