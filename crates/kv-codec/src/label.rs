// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: Telemetry labels for a KV-cache plan (OWAY: printed, never read back). The text
//! comes from each codec's spec row, so a label cannot name another codec.
//!
//! Owner: kv-codec.
//! Invariants:
//! - Layers are counted under their K-side codec's label, in label order.

use alloc::collections::BTreeMap;
use core::fmt;

use crate::{KvCachePlan, KvCodecId, spec};

/// 2026-10-18: Attention-layer count per K-side codec label, displayed as `60 Turbo4 + 4 BF16`
/// (labels in byte order, joined by ` + `).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KvPlanLabel {
    by_k: BTreeMap<&'static str, u64>,
}

impl KvPlanLabel {
    /// 2026-10-18: The label of layers whose K-side codecs are `k`, one item per layer.
    pub fn by_k_codec(k: impl IntoIterator<Item = KvCodecId>) -> Self {
        let mut by_k = BTreeMap::new();
        for id in k {
            *by_k.entry(spec(id).label).or_default() += 1;
        }
        KvPlanLabel { by_k }
    }

    /// 2026-10-18: Layers counted.
    pub fn layers(&self) -> u64 {
        self.by_k.values().sum()
    }
}

impl fmt::Display for KvPlanLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, (label, n)) in self.by_k.iter().enumerate() {
            if i > 0 {
                f.write_str(" + ")?;
            }
            write!(f, "{n} {label}")?;
        }
        Ok(())
    }
}

impl KvCachePlan {
    /// 2026-10-18: The plan's telemetry label.
    pub fn label(&self) -> KvPlanLabel {
        KvPlanLabel::by_k_codec(self.layers.iter().map(|l| l.k.codec))
    }
}
