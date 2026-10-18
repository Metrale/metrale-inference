// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The resolved KV-cache plan (ROTI): per attention layer, the codec and VBR rung of
//! each side plus the block geometry; and the variable-bit-rate ladder that enumerates which
//! downward re-quantizations are permitted. The plan is immutable once resolved; the runtime
//! rung state is a separate epoch-numbered [`KvLayoutEpoch`].
//!
//! Owner: kv-codec.
//! Invariants:
//! - A [`VbrLadder`]'s rungs strictly decrease in block bytes for every geometry it is built
//!   against, and each scope entry has a floor inside the ladder (checked in `VbrLadder::new`).
//! - `VbrLadder::permits` accepts only downward moves inside the scope, never below the floor.

use alloc::boxed::Box;
use alloc::format;

use metrale_error::{Error, ErrorCode, error};

use crate::spec::BlockGeometry;
use crate::{KvCodecId, spec};

/// 2026-10-18: Which half of an attention layer's cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum KvSide {
    /// 2026-10-18: Keys.
    K = 0,
    /// 2026-10-18: Values.
    V = 1,
}

impl KvSide {
    /// 2026-10-18: `"k"` or `"v"`.
    pub const fn name(self) -> &'static str {
        match self {
            KvSide::K => "k",
            KvSide::V => "v",
        }
    }
}

/// 2026-10-18: An attention layer's index among the cache's layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LayerIx(pub u32);

/// 2026-10-18: An index into [`VbrLadder::rungs`]; 0 is the highest-precision rung, and every
/// side is at rung 0 when the plan has no ladder.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RungIx(pub u8);

/// 2026-10-18: One side of one layer: its codec and current rung.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KvSidePlan {
    /// 2026-10-18: The codec the side is allocated and written with.
    pub codec: KvCodecId,
    /// 2026-10-18: The ladder rung that codec sits at.
    pub rung: RungIx,
}

/// 2026-10-18: One attention layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LayerKvPlan {
    /// 2026-10-18: Keys.
    pub k: KvSidePlan,
    /// 2026-10-18: Values.
    pub v: KvSidePlan,
    /// 2026-10-18: Block geometry, shared by both sides.
    pub geometry: BlockGeometry,
}

impl LayerKvPlan {
    /// 2026-10-18: Bytes of one block of `side`.
    pub const fn side_block_bytes(&self, side: KvSide) -> u64 {
        let codec = match side {
            KvSide::K => self.k.codec,
            KvSide::V => self.v.codec,
        };
        spec(codec).block_bytes(&self.geometry)
    }
}

/// 2026-10-18: The whole cache.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvCachePlan {
    /// 2026-10-18: Every attention layer, by `LayerIx`.
    pub layers: Box<[LayerKvPlan]>,
    /// 2026-10-18: The permitted downward re-quantizations, when VBR is on.
    pub vbr: Option<VbrLadder>,
}

impl KvCachePlan {
    /// 2026-10-18: K plus V bytes of one block index, summed over every layer.
    pub fn block_bytes_total(&self) -> u64 {
        self.layers
            .iter()
            .map(|l| l.side_block_bytes(KvSide::K) + l.side_block_bytes(KvSide::V))
            .sum()
    }
}

/// 2026-10-18: What asked for a downward transition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PressureTrigger {
    /// 2026-10-18: Free blocks fell below the scheduler's watermark at this tick.
    FreeBlocksBelow {
        /// 2026-10-18: The virtual-clock tick the pressure was observed at.
        tick: u64,
    },
    /// 2026-10-18: A replayed transition script (bench and gate runs).
    Scripted {
        /// 2026-10-18: The script entry.
        step: u32,
    },
}

/// 2026-10-18: The permitted states of a VBR cache. Rungs go from high precision to low.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VbrLadder {
    rungs: Box<[KvCodecId]>,
    scope: Box<[(LayerIx, KvSide)]>,
    floor: Box<[RungIx]>,
}

impl VbrLadder {
    /// 2026-10-18: A ladder over `rungs`, letting each `scope` entry fall to its `floor`.
    /// Refused unless the rungs strictly decrease in block bytes at every geometry in
    /// `geometries`, `scope` and `floor` have one entry each, and every floor is a rung.
    pub fn new(
        rungs: Box<[KvCodecId]>,
        scope: Box<[(LayerIx, KvSide)]>,
        floor: Box<[RungIx]>,
        geometries: &[BlockGeometry],
    ) -> Result<Self, Error> {
        let invalid = |rule: alloc::string::String| error!(ErrorCode::KvLadderInvalid, rule);
        if rungs.len() < 2 || rungs.len() > usize::from(u8::MAX) {
            return Err(invalid(format!("needs 2..=255 rungs, got {}", rungs.len())));
        }
        if scope.len() != floor.len() {
            return Err(invalid(format!(
                "needs one floor per scope entry, got {} scope and {} floor",
                scope.len(),
                floor.len()
            )));
        }
        for g in geometries {
            for pair in rungs.windows(2) {
                if spec(pair[1]).block_bytes(g) >= spec(pair[0]).block_bytes(g) {
                    return Err(invalid(format!(
                        "rung {} is not smaller than rung {} before it",
                        pair[1].name(),
                        pair[0].name()
                    )));
                }
            }
        }
        if let Some(f) = floor.iter().find(|f| usize::from(f.0) >= rungs.len()) {
            return Err(invalid(format!(
                "floor rung {} is outside the {}-rung ladder",
                f.0,
                rungs.len()
            )));
        }
        Ok(VbrLadder {
            rungs,
            scope,
            floor,
        })
    }

    /// 2026-10-18: The codecs, high precision first.
    pub fn rungs(&self) -> &[KvCodecId] {
        &self.rungs
    }

    /// 2026-10-18: Accept `t` only when it moves a scoped side strictly downward, by any number
    /// of rungs, to no lower than its floor.
    pub fn permits(&self, t: &KvTransition) -> Result<(), Error> {
        let refuse = |reason: &'static str| {
            error!(
                ErrorCode::KvTransitionNotPermitted,
                t.layer.0,
                t.side.name(),
                t.from.0,
                t.to.0,
                reason
            )
        };
        let Some(ix) = self
            .scope
            .iter()
            .position(|&(l, s)| l == t.layer && s == t.side)
        else {
            return Err(refuse("the layer side is outside the ladder's scope"));
        };
        if t.to <= t.from {
            return Err(refuse("only a downward move re-quantizes resident blocks"));
        }
        if usize::from(t.to.0) >= self.rungs.len() || t.to > self.floor[ix] {
            return Err(refuse("the target rung is below the side's floor"));
        }
        Ok(())
    }
}

/// 2026-10-18: A proposed move of one layer side between rungs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KvTransition {
    /// 2026-10-18: The layer.
    pub layer: LayerIx,
    /// 2026-10-18: The side.
    pub side: KvSide,
    /// 2026-10-18: The rung the side is at.
    pub from: RungIx,
    /// 2026-10-18: The rung it moves to.
    pub to: RungIx,
    /// 2026-10-18: Why.
    pub trigger: PressureTrigger,
}

/// 2026-10-18: The runtime rung of every scoped side, numbered so prefix and spill keys never
/// reuse a block written under an earlier layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvLayoutEpoch {
    /// 2026-10-18: Increments on every applied transition.
    pub epoch: u64,
    /// 2026-10-18: The current rung of each ladder scope entry, in scope order.
    pub rungs: Box<[RungIx]>,
}
