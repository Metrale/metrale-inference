// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The swap record of one sequence (LIFECYCLE-DESIGN.md 15.10: `kv_swap_out` /
//! `kv_swap_in`): the state a scheduler spill moves to the host and back, as an ordered list of
//! segments sized from the circuit's declared states. The scheduler decides when a sequence
//! spills (`Model::save_sequence_state` / `restore_sequence_state`); the executor runs the
//! segments as copies on its copy stream.
//!
//! Owner: metrale-circuit (FEATURES workstream).
//! Invariants:
//! - The record is legacy's byte for byte (`sequence/state_io.rs`): for each block of the
//!   sequence's table in order and each attention layer, the K block then the V block; then, for
//!   each recurrent state of the target in layer order (a GatedDeltaNet layer's `h` at its
//!   storage format, then its conv window), one slot unit. A spill file is readable by either
//!   forward.
//! - Only the target's states: legacy spills no draft-head KV and no cache.
//! - Every size comes from the declarations ([`crate::state::StatePlan`]'s unit rule): nothing is
//!   restated here.

use std::collections::BTreeMap;

use crate::ir::{Circuit, Section};
use crate::state::{StateDtype, StateFormat, StateKind};

/// 2026-10-03: Why a swap record could not be planned.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SwapError {
    /// 2026-10-03: A keyed format the caller did not state.
    #[error("state `{state}`: no format for `{key}`")]
    MissingFormat {
        /// 2026-10-03: The state.
        state: String,
        /// 2026-10-03: Its key.
        key: String,
    },
    /// 2026-10-03: An attention layer whose K or V side is not declared, or declared twice.
    #[error("attention layer {0} does not declare exactly one paged `k` and one paged `v`")]
    Sides(usize),
    /// 2026-10-03: A KV or recurrent state outside the layers.
    #[error("state `{0}` is swapped per layer but belongs to none")]
    Layerless(String),
    /// 2026-10-03: A size that overflows.
    #[error("the swap record of `{0}` overflows")]
    Overflow(String),
}

/// 2026-10-03: One attention layer's KV, per block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvLayer {
    /// 2026-10-03: The layer index in the target's stack.
    pub layer: usize,
    /// 2026-10-03: Bytes of one K block, and of one V block.
    pub k_block_bytes: u64,
    pub v_block_bytes: u64,
}

/// 2026-10-03: One recurrent state's slot unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecurrentUnit {
    /// 2026-10-03: The declaration's id (`l0.gdn.h`).
    pub state: String,
    /// 2026-10-03: Its layer.
    pub layer: usize,
    /// 2026-10-03: Its local id (`h`, `conv`).
    pub local: String,
    /// 2026-10-03: Bytes of one slot's unit at the stored format.
    pub bytes: u64,
}

/// 2026-10-03: The record's layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwapPlan {
    /// 2026-10-03: The attention layers, in layer order (the KV cache's layer order).
    pub kv: Vec<KvLayer>,
    /// 2026-10-03: The recurrent units, in record order.
    pub recurrent: Vec<RecurrentUnit>,
}

/// 2026-10-03: What a segment of the record is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Piece {
    /// 2026-10-03: The K side of block `block` (index into the sequence's table) of the
    /// `attn`-th attention layer.
    K { attn: usize, block: usize },
    /// 2026-10-03: Its V side.
    V { attn: usize, block: usize },
    /// 2026-10-03: Recurrent unit `index` of [`SwapPlan::recurrent`].
    Recurrent { index: usize },
}

/// 2026-10-03: One segment: what it is, where it starts in the record, its bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segment {
    pub piece: Piece,
    pub offset: u64,
    pub bytes: u64,
}

impl SwapPlan {
    /// 2026-10-03: The record of `circuit`'s target, with the keyed formats in `formats`
    /// (`kv_cache_dtype`, `ssm_h_storage`) and `block_size` tokens per KV block.
    pub fn new(
        circuit: &Circuit,
        formats: &BTreeMap<String, StateDtype>,
        block_size: u64,
    ) -> Result<Self, SwapError> {
        let mut kv: BTreeMap<usize, (Vec<u64>, Vec<u64>)> = BTreeMap::new();
        let mut recurrent = Vec::new();
        let swapped = |k: StateKind| matches!(k, StateKind::PagedKv | StateKind::Recurrent);
        for s in circuit
            .states
            .iter()
            .filter(|s| s.section == Section::Main && swapped(s.kind))
        {
            let dtype = match &s.format {
                StateFormat::Fixed(d) => *d,
                StateFormat::Keyed(k) => {
                    *formats.get(k).ok_or_else(|| SwapError::MissingFormat {
                        state: s.id.clone(),
                        key: k.clone(),
                    })?
                }
            };
            let unit = s
                .elements
                .checked_mul(dtype.size())
                .ok_or_else(|| SwapError::Overflow(s.id.clone()))?;
            let layer = s.layer.ok_or_else(|| SwapError::Layerless(s.id.clone()))?;
            match s.kind {
                StateKind::PagedKv => {
                    let block = unit
                        .checked_mul(block_size)
                        .ok_or_else(|| SwapError::Overflow(s.id.clone()))?;
                    let sides = kv.entry(layer).or_default();
                    match s.local.as_str() {
                        "k" => sides.0.push(block),
                        "v" => sides.1.push(block),
                        _ => return Err(SwapError::Sides(layer)),
                    }
                }
                StateKind::Recurrent => recurrent.push(RecurrentUnit {
                    state: s.id.clone(),
                    layer,
                    local: s.local.clone(),
                    bytes: unit,
                }),
                _ => unreachable!("filtered to the swapped kinds"),
            }
        }
        let kv = kv
            .into_iter()
            .map(|(layer, (k, v))| match (k.as_slice(), v.as_slice()) {
                ([k], [v]) => Ok(KvLayer {
                    layer,
                    k_block_bytes: *k,
                    v_block_bytes: *v,
                }),
                _ => Err(SwapError::Sides(layer)),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(SwapPlan { kv, recurrent })
    }

    /// 2026-10-03: The record's segments for a sequence of `blocks` KV blocks, in order.
    pub fn segments(&self, blocks: usize) -> Vec<Segment> {
        let mut out = Vec::with_capacity(blocks * self.kv.len() * 2 + self.recurrent.len());
        let mut at = 0u64;
        let mut push = |piece, bytes| {
            out.push(Segment {
                piece,
                offset: at,
                bytes,
            });
            at += bytes;
        };
        for block in 0..blocks {
            for (attn, l) in self.kv.iter().enumerate() {
                push(Piece::K { attn, block }, l.k_block_bytes);
                push(Piece::V { attn, block }, l.v_block_bytes);
            }
        }
        for (index, r) in self.recurrent.iter().enumerate() {
            push(Piece::Recurrent { index }, r.bytes);
        }
        out
    }

    /// 2026-10-03: Bytes of the record of `blocks` blocks.
    pub fn record_bytes(&self, blocks: u64) -> u64 {
        let per_block: u64 = self
            .kv
            .iter()
            .map(|l| l.k_block_bytes + l.v_block_bytes)
            .sum();
        per_block * blocks + self.recurrent.iter().map(|r| r.bytes).sum::<u64>()
    }

    /// 2026-10-03: Bytes of one of the runner's two page-locked staging chunks: twice the largest
    /// segment, so a chunk holds several KV blocks and any recurrent unit. The memory model's
    /// host term is twice this.
    pub fn staging_chunk(&self) -> u64 {
        2 * self.largest_segment().max(1)
    }

    /// 2026-10-03: The largest single segment: a staging chunk must hold it.
    pub fn largest_segment(&self) -> u64 {
        self.kv
            .iter()
            .flat_map(|l| [l.k_block_bytes, l.v_block_bytes])
            .chain(self.recurrent.iter().map(|r| r.bytes))
            .max()
            .unwrap_or(0)
    }
}

#[cfg(test)]
#[path = "swap_tests.rs"]
mod swap_tests;
