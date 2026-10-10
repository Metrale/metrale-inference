// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: `--draft-tp-size`: the DFlash drafter split across the tensor-parallel ranks,
//! so the propose runs on every rank in step instead of on rank 0 while the others wait.
//!
//! Split (`plan`): the KV heads by `tp_split` (8 over 3 ranks: 3/3/2), the query heads with
//! them (whole GQA groups: 12/12/8), the MLP intermediate in [`INTER_ALIGN`] units, the LM-head
//! rows in [`VOCAB_ALIGN`] units. q/k/v/gate/up and the LM head are row slices (pointer
//! offsets into the loaded weights); o_proj and down are column slices, repacked
//! ([`shard_weights`]). The norms, the DFlash2 convs, `fc` and the selector stay whole.
//!
//! The drafter context stays rank 0's: its taps, commits, slides and trims are unchanged, and
//! the other ranks hold no context accumulator. A propose (`tp_propose.rs`) is one collective
//! step: rank 0 broadcasts a [`Header`] (the block, its anchor and position, the context rows
//! to add and their positions), then `fc` of those rows; every rank writes its own KV heads'
//! K/V for them into its paged cache, runs the block forward with an all-reduce after o_proj
//! and after down, takes the top-16 of its vocab rows and all-gathers it; the merged lists
//! (the one-block kernel's order and ties, `dflash2.cu`) feed the selector on every rank.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants:
//! - [`plan`]'s ranges cover each dimension exactly once over the ranks, in rank order.
//! - Every rank makes the same collectives in the same order: the header, its overflow, the
//!   context `fc` rows in `ctx_window` chunks, two all-reduces per layer, one all-gather.

use anyhow::{Result, bail, ensure};
use metrale_config::tp_split;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_layers::weight_map::DenseWeight;

use crate::weight_loader::DflashWeights;

/// 2026-10-10: The LM-head row unit of the split (the target head's vocab split uses 64 too).
pub const VOCAB_ALIGN: usize = 64;
/// 2026-10-10: The MLP intermediate unit: whole 128-wide K tiles of the FP8 row GEMMs.
pub const INTER_ALIGN: usize = 128;

/// 2026-10-10: This rank's place in the split, as the head carries it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DflashTp {
    pub rank: usize,
    pub world: usize,
    /// 2026-10-10: The first global vocab row this rank's LM-head rows hold.
    pub vocab_base: usize,
}

/// 2026-10-10: This rank's `(start, len)` of each split dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TpPlan {
    pub rank: usize,
    pub world: usize,
    pub q_heads: (usize, usize),
    pub kv_heads: (usize, usize),
    pub inter: (usize, usize),
    pub vocab: (usize, usize),
}

/// 2026-10-10: The split of a drafter with `q_heads` query and `kv_heads` KV heads (whole GQA
/// groups per rank), an `inter`-wide MLP and a `vocab`-row LM head over `world` ranks, for
/// `rank`. Errors when the heads do not group evenly or a rank would own nothing.
pub fn plan(
    q_heads: usize,
    kv_heads: usize,
    inter: usize,
    vocab: usize,
    world: usize,
    rank: usize,
) -> Result<TpPlan> {
    ensure!(
        kv_heads > 0 && q_heads.is_multiple_of(kv_heads),
        "DFlash TP: {q_heads} query heads do not group over {kv_heads} KV heads"
    );
    let group = q_heads / kv_heads;
    let kv = tp_split(kv_heads, world, rank, 1)?;
    let mlp = tp_split(inter, world, rank, INTER_ALIGN)?;
    let v = tp_split(
        vocab.next_multiple_of(VOCAB_ALIGN),
        world,
        rank,
        VOCAB_ALIGN,
    )?;
    let (v0, v1) = (v.start.min(vocab), (v.start + v.len).min(vocab));
    ensure!(
        v1 > v0,
        "DFlash TP: rank {rank} of {world} owns no vocab row of {vocab}"
    );
    Ok(TpPlan {
        rank,
        world,
        q_heads: (kv.start * group, kv.len * group),
        kv_heads: (kv.start, kv.len),
        inter: (mlp.start, mlp.len),
        vocab: (v0, v1 - v0),
    })
}

/// 2026-10-10: Words of the propose header broadcast.
pub const HEADER_WORDS: usize = 64;
/// 2026-10-10: Shipped-row positions the header carries inline; more take a second broadcast.
pub const HEADER_INLINE: usize = HEADER_WORDS - 8;
const HEADER_OK: u32 = 0x5446_4431;
const HEADER_ABORT: u32 = 0x5446_4430;

/// 2026-10-10: One propose as rank 0 announces it: the block width (rows, anchor included),
/// the anchor token and its position, the context rows the attention reads, and the context
/// rows `[ship_start, ship_start + positions.len())` whose K/V the ranks write first, with
/// their RoPE positions. `None` from [`Header::decode`] is rank 0's abort.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Header {
    pub width: usize,
    pub anchor: u32,
    pub position: usize,
    pub ctx_count: usize,
    pub ship_start: usize,
    pub positions: Vec<i32>,
}

impl Header {
    /// 2026-10-10: The fixed header words; positions past [`HEADER_INLINE`] go in the overflow.
    pub fn encode(&self) -> [u32; HEADER_WORDS] {
        let mut w = [0u32; HEADER_WORDS];
        w[0] = HEADER_OK;
        w[1] = self.width as u32;
        w[2] = self.anchor;
        w[3] = self.position as u32;
        w[4] = self.ctx_count as u32;
        w[5] = self.ship_start as u32;
        w[6] = self.positions.len() as u32;
        for (i, p) in self.positions.iter().take(HEADER_INLINE).enumerate() {
            w[8 + i] = *p as u32;
        }
        w
    }

    /// 2026-10-10: The positions that do not fit inline (sent as a second broadcast).
    pub fn overflow(&self) -> &[i32] {
        &self.positions[self.positions.len().min(HEADER_INLINE)..]
    }

    /// 2026-10-10: Rank 0's abort header: the propose failed before its forward, and the
    /// other ranks return without one.
    pub fn abort_words() -> [u32; HEADER_WORDS] {
        let mut w = [0u32; HEADER_WORDS];
        w[0] = HEADER_ABORT;
        w
    }

    /// 2026-10-10: Decode `w`; the overflow positions are read by `read_overflow` when
    /// [`Header::overflow_len`] is non-zero. `Ok(None)` is an abort; anything else malformed
    /// is an error.
    pub fn decode(w: &[u32; HEADER_WORDS]) -> Result<Option<Self>> {
        match w[0] {
            HEADER_ABORT => return Ok(None),
            HEADER_OK => {}
            other => bail!("DFlash TP propose header: bad tag {other:#x}"),
        }
        let n = w[6] as usize;
        let inline = n.min(HEADER_INLINE);
        let width = w[1] as usize;
        ensure!(width >= 1, "DFlash TP propose header: an empty block");
        Ok(Some(Self {
            width,
            anchor: w[2],
            position: w[3] as usize,
            ctx_count: w[4] as usize,
            ship_start: w[5] as usize,
            positions: w[8..8 + inline].iter().map(|&p| p as i32).collect(),
        }))
    }

    /// 2026-10-10: The positions still to read after a decoded header whose word 6 said
    /// `total`.
    pub fn overflow_len(w: &[u32; HEADER_WORDS]) -> usize {
        (w[6] as usize).saturating_sub(HEADER_INLINE)
    }
}

/// 2026-10-10: A column-sliced weight's source: the whole `[rows, full_k]` BF16 weight and the
/// first of this rank's columns. The FP8 copy quantizes the whole rows and keeps this rank's
/// columns with the whole rows' scales (`from_weights/fp8_weights.rs`), so every FP8 byte and
/// scale equals the one-rank drafter's.
#[derive(Clone, Copy, Debug)]
pub struct ColSlice {
    pub full: DenseWeight,
    pub full_k: usize,
    pub start: usize,
}

/// 2026-10-10: `weights` cut to this rank's part under `plan`: the config's head counts,
/// intermediate size and vocab become the local ones; q/k/v/gate/up point at their row slice;
/// o_proj and down are repacked into new `[hidden, local]` buffers (2D copies on stream 0,
/// synchronised before return). Everything else is unchanged.
pub fn shard_weights(
    mut weights: DflashWeights,
    plan: &TpPlan,
    gpu: &dyn GpuBackend,
) -> Result<DflashWeights> {
    let c = &weights.config;
    let (h, hd, nq, inter) = (
        c.hidden_size,
        c.head_dim,
        c.num_attention_heads,
        c.intermediate_size,
    );
    let rows = |w: &DenseWeight, first_row: usize| DenseWeight {
        weight: w.weight.offset(first_row * h * 2),
    };
    let cols = |w: &DenseWeight, full: usize, (start, len): (usize, usize)| -> Result<_> {
        let dst = gpu.alloc(h * len * 2)?;
        gpu.copy_d2d_2d_async(
            w.weight.offset(start * 2),
            full * 2,
            dst,
            len * 2,
            len * 2,
            h,
            0,
        )?;
        Ok(DenseWeight { weight: dst })
    };
    let q_cols = (plan.q_heads.0 * hd, plan.q_heads.1 * hd);
    for l in weights.layers.iter_mut() {
        l.q_proj = rows(&l.q_proj, plan.q_heads.0 * hd);
        l.k_proj = rows(&l.k_proj, plan.kv_heads.0 * hd);
        l.v_proj = rows(&l.v_proj, plan.kv_heads.0 * hd);
        l.gate_proj = rows(&l.gate_proj, plan.inter.0);
        l.up_proj = rows(&l.up_proj, plan.inter.0);
        l.o_proj_cols = Some(ColSlice {
            full: l.o_proj,
            full_k: nq * hd,
            start: q_cols.0,
        });
        l.down_proj_cols = Some(ColSlice {
            full: l.down_proj,
            full_k: inter,
            start: plan.inter.0,
        });
        l.o_proj = cols(&l.o_proj, nq * hd, q_cols)?;
        l.down_proj = cols(&l.down_proj, inter, plan.inter)?;
    }
    gpu.synchronize(0)?;
    weights.config.num_attention_heads = plan.q_heads.1;
    weights.config.num_key_value_heads = plan.kv_heads.1;
    weights.config.intermediate_size = plan.inter.1;
    weights.config.vocab_size = plan.vocab.1;
    Ok(weights)
}

#[cfg(test)]
#[path = "tp_tests.rs"]
mod tests;
