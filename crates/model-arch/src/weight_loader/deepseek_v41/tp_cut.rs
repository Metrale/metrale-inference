// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: Which part of each per-layer tensor a tensor-parallel rank keeps, and how to read
//! just that part. The query heads split in whole output groups (`wq_b` and `attn_sink` keep
//! their heads' rows, `wo_a` its groups' rows, `wo_b` their columns); the shared and routed
//! experts split their intermediate width in whole 32-value blocks (`w1`/`w3` keep rows, `w2`
//! columns). Everything else is whole on every rank.
//!
//! Owner: model-arch weight loader.
//! Invariants:
//! - [`Local::of`]'s heads equal the config's rank-local head count (`shard_heads_for_tp`), or
//!   it refuses.
//! - A row cut reads only its rows; a column cut reads the whole tensor and keeps its columns.

use std::ops::Range;

use anyhow::{Result, ensure};
use metrale_config::ModelConfig;

use super::source::T;
use crate::tp_slice_v41::{rank_share, slice2d};

/// 2026-10-19: The MX / FP8 block edge the intermediate width splits in.
pub const BLOCK: usize = 32;

/// 2026-10-19: This rank's share of the split dimensions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Local {
    pub world: usize,
    pub heads: Range<usize>,
    pub groups: Range<usize>,
    pub inter: Range<usize>,
    pub head_dim: usize,
    pub o_rank: usize,
}

impl Local {
    /// 2026-10-19: Rank `config.tp_rank` of `config.tp_world_size`.
    pub fn of(config: &ModelConfig) -> Result<Local> {
        let (world, rank) = (config.tp_world_size.max(1), config.tp_rank);
        let total = config.pre_shard_heads()?.num_attention_heads;
        let og = config.o_groups;
        ensure!(
            og > 0 && total.is_multiple_of(og),
            "deepseek-v4.1: {total} heads in {og} output groups"
        );
        let unit = total / og;
        let heads = rank_share(total, unit, world, rank)?;
        ensure!(
            heads.len() == config.num_attention_heads,
            "deepseek-v4.1: rank {rank} keeps heads {heads:?}, the config says {} local heads",
            config.num_attention_heads
        );
        Ok(Local {
            world,
            groups: heads.start / unit..heads.end / unit,
            heads,
            inter: rank_share(config.moe_intermediate_size, BLOCK, world, rank)?,
            head_dim: config.head_dim,
            o_rank: config.o_lora_rank,
        })
    }

    /// 2026-10-19: One rank holding everything (the GGUF path, which serves one rank).
    pub fn whole(config: &ModelConfig) -> Local {
        Local {
            world: 1,
            heads: 0..config.num_attention_heads,
            groups: 0..config.o_groups.max(1),
            inter: 0..config.moe_intermediate_size,
            head_dim: config.head_dim,
            o_rank: config.o_lora_rank,
        }
    }
}

/// 2026-10-19: What a rank keeps of a `[rows, cols]` tensor, in stored elements.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Cut {
    Whole,
    Rows(Range<usize>),
    Cols(Range<usize>),
}

impl Cut {
    /// 2026-10-19: The same cut over a plane `d` times coarser (a scale plane, or packed
    /// nibbles). Refuses a range that is not whole units of `d`.
    pub fn div(&self, d: usize) -> Result<Cut> {
        let f = |r: &Range<usize>| -> Result<Range<usize>> {
            ensure!(
                r.start.is_multiple_of(d) && r.end.is_multiple_of(d),
                "cut {r:?} is not whole units of {d}"
            );
            Ok(r.start / d..r.end / d)
        };
        Ok(match self {
            Cut::Whole => Cut::Whole,
            Cut::Rows(r) => Cut::Rows(f(r)?),
            Cut::Cols(r) => Cut::Cols(f(r)?),
        })
    }
}

/// 2026-10-19: What `local`'s rank keeps of per-layer tensor `t`, in its value units (a
/// packed or scale plane takes [`Cut::div`]).
pub fn cut_of(t: T, local: &Local) -> Cut {
    if local.world == 1 {
        return Cut::Whole;
    }
    let scaled = |r: &Range<usize>, by: usize| r.start * by..r.end * by;
    match t {
        T::Sink => Cut::Rows(local.heads.clone()),
        T::WqB => Cut::Rows(scaled(&local.heads, local.head_dim)),
        T::WoA => Cut::Rows(scaled(&local.groups, local.o_rank)),
        T::WoB => Cut::Cols(scaled(&local.groups, local.o_rank)),
        T::SharedW1 | T::SharedW3 => Cut::Rows(local.inter.clone()),
        T::SharedW2 => Cut::Cols(local.inter.clone()),
        _ => Cut::Whole,
    }
}

/// 2026-10-19: What `local`'s rank keeps of routed-expert table `table` (`source::EXPERT_TABLES`
/// order), in stored bytes: w1 and w3 keep their intermediate rows (values and per-row scales
/// alike), w2 its intermediate columns, two packed values or 32 scaled values a byte.
pub fn expert_cut(table: usize, local: &Local) -> Result<Cut> {
    if local.world == 1 {
        return Ok(Cut::Whole);
    }
    let inter = local.inter.clone();
    match table {
        0..=3 => Ok(Cut::Rows(inter)),
        4 => Cut::Cols(inter).div(2),
        5 => Cut::Cols(inter).div(BLOCK),
        _ => anyhow::bail!("routed experts have six tables, not table {table}"),
    }
}

/// 2026-10-19: The bytes to read for a cut (`start`, `len` into the tensor) and the byte columns
/// to keep of them afterwards (`[_, row_bytes]` rows).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadPlan {
    pub start: usize,
    pub len: usize,
    pub row_bytes: usize,
    pub cols: Option<Range<usize>>,
}

/// 2026-10-19: The read for `cut` of a tensor of `shape` (one dimension reads as `[n, 1]`)
/// stored in `byte_size` bytes.
pub fn read_plan(shape: &[usize], byte_size: usize, cut: &Cut) -> Result<ReadPlan> {
    let (rows, cols) = match *shape {
        [n] => (n, 1),
        [r, c] => (r, c),
        _ => anyhow::bail!("a {}-dimensional tensor cannot be cut", shape.len()),
    };
    ensure!(
        rows > 0 && cols > 0 && byte_size.is_multiple_of(rows * cols),
        "{byte_size} bytes for shape {shape:?}"
    );
    let row_bytes = byte_size / rows;
    let elem = row_bytes / cols;
    Ok(match cut {
        Cut::Whole => ReadPlan {
            start: 0,
            len: byte_size,
            row_bytes,
            cols: None,
        },
        Cut::Rows(r) => {
            ensure!(r.start < r.end && r.end <= rows, "rows {r:?} of {shape:?}");
            ReadPlan {
                start: r.start * row_bytes,
                len: r.len() * row_bytes,
                row_bytes,
                cols: None,
            }
        }
        Cut::Cols(c) => {
            ensure!(
                shape.len() == 2 && c.start < c.end && c.end <= cols,
                "columns {c:?} of {shape:?}"
            );
            ReadPlan {
                start: 0,
                len: byte_size,
                row_bytes,
                cols: Some(c.start * elem..c.end * elem),
            }
        }
    })
}

/// 2026-10-19: The kept bytes of `read`, the bytes [`read_plan`] named.
pub fn finish(plan: &ReadPlan, read: Vec<u8>) -> Result<Vec<u8>> {
    ensure!(
        read.len() == plan.len,
        "read {} bytes, planned {}",
        read.len(),
        plan.len
    );
    match &plan.cols {
        None => Ok(read),
        Some(c) => slice2d(
            &read,
            plan.row_bytes,
            0..read.len() / plan.row_bytes,
            c.clone(),
        ),
    }
}

#[cfg(test)]
#[path = "tp_cut_tests.rs"]
pub(crate) mod tests;
