// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The per-rank tensor-parallel split: which contiguous range of a `total`-wide
//! dimension each TP rank owns, and the serve's head-count division built on it.
//!
//! Owner: config.
//! Invariants:
//! - [`tp_split`] over ranks `0..tp_size` partitions `0..total` into contiguous, ordered,
//!   non-empty ranges whose starts and lengths are multiples of `align`.
//! - When `total` is a multiple of `tp_size * align`, rank `r` owns
//!   `[r * total / tp_size, (r + 1) * total / tp_size)`: the even split every loader used
//!   before uneven splits existed, so an even geometry shards byte-identically.
//! - Otherwise the first `units % tp_size` ranks own one `align`-wide unit more than the rest,
//!   so the widest and narrowest ranks differ by exactly one unit.
//! - [`ModelConfig::shard_heads_for_tp`] runs at most once per config and records the
//!   pre-shard head counts it divided; [`ModelConfig::pre_shard_heads`] returns them.

use anyhow::{Result, bail};

use super::ModelConfig;

/// 2026-10-08: One rank's contiguous range `[start, start + len)` of a split dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TpSlice {
    pub start: usize,
    pub len: usize,
}

impl TpSlice {
    pub fn end(&self) -> usize {
        self.start + self.len
    }
    pub fn range(&self) -> std::ops::Range<usize> {
        self.start..self.end()
    }
}

/// 2026-10-08: Rank `tp_rank`'s share of a `total`-wide dimension split over `tp_size` ranks in
/// `align`-wide units (see the module invariants).
///
/// `align` is the granularity a consumer needs: 1 for a head count, 8 for a BF16 GEMM K
/// dimension whose kernels load 8 elements at a time. At `tp_size == 1` nothing is sliced, so
/// the whole dimension is returned whatever `align` is.
///
/// Errors when `tp_size` or `align` is 0, `tp_rank >= tp_size`, `total` is not a multiple of
/// `align`, or there are fewer units than ranks (a rank would own nothing).
pub fn tp_split(total: usize, tp_size: usize, tp_rank: usize, align: usize) -> Result<TpSlice> {
    if tp_size == 0 {
        bail!("tp_split: tp_size is 0");
    }
    if tp_rank >= tp_size {
        bail!("tp_split: tp_rank {tp_rank} >= tp_size {tp_size}");
    }
    if align == 0 {
        bail!("tp_split: align is 0");
    }
    if tp_size == 1 {
        return Ok(TpSlice {
            start: 0,
            len: total,
        });
    }
    if !total.is_multiple_of(align) {
        bail!("tp_split: {total} is not a multiple of the {align}-element alignment");
    }
    let units = total / align;
    if units < tp_size {
        bail!(
            "tp_split: {total} in {align}-element units gives {units} units for {tp_size} \
             ranks; at least one rank would own nothing"
        );
    }
    let (base, rem) = (units / tp_size, units % tp_size);
    let start_units = tp_rank * base + tp_rank.min(rem);
    let len_units = base + usize::from(tp_rank < rem);
    Ok(TpSlice {
        start: start_units * align,
        len: len_units * align,
    })
}

/// 2026-10-08: How a weight loader shards head counts under TP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TpSupport {
    /// 2026-10-08: The loader does not slice for TP; `--tp-size > 1` is refused.
    Unsupported,
    /// 2026-10-08: Every head count must divide over `tp_size`.
    Even,
    /// 2026-10-08: Head counts split by [`tp_split`], so ranks may own different counts. The
    /// loader's plans must read the pre-shard counts from [`ModelConfig::pre_shard_heads`]
    /// rather than multiply a local count back up.
    Uneven,
}

/// 2026-10-08: The head counts [`ModelConfig::shard_heads_for_tp`] divided, as they were
/// before the division.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TpPreShardHeads {
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
}

/// 2026-10-08: The `Even` refusal, worded as it was before uneven splits existed.
fn require_divisible(name: &str, count: usize, tp_size: usize) -> Result<()> {
    if !count.is_multiple_of(tp_size) {
        bail!("TP requires {name} ({count}) divisible by tp_size ({tp_size})");
    }
    Ok(())
}

/// 2026-10-08: This rank's `(outer, inner)` head counts for a grouped pair (KV heads and the Q
/// heads that share them; linear key heads and their value heads). The outer count is split
/// head by head and the inner count in whole groups, so every local inner head keeps its outer
/// head. A zero count stays zero.
fn split_grouped(
    outer_name: &str,
    outer: usize,
    inner_name: &str,
    inner: usize,
    tp_size: usize,
    tp_rank: usize,
) -> Result<(usize, usize)> {
    if outer == 0 {
        return Ok((0, tp_split(inner, tp_size, tp_rank, 1)?.len));
    }
    if !inner.is_multiple_of(outer) {
        bail!(
            "uneven TP needs {inner_name} ({inner}) to be a multiple of {outer_name} ({outer}), \
             so that a rank's {inner_name} share whole groups"
        );
    }
    let group = inner / outer;
    Ok((
        tp_split(outer, tp_size, tp_rank, 1)?.len,
        tp_split(inner, tp_size, tp_rank, group)?.len,
    ))
}

impl ModelConfig {
    /// 2026-10-08: Replace the attention and linear-attention head counts with this rank's
    /// (`tp_rank` of `tp_world_size`, set beforehand) and record the pre-shard counts. A
    /// no-op at `tp_world_size <= 1`.
    ///
    /// `Even` refuses a count that does not divide, as serve's topology always did; `Uneven`
    /// splits by [`tp_split`]. Linear-attention counts are touched only when either is nonzero.
    /// Errors when called twice, or with [`TpSupport::Unsupported`] above one rank.
    pub fn shard_heads_for_tp(&mut self, support: TpSupport) -> Result<()> {
        let (tp_size, tp_rank) = (self.tp_world_size, self.tp_rank);
        if tp_size <= 1 {
            return Ok(());
        }
        if self.tp_pre_shard_heads.is_some() {
            bail!("shard_heads_for_tp: the head counts are already divided for TP");
        }
        let pre = TpPreShardHeads {
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            linear_num_key_heads: self.linear_num_key_heads,
            linear_num_value_heads: self.linear_num_value_heads,
        };
        let has_linear = pre.linear_num_key_heads > 0 || pre.linear_num_value_heads > 0;
        let (q, kv, lk, lv) = match support {
            TpSupport::Unsupported => {
                bail!(
                    "shard_heads_for_tp: the {} loader does not support TP",
                    self.model_type
                )
            }
            TpSupport::Even => {
                require_divisible("num_attention_heads", pre.num_attention_heads, tp_size)?;
                require_divisible("num_key_value_heads", pre.num_key_value_heads, tp_size)?;
                if has_linear {
                    require_divisible("linear_num_key_heads", pre.linear_num_key_heads, tp_size)?;
                    require_divisible(
                        "linear_num_value_heads",
                        pre.linear_num_value_heads,
                        tp_size,
                    )?;
                }
                let d = |n: usize| n / tp_size;
                (
                    d(pre.num_attention_heads),
                    d(pre.num_key_value_heads),
                    d(pre.linear_num_key_heads),
                    d(pre.linear_num_value_heads),
                )
            }
            TpSupport::Uneven => {
                let (kv, q) = split_grouped(
                    "num_key_value_heads",
                    pre.num_key_value_heads,
                    "num_attention_heads",
                    pre.num_attention_heads,
                    tp_size,
                    tp_rank,
                )?;
                let (lk, lv) = if has_linear {
                    split_grouped(
                        "linear_num_key_heads",
                        pre.linear_num_key_heads,
                        "linear_num_value_heads",
                        pre.linear_num_value_heads,
                        tp_size,
                        tp_rank,
                    )?
                } else {
                    (0, 0)
                };
                (q, kv, lk, lv)
            }
        };
        self.num_attention_heads = q;
        self.num_key_value_heads = kv;
        self.linear_num_key_heads = lk;
        self.linear_num_value_heads = lv;
        self.tp_pre_shard_heads = Some(pre);
        Ok(())
    }

    /// 2026-10-08: The head counts before TP division: the recorded ones above one rank, the
    /// current ones at one. Errors above one rank when [`Self::shard_heads_for_tp`] has not run,
    /// since the per-rank counts alone cannot recover an uneven split.
    pub fn pre_shard_heads(&self) -> Result<TpPreShardHeads> {
        if self.tp_world_size <= 1 {
            return Ok(TpPreShardHeads {
                num_attention_heads: self.num_attention_heads,
                num_key_value_heads: self.num_key_value_heads,
                linear_num_key_heads: self.linear_num_key_heads,
                linear_num_value_heads: self.linear_num_value_heads,
            });
        }
        match self.tp_pre_shard_heads {
            Some(p) => Ok(p),
            None => bail!(
                "pre_shard_heads: tp_world_size is {} but the head counts were never divided \
                 (ModelConfig::shard_heads_for_tp)",
                self.tp_world_size
            ),
        }
    }
}

#[cfg(test)]
#[path = "tp_split_tests.rs"]
mod tests;
