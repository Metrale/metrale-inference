// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: Two opt-in levers for the zeroing a prefill does before its first launch, both
//! measured on GLM-5.3 TP3 (cold-4k nsys pf4kZ): with a communicator every prefill chunk zeroed
//! the WHOLE arena (`BufferArena::zero_all`, 2.66 GB at 8208 arena tokens, 14 ms of GPU time
//! whatever the prompt length), and every freshly allocated KV block was zeroed with two
//! memsets per layer (5742 calls of 8 KB for a 4096-token prompt: 19 ms of host issue time,
//! 6.4 ms of GPU time). Both sit on every cold TTFT.
//!
//! Owner: model-engine (prefill).
//! Invariants:
//! - Unset or `0` keeps today's behaviour exactly; another value than the ones named is an
//!   error when the model is built (`EngineLevers::from_settings`), never a silent default.
//!   `METRALE_PREFILL_ZERO_ROWS` (with a communicator, zero only the chunk's rows) is the
//!   settings switch `runtime.prefill_zero_rows`.

/// 2026-10-12: How a prefill zeroes the KV blocks it allocates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KvZeroBlocks {
    /// Two memsets per layer per block, as the block is allocated (today).
    Block,
    /// The chunk's fresh blocks coalesced into runs of consecutive indices, one memset per
    /// layer per pool per run (`PagedKvCache::zero_blocks_run`).
    Range,
}

/// 2026-10-17: How [`super::block_mgmt::ensure_blocks_through_prefill`] fills the blocks it
/// allocates: `ModelLevers::kv_poison` and `EngineLevers::kv_zero_blocks`, per model.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PrefillBlockFill {
    pub kv_poison: bool,
    pub zero_blocks: KvZeroBlocks,
}

impl super::types::TransformerModel {
    /// 2026-10-17: This model's [`PrefillBlockFill`].
    pub(crate) fn prefill_block_fill(&self) -> PrefillBlockFill {
        PrefillBlockFill {
            kv_poison: self.levers.kv_poison,
            zero_blocks: self.engine.kv_zero_blocks,
        }
    }
}

/// 2026-10-12: `METRALE_KV_ZERO_BLOCKS=range`; unset, `0` or `block` keeps per-block zeroing.
/// `METRALE_KV_POISON` keeps its per-block fill whatever this says. Pure.
pub(crate) fn parse_kv_zero_blocks(v: Option<&str>) -> Result<KvZeroBlocks, String> {
    match v {
        None | Some("0") | Some("block") => Ok(KvZeroBlocks::Block),
        Some("range") => Ok(KvZeroBlocks::Range),
        Some(other) => Err(format!(
            "METRALE_KV_ZERO_BLOCKS={other:?}: expected 0, block or range"
        )),
    }
}

/// 2026-10-12: The runs of consecutive block indices in `blocks` (any order), as
/// `(first, count)` in ascending order; pure.
pub(crate) fn coalesce_runs(blocks: &[u32]) -> Vec<(u32, u32)> {
    let mut sorted = blocks.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for b in sorted {
        match runs.last_mut() {
            Some((first, count)) if *first + *count == b => *count += 1,
            _ => runs.push((b, 1)),
        }
    }
    runs
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-12: The block mode parses strictly: 0/block/range; anything else is an error
    /// naming the variable.
    #[test]
    fn the_zeroing_levers_parse_strictly() {
        assert_eq!(parse_kv_zero_blocks(None), Ok(KvZeroBlocks::Block));
        assert_eq!(parse_kv_zero_blocks(Some("block")), Ok(KvZeroBlocks::Block));
        assert_eq!(parse_kv_zero_blocks(Some("range")), Ok(KvZeroBlocks::Range));
        for bad in ["1", "runs", ""] {
            assert!(parse_kv_zero_blocks(Some(bad)).is_err(), "{bad:?}");
        }
    }

    /// 2026-10-12: Runs cover every block once: a stack-order allocation (descending) is one
    /// run, gaps split runs, duplicates collapse, and the empty list has no run.
    #[test]
    fn runs_cover_every_block_once() {
        assert_eq!(coalesce_runs(&[]), vec![]);
        assert_eq!(coalesce_runs(&[7, 6, 5, 4]), vec![(4, 4)]);
        assert_eq!(
            coalesce_runs(&[3, 9, 4, 10, 11, 20]),
            vec![(3, 2), (9, 3), (20, 1)]
        );
        assert_eq!(coalesce_runs(&[5, 5, 6]), vec![(5, 2)]);
        let blocks: Vec<u32> = (100..356).rev().collect();
        let runs = coalesce_runs(&blocks);
        assert_eq!(runs, vec![(100, 256)]);
        assert_eq!(runs.iter().map(|r| r.1).sum::<u32>(), 256);
    }
}
