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
//!   error at first use, never a silent default.

use std::sync::OnceLock;

use anyhow::Result;

/// 2026-10-12: `METRALE_PREFILL_ZERO_ROWS=1`: with a communicator, a prefill chunk zeroes the
/// rows it will touch (`BufferArena::zero_all_rows` at the chunk length rounded up to 16, the
/// cuBLASLt pad) instead of the whole arena. Read once.
pub(crate) fn prefill_zero_rows() -> Result<bool> {
    static V: OnceLock<Result<bool, String>> = OnceLock::new();
    V.get_or_init(|| {
        parse_switch(
            "METRALE_PREFILL_ZERO_ROWS",
            std::env::var("METRALE_PREFILL_ZERO_ROWS").ok().as_deref(),
        )
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

/// 2026-10-12: How a prefill zeroes the KV blocks it allocates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KvZeroBlocks {
    /// Two memsets per layer per block, as the block is allocated (today).
    Block,
    /// The chunk's fresh blocks coalesced into runs of consecutive indices, one memset per
    /// layer per pool per run (`PagedKvCache::zero_blocks_run`).
    Range,
}

/// 2026-10-12: `METRALE_KV_ZERO_BLOCKS=range`; unset, `0` or `block` keeps per-block zeroing.
/// Read once. `METRALE_KV_POISON` keeps its per-block fill whatever this says.
pub(crate) fn kv_zero_blocks() -> Result<KvZeroBlocks> {
    static V: OnceLock<Result<KvZeroBlocks, String>> = OnceLock::new();
    V.get_or_init(|| parse_kv_zero_blocks(std::env::var("METRALE_KV_ZERO_BLOCKS").ok().as_deref()))
        .clone()
        .map_err(anyhow::Error::msg)
}

/// 2026-10-12: A 0/1 switch, pure.
pub(crate) fn parse_switch(name: &str, v: Option<&str>) -> Result<bool, String> {
    match v {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!("{name}={other:?}: expected 0 or 1")),
    }
}

/// 2026-10-12: [`kv_zero_blocks`]'s parse, pure.
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

    /// 2026-10-12: Both levers parse strictly: the switch takes 0/1, the block mode takes
    /// 0/block/range; anything else is an error naming the variable.
    #[test]
    fn the_zeroing_levers_parse_strictly() {
        assert_eq!(parse_switch("X", None), Ok(false));
        assert_eq!(parse_switch("X", Some("1")), Ok(true));
        assert!(parse_switch("X", Some("yes")).unwrap_err().contains("X="));
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
