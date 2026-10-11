// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The seeded mutations a contract must catch. Each is applied from the test side
//! only: to the operands handed to the kernel (a corrupted scale, a re-laid-out scale tensor, a
//! zeroed expert, swapped KV pages, a stale state), to a runtime scalar (the rope base), to the
//! output split (shard boundaries off by one), to the symbol launched (another entry point), or
//! as an emulated arm (a narrower accumulator). Production code is never changed.
//!
//! Owner: metrale-accuracy.
//! Invariants:
//! - The spelling is the contract's; `parse(m.name()) == Some(m)`.
//! - A mutation that does not apply to a contract's reference is a contract error, not a skip.

use crate::elem::{self, Elem};

/// 2026-10-09: One mutation.
#[derive(Debug, Clone, PartialEq)]
pub enum Mutation {
    /// 2026-10-09: One block-scale element of the weight has its exponent's lowest bit flipped
    /// (the scale doubles or halves).
    CorruptBlockScale,
    /// 2026-10-09: The weight's block scales are re-laid out at twice the group size (the odd
    /// groups read the even groups' scales).
    SwapScaleGranularity,
    /// 2026-10-09: The rope base handed to the kernel is ten times the model's.
    WrongRopeBase,
    /// 2026-10-09: The output placement of every shard after the first is off by one column
    /// against its weight rows (the misaligned-split bug class).
    SplitOffByOne,
    /// 2026-10-09: One routed expert's weights are zero (the generator routes to it).
    ZeroExpert,
    /// 2026-10-09: An emulated arm: the declared pipeline with every accumulation in `Elem`.
    Accumulate(Elem),
    /// 2026-10-09: Another entry point is launched on the same operands.
    Symbol(String),
    /// 2026-10-09: Two KV pages swap places in the block table.
    KvPageSwap,
    /// 2026-10-09: The recurrent state handed in is one step stale.
    StateStale,
}

const PLAIN: [(&str, fn() -> Mutation); 7] = [
    ("corrupt_block_scale", || Mutation::CorruptBlockScale),
    ("swap_scale_granularity", || Mutation::SwapScaleGranularity),
    ("wrong_rope_base", || Mutation::WrongRopeBase),
    ("split_off_by_one", || Mutation::SplitOffByOne),
    ("zero_expert", || Mutation::ZeroExpert),
    ("kv_page_swap", || Mutation::KvPageSwap),
    ("state_stale", || Mutation::StateStale),
];

impl Mutation {
    /// 2026-10-09: Parse the contract spelling (`accumulate:bf16`, `symbol:module::function`).
    pub fn parse(s: &str) -> Option<Self> {
        if let Some(f) = s.strip_prefix("accumulate:") {
            let e = [elem::BF16, elem::F16, elem::E4M3]
                .into_iter()
                .find(|e| e.name == f)?;
            return Some(Mutation::Accumulate(e));
        }
        if let Some(k) = s.strip_prefix("symbol:") {
            return k.contains("::").then(|| Mutation::Symbol(k.to_string()));
        }
        PLAIN.iter().find(|(n, _)| *n == s).map(|(_, f)| f())
    }

    /// 2026-10-09: The contract spelling.
    pub fn name(&self) -> String {
        match self {
            Mutation::Accumulate(e) => format!("accumulate:{}", e.name),
            Mutation::Symbol(k) => format!("symbol:{k}"),
            other => PLAIN
                .iter()
                .find(|(_, f)| &f() == other)
                .map_or("?".to_string(), |(n, _)| (*n).to_string()),
        }
    }

    /// 2026-10-09: The arm runs on a CPU emulation, not on the kernel (it proves the contract's
    /// sensitivity; the record labels it).
    pub fn emulated(&self) -> bool {
        matches!(self, Mutation::Accumulate(_))
    }
}

#[cfg(test)]
#[path = "mutation_tests.rs"]
mod tests;
