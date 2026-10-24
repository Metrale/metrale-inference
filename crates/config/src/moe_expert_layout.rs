// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: `--moe-expert-layout`: how a routed MoE's experts are laid out over the ranks.
//!
//! - `ep`: each expert is owned whole by one expert-parallel rank (`local_expert_range`), the
//!   layout every MoE loader has always used.
//! - `tp`: every rank holds a tensor-parallel slice of EVERY expert's intermediate width, as the
//!   shared expert is split, so all `top_k` routed slots of a token are local on every rank and
//!   each rank reads the same bytes per token. The MoE output stays a partial sum, reduced by
//!   the all-reduce that follows the MoE site under TP or EP.
//!
//! Owner: config.
//! Invariants:
//! - [`MoeExpertLayout::check_topology`] passes `ep` at every topology, so the `ep` layout's
//!   startup is unchanged.
//! - `tp` passes only with `tp_world_size >= 2` and `ep_world_size == 1`: the experts are not
//!   EP-partitioned under it, and a one-rank TP slice is the whole expert.

use anyhow::{Result, bail};

/// 2026-10-09: The `--moe-expert-layout` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MoeExpertLayout {
    /// 2026-10-09: Whole experts partitioned over EP ranks. `Default` only because
    /// `ModelConfig` is deserialized with serve-set fields skipped; serve sets it from the flag.
    #[default]
    Ep,
    /// 2026-10-09: Every expert's intermediate width split over TP ranks.
    Tp,
}

impl MoeExpertLayout {
    /// 2026-10-09: Every layout, in the order the flag lists them.
    pub const ALL: [Self; 2] = [Self::Ep, Self::Tp];

    /// 2026-10-09: The flag and recipe value.
    pub fn name(self) -> &'static str {
        match self {
            Self::Ep => "ep",
            Self::Tp => "tp",
        }
    }

    /// 2026-10-09: Refuse a topology the layout cannot run on (see the module invariants).
    pub fn check_topology(self, tp_world_size: usize, ep_world_size: usize) -> Result<()> {
        if self == Self::Ep {
            return Ok(());
        }
        if ep_world_size > 1 {
            bail!(
                "--moe-expert-layout tp slices every expert over the TP ranks, so the experts are \
                 not expert-parallel: run --ep-size 1 (got --ep-size {ep_world_size}) with \
                 --tp-size equal to the rank count"
            );
        }
        if tp_world_size < 2 {
            bail!(
                "--moe-expert-layout tp needs --tp-size 2 or more (got {tp_world_size}); at one \
                 rank there is nothing to slice, use --moe-expert-layout ep"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-09: `ep` passes every topology; `tp` passes only TP >= 2 at EP 1.
    #[test]
    fn tp_needs_tp_ranks_and_no_expert_parallelism() {
        for (tp, ep) in [(1, 1), (3, 3), (2, 2), (1, 3), (3, 1)] {
            assert!(MoeExpertLayout::Ep.check_topology(tp, ep).is_ok());
        }
        assert!(MoeExpertLayout::Tp.check_topology(3, 1).is_ok());
        assert!(MoeExpertLayout::Tp.check_topology(2, 1).is_ok());
        let e = MoeExpertLayout::Tp.check_topology(3, 3).unwrap_err();
        assert!(e.to_string().contains("--ep-size 1"), "{e}");
        let e = MoeExpertLayout::Tp.check_topology(1, 1).unwrap_err();
        assert!(e.to_string().contains("--tp-size 2"), "{e}");
    }

    /// 2026-10-09: The names are the flag values and are distinct.
    #[test]
    fn names_round_trip() {
        let names: Vec<_> = MoeExpertLayout::ALL.iter().map(|l| l.name()).collect();
        assert_eq!(names, ["ep", "tp"]);
    }
}
