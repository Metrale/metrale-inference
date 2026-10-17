// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: Which path an all-reduce takes on `NcclBackend`, as a pure function of the
//! communicator's state, so the choice is tested without a communicator.
//!
//! Owner: metrale-comm.
//! Invariants:
//! - At two ranks the 2-rank exchange (`bf16_add_inplace`) keeps priority whenever its kernel is
//!   set, so a plain all-reduce's bytes there do not move.
//! - At two ranks the one-shot is taken only with the rank-sum kernel set; without it the path is
//!   `ncclAllReduce`, which is what two ranks ran without the add kernel before the one-shot was
//!   offered there.
//! - At three or more ranks the one-shot is taken whenever it applies; a missing rank-sum kernel
//!   there fails the first collective (never a silent NCCL fallback), as before.

/// 2026-10-16: The path an all-reduce takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ArPath {
    /// 2026-10-17: The 2-rank exchange plus `bf16_add_inplace` (`__hadd`).
    TwoRank,
    /// 2026-10-17: The one-shot exchange plus `bf16_add_rank_sum`.
    OneShot,
    /// 2026-10-17: `ncclAllReduce` in place.
    Nccl,
}

/// 2026-10-16: Choose the path (see the module invariants). `oneshot_applies` is whether the
/// payload fits an enabled one-shot bound on this communicator.
pub(super) fn ar_path(
    world_size: usize,
    add_kernel_set: bool,
    oneshot_applies: bool,
    sum_kernel_set: bool,
) -> ArPath {
    if world_size == 2 && add_kernel_set {
        ArPath::TwoRank
    } else if oneshot_applies && (world_size >= 3 || sum_kernel_set) {
        ArPath::OneShot
    } else {
        ArPath::Nccl
    }
}

#[cfg(test)]
mod tests {
    use super::{ArPath, ar_path};

    /// 2026-10-16: Every combination of the four inputs at two and three ranks, against the table
    /// written out by hand. The rows that changed with the world-2 one-shot are the `(2, false,
    /// true, _)` ones: the kernel decides between the one-shot and NCCL.
    #[test]
    fn every_state_takes_the_tabled_path() {
        use ArPath::{Nccl, OneShot, TwoRank};
        // 2026-10-17: (world, add kernel, one-shot applies, rank-sum kernel) -> path
        let table = [
            ((2, true, true, true), TwoRank),
            ((2, true, true, false), TwoRank),
            ((2, true, false, true), TwoRank),
            ((2, true, false, false), TwoRank),
            ((2, false, true, true), OneShot),
            ((2, false, true, false), Nccl),
            ((2, false, false, true), Nccl),
            ((2, false, false, false), Nccl),
            ((3, true, true, true), OneShot),
            ((3, true, true, false), OneShot),
            ((3, true, false, true), Nccl),
            ((3, true, false, false), Nccl),
            ((3, false, true, true), OneShot),
            ((3, false, true, false), OneShot),
            ((3, false, false, true), Nccl),
            ((3, false, false, false), Nccl),
        ];
        for ((w, add, applies, sum), want) in table {
            assert_eq!(
                ar_path(w, add, applies, sum),
                want,
                "{:?}",
                (w, add, applies, sum)
            );
        }
    }

    /// 2026-10-16: Positive control: the two-rank rule differs from the three-rank one only in the
    /// kernel-less one-shot row, so a rule that ignored the world size would be caught.
    #[test]
    fn only_the_kernel_less_oneshot_row_depends_on_the_world() {
        let mut differ = Vec::new();
        for add in [false, true] {
            for applies in [false, true] {
                for sum in [false, true] {
                    let (two, eight) =
                        (ar_path(2, add, applies, sum), ar_path(8, add, applies, sum));
                    if two != eight {
                        differ.push((add, applies, sum, two, eight));
                    }
                }
            }
        }
        assert!(differ.contains(&(false, true, false, ArPath::Nccl, ArPath::OneShot)));
        assert!(
            differ
                .iter()
                .all(|&(add, applies, sum, ..)| add || (applies && !sum))
        );
    }
}
