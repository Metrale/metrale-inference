// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: The path table, and (2026-10-23) what a planned path refuses.

use super::{PathState, ar_path, ar_path_of, runnable};
use metrale_circuit::CollectivePath::{self, Nccl, OneShot, TwoRank};
use metrale_error::ErrorCode;

/// 2026-10-16: Every combination of the four inputs at two and three ranks, against the table
/// written out by hand. The rows that changed with the world-2 one-shot are the `(2, false,
/// true, _)` ones: the kernel decides between the one-shot and NCCL.
#[test]
fn every_state_takes_the_tabled_path() {
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
                let (two, eight) = (ar_path(2, add, applies, sum), ar_path(8, add, applies, sum));
                if two != eight {
                    differ.push((add, applies, sum, two, eight));
                }
            }
        }
    }
    assert!(differ.contains(&(false, true, false, Nccl, OneShot)));
    assert!(
        differ
            .iter()
            .all(|&(add, applies, sum, ..)| add || (applies && !sum))
    );
}

/// 2026-10-23: A served communicator: both kernels set, a 1024 KiB one-shot bound, an 8 MiB
/// 2-rank receive buffer (the 2-rank exchange refuses past it on either path choice).
fn served(world_size: usize) -> PathState {
    PathState {
        world_size,
        add_kernel_set: true,
        sum_kernel_set: true,
        oneshot_max_bytes: 1024 * 1024,
        recv_capacity: 8 << 20,
    }
}

fn refusal(path: CollectivePath, s: &PathState, bytes: usize) -> String {
    let e = runnable(path, s, bytes).expect_err("refused");
    assert_eq!(e.code(), ErrorCode::CollectivePathRefused);
    e.to_string()
}

/// 2026-10-23: The path the communicator itself takes is runnable (positive), at every world and
/// on both sides of the one-shot bound, for payloads the 2-rank buffer holds.
#[test]
fn the_communicators_own_path_is_runnable() {
    for w in [2, 3, 4, 8] {
        let s = served(w);
        for bytes in [2, 8192, 1024 * 1024, 1024 * 1024 + 2, 4 << 20] {
            let p = ar_path_of(&s, bytes);
            runnable(p, &s, bytes).unwrap_or_else(|e| panic!("world {w} {bytes} {p}: {e}"));
        }
    }
}

/// 2026-10-23: A path the state cannot run is refused with what is missing, never run as
/// another (negative): the 2-rank exchange at three ranks or without its kernel, the one-shot
/// past its bound, disabled, or without its kernel. NCCL runs anywhere.
#[test]
fn a_path_the_state_cannot_run_is_refused() {
    let at3 = served(3);
    assert!(refusal(TwoRank, &at3, 8192).contains("needs two ranks"));
    let no_add = PathState {
        add_kernel_set: false,
        ..served(2)
    };
    assert!(refusal(TwoRank, &no_add, 8192).contains("bf16_add_inplace"));
    assert!(refusal(TwoRank, &served(2), 16 << 20).contains("receive buffer"));
    assert!(refusal(OneShot, &at3, 1024 * 1024 + 2).contains("1..=1048576 bytes"));
    let off = PathState {
        oneshot_max_bytes: 0,
        ..served(3)
    };
    assert!(refusal(OneShot, &off, 8192).contains("1..=0 bytes"));
    let no_sum = PathState {
        sum_kernel_set: false,
        ..served(3)
    };
    assert!(refusal(OneShot, &no_sum, 8192).contains("bf16_add_rank_sum"));
    for s in [at3, no_add, off, no_sum] {
        runnable(Nccl, &s, 4 << 20).unwrap();
    }
}
