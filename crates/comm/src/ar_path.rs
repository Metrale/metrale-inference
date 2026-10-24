// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: Which path an all-reduce takes on `NcclBackend`, as a pure function of the
//! communicator's state, so the choice is tested without a communicator.
//!
//! 2026-10-23: Public and outside the NCCL gate (pure), so the circuit's collective rules are
//! checked against it (`tp_all_reduce_*`, model-arch `world_tests`). The path type is the plan's
//! (`metrale_circuit::CollectivePath`): a planned collective names its path and the backend
//! executes exactly that one (`CommBackend::all_reduce_via`), refusing what it cannot run
//! ([`runnable`]). [`ar_path`] remains the decision of the callers that run no plan, and the
//! boot's check of each planned group (model-layers `circuit_exec::collectives`); it goes when
//! the last such caller is plan-executed.
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
//! - [`runnable`] refuses a path the state cannot run (CERF `CollectivePathRefused`); it never
//!   names another path.

use metrale_circuit::CollectivePath;
use metrale_error::{Error, ErrorCode, error};

/// 2026-10-23: The communicator state a path depends on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PathState {
    pub world_size: usize,
    /// 2026-10-23: `bf16_add_inplace` set (`set_add_kernel`).
    pub add_kernel_set: bool,
    /// 2026-10-23: `bf16_add_rank_sum` set (`set_rank_sum_kernel`).
    pub sum_kernel_set: bool,
    /// 2026-10-23: The one-shot bound in bytes (`--all-reduce-oneshot-max-kb` x 1024; 0: off).
    pub oneshot_max_bytes: usize,
    /// 2026-10-23: The 2-rank receive buffer's capacity in bytes.
    pub recv_capacity: usize,
}

/// 2026-10-08: Whether a payload of `bytes` takes the one-shot path. The payload test is the
/// fuser's (`metrale_circuit::collective::oneshot_fits`), so a plan and the communicator agree.
pub fn oneshot_applies(world_size: usize, max_bytes: usize, bytes: usize) -> bool {
    world_size >= 2 && metrale_circuit::collective::oneshot_fits(bytes as u64, max_bytes as u64)
}

/// 2026-10-16: Choose the path (see the module invariants). `oneshot_applies` is whether the
/// payload fits an enabled one-shot bound on this communicator.
pub fn ar_path(
    world_size: usize,
    add_kernel_set: bool,
    oneshot_applies: bool,
    sum_kernel_set: bool,
) -> CollectivePath {
    if world_size == 2 && add_kernel_set {
        CollectivePath::TwoRank
    } else if oneshot_applies && (world_size >= 3 || sum_kernel_set) {
        CollectivePath::OneShot
    } else {
        CollectivePath::Nccl
    }
}

/// 2026-10-23: [`ar_path`] of a payload of `bytes` in state `s`.
pub fn ar_path_of(s: &PathState, bytes: usize) -> CollectivePath {
    ar_path(
        s.world_size,
        s.add_kernel_set,
        oneshot_applies(s.world_size, s.oneshot_max_bytes, bytes),
        s.sum_kernel_set,
    )
}

/// 2026-10-23: Whether state `s` can run `path` for `bytes`: the 2-rank exchange at two ranks
/// with its add kernel and the payload within its receive buffer; the one-shot with its sum
/// kernel and the payload within its bound; NCCL always. A refusal is CERF
/// `CollectivePathRefused`, naming what is missing.
pub fn runnable(path: CollectivePath, s: &PathState, bytes: usize) -> Result<(), Error> {
    let why = match path {
        CollectivePath::TwoRank if s.world_size != 2 => {
            format!("the 2-rank exchange needs two ranks, this communicator has {}", s.world_size)
        }
        CollectivePath::TwoRank if !s.add_kernel_set => "bf16_add_inplace is not set".into(),
        CollectivePath::TwoRank if bytes > s.recv_capacity => {
            format!("the 2-rank receive buffer holds {} bytes", s.recv_capacity)
        }
        CollectivePath::OneShot if !s.sum_kernel_set => "bf16_add_rank_sum is not set".into(),
        CollectivePath::OneShot if !oneshot_applies(s.world_size, s.oneshot_max_bytes, bytes) => {
            format!(
                "the one-shot takes 1..={} bytes at {} ranks",
                s.oneshot_max_bytes, s.world_size
            )
        }
        _ => return Ok(()),
    };
    Err(error!(ErrorCode::CollectivePathRefused, path.name(), bytes, why))
}

/// 2026-10-23: Whether state `s` has the one-shot exchange alone for `bytes` (what
/// `CommBackend::exchange_partials` sends, a fused group's consumer summing the partials): the
/// payload within the one-shot bound. A refusal is CERF `CollectivePathRefused`.
pub fn exchange_runnable(s: &PathState, bytes: usize) -> Result<(), Error> {
    if oneshot_applies(s.world_size, s.oneshot_max_bytes, bytes) {
        return Ok(());
    }
    let why = format!(
        "the one-shot exchange takes 1..={} bytes at {} ranks (--all-reduce-oneshot-max-kb)",
        s.oneshot_max_bytes, s.world_size
    );
    Err(error!(ErrorCode::CollectivePathRefused, CollectivePath::OneShot.name(), bytes, why))
}

#[cfg(test)]
#[path = "ar_path_tests.rs"]
mod tests;
