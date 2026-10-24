// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: A tensor-parallel rank's planned collectives against its communicator, at boot
//! (`impl_circuit_ranks`): the one-shot payload bound the plans fuse under is the
//! communicator's, resolved once (ROTI), and every planned collective group's path
//! (`metrale_circuit::Group::path`) must be the one the communicator takes for its payload, or
//! the boot is refused naming the group (CERF `CollectivePathMismatch`). The communicator
//! executes a planned group's path as named (`CommBackend::all_reduce_via`); this check is the
//! one place both sides meet, so neither re-decides per step.
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - The bound is overwritten only where the instance states it; an instance that states none
//!   keeps none, and a rule that reads it is refused by the fuser (PCND).
//! - Every group with a path is checked, the plain collectives and the fused exchanges (the mHC
//!   post-mix's, whose bound a serve without `--all-reduce-oneshot-max-kb` fails at the first
//!   step otherwise); a payload the circuit cannot size is a mismatch, never skipped.

use metrale_circuit::collective::{ONESHOT_MAX_KB, payload_bytes};
use metrale_circuit::{Circuit, CollectivePath, FusionPlan, Policy};
use metrale_comm::ar_path::{PathState, ar_path_of, exchange_runnable};
use metrale_error::{Error, ErrorCode, error};

/// 2026-10-23: `policy` with the one-shot bound the instance states replaced by the
/// communicator's (`oneshot_max_bytes`, `CommBackend::oneshot_max_bytes`), logged when they
/// differ. A bound that is not whole KiB is refused: the setting is in KiB, and a rounded one
/// would plan payloads the communicator sends elsewhere.
pub fn with_comm_bound(policy: &Policy, oneshot_max_bytes: usize) -> Result<Policy, Error> {
    let mut p = policy.clone();
    let Some(stated) = p.settings.get_mut(ONESHOT_MAX_KB) else {
        return Ok(p);
    };
    if !oneshot_max_bytes.is_multiple_of(1024) {
        return Err(error!(
            ErrorCode::CollectivePathRefused,
            CollectivePath::OneShot.name(),
            oneshot_max_bytes,
            format!("the plans state the one-shot bound in KiB (`{ONESHOT_MAX_KB}`)")
        ));
    }
    let live = (oneshot_max_bytes / 1024).to_string();
    if *stated != live {
        tracing::info!(
            "circuit: the instance states {ONESHOT_MAX_KB} = {stated}; this communicator's is \
             {live}, which the plans fuse under"
        );
        *stated = live;
    }
    Ok(p)
}

/// 2026-10-23: Check every collective group of `plans` (over rank circuit `circuit`) against
/// the communicator's state `comm` (`CommBackend::path_state`): a plain collective (one node)
/// must name the path the communicator takes for its payload (`ar_path`, else
/// `CollectivePathMismatch`); a fused exchange (a collective headed group whose own kernel sums,
/// `glm_ar_fused_post`) needs the one-shot exchange for its payload (`exchange_runnable`, else
/// `CollectivePathRefused`). Returns how many groups were checked.
pub fn check_paths(
    circuit: &Circuit,
    plans: &[FusionPlan],
    comm: &PathState,
) -> Result<usize, Error> {
    let mut checked = 0;
    for plan in plans {
        for (i, g) in plan.groups.iter().enumerate() {
            let Some(planned) = g.path else { continue };
            let node = &circuit.nodes[g.nodes[0]];
            let group = format!(
                "{} n={} g{i:04} `{}` (rule `{}`)",
                plan.mode.name(),
                plan.rows,
                node.id,
                g.rule
            );
            let bytes = payload_bytes(circuit, node, plan.rows)
                .and_then(|b| usize::try_from(b).ok());
            let Some(bytes) = bytes else {
                return Err(error!(
                    ErrorCode::CollectivePathMismatch,
                    group,
                    "unsized",
                    planned.name(),
                    "no path"
                ));
            };
            if g.nodes.len() > 1 {
                exchange_runnable(comm, bytes).map_err(|e| {
                    error!(
                        ErrorCode::CollectivePathRefused,
                        planned.name(),
                        bytes,
                        format!("{group} exchanges its partials for its own sum: {e}")
                    )
                })?;
            } else {
                let live = ar_path_of(comm, bytes);
                if live != planned {
                    return Err(error!(
                        ErrorCode::CollectivePathMismatch,
                        group,
                        bytes,
                        planned.name(),
                        live.name()
                    ));
                }
            }
            checked += 1;
        }
    }
    Ok(checked)
}

#[cfg(test)]
#[path = "collectives_tests.rs"]
mod tests;
