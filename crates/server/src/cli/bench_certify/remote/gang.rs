// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: Units that hold more than one box (`Unit::nodes`, from the recipe's
//! `min_nodes`): which node runs one, which nodes are held for it, and when a node waits.
//!
//! Owner: server CLI (`met benchmark certify`).
//! - A multi-node unit runs once, on the node that takes it (the head, which signs its
//!   record); `nodes - 1` idle nodes are held for it until it finishes and run nothing.
//! - Multi-node units go first: while one is pending and some live node may head it,
//!   a free node either starts it (enough idle peers) or waits. Without that, single
//!   units would keep a peer busy and the wide unit could wait to the end of the
//!   campaign. The cost is idle boxes until the head's and peers' current units end.
//! - A multi-node unit no live node may head, or wider than the live fleet, is left
//!   pending (the campaign reports it not run) and holds nothing.
//! - With no multi-node unit pending, this is [`next_for`] unchanged.
//! Invariants: none beyond the types.

use super::super::plan::Unit;
use super::super::runner::RunOutcome;
use super::schedule::{EnergyPin, SpeedMode, may_run, next_for};

/// 2026-10-15: A node as the scheduler sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    /// 2026-10-15: Running nothing and held by nothing: may take a unit or be held.
    Idle,
    /// 2026-10-15: Running a unit, or held as a peer of one.
    Busy,
    /// 2026-10-15: Its worker returned (retired, or nothing left it may run).
    Gone,
}

/// 2026-10-15: What a free node does next.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Take {
    /// 2026-10-15: Run `unit`, holding `peers` (empty for a one-box unit).
    Run { unit: usize, peers: Vec<usize> },
    /// 2026-10-15: This node is held as a peer of a running multi-node unit, or such a
    /// unit is pending but cannot start yet: take nothing.
    Wait,
}

/// 2026-10-15: The next step for `node`, given every node's `slots`. A node whose own slot
/// is `Busy` is held as a peer and waits. `None` means nothing it may run.
pub fn take(
    node: usize,
    units: &[Unit],
    pending: &[bool],
    placed: &[Option<usize>],
    mode: &SpeedMode,
    energy: Option<&EnergyPin>,
    slots: &[Slot],
) -> Option<Take> {
    if slots[node] == Slot::Busy {
        return Some(Take::Wait);
    }
    let live: Vec<usize> = (0..slots.len())
        .filter(|&k| k == node || slots[k] != Slot::Gone)
        .collect();
    let wide = (0..units.len())
        .filter(|&i| pending[i] && units[i].nodes > 1 && units[i].nodes <= live.len())
        .filter(|&i| live.iter().any(|&h| may_run(&units[i], h, mode, energy)))
        // 2026-10-15: Longest first, then the earlier unit, as `next_for` orders.
        .min_by_key(|&i| (std::cmp::Reverse(units[i].secs()), i));
    let Some(i) = wide else {
        return next_for(node, units, pending, placed, mode, energy).map(|unit| Take::Run {
            unit,
            peers: vec![],
        });
    };
    if !may_run(&units[i], node, mode, energy) {
        return Some(Take::Wait);
    }
    let peers: Vec<usize> = (0..slots.len())
        .filter(|&k| k != node && slots[k] == Slot::Idle)
        .take(units[i].nodes - 1)
        .collect();
    if peers.len() + 1 < units[i].nodes {
        return Some(Take::Wait);
    }
    Some(Take::Run { unit: i, peers })
}

/// 2026-10-15: Free a finished unit's head and the peers held for it. A node whose
/// worker returned meanwhile stays `Gone`.
pub fn release(slots: &mut [Slot], head: usize, peers: Vec<usize>) {
    for k in peers.into_iter().chain([head]) {
        if slots[k] == Slot::Busy {
            slots[k] = Slot::Idle;
        }
    }
}

/// 2026-10-15: The outcome of a multi-node unit, decided without submitting it: its
/// peers are held, but nothing launches their ranks over `--with-nodes` yet, and rank 0
/// alone would wait out its boot timeout. `None` for a one-box unit, which runs.
pub fn unlaunchable(unit: &Unit) -> Option<RunOutcome> {
    (unit.nodes > 1).then(|| RunOutcome::Harness {
        reason: format!(
            "{} holds {} nodes, but launching the peer ranks of a multi-node recipe over \
             --with-nodes is not built; nothing was submitted",
            unit.label(),
            unit.nodes
        ),
        retryable: false,
    })
}

#[cfg(test)]
#[path = "gang_tests.rs"]
mod gang_tests;
