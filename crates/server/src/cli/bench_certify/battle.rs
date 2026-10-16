// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: `met benchmark certify --gates <battle>`: a battle id (`bench/battles.toml`)
//! runs the battle's member gates on this box's hardware whose cell subjects are not
//! current at the anchor, beside any required gates named, and the battle's table is printed
//! at the end. A battle never decides `certified`.
//!
//! Owner: server CLI (`met benchmark certify`).
//! Invariants:
//! - `--gates` holding only battle ids runs no required gate; an empty `--gates` still means
//!   every open required gate.
//! - A member is current when the subject of each of its cells on this hardware is `Pass`
//!   in `check_models`, the kernel-aware per-model check: a change that reaches the model's
//!   kernels makes the battle owed.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Result, bail};
use metrale_bench::gate::battle::{self, Battle, Cell};
use metrale_bench::gate::check_models::{ModelStanding, SubjectState, check_models};
use metrale_bench::gate::kernel_map::{FsRepo, load};
use metrale_bench::gate::{self, GateStatus, battle_load};

use super::Emit;

/// 2026-10-16: The battles a campaign was asked for and the member gates it adds.
#[derive(Default)]
pub struct Plan {
    battles: Vec<(Battle, Vec<Cell>)>,
    hardware: String,
    members: Vec<&'static str>,
}

/// 2026-10-16: Move the battle ids out of `gates` and return them, in order. The battle file
/// is read only when `gates` names something that is not a required gate.
pub fn take(root: &Path, gates: &mut Vec<String>) -> Result<Vec<String>> {
    if gates
        .iter()
        .all(|g| gate::REQUIRED_GATES.contains(&g.as_str()))
    {
        return Ok(Vec::new());
    }
    let ids: Vec<String> = battle_load::battles(root)?
        .into_iter()
        .map(|b| b.id)
        .collect();
    Ok(split(gates, &ids))
}

/// 2026-10-16: The battle ids in `gates`, removed from it.
pub fn split(gates: &mut Vec<String>, battle_ids: &[String]) -> Vec<String> {
    let (battles, rest): (Vec<String>, Vec<String>) =
        gates.drain(..).partition(|g| battle_ids.contains(g));
    *gates = rest;
    battles
}

/// 2026-10-16: The required gates to run: none when `--gates` named only battles.
pub fn remaining(
    statuses: &BTreeMap<String, GateStatus>,
    gates: &[String],
    battles: &[String],
) -> Result<Vec<&'static str>> {
    if gates.is_empty() && !battles.is_empty() {
        return Ok(Vec::new());
    }
    super::plan::remaining(statuses, gates)
}

/// 2026-10-16: Whether every model on `cell`'s gate that is `cell`'s subject passes there.
pub fn current(standings: &BTreeMap<String, Vec<ModelStanding>>, cell: &Cell) -> bool {
    standings.get(&cell.gate).is_some_and(|list| {
        list.iter().any(|m| {
            m.subject.hw == cell.hw
                && m.subject.checkpoint == cell.checkpoint
                && m.state == SubjectState::Pass
        })
    })
}

impl Plan {
    pub fn new(root: &Path, anchor: &str, hardware: &str, wanted: &[String]) -> Result<Self> {
        if wanted.is_empty() {
            return Ok(Self::default());
        }
        let battles: Vec<(Battle, Vec<Cell>)> = battle_load::load(root)?
            .into_iter()
            .filter(|(b, _)| wanted.contains(&b.id))
            .collect();
        let standings =
            check_models(root, anchor, anchor, &load(&FsRepo { root: root.into() })).standings;
        let mut members = Vec::new();
        for (b, cells) in &battles {
            let open = battle::open_members(cells, hardware, &|c| current(&standings, c));
            for id in open {
                let Some(d) = metrale_bench::registry::find(&id) else {
                    bail!(
                        "{id}, a member of battle {}, is not a registered benchmark",
                        b.id
                    );
                };
                if !members.contains(&d.id) {
                    members.push(d.id);
                }
            }
        }
        Ok(Self {
            battles,
            hardware: hardware.to_string(),
            members,
        })
    }

    /// 2026-10-16: `gates` followed by the members it does not already hold.
    pub fn with_members(&self, mut gates: Vec<&'static str>) -> Vec<&'static str> {
        for m in &self.members {
            if !gates.contains(m) {
                gates.push(m);
            }
        }
        gates
    }

    /// 2026-10-16: Each battle's table on this hardware, from the records now in the tree.
    pub fn report(&self, emit: &Emit, root: &Path) -> Result<()> {
        for (b, cells) in &self.battles {
            let here: Vec<Cell> = cells
                .iter()
                .filter(|c| c.hw == self.hardware)
                .cloned()
                .collect();
            let r = battle::report(b, &here, &|c| battle_load::newest(root, c));
            emit.event("battle", serde_json::to_value(&r)?);
            if !emit.json {
                print!("{}", super::super::bench_battle::table(&r));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "battle_tests.rs"]
mod battle_tests;
