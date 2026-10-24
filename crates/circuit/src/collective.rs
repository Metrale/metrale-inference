// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The path a planned all-reduce takes, resolved once by the fuser from the rule that
//! names its kernel ([`crate::rules::Rule::path`] onto [`crate::fuser::Group::path`]), so the
//! engine executes the plan's choice instead of re-deciding it per call (PLEX). The communicator
//! (`metrale-comm`) executes exactly a [`CollectivePath`]; the boot compares each planned group
//! with the communicator's own resolution once.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - A one-shot rule is taken only when its payload fits the bound the policy states
//!   ([`ONESHOT_MAX_KB`], KiB): `0 < bytes <= kb * 1024`, the communicator's own test
//!   (`metrale_comm::ar_path::oneshot_applies`). A plan that would read the bound and finds it
//!   unstated is refused ([`crate::FuseError::PolicyMissing`]), never given a default.
//! - The payload is the collective node's output: the plan's rows times its width, in its format
//!   ([`payload_bytes`]), the bytes every rank passes the communicator.

use crate::prelude::*;

use crate::ir::{Circuit, Node};

/// 2026-10-23: The policy setting that states the one-shot all-reduce's payload bound in KiB
/// (`--all-reduce-oneshot-max-kb`; `0` disables the one-shot). The instance states the recipe's
/// value; a serve overwrites it once at boot from its communicator.
pub const ONESHOT_MAX_KB: &str = "all_reduce_oneshot_max_kb";

/// 2026-10-23: How an all-reduce runs on the communicator.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum CollectivePath {
    /// 2026-10-23: The 2-rank exchange plus `bf16_add_inplace` (`__hadd`).
    TwoRank,
    /// 2026-10-23: The one-shot exchange plus `bf16_add_rank_sum`.
    OneShot,
    /// 2026-10-23: The library's all-reduce (`ncclAllReduce`) in place.
    Nccl,
}

impl CollectivePath {
    /// 2026-10-23: Every path.
    pub const ALL: [CollectivePath; 3] = [
        CollectivePath::TwoRank,
        CollectivePath::OneShot,
        CollectivePath::Nccl,
    ];

    /// 2026-10-23: The spelling in FUSIONS.toml and plan renders.
    pub fn name(self) -> &'static str {
        match self {
            CollectivePath::TwoRank => "two_rank",
            CollectivePath::OneShot => "one_shot",
            CollectivePath::Nccl => "nccl",
        }
    }

    /// 2026-10-23: Parse [`CollectivePath::name`].
    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.name() == s)
    }
}

impl core::fmt::Display for CollectivePath {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

/// 2026-10-23: Bytes of collective node `n`'s payload at `rows` plan rows: its first output's row
/// expression at `n = rows` times its width, in its format. `None` when the node has no output,
/// its rows read an unbound dim, or the size overflows.
pub fn payload_bytes(c: &Circuit, n: &Node, rows: u64) -> Option<u64> {
    let e = &c.edges[*n.outputs.first()?];
    let mut dims = c.dims.clone();
    dims.insert("n".into(), rows);
    e.format.bytes(e.rows.eval(&dims).ok()?, e.dim_value)
}

/// 2026-10-23: The one-shot bound in bytes the policy states, `Ok(None)` when it states none,
/// `Err(value)` for a value that is not a whole number of KiB.
pub fn oneshot_max_bytes(policy: &crate::Policy) -> Result<Option<u64>, String> {
    let Some(v) = policy.settings.get(ONESHOT_MAX_KB) else {
        return Ok(None);
    };
    v.parse::<u64>()
        .ok()
        .and_then(|kb| kb.checked_mul(1024))
        .map(Some)
        .ok_or_else(|| v.clone())
}

/// 2026-10-23: Whether a payload of `bytes` takes the one-shot path under bound `max` (bytes):
/// the communicator's test, `0 < bytes <= max`.
pub fn oneshot_fits(bytes: u64, max: u64) -> bool {
    bytes > 0 && bytes <= max
}

impl crate::rules::Rule {
    /// 2026-10-23: The policy settings the rule reads: its `when` keys, and for a `one_shot`
    /// rule the payload bound ([`ONESHOT_MAX_KB`]).
    pub fn settings_read(&self) -> impl Iterator<Item = &str> {
        let bound = (self.path == Some(CollectivePath::OneShot)).then_some(ONESHOT_MAX_KB);
        self.when.keys().map(String::as_str).chain(bound)
    }
}

/// 2026-10-23: Set every group's path of `plan` from its rule in `rules` (the rule set the plan
/// was fused from, as its digest names it): what [`crate::fuse`] sets, for a plan decoded without
/// it. A group whose rule is not in `rules` is refused naming it (CERF `CircuitSource`).
pub fn derive_paths(
    plan: &mut crate::FusionPlan,
    rules: &[crate::rules::Rule],
) -> Result<(), metrale_error::Error> {
    for g in &mut plan.groups {
        let rule = rules.iter().find(|r| r.id == g.rule).ok_or_else(|| {
            metrale_error::error!(
                metrale_error::ErrorCode::CircuitSource,
                "FUSIONS.toml",
                alloc::format!("plan group rule `{}` is not in the rule set", g.rule)
            )
        })?;
        g.path = rule.path;
    }
    Ok(())
}
