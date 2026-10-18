// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The lever groups, one file each, and the flat table derived from them.
//!
//! Owner: settings.
//! Invariants: `GROUPS` and `RULE_GROUPS` list the same groups in the same order, and each
//! group's `RULES` is parallel to its `LEVERS`.

use alloc::vec::Vec;

use crate::spec::{LeverSpec, Rule};

pub mod build;
pub mod cache;
pub mod dev;
pub mod device;
pub mod device_cache;
pub mod device_drafter;
pub mod device_weights;
pub mod harness;
pub mod kernels;
pub mod model;
pub mod runtime;
pub mod sampling;
pub mod scheduler;
pub mod server;
pub mod speculative;
pub mod storage;
pub mod telemetry;
pub mod tool;

#[cfg(any(test, feature = "test-utils"))]
pub mod equivalence;

/// 2026-10-17: Group order is alphabetical by group name; `by_name` restores the table order.
pub const GROUPS: &[&[LeverSpec]] = &[
    build::BuildLevers::LEVERS,
    cache::CacheSettings::LEVERS,
    dev::DevLevers::LEVERS,
    harness::HarnessSettings::LEVERS,
    kernels::KernelsSettings::LEVERS,
    model::ModelSettings::LEVERS,
    runtime::RuntimeSettings::LEVERS,
    sampling::SamplingSettings::LEVERS,
    scheduler::SchedulerSettings::LEVERS,
    server::ServerSettings::LEVERS,
    speculative::SpeculativeSettings::LEVERS,
    storage::StorageSettings::LEVERS,
    telemetry::TelemetrySettings::LEVERS,
    tool::ToolLevers::LEVERS,
];

/// 2026-10-17: Each group's `RULES`, in `GROUPS` order.
pub const RULE_GROUPS: &[&[Rule]] = &[
    build::BuildLevers::RULES,
    cache::CacheSettings::RULES,
    dev::DevLevers::RULES,
    harness::HarnessSettings::RULES,
    kernels::KernelsSettings::RULES,
    model::ModelSettings::RULES,
    runtime::RuntimeSettings::RULES,
    sampling::SamplingSettings::RULES,
    scheduler::SchedulerSettings::RULES,
    server::ServerSettings::RULES,
    speculative::SpeculativeSettings::RULES,
    storage::StorageSettings::RULES,
    telemetry::TelemetrySettings::RULES,
    tool::ToolLevers::RULES,
];

/// Every lever, sorted by `env` (byte order == the old table order).
#[must_use]
pub fn by_name() -> Vec<&'static LeverSpec> {
    let mut all: Vec<&'static LeverSpec> = GROUPS.iter().flat_map(|g| g.iter()).collect();
    all.sort_unstable_by(|a, b| a.env.cmp(b.env));
    all
}

#[must_use]
pub fn lookup(env: &str) -> Option<&'static LeverSpec> {
    GROUPS.iter().flat_map(|g| g.iter()).find(|l| l.env == env)
}

/// Rule of a declared lever (Presence vs Switch), for tests and dump tooling.
#[must_use]
pub fn rule_of(env: &str) -> Option<Rule> {
    GROUPS
        .iter()
        .zip(RULE_GROUPS)
        .flat_map(|(levers, rules)| levers.iter().zip(rules.iter()))
        .find(|(l, _)| l.env == env)
        .map(|(_, &r)| r)
}
