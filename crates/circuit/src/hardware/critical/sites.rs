// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: Which headroom site each launch group of an event capture belongs to: the
//! headroom's own attribution ([`attribute`]) run over the capture's groups, so the what-if and
//! the headroom report agree on every kernel's site.
//!
//! Owner: metrale-circuit (hardware).
//! Invariants: one attribution rule (headroom's); a group it refuses fails the whole call.

use crate::prelude::*;

use super::capture::Group;
use crate::hardware::headroom_sites::{LaunchSite, attribute};
use crate::hardware::headroom_trace::TraceKernel;
use crate::hardware::plan::Resolved;

/// 2026-10-23: Per group, the `(run, site label)` of the launch site it is attributed to.
pub fn group_sites(
    r: &Resolved,
    sites: &[LaunchSite],
    groups: &[Group],
) -> Result<Vec<Option<(String, String)>>, Vec<String>> {
    let rows: Vec<TraceKernel> = groups
        .iter()
        .map(|g| TraceKernel {
            kernel: g.0.clone(),
            site: g.1.clone(),
            label: g.2.clone(),
            note: g.3.clone(),
            calls: 0,
            total_us: 0.0,
            exclusive_us: 0.0,
            energy_uj: None,
            step_total_us: Vec::new(),
            step_exclusive_us: Vec::new(),
        })
        .collect();
    let a = attribute(r, sites, &rows)?;
    let mut of = vec![None; groups.len()];
    for (&i, members) in &a.measured {
        let s = &sites[i];
        let name = (
            format!("{} n={}", s.run.mode.name(), s.run.rows),
            s.label.clone(),
        );
        for (k, _) in members {
            for (gi, row) in rows.iter().enumerate() {
                if row == k {
                    of[gi] = Some(name.clone());
                }
            }
        }
    }
    Ok(of)
}
