// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The derive: a [`Config`] and an already-read [`SourceSet`] in, a sealed [`Bundle`]
//! out. Pure (SBIO): the same inputs give the same bundle digest, whatever order the host read the
//! files in (design §4.1 invariant 2).
//!
//! Owner: metrale-kg-derive.
//! Invariants:
//! - A configured source that is absent from the set is an error (`KgSource`), never skipped.
//! - Adapters run in a fixed order so cross-references (principle ids, ledger ids) are known
//!   before the adapters that scan text for them.

use crate::bundle::{Builder, Bundle};
use crate::config::Config;
use crate::prelude::*;
use crate::sources::SourceSet;
use crate::{decisions, ledger_adapter, memory, principles, research, tooling};
use metrale_error::{ErrorCode, Result, error};

fn need<'a>(src: &'a SourceSet, path: &str) -> Result<&'a str> {
    src.get(path).ok_or_else(|| {
        error!(
            ErrorCode::KgSource,
            path.to_string(),
            "configured but not read"
        )
    })
}

/// 2026-10-23: Derive the campaign-record bundle (P0 sources).
pub fn derive(cfg: &Config, src: &SourceSet) -> Result<Bundle> {
    let mut b = Builder::default();
    let s = &cfg.sources;
    let mut texts = Vec::new();
    for p in &s.principles {
        texts.push((p.as_str(), need(src, p)?));
    }
    let known = principles::derive(&mut b, &texts, &texts);
    let mut ledger_ids = BTreeSet::new();
    for c in &s.campaigns {
        let ids = ledger_adapter::derive(&mut b, &c.id, &c.ledger, need(src, &c.ledger)?, &known);
        ledger_ids.extend(ids);
    }
    for c in &s.campaigns {
        decisions::charter(&mut b, &c.charter, need(src, &c.charter)?, &ledger_ids);
        tooling::derive(&mut b, &c.id, &c.tooling, need(src, &c.tooling)?, &known);
    }
    for d in &s.designs {
        decisions::design(&mut b, d, need(src, d)?, &ledger_ids);
    }
    research::graph_theory_map(&mut b, &s.graph_theory_map, need(src, &s.graph_theory_map)?);
    research::mathbomb(&mut b, &s.mathbomb, need(src, &s.mathbomb)?);
    memory::derive(&mut b, src.under(&s.memory_dir));
    if !cfg.origin.is_empty() {
        crate::settings::project_node(&mut b, &cfg.origin, need(src, &cfg.origin)?, &cfg.embedder);
    }
    Ok(b.seal(src.digests()))
}
