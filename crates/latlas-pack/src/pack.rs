// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: `impl Pack for Metrale`: the vocabulary, the sources, the adapters in their fixed
//! order, the status sections and the ledger views `la_impact` adds for a ledger row.
//!
//! Owner: metrale-latlas-pack (the adapter order is cut from metrale-kg-derive `derive.rs`).
//! Invariants:
//! - A configured source absent from the set is `LatlasSource`, never skipped.
//! - Adapters run in a fixed order so cross-references (principle ids, ledger ids) are known
//!   before the adapters that scan text for them.
//! - The LKB facts are given exactly when `[sources.lkb]` is configured.

use crate::config::{CONFIG_SCHEMA, Config, Sources};
use crate::ledger::{self, render_deps};
use crate::lkb_derive::{self, LkbFacts, LkbSources};
use crate::prelude::*;
use crate::vocab::{Label, NodeId, PACK_LABELS, PACK_RELS, Rel, labels, rels};
use crate::{decisions, ledger_adapter, owed, queries, research, runs, tooling};
use latlas_adapters::{memory, principles};
use latlas_core::bundle::{Builder, Bundle};
use latlas_core::derive::need;
use latlas_core::id::Vocab;
use latlas_core::pack::{Pack, SectionCtx};
use latlas_core::sources::SourceSet;
use latlas_error::{ErrorCode, Result, error};
use serde_json::{Map, Value};

/// 2026-10-23: The metrale pack.
#[derive(Clone, Copy, Debug)]
pub struct Metrale;

impl Vocab for Metrale {
    const LABELS: &'static [Label] = PACK_LABELS;
    const RELS: &'static [Rel] = PACK_RELS;
}

impl Pack for Metrale {
    const ID: &'static str = "metrale";
    const VERSION: &'static str = "1.0.0";
    const CONFIG_SCHEMA: &'static str = CONFIG_SCHEMA;
    const NAMED_LABELS: &'static [Label] = &[labels::LedgerEntry, labels::Lever];
    const SEARCH_EXPAND: &'static [Rel] = &[
        rels::Toggles,
        rels::PredictedBy,
        rels::LandsAs,
        rels::DependsOn,
        rels::Links,
        rels::Cites,
    ];
    const IMPACT_FOLLOW_OUT: &'static [Rel] = &[rels::Pins, rels::MeasuredIn, rels::DecidedBy];
    const SECTIONS: &'static [&'static str] = &[
        "progress",
        "blocked",
        "tainted",
        "claims",
        "owed_lkb",
        "calibration",
    ];
    /// `calibration` is the retrospective (design §5) and comes only when named.
    const DEFAULT_SECTIONS: &'static [&'static str] =
        &["progress", "blocked", "tainted", "claims", "owed_lkb"];
    const RECORD_KINDS: &'static [&'static str] = crate::record::KINDS;
    const INSTRUCTIONS: &'static str = "Derived index of campaign ledgers, frictions, decisions, \
designs, memory rules and the LKB; files stay the source of truth. Start with la_status. \
la_search before a new hypothesis, la_impact before a retraction. In la_query, n.id is the typed \
id; filter on n.name.";
    const STATUS_DESCRIPTION: &'static str = "Campaign progress, blockers, taint, claims, owed LKB rows. Default: compact (lists capped); name a section for all of it.";
    const IMPACT_DESCRIPTION: &'static str =
        "Dependency/taint closure of an id (== ledger.py tainted/deps).";

    type Sources = Sources;
    type Facts = LkbFacts;

    fn root(s: &Sources) -> &str {
        &s.root
    }

    fn validate(s: &Sources) -> Result<()> {
        s.validate()
    }

    fn source_paths(s: &Sources) -> Vec<String> {
        s.paths()
    }

    fn source_dirs(s: &Sources) -> Vec<String> {
        vec![s.memory_dir.clone()]
    }

    fn scopes(s: &Sources) -> Vec<String> {
        s.campaigns.iter().map(|c| c.id.clone()).collect()
    }

    fn default_instance(s: &Sources, scope: &str) -> Option<String> {
        s.campaign(scope).ok()?.default_instance.clone()
    }

    fn derive(
        cfg: &Config,
        src: &SourceSet,
        facts: Option<&LkbFacts>,
        b: &mut Builder,
    ) -> Result<()> {
        let s = &cfg.sources;
        let mut texts = Vec::new();
        for p in &s.principles {
            texts.push((p.as_str(), need(src, p)?));
        }
        let known = principles::derive(b, &texts, &texts);
        let mut ledger_ids = BTreeSet::new();
        for c in &s.campaigns {
            let ids = ledger_adapter::derive(b, &c.id, &c.ledger, need(src, &c.ledger)?, &known);
            ledger_ids.extend(ids);
        }
        for c in &s.campaigns {
            decisions::charter(b, &c.charter, need(src, &c.charter)?, &ledger_ids);
            tooling::derive(b, &c.id, &c.tooling, need(src, &c.tooling)?, &known);
            runs::derive(b, &c.runs, need(src, &c.runs)?);
        }
        for d in &s.designs {
            decisions::design(b, d, need(src, d)?, &ledger_ids);
        }
        research::graph_theory_map(b, &s.graph_theory_map, need(src, &s.graph_theory_map)?);
        research::mathbomb(b, &s.mathbomb, need(src, &s.mathbomb)?);
        match (&s.lkb, facts) {
            (Some(t), Some(f)) => lkb_derive::derive(b, &LkbSources::parse(t)?, src, f)?,
            (None, None) => {}
            (Some(_), None) => {
                return Err(error!(
                    ErrorCode::LatlasSource,
                    "sources.lkb", "is configured but the host gave no LKB facts"
                ));
            }
            (None, Some(_)) => {
                return Err(error!(
                    ErrorCode::LatlasSource,
                    "sources.lkb", "is not configured but LKB facts were given"
                ));
            }
        }
        memory::derive(b, src.under(&s.memory_dir));
        Ok(())
    }

    fn sealed(bundle: &mut Bundle, facts: Option<&LkbFacts>) {
        if let Some(f) = facts {
            owed::mark_stale(bundle, &f.file_last, &f.commit_date);
        }
    }

    fn section(ctx: &SectionCtx<'_, Self>, name: &str) -> Result<Value> {
        crate::sections::section(ctx, name)
    }

    fn impact_rels(label: Label, downstream: bool) -> Option<Vec<Rel>> {
        if label != labels::LedgerEntry {
            return None;
        }
        Some(if downstream {
            vec![
                rels::DependsOn,
                rels::Pins,
                rels::MeasuredIn,
                rels::DecidedBy,
            ]
        } else {
            vec![rels::DependsOn]
        })
    }

    fn impact_extra(
        b: &Bundle,
        cfg: &Config,
        root: &NodeId,
        out: &mut Map<String, Value>,
    ) -> Result<()> {
        if root.label() != labels::LedgerEntry {
            return Ok(());
        }
        let campaign = b.field(root, "campaign").unwrap_or("").to_string();
        let c = cfg.sources.campaign(&campaign)?;
        queries::ledger_valid(b, &c.ledger)?;
        let links = queries::ledger_links(b, &campaign);
        let chain = ledger::deps(&links, root.key());
        let mut upstream: Vec<&str> = chain.iter().map(String::as_str).collect();
        upstream.push(root.key());
        let tainted: Vec<Value> = ledger::tainted(&links)
            .into_iter()
            .filter(|(id, _)| upstream.contains(&id.as_str()))
            .map(|(id, why)| Value::from(vec![id, why]))
            .collect();
        out.insert("tainted_by".into(), Value::from(tainted));
        out.insert(
            "ledger_deps".into(),
            Value::String(render_deps(&chain).trim_end().to_string()),
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use latlas_core::id::{LABEL_LIMIT, labels as all_labels, vocab_is_valid};

    #[test]
    fn the_vocabulary_fits_one_collection_and_repeats_no_core_name() {
        assert!(vocab_is_valid::<Metrale>());
        assert!(all_labels::<Metrale>().count() <= LABEL_LIMIT);
        assert_eq!(all_labels::<Metrale>().count(), 35);
    }

    #[test]
    fn exactly_five_tools_with_short_descriptions_and_valid_schemas() {
        let t = latlas_core::tools::tools::<Metrale>();
        assert_eq!(t.len(), 5);
        for x in &t {
            let d = x["description"].as_str().unwrap();
            assert!(d.chars().count() <= 140, "{d}");
            assert_eq!(x["inputSchema"]["type"], "object", "{}", x["name"]);
        }
        assert!(Metrale::INSTRUCTIONS.split_whitespace().count() <= 120);
    }
}
