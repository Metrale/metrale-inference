// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The LKB half of `derive` (P1): the configured LKB files of one hardware directory,
//! read through metrale-circuit's own parsers, plus the host-supplied facts no file holds (the
//! `kernel_perf.py --json` inventory, git dates of kernel files and commits).
//!
//! Owner: metrale-latlas-pack (the metrale pack; was metrale-kg-derive).
//! Invariants:
//! - Every path is repo-relative in config and root-relative (`<repo>/<path>`) in the source set.
//! - A parser refusal is a `KgBadRow` finding at the file, never a silent skip of the file.
//! - Order: inventory, families, rules, critical pairs, allowlist, instances, circuits, plans, headroom,
//!   measurements (later steps edge into nodes earlier steps define).

use crate::lkb::{families, handwritten, inventory, kernels};
use crate::lkb_circuits::{circuit, instances, plan};
use crate::lkb_headroom::{predicted_by, report};
use crate::lkb_rules::{critical_pairs, measurements, rules};
use crate::prelude::*;
use crate::vocab::*;
use latlas_core::bundle::Builder;
use latlas_core::sources::SourceSet;
use latlas_error::{ErrorCode, Result, error};
use serde::Deserialize;
use serde_json::Value;

/// 2026-10-23: `[sources.lkb]`: one hardware directory of the LKB checkout.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LkbSources {
    /// 2026-10-23: The checkout, root-relative (`metrale/wt-mk`).
    pub repo: String,
    /// 2026-10-23: The hardware directory name (`gb10`).
    pub hardware: String,
    pub families: String,
    pub fusions: String,
    pub handwritten: String,
    pub measurements: String,
    pub instances: String,
    pub circuits_dir: String,
    pub plans_dir: String,
    pub headroom_dir: String,
    /// 2026-10-23: The critical-pairs allowlist (`kernels/circuits/CRITICAL_PAIRS.toml`, P4).
    pub critical_pairs: String,
    /// 2026-10-23: The inventory script the host runs (`scripts/kernel_perf.py`).
    pub inventory: String,
    /// 2026-10-23: What `metrale-kg refresh` moves the checkout to: `git fetch <remote> <branch>`,
    /// then a detached checkout of `<remote>/<branch>`. The checkout is a read-only source.
    pub remote: String,
    pub branch: String,
}

impl LkbSources {
    /// 2026-10-23: Parse the raw `[sources.lkb]` table (every field required, unknown refused).
    pub fn parse(t: &toml::Table) -> Result<LkbSources> {
        let s: LkbSources =
            toml::Value::Table(t.clone())
                .try_into()
                .map_err(|e: toml::de::Error| {
                    error!(ErrorCode::LatlasConfig, "sources.lkb", e.to_string())
                })?;
        let all = [
            &s.repo,
            &s.families,
            &s.fusions,
            &s.handwritten,
            &s.measurements,
            &s.instances,
            &s.circuits_dir,
            &s.plans_dir,
            &s.headroom_dir,
            &s.critical_pairs,
            &s.inventory,
            &s.remote,
            &s.branch,
        ];
        for p in all {
            if p.starts_with('/') || p.split('/').any(|c| c == "..") || p.is_empty() {
                return Err(error!(
                    ErrorCode::LatlasConfig,
                    format!("sources.lkb {p}"),
                    "must be relative, non-empty, without .."
                ));
            }
        }
        for (k, v) in [("remote", &s.remote), ("branch", &s.branch)] {
            let ok = !v.starts_with('-')
                && v.chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '/' | '-'));
            if !ok {
                return Err(error!(
                    ErrorCode::LatlasConfig,
                    format!("sources.lkb.{k}"),
                    "must be a git ref name ([A-Za-z0-9._/-], not starting with -)"
                ));
            }
        }
        Ok(s)
    }

    /// 2026-10-23: The root-relative path of a repo-relative one.
    pub fn path(&self, rel: &str) -> String {
        format!("{}/{rel}", self.repo)
    }

    /// 2026-10-23: The files and directories the host reads, root-relative.
    pub fn files(&self) -> [String; 6] {
        [
            self.families.as_str(),
            self.fusions.as_str(),
            self.handwritten.as_str(),
            self.measurements.as_str(),
            self.instances.as_str(),
            self.critical_pairs.as_str(),
        ]
        .map(|p| self.path(p))
    }

    /// 2026-10-23: The directories whose `.toml`, `.txt` and `.json` files the host reads.
    pub fn dirs(&self) -> [String; 3] {
        [
            self.circuits_dir.as_str(),
            self.plans_dir.as_str(),
            self.headroom_dir.as_str(),
        ]
        .map(|p| self.path(p))
    }
}

/// 2026-10-23: Facts the host computes that no SSOT file holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LkbFacts {
    /// 2026-10-23: The checkout's HEAD when the facts were gathered (every answer names it).
    pub head: String,
    /// 2026-10-23: `kernel_perf.py --json --partial` stdout, or why the script refused to build the
    /// inventory (its strict model rejects an LKB with, e.g., an unclassified kernel). A refusal is
    /// a finding and makes `owed_lkb` a declared unavailable section; the rest of the graph stays.
    pub inventory: core::result::Result<String, String>,
    /// 2026-10-23: kernel file (repo-relative) → (last commit sha, its commit time, unix seconds).
    pub file_last: BTreeMap<String, (String, String)>,
    /// 2026-10-23: commit sha (as a measurement cites it) → commit time, unix seconds.
    pub commit_date: BTreeMap<String, String>,
}

fn need<'a>(src: &'a SourceSet, p: &str) -> Result<&'a str> {
    src.get(p).ok_or_else(|| {
        error!(
            ErrorCode::LatlasSource,
            p.to_string(),
            "is configured but was not read"
        )
    })
}

/// 2026-10-23: A whole LKB file its own reader refuses (e.g. a schema field this build's
/// metrale-circuit does not know): fatal, since every node of the file would be missing.
fn rejected(b: &mut Builder, path: &str, why: String) {
    b.error(
        &error!(ErrorCode::LatlasLkbFileRejected, path.to_string(), why),
        path,
    );
}

/// 2026-10-23: Derive the LKB nodes and edges into `b`.
pub fn derive(b: &mut Builder, cfg: &LkbSources, src: &SourceSet, facts: &LkbFacts) -> Result<()> {
    let inv_path = cfg.path(&cfg.inventory);
    let inv = match &facts.inventory {
        Ok(json) => {
            let got = inventory(json, &cfg.hardware)
                .map_err(|e| error!(ErrorCode::LatlasSource, inv_path.clone(), e))?;
            for p in got.problems {
                b.error(
                    &error!(ErrorCode::LatlasInventoryProblem, inv_path.clone(), p),
                    &inv_path,
                );
            }
            got.kernels
        }
        Err(why) => {
            b.error(
                &error!(
                    ErrorCode::LatlasInventoryRefused,
                    inv_path.clone(),
                    why.clone()
                ),
                &inv_path,
            );
            BTreeMap::new()
        }
    };
    kernels(b, &cfg.repo, &inv);

    let fp = cfg.path(&cfg.families);
    let ftext = need(src, &fp)?;
    let mut emitters: BTreeMap<String, Vec<String>> = BTreeMap::new();
    match metrale_circuit::venn::parse_families(ftext) {
        Ok(f) => {
            for fam in &f.families {
                for e in &fam.emitters {
                    emitters.entry(e.clone()).or_default().push(fam.id.clone());
                }
            }
            families(b, &fp, ftext, &f, &inv);
        }
        Err(e) => rejected(b, &fp, format!("{e}")),
    }

    let rp = cfg.path(&cfg.fusions);
    let rtext = need(src, &rp)?;
    match metrale_circuit::runtime::parse_rule_set(rtext) {
        Ok(set) => rules(b, &rp, rtext, &set.rules, &set.runtime, &emitters),
        Err(e) => rejected(b, &rp, format!("{e}")),
    }

    let cp = cfg.path(&cfg.critical_pairs);
    let ctext = need(src, &cp)?;
    match metrale_circuit::critical_report::parse_allowlist(ctext) {
        Ok(pairs) => critical_pairs(b, &cp, ctext, &pairs),
        Err(e) => rejected(b, &cp, e),
    }

    let hp = cfg.path(&cfg.handwritten);
    handwritten(b, &hp, need(src, &hp)?);

    let ip = cfg.path(&cfg.instances);
    let itext = need(src, &ip)?;
    let mut arch_of: BTreeMap<String, String> = BTreeMap::new();
    let mut by_recipe: BTreeMap<String, (String, String)> = BTreeMap::new();
    match metrale_circuit::parse_instances(itext) {
        Ok(all) => {
            for i in &all {
                arch_of.insert(i.checkpoint.clone(), i.arch.clone());
                by_recipe.insert(
                    i.recipe.clone(),
                    (crate::lkb_circuits::instance_id(i), i.arch.clone()),
                );
            }
            instances(b, &ip, itext, &all);
        }
        Err(e) => rejected(b, &ip, format!("{e}")),
    }

    let [cdir, pdir, hdir] = cfg.dirs();
    let in_dir = |dir: &str, ext: &str| -> Vec<(String, String)> {
        src.under(dir)
            .filter(|(p, _)| p.ends_with(ext) && !p[dir.len() + 1..].contains('/'))
            .filter_map(|(p, t)| Some((p.to_string(), t?.to_string())))
            .collect()
    };
    let is_circuit = |p: &str| {
        let name = p.rsplit('/').next().unwrap_or(p);
        !name.contains(".config.") && name.starts_with(|c: char| c.is_ascii_lowercase())
    };
    for (p, t) in in_dir(&cdir, ".toml")
        .into_iter()
        .filter(|(p, _)| is_circuit(p))
    {
        circuit(b, &p, &t);
    }
    for (p, t) in in_dir(&pdir, ".txt") {
        plan(b, &p, &t);
    }
    let mut sites = Vec::new();
    for (p, t) in in_dir(&hdir, ".json") {
        sites.extend(report(b, &p, &t, &by_recipe));
    }
    let rows: Vec<_> = b
        .defined_of(labels::LedgerEntry)
        .map(|n| {
            (
                n.id.clone(),
                n.src.clone(),
                n.p.get("predicted_us_saved")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            )
        })
        .collect();
    predicted_by(b, &rows, &sites);

    let mp = cfg.path(&cfg.measurements);
    let mtext = need(src, &mp)?;
    match metrale_circuit::venn::parse_measurement_rows(mtext) {
        Ok(rows) => measurements(b, &mp, mtext, &rows, &arch_of),
        Err(e) => rejected(b, &mp, e),
    }
    Ok(())
}
