// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The P0 gates as commands (design §6): `check` (derive twice ⇒ one digest; fatal
//! findings; two fresh stores ⇒ one export; reopen ⇒ the same export; cold start), and `parity`
//! (`kg_impact`'s closures and the Rust check == the campaign's own ledger.py, byte for byte, run
//! through the real host). Each prints a JSON verdict and fails the process on any mismatch.
//!
//! Owner: metrale-kg-mcp.
//! Invariants: the gates never write a source file; stores are created under `store.scratch_dir`.

use crate::app::derive_once;
use crate::store::Store;
use anyhow::Result;
use metrale_abstract_io::Clock;
use metrale_io_host_kg::fs::{FsSources, read_config};
use metrale_io_host_kg::ledger_py::LedgerPy;
use metrale_io_host_std::SystemClock;
use metrale_kg_derive::jsonl::{digest, render};
use metrale_kg_derive::ledger::{self, Table, render_check, render_deps, render_tainted};
use metrale_kg_derive::{derive, queries};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// 2026-10-23: `derive`: the bundle JSONL to stdout or a file.
/// 2026-10-23: A gate whose report is already on stdout failed; `main` exits 1 without printing a
/// second JSON document.
#[derive(Debug)]
pub struct GateFailed(pub &'static str);

impl std::fmt::Display for GateFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} failed (report on stdout)", self.0)
    }
}

impl std::error::Error for GateFailed {}

pub fn derive_cmd(cfg_path: &Path, out: Option<&Path>) -> Result<()> {
    let cfg = read_config(cfg_path)?;
    let (_, b, _) = derive_once(&cfg, &FsSources::new(&cfg))?;
    let text = render(&b);
    match out {
        Some(p) => std::fs::write(p, text)?,
        None => print!("{text}"),
    }
    Ok(())
}

fn fresh_dir(base: &str, name: &str) -> Result<PathBuf> {
    let d = PathBuf::from(base).join(name);
    if d.exists() {
        std::fs::remove_dir_all(&d)?;
    }
    Ok(d)
}

/// 2026-10-23: `check`.
pub async fn check_cmd(cfg_path: &Path, roundtrip: bool) -> Result<()> {
    let clock = SystemClock;
    let cfg = read_config(cfg_path)?;
    let host = FsSources::new(&cfg);
    let t0 = clock.monotonic_ns().0;
    let (_, a, source_commit) = derive_once(&cfg, &host)?;
    let t1 = clock.monotonic_ns().0;
    let (_, b, _) = derive_once(&cfg, &host)?;
    let (da, db) = (digest(&a).hex(), digest(&b).hex());
    let mut by_code: BTreeMap<&str, u32> = BTreeMap::new();
    for f in &a.errors {
        *by_code.entry(f.code).or_default() += 1;
    }
    let fatal: Vec<Value> = a
        .fatal()
        .iter()
        .map(|f| json!([f.code, f.at, f.msg]))
        .collect();
    let mut v = json!({
        "digest": da, "digest_again": db, "fixpoint": da == db,
        "counts": {"nodes": a.nodes.len(), "edges": a.edges.len(), "texts": a.texts.len(), "sources": a.sources.len()},
        "findings_by_code": by_code, "fatal": fatal, "derive_ms": (t1 - t0) / 1_000_000,
        "source_commit": source_commit,
    });
    let mut ok = da == db && fatal.is_empty();
    if roundtrip {
        let s0 = clock.monotonic_ns().0;
        let dir_a = fresh_dir(&cfg.store.scratch_dir, "kg-check-a")?;
        let mut sa = Store::open(&dir_a).await?;
        sa.load(&a).await?;
        let cold_ms = (clock.monotonic_ns().0 - s0) / 1_000_000 + (t1 - t0) / 1_000_000;
        let ea = sa.export()?;
        drop(sa);
        let mut sb = Store::open(&fresh_dir(&cfg.store.scratch_dir, "kg-check-b")?).await?;
        sb.load(&b).await?;
        let eb = sb.export()?;
        let mut sa2 = Store::open(&dir_a).await?;
        let rewrote = sa2.load(&a).await?;
        let ea2 = sa2.export()?;
        let same = ea == eb && ea == ea2 && !rewrote;
        ok &= same && cold_ms < 2000;
        v["roundtrip"] = json!({
            "fresh_stores_identical": ea == eb, "reopen_identical": ea == ea2, "reopen_rewrote": rewrote,
            "export_lines": ea.lines().count(), "export_sha256": metrale_kg_derive::hash::sha256(ea.as_bytes()).hex(),
            "cold_start_ms": cold_ms, "vector_size": 1,
        });
    }
    println!("{}", serde_json::to_string_pretty(&v)?);
    if !ok {
        return Err(GateFailed("metrale-kg check").into());
    }
    Ok(())
}

/// 2026-10-23: `parity`.
pub fn parity_cmd(cfg_path: &Path, campaign: &str, ledger_file: Option<&Path>) -> Result<()> {
    let cfg = read_config(cfg_path)?;
    let c = cfg.campaign(campaign)?.clone();
    let host = FsSources::new(&cfg);
    let mut src = host.read(&cfg)?;
    if let Some(f) = ledger_file {
        src.files
            .insert(c.ledger.clone(), std::fs::read_to_string(f)?);
    }
    let text = src.get(&c.ledger).unwrap_or("").to_string();
    let py = LedgerPy {
        python: PathBuf::from(&cfg.validator.python),
        script: host.root.join(&c.ledger_check),
        scratch: PathBuf::from(&cfg.store.scratch_dir),
    };
    std::fs::create_dir_all(&py.scratch)?;
    let table = Table::parse(&text);
    let mut bad: Vec<String> = Vec::new();
    let chk = py.run(&text, &["check"])?;
    let rust_check = render_check(&table);
    if chk.stdout != rust_check || (chk.code == 0) != table.check().is_empty() {
        bad.push(format!(
            "check: python {:?} rust {:?}",
            chk.stdout, rust_check
        ));
    }
    // 2026-10-23: parity is over the campaign records; the LKB half is not needed for it.
    let mut records = cfg.clone();
    records.sources.lkb = None;
    let bundle = derive(&records, &src, None)?;
    let valid = queries::ledger_valid(&bundle, &c.ledger).is_ok();
    let links = queries::ledger_links(&bundle, campaign);
    let tnt = py.run(&text, &["tainted"])?;
    if valid {
        let rust = render_tainted(&ledger::tainted(&links));
        if tnt.stdout != rust {
            bad.push(format!("tainted: python {:?} rust {:?}", tnt.stdout, rust));
        }
    } else if tnt.code == 0 || !tnt.stderr.contains(ledger::INVALID) {
        bad.push("tainted: python accepted a ledger the bundle refuses".into());
    }
    let mut ids: Vec<String> = table.links().into_iter().map(|l| l.id).collect();
    ids.push("no-such-row".into());
    let mut deps_same = 0;
    if valid {
        for id in &ids {
            let p = py.run(&text, &["deps", id])?;
            let r = render_deps(&ledger::deps(&links, id));
            if p.stdout == r {
                deps_same += 1;
            } else {
                bad.push(format!("deps {id}: python {:?} rust {:?}", p.stdout, r));
            }
        }
    }
    let v = json!({
        "ledger": c.ledger, "rows": table.rows.len(), "ledger_valid": valid,
        "check_identical": !bad.iter().any(|m| m.starts_with("check")),
        "tainted_lines": tnt.stdout.lines().count(),
        "deps_identical": format!("{deps_same}/{}", if valid { ids.len() } else { 0 }),
        "ledger_sha256": metrale_kg_derive::hash::sha256(text.as_bytes()).hex(),
        "mismatches": bad,
    });
    println!("{}", serde_json::to_string_pretty(&v)?);
    if !v["mismatches"].as_array().is_some_and(Vec::is_empty) {
        return Err(GateFailed("ledger.py parity").into());
    }
    Ok(())
}

/// 2026-10-23: `tainted` in ledger.py's format.
pub fn tainted_cmd(cfg_path: &Path, campaign: &str) -> Result<()> {
    let cfg = read_config(cfg_path)?;
    let c = cfg.campaign(campaign)?;
    let (_, b, _) = derive_once(&cfg, &FsSources::new(&cfg))?;
    queries::ledger_valid(&b, &c.ledger)?;
    print!(
        "{}",
        render_tainted(&ledger::tainted(&queries::ledger_links(&b, campaign)))
    );
    Ok(())
}

/// 2026-10-23: `deps` in ledger.py's format.
pub fn deps_cmd(cfg_path: &Path, id: &str) -> Result<()> {
    let cfg = read_config(cfg_path)?;
    let (_, b, _) = derive_once(&cfg, &FsSources::new(&cfg))?;
    let root = metrale_kg_derive::KgId::new(metrale_kg_derive::Label::LedgerEntry, id)?;
    let campaign = b.field(&root, "campaign").unwrap_or("").to_string();
    let c = cfg.campaign(&campaign)?;
    queries::ledger_valid(&b, &c.ledger)?;
    print!(
        "{}",
        render_deps(&ledger::deps(&queries::ledger_links(&b, &campaign), id))
    );
    Ok(())
}
