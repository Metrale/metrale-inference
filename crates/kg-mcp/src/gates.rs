// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The metrale pack's gates as commands (the generic `check`, `derive` and `probes`
//! are latlas-cli's): `parity` (`la_impact`'s closures and the Rust check == the campaign's own
//! ledger.py, byte for byte, run through the real host), and `tainted`/`deps` in ledger.py's
//! format. Each prints its output and fails the process on any mismatch.
//!
//! Owner: metrale-kg-mcp.
//! Invariants: the gates never write a source file.

use crate::host::MetraleHost;
use anyhow::Result;
use latlas_cli::app::{derive_once, open_hosts};
use latlas_cli::gates::GateFailed;
use metrale_io_host_kg::ledger_py::LedgerPy;
use metrale_latlas_pack::ledger::{self, Table, render_check, render_deps, render_tainted};
use metrale_latlas_pack::{Metrale, derive, labels, queries};
use serde_json::json;
use std::path::{Path, PathBuf};

/// 2026-10-23: `parity`.
pub fn parity_cmd(cfg_path: &Path, campaign: &str, ledger_file: Option<&Path>) -> Result<()> {
    let (cfg, fs, host) = open_hosts::<Metrale, MetraleHost>(cfg_path)?;
    let c = cfg.sources.campaign(campaign)?.clone();
    let extra = latlas_cli::PackHost::extra_files(&host, &cfg, &fs)?;
    let mut src = fs.read::<Metrale>(&cfg, &extra)?;
    if let Some(f) = ledger_file {
        src.files
            .insert(c.ledger.clone(), std::fs::read_to_string(f)?);
    }
    let text = src.get(&c.ledger).unwrap_or("").to_string();
    let py = LedgerPy {
        python: PathBuf::from(&cfg.validator.python),
        script: fs.root.join(&c.ledger_check),
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
        "ledger_sha256": latlas_core::hash::sha256(text.as_bytes()).hex(),
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
    let (cfg, fs, host) = open_hosts::<Metrale, MetraleHost>(cfg_path)?;
    let c = cfg.sources.campaign(campaign)?;
    let (_, b, _) = derive_once::<Metrale, MetraleHost>(&cfg, &fs, &host)?;
    queries::ledger_valid(&b, &c.ledger)?;
    print!(
        "{}",
        render_tainted(&ledger::tainted(&queries::ledger_links(&b, campaign)))
    );
    Ok(())
}

/// 2026-10-23: `deps` in ledger.py's format.
pub fn deps_cmd(cfg_path: &Path, id: &str) -> Result<()> {
    let (cfg, fs, host) = open_hosts::<Metrale, MetraleHost>(cfg_path)?;
    let (_, b, _) = derive_once::<Metrale, MetraleHost>(&cfg, &fs, &host)?;
    let root = latlas_core::NodeId::new(labels::LedgerEntry, id)?;
    let campaign = b.field(&root, "campaign").unwrap_or("").to_string();
    let c = cfg.sources.campaign(&campaign)?;
    queries::ledger_valid(&b, &c.ledger)?;
    print!(
        "{}",
        render_deps(&ledger::deps(&queries::ledger_links(&b, &campaign), id))
    );
    Ok(())
}
