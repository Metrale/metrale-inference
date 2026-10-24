// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: `MetraleHost`: the metrale pack's process side for `latlas-cli` (K1): the LKB
//! checkout's files, stamp and facts, and `la_record` (parse the arguments, run the pack's
//! `record_flow::record` over latlas-host-native's file port and the ledger's validator of
//! record, answer with the taint preview).
//!
//! Owner: metrale-kg-mcp.
//! Invariants:
//! - The record flow itself (checks, preview, fail-closed write) lives in the pack; this is
//!   wiring only.
//! - Built once at open from the resolved config (ROTI).

use anyhow::Result;
use latlas_cli::PackHost;
use latlas_core::Bundle;
use latlas_error::{ErrorCode, error};
use latlas_host_native::fs::{FsSources, Stamp};
use latlas_host_native::record::FsRecord;
use metrale_io_host_kg::ledger_py::LedgerPy;
use metrale_io_host_kg::lkb;
use metrale_io_host_kg::record::LedgerValidator;
use metrale_latlas_pack::lkb_derive::{LkbFacts, LkbSources};
use metrale_latlas_pack::record::Kind;
use metrale_latlas_pack::record_flow::{Request, record};
use metrale_latlas_pack::{Config, Metrale};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// 2026-10-23: The metrale pack's process host.
pub struct MetraleHost {
    /// 2026-10-23: `[sources.lkb]`, parsed once (`None`: no LKB configured).
    lkb: Option<LkbSources>,
    python: PathBuf,
    scratch: PathBuf,
}

impl PackHost<Metrale> for MetraleHost {
    fn open(cfg: &Config, _fs: &FsSources) -> Result<Self> {
        let lkb = match &cfg.sources.lkb {
            Some(t) => Some(LkbSources::parse(t)?),
            None => None,
        };
        Ok(MetraleHost {
            lkb,
            python: PathBuf::from(&cfg.validator.python),
            scratch: PathBuf::from(&cfg.store.scratch_dir),
        })
    }

    fn extra_files(&self, _cfg: &Config, fs: &FsSources) -> Result<Vec<String>> {
        match &self.lkb {
            Some(l) => Ok(lkb::lkb_files(fs, l)?),
            None => Ok(Vec::new()),
        }
    }

    fn extra_stamp(&self, _cfg: &Config, fs: &FsSources) -> Result<Stamp> {
        match &self.lkb {
            Some(l) => Ok(vec![lkb::repo_stamp(&fs.root, l)?]),
            None => Ok(Vec::new()),
        }
    }

    fn facts(&self, _cfg: &Config, fs: &FsSources) -> Result<(Option<LkbFacts>, Option<String>)> {
        let Some(l) = &self.lkb else {
            return Ok((None, None));
        };
        let python = self.python.to_string_lossy();
        let f = lkb::facts_cached(&fs.root, l, &python, &self.scratch.join("lkb-facts"))?;
        let head = f.head.clone();
        Ok((Some(f), Some(head)))
    }

    async fn record(
        &self,
        cfg: &Config,
        fs: &FsSources,
        _bundle: &Bundle,
        args: &Value,
    ) -> latlas_error::Result<(Value, bool)> {
        let kind = Kind::parse(args.get("kind").and_then(Value::as_str).unwrap_or(""))?;
        let scope = args.get("scope").and_then(Value::as_str).unwrap_or("");
        let c = cfg.sources.campaign(scope)?.clone();
        let dry_run = args
            .get("dry_run")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let acknowledge_taint =
            match args.get("acknowledge_taint") {
                None | Some(Value::Null) => None,
                Some(v) => Some(v.as_u64().and_then(|n| u32::try_from(n).ok()).ok_or_else(
                    || {
                        error!(
                            ErrorCode::LatlasToolArgument,
                            "acknowledge_taint", "a count"
                        )
                    },
                )?),
            };
        let mut fields = BTreeMap::new();
        for (k, v) in args
            .get("fields")
            .and_then(Value::as_object)
            .ok_or_else(|| error!(ErrorCode::LatlasToolArgument, "fields", "must be an object"))?
        {
            let s = v.as_str().ok_or_else(|| {
                error!(ErrorCode::LatlasToolArgument, k.clone(), "must be a string")
            })?;
            fields.insert(k.clone(), s.to_string());
        }
        let file_port = FsRecord {
            root: fs.root.clone(),
        };
        let validator = LedgerValidator {
            ledger_py: LedgerPy {
                python: self.python.clone(),
                script: fs.root.join(&c.ledger_check),
                scratch: self.scratch.clone(),
            },
        };
        let req = Request {
            kind,
            fields,
            dry_run,
            acknowledge_taint,
        };
        let out = record(&file_port, &validator, &c, &req).await?;
        let mut o = Map::new();
        o.insert("id".into(), Value::String(out.id));
        o.insert("file".into(), Value::String(out.file));
        o.insert("line".into(), Value::from(out.line));
        o.insert("validator".into(), Value::String(out.validator));
        if let Some(cl) = out.claims {
            o.insert("claims".into(), Value::String(cl));
        }
        o.insert("would_taint".into(), Value::from(out.would_taint));
        o.insert("dry_run".into(), Value::Bool(dry_run));
        o.insert("written".into(), Value::Bool(out.written));
        o.insert("warnings".into(), Value::from(out.warnings));
        Ok((Value::Object(o), out.written))
    }
}
