// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The native hosts of `kg_record`'s ports: [`FsRecord`] (read, and the atomic
//! replace of [`crate::write::replace`]) and [`LedgerValidator`] (the campaign's `ledger.py check`
//! for LEDGER.tsv; the other kinds have no external validator and say so).
//!
//! Owner: metrale-io-host-kg.
//! Invariants:
//! - Paths are root-relative; one that is absolute or climbs out of the root is refused.

use std::path::{Path, PathBuf};

use metrale_error::{ErrorCode, Result, error};
use metrale_kg_derive::record::Kind;
use metrale_kg_derive::record_flow::{RecordFs, Validator};

use crate::ledger_py::LedgerPy;

/// 2026-10-23: The file port over `root`.
pub struct FsRecord {
    pub root: PathBuf,
}

impl FsRecord {
    fn abs(&self, rel: &str) -> Result<PathBuf> {
        let p = Path::new(rel);
        if p.is_absolute()
            || p.components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            return Err(error!(
                ErrorCode::KgRecordWrite,
                rel.to_string(),
                "must be root-relative without .."
            ));
        }
        Ok(self.root.join(p))
    }
}

impl RecordFs for FsRecord {
    async fn read(&self, path: &str) -> Result<Option<String>> {
        match std::fs::read_to_string(self.abs(path)?) {
            Ok(t) => Ok(Some(t)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(error!(ErrorCode::KgSource, path.to_string(), e.to_string())),
        }
    }

    async fn replace(&self, path: &str, expected: &str, new: &str) -> Result<()> {
        crate::write::replace(&self.abs(path)?, expected, new)
    }
}

/// 2026-10-23: The validators of record: `ledger.py check` for the ledger.
pub struct LedgerValidator {
    pub ledger_py: LedgerPy,
}

impl Validator for LedgerValidator {
    async fn validate(&self, kind: Kind, path: &str, new: &str) -> Result<String> {
        if kind != Kind::Ledger {
            return Ok("adapter check (no external validator for this kind)".into());
        }
        let run = self.ledger_py.run(new, &["check"])?;
        let said = run.stdout.trim_end().to_string();
        if run.code != 0 {
            return Err(error!(
                ErrorCode::KgValidatorRejected,
                path.to_string(),
                format!("ledger.py check: {said}")
            ));
        }
        Ok(format!("ledger.py check: {said}"))
    }
}
