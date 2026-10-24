// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The validator of record for `la_record` (the file port is latlas-host-native's
//! `FsRecord`): [`LedgerValidator`] runs the campaign's `ledger.py check` for LEDGER.tsv; the
//! other kinds have no external validator and say so.
//!
//! Owner: metrale-io-host-kg.
//! Invariants: a refusal names the validator's own output.

use latlas_error::{ErrorCode, Result, error};
use metrale_latlas_pack::record::Kind;
use metrale_latlas_pack::record_flow::Validator;

use crate::ledger_py::LedgerPy;

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
                ErrorCode::LatlasValidatorRejected,
                path.to_string(),
                format!("ledger.py check: {said}")
            ));
        }
        Ok(format!("ledger.py check: {said}"))
    }
}
