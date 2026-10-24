// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: A campaign's own `tools/ledger.py`, run on a private copy: the validator of record
//! for `la_record kind=ledger` (design §2.2) and the reference side of the P0 parity gate.
//! ledger.py resolves `LEDGER.tsv` beside its own parent directory, so a copy of the script in
//! `<scratch>/<run>/tools/` checks the candidate text in `<scratch>/<run>/LEDGER.tsv` and never
//! the live file.
//!
//! Owner: metrale-io-host-kg.
//! Invariants:
//! - The live ledger is never opened for writing here.
//! - Each run gets its own directory, removed afterwards.

use latlas_error::{ErrorCode, Result, error};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static RUN: AtomicU64 = AtomicU64::new(0);

/// 2026-10-23: The outcome of one ledger.py run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Run {
    pub stdout: String,
    pub stderr: String,
    pub code: i32,
}

/// 2026-10-23: The script, the interpreter and the scratch directory.
#[derive(Clone, Debug)]
pub struct LedgerPy {
    pub python: PathBuf,
    pub script: PathBuf,
    pub scratch: PathBuf,
}

fn io(what: &Path, e: &std::io::Error) -> latlas_error::Error {
    error!(
        ErrorCode::LatlasRecordWrite,
        what.display().to_string(),
        e.to_string()
    )
}

impl LedgerPy {
    /// 2026-10-23: Run `ledger.py <args>` against `ledger_text`.
    pub fn run(&self, ledger_text: &str, args: &[&str]) -> Result<Run> {
        let n = RUN.fetch_add(1, Ordering::Relaxed);
        let dir = self
            .scratch
            .join(format!("ledger-py-{}-{n}", std::process::id()));
        let tools = dir.join("tools");
        std::fs::create_dir_all(&tools).map_err(|e| io(&tools, &e))?;
        let out = (|| {
            let script = tools.join("ledger.py");
            std::fs::copy(&self.script, &script).map_err(|e| io(&self.script, &e))?;
            let ledger = dir.join("LEDGER.tsv");
            std::fs::write(&ledger, ledger_text).map_err(|e| io(&ledger, &e))?;
            let o = Command::new(&self.python)
                .arg("-I")
                .arg(&script)
                .args(args)
                .current_dir(&dir)
                .output()
                .map_err(|e| {
                    error!(
                        ErrorCode::LatlasValidatorRejected,
                        self.script.display().to_string(),
                        e.to_string()
                    )
                })?;
            Ok(Run {
                stdout: String::from_utf8_lossy(&o.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&o.stderr).into_owned(),
                code: o.status.code().unwrap_or(-1),
            })
        })();
        std::fs::remove_dir_all(&dir).map_err(|e| io(&dir, &e))?;
        out
    }
}
