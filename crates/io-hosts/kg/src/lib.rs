// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: Process hosts of the kernel knowledge graph (metrale-kg-derive's ports and inputs):
//! [`fs::FsSources`] reads and stamps the configured sources, [`ledger_py::LedgerPy`] runs a
//! campaign's own `ledger.py` on a private copy. The binary (metrale-kg-mcp) is the only place
//! these are chosen.
//!
//! Owner: metrale-io-host-kg.
//! Invariants:
//! - A credential-named file is never opened: it is listed by name only.
//! - Every failure is a registry error (`KgSource`, `KgValidatorRejected`, `KgRecordWrite`, ...).
#![forbid(unsafe_code)]

pub mod fs;
pub mod ledger_py;
pub mod settings;
