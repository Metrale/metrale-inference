// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: Process hosts of the metrale pack (the generic ones are latlas-host-native's):
//! [`ledger_py::LedgerPy`] runs a campaign's own `ledger.py` on a private copy, [`lkb`] lists the
//! LKB checkout's files and computes its facts (`kernel_perf.py`, git dates), [`record`] is the
//! ledger's validator of record. The binary (metrale-kg-mcp) is the only place these are chosen.
//!
//! Owner: metrale-io-host-kg.
//! Invariants:
//! - A credential-named file is never opened: it is listed by name only.
//! - Every failure is a registry error (`LatlasSource`, `LatlasValidatorRejected`, ...).
#![forbid(unsafe_code)]

pub mod ledger_py;
pub mod lkb;
pub mod record;
