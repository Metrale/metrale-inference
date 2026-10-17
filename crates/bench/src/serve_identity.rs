// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: A server's identity: the pure digests and wire shape live in
//! `metrale-serve-identity`, reading this process in `metrale-host-probe`;
//! both are re-exported so bench paths are unchanged (SSOT: the server
//! publishes and the bench checks the same functions).
//!
//! Owner: bench (serve identity).
//! Invariants: none beyond the re-exported functions.

pub use metrale_host_probe::serve_identity::{file_sha256, this_process};
pub use metrale_serve_identity::identity::{ServeIdentity, argv_fingerprint, env_is_unknown};
