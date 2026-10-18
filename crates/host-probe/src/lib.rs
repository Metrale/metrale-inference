// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Host probes the serve path shares with the benchmark harness:
//! the artifact home ([`artifacts`]), the box fingerprint and state
//! ([`hardware`]), the HTTP/1.1 response wire ([`http_wire`]), the harness lever values
//! ([`levers`]) and this
//! process's identity ([`serve_identity`]). Split out of metrale-bench so
//! `met` builds without the `bench` feature and still serves `/hardware` and
//! `/serve-config` (OWAY: serve never depends on bench).
//!
//! Owner: host-probe.
//! Invariants:
//! - Nothing here decides a benchmark verdict; policy stays in metrale-bench.

pub mod artifacts;
pub mod hardware;
pub mod http_wire;
pub mod levers;
pub mod serve_identity;
