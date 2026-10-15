// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the kernel knowledge base (`metrale.kkb/1`) as metprof writes
//! it. Everything here is pure except the store adapters: `schema` (records,
//! kinds, relations), `ids` (natural-key point ids), `canon` (canonical JSON
//! and `content_sha256`), `config` (`embedding.toml`), `kb_toml` (`kb.toml`), `time`, `embed` (the embedding
//! port and the hashed space), `rerank` (the reranker port), `port` (what
//! metprof hands the sink) and `derive` (a profile run and its kernel
//! samples, edges included); on the host also `spool`, `lease`, `lattice`,
//! `export`, `seal` and `sink` (the write path of DESIGN §3.1) over `atomic`,
//! and `local` (the KB a `kb.toml` names, behind the `KbOpen` port).

pub mod canon;
pub mod config;
pub mod derive;
pub mod embed;
pub mod error;
pub mod ids;
pub mod kb_toml;
pub mod port;
pub mod rerank;
pub mod schema;
pub mod time;

// The adapters: files, the lease and the durable LatticeDB store (host only).
#[cfg(not(target_arch = "wasm32"))]
pub mod atomic;
#[cfg(not(target_arch = "wasm32"))]
pub mod export;
#[cfg(not(target_arch = "wasm32"))]
pub mod lattice;
#[cfg(not(target_arch = "wasm32"))]
pub mod lease;
#[cfg(not(target_arch = "wasm32"))]
pub mod local;
#[cfg(not(target_arch = "wasm32"))]
pub mod seal;
#[cfg(not(target_arch = "wasm32"))]
pub mod sink;
#[cfg(not(target_arch = "wasm32"))]
pub mod spool;

pub use error::{Error, Result};
