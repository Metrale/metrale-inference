// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the kernel knowledge base (`metrale.kkb/1`) as metprof writes
//! it. Everything here is pure except the store adapters: `schema` (records,
//! kinds, relations), `ids` (natural-key point ids), `canon` (canonical JSON
//! and `content_sha256`), `config` (`embedding.toml`), `embed` (the embedding
//! port and the hashed space), `rerank` (the reranker port), `port` (what
//! metprof hands the sink) and `derive` (a profile run and its kernel
//! samples, edges included).

pub mod canon;
pub mod config;
pub mod derive;
pub mod embed;
pub mod error;
pub mod ids;
pub mod port;
pub mod rerank;
pub mod schema;

pub use error::{Error, Result};
