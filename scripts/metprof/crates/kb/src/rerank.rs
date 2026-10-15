// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the reranker port (ADDENDUM-EMBEDDINGS E1 §4). A reranker
//! re-scores first-stage candidates against the query. There is no default:
//! `embedding.toml` names one, and `none` (keep the first-stage scores) is
//! today's only setting. A `/v1/rerank`-style client plugs in behind the
//! same trait, outside this crate.

use crate::config::RerankerConfig;
use crate::{Result, bail};

#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub id: u64,
    pub text: String,
    /// The first-stage score (cosine, graph proximity, ...).
    pub score: f32,
}

pub trait Reranker {
    /// The setting that selects this reranker.
    fn name(&self) -> &'static str;
    /// The candidates re-scored, best first (ties keep their input order).
    fn rerank(&self, query: &str, candidates: Vec<Candidate>) -> Result<Vec<Candidate>>;
}

/// `none`: the first-stage scores stand.
pub struct NoRerank;

impl Reranker for NoRerank {
    fn name(&self) -> &'static str {
        "none"
    }

    fn rerank(&self, _query: &str, mut candidates: Vec<Candidate>) -> Result<Vec<Candidate>> {
        candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
        Ok(candidates)
    }
}

/// The reranker `embedding.toml` selects. This crate carries only `none`; an
/// `openai_compatible` reranker is built by the adapter that owns a transport.
pub fn select(config: &RerankerConfig) -> Result<Box<dyn Reranker>> {
    match config {
        RerankerConfig::None => Ok(Box::new(NoRerank)),
        RerankerConfig::OpenAiCompatible { .. } => bail!(
            "reranker \"openai_compatible\" needs a transport adapter; this crate selects only \"none\""
        ),
    }
}
