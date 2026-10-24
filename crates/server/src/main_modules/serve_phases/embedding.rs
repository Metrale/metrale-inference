// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: The serve phases an embedding model adds: its role prompts, read once from the
//! checkpoint's `config_sentence_transformers.json` before anything else reads the config, and
//! the `/v1/embeddings` endpoint (job channel, semantics, width) the router and scheduler
//! share.
//!
//! Owner: server startup (`met serve`).
//! Invariants:
//! - A config with an embedding spec boots only with its prompts and a model that reports an
//!   embedding width; a generative config never gets an endpoint.

use std::path::Path;

use anyhow::{Context, Result, bail};
use metrale_config::ModelConfig;
use tokio::sync::mpsc;

use crate::main_modules::app_state::EmbedEndpoint;
use crate::scheduler::EmbedJob;

/// 2026-10-24: The prompts file of a sentence-transformers checkpoint.
const PROMPTS_FILE: &str = "config_sentence_transformers.json";

/// 2026-10-24: Fill the embedding spec's prompts from `model_dir`; a no-op for a generative
/// config, and an error when an embedding checkpoint lacks the file.
pub(crate) fn load_embedding_prompts(model_dir: &Path, config: &mut ModelConfig) -> Result<()> {
    let Some(spec) = config.embedding.as_mut() else {
        return Ok(());
    };
    let path = model_dir.join(PROMPTS_FILE);
    let json = std::fs::read_to_string(&path).with_context(|| {
        format!(
            "embedding checkpoint has no {} (its role prompts are model semantics)",
            path.display()
        )
    })?;
    let prompts = metrale_config::embedding::parse_sentence_transformers_prompts(&json)?;
    tracing::info!(
        "Embedding prompts: query {:?}, document {:?} (from {PROMPTS_FILE})",
        prompts.query,
        prompts.document
    );
    spec.prompts = Some(prompts);
    Ok(())
}

/// 2026-10-24: The endpoint and the scheduler's half of its channel (`cap` jobs queued), for
/// an embedding model; `(None, None)` for a generative one. `dim` is what the built model
/// reports, and it must agree with the config.
pub(crate) fn embed_endpoint(
    config: &ModelConfig,
    dim: Option<usize>,
    cap: usize,
) -> Result<(Option<EmbedEndpoint>, Option<mpsc::Receiver<EmbedJob>>)> {
    match (&config.embedding, dim) {
        (None, None) => Ok((None, None)),
        (Some(spec), Some(dim)) => {
            if spec.prompts.is_none() {
                bail!("embedding prompts were never loaded");
            }
            let (tx, rx) = mpsc::channel::<EmbedJob>(cap.max(1));
            Ok((
                Some(EmbedEndpoint {
                    tx,
                    spec: spec.clone(),
                    bos_token_id: config.bos_token_id,
                    dim,
                }),
                Some(rx),
            ))
        }
        (Some(_), None) => bail!("the config is an embedding model but the built model is not"),
        (None, Some(_)) => {
            bail!("the built model computes embeddings but the config is generative")
        }
    }
}
