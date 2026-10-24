// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: `/v1/embeddings` work on the scheduler thread. Each request arrives as one
//! [`EmbedJob`] (its inputs already tokenized with the role prompt); every job drained in a
//! tick runs as one `Model::embed_batch` call, which packs all their inputs back to back, and
//! each job gets its own slice of the result. When the combined call fails, the jobs run one
//! by one, so one bad request cannot fail the others in its tick.
//!
//! Owner: scheduler.
//! Invariants:
//! - Every job gets exactly one reply.

use metrale_model_engine::traits::Model;

/// 2026-10-24: Why a job failed: the registry code when the error carried one (the handler
/// maps it to a status without reading text), and the rendered text.
#[derive(Debug, Clone)]
pub struct EmbedFailure {
    pub code: Option<metrale_error::ErrorCode>,
    pub text: String,
}

impl EmbedFailure {
    fn of(e: &anyhow::Error) -> Self {
        let code = e.downcast_ref::<metrale_error::Error>().map(|x| x.code());
        let text = match code {
            Some(c) => format!("[{:#06x} {}] {e:#}", c.as_u16(), c.name()),
            None => format!("{e:#}"),
        };
        Self { code, text }
    }
    fn uncoded(text: String) -> Self {
        Self { code: None, text }
    }
}

/// 2026-10-24: One `/v1/embeddings` request: token ids per input, and where the vectors go
/// (`[inputs.len() * dim]` FP32 in input order, or why not).
pub struct EmbedJob {
    pub inputs: Vec<Vec<u32>>,
    pub reply: tokio::sync::oneshot::Sender<Result<Vec<f32>, EmbedFailure>>,
}

/// 2026-10-24: Run every job of this tick and answer each.
pub(crate) fn run_embed_jobs(model: &dyn Model, jobs: Vec<EmbedJob>) {
    if jobs.is_empty() {
        return;
    }
    let Some(dim) = model.embedding_dim() else {
        for j in jobs {
            let _ = j.reply.send(Err(EmbedFailure::uncoded(
                "this model does not compute embeddings".into(),
            )));
        }
        return;
    };
    let all: Vec<Vec<u32>> = jobs.iter().flat_map(|j| j.inputs.iter().cloned()).collect();
    match model.embed_batch(&all) {
        Ok(v) if v.len() == all.len() * dim => {
            let mut off = 0;
            for j in jobs {
                let n = j.inputs.len() * dim;
                let _ = j.reply.send(Ok(v[off..off + n].to_vec()));
                off += n;
            }
        }
        Ok(v) => {
            let msg = format!(
                "embed_batch returned {} values for {} inputs of width {dim}",
                v.len(),
                all.len()
            );
            for j in jobs {
                let _ = j.reply.send(Err(EmbedFailure::uncoded(msg.clone())));
            }
        }
        Err(e) if jobs.len() > 1 => {
            tracing::warn!(
                "embedding batch of {} jobs failed ({e:#}); retrying each",
                jobs.len()
            );
            for j in jobs {
                let r = model
                    .embed_batch(&j.inputs)
                    .map_err(|e| EmbedFailure::of(&e));
                let _ = j.reply.send(r);
            }
        }
        Err(e) => {
            let f = EmbedFailure::of(&e);
            for j in jobs {
                let _ = j.reply.send(Err(f.clone()));
            }
        }
    }
}

#[cfg(test)]
#[path = "embed_jobs_tests.rs"]
mod tests;
