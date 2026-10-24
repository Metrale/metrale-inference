// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: The served embedding model: a bidirectional Llama text tower whose final-norm
//! hidden states are mean-pooled into one FP32 vector per input (today
//! `nvidia/llama-nemotron-embed-vl-1b-v2`, `model_type = "llama_nemotron_vl"`).
//!
//! `EmbedGpuModel` implements [`crate::traits::Model`], so the checkpoint is served by the
//! same `met serve`, scheduler and API as the generative models: the scheduler hands each
//! tick's `/v1/embeddings` inputs to [`crate::traits::ModelForward::embed_batch`], which packs
//! them back to back (no padding) into forwards of at most `max_pack_tokens` rows. It is
//! built, like NLLB, by `factory::build_model` before the decoder-only loaders run; every
//! kernel it launches is an LKB family the other models run, at its own parameter point.
//!
//! Owner: model-engine (embedding model).
//! Invariants:
//! - The model never generates: every decode/prefill entry point refuses with
//!   `GenerateNotSupported`.
//! - `embed_batch` returns vectors in input order whatever the packing.

#[cfg(feature = "cuda")]
mod engine;
#[cfg(feature = "cuda")]
mod forward;
#[cfg(feature = "cuda")]
mod kernels;
#[cfg(feature = "cuda")]
mod model_impl;
mod pack;
#[cfg(feature = "cuda")]
mod weights;

#[cfg(feature = "cuda")]
pub use engine::EmbedGpuModel;
pub use pack::{plan_packs, validate_inputs};
