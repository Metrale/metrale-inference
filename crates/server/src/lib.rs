// SPDX-License-Identifier: MIT OR Apache-2.0

#![deny(warnings)]
#![deny(clippy::all)]
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::large_enum_variant)]
#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::doc_overindented_list_items)]

//! 2026-09-26: Library target of met.
//!
//! 2026-10-24: Every module of `met`. The binary (crates/met) is a `main` that hands
//! [`serve_main`] the kernel registry and kernel tree it embeds ([`embedded::Embedded`]), so
//! this crate does not depend on the crates that compile or pack the kernels, and a kernel
//! edit relinks the binary without recompiling it.
//!
//! Owner: server.
//! Invariants: none beyond the types.

mod adaptive_sampler;
mod anthropic;
mod api;
pub mod auth;
mod citation;
mod citation_structured;
mod cli;
mod conversation_store;
mod disk_guard;
pub mod env_config;
mod error_hints;
pub mod grammar;
mod halluc_probe;
mod hint_injector;
mod identity;
mod ids;
mod ir;
#[cfg(test)]
mod lever_equivalence;
mod lever_value;
mod llmlingua;
mod lookback_lens;
mod loop_detector;
mod loop_simhash;
mod lqer;
mod main_modules;
pub mod metrics;
mod model_download;
mod model_resolver;
mod moe_quality;
mod openai;
pub mod rate_limiter;
pub mod reasoning_parser;
pub mod recipe;
pub mod refusal;
mod request_dumper;
mod response_store;
mod retrieval_heads;
mod scheduler;
mod scheduling_policy;
mod session_manager;
mod symbol_trie;
pub mod tokenizer;
mod tool_arg_dedup;
pub mod tool_parser;
mod tool_rag;
mod tscg;
pub mod tui;

pub mod embedded;
mod serve_main;

pub use serve_main::serve_main;

pub(crate) use crate::main_modules::AppState;

pub type ModelBehavior = metrale_kernel_set::ModelBehavior;
