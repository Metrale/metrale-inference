// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The chat-surface serve flags (flattened last into `ServeServiceArgs`):
//! the chat template file.
//!
//! Owner: server CLI.
//! Invariants: the `///` text on the struct's fields is the `--help` output and
//! carries no date.

use clap::Args;

#[derive(Args, Debug, Clone, PartialEq)]
pub struct ServeChatSurfaceArgs {
    /// Jinja chat template file to render every chat request with, in place of
    /// `jinja-templates/{model_type}.jinja` (and its `openai/` variant) and of the
    /// checkpoint's own template. A file that cannot be read fails the start.
    /// Precedence (highest wins): this flag → `jinja-templates/{model_type}.jinja`
    /// → the checkpoint's template → ChatML. Refused together with
    /// `--disable-template-overrides`.
    #[arg(
        long,
        value_name = "FILE",
        conflicts_with = "disable_template_overrides"
    )]
    pub chat_template: Option<std::path::PathBuf>,
}
