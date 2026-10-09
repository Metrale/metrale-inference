// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The chat-surface serve flags (flattened last into `ServeServiceArgs`):
//! the chat template file and uncapped thinking.
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

    /// Arm no thinking budget of the server's own: reasoning runs until the model
    /// closes it or `max_tokens` ends the response. Effort levels, the
    /// `--max-thinking-budget` / MODEL.toml default, a block the chat template
    /// opens, and the 90%-of-max_tokens clamp then set no budget; an explicit token
    /// budget (`thinking.budget_tokens`, `thinking_token_budget`, a `thinking_budget`
    /// kwarg from the request or `--default-chat-template-kwargs`) still applies. A
    /// block the model opens on its own later in the answer keeps the
    /// `--max-thinking-budget` budget. Refused with `--disable-thinking`.
    #[arg(long, default_value_t = false, conflicts_with = "disable_thinking")]
    pub uncapped_thinking: bool,
}
