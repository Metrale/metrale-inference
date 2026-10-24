// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The chat-surface serve flags (flattened last into `ServeServiceArgs`):
//! the chat template file, the per-request media caps and uncapped thinking.
//! 2026-10-17: Also the tokenizer heap trim after a long prompt.
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

    /// Most images one chat request may carry, counted over all its messages. A
    /// request with more is refused with a 400 before any image is fetched or
    /// decoded. Unset: no cap.
    #[arg(long, value_name = "N")]
    pub limit_images_per_prompt: Option<usize>,

    /// Most videos one chat request may carry, counted over all its messages; 0
    /// refuses any video. A request with more is refused with a 400 before any
    /// video is decoded. Unset: no cap.
    #[arg(long, value_name = "N")]
    pub limit_videos_per_prompt: Option<usize>,

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

    /// Give the heap the tokenizer freed back to the operating system (glibc
    /// `malloc_trim`) after encoding any prompt of N tokens or more. Encoding keeps
    /// per-token offsets and masks beside the ids, and glibc holds on to those pages
    /// once they are freed, so a very long prompt leaves host memory resident that a
    /// unified-memory GPU shares. The trim runs after the ids are copied out (they are
    /// unchanged) and takes longer the larger the prompt. 0 (the default) never trims.
    /// Refused above --max-seq-len, where no prompt could reach it, and on a build
    /// whose allocator is not glibc's.
    #[arg(long, default_value_t = 0, value_name = "N")]
    pub tokenizer_trim_min_tokens: usize,

    /// Memoize the token ids of long prompt pieces (the text between two special tokens) in
    /// up to this many MiB. A tokenizer encodes each such piece on its own, so a
    /// conversation re-sent with one more turn re-encodes only its new pieces; the ids are
    /// exactly those of a whole encode. Only prompts of 32 KiB or more and pieces of 4 KiB or
    /// more use it. 0 turns it off.
    #[arg(long, default_value_t = 64, value_name = "MIB")]
    pub tokenizer_memo_mb: usize,
}
