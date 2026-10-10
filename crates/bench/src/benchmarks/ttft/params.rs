// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The TTFT gates' parameter schema, moved out of `ttft.rs` (500-line cap) when the
//! `tokenizer` parameter was added.
//!
//! Owner: bench, ttft.
//! Invariants:
//! - `tokenizer` defaults to `token_prompt::NO_TOKENIZER`, which keeps every gate's committed
//!   prompts, so a record made without it measures what it measured before the parameter existed.

use super::{Mode, long_prompt, token_prompt};
use crate::params::{ParamKind, ParamSpec, ParamValue};

/// 2026-09-26: The schema of a synthetic (`fixture == false`) or high-ISL TTFT gate.
pub(super) fn parameters(fixture: bool, mode: Mode) -> Vec<ParamSpec> {
    let (repeats, median_limit, p90_limit) = if fixture {
        long_prompt::one_shot_defaults(mode)
    } else {
        (12, 3.0, 5.0)
    };
    let source = if fixture {
        long_prompt::source_parameters()
    } else {
        vec![ParamSpec::new(
            "prompt_lengths",
            "Prompt lengths",
            "Prompt sizes in tokens; one table row each.",
            ParamKind::IntList {
                min: 16,
                max: 131_072,
            },
            ParamValue::IntList(vec![256, 1024, 4096]),
        )]
    };
    source
        .into_iter()
        .chain([
            ParamSpec::new(
                "tokenizer",
                "Tokenizer",
                "The served model's tokenizer.json, or its checkpoint directory. Set, every user \
                 message is cut to exactly the requested token count of that tokenizer \
                 (prompt_lengths, or min_prompt_tokens for a high-ISL gate) and the server's \
                 count is checked on every sample. `none` keeps the committed prompts.",
                ParamKind::Text,
                ParamValue::Text(token_prompt::NO_TOKENIZER.to_string()),
            ),
            ParamSpec::new(
                "repeats",
                "Samples per length",
                "More samples narrow the median; each costs one request (two in warm mode).",
                ParamKind::Int { min: 1, max: 200 },
                ParamValue::Int(repeats),
            ),
            ParamSpec::new(
                "median_limit_pct",
                "Median limit",
                "Percent the median may rise over the baseline before this gate fails.",
                ParamKind::Float {
                    min: 0.0,
                    max: 100.0,
                },
                ParamValue::Float(median_limit),
            ),
            ParamSpec::new(
                "p90_limit_pct",
                "p90 limit",
                "Percent p90 may rise over the baseline before this gate fails.",
                ParamKind::Float {
                    min: 0.0,
                    max: 100.0,
                },
                ParamValue::Float(p90_limit),
            ),
            ParamSpec::new(
                "update_baseline",
                "Record as baseline",
                "Store this run's numbers as the new baseline. Turn off to compare without moving the bar.",
                ParamKind::Bool,
                ParamValue::Bool(true),
            ),
            ParamSpec::new(
                "request_timeout_s",
                "Request timeout",
                "Seconds before a single request is abandoned.",
                ParamKind::Int { min: 10, max: 3600 },
                ParamValue::Int(300),
            ),
        ])
        .collect()
}
