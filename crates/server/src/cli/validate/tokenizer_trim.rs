// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The `--tokenizer-trim-min-tokens` rules of `validate_serve_args`.
//!
//! Owner: server CLI.
//! Invariants: none beyond the types.

use super::super::ServeArgs;
use super::violation::Violation;
use crate::tokenizer::HeapTrimPolicy;

pub(super) fn check(args: &ServeArgs, v: &mut Vec<Violation>) {
    let min_tokens = args.chat_surface.tokenizer_trim_min_tokens;
    // 2026-10-17: No prompt is longer than --max-seq-len, so the trim could never run.
    if min_tokens > args.max_seq_len {
        v.push(Violation::new(
            format!(
                "--tokenizer-trim-min-tokens {min_tokens} is above --max-seq-len {}.",
                args.max_seq_len
            ),
            "a prompt longer than --max-seq-len is refused, so no encode could reach the \
             threshold and the flag would do nothing.",
            "use a threshold at or below --max-seq-len, or 0 to never trim.",
        ));
    }
    // 2026-10-17: The same refusal the tokenizer load gives (`HeapTrimPolicy::process_heap`).
    if let Err(why) = HeapTrimPolicy::process_heap(min_tokens) {
        v.push(Violation::new(
            format!("--tokenizer-trim-min-tokens {min_tokens} on this build."),
            why,
            "drop --tokenizer-trim-min-tokens (or set it to 0).",
        ));
    }
}
