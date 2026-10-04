// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: The `--prompt-lookup-*` rules of `validate_serve_args`.
//!
//! Owner: server CLI.
//! Invariants: none beyond the types.

use super::super::ServeArgs;
use super::violation::Violation;

pub(super) fn check(args: &ServeArgs, v: &mut Vec<Violation>) {
    // 2026-10-02: Prompt lookup takes MTP's round when it matches and leaves
    // it to MTP otherwise; with no MTP there is no round to take.
    if args.prompt_lookup.prompt_lookup_decoding && !args.speculative {
        v.push(Violation::new(
            "--prompt-lookup-decoding is set without --speculative.",
            "prompt-lookup copies are verified in the MTP speculative step, in place of \
             the drafter's chain; without MTP that step never runs, so the flag would do \
             nothing.",
            "add --speculative, or drop --prompt-lookup-decoding.",
        ));
    }

    // 2026-10-04: The cross-request cache carries one request's tokens into
    // another's drafts, the kind of channel `--hermetic` closes.
    if args.hermetic && args.prompt_lookup.prompt_lookup_shared_cache_mb > 0 {
        v.push(Violation::new(
            "--hermetic with --prompt-lookup-shared-cache-mb",
            "--hermetic closes cross-request channels, and the shared prompt-lookup cache \
             is one: what earlier requests held decides which drafts a later request \
             verifies, and so how fast it runs",
            "drop --prompt-lookup-shared-cache-mb (or set it to 0), or drop --hermetic if \
             you meant to measure WITH the cache",
        ));
    }
}
