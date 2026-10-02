// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The prompt-lookup settings read from `ServeArgs` (`--prompt-lookup-*`).
//!
//! Owner: server CLI.
//! Invariants: none beyond the types.

use super::serve_args::ServeArgs;

impl ServeArgs {
    /// 2026-10-02: The prompt-lookup settings, `None` without
    /// `--prompt-lookup-decoding`.
    pub fn prompt_lookup_config(
        &self,
    ) -> Option<metrale_speculative::prompt_lookup::PromptLookupConfig> {
        self.prompt_lookup_decoding.then(|| {
            metrale_speculative::prompt_lookup::PromptLookupConfig {
                ngram: self.prompt_lookup_ngram as usize,
                max_drafts: self.prompt_lookup_max_drafts as usize,
                max_seqs: self.prompt_lookup_max_seqs as usize,
            }
        })
    }

    /// 2026-10-02: Drafts the widest speculative verify can carry: the drafter's
    /// [`Self::resolved_num_drafts`], raised to `--prompt-lookup-max-drafts` when prompt
    /// lookup is on. It sizes the per-sequence verify rows the prefill budget reserves; the
    /// SSM pools take the copy depth from the published copy tier (`ssm_reserve::copy_tier`).
    pub fn verify_pool_drafts(&self) -> usize {
        let copy = self.prompt_lookup_config().map_or(0, |pl| pl.max_drafts);
        self.resolved_num_drafts().max(copy)
    }
}
