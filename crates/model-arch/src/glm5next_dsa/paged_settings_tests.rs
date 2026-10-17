// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `text_stack_indexer_cache` from the resolved settings against the reader it
//! replaced (`metrale_config::glm_dsa_indexer_paged`: flat only at `METRALE_GLM_DSA_INDEXER_FLAT=1`).
//!
//! Owner: model-arch (glm5next_dsa).
//! Invariants: the probes that read differently are exactly the listed behaviour changes.

use super::{IndexerCache, text_stack_indexer_cache};
use metrale_model_layers::lever_equivalence::{Old, Outcome, check};

#[test]
fn indexer_flat_reads_as_before_except_the_switch_spellings() {
    let changes = check(
        "METRALE_GLM_DSA_INDEXER_FLAT",
        Old::OptIn,
        |p| {
            if p == Some("1") {
                IndexerCache::Flat
            } else {
                IndexerCache::Paged
            }
        },
        text_stack_indexer_cache,
    );
    assert_eq!(
        changes,
        [
            (
                "METRALE_GLM_DSA_INDEXER_FLAT",
                Some("true"),
                Outcome::Changed
            ),
            (
                "METRALE_GLM_DSA_INDEXER_FLAT",
                Some("garbage"),
                Outcome::Rejected
            ),
        ]
    );
}
