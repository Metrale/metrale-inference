// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: Tests for `attention_route`: a batched first chunk is refused exactly
//! when the model engine's admission rule would not have admitted its wave.
//!
//! Owner: model-layers (qwen3 attention).
//! Invariants: none beyond the types.

use super::*;
use crate::layers::ops::batched_chunk_zero_admitted;

#[test]
fn a_varlen_admitted_batched_first_chunk_runs_paged() {
    // 2026-09-28: `--prefill-varlen-batch` without `--prefill-codispatch`: the wave is
    // admitted, so the layer must not refuse it (it did before 2026-09-28, and every
    // stream of the wave failed).
    assert!(batched_chunk_zero_admitted(false, true));
    assert_eq!(attention_route(true, 0, false, true), AttnRoute::Paged);
}

#[test]
fn the_route_refuses_exactly_what_admission_refuses() {
    for lever in [false, true] {
        for varlen in [false, true] {
            let admitted = batched_chunk_zero_admitted(lever, varlen);
            let route = attention_route(true, 0, lever, varlen);
            assert_eq!(
                route == AttnRoute::Refuse,
                !admitted,
                "lever={lever} varlen={varlen}: route {route:?}, admitted {admitted}"
            );
        }
    }
}

#[test]
fn unbatched_and_later_chunks_are_unaffected() {
    for lever in [false, true] {
        for varlen in [false, true] {
            assert_eq!(
                attention_route(false, 0, lever, varlen),
                AttnRoute::Contiguous
            );
            assert_eq!(attention_route(false, 512, lever, varlen), AttnRoute::Paged);
            assert_eq!(attention_route(true, 512, lever, varlen), AttnRoute::Paged);
        }
    }
}
