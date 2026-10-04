// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The DIAG lines both forwards' profiles log.
//!
//! Owner: model-engine (FEATURES workstream).
//! Invariants: none beyond the types.

use metrale_config::LayerType;

use super::{diag_embed_line, diag_layer_line, diag_top5_line};

fn bf16(xs: &[f32]) -> Vec<u8> {
    xs.iter()
        .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

#[test]
fn the_top5_line_lists_the_largest_logits_first_and_ties_by_lower_index() {
    let logits = bf16(&[1.0, 7.0, -3.0, 7.0, 2.0, 0.5, 9.0, 2.0]);
    assert_eq!(
        diag_top5_line(12, &logits),
        "DIAG tok=12 top5_logits: [(6, 9.0), (1, 7.0), (3, 7.0), (4, 2.0), (7, 2.0)]"
    );
}

#[test]
fn the_readback_lines_show_four_values_at_four_decimals() {
    let vals = [0.5, -1.25, 2.0, 3.125, 99.0, 99.0, 99.0, 99.0];
    assert_eq!(
        diag_embed_line(3, 1.5, &vals),
        "DIAG tok=3 after_embed (FP32): norm=1.5000 vals=[0.5000, -1.2500, 2.0000, 3.1250]"
    );
    assert_eq!(
        diag_layer_line(3, 7, LayerType::FullAttention, 2.0, &vals),
        "DIAG tok=3 after_L7 (FullAttention) [FP32]: norm=2.0000 vals=[0.5000, -1.2500, \
         2.0000, 3.1250]"
    );
}
