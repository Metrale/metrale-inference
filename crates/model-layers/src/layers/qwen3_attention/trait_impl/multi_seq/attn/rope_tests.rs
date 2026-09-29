// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: Batched-decode (and batch-verify) RoPE honours the layer's positional
//! encoding, on the per-row loop (1 row) and the strided launch (4 rows).
//!
//! Owner: model-layers (qwen3 attention).
//! Invariants: none beyond the types.

use metrale_config::AttnPositionEncoding;

use super::super::ctx::MultiSeqCtx;
use crate::layers::qwen3_attention::rope_site_fixture::Rig;

fn rope_launches(encoding: AttnPositionEncoding, rows: usize) -> usize {
    let rig = Rig::new(encoding);
    let layer = rig.layer();
    let fwd = rig.fwd();
    let c = MultiSeqCtx::new(
        &layer,
        &fwd,
        rig.buffers.hidden_states(),
        rig.buffers.residual(),
        rows,
        16,
        0,
    );
    let from = rig.gpu.launch_count();
    layer.ms_phase_rope(&c, rig.meta(rows)).unwrap();
    rig.rope_launches_since(from)
}

#[test]
fn batched_decode_applies_no_rope_without_position_encoding() {
    for rows in [1, 4] {
        assert_eq!(
            rope_launches(AttnPositionEncoding::None, rows),
            0,
            "{rows} rows"
        );
    }
}

/// 2026-09-29: Path B: one launch per row on the loop, one strided launch for 4 rows.
#[test]
fn batched_decode_applies_rope_under_rope() {
    assert_eq!(rope_launches(AttnPositionEncoding::Rope, 1), 1);
    assert_eq!(rope_launches(AttnPositionEncoding::Rope, 4), 1);
}
