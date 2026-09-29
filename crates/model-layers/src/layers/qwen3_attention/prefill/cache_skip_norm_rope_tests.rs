// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: Non-paged prefill RoPE honours the layer's positional encoding.
//!
//! Owner: model-layers (attention).
//! Invariants: none beyond the types.

use metrale_config::AttnPositionEncoding;
use metrale_gpu_runtime::gpu::GpuBackend;

use crate::layers::qwen3_attention::rope_site_fixture::{HEAD_DIM, Rig};

const TOKENS: usize = 4;

fn rope_launches(encoding: AttnPositionEncoding) -> usize {
    let rig = Rig::new(encoding);
    let layer = rig.layer();
    let fwd = rig.fwd();
    let meta = rig.meta(TOKENS);
    let q = rig.gpu.alloc(TOKENS * HEAD_DIM * 2).unwrap();
    let k = rig.gpu.alloc(TOKENS * HEAD_DIM * 2).unwrap();
    let from = rig.gpu.launch_count();
    layer
        .cache_skip_rope(
            &fwd,
            false,
            q,
            k,
            meta.positions,
            meta.positions_h,
            meta.positions_w,
            TOKENS as u32,
            1,
            1,
            HEAD_DIM as u32,
            TOKENS,
            2,
            0,
        )
        .unwrap();
    rig.rope_launches_since(from)
}

#[test]
fn prefill_applies_no_rope_without_position_encoding() {
    assert_eq!(rope_launches(AttnPositionEncoding::None), 0);
}

/// 2026-09-29: Path B: the same prefill under RoPE launches it once for all tokens.
#[test]
fn prefill_applies_rope_under_rope() {
    assert_eq!(rope_launches(AttnPositionEncoding::Rope), 1);
}
