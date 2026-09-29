// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: Single-token decode RoPE honours the layer's positional encoding.
//!
//! Owner: model-layers attention decode.
//! Invariants: none beyond the types.

use metrale_config::AttnPositionEncoding;
use metrale_gpu_runtime::gpu::GpuBackend;

use crate::layers::qwen3_attention::rope_site_fixture::{HEAD_DIM, Rig};

fn rope_launches(encoding: AttnPositionEncoding) -> usize {
    let rig = Rig::new(encoding);
    let layer = rig.layer();
    let fwd = rig.fwd();
    let meta = rig.meta(1);
    let q = rig.gpu.alloc(HEAD_DIM * 2).unwrap();
    let k = rig.gpu.alloc(HEAD_DIM * 2).unwrap();
    let from = rig.gpu.launch_count();
    layer
        .attention_forward_rope(
            &fwd,
            &meta,
            q,
            k,
            1,
            1,
            HEAD_DIM as u32,
            fwd.config.rotary_dim() as u32,
            false,
            0,
        )
        .unwrap();
    rig.rope_launches_since(from)
}

#[test]
fn decode_applies_no_rope_without_position_encoding() {
    assert_eq!(rope_launches(AttnPositionEncoding::None), 0);
}

/// 2026-09-29: Path B: the same layer under RoPE launches it once.
#[test]
fn decode_applies_rope_under_rope() {
    assert_eq!(rope_launches(AttnPositionEncoding::Rope), 1);
}
