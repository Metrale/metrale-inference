// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: Paged / chunked prefill RoPE honours the layer's positional encoding, and
//! skipping it still writes the KV cache.
//!
//! Owner: model-layers (attention).
//! Invariants: none beyond the types.

use metrale_cache::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use metrale_config::AttnPositionEncoding;
use metrale_gpu_runtime::gpu::GpuBackend;

use crate::layers::qwen3_attention::rope_site_fixture::{HEAD_DIM, Rig};

const TOKENS: usize = 4;

/// 2026-09-29: (RoPE launches, all launches) of one paged prefill RoPE + cache write.
fn launches(encoding: AttnPositionEncoding) -> (usize, usize) {
    let rig = Rig::new(encoding);
    let layer = rig.layer();
    let fwd = rig.fwd();
    let meta = rig.meta(TOKENS);
    let mut kv = PagedKvCache::new(
        KvCacheConfig {
            block_size: 16,
            num_kv_heads: 1,
            head_dim: HEAD_DIM,
            num_layers: 1,
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        },
        4,
        &rig.gpu,
    )
    .unwrap();
    let q = rig.gpu.alloc(TOKENS * HEAD_DIM * 2).unwrap();
    let k = rig.gpu.alloc(TOKENS * HEAD_DIM * 2).unwrap();
    let v = rig.gpu.alloc(TOKENS * HEAD_DIM * 2).unwrap();
    let from = rig.gpu.launch_count();
    layer
        .prefill_paged_rope_cache_write(
            &mut kv,
            q,
            k,
            v,
            None,
            meta.positions,
            meta.positions_h,
            meta.positions_w,
            meta.slot,
            TOKENS as u32,
            1,
            1,
            HEAD_DIM as u32,
            16,
            TOKENS,
            0,
            HEAD_DIM,
            2,
            &fwd,
            0,
        )
        .unwrap();
    (rig.rope_launches_since(from), rig.gpu.launch_count() - from)
}

#[test]
fn paged_prefill_applies_no_rope_but_still_writes_the_cache() {
    let (rope, all) = launches(AttnPositionEncoding::None);
    assert_eq!(rope, 0);
    assert!(all > 0, "the KV-cache write must still run");
}

/// 2026-09-29: Path B: the same prefill under RoPE launches it once, plus the cache write.
#[test]
fn paged_prefill_applies_rope_under_rope() {
    let (rope, all) = launches(AttnPositionEncoding::Rope);
    assert_eq!(rope, 1);
    assert!(all > rope, "the KV-cache write must run after RoPE");
}
