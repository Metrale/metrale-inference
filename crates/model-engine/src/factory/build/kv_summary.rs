// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The `--high-speed-swap` KV summary log line: attention-layer count per KV dtype label.
//!
//! Owner: model-engine factory.
//! Invariants: the labels are the registry's (`KvPlanLabel`); mixed K/V layers count under
//! their K codec.

use metrale_cache::kv_cache::KvCacheConfig;
use metrale_kv_codec::KvPlanLabel;

pub(super) fn log_hss_kv_summary(kv_config: &KvCacheConfig) {
    let label = if kv_config.layer_dtypes.is_empty() {
        let (k, _) = kv_config.dtype.pair();
        KvPlanLabel::by_k_codec(std::iter::repeat_n(k, kv_config.num_layers))
    } else {
        KvPlanLabel::by_k_codec(kv_config.layer_dtypes.iter().map(|d| d.pair().0))
    };
    tracing::info!(
        "--high-speed-swap KV: {} attn layers ({}); HBM-shrink applies to all \
         (Phase 6.2.c proper — host dequant for FP8/NVFP4/Turbo3/Turbo4/Turbo8; \
         predictor scoring uses LRU for non-BF16 layers)",
        label.layers(),
        label
    );
}
