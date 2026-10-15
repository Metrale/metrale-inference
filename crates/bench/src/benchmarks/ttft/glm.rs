// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: The TTFT gates of GLM-5.3-Flash: cold and warm over the synthetic 256, 1024 and
//! 4096-token prompts, and cold and warm over the committed 32k-token prompt, once per served
//! context length (64k, 256k and 512k). Each is the existing driver under its own id, so its
//! bounds and baseline are read against GLM's history only and one gate id has one default
//! subject (`concurrency-sweep-moe` and `high-isl-ttft-cold-moe` are the precedent).
//!
//! Owner: bench, ttft.
//! Invariants:
//! - The context length is not a driver parameter: each entry in
//!   kernels/gb10/glm-5.3-flash/BENCH.toml pins it as the `max_model_len` serve override, and
//!   the id names it so records at different lengths never share a bound.

use super::{Mode, TtftGate};
use crate::benchmark::{BenchmarkDescriptor, ModelExpectation};
use crate::hardware::Sensitivity;
use crate::metadata::PluginMetadata;

const GLM: ModelExpectation = ModelExpectation {
    families: &["glm-5.3-flash"],
    note: "The -glm TTFT gates are defined on GLM-5.3-Flash at the context length their id \
           names; the gates without the suffix measure whatever they are pointed at.",
};

macro_rules! glm_ttft {
    ($desc:ident, $meta:ident, $ctor:ident, $mode:expr, $id:literal, $name:literal,
     $summary:literal, $hint:literal, $secs:literal, $detail:literal) => {
        const $meta: PluginMetadata = PluginMetadata::metrale($summary);
        pub const $desc: BenchmarkDescriptor = BenchmarkDescriptor {
            id: $id,
            name: $name,
            summary: $summary,
            detail: $detail,
            duration_hint: $hint,
            expected_secs: $secs,
            updated: "2026-10-15",
            needs_confirmation: false,
            intended_for: Some(GLM),
            threshold_params: &[],
            // 2026-10-15: Speed, as for the TTFT gates these reuse.
            sensitivity: Sensitivity::Speed,
            ctor: || Box::new(TtftGate::$ctor($mode, &$desc, &$meta)),
        };
    };
}

glm_ttft!(
    GLM_COLD_64K,
    GLM_COLD_64K_META,
    synthetic,
    Mode::Cold,
    "ttft-cold-glm-64k",
    "Cold TTFT Gate (GLM, 64k)",
    "Uncached prefill TTFT on GLM-5.3-Flash served at 64k context",
    "~1 min",
    60,
    "ttft-cold-gate on GLM-5.3-Flash served with max_model_len 65536: the same driver and \
     prompts, with one median per prompt length (median_ms_256, median_ms_1024, \
     median_ms_4096), each bounded on its own."
);
glm_ttft!(
    GLM_COLD_256K,
    GLM_COLD_256K_META,
    synthetic,
    Mode::Cold,
    "ttft-cold-glm-256k",
    "Cold TTFT Gate (GLM, 256k)",
    "Uncached prefill TTFT on GLM-5.3-Flash served at 256k context",
    "~1 min",
    60,
    "ttft-cold-gate on GLM-5.3-Flash served with max_model_len 262144, with one median per \
     prompt length, each bounded on its own."
);
glm_ttft!(
    GLM_COLD_512K,
    GLM_COLD_512K_META,
    synthetic,
    Mode::Cold,
    "ttft-cold-glm-512k",
    "Cold TTFT Gate (GLM, 512k)",
    "Uncached prefill TTFT on GLM-5.3-Flash served at 512k context",
    "~1 min",
    60,
    "ttft-cold-gate on GLM-5.3-Flash served with max_model_len 524288, with one median per \
     prompt length, each bounded on its own."
);
glm_ttft!(
    GLM_WARM_64K,
    GLM_WARM_64K_META,
    synthetic,
    Mode::Warm,
    "ttft-warm-glm-64k",
    "Warm TTFT Gate (GLM, 64k)",
    "Cached-prefix TTFT on GLM-5.3-Flash served at 64k context",
    "~1 min",
    60,
    "ttft-warm-gate on GLM-5.3-Flash served with max_model_len 65536: each prompt is primed \
     and re-sent, with one median per prompt length, each bounded on its own."
);
glm_ttft!(
    GLM_WARM_256K,
    GLM_WARM_256K_META,
    synthetic,
    Mode::Warm,
    "ttft-warm-glm-256k",
    "Warm TTFT Gate (GLM, 256k)",
    "Cached-prefix TTFT on GLM-5.3-Flash served at 256k context",
    "~1 min",
    60,
    "ttft-warm-gate on GLM-5.3-Flash served with max_model_len 262144, with one median per \
     prompt length, each bounded on its own."
);
glm_ttft!(
    GLM_WARM_512K,
    GLM_WARM_512K_META,
    synthetic,
    Mode::Warm,
    "ttft-warm-glm-512k",
    "Warm TTFT Gate (GLM, 512k)",
    "Cached-prefix TTFT on GLM-5.3-Flash served at 512k context",
    "~1 min",
    60,
    "ttft-warm-gate on GLM-5.3-Flash served with max_model_len 524288, with one median per \
     prompt length, each bounded on its own."
);
glm_ttft!(
    GLM_HIGH_ISL_COLD_64K,
    GLM_HIGH_ISL_COLD_64K_META,
    high_isl,
    Mode::Cold,
    "high-isl-ttft-cold-glm-64k",
    "High-ISL Cold TTFT Gate (GLM, 64k)",
    "Uncached 32k-token prefill TTFT on GLM-5.3-Flash served at 64k context",
    "~1–2 min",
    90,
    "high-isl-ttft-cold on GLM-5.3-Flash served with max_model_len 65536: the same committed \
     prompt and the same prompt_tokens check, under its own id."
);
glm_ttft!(
    GLM_HIGH_ISL_COLD_256K,
    GLM_HIGH_ISL_COLD_256K_META,
    high_isl,
    Mode::Cold,
    "high-isl-ttft-cold-glm-256k",
    "High-ISL Cold TTFT Gate (GLM, 256k)",
    "Uncached 32k-token prefill TTFT on GLM-5.3-Flash served at 256k context",
    "~1–2 min",
    90,
    "high-isl-ttft-cold on GLM-5.3-Flash served with max_model_len 262144, under its own id."
);
glm_ttft!(
    GLM_HIGH_ISL_COLD_512K,
    GLM_HIGH_ISL_COLD_512K_META,
    high_isl,
    Mode::Cold,
    "high-isl-ttft-cold-glm-512k",
    "High-ISL Cold TTFT Gate (GLM, 512k)",
    "Uncached 32k-token prefill TTFT on GLM-5.3-Flash served at 512k context",
    "~1–2 min",
    90,
    "high-isl-ttft-cold on GLM-5.3-Flash served with max_model_len 524288, under its own id."
);
glm_ttft!(
    GLM_HIGH_ISL_WARM_64K,
    GLM_HIGH_ISL_WARM_64K_META,
    high_isl,
    Mode::Warm,
    "high-isl-ttft-warm-glm-64k",
    "High-ISL Warm TTFT Gate (GLM, 64k)",
    "Cached 32k-token prefix TTFT on GLM-5.3-Flash served at 64k context",
    "~1–2 min",
    90,
    "high-isl-ttft-warm on GLM-5.3-Flash served with max_model_len 65536: the same committed \
     prompt, primed and re-sent byte for byte, under its own id."
);
glm_ttft!(
    GLM_HIGH_ISL_WARM_256K,
    GLM_HIGH_ISL_WARM_256K_META,
    high_isl,
    Mode::Warm,
    "high-isl-ttft-warm-glm-256k",
    "High-ISL Warm TTFT Gate (GLM, 256k)",
    "Cached 32k-token prefix TTFT on GLM-5.3-Flash served at 256k context",
    "~1–2 min",
    90,
    "high-isl-ttft-warm on GLM-5.3-Flash served with max_model_len 262144, under its own id."
);
glm_ttft!(
    GLM_HIGH_ISL_WARM_512K,
    GLM_HIGH_ISL_WARM_512K_META,
    high_isl,
    Mode::Warm,
    "high-isl-ttft-warm-glm-512k",
    "High-ISL Warm TTFT Gate (GLM, 512k)",
    "Cached 32k-token prefix TTFT on GLM-5.3-Flash served at 512k context",
    "~1–2 min",
    90,
    "high-isl-ttft-warm on GLM-5.3-Flash served with max_model_len 524288, under its own id."
);

#[cfg(test)]
#[path = "glm_tests.rs"]
mod tests;
