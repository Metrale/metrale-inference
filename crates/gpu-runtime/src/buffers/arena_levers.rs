// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `ArenaLevers`, the four settings the buffer-arena sizing reads
//! ([`super::BufferSizes::from_config_with`]), projected from the resolved `DeviceSettings`.
//!
//! Owner: gpu-runtime.
//! Invariants: `splitk` is the policy the paged-decode dispatch resolves from the same lever
//! (`attn_splitk::target_policy`), so the workspace it sizes fits the split count a launch uses.

use metrale_settings::groups::device::DeviceSettings;
use metrale_target_defaults::attn_splitk::{self, SplitkPolicy};

/// 2026-10-17: The levers that size the env-gated arena entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaLevers {
    /// 2026-10-17: `METRALE_GGUF_NATIVE_Q2`: sizes `q2_dequant_scratch`.
    pub gguf_native_q2: bool,
    /// 2026-10-17: `METRALE_GGUF_NATIVE_Q2_MMQ`: sizes `q2_act_q8`.
    pub gguf_native_q2_mmq: bool,
    /// 2026-10-17: `METRALE_FP8_ROWWISE`: sizes `ssm_rowwise_w_bf16`.
    pub fp8_rowwise: bool,
    /// 2026-10-17: `METRALE_ATTN_DECODE_SPLITK` over the target's declaration: sizes
    /// `splitk_workspace`.
    pub splitk: SplitkPolicy,
}

impl From<&DeviceSettings> for ArenaLevers {
    fn from(d: &DeviceSettings) -> Self {
        Self {
            gguf_native_q2: d.gguf_native_q2,
            gguf_native_q2_mmq: d.gguf_native_q2_mmq,
            fp8_rowwise: d.fp8_rowwise,
            splitk: attn_splitk::target_policy(d.attn_decode_splitk.as_deref()),
        }
    }
}
