// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `ArenaLevers`, the four settings the buffer-arena sizing reads
//! ([`super::BufferSizes::from_config_with`]), projected from the resolved `DeviceSettings`, and
//! the declared-lever sizing that tests and examples use.
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

impl ArenaLevers {
    /// 2026-10-17: The levers at the declared device settings, for tests and examples; a serve
    /// projects the settings it resolved at startup.
    pub fn declared() -> Self {
        Self::from(&DeviceSettings::declared().expect("the declared device table projects"))
    }
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

impl super::BufferSizes {
    /// 2026-10-17: [`super::BufferSizes::from_config_with`] at the declared levers, for tests and
    /// examples.
    pub fn from_config(
        config: &metrale_config::ModelConfig,
        max_batch_tokens: usize,
        max_seq_len: usize,
        kv_block_size: usize,
        max_batch_size: usize,
    ) -> Self {
        Self::from_config_with(
            config,
            max_batch_tokens,
            max_seq_len,
            kv_block_size,
            max_batch_size,
            &ArenaLevers::declared(),
        )
    }
}
