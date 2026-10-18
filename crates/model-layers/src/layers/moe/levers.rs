// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The dispatch levers of the MoE layer and of the multi-sequence decode arms that
//! route through it, resolved once from the startup settings ([`MoeLevers::from_settings`]) and
//! carried on `ModelLevers::moe`. The forward paths read `ctx.levers.moe`.
//!
//! Owner: model-layers (MoE).
//! Invariants:
//! - `from_settings(&defaults)` equals [`MoeLevers::declared`]
//!   (`the_declared_defaults_are_the_old_unset_values`).
//! - `METRALE_MOE_ATOMIC_C4_DECODE` keeps its reader's text predicate: exactly `1`.
//! - `METRALE_MOE_GROUPED_ROUTED_DECODE_MIN` keeps its reader's parse: an unsigned row count,
//!   and 2 when unset or unparseable.
//! - [`MoeLevers::grouped_decode_for`] is false at every width while
//!   `METRALE_NO_MOE_GROUPED_DECODE` is set, even when the arm is forced.

use metrale_settings::InferenceServerSettings;

/// 2026-10-17: The grouped routed arm's minimum rows when
/// `METRALE_MOE_GROUPED_ROUTED_DECODE_MIN` is unset or unparseable.
const GROUPED_ROUTED_DECODE_MIN: usize = 2;

/// 2026-10-17: The MoE dispatch levers. Each field names its variable; the forward path that
/// takes it documents what it changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeLevers {
    /// 2026-09-25: `METRALE_DUMP_EXPERT_IDS`: the diagnostic dumps of `moe/dump.rs` and the SSM
    /// prefill's residual dump (`qwen3_ssm/trait_prefill.rs`).
    pub dump_expert_ids: bool,
    /// 2026-09-25: `METRALE_MOE_BATCHED_DECODE`: the multi-sequence decode runs all rows in one
    /// `forward_batched` call.
    pub batched_decode: bool,
    /// 2026-09-25: `METRALE_MOE_ATOMIC_C4_DECODE` is exactly `1`: the C=4 decode accumulates
    /// the routed down projections with FP32 atomic adds (`qwen3_ssm/trait_decode_multi_seq.rs`).
    pub atomic_c4_decode: bool,
    /// 2026-09-25: On unless `METRALE_MOE_PAIRWISE_DECODE` is switched off
    /// (`qwen3_attention/trait_impl/multi_seq/ffn.rs`).
    pub pairwise_decode: bool,
    /// 2026-09-25: `METRALE_MOE_GROUPED_ROUTED_DECODE`: the grouped routed arm from
    /// [`Self::grouped_routed_decode_min`] rows.
    pub grouped_routed_decode: bool,
    /// 2026-09-25: `METRALE_MOE_GROUPED_ROUTED_DECODE_MIN`.
    pub grouped_routed_decode_min: usize,
    /// 2026-09-25: On unless `METRALE_NO_MOE_GROUPED_DECODE` is present at any value
    /// (`moe_grouped_decode.rs`).
    pub grouped_decode: bool,
    /// 2026-09-25: `METRALE_MOE_GROUPED_DECODE`: the grouped decode arm runs below
    /// `moe_grouped_decode_min_rows()` too.
    pub grouped_decode_forced: bool,
    /// 2026-09-28: On unless `METRALE_NO_MOE_FP8_TC` is present at any value
    /// (`fp8_grouped_tc.rs`).
    pub fp8_tc: bool,
    /// 2026-09-25: On unless `METRALE_NO_FP8_MOE_GROUPED_DECODE` is present at any value
    /// (`forward_fp8_grouped_decode.rs`).
    pub fp8_grouped_decode: bool,
    /// 2026-10-02: On unless `METRALE_NO_MOE_NVFP4_TC` is present at any value
    /// (`forward_nvfp4_grouped_decode.rs`).
    pub nvfp4_tc: bool,
}

impl MoeLevers {
    /// 2026-10-17: The levers of the settings resolved once at startup.
    pub fn from_settings(s: &InferenceServerSettings) -> Self {
        let k = &s.kernels;
        Self {
            dump_expert_ids: s.telemetry.dump_expert_ids,
            batched_decode: s.scheduler.moe_batched_decode,
            atomic_c4_decode: k.moe_atomic_c4_decode.as_deref() == Some("1"),
            pairwise_decode: k.moe_pairwise_decode,
            grouped_routed_decode: k.moe_grouped_routed_decode,
            grouped_routed_decode_min: k
                .moe_grouped_routed_decode_min
                .as_deref()
                .and_then(|v| v.parse().ok())
                .unwrap_or(GROUPED_ROUTED_DECODE_MIN),
            grouped_decode: !k.no_moe_grouped_decode,
            grouped_decode_forced: k.moe_grouped_decode,
            fp8_tc: !k.no_moe_fp8_tc,
            fp8_grouped_decode: !k.no_fp8_moe_grouped_decode,
            nvfp4_tc: !k.no_moe_nvfp4_tc,
        }
    }

    /// 2026-10-17: What [`Self::from_settings`] returns at the declared defaults.
    pub const fn declared() -> Self {
        Self {
            dump_expert_ids: false,
            batched_decode: false,
            atomic_c4_decode: false,
            pairwise_decode: true,
            grouped_routed_decode: false,
            grouped_routed_decode_min: GROUPED_ROUTED_DECODE_MIN,
            grouped_decode: true,
            grouped_decode_forced: false,
            fp8_tc: true,
            fp8_grouped_decode: true,
            nvfp4_tc: true,
        }
    }

    /// 2026-10-17: Whether the grouped decode arm runs for `n` rows
    /// (`crate::layers::moe_grouped_decode_decide`).
    pub fn grouped_decode_for(&self, n: usize) -> bool {
        crate::layers::moe_grouped_decode_decide(n, self.grouped_decode, self.grouped_decode_forced)
    }
}

/// 2026-10-17: [`MoeLevers::declared`], so `ModelLevers` can keep deriving `Default`.
impl Default for MoeLevers {
    fn default() -> Self {
        Self::declared()
    }
}

#[cfg(test)]
#[path = "levers_tests.rs"]
mod tests;
