// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: LEGACY. Every process-environment read left in gpu-runtime, kept so that the
//! entry points the server and model crates still call (`MetraleCudaBackend::new`,
//! `BufferSizes::from_config`, `preflight_device_arch` and the process switches) keep their
//! current meaning until those callers pass resolved settings. Each `legacy_env_*_from` reproduces
//! the deleted reader's parse over an injected lookup, and `legacy_env_tests.rs` proves it equals
//! the `DeviceSettings` projection except for the listed deviations. The 2c-device-wire commit
//! deletes this module.
//!
//! Owner: gpu-runtime.
//! Invariants: the only module in gpu-runtime that reads the process environment.

use std::env::VarError;

use metrale_target_defaults::attn_splitk;
use metrale_settings::groups::device::{
    CUTLASS_WORKSPACE_MB_BUILT_IN, DeviceSettings, REDZONE_FILL_BUILT_IN, Redzone, StreamPriority,
};

use crate::buffers::{ArenaLevers, BufferSizes};

/// 2026-10-17: The process-environment lookup the `legacy_env_*` readers are given in production.
fn process_var(name: &str) -> Result<String, VarError> {
    std::env::var(name)
}

fn eq_one(var: &impl Fn(&str) -> Result<String, VarError>, name: &str) -> bool {
    var(name).as_deref() == Ok("1")
}

/// 2026-10-17: The deleted `cuda_host` readers: `METRALE_MAIN_STREAM_HIGH_PRIORITY` on only for
/// `1`; `METRALE_MAIN_STREAM_UNDER_GREATEST` unset or `0` off, `1` on, anything else an error.
///
/// # Errors
/// The old message for a malformed `METRALE_MAIN_STREAM_UNDER_GREATEST`.
pub fn legacy_env_stream_priority_from(
    var: impl Fn(&str) -> Result<String, VarError>,
) -> Result<StreamPriority, String> {
    let under_greatest = match var("METRALE_MAIN_STREAM_UNDER_GREATEST").ok().as_deref() {
        None | Some("0") => false,
        Some("1") => true,
        Some(other) => {
            return Err(format!(
                "METRALE_MAIN_STREAM_UNDER_GREATEST={other:?}: expected 0 or 1"
            ));
        }
    };
    Ok(StreamPriority {
        high: eq_one(&var, "METRALE_MAIN_STREAM_HIGH_PRIORITY"),
        under_greatest,
    })
}

/// 2026-10-17: [`legacy_env_stream_priority_from`] on the process environment.
///
/// # Errors
/// As [`legacy_env_stream_priority_from`].
pub fn legacy_env_stream_priority() -> Result<StreamPriority, String> {
    legacy_env_stream_priority_from(process_var)
}

/// 2026-10-17: `DeviceSettings` as the deleted gpu-runtime and kernels readers parsed them: the
/// `== "1"` switches, the redzone and CUTLASS numbers defaulting on a bad value, the raw texts.
///
/// # Errors
/// As [`legacy_env_stream_priority_from`], the one reader that could fail.
pub fn legacy_env_device_settings_from(
    var: impl Fn(&str) -> Result<String, VarError>,
) -> Result<DeviceSettings, String> {
    let parsed = |name: &str| var(name).ok().and_then(|v| v.parse::<usize>().ok());
    let workspace_mb =
        parsed("METRALE_CUTLASS_WORKSPACE_MB").unwrap_or(CUTLASS_WORKSPACE_MB_BUILT_IN);
    Ok(DeviceSettings {
        debug_sync_kernels: eq_one(&var, "METRALE_DEBUG_SYNC_KERNELS"),
        redzone: Redzone {
            bytes: parsed("METRALE_REDZONE")
                .map(|n| if n == 0 { 0 } else { n.next_multiple_of(16) })
                .unwrap_or(0),
            fill: var("METRALE_REDZONE_FILL")
                .ok()
                .and_then(|v| v.parse::<u8>().ok())
                .unwrap_or(REDZONE_FILL_BUILT_IN),
            min_idx: parsed("METRALE_REDZONE_MIN_IDX").unwrap_or(0),
            trace_idx: parsed("METRALE_REDZONE_TRACE_IDX"),
        },
        stream_priority: legacy_env_stream_priority_from(&var)?,
        nccl_host_hoist: eq_one(&var, "METRALE_NCCL_HOST_HOIST"),
        d2h_spin_sync: eq_one(&var, "METRALE_D2H_SPIN_SYNC"),
        d2h_trace: var("METRALE_D2H_TRACE")
            .ok()
            .map(|v| v.parse().unwrap_or(0)),
        cutlass_workspace_bytes: workspace_mb * 1024 * 1024,
        gguf_native_q2: eq_one(&var, "METRALE_GGUF_NATIVE_Q2"),
        gguf_native_q2_mmq: eq_one(&var, "METRALE_GGUF_NATIVE_Q2_MMQ"),
        fp8_rowwise: eq_one(&var, "METRALE_FP8_ROWWISE"),
        attn_decode_splitk: var("METRALE_ATTN_DECODE_SPLITK").ok(),
        attn_decode_gqa_pack: var("METRALE_ATTN_DECODE_GQA_PACK").ok(),
    })
}

/// 2026-10-17: [`legacy_env_device_settings_from`] on the process environment.
///
/// # Errors
/// As [`legacy_env_device_settings_from`].
pub fn legacy_env_device_settings() -> Result<DeviceSettings, String> {
    legacy_env_device_settings_from(process_var)
}

/// 2026-10-17: The arena-sizing levers as the deleted `sizes_q2`, `sizes_rowwise` and
/// `attn_splitk::policy_from_env` readers parsed them. Infallible, as they were.
pub fn legacy_env_arena_levers_from(var: impl Fn(&str) -> Result<String, VarError>) -> ArenaLevers {
    ArenaLevers {
        gguf_native_q2: eq_one(&var, "METRALE_GGUF_NATIVE_Q2"),
        gguf_native_q2_mmq: eq_one(&var, "METRALE_GGUF_NATIVE_Q2_MMQ"),
        fp8_rowwise: eq_one(&var, "METRALE_FP8_ROWWISE"),
        splitk: attn_splitk::target_policy(var("METRALE_ATTN_DECODE_SPLITK").ok().as_deref()),
    }
}

/// 2026-10-17: [`legacy_env_arena_levers_from`] on the process environment.
pub fn legacy_env_arena_levers() -> ArenaLevers {
    legacy_env_arena_levers_from(process_var)
}

impl BufferSizes {
    /// 2026-10-17: LEGACY. [`BufferSizes::from_config_with`] at [`legacy_env_arena_levers`].
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
            &legacy_env_arena_levers(),
        )
    }
}

#[cfg(feature = "cuda")]
impl crate::cuda_backend::MetraleCudaBackend {
    /// 2026-10-17: LEGACY. [`Self::with_settings`] at [`legacy_env_device_settings`]; a malformed
    /// `METRALE_MAIN_STREAM_UNDER_GREATEST` fails with the message the registry load gave it.
    pub fn new(
        ordinal: usize,
        ptx_modules: &[(&'static str, &'static [u8])],
    ) -> anyhow::Result<Self> {
        let device = legacy_env_device_settings().map_err(|msg| {
            anyhow::anyhow!(
                "MetraleRegistry load failed: {}",
                metrale_core::error::MetraleError::ModuleLoad(msg)
            )
        })?;
        Self::with_settings(ordinal, ptx_modules, &device)
    }
}

/// 2026-10-17: LEGACY. `preflight_device_arch_for` at [`legacy_env_stream_priority`]; a malformed
/// priority fails where the host would have been created, as before.
///
/// # Errors
/// As `preflight_device_arch_for`.
#[cfg(feature = "cuda")]
pub fn preflight_device_arch(ordinal: usize, compiled_arch: Option<&str>) -> anyhow::Result<()> {
    use crate::cuda_backend::arch_preflight::{DriverDeviceQuery, preflight_device_arch_with};
    let query = DriverDeviceQuery {
        priority: legacy_env_stream_priority(),
    };
    preflight_device_arch_with(ordinal, compiled_arch, &query)
}

/// 2026-10-17: The deleted `ssm_tail_ckpt_enabled` parse: on only for `1`.
pub fn legacy_env_ssm_tail_ckpt_from(var: impl Fn(&str) -> Result<String, VarError>) -> bool {
    eq_one(&var, "METRALE_SSM_TAIL_CKPT")
}

/// 2026-10-17: The deleted `hermetic_enabled` env parse: on only for `1`.
pub fn legacy_env_hermetic_from(var: impl Fn(&str) -> Result<String, VarError>) -> bool {
    eq_one(&var, "METRALE_HERMETIC")
}

/// 2026-10-17: The deleted `ssm_tail_midchunk_enabled` env parse: on unless `0`.
pub fn legacy_env_ssm_tail_midchunk_from(var: impl Fn(&str) -> Result<String, VarError>) -> bool {
    var("METRALE_SSM_TAIL_MIDCHUNK").as_deref() != Ok("0")
}

/// 2026-10-17: [`legacy_env_ssm_tail_ckpt_from`] on the process environment.
pub fn legacy_env_ssm_tail_ckpt() -> bool {
    legacy_env_ssm_tail_ckpt_from(process_var)
}

/// 2026-10-17: [`legacy_env_hermetic_from`] on the process environment.
pub fn legacy_env_hermetic() -> bool {
    legacy_env_hermetic_from(process_var)
}

/// 2026-10-17: [`legacy_env_ssm_tail_midchunk_from`] on the process environment.
pub fn legacy_env_ssm_tail_midchunk() -> bool {
    legacy_env_ssm_tail_midchunk_from(process_var)
}

#[cfg(test)]
#[path = "legacy_env_tests.rs"]
mod tests;
