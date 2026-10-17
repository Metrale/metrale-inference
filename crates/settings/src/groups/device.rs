// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `DeviceSettings`, the levers gpu-runtime and the kernels crate read at run time,
//! projected once from the resolved [`InferenceServerSettings`] and handed to the CUDA backend
//! when it is built (ROTI). Also the projection error and the declared-defaults settings the
//! other projections (`device_weights`, `cache::CacheTuning`, `speculative::DrafterSettings`)
//! share.
//!
//! The levers stay in the groups that declare them, so the other readers of the same lever (the
//! model crates) keep their field paths. A Text lever keeps the parse its old reader applied; an
//! Int, Float or Switch lever has the resolver's strict parse, and a value the device-side type
//! cannot hold is a [`ProjectionError`], never a silent default.
//!
//! Owner: settings.
//! Invariants: a projection reads only `InferenceServerSettings`; each field names its lever.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use crate::Env;
use crate::resolve::{InferenceServerSettings, SettingsError};
use crate::sources::{CliLayer, RecipeLayer};

/// 2026-10-17: A resolved value the device-side type cannot hold.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProjectionError {
    /// 2026-10-17: `env` resolved to `value`, outside `range`.
    OutOfRange {
        env: &'static str,
        value: String,
        range: &'static str,
    },
}

impl fmt::Display for ProjectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfRange { env, value, range } => {
                write!(f, "{env}: {value} is outside {range}")
            }
        }
    }
}

impl core::error::Error for ProjectionError {}

/// 2026-10-17: `v` as a `usize`; a negative or too large value is an error.
///
/// # Errors
/// `OutOfRange` naming `env`.
pub fn non_negative(env: &'static str, v: i64) -> Result<usize, ProjectionError> {
    usize::try_from(v).map_err(|_| out_of_range(env, v, "0..=usize::MAX"))
}

fn out_of_range(env: &'static str, v: i64, range: &'static str) -> ProjectionError {
    ProjectionError::OutOfRange {
        env,
        value: alloc::format!("{v}"),
        range,
    }
}

/// 2026-10-17: The environment with nothing set: what [`declared`] resolves against.
struct NoEnv;

impl Env for NoEnv {
    fn var(&self, _name: &str) -> metrale_error::Result<Option<String>> {
        Ok(None)
    }

    fn names(&self) -> Vec<String> {
        Vec::new()
    }
}

/// 2026-10-17: Every runtime lever at its declared default (no CLI, recipe or env value), for a
/// device built outside a `met` process (examples, device tests). A `met` process passes the
/// settings it resolved at startup instead.
///
/// # Errors
/// `BadDefault` when a declared default does not parse (a table bug `groups_tests` also
/// catches).
pub fn declared() -> Result<InferenceServerSettings, SettingsError> {
    crate::resolve(&NoEnv, &CliLayer::empty(), &RecipeLayer::empty())
}

/// 2026-10-17: `METRALE_REDZONE_FILL` when unset: the byte the guard bands are filled with.
pub const REDZONE_FILL_BUILT_IN: u8 = 0xEE;

/// 2026-10-17: `METRALE_CUTLASS_WORKSPACE_MB` when unset.
pub const CUTLASS_WORKSPACE_MB_BUILT_IN: usize = 512;

/// 2026-10-17: The process stream's priority (`METRALE_MAIN_STREAM_HIGH_PRIORITY`,
/// `METRALE_MAIN_STREAM_UNDER_GREATEST`). gpu-runtime rejects `under_greatest` without `high`
/// when it creates the stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamPriority {
    pub high: bool,
    pub under_greatest: bool,
}

/// 2026-10-17: The allocation guard bands (`METRALE_REDZONE*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Redzone {
    /// 2026-10-17: Guard-band bytes, a multiple of 16; 0 disables the bands.
    pub bytes: usize,
    pub fill: u8,
    /// 2026-10-17: Only allocations with creation index `>= min_idx` are padded.
    pub min_idx: usize,
    /// 2026-10-17: Log a backtrace when the padded allocation with this index is made.
    pub trace_idx: Option<usize>,
}

/// 2026-10-17: The levers gpu-runtime and the kernels crate read at run time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceSettings {
    /// 2026-10-17: `METRALE_DEBUG_SYNC_KERNELS`.
    pub debug_sync_kernels: bool,
    pub redzone: Redzone,
    pub stream_priority: StreamPriority,
    /// 2026-10-17: `METRALE_NCCL_HOST_HOIST`.
    pub nccl_host_hoist: bool,
    /// 2026-10-17: `METRALE_D2H_SPIN_SYNC`.
    pub d2h_spin_sync: bool,
    /// 2026-10-17: `METRALE_D2H_TRACE`: `None` unset (no counting); a value that does not parse
    /// as `u64` is 0 (count, no backtrace), as the old reader had it.
    pub d2h_trace: Option<u64>,
    /// 2026-10-17: `METRALE_CUTLASS_WORKSPACE_MB`, in bytes.
    pub cutlass_workspace_bytes: usize,
    /// 2026-10-17: `METRALE_GGUF_NATIVE_Q2`.
    pub gguf_native_q2: bool,
    /// 2026-10-17: `METRALE_GGUF_NATIVE_Q2_MMQ`.
    pub gguf_native_q2_mmq: bool,
    /// 2026-10-17: `METRALE_FP8_ROWWISE`.
    pub fp8_rowwise: bool,
    /// 2026-10-17: `METRALE_ATTN_DECODE_SPLITK`, raw: `attn_splitk::resolve_policy` parses it
    /// against the target's declaration.
    pub attn_decode_splitk: Option<String>,
    /// 2026-10-17: `METRALE_ATTN_DECODE_GQA_PACK`, raw: `attn_splitk::resolve_gqa_pack` parses it
    /// against the declaration.
    pub attn_decode_gqa_pack: Option<String>,
}

impl DeviceSettings {
    /// 2026-10-17: [`declared`], projected.
    ///
    /// # Errors
    /// A declared default that does not parse or project.
    pub fn declared() -> Result<Self, DeviceSettingsError> {
        Ok(Self::try_from(&declared()?)?)
    }
}

impl TryFrom<&InferenceServerSettings> for DeviceSettings {
    type Error = ProjectionError;

    fn try_from(s: &InferenceServerSettings) -> Result<Self, ProjectionError> {
        let rt = &s.runtime;
        let bytes = match rt.redzone {
            None => 0,
            Some(n) => non_negative("METRALE_REDZONE", n)?.next_multiple_of(16),
        };
        let fill = match rt.redzone_fill {
            None => REDZONE_FILL_BUILT_IN,
            Some(v) => {
                u8::try_from(v).map_err(|_| out_of_range("METRALE_REDZONE_FILL", v, "0..=255"))?
            }
        };
        let min_idx = rt
            .redzone_min_idx
            .map(|v| non_negative("METRALE_REDZONE_MIN_IDX", v))
            .transpose()?
            .unwrap_or(0);
        let trace_idx = s
            .telemetry
            .redzone_trace_idx
            .map(|v| non_negative("METRALE_REDZONE_TRACE_IDX", v))
            .transpose()?;
        let workspace_mb = s
            .kernels
            .cutlass_workspace_mb
            .map(|v| non_negative("METRALE_CUTLASS_WORKSPACE_MB", v))
            .transpose()?
            .unwrap_or(CUTLASS_WORKSPACE_MB_BUILT_IN);
        let cutlass_workspace_bytes = workspace_mb
            .checked_mul(1024 * 1024)
            .ok_or_else(|| out_of_range_mb(workspace_mb))?;
        Ok(Self {
            debug_sync_kernels: s.telemetry.debug_sync_kernels,
            redzone: Redzone {
                bytes,
                fill,
                min_idx,
                trace_idx,
            },
            stream_priority: StreamPriority {
                high: rt.main_stream_high_priority,
                under_greatest: rt.main_stream_under_greatest,
            },
            nccl_host_hoist: rt.nccl_host_hoist,
            d2h_spin_sync: s.server.d2h_spin_sync,
            d2h_trace: s
                .telemetry
                .d2h_trace
                .as_deref()
                .map(|v| v.parse().unwrap_or(0)),
            cutlass_workspace_bytes,
            gguf_native_q2: s.model.gguf_native_q2,
            gguf_native_q2_mmq: s.kernels.gguf_native_q2_mmq,
            fp8_rowwise: s.kernels.fp8_rowwise,
            attn_decode_splitk: s.kernels.attn_decode_splitk.clone(),
            attn_decode_gqa_pack: s.kernels.attn_decode_gqa_pack.clone(),
        })
    }
}

fn out_of_range_mb(mb: usize) -> ProjectionError {
    ProjectionError::OutOfRange {
        env: "METRALE_CUTLASS_WORKSPACE_MB",
        value: alloc::format!("{mb}"),
        range: "a byte count that fits usize",
    }
}

/// 2026-10-17: Why the declared-defaults projection failed.
#[derive(Clone, Debug, PartialEq)]
pub enum DeviceSettingsError {
    Resolve(SettingsError),
    Project(ProjectionError),
}

impl fmt::Display for DeviceSettingsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Resolve(e) => write!(f, "{e}"),
            Self::Project(e) => write!(f, "{e}"),
        }
    }
}

impl core::error::Error for DeviceSettingsError {}

impl From<SettingsError> for DeviceSettingsError {
    fn from(e: SettingsError) -> Self {
        Self::Resolve(e)
    }
}

impl From<ProjectionError> for DeviceSettingsError {
    fn from(e: ProjectionError) -> Self {
        Self::Project(e)
    }
}

#[cfg(test)]
#[path = "device_tests.rs"]
mod tests;
