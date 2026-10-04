// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Why a plan could not be made. Every variant names what was wrong; nothing is
//! approximated or defaulted instead.
//!
//! Owner: metrale-ml-utils.
//! Invariants: none beyond the types.

/// 2026-10-03: A refusal from planning, synthesis or extrapolation.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum MlError {
    /// 2026-10-03: The spec file is malformed or states a value this build does not implement.
    #[error("spec: {0}")]
    Spec(String),
    /// 2026-10-03: A safetensors header or index is malformed.
    #[error("tensor index: {0}")]
    Index(String),
    /// 2026-10-03: The checkpoint has no circuit, or its config does not map.
    #[error("checkpoint: {0}")]
    Checkpoint(String),
    /// 2026-10-03: A tensor whose storage scheme or value class is not known.
    #[error("tensor `{name}`: {why}")]
    Tensor {
        /// 2026-10-03: The source tensor.
        name: String,
        /// 2026-10-03: Why it cannot be planned.
        why: String,
    },
    /// 2026-10-03: The mock's quantization metadata would declare a different precision for a
    /// module than the source declares for the module it came from.
    #[error("quantization metadata: {0}")]
    Quant(String),
    /// 2026-10-03: The routing profile is malformed or does not fit the checkpoint.
    #[error("routing profile: {0}")]
    Routing(String),
    /// 2026-10-03: Records that do not determine the extrapolation.
    #[error("extrapolation: {0}")]
    Extrapolate(String),
    /// 2026-10-03: An I/O failure reported by a `CheckpointSource` or `CheckpointSink`.
    #[error("{0}")]
    Io(String),
}

impl From<metrale_circuit::CheckpointError> for MlError {
    fn from(e: metrale_circuit::CheckpointError) -> Self {
        MlError::Checkpoint(e.to_string())
    }
}

/// 2026-10-03: Results of this crate.
pub type Result<T> = std::result::Result<T, MlError>;
