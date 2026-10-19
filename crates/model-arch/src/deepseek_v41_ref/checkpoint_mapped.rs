// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: The memory-mapped safetensors checkpoint as a [`TensorSource`]: the header's
//! dtype strings become [`RawDtype`]s here and stop at this edge.
//!
//! Owner: model-arch, DeepSeek-V4.1 reference.
//! Invariants: a stored dtype outside the reference's five is refused, never reinterpreted
//! (`U8` is not read as packed fp4).

use metrale_error::Error;
use metrale_model_weights::fast_weights::MappedSafetensors;

use super::checkpoint::{RawDtype, RawTensor, TensorSource, refuse};

impl TensorSource for MappedSafetensors {
    fn tensor(&self, name: &str) -> Result<RawTensor<'_>, Error> {
        let t = self
            .get(name)
            .ok_or_else(|| refuse(name, "absent from the checkpoint".into()))?;
        let dtype = match t.stored_dtype {
            "BF16" => RawDtype::Bf16,
            "F32" => RawDtype::F32,
            "F8_E4M3" => RawDtype::F8E4m3,
            "F8_E8M0" => RawDtype::F8E8m0,
            "I8" => RawDtype::I8,
            other => return Err(refuse(name, format!("stored as {other}"))),
        };
        Ok(RawTensor {
            dtype,
            shape: t.shape.iter().map(|&d| d as u64).collect(),
            bytes: t.bytes,
        })
    }
}

#[cfg(test)]
#[path = "checkpoint_mapped_tests.rs"]
mod tests;
