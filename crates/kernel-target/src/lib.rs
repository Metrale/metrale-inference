// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Kernel target descriptor: the `(arch, model, quant)` key of a compiled kernel set.
//!
//! 2026-10-22: One type for both of its uses. The engine's compiled-in key is
//! `KernelTarget` (`&'static str` fields, the default parameter, built into `const`s and the
//! generated `TargetPtxSet` statics); the blueprint body carries `KernelTarget<String>`, the
//! same struct decoded from the wire. Both serialize as three strings in field order.
//!
//! Owner: metrale-core (kernel target).
//! Invariants:
//! - `no_std` + `alloc`, no `unsafe`, no I/O; builds for both wasm32 targets (NSCC).
//! - Equality, hashing and display read only the three strings, whatever holds them.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]
#![deny(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

extern crate alloc;

use alloc::string::String;
use core::fmt;

use serde::{Deserialize, Serialize};

/// 2026-09-25: The `(arch, model, quant)` key of one compiled kernel set; each
/// `metrale_kernels::TargetPtxSet` carries one as its `target`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KernelTarget<S = &'static str> {
    /// 2026-09-25: Architecture identifier. For CUDA it is the base SM
    /// (`sm_121`, `sm_90`): the build strips a HARDWARE.toml feature suffix
    /// (`sm_90a` -> `sm_90`). Other backends' names are kept verbatim.
    pub arch: S,
    /// 2026-09-25: Model directory name under `kernels/<hardware>/` (e.g. `qwen3-next-80b-a3b`).
    pub model: S,
    /// 2026-09-25: Quantization directory name under the model (e.g. `nvfp4`, `fp8`, `bf16`).
    pub quant: S,
}

impl KernelTarget {
    pub const GB10_QWEN3_NVFP4: Self = Self {
        arch: "sm_121",
        model: "qwen3-next-80b-a3b",
        quant: "nvfp4",
    };

    pub const GB10_QWEN35_NVFP4: Self = Self {
        arch: "sm_121",
        model: "qwen3.5-35b-a3b",
        quant: "nvfp4",
    };

    pub const GB10_QWEN35_122B_NVFP4: Self = Self {
        arch: "sm_121",
        model: "qwen3.5-122b-a10b",
        quant: "nvfp4",
    };
}

impl<S: AsRef<str>> KernelTarget<S> {
    pub fn model_contains(&self, substring: &str) -> bool {
        self.model.as_ref().contains(substring)
    }

    /// 2026-10-22: The same key, borrowing its strings.
    pub fn as_strs(&self) -> KernelTarget<&str> {
        KernelTarget {
            arch: self.arch.as_ref(),
            model: self.model.as_ref(),
            quant: self.quant.as_ref(),
        }
    }

    /// 2026-10-22: The same key, owning its strings (the blueprint wire form).
    pub fn to_owned_strings(&self) -> KernelTarget<String> {
        KernelTarget {
            arch: String::from(self.arch.as_ref()),
            model: String::from(self.model.as_ref()),
            quant: String::from(self.quant.as_ref()),
        }
    }
}

impl<S: AsRef<str>> fmt::Display for KernelTarget<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({}, {}, {})",
            self.arch.as_ref(),
            self.model.as_ref(),
            self.quant.as_ref()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_target_display() {
        let t = KernelTarget::GB10_QWEN3_NVFP4;
        assert_eq!(t.to_string(), "(sm_121, qwen3-next-80b-a3b, nvfp4)");
        assert_eq!(t.to_owned_strings().to_string(), t.to_string());
    }

    #[test]
    fn target_equality_observes_each_dispatch_dimension() {
        let a = KernelTarget::GB10_QWEN3_NVFP4;
        assert_eq!(a, KernelTarget::GB10_QWEN3_NVFP4);
        assert_ne!(
            a,
            KernelTarget {
                arch: "sm_100a",
                ..a
            }
        );
        assert_ne!(
            a,
            KernelTarget {
                model: "llama-70b",
                ..a
            }
        );
        assert_ne!(a, KernelTarget { quant: "fp8", ..a });
    }

    /// 2026-10-22: The owned and borrowed forms are one key: they convert both ways, and the
    /// wire bytes are three strings in field order (the blueprint body's encoding of the
    /// Phase-1 mirror, so blueprint ids do not move).
    #[test]
    fn owned_and_static_forms_are_one_key_on_the_wire() {
        let a = KernelTarget::GB10_QWEN35_NVFP4;
        let owned = a.to_owned_strings();
        assert_eq!(owned.as_strs(), a);
        let bytes = postcard::to_allocvec(&owned).unwrap();
        assert_eq!(bytes, postcard::to_allocvec(&a).unwrap());
        let mut want = alloc::vec::Vec::new();
        for s in ["sm_121", "qwen3.5-35b-a3b", "nvfp4"] {
            want.push(u8::try_from(s.len()).unwrap());
            want.extend_from_slice(s.as_bytes());
        }
        assert_eq!(bytes, want);
        assert_eq!(
            postcard::from_bytes::<KernelTarget<String>>(&bytes).unwrap(),
            owned
        );
    }
}
