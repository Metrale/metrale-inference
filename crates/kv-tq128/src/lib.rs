// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The 128-group rotated KV cache codecs, as a bit-exact CPU reference.
//!
//! Every codec here cuts a head row into 128-element groups and, per group:
//! 1. takes the L2 norm and scales the group to unit norm;
//! 2. rotates it: Rademacher signs, a 128-point Walsh-Hadamard transform, signs again
//!    (arXiv:2504.19874 — TurboQuant rotation);
//! 3. quantizes the rotated coordinates against a codebook, either along a bitshift trellis
//!    (arXiv:2406.11235 — trellis (QTIP); turbo1/2/3_tcq) or one value at a time (turbo4_n128);
//! 4. stores one fp16 norm, corrected so the reconstruction keeps the input's norm.
//!
//! Decode is `codebook[index] * norm` in the rotated domain, so attention scores against a
//! rotated query without undoing the rotation per key.
//!
//! The codebooks, the bitshift-trellis layout and the V decode scales are derived from
//! spiritbuun/buun-llama-cpp (MIT, Copyright (c) 2023-2026 The ggml authors;
//! https://github.com/spiritbuun/buun-llama-cpp, pinned at 34ef0ef7c6ea); credit spiritbuun.
//! Codebook training: arXiv:2410.13780. Report: "Closing the Gap: Trellis-Coded Quantization for
//! KV Cache at 2-3 Bits" (spiritbuun, 2026). Format notes: docs/kv-cache-codecs.md.
//!
//! This crate is the oracle the GPU kernels (kernels/gb10/common/*tq128*, *tcq*) are tested
//! against byte for byte: every reduction runs in the order the kernels run it, and the build
//! of those kernels contracts no multiply-add (`--fmad=false`).
//!
//! Owner: kv-tq128.
//! Invariants:
//! - No I/O and no `std`: everything is a function of its arguments.
//! - Inputs are finite; a non-finite input is refused, not encoded.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]
#![deny(warnings)]
#![deny(clippy::all)]
#![deny(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

extern crate alloc;

pub mod codebooks;
mod codec;
mod group;
pub mod numeric;
mod pack;
pub mod rotation;
pub mod trellis;

pub use codec::{GROUP, KvSide, Tq128Codec};
pub use group::{
    decode_group_rotated, dequant_group, dequant_group_bf16, dequant_head_bf16, encode_group,
    encode_head, group_indices,
};
pub use pack::{nibble_at, pack_nibbles, pack_trellis, trellis_state_at};

/// 2026-10-18: Why a codec call was refused. Phase 2 maps these onto the engine's coded error
/// registry at the crate edge that hands them to the server.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tq128Error {
    /// A head dimension that is zero or not a multiple of the 128-element group.
    HeadDim { head_dim: usize },
    /// A group slice whose length is not 128.
    GroupLength { found: usize },
    /// An encoded buffer whose length is not the codec's block size times the group count.
    BlockLength { expected: usize, found: usize },
    /// A codebook whose length is not the codec's index count.
    CodebookLength { expected: usize, found: usize },
    /// A trellis whose shape the bitstream cannot carry (needs 1 <= k < L <= 9).
    TrellisShape { k: u32, l: u32 },
    /// An input value that is NaN or infinite.
    NonFinite { index: usize },
}

impl core::fmt::Display for Tq128Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::HeadDim { head_dim } => {
                write!(
                    f,
                    "head_dim {head_dim} is not a positive multiple of {GROUP}"
                )
            }
            Self::GroupLength { found } => write!(f, "group of {found} values, expected {GROUP}"),
            Self::BlockLength { expected, found } => {
                write!(f, "encoded buffer of {found} bytes, expected {expected}")
            }
            Self::CodebookLength { expected, found } => {
                write!(f, "codebook of {found} entries, expected {expected}")
            }
            Self::TrellisShape { k, l } => {
                write!(f, "trellis k={k} L={l} outside 1 <= k < L <= 9")
            }
            Self::NonFinite { index } => write!(f, "input value {index} is not finite"),
        }
    }
}
