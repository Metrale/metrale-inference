// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The kernel rows of the 128-group rotated codecs (`AbiId::Trellis`): turbo1/2/3_tcq
//! and turbo4_n128 (`kernels/gb10/common/*tq128*`, `reshape_and_cache_tcq.cu`). Each codec is one
//! [`tq128!`] row set: the write, decode attention, paged prefill attention and bf16 dequant. The kernels
//! rotate per 128 values themselves, so no WHT bookend row exists for them.
//!
//! Owner: kv-codec.
//! Invariants:
//! - Every `AbiId::Trellis` codec has exactly these four rows (tested in
//!   `kernel_table_tests.rs`).

use crate::KvCodecId::{Turbo1Tcq, Turbo2Tcq, Turbo3Tcq, Turbo4N128};
use crate::kernel_key::{HeadDimClass::Model, KvOp};
use crate::kernel_table::{KernelRow, r};

/// 2026-10-18: The module holding every 128-group decode and prefill attention entry.
const ATTN: &str = "paged_decode_attn_tq128";

/// 2026-10-18: The write, decode, prefill and dequant rows of one codec: `$write` is the write module,
/// `$s` the entry suffix the `.cu` files use.
macro_rules! tq128 {
    ($id:expr, $write:literal, $s:literal) => {
        [
            r(
                KvOp::Write,
                $id,
                None,
                Model,
                $write,
                concat!("reshape_and_cache_", $s),
            ),
            r(
                KvOp::DecodeAttn,
                $id,
                None,
                Model,
                ATTN,
                concat!("paged_decode_attn_", $s),
            ),
            r(
                KvOp::PrefillAttn,
                $id,
                None,
                Model,
                ATTN,
                concat!("paged_prefill_attn_", $s),
            ),
            r(
                KvOp::DequantBf16,
                $id,
                None,
                Model,
                "tq128_dequant_bf16",
                concat!("tq128_dequant_bf16_", $s),
            ),
        ]
    };
}

/// 2026-10-18: One slice per codec, appended to the kernel table.
pub(crate) static TQ128_FAMILIES: &[&[KernelRow]] = &[
    &tq128!(Turbo3Tcq, "reshape_and_cache_tcq", "tcq3"),
    &tq128!(Turbo2Tcq, "reshape_and_cache_tcq", "tcq2"),
    &tq128!(Turbo1Tcq, "reshape_and_cache_tcq", "tcq1"),
    &tq128!(Turbo4N128, "reshape_and_cache_turbo4_n128", "turbo4_n128"),
];

#[cfg(test)]
#[path = "kernel_table_tq128_tests.rs"]
mod tests;
