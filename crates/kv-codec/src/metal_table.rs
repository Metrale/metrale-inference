// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The Metal backend's KV kernel table, keyed by [`KvKernelKey`] like the CUDA one.
//! The Metal cache is contiguous per sequence and its kernels are a different set (a quantizing
//! append and a dequantizing single-token decode per codec pair), so it is its own table, read
//! by its own resolver. It replaces the Metal forward's private dtype enum.
//!
//! Owner: kv-codec.
//! Invariants:
//! - Every key appears at most once (tested).
//! - A pair the table has no append or decode for is refused with `KvCapabilityMissing`; no
//!   lookup falls back to another codec's kernel.
//! - The WHT bookends are `Some` exactly when a side is rotated; the Metal rotation kernel takes
//!   the head dim at run time.

use metrale_error::{Error, ErrorCode, error};

use crate::kernel_key::{HeadDimClass, KvKernelKey, KvOp, ResolvedKvKernels};
use crate::kernel_table::{KernelName, KernelRow, r};
use crate::{KvCodecId, KvPair, spec};

use HeadDimClass::Model;
use KvCodecId::{Bf16, Turbo2, Turbo3, Turbo4, Turbo8};

/// 2026-10-18: The append, decode and WHT rows of one rotated symmetric codec. `$s` is the
/// codec's name, which the Metal kernels carry as their suffix.
macro_rules! metal_turbo {
    ($k:expr, $s:literal) => {
        [
            r(
                KvOp::Write,
                $k,
                None,
                Model,
                concat!("kv_cache_append_", $s),
                concat!("kv_cache_append_", $s),
            ),
            r(
                KvOp::DecodeAttn,
                $k,
                None,
                Model,
                concat!("attention_decode_", $s),
                concat!("attention_decode_", $s),
            ),
            r(
                KvOp::WhtFwd,
                $k,
                None,
                Model,
                "wht_bf16",
                "wht_bf16_inplace",
            ),
            r(
                KvOp::WhtInv,
                $k,
                None,
                Model,
                "wht_bf16",
                "wht_bf16_inplace_inv",
            ),
        ]
    };
}

/// 2026-10-18: The append and decode rows of a bf16-K, rotated-V pair. `$s` is the pair's
/// canonical name.
macro_rules! metal_bf16k {
    ($v:expr, $s:literal) => {
        [
            r(
                KvOp::Write,
                Bf16,
                Some($v),
                Model,
                "kv_cache_append_bf16k_turbov",
                concat!("kv_cache_append_", $s),
            ),
            r(
                KvOp::DecodeAttn,
                Bf16,
                Some($v),
                Model,
                "attention_decode_bf16k_turbov",
                concat!("attention_decode_", $s),
            ),
        ]
    };
}

/// 2026-10-18: The table, one slice per codec or pair.
static METAL: &[&[KernelRow]] = &[
    &[
        r(
            KvOp::Write,
            Bf16,
            None,
            Model,
            "kv_cache_append",
            "kv_cache_append",
        ),
        r(
            KvOp::DecodeAttn,
            Bf16,
            None,
            Model,
            "attention_decode",
            "attention_decode",
        ),
    ],
    &metal_turbo!(Turbo8, "turbo8"),
    &metal_turbo!(Turbo4, "turbo4"),
    &metal_turbo!(Turbo3, "turbo3"),
    &metal_turbo!(Turbo2, "turbo2"),
    &metal_bf16k!(Turbo4, "bf16k_turbo4v"),
    &metal_bf16k!(Turbo3, "bf16k_turbo3v"),
    &metal_bf16k!(Turbo2, "bf16k_turbo2v"),
];

fn rows() -> impl Iterator<Item = &'static KernelRow> {
    METAL.iter().flat_map(|family| family.iter())
}

/// 2026-10-18: The Metal entry point `key` names, or `None` when Metal has no kernel for it.
fn metal_name(key: KvKernelKey) -> Option<KernelName> {
    rows().find(|row| row.key == key).map(|row| row.name)
}

/// 2026-10-18: Every pair the Metal table has an append for, in table order.
pub fn metal_kv_pairs() -> impl Iterator<Item = KvPair> {
    rows()
        .filter(|row| row.key.op == KvOp::Write)
        .map(|row| KvPair::from_parts(row.key.k, row.key.v.unwrap_or(row.key.k)))
}

/// 2026-10-18: The Metal kernels of a cache holding `pair`: the append (`write`), the decode,
/// and the WHT bookends when a side is rotated. Metal has no head_dim-512, split-K or prefill
/// KV kernels, so those are `None`. A pair without an append or decode is refused with
/// `KvCapabilityMissing`.
pub fn metal_kv_kernel_names(pair: KvPair) -> Result<ResolvedKvKernels<KernelName>, Error> {
    let at = |op| metal_name(KvKernelKey::new(op, pair, Model));
    let missing = |what| error!(ErrorCode::KvCapabilityMissing, pair.name(), what);
    let write = at(KvOp::Write).ok_or_else(|| missing("a Metal append kernel"))?;
    let decode = at(KvOp::DecodeAttn).ok_or_else(|| missing("a Metal decode kernel"))?;
    let wht = match [pair.k(), pair.v()]
        .into_iter()
        .find(|&id| spec(id).rotates())
    {
        None => None,
        Some(side) => {
            let key = |op| metal_name(KvKernelKey::new(op, KvPair::symmetric(side), Model));
            let both = key(KvOp::WhtFwd).zip(key(KvOp::WhtInv));
            Some(both.ok_or_else(|| missing("the Metal WHT bookends"))?)
        }
    };
    Ok(ResolvedKvKernels {
        write,
        decode,
        decode_512: None,
        splitk: None,
        prefill: None,
        wht,
    })
}

#[cfg(test)]
#[path = "metal_table_tests.rs"]
mod tests;
