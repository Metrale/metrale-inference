// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The kernel table: which compiled entry point each [`KvKernelKey`] names, and the
//! one resolver that turns a codec pair and a head dim into the kernel names a layer loads.
//! Engine code never spells a KV kernel name; it asks [`kv_kernel_names`] and looks the names up.
//!
//! Owner: kv-codec.
//! Invariants:
//! - Every key appears at most once (tested).
//! - Every registered codec and every pair alias has a write, a decode and a prefill row
//!   (tested), and every WHT-rotated codec has both bookend rows.
//! - A missing required kernel is an error and a missing optional one is `None`; no lookup
//!   falls back to another codec's kernel.

use metrale_error::{Error, ErrorCode, error};

use crate::kernel_key::{HeadDimClass, KvKernelKey, KvOp, ResolvedKvKernels};
use crate::kernel_table_tq128::TQ128_FAMILIES;
use crate::spec::Rotation;
use crate::{KvCodecId, KvPair, spec};

/// 2026-10-18: A compiled kernel entry point: the module a KERNEL.toml `[modules]` table (or
/// the file stem) names, and the function in it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KernelName {
    /// 2026-10-18: The module name.
    pub module: &'static str,
    /// 2026-10-18: The entry function.
    pub func: &'static str,
}

impl core::fmt::Display for KernelName {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}::{}", self.module, self.func)
    }
}

/// 2026-10-18: One table row.
#[derive(Clone, Copy, Debug)]
pub(crate) struct KernelRow {
    pub(crate) key: KvKernelKey,
    pub(crate) name: KernelName,
}

pub(crate) const fn r(
    op: KvOp,
    k: KvCodecId,
    v: Option<KvCodecId>,
    hd: HeadDimClass,
    module: &'static str,
    func: &'static str,
) -> KernelRow {
    KernelRow {
        key: KvKernelKey { op, k, v, hd },
        name: KernelName { module, func },
    }
}

use HeadDimClass::{Hd128, Hd512, Model};
use KvCodecId::{Bf16, Fp8, Nvfp4, Turbo2, Turbo3, Turbo4, Turbo8};

/// 2026-10-18: The rows every WHT-rotated family shares: the write through the turbo reshape
/// module, decode at the model's head dim and its head_dim-128 build. `$s` is the pair's
/// canonical name (tested against [`KvPair::name`]).
macro_rules! rotated {
    ($k:expr, $v:expr, $s:literal) => {
        [
            r(
                KvOp::Write,
                $k,
                $v,
                Model,
                "reshape_and_cache_turbo",
                concat!("reshape_and_cache_flash_", $s),
            ),
            r(
                KvOp::DecodeAttn,
                $k,
                $v,
                Model,
                concat!("paged_decode_", $s),
                concat!("paged_decode_attn_", $s),
            ),
            r(
                KvOp::DecodeAttn,
                $k,
                $v,
                Hd128,
                concat!("paged_decode_", $s, "_128"),
                concat!("paged_decode_attn_", $s),
            ),
        ]
    };
}

/// 2026-10-18: The 64-row paged prefill kernel of a family.
macro_rules! prefill64 {
    ($k:expr, $v:expr, $s:literal) => {
        r(
            KvOp::PrefillAttn64,
            $k,
            $v,
            Model,
            concat!("prefill_paged_", $s),
            concat!("attn_prefill_paged_", $s, "_64"),
        )
    };
}

/// 2026-10-18: The WHT bookends a rotated codec wraps attention in.
macro_rules! wht {
    ($k:expr) => {
        [
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

/// 2026-10-18: The table, one slice per family.
static FAMILIES: &[&[KernelRow]] = &[
    &[
        r(
            KvOp::Write,
            Bf16,
            None,
            Model,
            "reshape_and_cache",
            "reshape_and_cache_flash",
        ),
        r(
            KvOp::DecodeAttn,
            Bf16,
            None,
            Model,
            "paged_decode",
            "paged_decode_attn",
        ),
        r(
            KvOp::DecodeAttn512,
            Bf16,
            None,
            Hd512,
            "paged_decode_attn_512",
            "paged_decode_attn",
        ),
        r(
            KvOp::PrefillAttn,
            Bf16,
            None,
            Model,
            "prefill_paged",
            "attn_prefill_paged",
        ),
        r(
            KvOp::PrefillAttn64,
            Bf16,
            None,
            Model,
            "prefill_paged",
            "attn_prefill_paged_64",
        ),
    ],
    &[
        r(
            KvOp::Write,
            Fp8,
            None,
            Model,
            "reshape_and_cache",
            "reshape_and_cache_flash_fp8",
        ),
        r(
            KvOp::DecodeAttn,
            Fp8,
            None,
            Model,
            "paged_decode_fp8",
            "paged_decode_attn_fp8",
        ),
        r(
            KvOp::DecodeAttn512,
            Fp8,
            None,
            Hd512,
            "paged_decode_attn_fp8_512",
            "paged_decode_attn_fp8",
        ),
        r(
            KvOp::DecodeSplitK,
            Fp8,
            None,
            Model,
            "paged_decode_fp8",
            "paged_decode_attn_splitk_fp8",
        ),
        r(
            KvOp::DecodeReduce,
            Fp8,
            None,
            Model,
            "paged_decode_fp8",
            "paged_decode_attn_reduce_fp8",
        ),
        r(
            KvOp::PrefillAttn,
            Fp8,
            None,
            Model,
            "prefill_paged_fp8",
            "attn_prefill_paged_fp8",
        ),
        prefill64!(Fp8, None, "fp8"),
    ],
    &[
        r(
            KvOp::Write,
            Nvfp4,
            None,
            Model,
            "reshape_and_cache",
            "reshape_and_cache_flash_nvfp4",
        ),
        r(
            KvOp::DecodeAttn,
            Nvfp4,
            None,
            Model,
            "paged_decode_nvfp4",
            "paged_decode_attn_nvfp4",
        ),
        r(
            KvOp::DecodeSplitK,
            Nvfp4,
            None,
            Model,
            "paged_decode_nvfp4",
            "paged_decode_attn_splitk_nvfp4",
        ),
        r(
            KvOp::DecodeReduce,
            Nvfp4,
            None,
            Model,
            "paged_decode_nvfp4",
            "paged_decode_attn_reduce_nvfp4",
        ),
        r(
            KvOp::PrefillAttn,
            Nvfp4,
            None,
            Model,
            "prefill_paged_nvfp4",
            "attn_prefill_paged_nvfp4",
        ),
        prefill64!(Nvfp4, None, "nvfp4"),
    ],
    &rotated!(Turbo4, None, "turbo4"),
    &[
        r(
            KvOp::DecodeAttn512,
            Turbo4,
            None,
            Hd512,
            "paged_decode_turbo4_512",
            "paged_decode_attn_turbo4",
        ),
        r(
            KvOp::PrefillAttn,
            Turbo4,
            None,
            Model,
            "prefill_paged_turbo4",
            "attn_prefill_paged_turbo4",
        ),
        prefill64!(Turbo4, None, "turbo4"),
    ],
    &wht!(Turbo4),
    &rotated!(Turbo3, None, "turbo3"),
    &[prefill64!(Turbo3, None, "turbo3")],
    &wht!(Turbo3),
    &rotated!(Turbo2, None, "turbo2"),
    &[r(
        KvOp::PrefillAttn,
        Turbo2,
        None,
        Model,
        "prefill_paged_turbo2",
        "attn_prefill_paged_turbo2",
    )],
    &wht!(Turbo2),
    &rotated!(Turbo8, None, "turbo8"),
    &[
        r(
            KvOp::DecodeAttn512,
            Turbo8,
            None,
            Hd512,
            "paged_decode_turbo8_512",
            "paged_decode_attn_turbo8",
        ),
        prefill64!(Turbo8, None, "turbo8"),
    ],
    &wht!(Turbo8),
    &rotated!(Turbo4, Some(Turbo3), "turbo4k_turbo3v"),
    &[prefill64!(Turbo4, Some(Turbo3), "turbo4k_turbo3v")],
    &rotated!(Turbo4, Some(Turbo8), "turbo4k_turbo8v"),
    &[prefill64!(Turbo4, Some(Turbo8), "turbo4k_turbo8v")],
    &rotated!(Turbo3, Some(Turbo8), "turbo3k_turbo8v"),
    &[prefill64!(Turbo3, Some(Turbo8), "turbo3k_turbo8v")],
    &rotated!(Bf16, Some(Turbo4), "bf16k_turbo4v"),
    &[prefill64!(Bf16, Some(Turbo4), "bf16k_turbo4v")],
    &rotated!(Bf16, Some(Turbo3), "bf16k_turbo3v"),
    &[prefill64!(Bf16, Some(Turbo3), "bf16k_turbo3v")],
    &rotated!(Fp8, Some(Turbo4), "fp8k_turbo4v"),
    &[prefill64!(Fp8, Some(Turbo4), "fp8k_turbo4v")],
    &rotated!(Fp8, Some(Turbo3), "fp8k_turbo3v"),
    &[prefill64!(Fp8, Some(Turbo3), "fp8k_turbo3v")],
    &rotated!(Bf16, Some(Turbo2), "bf16k_turbo2v"),
    &[prefill64!(Bf16, Some(Turbo2), "bf16k_turbo2v")],
    &rotated!(Fp8, Some(Turbo2), "fp8k_turbo2v"),
    &[prefill64!(Fp8, Some(Turbo2), "fp8k_turbo2v")],
];

fn rows() -> impl Iterator<Item = &'static KernelRow> {
    FAMILIES
        .iter()
        .chain(TQ128_FAMILIES)
        .flat_map(|family| family.iter())
}

impl KvKernelKey {
    /// 2026-10-18: The key of `op` on `pair` at `hd`: a symmetric pair keys by its one codec, an
    /// asymmetric pair by both.
    pub fn new(op: KvOp, pair: KvPair, hd: HeadDimClass) -> Self {
        KvKernelKey {
            op,
            k: pair.k(),
            v: pair.is_asymmetric().then_some(pair.v()),
            hd,
        }
    }

    /// 2026-10-18: The entry point this key names, or `None` when no kernel is compiled for it.
    pub fn kernel_name(&self) -> Option<KernelName> {
        rows().find(|row| row.key == *self).map(|row| row.name)
    }
}

/// 2026-10-18: The kernel names a layer holding `pair` at `head_dim` loads. Decode prefers the
/// head_dim-128 build when `head_dim <= 128` and one exists. The head_dim-512 decode, split-K
/// and prefill are `None` when the pair has none; the WHT bookends are `Some` exactly when a
/// side is WHT-rotated at a head dim its rotation covers. A pair without a write or decode
/// kernel, or a side whose kernels cannot hold a `head_dim` row, is refused with
/// `KvPairUnsupported`.
pub fn kv_kernel_names(
    pair: KvPair,
    head_dim: u32,
) -> Result<ResolvedKvKernels<KernelName>, Error> {
    let at = |op, hd| KvKernelKey::new(op, pair, hd).kernel_name();
    let unsupported = || {
        error!(
            ErrorCode::KvPairUnsupported,
            pair.k().name(),
            pair.v().name(),
            head_dim
        )
    };
    if !(spec(pair.k()).fits_head_dim(head_dim) && spec(pair.v()).fits_head_dim(head_dim)) {
        return Err(unsupported());
    }
    let write = at(KvOp::Write, Model).ok_or_else(unsupported)?;
    let decode_128 = if head_dim <= 128 {
        at(KvOp::DecodeAttn, Hd128)
    } else {
        None
    };
    let decode = decode_128
        .or_else(|| at(KvOp::DecodeAttn, Model))
        .ok_or_else(unsupported)?;
    let splitk = at(KvOp::DecodeSplitK, Model).zip(at(KvOp::DecodeReduce, Model));
    let prefill = at(KvOp::PrefillAttn64, Model).or_else(|| at(KvOp::PrefillAttn, Model));
    Ok(ResolvedKvKernels {
        write,
        decode,
        decode_512: at(KvOp::DecodeAttn512, Hd512),
        splitk,
        prefill,
        wht: wht_bookends(pair, head_dim)?,
    })
}

/// 2026-10-18: The WHT bookends of the first side whose rotation covers `head_dim`.
fn wht_bookends(pair: KvPair, head_dim: u32) -> Result<Option<(KernelName, KernelName)>, Error> {
    let Some(side) = [pair.k(), pair.v()]
        .into_iter()
        .find(|&id| match spec(id).rotation {
            Rotation::Wht { sizes } => sizes.contains(&head_dim),
            Rotation::None | Rotation::Fwht { .. } | Rotation::InKernel { .. } => false,
        })
    else {
        return Ok(None);
    };
    let symmetric = KvPair::symmetric(side);
    let fwd = KvKernelKey::new(KvOp::WhtFwd, symmetric, Model).kernel_name();
    let inv = KvKernelKey::new(KvOp::WhtInv, symmetric, Model).kernel_name();
    fwd.zip(inv).map(Some).ok_or_else(|| {
        error!(
            ErrorCode::KvCapabilityMissing,
            side.name(),
            "the WHT bookends"
        )
    })
}

#[cfg(test)]
#[path = "kernel_table_tests.rs"]
mod tests;
