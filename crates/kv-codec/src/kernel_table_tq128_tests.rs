// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: Golden tests of the 128-group codecs' kernel rows, written out literally so a
//! slip in the [`tq128!`] macro cannot pass, and of the head dims their kernels refuse.

use crate::kernel_key::{HeadDimClass, KvKernelKey, KvOp};
use crate::kernel_table::{KernelName, kv_kernel_names};
use crate::{AbiId, KvCodecId, KvPair, spec};
use metrale_error::ErrorCode;

fn n(m: &'static str, f: &'static str) -> KernelName {
    KernelName { module: m, func: f }
}

/// 2026-10-18: (codec name, write module, write fn, decode fn, prefill fn); decode and prefill
/// live in `paged_decode_attn_tq128`.
const GOLDEN: &[(&str, &str, &str, &str, &str)] = &[
    (
        "turbo3_tcq",
        "reshape_and_cache_tcq",
        "reshape_and_cache_tcq3",
        "paged_decode_attn_tcq3",
        "paged_prefill_attn_tcq3",
    ),
    (
        "turbo2_tcq",
        "reshape_and_cache_tcq",
        "reshape_and_cache_tcq2",
        "paged_decode_attn_tcq2",
        "paged_prefill_attn_tcq2",
    ),
    (
        "turbo1_tcq",
        "reshape_and_cache_tcq",
        "reshape_and_cache_tcq1",
        "paged_decode_attn_tcq1",
        "paged_prefill_attn_tcq1",
    ),
    (
        "turbo4_n128",
        "reshape_and_cache_turbo4_n128",
        "reshape_and_cache_turbo4_n128",
        "paged_decode_attn_turbo4_n128",
        "paged_prefill_attn_turbo4_n128",
    ),
];

fn trellis_codecs() -> Vec<KvCodecId> {
    KvCodecId::ATTENTION
        .into_iter()
        .filter(|&id| spec(id).abi == AbiId::Trellis)
        .collect()
}

/// 2026-10-18: Every Trellis codec resolves exactly its golden write, decode, prefill and dequant at
/// head_dim 128, 256 and 512, with no split-K, no head_dim-512 build and no WHT bookends (its
/// kernels rotate in place).
#[test]
fn trellis_codecs_resolve_their_own_kernels() {
    assert_eq!(GOLDEN.len(), trellis_codecs().len());
    for &(name, wm, wf, df, pf) in GOLDEN {
        let p: KvPair = name.parse().expect(name);
        assert_eq!(spec(p.k()).abi, AbiId::Trellis, "{name}");
        for hd in [128, 256, 512] {
            let got = kv_kernel_names(p, hd).unwrap_or_else(|e| panic!("{name} {hd}: {e}"));
            assert_eq!(got.write, n(wm, wf), "{name} {hd} write");
            assert_eq!(
                got.decode,
                n("paged_decode_attn_tq128", df),
                "{name} {hd} decode"
            );
            assert_eq!(
                got.prefill,
                Some(n("paged_decode_attn_tq128", pf)),
                "{name} {hd} prefill"
            );
            assert_eq!(
                (got.decode_512, got.splitk, got.wht),
                (None, None, None),
                "{name} {hd}"
            );
            let dequant = KvKernelKey::new(KvOp::DequantBf16, p, HeadDimClass::Model);
            let df_suffix = df.trim_start_matches("paged_decode_attn_");
            assert_eq!(
                dequant
                    .kernel_name()
                    .map(|k| (k.module, k.func.trim_start_matches("tq128_dequant_bf16_"))),
                Some(("tq128_dequant_bf16", df_suffix)),
                "{name} {hd} dequant"
            );
        }
    }
}

/// 2026-10-18: A head dim that is not a whole number of 128-value groups, or one the kernels are
/// not built for, is refused with `KvPairUnsupported` before any row is looked up.
#[test]
fn trellis_codecs_refuse_head_dims_their_kernels_cannot_hold() {
    for id in trellis_codecs() {
        let p = KvPair::symmetric(id);
        for hd in [0, 64, 96, 192, 320, 384, 640, 1024] {
            let err = kv_kernel_names(p, hd).expect_err("unsupported head dim");
            assert_eq!(err.code(), ErrorCode::KvPairUnsupported, "{id:?} {hd}");
        }
    }
}

/// 2026-10-18: No Trellis codec has a WHT bookend row: a bookend around an in-kernel rotation
/// would rotate twice.
#[test]
fn trellis_codecs_have_no_bookend_rows() {
    for id in trellis_codecs() {
        for op in [KvOp::WhtFwd, KvOp::WhtInv] {
            let key = KvKernelKey::new(op, KvPair::symmetric(id), HeadDimClass::Model);
            assert_eq!(key.kernel_name(), None, "{id:?} {op:?}");
            assert!(!spec(id).rotates());
        }
    }
}
