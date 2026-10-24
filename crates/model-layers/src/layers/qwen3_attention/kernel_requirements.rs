// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The startup kernel check for a `--kv-cache-dtype`. 2026-10-18: The names come
//! from the codec kernel table (`init_kv_kernels`), the one source `init.rs` also resolves
//! from: the write and decode kernels, plus the kernels a rotated side cannot run without (its
//! chunked-prefill kernel and the WHT bookends).
//!
//! Owner: model-layers (attention).
//! Invariants: no kernel name is spelled here.

use metrale_cache::kv_cache::KvCacheDtype;
use metrale_kv_codec::{KernelName, spec};

/// 2026-10-18: True when either side of `kv_dtype` stores rotated values, by the bookends or in
/// its own kernels (`KvCodecSpec::stores_rotated`). Such a pair has no dense prefill to fall
/// back to, so its chunked-prefill kernel is required, and it is the experimental set the error
/// note names.
fn rotates(kv_dtype: KvCacheDtype) -> bool {
    let pair = kv_dtype.codec_pair();
    spec(pair.k()).stores_rotated() || spec(pair.v()).stores_rotated()
}

/// 2026-10-18: Every kernel `kv_dtype` at `head_dim` cannot run without: the write and decode
/// kernels; for a rotated pair, its chunked-prefill kernel; and the WHT bookends when the table
/// names them (a WHT-rotated side at a head dim its rotation covers).
pub(super) fn required_kernels(
    kv_dtype: KvCacheDtype,
    head_dim: usize,
) -> anyhow::Result<Vec<KernelName>> {
    let names = super::init_kv_kernels::kv_kernel_names_for(kv_dtype.codec_pair(), head_dim)?;
    let mut required = vec![names.write, names.decode];
    if rotates(kv_dtype) {
        required.extend(names.prefill);
    }
    required.extend(names.wht.into_iter().flat_map(|(fwd, inv)| [fwd, inv]));
    Ok(required)
}

/// 2026-09-25: Resolves `required_kernels` for one dtype, and returns an error naming every
/// kernel that is missing.
pub(super) fn validate_required_kernels(
    gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
    kv_dtype: KvCacheDtype,
    head_dim: usize,
) -> anyhow::Result<()> {
    // 2026-09-25: The write/decode pair is checked here too.
    // `Qwen3AttentionLayer::new` resolves it with `gpu.kernel(..)?`, whose
    // error ("Kernel lookup <module>::<fn>") does not name the dtype.
    let missing: Vec<String> = required_kernels(kv_dtype, head_dim)?
        .into_iter()
        .filter(|name| gpu.kernel(name.module, name.func).is_err())
        .map(|name| name.to_string())
        .collect();
    if !missing.is_empty() {
        let note = if rotates(kv_dtype) {
            " This KV-cache dtype is EXPERIMENTAL and is not supported on this kernel target; \
             use --kv-cache-dtype fp8, bf16 or nvfp4."
        } else {
            " Rebuild kernels or pick a supported dtype."
        };
        anyhow::bail!(
            "kv-cache-dtype {kv_dtype:?} (head_dim {head_dim}) requires kernel(s) \
             missing from this build: {}.{note}",
            missing.join(", ")
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn funcs(kv_dtype: KvCacheDtype, head_dim: usize) -> Vec<&'static str> {
        required_kernels(kv_dtype, head_dim)
            .unwrap()
            .into_iter()
            .map(|n| n.func)
            .collect()
    }

    /// 2026-10-18: A plain dtype requires only its write and decode kernels: its prefill has a
    /// dense fallback, so a target without the paged prefill kernel still boots.
    #[test]
    fn plain_dtypes_require_write_and_decode_only() {
        for d in [KvCacheDtype::Bf16, KvCacheDtype::Fp8, KvCacheDtype::Nvfp4] {
            for hd in [64, 128, 256, 512] {
                assert_eq!(required_kernels(d, hd).unwrap().len(), 2, "{d} hd={hd}");
            }
        }
    }

    /// 2026-10-18: Every dtype with a WHT-rotated side requires its chunked-prefill kernel, and
    /// the bookends exactly at head dims 128, 256 and 512.
    fn check_bookended(
        d: impl core::fmt::Display + Copy,
        funcs: impl Fn(usize) -> Vec<&'static str>,
    ) {
        for hd in [128, 256, 512] {
            let f = funcs(hd);
            assert_eq!(f.len(), 5, "{d} hd={hd}: {f:?}");
            assert!(
                f[2].starts_with("attn_prefill_paged_"),
                "{d} hd={hd}: {f:?}"
            );
            assert_eq!(&f[3..], ["wht_bf16_inplace", "wht_bf16_inplace_inv"]);
        }
        for hd in [64, 513] {
            assert_eq!(funcs(hd).len(), 3, "{d} hd={hd}: no bookends");
        }
    }

    /// 2026-10-18: A codec that rotates inside its own kernels requires its chunked-prefill
    /// kernel at every head dim it fits, and no bookends: no dense prefill could read its
    /// records, so a target without that kernel must refuse it at boot. Returns the head dims
    /// checked.
    fn check_in_kernel(
        d: impl core::fmt::Display + Copy,
        fits: impl Fn(u32) -> bool,
        funcs: impl Fn(usize) -> Vec<&'static str>,
    ) -> usize {
        let dims: Vec<u32> = [128u32, 256].into_iter().filter(|&hd| fits(hd)).collect();
        for &hd in &dims {
            let f = funcs(hd as usize);
            assert_eq!(f.len(), 3, "{d} hd={hd}: {f:?}");
            assert!(
                f[2].starts_with("paged_prefill_attn_"),
                "{d} hd={hd}: {f:?}"
            );
        }
        dims.len()
    }

    /// 2026-10-18: Each rotated dtype requires the kernels its rotation needs: the four turbo
    /// codecs and nine pairs (WHT bookends) per `check_bookended`, the codecs that rotate in
    /// their own kernels per `check_in_kernel`.
    #[test]
    fn rotated_dtypes_require_their_prefill_and_rotation_kernels() {
        let (mut bookended, mut in_kernel) = (0, 0);
        for d in KvCacheDtype::all().filter(|&d| rotates(d)) {
            let pair = d.codec_pair();
            let (k, v) = (spec(pair.k()), spec(pair.v()));
            if k.rotates() || v.rotates() {
                bookended += 1;
                check_bookended(d, |hd| funcs(d, hd));
            } else {
                in_kernel += check_in_kernel(d, |hd| k.fits_head_dim(hd), |hd| funcs(d, hd));
            }
        }
        assert_eq!(bookended, 13, "four turbo codecs and nine pairs");
        assert!(
            in_kernel >= 4,
            "no in-kernel rotated codec was checked ({in_kernel})"
        );
    }

    /// 2026-10-18: Turbo2's prefill is its one non-`_64` build, at every head dim.
    #[test]
    fn turbo2_requires_its_plain_prefill() {
        assert_eq!(
            funcs(KvCacheDtype::Turbo2, 64)[2],
            "attn_prefill_paged_turbo2"
        );
        assert_eq!(
            funcs(KvCacheDtype::Turbo3, 64)[2],
            "attn_prefill_paged_turbo3_64"
        );
    }
}
