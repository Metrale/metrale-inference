// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The KV-codec kernel names of one attention layer, read from the codec kernel
//! table (`metrale_kv_codec::kv_kernel_names`). `init.rs` resolves them into handles and
//! `kernel_requirements.rs` checks them at startup, so both read the one table.
//!
//! Owner: model-layers (attention).
//! Invariants: no kernel name of a KV codec is spelled in this crate's resolvers; a pair the
//! table has no write or decode kernel for is refused with `KvPairUnsupported`.

use anyhow::Result;
use metrale_kv_codec::{KernelName, KvPair, ResolvedKvKernels, kv_kernel_names};

/// 2026-10-18: The kernel names a layer holding `pair` at `head_dim` loads.
pub(super) fn kv_kernel_names_for(
    pair: KvPair,
    head_dim: usize,
) -> Result<ResolvedKvKernels<KernelName>> {
    Ok(kv_kernel_names(pair, u32::try_from(head_dim)?)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use metrale_error::ErrorCode;
    use metrale_kv_codec::{KvCodecId, spec};

    /// 2026-10-18: Every accepted pair (`KvPair::all`) resolves a write and a decode kernel at
    /// each head dim the shipped attention models use that both its codecs can hold
    /// (`fits_head_dim`), and is refused with `KvPairUnsupported` at the others, so a layer is
    /// either built with real kernels or refused at construction. Every codec holds head_dim 128
    /// and 256.
    #[test]
    fn every_pair_resolves_at_the_shipped_head_dims_it_fits() {
        for pair in KvPair::all() {
            for hd in [64u32, 128, 256] {
                let fits = spec(pair.k()).fits_head_dim(hd) && spec(pair.v()).fits_head_dim(hd);
                assert!(fits || hd == 64, "{pair} must hold hd={hd}");
                match kv_kernel_names_for(pair, hd as usize) {
                    Ok(names) => {
                        assert!(fits, "{pair} hd={hd} resolved a head dim it cannot hold");
                        assert!(!names.write.func.is_empty() && !names.decode.func.is_empty());
                    }
                    Err(e) => {
                        assert!(!fits, "{pair} hd={hd}: {e}");
                        let code = e.downcast_ref::<metrale_error::Error>().map(|e| e.code());
                        assert_eq!(code, Some(ErrorCode::KvPairUnsupported), "{pair} hd={hd}");
                    }
                }
            }
        }
    }

    /// 2026-10-18: A head dim that does not fit the table's `u32` is an error, not a wrap.
    #[test]
    fn an_oversized_head_dim_is_refused() {
        let any = KvPair::symmetric(KvCodecId::ALL[0]);
        assert!(kv_kernel_names_for(any, usize::MAX).is_err());
    }
}
