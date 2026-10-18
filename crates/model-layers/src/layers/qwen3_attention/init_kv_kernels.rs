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
    use metrale_kv_codec::{KvCodecId, PAIR_ALIASES};

    /// 2026-10-18: Every accepted pair (each codec symmetric, then each alias) resolves a write
    /// and a decode kernel at the head dims the shipped attention models use, so none is refused
    /// at layer construction.
    #[test]
    fn every_pair_resolves_at_the_shipped_head_dims() {
        let pairs = KvCodecId::ALL.iter().map(|&c| KvPair::symmetric(c)).chain(
            PAIR_ALIASES
                .iter()
                .map(|a| KvPair::new(a.k, a.v).expect("an alias is an accepted pair")),
        );
        for pair in pairs {
            for hd in [64, 128, 256] {
                let names =
                    kv_kernel_names_for(pair, hd).unwrap_or_else(|e| panic!("{pair} hd={hd}: {e}"));
                assert!(!names.write.func.is_empty() && !names.decode.func.is_empty());
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
