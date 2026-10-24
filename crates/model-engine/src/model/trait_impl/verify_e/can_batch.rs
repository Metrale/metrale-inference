// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: Whether a set of sequences can take the batched verify.
//!
//! Owner: model-engine (speculative verify).
//! Invariants: none beyond the types.

use super::super::super::types::TransformerModel;

impl TransformerModel {
    /// 2026-09-25: Whether `decode_verify_batched_dispatch` can take `ks.len()`
    /// sequences with `ks[i]` rows each. The scheduler verifies the sequences
    /// one at a time when it returns false.
    ///
    /// It requires `2 <= n <= VERIFY_WY_TABLE_SEQS`, `Σ ks <= VERIFY_ROW_CAP`,
    /// no EP comm backend unless every layer answers `batch_verify_across_ranks`
    /// (2026-10-09), a verify hidden stash (allocated only with a
    /// proposer), no layer that declines `decode_verify_multi`, no HSS
    /// (`cache_blocks_per_seq`), and, with an adapter loaded, no
    /// `METRALE_LORA_NO_BATCH_VERIFY=1`. Without DFlash every `ks[i]` must be
    /// in 2..=4. With DFlash the batch must be uniform with k in
    /// 2..=`dflash_kgamma`, and `n * dflash_kgamma` must fit
    /// `dflash_hidden_save_rows`.
    pub(in super::super) fn can_batch_verify_dispatch(&self, ks: &[usize]) -> bool {
        let n = ks.len();
        let shape_ok = if self.dflash_hidden_save.is_some() {
            // 2026-09-25: DFlash: one k for the whole batch, any value a
            // capture band holds, and a full band per sequence.
            self.dflash_kgamma >= 2
                && ks.iter().all(|&k| (2..=self.dflash_kgamma).contains(&k))
                && ks.iter().all(|&k| k == ks[0])
                && n * self.dflash_kgamma <= self.dflash_hidden_save_rows
        } else {
            ks.iter().all(|k| (2..=4).contains(k))
        };
        (2..=metrale_model_layers::layer::VERIFY_WY_TABLE_SEQS).contains(&n)
            && shape_ok
            && ks.iter().sum::<usize>() <= super::super::verify_e2::VERIFY_ROW_CAP
            // 2026-10-09: Multi-rank only under the slot-addressed worker protocol (v2) and
            // when every layer runs its batched verify the same way on every rank; the
            // scheduler then announces the batch to the workers (`EP_CMD_VERIFY_BATCH`).
            && (self.comm.is_none()
                || (self.ep_protocol_v2
                    && self.layers.iter().all(|l| l.batch_verify_across_ranks())))
            // 2026-09-30: Under `--forward circuit` only when the executor compiles batched
            // verifies (`CircuitExec::verify_batch`).
            && self
                .circuit
                .read()
                .as_ref()
                .is_none_or(|e| {
                    e.verify_batch
                        .as_ref()
                        .is_some_and(|v| ks.iter().sum::<usize>() as u64 <= v.max_rows)
                })
            && !(self.lora.is_some() && metrale_model_layers::lora::no_batch_verify(&self.settings))
            && !self.verify_hidden_stash.is_null()
            && !self
                .layers
                .iter()
                .any(|l| l.decode_verify_multi_unsupported())
            // 2026-09-25: HSS: `decode_multi_seq` takes no disk block ids, so
            // history offloaded to disk would be missing.
            && self
                .kv_cache
                .lock()
                .config()
                .cache_blocks_per_seq
                .is_none()
    }
}
