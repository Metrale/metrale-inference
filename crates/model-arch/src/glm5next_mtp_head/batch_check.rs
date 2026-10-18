// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: The batched propose's parity check (`METRALE_GLM_MTP_PROPOSE_BATCH_CHECK`), split
//! out of `batch.rs`.
//!
//! Owner: model-arch (GLM-5.3 MTP drafter).
//! Invariants: none beyond the types.

use super::*;

impl Glm5NextMtpHead {
    /// 2026-10-11: `METRALE_GLM_MTP_PROPOSE_BATCH_CHECK=1`: rewind every sequence to the
    /// drafter row the batched propose started from (`starts`, `batch::start_row`; NOT the
    /// target position, which a drafter without its prompt context lags) and run the
    /// per-sequence `propose` over the same inputs, then count the
    /// sequences whose drafts differ from the batched ones. The per-sequence rows overwrite
    /// the batched rows at the same slots, so the drafter state afterwards is the per-sequence
    /// one. Every rank runs it (it is inside the shared forward), so its collectives pair.
    /// Under the BF16 head (`METRALE_GLM_MTP_HEAD_FP8=0`) a mismatch is a defect; under the
    /// FP8 batch head some are expected.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn check_against_rows(
        &self,
        batched: &[Vec<u32>],
        last_tokens: &[u32],
        target_hiddens: &[DevicePtr],
        positions: &[usize],
        starts: &[usize],
        num_drafts: usize,
        sts: &mut [&mut Glm5NextMtpProposerState],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQS: AtomicU64 = AtomicU64::new(0);
        static DIFF: AtomicU64 = AtomicU64::new(0);
        let mut diff = 0u64;
        for (i, st) in sts.iter_mut().enumerate() {
            st.dsa.rewind_to(starts[i])?;
            st.seq_len = starts[i];
            let rows = self.propose(
                last_tokens[i],
                target_hiddens[i],
                positions[i],
                num_drafts,
                &mut **st,
                ctx,
                stream,
                None,
                None,
                None,
            )?;
            // 2026-10-11: Under the confidence stop the per-sequence propose ends its own chain
            // where the batched one keeps drafting for the scheduler to truncate, so the
            // shorter chain is compared with the other's prefix.
            let k = rows.len().min(batched[i].len());
            if rows[..k] != batched[i][..k] {
                diff += 1;
                tracing::warn!(
                    "GLM MTP batch check: sequence {i} of {} at row {}: batched {:?} vs \
                     per-sequence {rows:?}",
                    batched.len(),
                    positions[i],
                    batched[i]
                );
            }
        }
        let seqs = SEQS.fetch_add(batched.len() as u64, Ordering::Relaxed) + batched.len() as u64;
        let total = DIFF.fetch_add(diff, Ordering::Relaxed) + diff;
        if diff > 0 || seqs.is_multiple_of(256) {
            tracing::info!("GLM MTP batch check: {total} of {seqs} sequences differ so far");
        }
        Ok(())
    }
}
