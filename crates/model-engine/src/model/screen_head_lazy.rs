// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: The lazy logits of a screened K=2 verify (`screen_head.rs`). The verify writes
//! no logits, only a stamp (16 bytes of 0xFF) at the head of the logits buffer; every head call
//! writes row 0 from its first element, so an intact stamp means no head output has replaced
//! the screened verify's. A host read of the logits while the stamp holds gets what the full head
//! would have written, recomputed from the verify's normed rows (kept in the stash) with the full
//! head's arithmetic:
//!
//! - a read of up to [`SINGLE_READ_MAX`] logits (the fast paths' single-logit reads: the
//!   reduce-only penalty check, `logit_bias`): each recomputed on its own (`screen_head_exact_id`,
//!   which checks the stamp on the device and copies the logit instead when it is gone);
//! - any larger read, and a device argmax over the rows: both rows recomputed whole on this rank
//!   (`dense_gemv_bf16_batchm` over the whole head, which every rank holds), once.
//!
//! Owner: model-engine (LM head).
//! Invariants:
//! - Only reads overlapping the verify's rows are served here; a host head call clears the lazy
//!   flag, and a replayed head call breaks the stamp.

use std::sync::atomic::Ordering;

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_layers::layers::ops::{self, SCREEN_STAMP_BYTES};

use super::screen_head::SCREEN_ROWS;
use super::types::TransformerModel;

/// 2026-10-21: Most logits a read may ask for to be recomputed one by one.
const SINGLE_READ_MAX: usize = 4;

impl TransformerModel {
    /// 2026-10-21: A host head call is about to write the logits: nothing is lazy any more.
    pub(crate) fn screen_clear_lazy(&self) {
        if let Some(s) = self.screen_head() {
            s.lazy.store(false, Ordering::Relaxed);
        }
    }

    /// 2026-10-21: Serve a host read of `dst.len()` bytes at `ptr` when it falls on the lazy rows:
    /// `true` when `dst` was filled here, `false` for the caller's plain copy (after recomputing
    /// the rows whole when the read needs them).
    pub(crate) fn screen_lazy_copy(&self, ptr: DevicePtr, dst: &mut [u8]) -> Result<bool> {
        let Some(sh) = self.screen_head() else {
            return Ok(false);
        };
        let base = self.buffers.logits();
        let vocab = self.config.vocab_size;
        let span = SCREEN_ROWS * vocab * 2;
        let overlaps = ptr.0 < base.0 + span as u64 && ptr.0 + dst.len() as u64 > base.0;
        if !sh.lazy.load(Ordering::Relaxed) || !overlaps {
            return Ok(false);
        }
        let first = ptr.0.checked_sub(base.0).map(|o| o as usize);
        let single = match first {
            Some(o)
                if o.is_multiple_of(2) && dst.len().is_multiple_of(2) && o + dst.len() <= span =>
            {
                dst.len() / 2 <= SINGLE_READ_MAX
            }
            _ => false,
        };
        if !single {
            self.screen_materialize()?;
            return Ok(false);
        }
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size as u32;
        for e in 0..dst.len() / 2 {
            let at = first.unwrap_or(0) / 2 + e;
            ops::screen_head_exact_id(
                self.gpu.as_ref(),
                sh.exact_id,
                self.lm_head_weight.weight,
                sh.stash,
                h,
                ((at / vocab) as u32, (at % vocab) as u32, vocab as u32),
                base,
                base,
                sh.probe.offset(e * 2),
                stream,
            )?;
        }
        self.gpu.synchronize(stream)?;
        self.gpu.copy_d2h(sh.probe, dst)?;
        Ok(true)
    }

    /// 2026-10-21: Before a read of whole lazy rows or a device argmax over them: recompute both
    /// rows on this rank while the stamp holds, and clear the flag.
    pub(crate) fn screen_materialize(&self) -> Result<()> {
        let Some(sh) = self.screen_head() else {
            return Ok(());
        };
        if !sh.lazy.swap(false, Ordering::Relaxed) {
            return Ok(());
        }
        let stream = self.gpu.default_stream();
        self.gpu.synchronize(stream)?;
        let base = self.buffers.logits();
        let mut stamp = [0u8; SCREEN_STAMP_BYTES];
        self.gpu.copy_d2h(base, &mut stamp)?;
        if stamp.iter().any(|&b| b != 0xFF) {
            return Ok(());
        }
        let (h, v) = (
            self.config.hidden_size as u32,
            self.config.vocab_size as u32,
        );
        ops::dense_gemv_batchm(
            self.gpu.as_ref(),
            self.dense_gemv_batchm_kernel,
            sh.stash,
            &self.lm_head_weight,
            base,
            SCREEN_ROWS as u32,
            v,
            h,
            v,
            stream,
        )?;
        self.gpu.synchronize(stream)?;
        let n = sh.materialized.fetch_add(1, Ordering::Relaxed) + 1;
        if n.is_power_of_two() {
            tracing::info!(
                "screened LM head: a host read needed whole logits rows; recomputed both on this \
                 rank ({n} times so far, {} screened verifies)",
                sh.screened.load(Ordering::Relaxed)
            );
        }
        Ok(())
    }
}
