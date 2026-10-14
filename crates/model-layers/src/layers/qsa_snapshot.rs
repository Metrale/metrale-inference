// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Serialise and restore one sequence's QSA indexer state, the aux blob
//! stored with an SSM prefix-cache snapshot.
//!
//! Owner: model-layers (QSA).
//! Invariants:
//! - `restore_aux` sets the counters only after the size checks pass and both uploads
//!   are issued.
//! - 2026-10-13 (L-K9a): a blob holds `pooled == ingested / ratio` block keys and the
//!   `ingested % ratio` raw keys of the incomplete block; any other shape is refused.

use anyhow::Result;
use metrale_gpu_runtime::gpu::GpuBackend;

use super::{QsaIndexer, QsaSeqState};

impl QsaIndexer {
    /// 2026-10-13: Aux blob: `[ingested u64][pooled u64][block_keys bf16, pooled rows]
    /// [raw tail bf16, ingested % ratio rows]`. The block keys are stored as pooled, so a
    /// restore holds the same bytes as the sequence that was snapshotted (L-K9a: the raw
    /// keys of complete blocks are no longer kept, so they cannot be re-pooled).
    pub fn snapshot_aux(
        &self,
        st: &QsaSeqState,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<u8>> {
        let row = self.hd as usize * 2;
        let (blocks, tail) = self.aux_rows(st.ingested, st.pooled)?;
        let mut blob = vec![0u8; 16 + (blocks + tail) * row];
        blob[..8].copy_from_slice(&(st.ingested as u64).to_le_bytes());
        blob[8..16].copy_from_slice(&(st.pooled as u64).to_le_bytes());
        let (block_part, tail_part) = blob[16..].split_at_mut(blocks * row);
        if blocks > 0 {
            gpu.copy_d2h_on_stream(st.block_keys, block_part, stream)?;
        }
        if tail > 0 {
            gpu.copy_d2h_on_stream(st.raw_tail, tail_part, stream)?;
        }
        Ok(blob)
    }

    /// 2026-09-25: Restore the blob from [`Self::snapshot_aux`] on a prefix-cache hit.
    /// 2026-10-13: Uploads the block keys and the raw tail as they were snapshotted.
    pub fn restore_aux(
        &self,
        st: &mut QsaSeqState,
        blob: &[u8],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(blob.len() >= 16, "QSA aux blob truncated");
        let ingested = u64::from_le_bytes(blob[..8].try_into()?) as usize;
        let pooled = u64::from_le_bytes(blob[8..16].try_into()?) as usize;
        let row = self.hd as usize * 2;
        anyhow::ensure!(ingested <= self.max_tokens, "QSA aux exceeds key cache");
        let (blocks, tail) = self.aux_rows(ingested, pooled)?;
        anyhow::ensure!(
            blob.len() == 16 + (blocks + tail) * row,
            "QSA aux blob size mismatch: {} B for {ingested} tokens",
            blob.len()
        );
        let (block_part, tail_part) = blob[16..].split_at(blocks * row);
        if blocks > 0 {
            gpu.copy_h2d_async(block_part, st.block_keys, stream)?;
        }
        if tail > 0 {
            gpu.copy_h2d_async(tail_part, st.raw_tail, stream)?;
        }
        st.ingested = ingested;
        st.pooled = pooled;
        Ok(())
    }

    /// 2026-10-13: The (block-key, raw-tail) row counts of a state with these counters;
    /// errors unless every complete block is pooled.
    fn aux_rows(&self, ingested: usize, pooled: usize) -> Result<(usize, usize)> {
        let ratio = self.ratio as usize;
        anyhow::ensure!(
            pooled == ingested / ratio,
            "QSA aux: {pooled} blocks pooled of {ingested} tokens (ratio {ratio})"
        );
        Ok((pooled, ingested % ratio))
    }
}
