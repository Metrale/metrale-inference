// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: Host rows standing in for `PreemptStubModel`'s device logits
//! buffer, so the verify and bootstrap picks run their device-argmax fast paths
//! and their host fallbacks without a GPU. The rows are BF16, `vocab * 2`
//! bytes apart from `BASE`, as the engine lays out its logits buffer.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;
use std::sync::atomic::{AtomicUsize, Ordering};

/// 2026-10-21: The address the stub reports as its logits buffer.
const BASE: u64 = 0x1000_0000;

pub(in crate::scheduler) struct HostLogits {
    vocab: usize,
    bf16: Vec<u8>,
    /// 2026-10-21: Copies of at least one whole row: the host fallback's D2H,
    /// as opposed to the fast paths' single-logit reads.
    pub(in crate::scheduler) row_copies: AtomicUsize,
}

impl HostLogits {
    pub(in crate::scheduler) fn new(vocab: usize, rows: &[Vec<f32>]) -> Self {
        assert!(rows.iter().all(|r| r.len() == vocab), "every row is one vocab wide");
        Self {
            vocab,
            bf16: rows.iter().flat_map(|r| bf16_bytes(r)).collect(),
            row_copies: AtomicUsize::new(0),
        }
    }

    pub(super) fn vocab(&self) -> usize {
        self.vocab
    }

    pub(super) fn base(&self) -> DevicePtr {
        DevicePtr(BASE)
    }

    pub(super) fn copy(&self, p: DevicePtr, dst: &mut [u8]) -> Result<()> {
        let off = (p.0 - BASE) as usize;
        let src = self
            .bf16
            .get(off..off + dst.len())
            .ok_or_else(|| anyhow::anyhow!("copy past the host logits"))?;
        dst.copy_from_slice(src);
        if dst.len() >= self.vocab * 2 {
            self.row_copies.fetch_add(1, Ordering::SeqCst);
        }
        Ok(())
    }

    pub(super) fn argmax(&self, p: DevicePtr) -> u32 {
        let row = (p.0 - BASE) as usize / (self.vocab * 2);
        let bytes = &self.bf16[row * self.vocab * 2..(row + 1) * self.vocab * 2];
        let vals: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|c| crate::scheduler::helpers::bf16_to_f32(c[0], c[1]))
            .collect();
        metrale_sampling::argmax_first_wins_f32(&vals)
    }
}

/// 2026-10-21: One row as the little-endian BF16 bytes the engine's logits
/// buffer holds (the top half of each f32).
pub(in crate::scheduler) fn bf16_bytes(row: &[f32]) -> Vec<u8> {
    row.iter()
        .flat_map(|&v| {
            let b = v.to_bits();
            [(b >> 16) as u8, (b >> 24) as u8]
        })
        .collect()
}
