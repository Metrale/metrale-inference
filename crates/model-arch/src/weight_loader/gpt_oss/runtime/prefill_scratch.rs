// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Caller-owned bounded diagnostic chunk scratch; no serving admission.
use super::*;
/// 2026-10-07: Allocate once and explicitly release after all chunk work completes.
/// At most16 tokens of one sequence; never a multi-request batch.
pub struct PrefillScratch {
    allocation: DevicePtr,
    stream: Option<u64>,
    pub(super) norm: DevicePtr,
    pub(super) q: DevicePtr,
    pub(super) k: DevicePtr,
    pub(super) v: DevicePtr,
    pub(super) attn: DevicePtr,
    pub(super) projection: DevicePtr,
    pub(super) logits: DevicePtr,
    pub(super) ids: DevicePtr,
    pub(super) scores: DevicePtr,
    pub(super) accum: DevicePtr,
    pub(super) positions: DevicePtr,
    pub(super) slots: DevicePtr,
    pub(super) lengths: DevicePtr,
    pub(super) tables: DevicePtr,
    pub(super) rows: usize,
    pub(super) max_blocks: usize,
}
impl PrefillScratch {
    /// 2026-10-07: max_blocks is the single-sequence logical page capacity.
    pub fn new(gpu: &dyn GpuBackend, rows: usize, max_blocks: usize) -> Result<Self> {
        ensure!(
            (1..=16).contains(&rows) && (1..=131072).contains(&max_blocks),
            "GPT chunk scratch geometry"
        );
        let widths = [
            5760,
            8192,
            1024,
            1024,
            8192,
            5760,
            64,
            16,
            64,
            16384,
            4,
            8,
            4,
            max_blocks * 4,
        ];
        let sizes: Vec<_> = widths
            .iter()
            .map(|w| (rows * w).next_multiple_of(16))
            .collect();
        let allocation = gpu.alloc(sizes.iter().sum())?;
        let mut offset = 0;
        let p: Vec<_> = sizes
            .iter()
            .map(|n| {
                let p = allocation.offset(offset);
                offset += n;
                p
            })
            .collect();
        Ok(Self {
            allocation,
            stream: None,
            norm: p[0],
            q: p[1],
            k: p[2],
            v: p[3],
            attn: p[4],
            projection: p[5],
            logits: p[6],
            ids: p[7],
            scores: p[8],
            accum: p[9],
            positions: p[10],
            slots: p[11],
            lengths: p[12],
            tables: p[13],
            rows,
            max_blocks,
        })
    }
    /// 2026-10-07: Synchronize before freeing; caller must use the work stream.
    pub fn release(&mut self, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        self.check_stream(stream)?;
        if !self.allocation.is_null() {
            gpu.synchronize(stream)?;
            gpu.free(self.allocation)?;
            self.allocation = DevicePtr::NULL;
        }
        Ok(())
    }
    pub(super) fn admit(&mut self, rows: usize, blocks: usize, stream: u64) -> Result<()> {
        ensure!(
            !self.allocation.is_null()
                && rows > 0
                && rows <= self.rows
                && blocks <= self.max_blocks,
            "GPT chunk exceeds live scratch"
        );
        self.check_stream(stream)?;
        self.stream = Some(stream);
        Ok(())
    }
    fn check_stream(&self, stream: u64) -> Result<()> {
        ensure!(
            self.stream.is_none_or(|s| s == stream),
            "GPT chunk scratch stream changed while work may be pending"
        );
        Ok(())
    }
}
