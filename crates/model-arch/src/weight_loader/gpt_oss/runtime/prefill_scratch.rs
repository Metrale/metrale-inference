// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Bounded chunk scratch for explicit experimental C1 admission.
use super::*;
/// 2026-10-07: Allocate once and explicitly release after all chunk work completes.
/// At most16 tokens of one sequence; never a multi-request batch.
pub struct PrefillScratch {
    allocation: DevicePtr,
    stream: Option<u64>,
    pub(super) expert_gemm: metrale_gpu_runtime::gpu::KernelHandle,
    pub(super) expert_bias: metrale_gpu_runtime::gpu::KernelHandle,
    pub(super) gate_up: DevicePtr,
    pub(super) activation: DevicePtr,
    pub(super) selected: DevicePtr,
    pub(super) moe: DevicePtr,
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
    fn sizes(rows: usize, max_blocks: usize) -> Result<Vec<usize>> {
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
            4 * 11520,
            4 * 5760,
            4 * 5760,
            5760,
        ];
        Ok(widths
            .iter()
            .map(|w| (rows * w).next_multiple_of(16))
            .collect())
    }
    /// 2026-10-07: The same allocation geometry is charged before KV pool sizing.
    pub fn required_bytes(rows: usize, max_blocks: usize) -> Result<usize> {
        Ok(Self::sizes(rows, max_blocks)?.iter().sum())
    }
    pub fn new(gpu: &dyn GpuBackend, rows: usize, max_blocks: usize) -> Result<Self> {
        let sizes = Self::sizes(rows, max_blocks)?;
        let expert_gemm = gpu.kernel("gpt_oss_mxfp4_gemv", "gpt_oss_mxfp4_selected_tokens_bf16")?;
        let expert_bias = gpu.kernel("gpt_oss_expert_ops", "gpt_oss_selected_bias_tokens_bf16")?;
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
            expert_gemm,
            expert_bias,
            gate_up: p[14],
            activation: p[15],
            selected: p[16],
            moe: p[17],
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
    // 2026-10-07: Teardown must drain the actual work stream, including failed calls.
    pub(super) fn release_bound(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        self.release(gpu, self.stream.unwrap_or_else(|| gpu.default_stream()))
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
