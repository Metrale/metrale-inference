// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Bounded chunk scratch for explicit experimental C1 admission.
use super::*;
/// 2026-10-07: Allocate once and explicitly release after all chunk work completes.
/// At most128 explicitly requested tokens of one sequence; never a multi-request batch.
pub struct PrefillScratch {
    allocation: DevicePtr,
    pub(super) tc: Option<super::tc_scratch::TcScratch>,
    stream: Option<u64>,
    pub(super) expert_gemm: metrale_gpu_runtime::gpu::KernelHandle,
    pub(super) expert_reuse_wide: metrale_gpu_runtime::gpu::KernelHandle,
    pub(super) expert_reuse: metrale_gpu_runtime::gpu::KernelHandle,
    pub(super) expert_plan: DevicePtr,
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
            (1..=128).contains(&rows) && (1..=131072).contains(&max_blocks),
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
        let mut sizes: Vec<usize> = widths
            .iter()
            .map(|w| (rows * w).next_multiple_of(16))
            .collect();
        sizes.push((32 * (rows + 1) * 4).next_multiple_of(16));
        Ok(sizes)
    }
    /// 2026-10-07: The same allocation geometry is charged before KV pool sizing.
    pub fn required_bytes(rows: usize, max_blocks: usize) -> Result<usize> {
        Ok(Self::sizes(rows, max_blocks)?.iter().sum())
    }
    pub fn new(gpu: &dyn GpuBackend, rows: usize, max_blocks: usize) -> Result<Self> {
        let sizes = Self::sizes(rows, max_blocks)?;
        let expert_gemm = gpu.kernel("gpt_oss_mxfp4_gemv", "gpt_oss_mxfp4_selected_tokens_bf16")?;
        let expert_bias = gpu.kernel("gpt_oss_expert_ops", "gpt_oss_selected_bias_tokens_bf16")?;
        let expert_reuse = gpu.kernel("gpt_oss_mxfp4_gemv", "gpt_oss_mxfp4_reuse_tokens_bf16")?;
        let expert_reuse_wide =
            gpu.kernel("gpt_oss_mxfp4_gemv", "gpt_oss_mxfp4_reuse_wide_bf16")?;
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
            tc: None,
            stream: None,
            expert_gemm,
            expert_bias,
            expert_reuse,
            expert_reuse_wide,
            expert_plan: p[18],
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
    /// 2026-10-07: Explicit diagnostic reduction policy, only for full128-token chunks.
    /// This never changes serving admission or the default constructor.
    pub fn new_packed_tc_diagnostic(gpu: &dyn GpuBackend, max_blocks: usize) -> Result<Self> {
        let mut scratch = Self::new(gpu, 128, max_blocks)?;
        match super::tc_scratch::TcScratch::new(gpu) {
            Ok(tc) => {
                scratch.tc = Some(tc);
                Ok(scratch)
            }
            Err(error) => {
                scratch.release(gpu, gpu.default_stream())?;
                Err(error)
            }
        }
    }
    /// 2026-10-07: Extra allocation is explicit, derived from the allocator's fixed layout.
    pub fn packed_tc_extra_bytes() -> usize {
        super::tc_scratch::TcScratch::required_bytes()
    }
    /// 2026-10-07: Synchronize before freeing; caller must use the work stream.
    pub fn release(&mut self, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        self.check_stream(stream)?;
        if !self.allocation.is_null() {
            gpu.synchronize(stream)?;
            if let Some(tc) = &mut self.tc {
                tc.release(gpu)?;
            }
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
    /// 2026-10-07: Read the latest successful chunk before scratch is reused by another layer.
    /// The caller supplies that layer state and the original forward stream.
    pub fn diagnostic_router_snapshot(
        &self,
        state: &dyn LayerState,
        start: usize,
        rows: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<DiagnosticTensor>> {
        self.check_stream(stream)?;
        let state = state
            .as_any()
            .downcast_ref::<State>()
            .context("GPT state type")?;
        ensure!(
            !self.allocation.is_null()
                && self.stream == Some(stream)
                && rows > 0
                && rows <= self.rows
                && !state.failed
                && !state.allocation.is_null()
                && start.checked_add(rows) == Some(state.next_position),
            "GPT chunk snapshot is stale, failed or unreadable"
        );
        [
            ("post_attention_norm", self.norm, 2880, "BF16", 2),
            ("router_logits", self.logits, 32, "BF16", 2),
            ("router_scores", self.scores, 32, "BF16", 2),
            ("router_ids", self.ids, 4, "U32", 4),
        ]
        .into_iter()
        .map(|(name, ptr, cols, dtype, width)| {
            let mut bytes = vec![0; rows * cols * width];
            gpu.copy_d2h_on_stream(ptr, &mut bytes, stream)?;
            Ok(DiagnosticTensor {
                name,
                dtype,
                shape: vec![rows, cols],
                bytes,
            })
        })
        .collect()
    }
    fn check_stream(&self, stream: u64) -> Result<()> {
        ensure!(
            self.stream.is_none_or(|s| s == stream),
            "GPT chunk scratch stream changed while work may be pending"
        );
        Ok(())
    }
}
