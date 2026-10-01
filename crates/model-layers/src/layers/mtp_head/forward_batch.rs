// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Batched cross-sequence MTP propose. Drafts chain within a
//! sequence but not across sequences, so each draft position is one drafter
//! forward over n sequences, with n-row weight projections (`proj_rows`,
//! `gemm_rows`) and per-row attention.
//!
//! Owner: model-layers (MTP head).
//! Invariants:
//! - `propose_batch_impl` is called only from `DraftProposer::propose_batch`,
//!   after `MtpHead::can_propose_batch` admits n (2 <= n <= `propose_batch_max`).
//! - fc, k and v must be BF16 and q and o BF16 or weight-only NVFP4; any other
//!   projection returns an error.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::DevicePtr;

use super::row_dispatch;
use super::{MtpHead, MtpProposerState, ProjectionWeight};
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::DenseWeight;

mod position;

/// 2026-09-25: Byte offset in `scratch` of the per-row FP32 top-1
/// log-probabilities. The n argmax ids occupy `scratch[0..n*4)`, and
/// n <= `PROPOSE_META_SEQS` keeps them below this offset (2026-09-29: derived; it was 256,
/// which ids overwrite past 64 rows).
pub(super) const LP_SCRATCH_OFF: usize = 4 * super::batch_caps::PROPOSE_META_SEQS;

/// 2026-09-30: The LM-head launch of an n-row draft position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LmHeadRowsArm {
    /// 2026-09-30: `w4a16_gemm_n128_ldb` on the transposed twin.
    TileTwin,
    /// 2026-09-30: `w4a16_gemv_batchm` on `lm_head_batch_kernel(n)`, which takes the
    /// tensor-core entry `gemv_tc::tc_kernel` resolves, if any.
    Gemv,
}

/// 2026-09-30: The arm `forward_batch_position` takes at `n` rows: the tile twin when the
/// tensor-core LM head is not taken (`tc_lm_head`), from 5 rows, with the twin and its kernel
/// present (`twin_ready`); else the batched GEMV. The legacy forward and the circuit executor's
/// plan check both call it.
pub(crate) fn lm_head_rows_arm(n: usize, tc_lm_head: bool, twin_ready: bool) -> LmHeadRowsArm {
    if !tc_lm_head && n >= 5 && twin_ready {
        LmHeadRowsArm::TileTwin
    } else {
        LmHeadRowsArm::Gemv
    }
}

/// 2026-09-30: True when the tensor-core drafter path is on
/// (`ops::dense_gemv_tc::mtp_tc_enabled`) and `gemv_tc::tc_kernel` resolves a
/// `w4a16_gemv_tc8`/`tc16` entry for an `n`-row LM head of `v` rows over `h`.
pub(crate) fn tc_lm_head(
    gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
    n: usize,
    v: u32,
    h: u32,
) -> bool {
    ops::dense_gemv_tc::mtp_tc_enabled() && ops::gemv_tc::tc_kernel(gpu, n as u32, v, h).is_some()
}

impl MtpHead {
    /// 2026-09-25: n-row projection of a BF16 or weight-only NVFP4 weight.
    /// BF16 goes to [`Self::gemm_rows`]. NVFP4 runs one `ops::w4a16_gemv_batchm`
    /// launch on the narrowest resolved `w4a16_gemv_batch{4..8}` tier that
    /// covers `m` (that op takes a tensor-core entry when `gemv_tc::tc_kernel`
    /// resolves one), else one `w4a16_decode_gemv` per row. Any other
    /// precision returns an error.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn proj_rows(
        &self,
        gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
        input: DevicePtr,
        w: &ProjectionWeight,
        output: DevicePtr,
        m: usize,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        match w {
            ProjectionWeight::Bf16(d) => self.gemm_rows(gpu, input, d, output, m, n, k, stream),
            ProjectionWeight::Nvfp4(q) => {
                let kh = self.w4a16_batchm.kernel(m as u32);
                if kh.0 != 0 {
                    return ops::w4a16_gemv_batchm(
                        gpu, kh, input, q, output, m as u32, n, k, stream,
                    );
                }
                for r in 0..m {
                    ops::w4a16_decode_gemv(
                        gpu,
                        self.w4a16_gemv_k,
                        self.w4a16_gemv_sw_k,
                        self.gemv_sw,
                        input.offset(r * k as usize * 2),
                        q,
                        output.offset(r * n as usize * 2),
                        n,
                        k,
                        stream,
                    )?;
                }
                Ok(())
            }
            _ => anyhow::bail!("propose_batch: FP8 projection (can_propose_batch lied)"),
        }
    }

    /// 2026-09-25: n-row BF16 projection. The tensor-core GEMV
    /// (`ops::dense_gemv_tc::try_dense_gemv_tc`, off when `METRALE_NO_MTP_TC`
    /// is set non-empty) runs first. When it launches nothing,
    /// [`row_dispatch::drafter_row_kernel`] picks the batched GEMV, the
    /// pipelined tile GEMM or a per-row GEMV loop. Every arm reads `input` as
    /// `[m, k]` contiguous and writes m contiguous rows of n elements.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gemm_rows(
        &self,
        gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
        input: DevicePtr,
        w: &DenseWeight,
        output: DevicePtr,
        m: usize,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        if ops::dense_gemv_tc::try_dense_gemv_tc(gpu, input, w, output, m as u32, n, k, n, stream)?
        {
            return Ok(());
        }
        match row_dispatch::drafter_row_kernel(
            m,
            n,
            k,
            self.dense_gemv_batchm_k.0 != 0,
            row_dispatch::kv_gemv_pinned(),
            row_dispatch::small_m_tier_off(),
        ) {
            row_dispatch::RowKernel::Batchm => ops::dense_gemv_batchm(
                gpu,
                self.dense_gemv_batchm_k,
                input,
                w,
                output,
                m as u32,
                n,
                k,
                n,
                stream,
            ),
            row_dispatch::RowKernel::Pipelined => ops::dense_gemm_bf16_pipelined(
                gpu,
                self.dense_gemm_pipelined_k,
                input,
                w,
                output,
                m as u32,
                n,
                k,
                stream,
            ),
            row_dispatch::RowKernel::GemvLoop => {
                let gemv_k = self.dense_gemv_k.unwrap();
                for r in 0..m {
                    ops::dense_gemv(
                        gpu,
                        gemv_k,
                        input.offset(r * k as usize * 2),
                        w,
                        output.offset(r * n as usize * 2),
                        n,
                        k,
                        stream,
                    )?;
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{LmHeadRowsArm, lm_head_rows_arm};

    /// 2026-09-30: The legacy dispatch's edges: the tile twin from 5 rows, only off the
    /// tensor-core head and only when ready.
    #[test]
    fn the_lm_head_arm_edges() {
        use LmHeadRowsArm::{Gemv, TileTwin};
        for (n, tc, twin, want) in [
            (4, false, true, Gemv),
            (5, false, true, TileTwin),
            (32, false, true, TileTwin),
            (5, true, true, Gemv),
            (16, true, true, Gemv),
            (17, false, false, Gemv),
            (2, false, false, Gemv),
        ] {
            assert_eq!(
                lm_head_rows_arm(n, tc, twin),
                want,
                "n={n} tc={tc} twin={twin}"
            );
        }
    }
}
