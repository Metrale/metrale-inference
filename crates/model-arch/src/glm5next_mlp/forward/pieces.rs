// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The two launches of a routed-MoE site (`forward_moe_pieces`) whose arithmetic
//! depends on the row count, run once per piece of rows: the router logits and the shared
//! expert.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - A piece of `r` rows gets exactly the launches a `forward_moe` of `r` rows makes for them:
//!   the router as one cuBLASLt GEMM above `DENSE_GEMV_BATCHM_MAX_M` rows (with
//!   `cublas_wide_proj`), else one M = 1 GEMV per row; the shared expert as `forward_dense` over
//!   the piece. cuBLASLt picks its algorithm by the row count, so a row matches the
//!   single-sequence call only when its piece has that call's row count.

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::super::weights::Glm5NextMoeWeights;
use super::super::{Glm5NextMlpConfig, Glm5NextMlpKernels};
use super::Glm5NextMlpWorkspace;
use super::dense::forward_dense;
use super::launch::gemm;

/// 2026-10-09: Errors when `pieces` do not split `rows` rows into non-empty pieces.
pub(crate) fn check_pieces(rows: usize, pieces: &[usize]) -> Result<()> {
    if pieces.contains(&0) || pieces.iter().sum::<usize>() != rows {
        bail!("GLM MoE: pieces {pieces:?} do not split {rows} rows");
    }
    Ok(())
}

/// 2026-10-09: `[rows, num_experts]` F32 logits into `ws.logits`, piece by piece; the caller
/// has checked the pieces (`check_pieces`).
#[allow(clippy::too_many_arguments)]
pub(super) fn router_logits(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    w: &Glm5NextMoeWeights,
    x: DevicePtr,
    ws: &Glm5NextMlpWorkspace,
    pieces: &[usize],
    stream: u64,
) -> Result<()> {
    let mut r0 = 0usize;
    for &n in pieces {
        if n > metrale_model_layers::layers::ops::DENSE_GEMV_BATCHM_MAX_M as usize
            && crate::glm5next_layer::cublas_wide_proj()
        {
            metrale_model_layers::layers::ops::cublas_bf16_proj_dense_f32_out(
                x.offset(r0 * cfg.hidden * 2),
                w.router,
                ws.logits.offset(r0 * cfg.num_experts * 4),
                n as u32,
                cfg.num_experts as u32,
                cfg.hidden as u32,
                stream,
            )?;
        } else {
            for r in r0..r0 + n {
                gemm(
                    gpu,
                    k.gemm_f32,
                    k.gemv_f32,
                    KernelHandle(0),
                    x.offset(r * cfg.hidden * 2),
                    w.router,
                    ws.logits.offset(r * cfg.num_experts * 4),
                    1,
                    cfg.num_experts,
                    cfg.hidden,
                    stream,
                )?;
            }
        }
        r0 += n;
    }
    Ok(())
}

/// 2026-10-09: The shared expert over `x` into `ws.shared_out`, piece by piece; the caller has
/// checked the pieces (`check_pieces`).
#[allow(clippy::too_many_arguments)]
pub(super) fn shared_expert(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    w: &Glm5NextMoeWeights,
    x: DevicePtr,
    ws: &Glm5NextMlpWorkspace,
    pieces: &[usize],
    stream: u64,
) -> Result<()> {
    let mut r0 = 0usize;
    for &n in pieces {
        forward_dense(
            gpu,
            k,
            cfg,
            &w.shared,
            cfg.local_shared_intermediate,
            x.offset(r0 * cfg.hidden * 2),
            ws.shared_out.offset(r0 * cfg.hidden * 2),
            n,
            ws,
            stream,
        )?;
        r0 += n;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::check_pieces;

    /// 2026-10-09: Pieces must cover the rows exactly, with none empty; otherwise the site
    /// would launch the router or the shared expert over rows it does not own.
    #[test]
    fn pieces_must_split_the_rows_exactly() {
        assert!(check_pieces(396, &[198, 198]).is_ok());
        assert!(check_pieces(7, &[7]).is_ok());
        assert!(check_pieces(396, &[198, 197]).is_err());
        assert!(check_pieces(396, &[198, 199]).is_err());
        assert!(check_pieces(5, &[5, 0]).is_err());
    }
}
