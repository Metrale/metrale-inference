// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `METRALE_GLM_KDA_SCAN_ONORM=1`: where the 64-token chunked scan (`chunk64.rs`)
//! takes a forward through [`Glm5NextKdaLayer::stateful_chunk64`], each slab's scan is
//! `kda_chunk64_scan_onorm` (`kernels/gb10/common/kda_chunk64.cu`, epilogue in
//! `kda_chunk64_onorm.cuh`): the sigmoid-gated output RMSNorm of every row it produces runs in
//! the scan, which writes the BF16 `o_norm_out` rows itself, so the FP32 `core` rows are never
//! written and `back_end` skips the `kda_o_norm_gated_bf16` launch.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants:
//! - Every byte of `o_norm_out` and of the recurrent state equals the plain scan's then the
//!   norm's: the epilogue is the norm's reduction tree and expressions in its order (the kernel
//!   header has the argument); `examples/glm5next_kda_scan_onorm_gate.rs` compares them on a GPU.
//! - Under it `core` holds no rows of the forward; nothing after the scan reads `core` but the
//!   norm, which does not run.
//! - The two column blocks of a head are one thread-block cluster, which the hardware schedules
//!   whole: the per-chunk exchange cannot deadlock at any co-residency, and a cluster that cannot
//!   be placed fails the launch (an error, never a hang).
//! - The tail-merge recurrence (`cut.rs`) keeps the plain scan and the norm launch.
//! - The lever set without the kernel, at a head_dim other than 128, or where the device places
//!   no cluster of the kernel (`GpuBackend::max_active_clusters` 0) fails the forward before
//!   any launch.

use super::*;

/// 2026-10-15: `KDA64_SCAN_ONORM_SMEM`: the scan's shared memory plus the partial-sum buffer
/// (`[parity 2][segment 2][row 64]` floats).
pub const CHUNK64_SCAN_ONORM_SMEM: u32 = CHUNK64_SCAN_SMEM + 2 * 2 * 64 * 4;

/// 2026-10-15: `METRALE_GLM_KDA_SCAN_ONORM`, read once: `1` fuses the norm into the scan, unset
/// or `0` does not.
pub(crate) fn kda_scan_onorm() -> Result<bool> {
    static V: std::sync::OnceLock<Result<bool, String>> = std::sync::OnceLock::new();
    V.get_or_init(|| parse_scan_onorm(std::env::var("METRALE_GLM_KDA_SCAN_ONORM").ok().as_deref()))
        .clone()
        .map_err(anyhow::Error::msg)
}

/// 2026-10-15: [`kda_scan_onorm`]'s parse, pure.
pub(crate) fn parse_scan_onorm(v: Option<&str>) -> Result<bool, String> {
    match v {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(s) => Err(format!("METRALE_GLM_KDA_SCAN_ONORM={s:?}: expected 0 or 1")),
    }
}

impl Glm5NextKdaLayer {
    /// 2026-10-15: The refusals of the fused norm, before any launch: no kernel, a head_dim
    /// other than 128, or a device that cannot place one (1, 2, 1) cluster of it (the launch
    /// would fail; its blocks exchange sums, so they never run apart).
    pub(crate) fn scan_onorm_check(&self, gpu: &dyn GpuBackend) -> Result<()> {
        let s = &self.kernels.seq;
        if s.chunk64_scan_onorm.0 == 0 {
            bail!("METRALE_GLM_KDA_SCAN_ONORM=1: needs kda_chunk64_scan_onorm");
        }
        if self.cfg.head_dim != CHUNK64_D {
            bail!(
                "METRALE_GLM_KDA_SCAN_ONORM=1: head_dim {}, not {}",
                self.cfg.head_dim,
                CHUNK64_D
            );
        }
        let (grid, block) = self.scan_shape();
        let n = gpu
            .max_active_clusters(s.chunk64_scan_onorm, grid, block, CHUNK64_SCAN_ONORM_SMEM)
            .map_err(|e| anyhow::anyhow!("METRALE_GLM_KDA_SCAN_ONORM=1: {e}"))?;
        if n == 0 {
            bail!(
                "METRALE_GLM_KDA_SCAN_ONORM=1: the device places no cluster of \
                 kda_chunk64_scan_onorm (grid {grid:?}, block {block:?}, {} B shared)",
                CHUNK64_SCAN_ONORM_SMEM
            );
        }
        Ok(())
    }

    /// 2026-10-15: The scan's grid (heads x column blocks) and block.
    fn scan_shape(&self) -> ([u32; 3], [u32; 3]) {
        let c = &self.cfg;
        (
            [c.heads as u32, (c.head_dim / CHUNK64_VB) as u32, 1],
            [CHUNK64_SCAN_THREADS, 1, 1],
        )
    }

    /// 2026-10-15: One slab's scan (chunks `c0 .. c0 + n` of rows `0..k`): `kda_chunk64_scan`
    /// into `core`, or with `onorm` `kda_chunk64_scan_onorm` into `o_norm_out`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn chunk64_scan(
        &self,
        gpu: &dyn GpuBackend,
        k: usize,
        c0: usize,
        n: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        onorm: bool,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let s = &self.kernels.seq;
        let (kernel, smem) = if onorm {
            (s.chunk64_scan_onorm, CHUNK64_SCAN_ONORM_SMEM)
        } else {
            (s.chunk64_scan, CHUNK64_SCAN_SMEM)
        };
        let (grid, block) = self.scan_shape();
        let l = KernelLaunch::new(gpu, kernel)
            .grid(grid)
            .block(block)
            .shared_mem(smem)
            .arg_ptr(ws.chunk_u)
            .arg_ptr(ws.chunk_w)
            .arg_ptr(ws.q_f32)
            .arg_ptr(ws.k_f32)
            .arg_ptr(ws.v_f32)
            .arg_ptr(ws.chunk_gc)
            .arg_ptr(state.recurrent);
        let l = if onorm {
            l.arg_ptr(ws.out_gate)
                .arg_ptr(self.weights.o_norm.weight)
                .arg_ptr(ws.o_norm_out)
                .arg_f32(c.rms_norm_eps)
        } else {
            l.arg_ptr(ws.core)
        };
        l.arg_u32(c.heads as u32)
            .arg_u32(k as u32)
            .arg_u32(c0 as u32)
            .arg_u32(n as u32)
            .arg_u32(c.qkv_dim() as u32)
            .launch(stream)
    }
}

/// 2026-10-15: Once per process, where the fused norm first runs: the proof in the serve log
/// that the lever reached the scan (a serve A/B is unreadable without it).
pub(super) fn scan_onorm_readback() {
    static APPLIED: std::sync::Once = std::sync::Once::new();
    APPLIED.call_once(|| {
        tracing::info!(
            "GLM KDA: METRALE_GLM_KDA_SCAN_ONORM=1 applied at {}:{}: kda_chunk64_scan_onorm \
             writes o_norm_out, kda_o_norm_gated_bf16 skipped",
            file!(),
            line!()
        );
    });
}

#[cfg(test)]
#[path = "chunk64_onorm_tests.rs"]
mod tests;
