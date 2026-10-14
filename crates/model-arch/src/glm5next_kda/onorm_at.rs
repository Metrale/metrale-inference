// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: `METRALE_GLM_KDA_ONORM_IN_SCAN=1`: where the 64-token chunked scan
//! (`chunk64.rs`) takes a forward, the scan applies the KDA output's sigmoid-gated RMSNorm
//! itself (`kda_chunk64_scan_onorm` in `kernels/gb10/common/kda_chunk64.cu`, with
//! `kda_chunk64_onorm.cuh`). Each chunk's FP32 core rows are normalised while they are still in
//! L2, and the back end runs only `o_proj`. Without it, one `kda_o_norm_gated_bf16` launch reads
//! every core row back from DRAM. Also the back end ([`Glm5NextKdaLayer::back_end`]) for
//! either placement.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants:
//! - Every byte of `o_norm_out` equals the o_norm launch's: o_norm's arithmetic, operation for
//!   operation, in the same compile flags. `core`, the recurrent state and the slab factors are
//!   the plain scan's bytes. `examples/glm5next_kda_onorm_in_scan_parity.rs` compares them on a
//!   GPU.
//! - One decision per forward ([`Glm5NextKdaLayer::norm_at_for`]), handed to both the scan
//!   (which picks the o_norm entry) and the back end (which then skips the o_norm launch).
//! - The lever set without the kernel fails the forward before any launch.
//! - `ws.onorm_count` is zero between launches: it is zeroed when the workspace is created,
//!   and each chunk's last block puts its counter back to zero.

use super::*;

/// 2026-10-14: Where a forward applies the sigmoid-gated RMSNorm of the KDA output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NormAt {
    /// `kda_o_norm_gated_bf16` in the back end.
    Back,
    /// Inside the 64-token scan (`kda_chunk64_scan_onorm`).
    Scan,
}

/// 2026-10-14: `METRALE_GLM_KDA_ONORM_IN_SCAN`, read once: `1` moves the o_norm, unset or `0`
/// does not.
pub(crate) fn kda_onorm_in_scan() -> Result<bool> {
    static V: std::sync::OnceLock<Result<bool, String>> = std::sync::OnceLock::new();
    V.get_or_init(|| {
        parse_onorm_in_scan(
            std::env::var("METRALE_GLM_KDA_ONORM_IN_SCAN")
                .ok()
                .as_deref(),
        )
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

/// 2026-10-14: [`kda_onorm_in_scan`]'s parse, pure.
pub(crate) fn parse_onorm_in_scan(v: Option<&str>) -> Result<bool, String> {
    match v {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(s) => Err(format!(
            "METRALE_GLM_KDA_ONORM_IN_SCAN={s:?}: expected 0 or 1"
        )),
    }
}

/// 2026-10-14: `[slab chunk][heads][64 rows][head_dim / 32]` FP32 warp sums and
/// `[slab chunk][heads]` u32 arrival counters (`kda_chunk64_onorm.cuh`), in elements.
pub(super) fn onorm_scratch_elems(cfg: &Glm5NextKdaConfig) -> (usize, usize) {
    let n = chunk64::CHUNK64_SLAB * cfg.heads;
    (n * chunk64::CHUNK64 * (cfg.head_dim / 32), n)
}

impl Glm5NextKdaLayer {
    /// 2026-10-14: Where a forward on the 64-token scan (`chunked64`) applies the o_norm: in the
    /// scan when the lever is set (an error without the kernel), else in the back end.
    pub(crate) fn norm_at_for(&self, chunked64: bool) -> Result<NormAt> {
        if !chunked64 || !kda_onorm_in_scan()? {
            return Ok(NormAt::Back);
        }
        self.onorm_in_scan_check()?;
        Ok(NormAt::Scan)
    }

    /// 2026-10-14: The refusal of [`NormAt::Scan`], before any launch.
    pub(crate) fn onorm_in_scan_check(&self) -> Result<()> {
        if self.kernels.seq.chunk64_scan_onorm.0 == 0 {
            bail!("METRALE_GLM_KDA_ONORM_IN_SCAN=1: needs kda_chunk64_scan_onorm");
        }
        Ok(())
    }

    /// 2026-10-14: The scan's o_norm arguments at [`NormAt::Scan`], after its own: the output
    /// gate, the norm weight, the output rows (all three from workspace row `row0`), the warp
    /// sums, the counters and eps; nothing at [`NormAt::Back`].
    pub(super) fn onorm_args<'a>(
        &self,
        l: KernelLaunch<'a>,
        ws: &Glm5NextKdaWorkspace,
        row0: usize,
        norm: NormAt,
    ) -> KernelLaunch<'a> {
        let at = row0 * self.cfg.qkv_dim() * 2;
        match norm {
            NormAt::Back => l,
            NormAt::Scan => l
                .arg_ptr(ws.out_gate.offset(at))
                .arg_ptr(self.weights.o_norm.weight)
                .arg_ptr(ws.o_norm_out.offset(at))
                .arg_ptr(ws.onorm_part)
                .arg_ptr(ws.onorm_count)
                .arg_f32(self.cfg.rms_norm_eps),
        }
    }

    /// 2026-09-25: Sigmoid-gated RMSNorm, then `o_proj`.
    /// 2026-10-14: At [`NormAt::Scan`] the scan wrote `o_norm_out`, so only `o_proj` runs.
    pub(super) fn back_end(
        &self,
        gpu: &dyn GpuBackend,
        t: usize,
        ws: &Glm5NextKdaWorkspace,
        norm: NormAt,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        if norm == NormAt::Back {
            KernelLaunch::new(gpu, self.kernels.o_norm)
                .grid([(t * c.heads) as u32, 1, 1])
                .block([c.head_dim as u32, 1, 1])
                .arg_ptr(ws.core)
                .arg_ptr(ws.out_gate)
                .arg_ptr(self.weights.o_norm.weight)
                .arg_ptr(ws.o_norm_out)
                .arg_u32(c.head_dim as u32)
                .arg_f32(c.rms_norm_eps)
                .launch(stream)?;
        }
        self.gemm(
            gpu,
            ws.o_norm_out,
            &self.weights.o_proj,
            ws.final_out,
            t,
            c.hidden,
            c.qkv_dim(),
            stream,
        )
    }
}

#[cfg(test)]
#[path = "onorm_at_tests.rs"]
mod tests;
