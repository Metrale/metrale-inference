// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: `METRALE_GLM_KDA_CHUNK64=<rows>`: a prefill sub-chunk of at least `rows` rows of
//! one sequence (no per-row snapshot) steps the KDA recurrence in 64-token chunks with the
//! chunk-level products on tensor cores: `kda_chunk64_prepare`
//! (`kernels/gb10/common/kda_chunk64_prepare.cu`: every chunk's WY factors in parallel, FP32 on
//! CUDA cores) then `kda_chunk64_scan` (`kda_chunk64.cu`: one scan over the chunks per head and
//! 64-value-column block on BF16 mma.sync with FP32 accumulation and an FP32 state) instead of
//! token by token (~1.7 us per token per layer) or in 16-token FP32 chunks (`chunk16.rs`,
//! ~1.15 us). Takes precedence over `METRALE_GLM_KDA_CHUNK_SCAN` and
//! `METRALE_GLM_KDA_CHUNK_PREFILL` where it applies. The conv is the per-token path's
//! (`conv_tokens`), so the scan reads the q/k/v/gate/beta rows the per-token walk reads and
//! writes the same `core` rows.
//!
//! Numerics: the same recurrence in the chunk-level WY form, the scan's products with BF16
//! operands (as FLA's chunked KDA), so the outputs and the carried state differ from the per-token
//! walk at BF16 rounding level (~2e-3 relative RMS); opt-in and accuracy-gated
//! (`glm5next_kda_chunk64_bench`). A sub-chunk narrower than `rows`, a verify, and decode are
//! unchanged.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants:
//! - Runs only where [`Glm5NextKdaLayer::chunk64_ready`] holds; the scratch is the workspace's
//!   chunked-prefill buffers, each `t_pad * qkv` FP32 (`t_pad` a multiple of [`CHUNK64`]), laid
//!   out per (slab-local chunk, head) as `kda_chunk64.cuh` says: U (`chunk_u`, FP32 `[C, D]`),
//!   W, Qg, Kd (`chunk_w`, `q_f32`, `k_f32`, BF16 `[C, D]` in the first half), P (`v_f32`, BF16
//!   `[C, C]`) and A + decay (`chunk_gc`, FP32 `[C, C]` then `[D]`, [`CHUNK64_AD_STRIDE`] floats
//!   apart; A is the prepare's own read-back scratch).
//! - The chunks run in slabs of [`CHUNK64_SLAB`]: one prepare and one scan launch per slab over
//!   the same scratch, so a slab's factors (4 chunks x 22 heads x 88.5 KiB = 7.8 MiB; 4 measured
//!   best of 4/8/16/32 for the scan) stay in GB10's 24 MiB L2 between the two instead of
//!   round-tripping through DRAM; the scan carries
//!   the state across slabs in the layer's state buffer.

use super::decode::RowIo;
use super::*;

/// 2026-10-12: `KDA64_C` and `KDA64_D` of `kda_chunk64.cuh`.
pub const CHUNK64: usize = 64;
pub const CHUNK64_D: usize = 128;
/// 2026-10-12: `KDA64_SLAB`: chunks per prepare + scan launch pair (L2-sized, see above).
pub const CHUNK64_SLAB: usize = 4;
/// 2026-10-12: `KDA64_AD_STRIDE`: floats per (chunk, head) in the A + decay buffer.
pub const CHUNK64_AD_STRIDE: usize = CHUNK64 * CHUNK64 + CHUNK64_D;
/// 2026-10-12: `KDA64_VB`, `KDA64_PREP_THREADS`, `KDA64_SCAN_THREADS`.
pub const CHUNK64_VB: usize = 64;
pub const CHUNK64_PREP_THREADS: u32 = 256;
pub const CHUNK64_SCAN_THREADS: u32 = 256;
/// 2026-10-12: `KDA64_PREP_SMEM` and `KDA64_SCAN_SMEM` in bytes.
pub const CHUNK64_PREP_SMEM: u32 =
    (CHUNK64 * CHUNK64_D * 4 + CHUNK64 * CHUNK64_D * 2 + CHUNK64 * 4) as u32;
pub const CHUNK64_SCAN_SMEM: u32 = ((3 * CHUNK64 * (CHUNK64_D + 8)
    + CHUNK64 * (CHUNK64 + 8)
    + CHUNK64_VB * (CHUNK64_D + 8)
    + CHUNK64_VB * (CHUNK64 + 8))
    * 2
    + CHUNK64_D * 4) as u32;

/// 2026-10-12: `METRALE_GLM_KDA_CHUNK64`, read once: `None` when unset or `0`, else the narrowest
/// prefill sub-chunk the 64-token scan takes. Below [`CHUNK64`], or unparsable, is an error.
pub fn kda_chunk64_min_rows() -> Result<Option<usize>> {
    static V: std::sync::OnceLock<Result<Option<usize>, String>> = std::sync::OnceLock::new();
    V.get_or_init(|| parse_min_rows64(std::env::var("METRALE_GLM_KDA_CHUNK64").ok().as_deref()))
        .clone()
        .map_err(anyhow::Error::msg)
}

/// 2026-10-12: [`kda_chunk64_min_rows`]'s parse, pure.
pub(crate) fn parse_min_rows64(v: Option<&str>) -> Result<Option<usize>, String> {
    match v {
        None | Some("0") => Ok(None),
        Some(s) => match s.parse::<usize>() {
            Ok(r) if r >= CHUNK64 => Ok(Some(r)),
            _ => Err(format!(
                "METRALE_GLM_KDA_CHUNK64={s:?}: expected 0 or a row count of at least {CHUNK64}"
            )),
        },
    }
}

impl Glm5NextKdaLayer {
    /// 2026-10-12: Whether this target and config can run [`Self::stateful_chunk64`]: the
    /// per-sequence conv kernels, both chunk kernels, and head_dim 128.
    pub(super) fn chunk64_ready(&self) -> bool {
        let s = &self.kernels.seq;
        s.conv_tokens.0 != 0
            && s.conv_window.0 != 0
            && s.chunk64_prepare.0 != 0
            && s.chunk64_scan.0 != 0
            && self.cfg.head_dim == CHUNK64_D
            && self.cfg.conv_kernel <= 4
    }

    /// 2026-10-12: Whether `k` rows of one sequence at floor `min_rows`
    /// ([`kda_chunk64_min_rows`]) take [`Self::stateful_chunk64`] where no per-row snapshot is
    /// asked for: the floor is set and reached, and [`Self::chunk64_ready`] holds. `decode_k_modes`
    /// and the prefill routing in `glm5next_layer` share this one predicate.
    pub fn chunk64_applies(&self, k: usize, min_rows: Option<usize>) -> bool {
        min_rows.is_some_and(|r| k >= r) && self.chunk64_ready()
    }

    /// 2026-10-12: Workspace rows `0..k` of one sequence through the conv, then the 64-token
    /// chunked recurrence into `core`, advancing `state` past all of them.
    /// 2026-10-14: `METRALE_GLM_KDA_CHUNK64_CONV` (`chunk64_conv.rs`) picks `fused`.
    pub(super) fn stateful_chunk64(
        &self,
        gpu: &dyn GpuBackend,
        k: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        pack: Pack,
        stream: u64,
    ) -> Result<()> {
        let fused = self.chunk64_conv_on()?;
        self.stateful_chunk64_with(gpu, k, state, ws, pack, fused, stream)
    }

    /// 2026-10-14: [`Self::stateful_chunk64`] with the conv in front of the prepare (`fused`:
    /// `kda_chunk64_prepare_conv` per slab, the window advanced after the last) or as its own
    /// launch.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn stateful_chunk64_with(
        &self,
        gpu: &dyn GpuBackend,
        k: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        pack: Pack,
        fused: bool,
        stream: u64,
    ) -> Result<()> {
        if !self.chunk64_ready() || k == 0 || k > ws.max_tokens {
            bail!("KDA: the 64-token chunked scan cannot run {k} rows here");
        }
        if fused {
            self.chunk64_conv_check()?;
        }
        let c = &self.cfg;
        let (qkv, d, h) = (c.qkv_dim(), c.head_dim, c.heads);
        let nchunks = k.div_ceil(CHUNK64);
        // 2026-10-12: Each scratch buffer holds `t_pad * qkv` floats; a slab's widest factor
        // needs `slab * heads * 64 * head_dim` of them.
        if nchunks * CHUNK64 > ws.t_pad {
            bail!(
                "KDA 64-token chunked scan: {nchunks} chunks of {CHUNK64} exceed the workspace's \
                 {} rows",
                ws.t_pad
            );
        }
        // 2026-10-12: Once per process: the proof in the serve log that the lever reached this
        // path (a serve A/B is unreadable without it).
        static APPLIED: std::sync::Once = std::sync::Once::new();
        APPLIED.call_once(|| {
            tracing::info!(
                "GLM KDA: METRALE_GLM_KDA_CHUNK64 applied at {}:{}: {k} rows as {nchunks} chunks \
                 of {CHUNK64} in slabs of {CHUNK64_SLAB}, conv fused into the prepare: {fused}",
                file!(),
                line!()
            );
        });
        if !fused {
            self.conv_tokens(gpu, k, state, ws, pack, stream)?;
        }
        let io = RowIo::workspace(c, ws, 0);
        for c0 in (0..nchunks).step_by(CHUNK64_SLAB) {
            let n = CHUNK64_SLAB.min(nchunks - c0);
            if fused {
                self.chunk64_prepare_conv(gpu, k, c0, n, state, ws, pack, stream)?;
            } else {
                self.chunk64_prepare(gpu, k, c0, n, ws, stream)?;
            }
            KernelLaunch::new(gpu, self.kernels.seq.chunk64_scan)
                .grid([h as u32, (d / CHUNK64_VB) as u32, 1])
                .block([CHUNK64_SCAN_THREADS, 1, 1])
                .shared_mem(CHUNK64_SCAN_SMEM)
                .arg_ptr(ws.chunk_u)
                .arg_ptr(ws.chunk_w)
                .arg_ptr(ws.q_f32)
                .arg_ptr(ws.k_f32)
                .arg_ptr(ws.v_f32)
                .arg_ptr(ws.chunk_gc)
                .arg_ptr(state.recurrent)
                .arg_ptr(io.core)
                .arg_u32(h as u32)
                .arg_u32(k as u32)
                .arg_u32(c0 as u32)
                .arg_u32(n as u32)
                .arg_u32(qkv as u32)
                .launch(stream)?;
        }
        if fused {
            self.chunk64_conv_window(gpu, k, state, ws, pack, stream)?;
        }
        Ok(())
    }

    /// 2026-10-14: One slab's `kda_chunk64_prepare` over the conv's rows (chunks `c0 .. c0 + n`).
    fn chunk64_prepare(
        &self,
        gpu: &dyn GpuBackend,
        k: usize,
        c0: usize,
        n: usize,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let (qkv, cd, d, h) = (c.qkv_dim(), c.conv_dim(), c.head_dim, c.heads);
        let io = RowIo::workspace(c, ws, 0);
        let scale = 1.0 / (d as f32).sqrt();
        KernelLaunch::new(gpu, self.kernels.seq.chunk64_prepare)
            .grid([n as u32, h as u32, 1])
            .block([CHUNK64_PREP_THREADS, 1, 1])
            .shared_mem(CHUNK64_PREP_SMEM)
            .arg_ptr(io.conv_out)
            .arg_ptr(io.conv_out.offset(qkv * 2))
            .arg_ptr(io.conv_out.offset(qkv * 4))
            .arg_ptr(io.gate)
            .arg_ptr(io.beta)
            .arg_ptr(ws.chunk_u)
            .arg_ptr(ws.chunk_w)
            .arg_ptr(ws.q_f32)
            .arg_ptr(ws.k_f32)
            .arg_ptr(ws.v_f32)
            .arg_ptr(ws.chunk_gc)
            .arg_u32(h as u32)
            .arg_u32(k as u32)
            .arg_u32(c0 as u32)
            .arg_u32(cd as u32)
            .arg_u32(qkv as u32)
            .arg_f32(scale)
            .launch(stream)
    }
}

// 2026-10-14: `METRALE_GLM_KDA_CHUNK64_CONV`: the conv inside the prepare.
#[path = "chunk64_conv.rs"]
pub(crate) mod conv;

#[cfg(test)]
#[path = "chunk64_tests.rs"]
mod tests;
