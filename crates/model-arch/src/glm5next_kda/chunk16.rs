// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: `METRALE_GLM_KDA_CHUNK_SCAN=<rows>`: a prefill sub-chunk of at least `rows` rows
//! of one sequence (no per-row snapshot) steps the KDA recurrence in 16-token chunks
//! (`kernels/gb10/common/kda_chunk16.cu`: every chunk's WY factors in parallel, then one scan
//! over the chunks per head and value-column block) instead of token by token
//! (`kda_recurrent_decode_bf16_seq_reg`, whose per-token latency is ~1.5 us per layer on GB10,
//! O(T) sequential). The conv is the per-token path's (`conv_tokens`), so the scan reads the
//! q/k/v/gate/beta rows the per-token walk reads and writes the same `core` rows.
//!
//! Numerics: FP32 throughout, the same recurrence in another order (chunk-level WY form), so the
//! outputs and the carried state differ from the per-token walk at FP32 rounding level; opt-in
//! and accuracy-gated. A sub-chunk narrower than `rows`, a verify, and decode are unchanged.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants:
//! - Runs only where [`Glm5NextKdaLayer::chunk16_ready`] holds; the scratch is the workspace's
//!   chunked-prefill buffers (`chunk_u`, `chunk_w`, `q_f32`, `k_f32`, `v_f32`, `chunk_gc`), each
//!   `t_pad * qkv` FP32, which hold `ceil(T / 16) * heads * 16 * head_dim` floats.

use super::decode::RowIo;
use super::*;

/// 2026-10-10: `KDA16_C` and `KDA16_D` of `kda_chunk16.cu`.
pub const CHUNK16: usize = 16;
pub const CHUNK16_D: usize = 128;
/// 2026-10-10: `KDA16_VB` and `KDA16_SCAN_THREADS`: value columns and threads per scan block.
const CHUNK16_VB: usize = 32;
const CHUNK16_SCAN_THREADS: u32 = 256;

impl Glm5NextKdaLayer {
    /// 2026-10-10: Whether this target and config can run [`Self::stateful_chunk16`]: the
    /// per-sequence conv kernels, both chunk kernels, and head_dim 128.
    pub(super) fn chunk16_ready(&self) -> bool {
        let s = &self.kernels.seq;
        s.conv_tokens.0 != 0
            && s.conv_window.0 != 0
            && s.chunk16_prepare.0 != 0
            && s.chunk16_scan.0 != 0
            && self.cfg.head_dim == CHUNK16_D
            && self.cfg.conv_kernel <= 4
    }

    /// 2026-10-10: Workspace rows `0..k` of one sequence through the conv, then the chunked
    /// recurrence into `core`, advancing `state` past all of them.
    pub(super) fn stateful_chunk16(
        &self,
        gpu: &dyn GpuBackend,
        k: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        pack: Pack,
        stream: u64,
    ) -> Result<()> {
        if !self.chunk16_ready() || k == 0 || k > ws.max_tokens {
            bail!("KDA: the chunked scan cannot run {k} rows here");
        }
        let c = &self.cfg;
        let (qkv, cd, d, h) = (c.qkv_dim(), c.conv_dim(), c.head_dim, c.heads);
        let nchunks = k.div_ceil(CHUNK16);
        // 2026-10-10: Each scratch buffer holds `t_pad * qkv` floats; the scan's factors need
        // `nchunks * heads * 16 * head_dim` (and P `nchunks * heads * 256`, decay less).
        if nchunks * CHUNK16 > ws.t_pad || CHUNK16 > d {
            bail!(
                "KDA chunked scan: {nchunks} chunks of {CHUNK16} exceed the workspace's {} rows",
                ws.t_pad
            );
        }
        self.conv_tokens(gpu, k, state, ws, pack, stream)?;
        let io = RowIo::workspace(c, ws, 0);
        let scale = 1.0 / (d as f32).sqrt();
        KernelLaunch::new(gpu, self.kernels.seq.chunk16_prepare)
            .grid([nchunks as u32, h as u32, 1])
            .block([d as u32, 1, 1])
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
            .arg_u32(cd as u32)
            .arg_u32(qkv as u32)
            .arg_f32(scale)
            .launch(stream)?;
        KernelLaunch::new(gpu, self.kernels.seq.chunk16_scan)
            .grid([h as u32, (d / CHUNK16_VB) as u32, 1])
            .block([CHUNK16_SCAN_THREADS, 1, 1])
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
            .arg_u32(nchunks as u32)
            .arg_u32(qkv as u32)
            .launch(stream)
    }
}

#[cfg(test)]
#[path = "chunk16_tests.rs"]
mod tests;
