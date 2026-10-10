// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: `stateful_tokens`, the stateful half of `k` consecutive rows of ONE sequence in
//! three launches (conv over all rows, conv window advance, recurrence over all rows) instead of
//! two launches per row (`stateful_row`).
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants:
//! - Only for rows that need no per-row state snapshot: the state is advanced past all `k`
//!   rows at once.
//!
//! # Why every row's bits are the per-row walk's
//!
//! `causal_conv1d_update_l2norm_tokens` gives block (x, t) the window the per-row conv state
//! holds when row `t` arrives (the inputs are known up front; the conv state is a window of
//! them), and runs `causal_conv1d_update_l2norm`'s body on it; `causal_conv1d_window_advance`
//! then leaves the window the walk leaves after row `k - 1`.
//! `kda_recurrent_decode_bf16_seq_reg` keeps each v column of the state in registers across
//! the rows and runs, per row, the per-row step's expressions in their order on the column the
//! previous row left; the per-row kernels read that column back from memory, the same floats.

use super::decode::RowIo;
use super::*;

impl Glm5NextKdaLayer {
    /// 2026-10-09: Whether this target and config can run [`Self::stateful_tokens`].
    pub(super) fn seq_tokens_ready(&self) -> bool {
        self.kernels.seq.ready() && self.cfg.head_dim == KDA_REG_D && self.cfg.conv_kernel <= 4
    }

    /// 2026-10-09: Workspace rows `0..k` of one sequence through the conv and the recurrence,
    /// advancing `state` past all of them. Refuses where [`Self::seq_tokens_ready`] is false.
    pub(super) fn stateful_tokens(
        &self,
        gpu: &dyn GpuBackend,
        k: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        if !self.seq_tokens_ready() || k == 0 {
            bail!("KDA: the per-sequence token kernels cannot run {k} rows here");
        }
        self.conv_tokens(gpu, k, state, ws, stream)?;
        let c = &self.cfg;
        let (qkv, cd, d) = (c.qkv_dim(), c.conv_dim(), c.head_dim);
        let io = RowIo::workspace(c, ws, 0);
        KernelLaunch::new(gpu, self.kernels.seq.recurrent)
            .grid([c.heads as u32, (d / SEQ_VB) as u32, 1])
            .block([SEQ_VB as u32, 1, 1])
            .shared_mem((3 * d * 4) as u32)
            .arg_ptr(io.conv_out)
            .arg_ptr(io.conv_out.offset(qkv * 2))
            .arg_ptr(io.conv_out.offset(qkv * 4))
            .arg_ptr(io.gate)
            .arg_ptr(io.beta)
            .arg_ptr(state.recurrent)
            .arg_ptr(io.core)
            .arg_u32(c.heads as u32)
            .arg_f32(1.0 / (d as f32).sqrt())
            .arg_u32(k as u32)
            .arg_u32(cd as u32)
            .arg_u32(qkv as u32)
            .arg_u32(c.heads as u32)
            .arg_u32(qkv as u32)
            .launch(stream)
    }

    /// 2026-10-09: The conv, SiLU and L2 of workspace rows `0..k` of one sequence, each row as
    /// the per-row `causal_conv1d_update_l2norm` computes it, then the conv window advanced past
    /// the `k` rows: two launches. Also the chunked prefill's conv (`prefill.rs`).
    pub(super) fn conv_tokens(
        &self,
        gpu: &dyn GpuBackend,
        k: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let (cd, d) = (c.conv_dim(), c.head_dim);
        let io = RowIo::workspace(c, ws, 0);
        let s = &self.kernels.seq;
        KernelLaunch::new(gpu, s.conv_tokens)
            .grid([div_ceil(cd as u32, 256), k as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(state.conv)
            .arg_ptr(io.qkv_in)
            .arg_ptr(self.weights.conv.weight)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(io.conv_out)
            .arg_u32(cd as u32)
            .arg_u32(c.conv_kernel as u32)
            .arg_u32(c.qk_channels() as u32)
            .arg_u32(d as u32)
            .arg_f32(c.l2_eps)
            .launch(stream)?;
        KernelLaunch::new(gpu, s.conv_window)
            .grid([div_ceil(cd as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(state.conv)
            .arg_ptr(io.qkv_in)
            .arg_u32(cd as u32)
            .arg_u32(c.conv_kernel as u32)
            .arg_u32(k as u32)
            .launch(stream)
    }
}

/// 2026-10-09: V columns per block of `kda_recurrent_decode_bf16_seq_reg`: one warp, so a
/// head's 128 columns run as four blocks.
const SEQ_VB: usize = 32;

/// 2026-10-10: `METRALE_GLM_KDA_ROW_REG=1` runs each single-row recurrent step
/// (`stateful_row`: the C1 decode and every per-row walk) on `kda_recurrent_decode_bf16_seq_reg`
/// at one token instead of `kda_recurrent_decode_bf16_smem`: the same expressions in the same
/// order, with all of a column's state loads in flight at once. Off until measured end to end.
/// Read once.
pub(super) fn kda_row_reg() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("METRALE_GLM_KDA_ROW_REG").as_deref() == Ok("1"))
}

impl Glm5NextKdaLayer {
    /// 2026-10-10: One recurrent step of the row buffers `io` on `state.recurrent` with
    /// `kda_recurrent_decode_bf16_seq_reg` at T = 1, as [`Self::stateful_tokens`] launches it for
    /// one row. Refuses a target without the kernel or a head_dim other than `KDA_REG_D`: the
    /// lever asked for it.
    pub(super) fn recurrent_row_reg(
        &self,
        gpu: &dyn GpuBackend,
        io: RowIo,
        state: &KdaSeqState,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let (qkv, d) = (c.qkv_dim(), c.head_dim);
        if self.kernels.seq.recurrent.0 == 0 || d != KDA_REG_D {
            bail!(
                "METRALE_GLM_KDA_ROW_REG=1: needs kda_recurrent_decode_bf16_seq_reg and \
                 head_dim {KDA_REG_D} (head_dim {d})"
            );
        }
        KernelLaunch::new(gpu, self.kernels.seq.recurrent)
            .grid([c.heads as u32, (d / SEQ_VB) as u32, 1])
            .block([SEQ_VB as u32, 1, 1])
            .shared_mem((3 * d * 4) as u32)
            .arg_ptr(io.conv_out)
            .arg_ptr(io.conv_out.offset(qkv * 2))
            .arg_ptr(io.conv_out.offset(qkv * 4))
            .arg_ptr(io.gate)
            .arg_ptr(io.beta)
            .arg_ptr(state.recurrent)
            .arg_ptr(io.core)
            .arg_u32(c.heads as u32)
            .arg_f32(1.0 / (d as f32).sqrt())
            // 2026-10-10: One token, so the per-token strides are never applied.
            .arg_u32(1)
            .arg_u32(0)
            .arg_u32(0)
            .arg_u32(0)
            .arg_u32(0)
            .launch(stream)
    }
}
