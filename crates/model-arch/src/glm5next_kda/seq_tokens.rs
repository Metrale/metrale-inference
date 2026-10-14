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
        pack: Pack,
        stream: u64,
    ) -> Result<()> {
        if !self.seq_tokens_ready() || k == 0 {
            bail!("KDA: the per-sequence token kernels cannot run {k} rows here");
        }
        self.conv_tokens(gpu, k, state, ws, pack, stream)?;
        self.recurrence_tokens_rows(gpu, 0, k, state, ws, stream)
    }

    /// 2026-10-14: The recurrence of [`Self::stateful_tokens`] over workspace rows
    /// `row0 .. row0 + k`, whose conv rows are already in `conv_out`, advancing `state` past
    /// them: the launch indexes rows from the pointers it is given, so a forward can step its rows
    /// in pieces, a state copy between them (`cut.rs`).
    pub(super) fn recurrence_tokens_rows(
        &self,
        gpu: &dyn GpuBackend,
        row0: usize,
        k: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        if !self.seq_tokens_ready() || k == 0 || row0 + k > ws.max_tokens {
            bail!(
                "KDA: the per-sequence token recurrence cannot run rows {row0}..{} here",
                row0 + k
            );
        }
        let c = &self.cfg;
        let (qkv, cd, d) = (c.qkv_dim(), c.conv_dim(), c.head_dim);
        let io = RowIo::workspace(c, ws, row0);
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

    /// 2026-10-11: [`Self::stateful_tokens`] for a K-row verify that keeps the state after
    /// rows `0..k - 1` in `snapshots[t]` = `(h_dst, conv_dst)`, as the per-row walk with a
    /// snapshot per row leaves them (`decode_k_modes`): each conv intermediate is the conv
    /// window copied and advanced past rows `0..=t` (the advance only moves inputs), before
    /// `conv_tokens` advances the live window; each recurrent intermediate is stored by
    /// `kda_recurrent_decode_bf16_seq_reg_snap` as the column passes row `t`. Refuses a
    /// shape [`snap_tokens_apply`] refuses or a missing kernel.
    pub(super) fn stateful_tokens_snap(
        &self,
        gpu: &dyn GpuBackend,
        k: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        snapshots: &[(DevicePtr, DevicePtr)],
        stream: u64,
    ) -> Result<()> {
        let s = &self.kernels.seq;
        if !self.seq_tokens_ready()
            || s.recurrent_snap.0 == 0
            || !snap_tokens_apply(k, snapshots.len())
        {
            bail!(
                "KDA: the snapshotting token kernels cannot run {k} rows with {} snapshots here",
                snapshots.len()
            );
        }
        let c = &self.cfg;
        let (qkv, cd, d) = (c.qkv_dim(), c.conv_dim(), c.head_dim);
        let io = RowIo::workspace(c, ws, 0);
        for (t, &(_, conv_dst)) in snapshots.iter().enumerate() {
            gpu.copy_d2d_async(state.conv, conv_dst, c.conv_state_elems() * 4, stream)?;
            KernelLaunch::new(gpu, s.conv_window)
                .grid([div_ceil(cd as u32, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(conv_dst)
                .arg_ptr(io.qkv_in)
                .arg_u32(cd as u32)
                .arg_u32(c.conv_kernel as u32)
                .arg_u32(t as u32 + 1)
                .launch(stream)?;
        }
        // 2026-10-13: The snapshots above read the packed rows, so the pack ran in `front_end`.
        self.conv_tokens(gpu, k, state, ws, Pack::InFrontEnd, stream)?;
        let snap = |t: usize| snapshots.get(t).map_or(DevicePtr::NULL, |p| p.0);
        KernelLaunch::new(gpu, s.recurrent_snap)
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
            .arg_ptr(snap(0))
            .arg_ptr(snap(1))
            .arg_ptr(snap(2))
            .arg_u32(snapshots.len() as u32)
            .launch(stream)
    }

    /// 2026-10-09: The conv, SiLU and L2 of workspace rows `0..k` of one sequence, each row as
    /// the per-row `causal_conv1d_update_l2norm` computes it, then the conv window advanced past
    /// the `k` rows: two launches. Also the chunked prefill's conv (`prefill.rs`). 2026-10-13:
    /// under [`Pack::InConv`] the first launch also writes the pack `front_end` skipped
    /// (`conv_pack.rs`).
    pub(super) fn conv_tokens(
        &self,
        gpu: &dyn GpuBackend,
        k: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        pack: Pack,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let cd = c.conv_dim();
        let io = RowIo::workspace(c, ws, 0);
        let s = &self.kernels.seq;
        if pack == Pack::InConv {
            self.conv_tokens_parts(gpu, k, state, ws, stream)?;
        } else {
            self.conv_tokens_packed(gpu, k, state, io, stream)?;
        }
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

    /// 2026-10-13: `causal_conv1d_update_l2norm_tokens` over the packed rows `0..k`.
    fn conv_tokens_packed(
        &self,
        gpu: &dyn GpuBackend,
        k: usize,
        state: &KdaSeqState,
        io: RowIo,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let (cd, d) = (c.conv_dim(), c.head_dim);
        KernelLaunch::new(gpu, self.kernels.seq.conv_tokens)
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
            .launch(stream)
    }
}

/// 2026-10-11: The verify shapes `stateful_tokens_snap` takes: 2..=4 rows with one snapshot
/// per row but the last (the kernel stores at most three).
pub(super) fn snap_tokens_apply(k: usize, snapshots: usize) -> bool {
    (2..=4).contains(&k) && snapshots + 1 == k
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

#[cfg(test)]
mod snap_tokens_tests {
    use super::snap_tokens_apply;

    /// 2026-10-11: The snapshotting kernels take a K=2..=4 verify with its K - 1 intermediates
    /// and nothing else: no prefill (no snapshots), no partial snapshot list, no width the
    /// kernel's three snapshot slots cannot hold.
    #[test]
    fn only_verify_shapes_with_every_intermediate_apply() {
        assert!(snap_tokens_apply(2, 1));
        assert!(snap_tokens_apply(3, 2));
        assert!(snap_tokens_apply(4, 3));
        assert!(!snap_tokens_apply(1, 0));
        assert!(!snap_tokens_apply(2, 0));
        assert!(!snap_tokens_apply(3, 1));
        assert!(!snap_tokens_apply(5, 4));
        assert!(!snap_tokens_apply(16, 0));
    }
}
