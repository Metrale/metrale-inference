// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The KDA recurrent path: single-token `decode`, K-row `decode_k`, and
//! (2026-10-08) `decode_rows`, one row per sequence.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants:
//! - The recurrent state advances one row at a time, in row order (`stateful_row`).

use super::*;

impl Glm5NextKdaLayer {
    /// 2026-09-25: The stateful half of one KDA token: the conv window update (SiLU and L2 fused),
    /// then one recurrent step, both on row `row` of the workspace, updating `state` in place.
    ///
    /// The state after row `t + 1` depends on the state after row `t`, so [`Self::decode_k`] calls
    /// this once per row, in order, and batches only the projections around it. The chunked
    /// [`Self::prefill`] computes the same recurrence in a different order, so its results are not
    /// guaranteed to match this path bit for bit.
    ///
    /// `q`/`k` reach `kda_recurrent` already L2-normalised by the conv, which is the input that
    /// kernel expects; normalising them again would change the bf16-rounded values.
    fn stateful_row(
        &self,
        gpu: &dyn GpuBackend,
        row: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        self.conv_row(gpu, row, state, ws, stream)?;
        self.recurrent_row(gpu, row, state, ws, stream)
    }

    /// 2026-10-09: The conv window update (SiLU and L2 fused) of row `row` on `state.conv`, the
    /// first half of [`Self::stateful_row`].
    fn conv_row(
        &self,
        gpu: &dyn GpuBackend,
        row: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let cd = c.conv_dim();
        ops::conv1d_update_l2norm(
            gpu,
            self.kernels.conv_decode,
            state.conv,
            ws.qkv_proj.offset(row * cd * 2),
            &self.weights.conv,
            ws.conv_out.offset(row * cd * 2),
            c.conv_dim() as u32,
            c.conv_kernel as u32,
            1,
            c.qk_channels() as u32,
            c.head_dim as u32,
            c.l2_eps,
            stream,
        )
    }

    /// 2026-10-09: The 1R+1W kernel's launch geometry for this config, `(vpb, shared bytes)`,
    /// or `None` when the 2R+2W kernel runs instead (no 1R+1W kernel, a `head_dim` the slice
    /// does not divide, a request over `KDA_SMEM_BUDGET`, or `METRALE_GLM_KDA_NO_SMEM=1`).
    fn smem_geometry(&self) -> Option<(usize, usize)> {
        let d = self.cfg.head_dim;
        let vpb = KDA_V_PER_BLOCK.min(d);
        let smem = (3 * d + vpb * (d + 1)) * 4;
        (self.kernels.recurrent_smem.0 != 0
            && d.is_multiple_of(vpb)
            && smem <= KDA_SMEM_BUDGET
            && !kda_no_smem())
        .then_some((vpb, smem))
    }

    /// 2026-10-09: One recurrent step of row `row` on `state.recurrent`, the second half of
    /// [`Self::stateful_row`].
    fn recurrent_row(
        &self,
        gpu: &dyn GpuBackend,
        row: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let qkv = c.qkv_dim();
        let cd = c.conv_dim();
        let d = c.head_dim;
        // 2026-09-25: 1R+1W when the target has the shared-memory kernel. Each block owns `vpb`
        // V columns and grid.y covers the rest; the request is `3 * d` floats plus
        // `vpb * (d + 1)` floats of column scratch.
        if let Some((vpb, smem_smem)) = self.smem_geometry() {
            KernelLaunch::new(gpu, self.kernels.recurrent_smem)
                .grid([c.heads as u32, (d / vpb) as u32, 1])
                .block([vpb as u32, 1, 1])
                .shared_mem(smem_smem as u32)
                .arg_ptr(ws.conv_out.offset(row * cd * 2))
                .arg_ptr(ws.conv_out.offset(row * cd * 2 + qkv * 2))
                .arg_ptr(ws.conv_out.offset(row * cd * 2 + qkv * 4))
                .arg_ptr(ws.gate.offset(row * qkv * 4))
                .arg_ptr(ws.beta.offset(row * c.heads * 4))
                .arg_ptr(state.recurrent)
                .arg_ptr(ws.core.offset(row * qkv * 4))
                .arg_u32(c.heads as u32)
                .arg_u32(d as u32)
                .arg_f32(1.0 / (d as f32).sqrt())
                .arg_u32(vpb as u32)
                .launch(stream)?;
        } else {
            KernelLaunch::new(gpu, self.kernels.recurrent)
                .grid([c.heads as u32, 1, 1])
                .block([BLOCK.min(d as u32), 1, 1])
                .shared_mem((3 * d * 4) as u32)
                .arg_ptr(ws.conv_out.offset(row * cd * 2))
                .arg_ptr(ws.conv_out.offset(row * cd * 2 + qkv * 2))
                .arg_ptr(ws.conv_out.offset(row * cd * 2 + qkv * 4))
                .arg_ptr(ws.gate.offset(row * qkv * 4))
                .arg_ptr(ws.beta.offset(row * c.heads * 4))
                .arg_ptr(state.recurrent)
                .arg_ptr(ws.core.offset(row * qkv * 4))
                .arg_u32(c.heads as u32)
                .arg_u32(d as u32)
                .arg_f32(1.0 / (d as f32).sqrt())
                .launch(stream)?;
        }

        Ok(())
    }

    /// 2026-09-25: Single-token decode, carrying both states. The result lands in `ws.final_out`;
    /// `state` is updated in place.
    pub fn decode(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        self.front_end(gpu, hidden, 1, ws, stream)?;
        self.stateful_row(gpu, 0, state, ws, stream)?;
        self.back_end(gpu, 1, ws, stream)
    }

    /// 2026-09-25: `k` tokens of one sequence: the projections batched, the recurrence one row at a
    /// time. Used for the speculative verify, and for prefill sub-chunks that do not take the
    /// chunked arm (`glm5next_layer/steps/forward.rs`).
    ///
    /// `front_end` and `back_end` run once over all `k` rows. For `2 <= k <=
    /// DENSE_GEMV_BATCHM_MAX_M` on a target with `dense_gemv_bf16_batchm`, the output matches `k`
    /// serial [`Self::decode`] calls bit for bit: the batched GEMV gives each row the bits of the
    /// M = 1 GEMV, the pack, gate, sigmoid and `o_norm` kernels treat rows independently, and
    /// `stateful_row` walks the state one row at a time as `decode` does.
    ///
    /// `snapshots[t]`, `(h_dst, conv_dst)`, receives the state after row `t`; rows past
    /// `snapshots.len()` take no snapshot. Refuses `k == 0` or `k` above the workspace.
    pub fn decode_k(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        k: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        snapshots: &[(DevicePtr, DevicePtr)],
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let (h_bytes, conv_bytes) = (c.recurrent_state_elems() * 4, c.conv_state_elems() * 4);
        self.rows_with(
            gpu,
            hidden,
            k,
            ws,
            stream,
            |row| {
                self.stateful_row(gpu, row, state, ws, stream)?;
                if let Some((h_dst, conv_dst)) = snapshots.get(row) {
                    gpu.copy_d2d_async(state.recurrent, *h_dst, h_bytes, stream)?;
                    gpu.copy_d2d_async(state.conv, *conv_dst, conv_bytes, stream)?;
                }
                Ok(())
            },
            || Ok(()),
        )
    }

    /// 2026-10-08: One token for each of `states.len()` sequences: row `r` of `hidden` advances
    /// `states[r]`. The projections run once over all rows, as in [`Self::decode_k`], and each
    /// row's conv and recurrent step is [`Self::decode`]'s own arithmetic on that row's state
    /// (2026-10-09: from two rows on, with the 1R+1W kernel present, the recurrent steps of
    /// all rows run as one `kda_recurrent_decode_bf16_smem_rows` launch, block for block the
    /// single-row kernel's), so
    /// for up to `DENSE_GEMV_BATCHM_MAX_M` rows a row's output and state equal a lone
    /// [`Self::decode`] of its sequence bit for bit (the argument in [`Self::decode_k`]).
    /// Refuses no rows or more rows than the workspace holds.
    pub fn decode_rows(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        states: &[KdaSeqState],
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        let n = states.len();
        let batched = (2..=KDA_ROWS_MAX).contains(&n)
            && self.kernels.recurrent_smem_rows.0 != 0
            && self.smem_geometry().is_some();
        if !batched {
            return self.rows_with(
                gpu,
                hidden,
                n,
                ws,
                stream,
                |row| self.stateful_row(gpu, row, &states[row], ws, stream),
                || Ok(()),
            );
        }
        // 2026-10-09: Every row's conv first, then one recurrent launch for all rows. Row
        // `r`'s recurrence reads only row `r`'s conv output and state, so the order across
        // rows does not change a row's bits, and block (h, v-slice, r) of the rows kernel is
        // the single-row kernel's block (h, v-slice) on row `r`.
        self.rows_with(
            gpu,
            hidden,
            n,
            ws,
            stream,
            |row| self.conv_row(gpu, row, &states[row], ws, stream),
            || self.recurrent_rows(gpu, states, ws, stream),
        )
    }

    /// 2026-10-09: `kda_recurrent_decode_bf16_smem_rows` over `states.len()` rows (at most
    /// `KDA_ROWS_MAX`), with the 1R+1W geometry.
    fn recurrent_rows(
        &self,
        gpu: &dyn GpuBackend,
        states: &[KdaSeqState],
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let (qkv, cd, d) = (c.qkv_dim(), c.conv_dim(), c.head_dim);
        let Some((vpb, smem)) = self.smem_geometry() else {
            bail!("KDA batched recurrence without the 1R+1W geometry");
        };
        if states.is_empty() || states.len() > KDA_ROWS_MAX {
            bail!(
                "KDA batched recurrence of {} rows (1..={KDA_ROWS_MAX})",
                states.len()
            );
        }
        let mut launch = KernelLaunch::new(gpu, self.kernels.recurrent_smem_rows)
            .grid([c.heads as u32, (d / vpb) as u32, states.len() as u32])
            .block([vpb as u32, 1, 1])
            .shared_mem(smem as u32)
            .arg_ptr(ws.conv_out)
            .arg_ptr(ws.conv_out.offset(qkv * 2))
            .arg_ptr(ws.conv_out.offset(qkv * 4))
            .arg_ptr(ws.gate)
            .arg_ptr(ws.beta)
            .arg_ptr(ws.core)
            .arg_u32(c.heads as u32)
            .arg_u32(d as u32)
            .arg_f32(1.0 / (d as f32).sqrt())
            .arg_u32(vpb as u32)
            .arg_u32(cd as u32)
            .arg_u32(qkv as u32)
            .arg_u32(c.heads as u32)
            .arg_u32(qkv as u32);
        for r in 0..KDA_ROWS_MAX {
            launch = launch.arg_u64(states.get(r).map_or(0, |s| s.recurrent.0));
        }
        launch.launch(stream)
    }

    /// 2026-10-08: `front_end` over `k` rows, `per_row(row)` for each row in order,
    /// (2026-10-09) `after_rows`, then `back_end`, each span in its profile bucket. Refuses `k == 0` or `k` above the
    /// workspace before any launch.
    fn rows_with(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        k: usize,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
        mut per_row: impl FnMut(usize) -> Result<()>,
        after_rows: impl FnOnce() -> Result<()>,
    ) -> Result<()> {
        if k == 0 || k > ws.max_tokens {
            bail!(
                "KDA decode of {k} rows does not fit a workspace built for {}",
                ws.max_tokens
            );
        }
        use crate::glm5next_layer::profile;
        // 2026-09-25: Three profile buckets (front, recurrence, back), each closed where its span
        // ends.
        let t_front = profile::start();
        self.front_end(gpu, hidden, k, ws, stream)?;
        profile::end(profile::KDA_FRONT, t_front, gpu, stream);
        let t_recur = profile::start();
        for row in 0..k {
            per_row(row)?;
        }
        after_rows()?;
        profile::end(profile::KDA_RECUR, t_recur, gpu, stream);
        let t_back = profile::start();
        let r = self.back_end(gpu, k, ws, stream);
        profile::end(profile::KDA_BACK, t_back, gpu, stream);
        r
    }
}
