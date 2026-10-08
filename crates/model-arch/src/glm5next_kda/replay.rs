// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: Speculative-verify rollback by replay for a KDA layer: one checkpoint and a record
//! of each verify row's recurrent inputs per sequence, instead of a full state per verify row.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants:
//! - A replayed row runs `stateful_row` with the kernels, grid, block, shared memory and scalar
//!   arguments of the verify row it repeats, on the bytes that row read: the recorded pre-conv
//!   input, decay and beta. Starting from the checkpoint, the state after `n` replayed rows is
//!   therefore the state the verify held after its row `n - 1`, bit for bit.
//! - The record holds rows `0..k - 1` of a `k`-row verify: a partial accept keeps at most
//!   `k - 1` rows, and a full accept keeps the verify's own final state.
//!
//! ```text
//! verify (k rows):  state -> checkpoint;  decode_k in place;  record rows 0..k-1
//! commit (n of k):  n == k: nothing;  else checkpoint -> state, replay rows 0..n
//! ```

use super::decode::RowIo;
use super::*;

impl Glm5NextKdaConfig {
    /// 2026-10-08: Bytes of one recorded verify row: the pre-conv q|k|v input (`[conv_dim]`
    /// BF16), the bounded log-decay (`[heads, head_dim]` FP32) and beta (`[heads]` FP32).
    pub fn replay_row_bytes(&self) -> usize {
        self.conv_dim() * 2 + self.qkv_dim() * 4 + self.heads * 4
    }
}

/// 2026-10-08: One sequence's verify record for one KDA layer, carved from a region of
/// `bytes` bytes into three arrays of `rows` rows each: inputs, decays, betas.
#[derive(Clone, Copy, Debug)]
pub struct KdaVerifyRecord {
    base: DevicePtr,
    rows: usize,
    cd: usize,
    qkv: usize,
    heads: usize,
}

impl KdaVerifyRecord {
    /// 2026-10-08: The record over `bytes` bytes at `base`: as many whole rows as fit.
    pub fn new(cfg: &Glm5NextKdaConfig, base: DevicePtr, bytes: usize) -> Self {
        Self {
            base,
            rows: bytes / cfg.replay_row_bytes(),
            cd: cfg.conv_dim(),
            qkv: cfg.qkv_dim(),
            heads: cfg.heads,
        }
    }

    /// 2026-10-08: The rows this record holds.
    pub fn rows(&self) -> usize {
        self.rows
    }

    fn input_row(&self, r: usize) -> DevicePtr {
        self.base.offset(r * self.cd * 2)
    }

    fn gate_row(&self, r: usize) -> DevicePtr {
        self.base.offset(self.rows * self.cd * 2 + r * self.qkv * 4)
    }

    fn beta_row(&self, r: usize) -> DevicePtr {
        self.base
            .offset(self.rows * (self.cd * 2 + self.qkv * 4) + r * self.heads * 4)
    }
}

impl Glm5NextKdaLayer {
    fn state_bytes(&self) -> (usize, usize) {
        (
            self.cfg.recurrent_state_elems() * 4,
            self.cfg.conv_state_elems() * 4,
        )
    }

    /// 2026-10-08: Copy `state` into `checkpoint` (recurrent and conv), before a verify that
    /// updates `state` in place.
    pub fn checkpoint_state(
        &self,
        gpu: &dyn GpuBackend,
        state: &KdaSeqState,
        checkpoint: &KdaSeqState,
        stream: u64,
    ) -> Result<()> {
        let (h_bytes, conv_bytes) = self.state_bytes();
        gpu.copy_d2d_async(state.recurrent, checkpoint.recurrent, h_bytes, stream)?;
        gpu.copy_d2d_async(state.conv, checkpoint.conv, conv_bytes, stream)
    }

    /// 2026-10-08: After `decode_k`, copy the workspace's first `rows` rows of pre-conv input,
    /// decay and beta into `record`: one copy per array. Errors when the record holds fewer
    /// rows or the workspace was built for fewer.
    pub fn record_verify_rows(
        &self,
        gpu: &dyn GpuBackend,
        ws: &Glm5NextKdaWorkspace,
        rows: usize,
        record: &KdaVerifyRecord,
        stream: u64,
    ) -> Result<()> {
        if rows > record.rows || rows > ws.max_tokens {
            bail!(
                "KDA layer {}: a verify record of {rows} rows does not fit (record {} rows, \
                 workspace {} rows)",
                self.layer_idx,
                record.rows,
                ws.max_tokens
            );
        }
        if rows == 0 {
            return Ok(());
        }
        let c = &self.cfg;
        gpu.copy_d2d_async(
            ws.qkv_proj,
            record.input_row(0),
            rows * c.conv_dim() * 2,
            stream,
        )?;
        gpu.copy_d2d_async(ws.gate, record.gate_row(0), rows * c.qkv_dim() * 4, stream)?;
        gpu.copy_d2d_async(ws.beta, record.beta_row(0), rows * c.heads * 4, stream)
    }

    /// 2026-10-08: Commit `accepted` of the `k` rows a replay-mode verify ran. `accepted == k`
    /// leaves the verify's final state in place. Otherwise `checkpoint` is copied back into
    /// `state` and rows `0..accepted` are replayed from `record`, writing their conv and core
    /// outputs to workspace rows `0..accepted` (scratch between forwards). Errors when
    /// `accepted` is 0 or above `k`, or when the record or workspace is too short.
    #[allow(clippy::too_many_arguments)]
    pub fn commit_replay(
        &self,
        gpu: &dyn GpuBackend,
        state: &KdaSeqState,
        checkpoint: &KdaSeqState,
        record: &KdaVerifyRecord,
        accepted: usize,
        k: usize,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        if accepted == 0 || accepted > k {
            bail!(
                "KDA layer {}: commit of {accepted} accepted rows of a {k}-row verify; row 0 is \
                 always accepted and no more than k rows exist",
                self.layer_idx
            );
        }
        if accepted == k {
            return Ok(());
        }
        if accepted > record.rows || accepted > ws.max_tokens {
            bail!(
                "KDA layer {}: replay of {accepted} rows exceeds the verify record ({} rows) or \
                 the workspace ({} rows)",
                self.layer_idx,
                record.rows,
                ws.max_tokens
            );
        }
        let (h_bytes, conv_bytes) = self.state_bytes();
        gpu.copy_d2d_async(checkpoint.recurrent, state.recurrent, h_bytes, stream)?;
        gpu.copy_d2d_async(checkpoint.conv, state.conv, conv_bytes, stream)?;
        for r in 0..accepted {
            let scratch = RowIo::workspace(&self.cfg, ws, r);
            let io = RowIo {
                qkv_in: record.input_row(r),
                gate: record.gate_row(r),
                beta: record.beta_row(r),
                ..scratch
            };
            self.stateful_row(gpu, io, state, stream)?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "replay_tests.rs"]
mod tests;
