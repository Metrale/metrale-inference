// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The per-call staging of `decode_k`'s host path (no metadata: a prefill
//! sub-chunk or a verify without row metadata): every row's KV slot, query position and
//! `seq_len`, and the shared block table, computed on the host and uploaded once per call.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - The uploads are stream-ordered (`copy_h2d_async` on the call's stream), so a later
//!   kernel on that stream reads them and an earlier one has finished with the buffers.
//! - The values are the ones the per-row path computed and copied one by one before
//!   2026-10-09, at the same device offsets, so every kernel reads the same bytes.

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::Glm5NextDsaLayer;
use super::decode_k::bt_entries_needed;

/// 2026-10-09: Device pointers of the staged rows: `slot` `[k]` i64, `q_pos` `[k]` i32,
/// `sl` `[k]` i32, and `bt`, the sequence's block-table prefix. `owned` is true when `bt`
/// and `sl` were allocated for this call (`METRALE_GLM_DSA_ALLOC_PER_STEP=1`) and must be
/// freed after the attend.
pub(super) struct HostRows {
    pub(super) slot: DevicePtr,
    pub(super) q_pos: DevicePtr,
    pub(super) bt: DevicePtr,
    pub(super) sl: DevicePtr,
    pub(super) owned: bool,
}

/// 2026-10-09: Row `r`'s KV slot for position `seq_len + r`: `block_table[pos / block_size]`
/// as the physical block, at offset `pos % block_size`. Errors when the table is too short.
pub(super) fn row_slots(
    layer_idx: usize,
    seq_len: usize,
    k: usize,
    block_table: &[u32],
    block_size: usize,
) -> Result<Vec<i64>> {
    (0..k)
        .map(|row| {
            let pos = seq_len + row;
            let logical = pos / block_size;
            let Some(&physical) = block_table.get(logical) else {
                bail!(
                    "DSA layer {layer_idx}: block table has {} entries, needs logical block \
                     {logical} for position {pos}",
                    block_table.len()
                );
            };
            Ok((physical as usize * block_size + pos % block_size) as i64)
        })
        .collect()
}

fn le<T: Copy, const N: usize>(v: &[T], f: impl Fn(T) -> [u8; N]) -> Vec<u8> {
    v.iter().flat_map(|x| f(*x)).collect()
}

impl Glm5NextDsaLayer {
    /// 2026-10-09: Stage `k` host-path rows from position `seq_len`: four stream-ordered
    /// uploads per call instead of four synchronous copies per row. Every check runs before
    /// the first upload.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn stage_host_rows(
        &self,
        gpu: &dyn GpuBackend,
        k: usize,
        seq_len: usize,
        block_table: &[u32],
        block_size: usize,
        bt_block_size: usize,
        stream: u64,
    ) -> Result<HostRows> {
        let w = &self.workspace;
        let slots = row_slots(self.layer_idx, seq_len, k, block_table, block_size)?;
        let positions: Vec<i32> = (0..k).map(|r| (seq_len + r) as i32).collect();
        let lens: Vec<i32> = (0..k).map(|r| (seq_len + r + 1) as i32).collect();
        // 2026-09-25: Upload only the prefix the gather can index ([`bt_entries_needed`]),
        // not the caller's whole table.
        let bt_used = {
            let needed = bt_entries_needed(seq_len, k, bt_block_size);
            &block_table[..needed.min(block_table.len())]
        };
        if bt_used.len() > w.bt_cap {
            bail!(
                "DSA layer {}: block table needs {} entries for seq_len {} + {} rows but the \
                 persistent buffer holds {}. This is a BLOCK count against a buffer sized by \
                 max_dsa_context (a TOKEN count); do not write past the allocation.",
                self.layer_idx,
                bt_used.len(),
                seq_len,
                k,
                w.bt_cap
            );
        }
        let bt = le(bt_used, u32::to_le_bytes);
        let (d_bt, d_sl, owned) = if self.persist_bt {
            (w.bt, w.sl, false)
        } else {
            (gpu.alloc(bt.len().max(4))?, gpu.alloc(k * 4)?, true)
        };
        gpu.copy_h2d_async(&le(&slots, i64::to_le_bytes), w.slot, stream)?;
        gpu.copy_h2d_async(&le(&positions, i32::to_le_bytes), w.q_pos, stream)?;
        gpu.copy_h2d_async(&bt, d_bt, stream)?;
        gpu.copy_h2d_async(&le(&lens, i32::to_le_bytes), d_sl, stream)?;
        Ok(HostRows {
            slot: w.slot,
            q_pos: w.q_pos,
            bt: d_bt,
            sl: d_sl,
            owned,
        })
    }
}

#[cfg(test)]
#[path = "host_rows_tests.rs"]
mod tests;
