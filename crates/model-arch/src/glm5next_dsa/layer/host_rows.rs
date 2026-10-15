// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The per-call staging of `decode_k`'s host path (no metadata: a prefill
//! sub-chunk or a verify without row metadata): every row's KV slot, query position and
//! `seq_len`, and the shared block table, computed on the host and uploaded once per call.
//!
//! 2026-10-15: `METRALE_GLM_DSA_HOST_ROWS_PINNED=1` stages those four uploads in page-locked
//! memory ([`HostRowsStaging`]) instead of handing the driver pageable vectors. A pageable
//! `cuMemcpyHtoDAsync` above 64 KiB (the slot array of an 8,208-row sub-chunk is 65,664 B)
//! does not return until the stream has drained, so the host lost its whole lead (about
//! 220 ms) once per DSA layer of such a sub-chunk; a page-locked source never waits.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - The uploads are stream-ordered (`copy_h2d_async` on the call's stream), so a later
//!   kernel on that stream reads them and an earlier one has finished with the buffers.
//! - The values are the ones the per-row path computed and copied one by one before
//!   2026-10-09, at the same device offsets, so every kernel reads the same bytes.
//! - 2026-10-15: Pinned or not, the same four copies of the same bytes to the same
//!   destinations in the same order ([`HostRowBytes::parts`]); only the host source differs.
//! - 2026-10-15: A staging region is rewritten only after the device has finished every copy
//!   enqueued from it (its event), and a call whose bytes equal the region it last wrote reads
//!   that region again without rewriting it, so the DSA layers of one sub-chunk never wait.

use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result, bail, ensure};
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

/// 2026-10-15: The little-endian bytes of one host-path call: `[k]` i64 slots, `[k]` i32
/// positions, the used block-table prefix as u32, and `[k]` i32 `seq_len`s.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostRowBytes {
    pub slot: Vec<u8>,
    pub q_pos: Vec<u8>,
    pub bt: Vec<u8>,
    pub sl: Vec<u8>,
}

impl HostRowBytes {
    /// 2026-10-15: The bytes of `k` rows from position `seq_len` (moved out of
    /// `stage_host_rows` unchanged). Errors, before anything is uploaded, when the table is
    /// too short or its used prefix exceeds `bt_cap` entries.
    pub fn new(
        layer_idx: usize,
        k: usize,
        seq_len: usize,
        block_table: &[u32],
        block_size: usize,
        bt_block_size: usize,
        bt_cap: usize,
    ) -> Result<Self> {
        let slots = row_slots(layer_idx, seq_len, k, block_table, block_size)?;
        let positions: Vec<i32> = (0..k).map(|r| (seq_len + r) as i32).collect();
        let lens: Vec<i32> = (0..k).map(|r| (seq_len + r + 1) as i32).collect();
        // 2026-09-25: Upload only the prefix the gather can index ([`bt_entries_needed`]),
        // not the caller's whole table.
        let bt_used = {
            let needed = bt_entries_needed(seq_len, k, bt_block_size);
            &block_table[..needed.min(block_table.len())]
        };
        if bt_used.len() > bt_cap {
            bail!(
                "DSA layer {layer_idx}: block table needs {} entries for seq_len {seq_len} + {k} \
                 rows but the persistent buffer holds {bt_cap}. This is a BLOCK count against a \
                 buffer sized by max_dsa_context (a TOKEN count); do not write past the \
                 allocation.",
                bt_used.len(),
            );
        }
        Ok(Self {
            slot: le(&slots, i64::to_le_bytes),
            q_pos: le(&positions, i32::to_le_bytes),
            bt: le(bt_used, u32::to_le_bytes),
            sl: le(&lens, i32::to_le_bytes),
        })
    }

    /// 2026-10-15: The four uploads in their order, each with its destination.
    pub fn parts(
        &self,
        slot: DevicePtr,
        q_pos: DevicePtr,
        bt: DevicePtr,
        sl: DevicePtr,
    ) -> [(&[u8], DevicePtr); 4] {
        [
            (&self.slot, slot),
            (&self.q_pos, q_pos),
            (&self.bt, bt),
            (&self.sl, sl),
        ]
    }
}

/// 2026-10-15: `METRALE_GLM_DSA_HOST_ROWS_PINNED`, read once: `1` stages the host path's
/// uploads in page-locked memory; unset or `0` keeps the pageable copies; another value is an
/// error.
pub fn host_rows_pinned() -> Result<bool> {
    static V: OnceLock<Result<bool, String>> = OnceLock::new();
    V.get_or_init(|| {
        parse_pinned(
            std::env::var("METRALE_GLM_DSA_HOST_ROWS_PINNED")
                .ok()
                .as_deref(),
        )
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

/// 2026-10-15: [`host_rows_pinned`]'s parse, pure.
pub(super) fn parse_pinned(v: Option<&str>) -> Result<bool, String> {
    match v {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!(
            "METRALE_GLM_DSA_HOST_ROWS_PINNED={other:?}: expected 0 or 1"
        )),
    }
}

/// 2026-10-15: What a staged call does: read `region` again (its bytes are this call's) or
/// write it first; either way after waiting for its last copies when `wait`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Step {
    Reuse { region: usize, wait: bool },
    Write { region: usize, wait: bool },
}

/// 2026-10-15: The step of a call, pure: `last` is the region the previous call read (None
/// before the first), `same` whether this call's bytes equal it, `recorded[r]` whether a copy
/// from region `r` was ever enqueued, `on_stream[r]` whether its last copies were enqueued on
/// this call's stream. New bytes go to the other region, so the region a pending copy may
/// still read is never the one rewritten. A region's event marks only its last stream's
/// copies, so a region read again from another stream is waited on first, and its event
/// always covers every copy from it.
pub(super) fn step(
    last: Option<usize>,
    same: bool,
    recorded: [bool; 2],
    on_stream: [bool; 2],
) -> Step {
    match last {
        Some(r) if same => Step::Reuse {
            region: r,
            wait: recorded[r] && !on_stream[r],
        },
        _ => {
            let region = last.map_or(0, |r| 1 - r);
            Step::Write {
                region,
                wait: recorded[region],
            }
        }
    }
}

struct Ring {
    /// 2026-10-15: Host addresses of the two page-locked regions, `bytes` each.
    base: [usize; 2],
    bytes: usize,
    events: [u64; 2],
    recorded: [bool; 2],
    /// 2026-10-15: The stream each region's event was last recorded on.
    streams: [u64; 2],
    /// 2026-10-15: Bytes of each region holding the last call's parts, back to back.
    filled: [usize; 2],
    last: Option<usize>,
}

impl Ring {
    /// 2026-10-15: Region `r`'s first `filled[r]` bytes.
    fn held(&self, r: usize) -> &[u8] {
        // SAFETY: `base[r]` is a live page-locked allocation of `bytes` from
        // `alloc_host_pinned` (zeroed, never freed), `filled[r] <= bytes`, and it is written
        // only through `&mut self` (`fill`), which this borrow excludes.
        unsafe { std::slice::from_raw_parts(self.base[r] as *const u8, self.filled[r]) }
    }

    /// 2026-10-15: Write `parts` back to back into region `r`; the caller checked that they
    /// fit in `bytes`.
    fn fill(&mut self, r: usize, parts: &[(&[u8], DevicePtr)]) {
        let mut at = 0;
        for (b, _) in parts {
            // SAFETY: as in `held`; `at + b.len()` stays within the `bytes` the caller
            // checked, and `b` is caller memory, never the region.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    b.as_ptr(),
                    (self.base[r] as *mut u8).add(at),
                    b.len(),
                )
            };
            at += b.len();
        }
        self.filled[r] = at;
    }
}

/// 2026-10-15: Two page-locked regions of `bytes` (enough for `max_rows` rows and a `bt_cap`
/// table), each with an event recorded after the copies read from it. Allocated at load;
/// held for the process's life, like the workspace's device buffers.
pub struct HostRowsStaging {
    ring: Mutex<Ring>,
}

impl HostRowsStaging {
    /// 2026-10-15: Fails (a load error, never a silent pageable fallback) when the page-locked
    /// memory or the events cannot be had.
    pub fn alloc(gpu: &dyn GpuBackend, max_rows: usize, bt_cap: usize) -> Result<Self> {
        let bytes = (max_rows * 16 + bt_cap * 4).max(1);
        let mut base = [0usize; 2];
        let mut events = [0u64; 2];
        for r in 0..2 {
            base[r] = gpu.alloc_host_pinned(bytes).with_context(|| {
                format!("METRALE_GLM_DSA_HOST_ROWS_PINNED=1: page-locked staging of {bytes} B")
            })? as usize;
            events[r] = gpu
                .create_event()
                .context("METRALE_GLM_DSA_HOST_ROWS_PINNED=1: staging event")?;
        }
        Ok(Self {
            ring: Mutex::new(Ring {
                base,
                bytes,
                events,
                recorded: [false; 2],
                streams: [0; 2],
                filled: [0; 2],
                last: None,
            }),
        })
    }

    /// 2026-10-15: Enqueue `parts` on `stream` from page-locked memory ([`step`]).
    pub fn upload(
        &self,
        gpu: &dyn GpuBackend,
        parts: &[(&[u8], DevicePtr)],
        stream: u64,
    ) -> Result<()> {
        // 2026-10-15: A captured copy would replay whatever the region holds at launch, not
        // this call's bytes; refused loudly, never staged.
        ensure!(
            !gpu.stream_is_capturing(stream),
            "METRALE_GLM_DSA_HOST_ROWS_PINNED=1: host-path rows staged under graph capture"
        );
        let mut ring = self.ring.lock().unwrap_or_else(|e| e.into_inner());
        let total: usize = parts.iter().map(|(b, _)| b.len()).sum();
        ensure!(
            total <= ring.bytes,
            "METRALE_GLM_DSA_HOST_ROWS_PINNED=1: {total} B of rows exceed the {} B staging",
            ring.bytes
        );
        let same = ring.last.is_some_and(|r| {
            let held = ring.held(r);
            held.len() == total && parts.iter().flat_map(|(b, _)| b.iter()).eq(held.iter())
        });
        let on_stream = ring.streams.map(|s| s == stream);
        let s = step(ring.last, same, ring.recorded, on_stream);
        let (Step::Reuse { region: r, wait } | Step::Write { region: r, wait }) = s;
        if wait {
            gpu.event_synchronize(ring.events[r])?;
        }
        if matches!(s, Step::Write { .. }) {
            ring.fill(r, parts);
        }
        let mut at = 0;
        for &(b, d) in parts {
            gpu.copy_h2d_async_retained(&ring.held(r)[at..at + b.len()], d, stream)?;
            at += b.len();
        }
        gpu.record_event(ring.events[r], stream)?;
        ring.recorded[r] = true;
        ring.streams[r] = stream;
        ring.last = Some(r);
        Ok(())
    }
}

/// 2026-10-15: The workspace's staging: built when [`host_rows_pinned`], else none (no
/// allocation). Logs once per process, either way, the page-locked bytes one workspace holds,
/// so every rank's serve log proves the lever's state and cost.
pub(super) fn staging_for(
    gpu: &dyn GpuBackend,
    max_rows: usize,
    bt_cap: usize,
) -> Result<Option<HostRowsStaging>> {
    let staging = if host_rows_pinned()? {
        Some(HostRowsStaging::alloc(gpu, max_rows, bt_cap)?)
    } else {
        None
    };
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| tracing::info!("{}", staging_readback(staging.as_ref(), max_rows, bt_cap)));
    Ok(staging)
}

/// 2026-10-15: [`staging_for`]'s log line, pure.
pub(super) fn staging_readback(
    staging: Option<&HostRowsStaging>,
    max_rows: usize,
    bt_cap: usize,
) -> String {
    match staging {
        Some(s) => {
            let bytes = s.ring.lock().unwrap_or_else(|e| e.into_inner()).bytes;
            format!(
                "METRALE_GLM_DSA_HOST_ROWS_PINNED=1: DSA host-path rows staged in page-locked \
                 memory, 2 x {bytes} B = {} B per DSA workspace ({max_rows} rows, {bt_cap} \
                 table entries)",
                2 * bytes
            )
        }
        None => "METRALE_GLM_DSA_HOST_ROWS_PINNED=0: DSA host-path rows uploaded from pageable \
                 memory, 0 B page-locked staging allocated"
            .to_string(),
    }
}

/// 2026-10-15: The four uploads of a host-path call: from `staging` when the lever built it,
/// else each from its pageable vector (the path before 2026-10-15).
pub fn upload_rows(
    gpu: &dyn GpuBackend,
    staging: Option<&HostRowsStaging>,
    parts: &[(&[u8], DevicePtr)],
    stream: u64,
) -> Result<()> {
    match staging {
        Some(s) => s.upload(gpu, parts, stream),
        None => {
            for &(b, d) in parts {
                gpu.copy_h2d_async(b, d, stream)?;
            }
            Ok(())
        }
    }
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
        let rows = HostRowBytes::new(
            self.layer_idx,
            k,
            seq_len,
            block_table,
            block_size,
            bt_block_size,
            w.bt_cap,
        )?;
        let (d_bt, d_sl, owned) = if self.persist_bt {
            (w.bt, w.sl, false)
        } else {
            (gpu.alloc(rows.bt.len().max(4))?, gpu.alloc(k * 4)?, true)
        };
        upload_rows(
            gpu,
            w.host_staging.as_ref(),
            &rows.parts(w.slot, w.q_pos, d_bt, d_sl),
            stream,
        )?;
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
