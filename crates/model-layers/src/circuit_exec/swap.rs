// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: `kv_swap_out` / `kv_swap_in` under the circuit (LIFECYCLE-DESIGN.md 15.10): a
//! sequence's swap record (`metrale_circuit::swap::SwapPlan`) moved between the device and a
//! writer or reader as asynchronous copies on a copy stream, through two page-locked staging
//! chunks so that one chunk's copies overlap the other's file I/O. The scheduler decides when a
//! sequence spills; this runs the copies.
//!
//! Owner: model-layers circuit executor (FEATURES workstream).
//! Invariants:
//! - The record's bytes are the plan's segments in order: the same file legacy writes.
//! - A swap-out's copies start after the work the compute stream has queued (an event), and a
//!   swap-in returns only once its copies are done and the compute stream is ordered after them.
//! - Every size is the plan's; the binding's pool strides and slot sizes are checked against it
//!   at build, so a layout the plan misdescribes refuses the build.
//! - A staging chunk holds the largest segment; segments never straddle chunks.

use std::io::{Read, Write};

use anyhow::{Context, Result, ensure};
use metrale_circuit::swap::{Piece, Segment, SwapPlan};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

/// 2026-10-03: The device side of the record, from the model.
#[derive(Debug, Clone)]
pub struct SwapBinding {
    /// 2026-10-03: Per attention layer, its K pool and K block stride, V pool and V block stride
    /// (`PagedKvCache::{k,v}_cache_ptr`: pool + block x stride).
    pub kv: Vec<[(DevicePtr, u64); 2]>,
    /// 2026-10-03: Per recurrent unit of the plan, the bytes one slot's unit takes in its pool
    /// (`SsmStatePool::h_stored_bytes`, `conv_bytes`).
    pub recurrent_bytes: Vec<u64>,
}

/// 2026-10-03: What a build needs for the runner: the binding, the stored formats of the keyed
/// states (`kv_cache_dtype`, `ssm_h_storage`) and the KV block size.
#[derive(Debug, Clone)]
pub struct SwapBoot {
    pub bind: SwapBinding,
    pub formats: std::collections::BTreeMap<String, metrale_circuit::state::StateDtype>,
    pub block_size: u64,
}

/// 2026-10-03: The runner of `circuit`'s swap record over `boot`.
pub fn build(
    gpu: &dyn GpuBackend,
    circuit: &metrale_circuit::Circuit,
    boot: SwapBoot,
) -> Result<SwapRunner> {
    let plan = SwapPlan::new(circuit, &boot.formats, boot.block_size)
        .context("planning the swap record")?;
    SwapRunner::new(gpu, plan, boot.bind)
}

/// 2026-10-03: The staging chunks of a runner: two of `chunk` bytes, page-locked.
struct Staging {
    host: [*mut u8; 2],
    chunk: usize,
}

// 2026-10-03: The chunks are owned by the runner, touched only through `&self` methods that the
// model serializes (one swap at a time, under the scheduler's effect), and freed in `free`.
unsafe impl Send for Staging {}
unsafe impl Sync for Staging {}

/// 2026-10-03: A built swap: the plan, the binding it was checked against, its staging and
/// copy stream.
pub struct SwapRunner {
    plan: SwapPlan,
    bind: SwapBinding,
    staging: Staging,
    copy_stream: u64,
    /// 2026-10-03: One per chunk (its copies are done), and one ordering the copy stream after
    /// the compute stream or back.
    done: [u64; 2],
    order: u64,
}

/// 2026-10-03: Consecutive segments grouped so that each group fits `chunk` bytes.
pub fn chunks(segments: &[Segment], chunk: u64) -> Vec<std::ops::Range<usize>> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut bytes = 0;
    for (i, s) in segments.iter().enumerate() {
        if bytes + s.bytes > chunk && i > start {
            out.push(start..i);
            start = i;
            bytes = 0;
        }
        bytes += s.bytes;
    }
    if start < segments.len() {
        out.push(start..segments.len());
    }
    out
}

impl SwapRunner {
    /// 2026-10-03: The runner of `plan` over `bind`, its two chunks each twice the largest
    /// segment (so a chunk holds several KV blocks, and any recurrent unit). Refuses a binding
    /// the plan misdescribes.
    pub fn new(gpu: &dyn GpuBackend, plan: SwapPlan, bind: SwapBinding) -> Result<Self> {
        ensure!(
            bind.kv.len() == plan.kv.len(),
            "the model has {} KV layers, the swap plan {}",
            bind.kv.len(),
            plan.kv.len()
        );
        for (i, (l, [(_, ks), (_, vs)])) in plan.kv.iter().zip(&bind.kv).enumerate() {
            ensure!(
                (l.k_block_bytes, l.v_block_bytes) == (*ks, *vs),
                "KV layer {i}: the plan's block bytes {:?} are not the pool's strides {:?}",
                (l.k_block_bytes, l.v_block_bytes),
                (ks, vs)
            );
        }
        let plan_rec: Vec<u64> = plan.recurrent.iter().map(|r| r.bytes).collect();
        ensure!(
            plan_rec == bind.recurrent_bytes,
            "the plan's recurrent units {plan_rec:?} are not the pool's {:?}",
            bind.recurrent_bytes
        );
        let chunk = usize::try_from(plan.staging_chunk())?;
        let a = gpu.alloc_host_pinned(chunk)?;
        let b = match gpu.alloc_host_pinned(chunk) {
            Ok(b) => b,
            Err(e) => {
                gpu.free_host_pinned(a, chunk).ok();
                return Err(e);
            }
        };
        Ok(SwapRunner {
            plan,
            bind,
            staging: Staging {
                host: [a, b],
                chunk,
            },
            copy_stream: gpu.create_stream()?,
            done: [gpu.create_event()?, gpu.create_event()?],
            order: gpu.create_event()?,
        })
    }

    /// 2026-10-03: Bytes of the record of a sequence of `blocks` blocks.
    pub fn record_bytes(&self, blocks: u64) -> u64 {
        self.plan.record_bytes(blocks)
    }

    /// 2026-10-03: Page-locked host bytes the runner holds (its two chunks).
    pub fn staging_bytes(&self) -> u64 {
        2 * self.staging.chunk as u64
    }

    /// 2026-10-03: The device address of `piece` for a sequence whose blocks are `table` and
    /// whose recurrent units are at `recurrent`.
    fn address(&self, piece: Piece, table: &[u32], recurrent: &[DevicePtr]) -> Result<DevicePtr> {
        let block = |b: usize| -> Result<usize> {
            Ok(*table.get(b).context("a segment past the block table")? as usize)
        };
        Ok(match piece {
            Piece::K { attn, block: b } => {
                let (pool, stride) = self.bind.kv[attn][0];
                pool.offset(block(b)? * stride as usize)
            }
            Piece::V { attn, block: b } => {
                let (pool, stride) = self.bind.kv[attn][1];
                pool.offset(block(b)? * stride as usize)
            }
            Piece::Recurrent { index } => *recurrent
                .get(index)
                .context("a recurrent unit the sequence does not hold")?,
        })
    }

    /// 2026-10-03: `len` bytes of chunk `buf` from byte `at`. Refuses a range past the chunk.
    fn region(&self, buf: usize, at: usize, len: usize) -> Result<*mut u8> {
        ensure!(
            at + len <= self.staging.chunk,
            "a staging range past its chunk"
        );
        // 2026-10-03: SAFETY: in bounds of the live `chunk`-byte allocation `host[buf]`.
        Ok(unsafe { self.staging.host[buf].add(at) })
    }

    /// 2026-10-03: `kv_swap_out`: write the record of the sequence (`table`, `recurrent`) to
    /// `w`, after the work `compute_stream` has queued.
    pub fn swap_out(
        &self,
        gpu: &dyn GpuBackend,
        compute_stream: u64,
        (table, recurrent): (&[u32], &[DevicePtr]),
        w: &mut dyn Write,
    ) -> Result<()> {
        ensure!(
            recurrent.len() == self.plan.recurrent.len(),
            "recurrent units"
        );
        gpu.record_event(self.order, compute_stream)?;
        gpu.stream_wait_event(self.copy_stream, self.order)?;
        let segs = self.plan.segments(table.len());
        let groups = chunks(&segs, self.staging.chunk as u64);
        let mut pending: Option<(usize, usize)> = None;
        for (i, g) in groups.iter().enumerate() {
            let buf = i % 2;
            let base = segs[g.start].offset;
            for s in &segs[g.clone()] {
                let len = s.bytes as usize;
                let p = self.region(buf, (s.offset - base) as usize, len)?;
                // 2026-10-03: SAFETY: `region` checked the range; the model runs one swap at a
                // time, and this range is not read before `done[buf]` is waited on.
                let dst = unsafe { std::slice::from_raw_parts_mut(p, len) };
                gpu.copy_d2h_async(
                    self.address(s.piece, table, recurrent)?,
                    dst,
                    self.copy_stream,
                )?;
            }
            gpu.record_event(self.done[buf], self.copy_stream)?;
            if let Some((pb, plen)) = pending.take() {
                gpu.event_synchronize(self.done[pb])?;
                self.write_chunk(pb, plen, w)?;
            }
            let end = segs[g.end - 1].offset + segs[g.end - 1].bytes;
            pending = Some((buf, (end - base) as usize));
        }
        if let Some((pb, plen)) = pending {
            gpu.event_synchronize(self.done[pb])?;
            self.write_chunk(pb, plen, w)?;
        }
        w.flush()?;
        Ok(())
    }

    /// 2026-10-03: `kv_swap_in`: read the record of a sequence of `table.len()` blocks from `r`
    /// into its (freshly allocated) blocks and its recurrent units; the compute stream is ordered
    /// after the copies.
    pub fn swap_in(
        &self,
        gpu: &dyn GpuBackend,
        compute_stream: u64,
        (table, recurrent): (&[u32], &[DevicePtr]),
        r: &mut dyn Read,
    ) -> Result<()> {
        ensure!(
            recurrent.len() == self.plan.recurrent.len(),
            "recurrent units"
        );
        gpu.record_event(self.order, compute_stream)?;
        gpu.stream_wait_event(self.copy_stream, self.order)?;
        let segs = self.plan.segments(table.len());
        let groups = chunks(&segs, self.staging.chunk as u64);
        let mut used = [false; 2];
        for (i, g) in groups.iter().enumerate() {
            let buf = i % 2;
            if used[buf] {
                gpu.event_synchronize(self.done[buf])?;
            }
            let base = segs[g.start].offset;
            let end = segs[g.end - 1].offset + segs[g.end - 1].bytes;
            let len = (end - base) as usize;
            let p = self.region(buf, 0, len)?;
            // 2026-10-03: SAFETY: `region` checked the range, and `done[buf]` was waited on, so no
            // copy still reads it.
            r.read_exact(unsafe { std::slice::from_raw_parts_mut(p, len) })?;
            for s in &segs[g.clone()] {
                let len = s.bytes as usize;
                let p = self.region(buf, (s.offset - base) as usize, len)?;
                // 2026-10-03: SAFETY: in bounds; it stays unchanged until `done[buf]` is waited on.
                let src = unsafe { std::slice::from_raw_parts(p, len) };
                let dst = self.address(s.piece, table, recurrent)?;
                gpu.copy_h2d_async_retained(src, dst, self.copy_stream)?;
            }
            gpu.record_event(self.done[buf], self.copy_stream)?;
            used[buf] = true;
        }
        gpu.record_event(self.order, self.copy_stream)?;
        gpu.stream_wait_event(compute_stream, self.order)?;
        for (buf, u) in used.iter().enumerate() {
            if *u {
                gpu.event_synchronize(self.done[buf])?;
            }
        }
        Ok(())
    }

    fn write_chunk(&self, buf: usize, len: usize, w: &mut dyn Write) -> Result<()> {
        let p = self.region(buf, 0, len)?;
        // 2026-10-03: SAFETY: in bounds, and its copies are done (the caller waited on them).
        w.write_all(unsafe { std::slice::from_raw_parts(p, len) })?;
        Ok(())
    }

    /// 2026-10-03: Free the staging and the events.
    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        for h in self.staging.host {
            gpu.free_host_pinned(h, self.staging.chunk)?;
        }
        for e in self.done.into_iter().chain([self.order]) {
            gpu.destroy_event(e)?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "swap_tests.rs"]
mod swap_tests;
