// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: The batched decode's and the batched verify's attention block tables, in a
//! device buffer of their own, refreshed per step by uploading only the entries that changed.
//!
//! Each region holds rows of `stride = max_blocks_per_seq` i32 entries, the fixed stride the
//! captured graphs read. A step describes each row as its live blocks followed by a fill value
//! ([`RowSpec`]); the full upload this replaces wrote exactly that, every entry of every row,
//! every step: `rows * (--max-seq-len / block_size + 1) * 4` bytes, built and copied on the
//! host. [`TableMirror`] remembers what each row holds on the device, so a step uploads only
//! the entries that differ, and its host work and copy size follow the live context, not
//! `--max-seq-len`. The device bytes of every row a step describes are unchanged.
//!
//! Owner: model-engine batched decode / verify metadata.
//! Invariants:
//! - Only [`BlockTables::upload`] writes the buffer, so a row the mirror knows holds exactly
//!   what the mirror says; an unknown row (first use, or after a failed copy) is written whole.
//! - After an upload, row `r < specs.len()` of the region equals `specs[r]`'s full row.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::buffers::{BufferArena, VERIFY_ROW_CAP};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use parking_lot::Mutex;

/// 2026-10-13: One row as a step wants it on the device: `blocks` (cut at the stride), then
/// `fill` in every later entry.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RowSpec<'a> {
    pub(crate) blocks: &'a [u32],
    pub(crate) fill: i32,
}

impl RowSpec<'_> {
    fn live(&self, stride: usize) -> usize {
        self.blocks.len().min(stride)
    }

    fn at(&self, i: usize) -> i32 {
        self.blocks.get(i).map_or(self.fill, |&b| b as i32)
    }
}

/// 2026-10-13: What one row holds on the device: `live` entries, then `fill` to the stride.
#[derive(Clone, Debug, PartialEq, Eq)]
struct RowMirror {
    live: Vec<i32>,
    fill: i32,
}

/// 2026-10-13: A run of entries to copy: `values` go to entries `start..start + values.len()`
/// of the region (row-major, `stride` entries per row).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Span {
    pub(crate) start: usize,
    pub(crate) values: Vec<i32>,
}

/// 2026-10-13: The host's record of a region's rows (`None` = unknown).
#[derive(Debug)]
pub(crate) struct TableMirror {
    stride: usize,
    rows: Vec<Option<RowMirror>>,
}

impl TableMirror {
    pub(crate) fn new(stride: usize, rows: usize) -> Self {
        Self {
            stride,
            rows: vec![None; rows],
        }
    }

    /// 2026-10-13: Forget every row, so the next plan writes each row it describes whole.
    pub(crate) fn forget(&mut self) {
        self.rows.iter_mut().for_each(|r| *r = None);
    }

    /// 2026-10-13: The copies that make rows `0..specs.len()` equal `specs`, and the mirror
    /// updated as if they ran. A known row with the same fill copies only its entries from the
    /// first to the last that differ, all within `0..max(old live, new live)`; an unknown row
    /// or a fill change copies the whole row. Adjacent copies are merged.
    pub(crate) fn plan(&mut self, specs: &[RowSpec<'_>]) -> Result<Vec<Span>> {
        ensure!(
            specs.len() <= self.rows.len(),
            "block-table plan: {} rows exceed the region's {}",
            specs.len(),
            self.rows.len()
        );
        let stride = self.stride;
        let mut spans: Vec<Span> = Vec::new();
        for (r, (spec, slot)) in specs.iter().zip(self.rows.iter_mut()).enumerate() {
            let new_len = spec.live(stride);
            let changed = match slot {
                Some(m) if m.fill == spec.fill => {
                    let end = m.live.len().max(new_len);
                    let differs = |i: usize| m.live.get(i).copied().unwrap_or(m.fill) != spec.at(i);
                    let first = (0..end).find(|&i| differs(i)).unwrap_or(end);
                    let keep = first.min(m.live.len()).min(new_len);
                    let range = (first < end).then(|| {
                        let last = (first..end).rev().find(|&i| differs(i)).unwrap_or(first);
                        first..last + 1
                    });
                    m.live.truncate(keep);
                    m.live
                        .extend(spec.blocks[keep..new_len].iter().map(|&b| b as i32));
                    range
                }
                _ => {
                    *slot = Some(RowMirror {
                        live: spec.blocks[..new_len].iter().map(|&b| b as i32).collect(),
                        fill: spec.fill,
                    });
                    Some(0..stride)
                }
            };
            let Some(range) = changed else { continue };
            let start = r * stride + range.start;
            let values = range.map(|i| spec.at(i));
            match spans.last_mut() {
                Some(s) if s.start + s.values.len() == start => s.values.extend(values),
                _ => spans.push(Span {
                    start,
                    values: values.collect(),
                }),
            }
        }
        Ok(spans)
    }
}

/// 2026-10-13: Which region of [`BlockTables`] a step fills.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Region {
    /// The batched decode (`upload_batch_metadata_fixed`): `decode_meta().rows()` rows.
    Decode,
    /// The batched verify (`stage_verify_metadata`): `VERIFY_ROW_CAP` rows.
    Verify,
}

/// 2026-10-13: The two regions' device buffer (decode rows, then verify rows), allocated once
/// at a fixed address so captured graphs stay valid, and their mirrors.
pub(crate) struct BlockTables {
    dev: DevicePtr,
    stride: usize,
    decode_rows: usize,
    decode: Mutex<TableMirror>,
    verify: Mutex<TableMirror>,
}

impl BlockTables {
    /// 2026-10-13: `stride` entries per row; `buffers.decode_meta().rows()` decode rows and
    /// `VERIFY_ROW_CAP` verify rows. The device bytes start unknown to both mirrors.
    pub(crate) fn alloc(gpu: &dyn GpuBackend, stride: u32, buffers: &BufferArena) -> Result<Self> {
        Self::with_rows(
            gpu,
            stride as usize,
            buffers.decode_meta().rows(),
            VERIFY_ROW_CAP,
        )
    }

    fn with_rows(
        gpu: &dyn GpuBackend,
        stride: usize,
        decode_rows: usize,
        verify_rows: usize,
    ) -> Result<Self> {
        let dev = gpu.alloc((decode_rows + verify_rows) * stride * 4)?;
        Ok(Self {
            dev,
            stride,
            decode_rows,
            decode: Mutex::new(TableMirror::new(stride, decode_rows)),
            verify: Mutex::new(TableMirror::new(stride, verify_rows)),
        })
    }

    /// 2026-10-13: Where `region`'s row 0 starts.
    pub(crate) fn ptr(&self, region: Region) -> DevicePtr {
        match region {
            Region::Decode => self.dev,
            Region::Verify => self.dev.offset(self.decode_rows * self.stride * 4),
        }
    }

    /// 2026-10-13: Make rows `0..specs.len()` of `region` equal `specs` on `stream`, copying
    /// only what differs from the mirror. A failed copy leaves the region unknown.
    pub(crate) fn upload(
        &self,
        region: Region,
        gpu: &dyn GpuBackend,
        specs: &[RowSpec<'_>],
        stream: u64,
    ) -> Result<()> {
        let mut mirror = match region {
            Region::Decode => self.decode.lock(),
            Region::Verify => self.verify.lock(),
        };
        let base = self.ptr(region);
        let spans = mirror.plan(specs)?;
        for span in &spans {
            let bytes: Vec<u8> = span.values.iter().flat_map(|v| v.to_le_bytes()).collect();
            if let Err(e) = gpu.copy_h2d_async(&bytes, base.offset(span.start * 4), stream) {
                mirror.forget();
                return Err(e);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "block_tables_tests.rs"]
mod tests;
