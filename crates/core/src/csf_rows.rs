// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The row-major device form of an NVFP4 `csf_32x2_v1` scale plane, which kernels
//! decode in place of the native `[R, C]` scale bytes.
//!
//! Owner: metrale-core.
//!
//! The checkpoint's block order (`csf_blocks`) suits storage, not a kernel that wants the scales
//! of one row and a few adjacent groups. This form keeps the same information row-major, in one
//! blob per projection:
//!
//! | region     | bytes                    | contents                                       |
//! |------------|--------------------------|------------------------------------------------|
//! | codes      | `R * C / 2`              | 4-bit fields, byte `(r*C + c) / 2`, even `c` low |
//! | bases      | `R`                      | one base per row                               |
//! | row_ptr    | `4 * (R + 1)`            | u32 exception prefix per row                   |
//! | (pad)      | to a multiple of 16      | zero                                           |
//! | exceptions | `round_up(2 * E, 16)`    | u16 `(value << 8) \| c`, increasing `c` per row |
//!
//! Decode: `scale[r][c] = bases[r] + field(r, c)`, then a record of row `r` whose column is `c`
//! overwrites it with its value. `E` is a capacity: records `row_ptr[R]..E` are zero padding.
//!
//! [`encode_csf_rows`] builds the blob from the plane's decoded native scales and the plane's own
//! bases, so a position is a record exactly when `scale - base` is outside `0..16`; every such
//! position is a record of the checkpoint plane too, so the checkpoint's record count is always
//! a sufficient capacity. [`CsfRowsLayout::component_charge`] gives the blob size per checkpoint
//! component, so a pre-flight can charge it from tensor shapes alone.
//!
//! Invariants enforced (each failure is an error, never a clamp):
//! - `R` is a positive multiple of 32 (as in `csf_blocks`), `C` a multiple of 4 in `4..=256`
//!   (a column fits a record byte, and 4 scales are one aligned 2-byte code load);
//! - every base is at most 240, so a base plus a field never wraps;
//! - row prefixes start at 0, never decrease and end within the capacity; a row's record
//!   columns are inside the row and strictly increasing.

use crate::csf_blocks::{
    BLOCK_ROWS, CsfOffsetWidth, SUFFIX_BASES, SUFFIX_CODES, SUFFIX_EXCEPTIONS,
};
use anyhow::{Context, Result, ensure};

/// 2026-10-17: Region alignment inside a blob, and the alignment a blob needs in device memory.
pub const CSF_ROWS_ALIGN: usize = 16;
/// 2026-10-17: Largest supported `C` (a record stores its column in one byte).
pub const CSF_ROWS_MAX_COLUMNS: usize = 256;
/// 2026-10-17: `C` must be a multiple of this (4 fields are one aligned u16 code load).
pub const CSF_ROWS_COLUMN_QUANTUM: usize = 4;

fn round16(n: usize) -> Option<usize> {
    n.checked_next_multiple_of(CSF_ROWS_ALIGN)
}

/// 2026-10-17: Whether an `R x C` plane has a device form.
pub fn csf_rows_supported(rows: usize, cols: usize) -> bool {
    rows > 0
        && rows.is_multiple_of(BLOCK_ROWS)
        && cols.is_multiple_of(CSF_ROWS_COLUMN_QUANTUM)
        && (CSF_ROWS_COLUMN_QUANTUM..=CSF_ROWS_MAX_COLUMNS).contains(&cols)
        && rows
            .checked_mul(cols)
            .is_some_and(|n| n <= u32::MAX as usize)
}

/// 2026-10-17: Byte offsets of the four regions of one blob.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CsfRowsLayout {
    /// 2026-10-17: Scale rows `R`.
    pub rows: usize,
    /// 2026-10-17: Scale columns `C` (`K / 16`).
    pub cols: usize,
    /// 2026-10-17: Exception records the blob has room for.
    pub capacity: usize,
}

impl CsfRowsLayout {
    /// 2026-10-17: Validate an `R x C` plane with room for `capacity` records.
    pub fn new(rows: usize, cols: usize, capacity: usize) -> Result<Self> {
        ensure!(
            csf_rows_supported(rows, cols),
            "csf rows: [{rows}, {cols}] has no device form (rows a positive multiple of \
             {BLOCK_ROWS}, columns a multiple of {CSF_ROWS_COLUMN_QUANTUM} up to \
             {CSF_ROWS_MAX_COLUMNS})"
        );
        let layout = Self {
            rows,
            cols,
            capacity,
        };
        ensure!(
            capacity <= u32::MAX as usize && layout.checked_bytes().is_some(),
            "csf rows: capacity {capacity} overflows"
        );
        Ok(layout)
    }

    /// 2026-10-17: Bytes of the codes region (offset 0).
    pub fn codes_bytes(&self) -> usize {
        self.rows * self.cols / 2
    }

    /// 2026-10-17: Offset of the bases.
    pub fn bases_offset(&self) -> usize {
        self.codes_bytes()
    }

    /// 2026-10-17: Offset of the row prefixes (4-aligned: `R*C/2` and `R` are multiples of 32).
    pub fn row_ptr_offset(&self) -> usize {
        self.codes_bytes() + self.rows
    }

    fn checked_bytes(&self) -> Option<usize> {
        let head = round16(self.rows.checked_mul(5)?.checked_add(4)?)?;
        let tail = round16(self.capacity.checked_mul(2)?)?;
        self.codes_bytes().checked_add(head)?.checked_add(tail)
    }

    /// 2026-10-17: Offset of the exception records (16-aligned).
    pub fn exceptions_offset(&self) -> usize {
        self.codes_bytes() + (5 * self.rows + 4).next_multiple_of(CSF_ROWS_ALIGN)
    }

    /// 2026-10-17: Total blob bytes, a multiple of [`CSF_ROWS_ALIGN`].
    pub fn bytes(&self) -> usize {
        self.exceptions_offset() + (2 * self.capacity).next_multiple_of(CSF_ROWS_ALIGN)
    }

    /// 2026-10-17: The share of a blob charged to the checkpoint component `suffix` of `elems`
    /// elements: the codes charge their own bytes, the bases the bases, prefixes and padding
    /// (`R = elems`), the exceptions their padded records; the other components nothing. The
    /// three shares of one plane sum to [`Self::bytes`] with the checkpoint's record count as
    /// the capacity. `None` on overflow.
    pub fn component_charge(suffix: &str, elems: usize) -> Option<usize> {
        match suffix {
            SUFFIX_CODES => Some(elems),
            SUFFIX_BASES => round16(elems.checked_mul(5)?.checked_add(4)?),
            SUFFIX_EXCEPTIONS => round16(elems.checked_mul(2)?),
            _ => Some(0),
        }
    }
}

fn is_record(scale: u8, base: u8) -> bool {
    scale < base || scale - base >= 16
}

/// 2026-10-17: Records the device form of `scales` (row-major `[R, C]`) needs under `bases`.
pub fn csf_rows_records(scales: &[u8], bases: &[u8], cols: usize) -> usize {
    scales
        .chunks(cols.max(1))
        .zip(bases)
        .map(|(row, &b)| row.iter().filter(|&&s| is_record(s, b)).count())
        .sum()
}

/// 2026-10-17: The device blob of the native row-major scales `scales` under the plane's
/// `bases`, padded to `layout.bytes()`.
pub fn encode_csf_rows(scales: &[u8], bases: &[u8], layout: &CsfRowsLayout) -> Result<Vec<u8>> {
    let (rows, cols) = (layout.rows, layout.cols);
    ensure!(
        scales.len() == rows * cols && bases.len() == rows,
        "csf rows: {} scales and {} bases for [{rows}, {cols}]",
        scales.len(),
        bases.len()
    );
    let max_base = CsfOffsetWidth::Nvfp4.max_base();
    if let Some(r) = bases.iter().position(|&b| b > max_base) {
        anyhow::bail!("csf rows: base {} of row {r} exceeds {max_base}", bases[r]);
    }
    let need = csf_rows_records(scales, bases, cols);
    ensure!(
        need <= layout.capacity,
        "csf rows: {need} records exceed the capacity {}",
        layout.capacity
    );
    let mut blob = vec![0u8; layout.bytes()];
    let (ptr_at, exc_at) = (layout.row_ptr_offset(), layout.exceptions_offset());
    let mut next = 0usize;
    for r in 0..rows {
        let base = bases[r];
        blob[layout.bases_offset() + r] = base;
        blob[ptr_at + 4 * r..ptr_at + 4 * r + 4].copy_from_slice(&(next as u32).to_le_bytes());
        for c in 0..cols {
            let s = scales[r * cols + c];
            let field = if is_record(s, base) {
                let rec = (u16::from(s) << 8) | c as u16;
                blob[exc_at + 2 * next..exc_at + 2 * next + 2].copy_from_slice(&rec.to_le_bytes());
                next += 1;
                0
            } else {
                s - base
            };
            blob[(r * cols + c) / 2] |= field << (4 * (c & 1));
        }
    }
    blob[ptr_at + 4 * rows..ptr_at + 4 * rows + 4].copy_from_slice(&(next as u32).to_le_bytes());
    Ok(blob)
}

/// 2026-10-17: The native row-major scales of a device blob (the kernels' reference).
pub fn decode_csf_rows(blob: &[u8], layout: &CsfRowsLayout) -> Result<Vec<u8>> {
    let (rows, cols) = (layout.rows, layout.cols);
    ensure!(
        blob.len() == layout.bytes(),
        "csf rows: blob of {} bytes, layout needs {}",
        blob.len(),
        layout.bytes()
    );
    let max_base = CsfOffsetWidth::Nvfp4.max_base();
    let ptr = |r: usize| -> usize {
        let at = layout.row_ptr_offset() + 4 * r;
        u32::from_le_bytes(blob[at..at + 4].try_into().expect("4 bytes")) as usize
    };
    ensure!(ptr(0) == 0, "csf rows: the first row prefix is {}", ptr(0));
    let exc_at = layout.exceptions_offset();
    let mut out = vec![0u8; rows * cols];
    for r in 0..rows {
        let base = blob[layout.bases_offset() + r];
        ensure!(
            base <= max_base,
            "csf rows: base {base} of row {r} exceeds {max_base}"
        );
        for c in 0..cols {
            out[r * cols + c] = base + ((blob[(r * cols + c) / 2] >> (4 * (c & 1))) & 0xF);
        }
        let (lo, hi) = (ptr(r), ptr(r + 1));
        ensure!(
            lo <= hi && hi <= layout.capacity,
            "csf rows: row {r} records {lo}..{hi} outside 0..{}",
            layout.capacity
        );
        let mut last: Option<usize> = None;
        for e in lo..hi {
            let rec = u16::from_le_bytes([blob[exc_at + 2 * e], blob[exc_at + 2 * e + 1]]);
            let c = usize::from(rec & 0xFF);
            ensure!(
                c < cols && last.is_none_or(|l| l < c),
                "csf rows: row {r} record {e} column {c} is outside the row or out of order"
            );
            last = Some(c);
            out[r * cols + c] = (rec >> 8) as u8;
        }
    }
    Ok(out)
}

/// 2026-10-17: [`encode_csf_rows`] with the capacity the checkpoint plane's `records` give.
pub fn encode_csf_rows_for_plane(
    scales: &[u8],
    bases: &[u8],
    cols: usize,
    records: usize,
) -> Result<(CsfRowsLayout, Vec<u8>)> {
    let layout = CsfRowsLayout::new(bases.len(), cols, records)
        .with_context(|| format!("csf rows: plane of {} rows", bases.len()))?;
    let blob = encode_csf_rows(scales, bases, &layout)?;
    Ok((layout, blob))
}

#[cfg(test)]
#[path = "csf_rows_tests.rs"]
mod tests;
