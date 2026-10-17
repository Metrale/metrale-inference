// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Host decoder for the `csf_32x2_v1` lossless FP4 scale encoding.
//!
//! Owner: metrale-core.
//!
//! A checkpoint that declares `weight_scale_encoding: "csf_32x2_v1"` stores
//! each FP4 `weight_scale` matrix (`R` rows, `C` columns, `C = K/16` for
//! NVFP4 and `K/32` for MXFP4) as five components:
//!
//! | suffix              | dtype | contents                                   |
//! |---------------------|-------|--------------------------------------------|
//! | `.csf32_bases`      | u8    | one palette base per logical row           |
//! | `.csf32_codes`      | u8    | packed offset fields, block order          |
//! | `.csf32_page_bases` | u32   | byte origin of each page in the arena      |
//! | `.csf32_offsets`    | u16   | page-relative record prefixes, `B+P` items |
//! | `.csf32_exceptions` | u16   | `(value << 8) \| (2*local_row + local_col)`|
//!
//! Geometry. A block is 32 rows by 2 columns; a final odd column gives a
//! width-1 block. Blocks form 128-row by 4-column tiles (4 row blocks by 2
//! column blocks, clipped at the matrix edge). Tiles are visited row-major;
//! inside a tile the first column pair's row blocks (top to bottom) precede
//! the second pair's. A page is 32 tiles (the last may be shorter).
//!
//! Codes. Each block packs `32 * width` fields row-major
//! (`local_row * width + local_col`), least-significant field first, 4 bits
//! per field for NVFP4 and 1 bit for MXFP4. The stream holds exactly
//! `R * C * bits / 8` bytes.
//!
//! Decode. `scale[r][c] = bases[r] + field`, then every exception record of
//! the block overwrites its position with its value byte. Block `b` of page
//! `p` owns records `page_bases[p] / 2 + offsets[b + p] ..
//! page_bases[p] / 2 + offsets[b + p + 1]`.
//!
//! Invariants enforced (each failure is an error, never a clamp):
//! - `R` is a positive multiple of 32, `C` is positive;
//! - every base is at most `256 - 2^bits`;
//! - component byte lengths match the geometry exactly;
//! - each page's first prefix is 0, and the records form one consecutive
//!   arena that the blocks consume completely (smaller than 4 GiB);
//! - a block holds at most `32 * width` records, positions below 64 whose
//!   column is inside the block, strictly increasing.
//!
//! The output is the row-major `[R, C]` byte matrix a native checkpoint
//! stores under `weight_scale`, so every downstream consumer is unchanged.

use anyhow::{Result, bail, ensure};

/// 2026-10-17: Rows per block.
pub const BLOCK_ROWS: usize = 32;
/// 2026-10-17: Columns per block (a final odd column gives width 1).
pub const BLOCK_COLUMNS: usize = 2;
/// 2026-10-17: Row blocks per tile (128 rows).
pub const TILE_ROW_BLOCKS: usize = 4;
/// 2026-10-17: Column blocks per tile (4 columns).
pub const TILE_COLUMN_BLOCKS: usize = 2;
/// 2026-10-17: Tiles per index page.
pub const PAGE_GROUPS: usize = 32;
/// 2026-10-17: The `weight_scale_encoding` value this module decodes.
pub const ENCODING_NAME: &str = "csf_32x2_v1";
/// 2026-10-17: Tensor-name suffixes of the five components, appended to the
/// native scale name (e.g. `...down_proj.weight_scale`).
pub const SUFFIX_BASES: &str = ".csf32_bases";
/// 2026-10-17: See [`SUFFIX_BASES`].
pub const SUFFIX_CODES: &str = ".csf32_codes";
/// 2026-10-17: See [`SUFFIX_BASES`].
pub const SUFFIX_PAGE_BASES: &str = ".csf32_page_bases";
/// 2026-10-17: See [`SUFFIX_BASES`].
pub const SUFFIX_OFFSETS: &str = ".csf32_offsets";
/// 2026-10-17: See [`SUFFIX_BASES`].
pub const SUFFIX_EXCEPTIONS: &str = ".csf32_exceptions";

/// 2026-10-17: Offset-field width, fixed by the quantization recipe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CsfOffsetWidth {
    /// 2026-10-17: NVFP4 (E4M3 scale bytes), 4-bit fields.
    Nvfp4,
    /// 2026-10-17: MXFP4 (E8M0 scale bytes), 1-bit fields.
    Mxfp4,
}

impl CsfOffsetWidth {
    /// 2026-10-17: Bits per offset field.
    pub const fn bits(self) -> usize {
        match self {
            Self::Nvfp4 => 4,
            Self::Mxfp4 => 1,
        }
    }

    /// 2026-10-17: Largest legal row base, `256 - 2^bits`.
    pub const fn max_base(self) -> u8 {
        (256 - (1usize << self.bits())) as u8
    }
}

/// 2026-10-17: Geometry derived from `(rows, columns, width)` alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CsfGeometry {
    /// 2026-10-17: Logical scale rows `R`.
    pub rows: usize,
    /// 2026-10-17: Logical scale columns `C`.
    pub columns: usize,
    /// 2026-10-17: Offset-field width.
    pub width: CsfOffsetWidth,
    /// 2026-10-17: `R / 32`.
    pub row_blocks: usize,
    /// 2026-10-17: `ceil(C / 2)`.
    pub column_blocks: usize,
    /// 2026-10-17: Tiles (index groups), clipped tiles included.
    pub groups: usize,
    /// 2026-10-17: Index pages, `ceil(groups / 32)`.
    pub pages: usize,
}

impl CsfGeometry {
    /// 2026-10-17: Validate the extents and derive the counts.
    pub fn new(rows: usize, columns: usize, width: CsfOffsetWidth) -> Result<Self> {
        ensure!(
            rows > 0 && rows.is_multiple_of(BLOCK_ROWS),
            "csf: rows {rows} must be a positive multiple of {BLOCK_ROWS}"
        );
        ensure!(columns > 0, "csf: columns must be positive");
        ensure!(
            rows.checked_mul(columns)
                .is_some_and(|n| n <= isize::MAX as usize),
            "csf: {rows} x {columns} overflows"
        );
        let row_blocks = rows / BLOCK_ROWS;
        let column_blocks = columns.div_ceil(BLOCK_COLUMNS);
        let groups =
            row_blocks.div_ceil(TILE_ROW_BLOCKS) * column_blocks.div_ceil(TILE_COLUMN_BLOCKS);
        Ok(Self {
            rows,
            columns,
            width,
            row_blocks,
            column_blocks,
            groups,
            pages: groups.div_ceil(PAGE_GROUPS),
        })
    }

    /// 2026-10-17: Total blocks `B`.
    pub fn blocks(&self) -> usize {
        self.row_blocks * self.column_blocks
    }

    /// 2026-10-17: Exact byte length of `.csf32_codes`.
    pub fn codes_len(&self) -> usize {
        self.rows * self.columns * self.width.bits() / 8
    }

    /// 2026-10-17: Number of `.csf32_offsets` entries, `B + P`.
    pub fn offsets_entries(&self) -> usize {
        self.blocks() + self.pages
    }

    /// 2026-10-17: Encoded bytes excluding the variable exception arena.
    pub fn fixed_bytes(&self) -> usize {
        self.rows + self.codes_len() + 4 * self.pages + 2 * self.offsets_entries()
    }
}

/// 2026-10-17: Borrowed raw component bytes of one encoded scale matrix.
/// Integer components are little-endian byte views, as stored on disk.
#[derive(Clone, Copy, Debug)]
pub struct CsfPlaneRef<'a> {
    /// 2026-10-17: Validated geometry.
    pub geometry: CsfGeometry,
    /// 2026-10-17: `.csf32_bases`, `R` bytes.
    pub bases: &'a [u8],
    /// 2026-10-17: `.csf32_codes`.
    pub codes: &'a [u8],
    /// 2026-10-17: `.csf32_page_bases`, `4 * P` bytes.
    pub page_bases: &'a [u8],
    /// 2026-10-17: `.csf32_offsets`, `2 * (B + P)` bytes.
    pub offsets: &'a [u8],
    /// 2026-10-17: `.csf32_exceptions`, `2 * records` bytes.
    pub exceptions: &'a [u8],
}

#[inline(always)]
fn le_u16(bytes: &[u8], index: usize) -> u16 {
    u16::from_le_bytes([bytes[2 * index], bytes[2 * index + 1]])
}

#[inline(always)]
fn le_u32(bytes: &[u8], index: usize) -> u32 {
    let i = 4 * index;
    u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]])
}

fn check_lengths(plane: &CsfPlaneRef<'_>) -> Result<()> {
    let g = &plane.geometry;
    for (name, got, want) in [
        ("bases", plane.bases.len(), g.rows),
        ("codes", plane.codes.len(), g.codes_len()),
        ("page_bases", plane.page_bases.len(), 4 * g.pages),
        ("offsets", plane.offsets.len(), 2 * g.offsets_entries()),
    ] {
        ensure!(
            got == want,
            "csf: {name} holds {got} bytes, geometry needs {want}"
        );
    }
    ensure!(
        plane.exceptions.len().is_multiple_of(2),
        "csf: exceptions byte length {} is odd",
        plane.exceptions.len()
    );
    ensure!(
        (plane.exceptions.len() as u64) < (1u64 << 32),
        "csf: exception arena must be smaller than 4 GiB"
    );
    let max_base = g.width.max_base();
    if let Some((row, &base)) = plane.bases.iter().enumerate().find(|(_, b)| **b > max_base) {
        bail!("csf: row {row} base {base} exceeds {max_base}");
    }
    Ok(())
}

/// 2026-10-17: Decode one plane into the row-major `[R, C]` scale bytes.
///
/// `out` must hold exactly `R * C` bytes. On error `out` is unspecified.
pub fn decode_csf_32x2_v1(plane: &CsfPlaneRef<'_>, out: &mut [u8]) -> Result<()> {
    check_lengths(plane)?;
    let g = plane.geometry;
    ensure!(
        out.len() == g.rows * g.columns,
        "csf: output holds {} bytes, needs {}",
        out.len(),
        g.rows * g.columns
    );
    let bits = g.width.bits();
    let mask = (1u8 << bits) - 1;
    let records = plane.exceptions.len() / 2;
    let mut field = 0usize;
    let mut block = 0usize;
    let mut group = 0usize;
    let mut consumed = 0usize;
    let mut page_origin = 0usize;
    for tile_row in (0..g.row_blocks).step_by(TILE_ROW_BLOCKS) {
        for tile_col in (0..g.column_blocks).step_by(TILE_COLUMN_BLOCKS) {
            let page = group / PAGE_GROUPS;
            if group.is_multiple_of(PAGE_GROUPS) {
                let first = le_u16(plane.offsets, block + page);
                ensure!(
                    first == 0,
                    "csf: page {page} first prefix is {first}, not 0"
                );
                let origin = le_u32(plane.page_bases, page) as usize;
                ensure!(
                    origin == 2 * consumed,
                    "csf: page {page} origin {origin} != arena position {}",
                    2 * consumed
                );
                page_origin = consumed;
            }
            let pair_end = (tile_col + TILE_COLUMN_BLOCKS).min(g.column_blocks);
            let row_end = (tile_row + TILE_ROW_BLOCKS).min(g.row_blocks);
            for pair in tile_col..pair_end {
                let col0 = pair * BLOCK_COLUMNS;
                let width = (g.columns - col0).min(BLOCK_COLUMNS);
                for row_block in tile_row..row_end {
                    let row0 = row_block * BLOCK_ROWS;
                    for local_row in 0..BLOCK_ROWS {
                        let row = row0 + local_row;
                        let base = plane.bases[row];
                        let dst = row * g.columns + col0;
                        for slot in out[dst..dst + width].iter_mut() {
                            let bit = field * bits;
                            let value = (plane.codes[bit / 8] >> (bit % 8)) & mask;
                            *slot = base + value;
                            field += 1;
                        }
                    }
                    let begin = le_u16(plane.offsets, block + page) as usize;
                    let end = le_u16(plane.offsets, block + page + 1) as usize;
                    ensure!(
                        end >= begin && end - begin <= BLOCK_ROWS * width,
                        "csf: block {block} prefix {begin}..{end} exceeds its {} fields",
                        BLOCK_ROWS * width
                    );
                    ensure!(
                        page_origin + begin == consumed,
                        "csf: block {block} records start at {} not {consumed}",
                        page_origin + begin
                    );
                    ensure!(
                        consumed + (end - begin) <= records,
                        "csf: block {block} reads past the {records}-record arena"
                    );
                    let mut previous: Option<u8> = None;
                    for record in consumed..consumed + (end - begin) {
                        let packed = le_u16(plane.exceptions, record);
                        let position = (packed & 0xFF) as u8;
                        let value = (packed >> 8) as u8;
                        let local_row = (position / 2) as usize;
                        let local_col = (position % 2) as usize;
                        ensure!(
                            local_row < BLOCK_ROWS && local_col < width,
                            "csf: block {block} record {record} position {position} out of bounds"
                        );
                        ensure!(
                            previous.is_none_or(|p| position > p),
                            "csf: block {block} positions are not strictly increasing"
                        );
                        previous = Some(position);
                        out[(row0 + local_row) * g.columns + col0 + local_col] = value;
                    }
                    consumed += end - begin;
                    block += 1;
                }
            }
            group += 1;
        }
    }
    debug_assert_eq!(field * bits, g.codes_len() * 8);
    ensure!(
        consumed == records,
        "csf: {records} records stored, {consumed} consumed"
    );
    Ok(())
}

/// 2026-10-17: Allocating convenience wrapper around [`decode_csf_32x2_v1`].
pub fn decode_csf_32x2_v1_to_vec(plane: &CsfPlaneRef<'_>) -> Result<Vec<u8>> {
    let mut out = vec![0u8; plane.geometry.rows * plane.geometry.columns];
    decode_csf_32x2_v1(plane, &mut out)?;
    Ok(out)
}

#[cfg(test)]
#[path = "csf_blocks_tests.rs"]
mod tests;
