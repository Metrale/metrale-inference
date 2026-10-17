// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Spec-derived reference encoder for the `csf_32x2_v1` tests.
//!
//! Owner: metrale-core (tests only).
//!
//! Built independently of the decoder's nested tile walk: it materialises
//! the block visiting order as an explicit permutation, packs fields per
//! block, and derives the page index from per-group block counts and a
//! prefix sum, mirroring the spec text step by step.

use super::super::{
    BLOCK_COLUMNS, BLOCK_ROWS, CsfGeometry, CsfOffsetWidth, CsfPlaneRef, PAGE_GROUPS,
};

/// 2026-10-17: Owned component bytes, little-endian as on disk.
#[derive(Clone, Debug)]
pub struct Encoded {
    pub geometry: CsfGeometry,
    pub bases: Vec<u8>,
    pub codes: Vec<u8>,
    pub page_bases: Vec<u8>,
    pub offsets: Vec<u8>,
    pub exceptions: Vec<u8>,
}

impl Encoded {
    pub fn plane(&self) -> CsfPlaneRef<'_> {
        CsfPlaneRef {
            geometry: self.geometry,
            bases: &self.bases,
            codes: &self.codes,
            page_bases: &self.page_bases,
            offsets: &self.offsets,
            exceptions: &self.exceptions,
        }
    }

    pub fn records(&self) -> usize {
        self.exceptions.len() / 2
    }
}

/// 2026-10-17: Block ids (`row_block * column_blocks + column_block`) in
/// visiting order, plus the number of blocks in each tile group.
pub fn block_order(row_blocks: usize, column_blocks: usize) -> (Vec<usize>, Vec<usize>) {
    let mut order = Vec::new();
    let mut group_sizes = Vec::new();
    let mut r = 0;
    while r < row_blocks {
        let mut c = 0;
        while c < column_blocks {
            let before = order.len();
            let pairs: Vec<usize> = (c..column_blocks).take(2).collect();
            let rows: Vec<usize> = (r..row_blocks).take(4).collect();
            for &cb in &pairs {
                for &rb in &rows {
                    order.push(rb * column_blocks + cb);
                }
            }
            group_sizes.push(order.len() - before);
            c += 2;
        }
        r += 4;
    }
    (order, group_sizes)
}

/// 2026-10-17: Row base = row minimum clamped to the legal palette range.
pub fn min_bases(logical: &[u8], rows: usize, columns: usize, width: CsfOffsetWidth) -> Vec<u8> {
    (0..rows)
        .map(|r| {
            let min = *logical[r * columns..(r + 1) * columns]
                .iter()
                .min()
                .unwrap();
            min.min(width.max_base())
        })
        .collect()
}

/// 2026-10-17: Encode a row-major logical scale matrix with the given row
/// bases; any byte outside `[base, base + 2^bits)` becomes an exception.
pub fn encode(
    logical: &[u8],
    rows: usize,
    columns: usize,
    width: CsfOffsetWidth,
    bases: &[u8],
) -> Encoded {
    let geometry = CsfGeometry::new(rows, columns, width).unwrap();
    assert_eq!(logical.len(), rows * columns);
    assert_eq!(bases.len(), rows);
    let bits = width.bits();
    let span = 1usize << bits;
    let (order, group_sizes) = block_order(geometry.row_blocks, geometry.column_blocks);
    let mut fields: Vec<u8> = Vec::new();
    let mut per_block_records: Vec<Vec<u16>> = Vec::new();
    for &id in &order {
        let rb = id / geometry.column_blocks;
        let cb = id % geometry.column_blocks;
        let w = (columns - cb * BLOCK_COLUMNS).min(BLOCK_COLUMNS);
        let mut records = Vec::new();
        for lr in 0..BLOCK_ROWS {
            let row = rb * BLOCK_ROWS + lr;
            for lc in 0..w {
                let value = logical[row * columns + cb * BLOCK_COLUMNS + lc];
                let delta = value as isize - bases[row] as isize;
                if (0..span as isize).contains(&delta) {
                    fields.push(delta as u8);
                } else {
                    fields.push(0);
                    records.push(((value as u16) << 8) | (2 * lr + lc) as u16);
                }
            }
        }
        per_block_records.push(records);
    }
    let per_byte = 8 / bits;
    let codes: Vec<u8> = fields
        .chunks(per_byte)
        .map(|chunk| {
            chunk
                .iter()
                .enumerate()
                .fold(0u8, |acc, (i, &f)| acc | (f << (i * bits)))
        })
        .collect();
    let mut prefix = vec![0usize];
    for records in &per_block_records {
        prefix.push(prefix.last().unwrap() + records.len());
    }
    let mut group_starts = vec![0usize];
    for size in &group_sizes {
        group_starts.push(group_starts.last().unwrap() + size);
    }
    let mut page_blocks: Vec<usize> = group_starts[..group_sizes.len()]
        .iter()
        .step_by(PAGE_GROUPS)
        .copied()
        .collect();
    page_blocks.push(*group_starts.last().unwrap());
    let mut page_bases = Vec::new();
    let mut offsets = Vec::new();
    for window in page_blocks.windows(2) {
        let (start, stop) = (window[0], window[1]);
        page_bases.extend_from_slice(&((2 * prefix[start]) as u32).to_le_bytes());
        for &p in &prefix[start..=stop] {
            offsets.extend_from_slice(&((p - prefix[start]) as u16).to_le_bytes());
        }
    }
    let exceptions = per_block_records
        .iter()
        .flatten()
        .flat_map(|r| r.to_le_bytes())
        .collect();
    Encoded {
        geometry,
        bases: bases.to_vec(),
        codes,
        page_bases,
        offsets,
        exceptions,
    }
}

/// 2026-10-17: Deterministic xorshift64* byte source.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u8(&mut self) -> u8 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 56) as u8
    }
}

/// 2026-10-17: Rows whose bytes mostly sit in a 16-wide window above a
/// per-row anchor, with about one in eight anywhere in `0..=0x7E`.
pub fn scale_like(rows: usize, columns: usize, rng: &mut Rng) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows * columns);
    for _ in 0..rows {
        let anchor = rng.next_u8() % 100;
        for _ in 0..columns {
            let pick = rng.next_u8();
            out.push(if pick < 32 {
                rng.next_u8() % 0x7F
            } else {
                anchor + rng.next_u8() % 16
            });
        }
    }
    out
}
