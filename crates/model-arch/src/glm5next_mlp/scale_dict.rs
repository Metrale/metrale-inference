// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: `METRALE_GLM_MOE_SCALE_DICT`: a routed expert tensor's E4M3 block scales stored as
//! 5-bit offsets from one per-tensor base code, the layout the `_cs` union sweeps read
//! (`w4a4_mx_core.cuh` CS). GLM-5.3-Flash's expert tensors use 23..=52 distinct scale codes and
//! almost all of them span at most 32 consecutive codes, so a k128 chunk's 8 scales of a row take
//! 5 bytes instead of 8: 4.5 -> 4.3125 bits per weight. A tensor whose codes span more than 32
//! keeps its plain scales ([`compact`] returns `None`) and the sweep reads it as before.
//!
//! The kernel rebuilds each byte as `base + offset`, the stored code itself, so every MMA scale
//! word, and every output byte, is the plain path's.
//!
//! Layout (`rows` rows, `num_c = ceil(groups / 8)` chunks per row): a lo plane of one
//! little-endian `u32` per (row, chunk), then a hi plane of one byte per (row, chunk). Word byte
//! `b` holds the low 4 offset bits of group `S0[b]` (low nibble) and `S1[b]` (high nibble); hi bit
//! `b` and bit `4 + b` are their fifth bits. A group past `groups` (a K % 128 == 64 tail) is
//! offset 0; the kernel zeroes those scale bytes. The hi plane is zero-padded to whole 4-byte
//! words: the grouped prefill kernels copy the aligned word that holds a (row, chunk)'s byte.
//!
//! The loader decides once per routed layer ([`plan`]) whether its experts' scales are packed:
//! only under the `tp` expert layout (whose slices pass through the host; the whole layout binds
//! the store's tensors in place), only when every row count of the layer runs W4A4 (the W4A16
//! paths read plain scales) and only when every `_cs` twin resolved (`scale_dict_kernels`). A
//! packed tensor replaces its plain bytes in the same slab slot ([`pack`]): one copy.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: [`compact`] is exact: [`expand`] of its result is the input.

use super::weights::ScaleLayout;

/// 2026-10-21: The widest code span one tensor may have (a 5-bit offset).
pub const SCALE_DICT_SPAN: u8 = 32;
/// 2026-10-21: Groups of a chunk in the order of the MMA's two scale words
/// (`__byte_perm(.., 0x6240)` and `0x7351` in `w4a4_mx_core.cuh`).
const S0: [usize; 4] = [0, 4, 2, 6];
const S1: [usize; 4] = [1, 5, 3, 7];

/// 2026-10-21: The per-expert base table's entry for an expert whose scales are plain.
pub const PLAIN_BASE4: u32 = 0xFFFF_FFFF;

/// 2026-10-21: One tensor's compact scales and its base code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactScales {
    pub bytes: Vec<u8>,
    pub base: u8,
}

impl CompactScales {
    /// 2026-10-21: The base code in every byte, as the `_cs` sweeps' per-expert table holds it.
    pub fn base4(&self) -> u32 {
        u32::from(self.base) * 0x0101_0101
    }
}

/// 2026-10-21: The compact bytes of a `[rows, groups]` E4M3 scale tensor, or `None` when its
/// codes span more than [`SCALE_DICT_SPAN`] (or `groups` is not a multiple of 4: a tensor the
/// sweeps never read).
pub fn compact(scales: &[u8], rows: usize, groups: usize) -> Option<CompactScales> {
    if scales.len() != rows * groups || !groups.is_multiple_of(4) || scales.is_empty() {
        return None;
    }
    let (lo, hi) = scales
        .iter()
        .fold((u8::MAX, 0u8), |(a, b), &c| (a.min(c), b.max(c)));
    if hi - lo >= SCALE_DICT_SPAN || hi > 0x7E {
        return None;
    }
    let num_c = groups.div_ceil(8);
    let mut words = Vec::with_capacity(rows * num_c * 4);
    let mut fifth = Vec::with_capacity(rows * num_c);
    for r in 0..rows {
        let row = &scales[r * groups..(r + 1) * groups];
        for c in 0..num_c {
            let off = |g: usize| row.get(c * 8 + g).map_or(0, |&s| s - lo);
            let (mut w, mut h) = (0u32, 0u8);
            for b in 0..4 {
                let (x, y) = (off(S0[b]), off(S1[b]));
                w |= u32::from((x & 0xF) | ((y & 0xF) << 4)) << (8 * b);
                h |= ((x >> 4) << b) | ((y >> 4) << (4 + b));
            }
            words.extend_from_slice(&w.to_le_bytes());
            fifth.push(h);
        }
    }
    words.extend_from_slice(&fifth);
    words.resize(words.len().next_multiple_of(4), 0);
    Some(CompactScales {
        bytes: words,
        base: lo,
    })
}

/// 2026-10-21: The `[rows, groups]` codes [`compact`] encoded (its inverse).
pub fn expand(c: &CompactScales, rows: usize, groups: usize) -> Vec<u8> {
    let num_c = groups.div_ceil(8);
    let (words, fifth) = c.bytes.split_at(rows * num_c * 4);
    let mut out = vec![0u8; rows * groups];
    for r in 0..rows {
        for ch in 0..num_c {
            let i = r * num_c + ch;
            let w = u32::from_le_bytes(words[4 * i..4 * i + 4].try_into().expect("4 bytes"));
            let h = fifth[i];
            for b in 0..4 {
                let byte = (w >> (8 * b)) as u8;
                let x = (byte & 0xF) | (((h >> b) & 1) << 4);
                let y = (byte >> 4) | (((h >> (4 + b)) & 1) << 4);
                for (g, v) in [(S0[b], x), (S1[b], y)] {
                    if ch * 8 + g < groups {
                        out[r * groups + ch * 8 + g] = c.base + v;
                    }
                }
            }
        }
    }
    out
}

/// 2026-10-21: The bytes a projection's scale buffer holds and their layout: the compact bytes
/// zero-padded to the plain length (a slab slot keeps one size for every expert), or `scales`
/// itself when [`compact`] refuses the tensor.
pub fn pack(scales: Vec<u8>, rows: usize, groups: usize) -> (Vec<u8>, ScaleLayout) {
    match compact(&scales, rows, groups) {
        Some(c) => {
            let mut b = c.bytes;
            b.resize(scales.len(), 0);
            (b, ScaleLayout::Compact { base: c.base })
        }
        None => (scales, ScaleLayout::Plain),
    }
}

/// 2026-10-21: A routed layer's packing decision, made once at load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleDictPlan {
    /// 2026-10-21: Every tensor whose codes fit is packed.
    Compact,
    /// 2026-10-21: `METRALE_GLM_MOE_SCALE_DICT=0`.
    Off,
    /// 2026-10-21: The whole (ep) expert layout binds the store's scale tensors in place.
    WholeLayout,
    /// 2026-10-21: Some row count runs the W4A16 path, which reads plain scales.
    NotW4a4EveryRow,
    /// 2026-10-21: A `_cs` twin of a routed W4A4 entry is absent from the target's PTX.
    NoTwins,
}

/// 2026-10-21: The decision from the lever, the layout (`sliced`: `tp`), whether the layer's
/// precision plan runs W4A4 at every row count, and whether every `_cs` twin resolved.
pub fn plan(lever: bool, sliced: bool, w4a4_every_row: bool, twins_ready: bool) -> ScaleDictPlan {
    if !lever {
        ScaleDictPlan::Off
    } else if !sliced {
        ScaleDictPlan::WholeLayout
    } else if !w4a4_every_row {
        ScaleDictPlan::NotW4a4EveryRow
    } else if !twins_ready {
        ScaleDictPlan::NoTwins
    } else {
        ScaleDictPlan::Compact
    }
}

impl ScaleDictPlan {
    /// 2026-10-21: For the load log.
    pub fn describe(self) -> &'static str {
        match self {
            Self::Compact => "compact scales where a tensor's codes span at most 32",
            Self::Off => "off (METRALE_GLM_MOE_SCALE_DICT=0): plain scales",
            Self::WholeLayout => {
                "plain scales: the ep expert layout binds the checkpoint's tensors"
            }
            Self::NotW4a4EveryRow => "plain scales: some row count runs W4A16, which reads them",
            Self::NoTwins => "plain scales: a `_cs` entry is absent from this target's PTX",
        }
    }
}

#[cfg(test)]
#[path = "scale_dict_tests.rs"]
mod tests;
