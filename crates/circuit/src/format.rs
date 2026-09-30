// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: Tensor formats of edges and weights, and their one text spelling.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - `Format::parse(f.name()) == Ok(f)` for every format: the spelling is canonical, so the
//!   plan digest and the golden files see one text per format.
//! - Edge formats carry activation scales only (`token`, `tensor`, `g<n>`); `channel` and
//!   `block` are weight layouts, refused on an edge by the circuit loader.

use std::fmt;

/// 2026-09-28: How the scales of an 8-bit float tensor are shared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Scale {
    /// 2026-09-28: One scale for the whole tensor.
    PerTensor,
    /// 2026-09-28: One scale per row (activations).
    PerToken,
    /// 2026-09-28: One scale per output channel (weights).
    PerChannel,
    /// 2026-09-28: One scale per `n` consecutive values along K.
    Group(u32),
    /// 2026-09-28: One scale per `[rows, cols]` block (weights).
    Block(u32, u32),
}

/// 2026-09-28: A tensor format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Format {
    /// 2026-09-28: bfloat16.
    Bf16,
    /// 2026-09-28: float32.
    F32,
    /// 2026-09-28: int32 (token and expert ids).
    I32,
    /// 2026-09-28: FP8 E4M3 with the given scale layout.
    Fp8E4m3 {
        /// 2026-09-28: Scale sharing.
        scale: Scale,
    },
    /// 2026-09-28: NVFP4: E2M1 values, one E4M3 scale per `group` values, one F32 global.
    Nvfp4 {
        /// 2026-09-28: Values per scale.
        group: u32,
    },
}

/// 2026-09-28: A format string that is not one of the spellings [`Format::parse`] accepts.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "unknown format `{0}` (expected bf16, f32, i32, fp8/<token|tensor|channel|g<n>|block<r>x<c>> or nvfp4/g<n>)"
)]
pub struct FormatError(pub String);

impl Format {
    /// 2026-09-28: Parse the canonical spelling, e.g. `bf16`, `fp8/token`, `fp8/block128x128`,
    /// `nvfp4/g16`.
    pub fn parse(s: &str) -> Result<Self, FormatError> {
        let bad = || FormatError(s.to_string());
        match s {
            "bf16" => return Ok(Format::Bf16),
            "f32" => return Ok(Format::F32),
            "i32" => return Ok(Format::I32),
            _ => {}
        }
        let (head, tail) = s.split_once('/').ok_or_else(bad)?;
        match head {
            "fp8" => Ok(Format::Fp8E4m3 {
                scale: parse_scale(tail).ok_or_else(bad)?,
            }),
            "nvfp4" => {
                let group = tail.strip_prefix('g').and_then(positive).ok_or_else(bad)?;
                Ok(Format::Nvfp4 { group })
            }
            _ => Err(bad()),
        }
    }

    /// 2026-09-28: The canonical spelling.
    pub fn name(&self) -> String {
        match self {
            Format::Bf16 => "bf16".into(),
            Format::F32 => "f32".into(),
            Format::I32 => "i32".into(),
            Format::Fp8E4m3 { scale } => format!("fp8/{}", scale_name(*scale)),
            Format::Nvfp4 { group } => format!("nvfp4/g{group}"),
        }
    }

    /// 2026-09-28: True for an unquantized format (16/32-bit floats and int32).
    pub fn is_plain(&self) -> bool {
        matches!(self, Format::Bf16 | Format::F32 | Format::I32)
    }

    /// 2026-09-28: True when the scale layout is one an activation edge can carry.
    pub fn is_edge_format(&self) -> bool {
        match self {
            Format::Bf16 | Format::F32 | Format::I32 | Format::Nvfp4 { .. } => true,
            Format::Fp8E4m3 { scale } => {
                matches!(scale, Scale::PerTensor | Scale::PerToken | Scale::Group(_))
            }
        }
    }

    /// 2026-09-30: Bytes of a `[n, k]` weight in this format, scales included: as
    /// [`Format::bytes`], except that an NVFP4 weight has one F32 global scale for the whole
    /// tensor (`weight_scale_2`), where an activation has one per row.
    pub fn weight_bytes(&self, n: u64, k: u64) -> Option<u64> {
        match self {
            Format::Nvfp4 { .. } => self
                .bytes(n, k)?
                .checked_sub(n.checked_mul(4)?)?
                .checked_add(4),
            _ => self.bytes(n, k),
        }
    }

    /// 2026-09-28: Bytes of a `rows x dim` tensor in this format, scales included. Used by
    /// the buffer planner. `None` when `dim` is not a multiple of the scale group.
    pub fn bytes(&self, rows: u64, dim: u64) -> Option<u64> {
        let elems = rows.checked_mul(dim)?;
        match self {
            Format::Bf16 => elems.checked_mul(2),
            Format::F32 | Format::I32 => elems.checked_mul(4),
            Format::Fp8E4m3 { scale } => {
                let scales = match scale {
                    Scale::PerTensor => 1,
                    Scale::PerToken => rows,
                    Scale::PerChannel => dim,
                    Scale::Group(g) => {
                        let g = u64::from(*g);
                        if !dim.is_multiple_of(g) {
                            return None;
                        }
                        rows.checked_mul(dim / g)?
                    }
                    Scale::Block(r, c) => {
                        let (r, c) = (u64::from(*r), u64::from(*c));
                        rows.div_ceil(r).checked_mul(dim.div_ceil(c))?
                    }
                };
                elems.checked_add(scales.checked_mul(4)?)
            }
            Format::Nvfp4 { group } => {
                let g = u64::from(*group);
                if !dim.is_multiple_of(g) || !elems.is_multiple_of(2) {
                    return None;
                }
                // 2026-09-28: Packed E2M1 pairs, one E4M3 byte per group. 2026-09-30: One F32
                // global per row, as the activation quantizer writes it (`w4a4_quant_rows`);
                // an edge is always an activation.
                (elems / 2)
                    .checked_add(elems / g)?
                    .checked_add(rows.checked_mul(4)?)
            }
        }
    }
}

impl fmt::Display for Format {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name())
    }
}

fn positive(s: &str) -> Option<u32> {
    s.parse::<u32>().ok().filter(|&n| n > 0)
}

fn parse_scale(s: &str) -> Option<Scale> {
    match s {
        "tensor" => Some(Scale::PerTensor),
        "token" => Some(Scale::PerToken),
        "channel" => Some(Scale::PerChannel),
        _ => {
            if let Some(g) = s.strip_prefix('g') {
                return positive(g).map(Scale::Group);
            }
            let (r, c) = s.strip_prefix("block")?.split_once('x')?;
            Some(Scale::Block(positive(r)?, positive(c)?))
        }
    }
}

fn scale_name(s: Scale) -> String {
    match s {
        Scale::PerTensor => "tensor".into(),
        Scale::PerToken => "token".into(),
        Scale::PerChannel => "channel".into(),
        Scale::Group(g) => format!("g{g}"),
        Scale::Block(r, c) => format!("block{r}x{c}"),
    }
}

#[cfg(test)]
#[path = "format_tests.rs"]
mod format_tests;
