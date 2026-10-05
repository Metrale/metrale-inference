// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: The layout type and its evaluation.
//!
//! Owner: metrale-layout.
//! Invariants: a digit's size is at least 1; a mode has at least one digit.

/// 2026-10-05: One mixed-radix digit: it takes `size` values, each step `stride` apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Digit {
    /// 2026-10-05: Values the digit takes.
    pub size: u64,
    /// 2026-10-05: Offset of one step.
    pub stride: i64,
}

/// 2026-10-05: A coordinate axis: its digits, fastest first (a row of a tile split into
/// in-warp and per-warp parts is one mode of two digits).
pub type Mode = Vec<Digit>;

/// 2026-10-05: Why a layout operation has no answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutError(pub String);

impl std::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for LayoutError {}

/// 2026-10-05: A layout: modes, the first fastest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    /// 2026-10-05: The modes.
    pub modes: Vec<Mode>,
}

impl Layout {
    /// 2026-10-05: A layout of one-digit modes from `(size, stride)` pairs.
    pub fn new(pairs: &[(u64, i64)]) -> Result<Self, LayoutError> {
        Self::from_modes(
            pairs
                .iter()
                .map(|&(size, stride)| vec![Digit { size, stride }])
                .collect(),
        )
    }

    /// 2026-10-05: A layout from modes, refusing empty modes and zero sizes.
    pub fn from_modes(modes: Vec<Mode>) -> Result<Self, LayoutError> {
        if modes
            .iter()
            .any(|m| m.is_empty() || m.iter().any(|d| d.size == 0))
        {
            return Err(LayoutError(
                "a mode has no digits or a digit has size 0".into(),
            ));
        }
        Ok(Layout { modes })
    }

    /// 2026-10-05: A row-major (last mode contiguous) or column-major (first mode contiguous)
    /// compact layout of `shape`.
    pub fn compact(shape: &[u64], row_major: bool) -> Result<Self, LayoutError> {
        let mut strides = vec![0i64; shape.len()];
        let mut s: i64 = 1;
        let order: Vec<usize> = if row_major {
            (0..shape.len()).rev().collect()
        } else {
            (0..shape.len()).collect()
        };
        for k in order {
            strides[k] = s;
            s = s
                .checked_mul(shape[k] as i64)
                .ok_or_else(|| LayoutError("compact layout overflows".into()))?;
        }
        Self::new(&shape.iter().copied().zip(strides).collect::<Vec<_>>())
    }

    /// 2026-10-05: The digits of every mode, in order.
    pub fn digits(&self) -> impl Iterator<Item = &Digit> {
        self.modes.iter().flatten()
    }

    /// 2026-10-05: The number of indices it maps.
    pub fn size(&self) -> u64 {
        self.digits().map(|d| d.size).product()
    }

    /// 2026-10-05: The size of mode `m`.
    pub fn mode_size(&self, m: usize) -> u64 {
        self.modes[m].iter().map(|d| d.size).product()
    }

    /// 2026-10-05: The offset of index `i` (taken modulo the size).
    pub fn eval(&self, mut i: u64) -> i64 {
        let mut off = 0i64;
        for d in self.digits() {
            off += (i % d.size) as i64 * d.stride;
            i /= d.size;
        }
        off
    }

    /// 2026-10-05: The offset of a per-mode coordinate.
    pub fn eval_coord(&self, coord: &[u64]) -> i64 {
        let mut i = 0u64;
        let mut scale = 1u64;
        for (m, &c) in self.modes.iter().zip(coord) {
            i += c * scale;
            scale *= m.iter().map(|d| d.size).product::<u64>();
        }
        self.eval(i)
    }

    /// 2026-10-05: One past the largest offset (the extent it addresses); 0 for a layout with a
    /// negative offset, which no kernel buffer uses.
    pub fn cosize(&self) -> u64 {
        let max: i64 = self
            .digits()
            .map(|d| (d.size as i64 - 1) * d.stride.max(0))
            .sum();
        let min: i64 = self
            .digits()
            .map(|d| (d.size as i64 - 1) * d.stride.min(0))
            .sum();
        if min < 0 { 0 } else { max as u64 + 1 }
    }

    /// 2026-10-05: Every offset, in index order.
    pub fn offsets(&self) -> Vec<i64> {
        (0..self.size()).map(|i| self.eval(i)).collect()
    }

    /// 2026-10-05: No two indices share an offset.
    pub fn is_injective(&self) -> bool {
        let mut o = self.offsets();
        o.sort_unstable();
        o.windows(2).all(|w| w[0] != w[1])
    }

    /// 2026-10-05: Injective onto exactly `[0, size)`.
    pub fn is_bijective(&self) -> bool {
        let mut o = self.offsets();
        o.sort_unstable();
        o.iter().enumerate().all(|(k, &v)| v == k as i64)
    }

    /// 2026-10-05: Both layouts are the same function (same size, same offsets).
    pub fn same_function(&self, other: &Layout) -> bool {
        self.size() == other.size() && (0..self.size()).all(|i| self.eval(i) == other.eval(i))
    }
}
