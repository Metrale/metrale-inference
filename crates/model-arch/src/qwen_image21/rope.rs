// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Pinned Qwen Image three-axis geometry, no decoder-RoPE alias.
use anyhow::{Result, ensure};

/// Shared sequence geometry across samples; every row is frame/height/width.
/// Image grids use the reference's asymmetric centred range for odd dimensions.
pub struct ImageRopeLayout {
    positions: Vec<[i32; 3]>,
    image_spans: Vec<(usize, usize)>,
}
impl ImageRopeLayout {
    pub fn new(mask: &[bool], shapes: &[[usize; 3]]) -> Result<Self> {
        ensure!(!mask.is_empty(), "empty image rotary sequence");
        let mut positions = Vec::with_capacity(mask.len());
        let mut image_spans = Vec::with_capacity(shapes.len());
        let (mut cursor, mut position) = (0usize, 0i32);
        for &[frames, height, width] in shapes {
            ensure!(
                frames == 1 && height > 0 && width > 0,
                "image rotary supports one nonempty frame"
            );
            let start = mask[cursor..]
                .iter()
                .position(|v| *v)
                .map(|i| cursor + i)
                .ok_or_else(|| anyhow::anyhow!("missing image rotary block"))?;
            for _ in cursor..start {
                positions.push([position; 3]);
                position = position
                    .checked_add(1)
                    .ok_or_else(|| anyhow::anyhow!("rotary position overflow"))?;
            }
            let count = height
                .checked_mul(width)
                .ok_or_else(|| anyhow::anyhow!("image grid overflow"))?;
            let end = start
                .checked_add(count)
                .ok_or_else(|| anyhow::anyhow!("image span overflow"))?;
            ensure!(
                end <= mask.len() && mask[start..end].iter().all(|v| *v),
                "image rotary block contains text or exceeds sequence"
            );
            image_spans.push((start, end));
            let height = i32::try_from(height)?;
            let width = i32::try_from(width)?;
            for h in -(height - height / 2)..height / 2 {
                for w in -(width - width / 2)..width / 2 {
                    positions.push([position, h, w]);
                }
            }
            position = position
                .checked_add(height.max(width))
                .ok_or_else(|| anyhow::anyhow!("rotary position overflow"))?;
            cursor = end;
        }
        ensure!(
            mask[cursor..].iter().all(|v| !*v),
            "unaccounted image rotary tokens"
        );
        for _ in cursor..mask.len() {
            positions.push([position; 3]);
            position = position
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("rotary position overflow"))?;
        }
        ensure!(
            positions
                .iter()
                .flatten()
                .all(|p| (-1024..8192).contains(p)),
            "rotary position outside pinned frequency table"
        );
        Ok(Self {
            positions,
            image_spans,
        })
    }
    pub(super) fn matches_image_ids(&self, ids: &[i32]) -> bool {
        if ids.len() != self.positions.len() {
            return false;
        }
        let mut cursor = 0;
        for &(start, end) in &self.image_spans {
            if ids[cursor..start].iter().any(|id| *id != -1)
                || ids[start] < 0
                || ids[start..end].iter().any(|id| *id != ids[start])
            {
                return false;
            }
            if start > 0 && ids[start - 1] == ids[start]
                || end < ids.len() && ids[end] == ids[start]
            {
                return false;
            }
            cursor = end;
        }
        ids[cursor..].iter().all(|id| *id == -1)
    }
    pub fn positions(&self) -> &[[i32; 3]] {
        &self.positions
    }
    /// FP32 complex cis table `[sequence,64,2]`, real then imaginary. This native
    /// table's trigonometric rounding must be compared on the execution platform.
    pub fn frequencies(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.positions.len() * 128);
        for position in &self.positions {
            for (axis, dim) in [16usize, 56, 56].into_iter().enumerate() {
                for j in 0..dim / 2 {
                    let inverse = 1.0f32 / 10000.0f32.powf((2 * j) as f32 / dim as f32);
                    let (sin, cos) = (position[axis] as f32 * inverse).sin_cos();
                    result.extend([cos, sin]);
                }
            }
        }
        result
    }
}
