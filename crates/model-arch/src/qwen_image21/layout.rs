// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Joint VLM/image placement; no dense attention mask allocation.
use super::rope::ImageRopeLayout;
use anyhow::{Result, ensure};

/// Image slots expand fourfold, then receive projected image latents. Text
/// projections occupy the non-image slots; the last image is the target.
pub struct JointLayout {
    pub(super) samples: u32,
    pub(super) text_tokens: u32,
    pub(super) image_tokens: u32,
    pub(super) joint_tokens: u32,
    pub(super) target_tokens: u32,
    pub(super) gather: Vec<u32>,
    pub(super) target_gather: Vec<u32>,
    image_ids: Vec<i32>,
    target_mask: Vec<bool>,
    key_valid: Vec<bool>,
    rope: ImageRopeLayout,
}
impl JointLayout {
    pub fn new(
        samples: u32,
        text_tokens: u32,
        image_slots: &[bool],
        shapes: &[[usize; 3]],
        text_key_valid: &[bool],
    ) -> Result<Self> {
        ensure!(
            samples > 0 && text_tokens > 0 && !shapes.is_empty(),
            "empty visual input geometry"
        );
        let lengths: Vec<usize> = shapes
            .iter()
            .map(|s| {
                s.iter()
                    .try_fold(1usize, |n, d| n.checked_mul(*d))
                    .ok_or_else(|| anyhow::anyhow!("image shape overflow"))
            })
            .collect::<Result<_>>()?;
        let target = *lengths.last().unwrap();
        ensure!(
            target > 0 && target.is_multiple_of(4),
            "target image requires whole four-token slots"
        );
        ensure!(
            image_slots.len() == text_tokens as usize + target / 4
                && image_slots[text_tokens as usize..].iter().all(|x| *x),
            "target slots must be appended image slots"
        );
        let image_tokens = lengths
            .iter()
            .try_fold(0usize, |n, d| n.checked_add(*d))
            .ok_or_else(|| anyhow::anyhow!("image count overflow"))?;
        ensure!(
            image_slots.iter().filter(|v| **v).count().checked_mul(4) == Some(image_tokens),
            "image slots and latent shapes differ"
        );
        let text_rows = samples
            .checked_mul(text_tokens)
            .ok_or_else(|| anyhow::anyhow!("text row overflow"))?;
        ensure!(
            text_key_valid.len() == text_rows as usize,
            "text key validity shape differs"
        );
        let expanded: Vec<bool> = image_slots
            .iter()
            .flat_map(|v| std::iter::repeat_n(*v, if *v { 4 } else { 1 }))
            .collect();
        let joint_tokens = u32::try_from(expanded.len())?;
        let image_tokens = u32::try_from(image_tokens)?;
        samples
            .checked_mul(
                text_tokens
                    .checked_add(image_tokens)
                    .ok_or_else(|| anyhow::anyhow!("projection row overflow"))?,
            )
            .ok_or_else(|| anyhow::anyhow!("batch projection overflow"))?;
        samples
            .checked_mul(joint_tokens)
            .ok_or_else(|| anyhow::anyhow!("joint row overflow"))?;
        let rope = ImageRopeLayout::new(&expanded, shapes)?;
        let mut image_ids = vec![-1; expanded.len()];
        let mut image_positions = expanded
            .iter()
            .enumerate()
            .filter_map(|(i, v)| v.then_some(i));
        for (id, len) in lengths.iter().enumerate() {
            for _ in 0..*len {
                image_ids[image_positions
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("missing image position"))?] =
                    i32::try_from(id)?;
            }
        }
        let target_id = i32::try_from(shapes.len() - 1)?;
        let target_mask: Vec<bool> = image_ids.iter().map(|id| *id == target_id).collect();
        ensure!(
            target_mask[..target_mask.len() - target]
                .iter()
                .all(|v| !*v)
                && target_mask[target_mask.len() - target..].iter().all(|v| *v),
            "target tokens must be trailing"
        );
        let mut gather = Vec::new();
        let mut key_valid = Vec::new();
        let mut target_gather = Vec::new();
        for sample in 0..samples {
            let mut image = 0;
            for (text, slot) in image_slots.iter().enumerate() {
                if *slot {
                    for _ in 0..4 {
                        gather.push(text_rows + sample * image_tokens + image);
                        key_valid.push(true);
                        image += 1;
                    }
                } else {
                    gather.push(sample * text_tokens + text as u32);
                    key_valid.push(text_key_valid[sample as usize * text_tokens as usize + text]);
                }
            }
            for target in joint_tokens - target as u32..joint_tokens {
                target_gather.push(sample * joint_tokens + target);
            }
        }
        Ok(Self {
            samples,
            text_tokens,
            image_tokens,
            joint_tokens,
            target_tokens: target as u32,
            gather,
            target_gather,
            image_ids,
            target_mask,
            key_valid,
            rope,
        })
    }
    pub fn image_ids(&self) -> &[i32] {
        &self.image_ids
    }
    pub fn target_mask(&self) -> &[bool] {
        &self.target_mask
    }
    pub fn key_valid(&self) -> &[bool] {
        &self.key_valid
    }
    pub fn rope(&self) -> &ImageRopeLayout {
        &self.rope
    }
    pub fn geometry(&self) -> (u32, u32, u32, u32, u32) {
        (
            self.samples,
            self.text_tokens,
            self.image_tokens,
            self.joint_tokens,
            self.target_tokens,
        )
    }
    pub fn projection_gather(&self) -> &[u32] {
        &self.gather
    }
    pub fn target_gather(&self) -> &[u32] {
        &self.target_gather
    }
}
