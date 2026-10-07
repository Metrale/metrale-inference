// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Checked token/slot coverage for bounded expert-weight reuse.
use anyhow::{Result, ensure};
pub(super) struct ExpertTokenPlan {
    words: Vec<u32>,
}
impl ExpertTokenPlan {
    pub(super) fn new(ids: &[u32]) -> Result<Self> {
        ensure!(
            (4..=64).contains(&ids.len()) && ids.len().is_multiple_of(4),
            "GPT expert plan requires1..16 top4 token rows"
        );
        let tokens = ids.len() / 4;
        let stride = tokens + 1;
        let mut words = vec![0; 32 * stride];
        for (token, row) in ids.chunks_exact(4).enumerate() {
            ensure!(
                row.iter().all(|&e| e < 32)
                    && row.iter().enumerate().all(|(i, e)| !row[..i].contains(e)),
                "GPT chunk invalid/duplicate experts"
            );
            for (slot, &expert) in row.iter().enumerate() {
                let at = expert as usize * stride;
                let count = words[at] as usize;
                ensure!(count < tokens, "GPT expert plan count overflow");
                words[at + 1 + count] = (token * 4 + slot) as u32;
                words[at] += 1;
            }
        }
        Self::validate(&words, ids)?;
        Ok(Self { words })
    }
    fn validate(words: &[u32], ids: &[u32]) -> Result<()> {
        let tokens = ids.len() / 4;
        let stride = tokens + 1;
        ensure!(words.len() == 32 * stride, "GPT expert plan size");
        let mut seen = vec![false; ids.len()];
        for expert in 0..32 {
            let count = words[expert * stride] as usize;
            ensure!(count <= tokens, "GPT expert plan count");
            for &entry in &words[expert * stride + 1..expert * stride + 1 + count] {
                let i = entry as usize;
                ensure!(
                    i < ids.len() && !seen[i] && ids[i] == expert as u32,
                    "GPT expert plan invalid/duplicate token slot"
                );
                seen[i] = true;
            }
        }
        ensure!(
            seen.iter().all(|&v| v),
            "GPT expert plan incomplete coverage"
        );
        Ok(())
    }
    pub(super) fn bytes(&self) -> Vec<u8> {
        self.words.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_experts_cover_every_permuted_token_slot_once() {
        for tokens in [1, 2, 3, 4, 5, 15, 16] {
            let ids: Vec<_> = (0..tokens)
                .flat_map(|t| [31, (t * 3) % 31, (t * 3 + 7) % 31, (t * 3 + 19) % 31])
                .collect();
            let plan = ExpertTokenPlan::new(&ids).unwrap();
            let stride = tokens as usize + 1;
            assert_eq!(plan.words[31 * stride], tokens);
            assert_eq!(plan.bytes().len(), 32 * stride * 4);
            let mut reversed = plan.words.clone();
            for expert in 0..32 {
                let at = expert * stride;
                let count = reversed[at] as usize;
                reversed[at + 1..at + 1 + count].reverse();
            }
            ExpertTokenPlan::validate(&reversed, &ids).unwrap();
            let mut overflow = plan.words.clone();
            overflow[31 * stride] = tokens + 1;
            assert!(ExpertTokenPlan::validate(&overflow, &ids).is_err());
            let mut omitted = plan.words.clone();
            omitted[31 * stride] = 0;
            assert!(ExpertTokenPlan::validate(&omitted, &ids).is_err());
            let mut duplicate = plan.words.clone();
            if tokens > 1 {
                duplicate[31 * stride + 2] = duplicate[31 * stride + 1];
                assert!(ExpertTokenPlan::validate(&duplicate, &ids).is_err());
            }
            let mut bad_entry = plan.words.clone();
            bad_entry[31 * stride + 1] = ids.len() as u32;
            assert!(ExpertTokenPlan::validate(&bad_entry, &ids).is_err());
            bad_entry[31 * stride + 1] = 1;
            assert!(ExpertTokenPlan::validate(&bad_entry, &ids).is_err());
        }
        for bad in [
            vec![],
            vec![0; 3],
            vec![0; 68],
            vec![0, 1, 2, 32],
            vec![0, 1, 2, u32::MAX],
            vec![0, 0, 1, 2],
        ] {
            assert!(ExpertTokenPlan::new(&bad).is_err());
        }
    }
}
