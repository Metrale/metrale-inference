// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: Bounded host-compatible greedy sampling after per-row token masking.
//! Owner: model-engine. Invariant: called after forward, before scratch is reused.

use super::super::types::TransformerModel;
use anyhow::Result;
use metrale_gpu_runtime::{gpu::DevicePtr, kernel_args::KernelLaunch};

// 2026-10-07: Mask IDs per row; must equal the `masks[row * 8 + j]` stride of
// `argmax_bf16_batch_masked_host` in kernels/gb10/common/argmax_feed.cu.
const MASK_CAPACITY: usize = 8;

fn pack_masks(masks: &[Vec<u32>], vocab: usize) -> Option<Vec<u8>> {
    let mut bytes = Vec::with_capacity(masks.len().checked_mul(MASK_CAPACITY * 4)?);
    for row in masks {
        let mut ids = [u32::MAX; MASK_CAPACITY];
        let mut used = 0;
        for &id in row {
            if id as usize >= vocab || ids[..used].contains(&id) {
                continue;
            }
            if used == MASK_CAPACITY {
                return None;
            }
            ids[used] = id;
            used += 1;
        }
        bytes.extend(ids.into_iter().flat_map(u32::to_le_bytes));
    }
    Some(bytes)
}

impl TransformerModel {
    pub(super) fn argmax_batch_masked_dispatch(
        &self,
        logits: DevicePtr,
        n: usize,
        masks: &[Vec<u32>],
        _stream: u64,
    ) -> Result<Option<Vec<u32>>> {
        // 2026-10-07: decode_batch uses the model default stream despite the
        // caller's conventional zero; keep scratch reuse ordered after that forward.
        let stream = self.gpu.default_stream();
        let vocab = self.config.vocab_size;
        let Some(bytes) = n.checked_mul(4 * (MASK_CAPACITY + 1)) else {
            return Ok(None);
        };
        let Some(logit_bytes) = n.checked_mul(vocab).and_then(|x| x.checked_mul(2)) else {
            return Ok(None);
        };
        if n == 0
            || n > u32::MAX as usize
            || masks.len() != n
            || vocab == 0
            || vocab >= u32::MAX as usize
            || logits.0 == 0
            || self.use_fp32_logits
            || bytes > self.buffers.scratch_bytes()
            || logit_bytes > self.buffers.logits_bytes()
        {
            return Ok(None);
        }
        let Some(packed) = pack_masks(masks, vocab) else {
            return Ok(None);
        };
        let gpu = self.gpu.as_ref();
        let Ok(kernel) = gpu
            .op_cache()
            .kernel(gpu, "argmax_feed", "argmax_bf16_batch_masked_host")
        else {
            return Ok(None);
        };
        let output = self.buffers.scratch();
        let mask_ptr = output.offset(n * 4);
        gpu.copy_h2d_async(&packed, mask_ptr, stream)?;
        KernelLaunch::new(gpu, kernel)
            .grid([n as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(logits)
            .arg_ptr(mask_ptr)
            .arg_ptr(output)
            .arg_u32(vocab as u32)
            .arg_u32(vocab as u32)
            .launch(stream)?;
        let mut host = vec![0u8; n * 4];
        gpu.copy_d2h_on_stream(output, &mut host, stream)?;
        let ids: Vec<u32> = host
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        if ids.iter().any(|&id| id as usize >= vocab) {
            return Ok(None);
        }
        Ok(Some(ids))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn masks_deduplicate_ignore_out_of_range_and_refuse_capacity() {
        let packed = pack_masks(&[vec![2, 2, 99, 4], vec![]], 10).unwrap();
        let ids: Vec<_> = packed
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(&ids[..3], &[2, 4, u32::MAX]);
        assert!(ids[2..].iter().all(|&id| id == u32::MAX));
        assert!(pack_masks(&[(0..9).collect()], 10).is_none());
        assert!(pack_masks(&[(0..9).collect()], 8).is_some());
    }
}
