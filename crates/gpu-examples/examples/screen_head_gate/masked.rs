// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: The one-row masked pick of the screened head (the sharded device argmax decode,
//! `device_argmax.rs`): the three shards' screen key pairs, resolved by `argmax_key_resolve`,
//! must be the cell `argmax_bf16_batch_feed` writes over the full head's logits row under the
//! same row mask. The masks cycle through the cases the feed re-pick meets: the argmax masked
//! (the q key decides), the argmax and the runner-up masked (the third row decides), and a
//! random id masked (the p key decides).

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::ops::{self, SCREEN_STATE_BYTES};

use super::rows::Rng;
use super::{Rig, V};

const NO_ID: u32 = u32::MAX;

pub(crate) struct Masked {
    k_feed: KernelHandle,
    k_resolve: KernelHandle,
    /// 2026-10-21: `[m0, m1]` u32.
    masks: DevicePtr,
    /// 2026-10-21: The feed cell (u32), then the screen's cell.
    cells: DevicePtr,
    pub(crate) picks: usize,
    pub(crate) q_decided: usize,
    pub(crate) diffs: usize,
}

impl Masked {
    pub(crate) fn new(g: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            k_feed: g.kernel("argmax_feed", "argmax_bf16_batch_feed")?,
            k_resolve: g.kernel("argmax_key", "argmax_key_resolve")?,
            masks: g.alloc(8)?,
            cells: g.alloc(8)?,
            picks: 0,
            q_decided: 0,
            diffs: 0,
        })
    }

    /// 2026-10-21: One masked pick over the one row at `rig.x`, whose full-head logits
    /// `rig.reference(1)` left in `rig.logits` and whose plain argmax is `top`.
    pub(crate) fn check(&mut self, rig: &Rig, g: &mut Rng, top: u32, kappa: f32) -> Result<()> {
        let s = rig.g.default_stream();
        let mask = match self.picks % 3 {
            0 => [top, NO_ID],
            1 => [top, runner_up(rig, top)?],
            _ => [g.below(V) as u32, NO_ID],
        };
        let bytes: Vec<u8> = mask.iter().flat_map(|w| w.to_le_bytes()).collect();
        rig.g.copy_h2d(&bytes, self.masks)?;
        ops::argmax_bf16_batch_feed(
            rig.g,
            self.k_feed,
            rig.logits,
            self.masks,
            self.cells,
            V as u32,
            1,
            V as u32,
            s,
        )?;
        for sh in &rig.shards {
            ops::screen_head_keys(rig.g, &rig.k, sh, rig.x, 1, self.masks, kappa, s)?;
        }
        ops::argmax_key_resolve(
            rig.g,
            self.k_resolve,
            rig.states,
            rig.states.offset(SCREEN_STATE_BYTES),
            SCREEN_STATE_BYTES as u64,
            2,
            self.masks,
            self.cells.offset(4),
            s,
        )?;
        rig.g.synchronize(s)?;
        let got = rig.u32s(self.cells, 2);
        self.picks += 1;
        if mask.contains(&top) {
            self.q_decided += 1;
        }
        if got[0] != got[1] {
            self.diffs += 1;
            if self.diffs <= 3 {
                eprintln!(
                    "  masked pick under {mask:?}: screened {} != feed argmax {}",
                    got[1], got[0]
                );
            }
        }
        Ok(())
    }
}

/// 2026-10-21: The largest BF16 logit of `rig.logits` row 0 other than `top` (the lowest index
/// among equals; any id serves as a mask, the gate only wants a decisive one).
fn runner_up(rig: &Rig, top: u32) -> Result<u32> {
    let mut row = vec![0u8; V * 2];
    rig.g.copy_d2h(rig.logits, &mut row)?;
    let mut best = (f32::NEG_INFINITY, NO_ID);
    for (i, c) in row.chunks_exact(2).enumerate() {
        let v = half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32();
        if i as u32 != top && v > best.0 {
            best = (v, i as u32);
        }
    }
    Ok(best.1)
}
