// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: The MTP drafter's shard head from the screen (`screen_head_local`): per shard and
//! activation row, the local argmax and its BF16 logit must be what the drafter's BF16 shard
//! sweep leaves for its exchange (`dense_gemv_bf16_batchm` over the shard's rows, then
//! `argmax_bf16_batch`): the same index and the same two logit bytes. The drafter keys its shard
//! at local indices (a shard view with base 0): argmax_bf16_batch breaks ties by the local index,
//! and a shard start that is not a multiple of 1,024 moves that order.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::ops::{self, ScreenShard};
use metrale_model_layers::weight_map::DenseWeight;

use super::{K, Rig};

pub(crate) struct Draft {
    k_local: KernelHandle,
    k_argmax_batch: KernelHandle,
    /// 2026-10-21: `[2, 4]` u32: the reference's indices, then the screen's.
    args: DevicePtr,
    pub(crate) rows: usize,
    pub(crate) diffs: usize,
}

impl Draft {
    pub(crate) fn new(g: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            k_local: g.kernel("screen_head", "screen_head_local")?,
            k_argmax_batch: g.kernel("argmax", "argmax_bf16_batch")?,
            args: g.alloc(32)?,
            rows: 0,
            diffs: 0,
        })
    }

    /// 2026-10-21: Every shard, the `m` rows at `rig.x`.
    pub(crate) fn check(&mut self, rig: &Rig, m: usize, kappa: f32) -> Result<()> {
        let s = rig.g.default_stream();
        for full in &rig.shards {
            // 2026-10-21: Keyed at local indices (base 0), as argmax_bf16_batch over the shard
            // breaks ties; the drafter's exchange adds its own shard start.
            let sh = &ScreenShard { base: 0, ..*full };
            let n = sh.rows;
            ops::dense_gemv_batchm(
                rig.g,
                rig.k_batchm,
                rig.x,
                &DenseWeight { weight: sh.w },
                rig.logits,
                m as u32,
                n,
                K as u32,
                n,
                s,
            )?;
            ops::argmax_bf16_batch(
                rig.g,
                self.k_argmax_batch,
                rig.logits,
                self.args,
                n,
                m as u32,
                n,
                s,
            )?;
            ops::screen_head_keys(
                rig.g,
                &rig.k,
                sh,
                rig.x,
                m as u32,
                DevicePtr::NULL,
                kappa,
                s,
            )?;
            ops::screen_head_local(
                rig.g,
                self.k_local,
                sh,
                rig.x,
                m as u32,
                self.args.offset(16),
                rig.probe,
                n,
                s,
            )?;
            rig.g.synchronize(s)?;
            let a = rig.u32s(self.args, 8);
            for r in 0..m {
                let (want, got) = (a[r] as usize, a[4 + r] as usize);
                let at = (r * n as usize + want) * 2;
                let mut lw = [0u8; 2];
                let mut lg = [0u8; 2];
                rig.g.copy_d2h(rig.logits.offset(at), &mut lw)?;
                rig.g
                    .copy_d2h(rig.probe.offset((r * n as usize + got) * 2), &mut lg)?;
                self.rows += 1;
                if want != got || lw != lg {
                    self.diffs += 1;
                    if self.diffs <= 3 {
                        eprintln!(
                            "  draft shard @{}: screen ({got}, {lg:?}) != BF16 sweep ({want}, {lw:?})",
                            full.base
                        );
                    }
                }
            }
        }
        Ok(())
    }
}
