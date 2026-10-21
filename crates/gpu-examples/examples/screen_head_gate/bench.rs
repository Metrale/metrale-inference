// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: The `bench` mode of the screen gate: the old shard head against the screen at
//! M = 1 and 2, one rank's shard.

use std::time::Instant;

use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_layers::layers::ops::{self, SCREEN_STATE_COUNTS};
use metrale_model_layers::weight_map::DenseWeight;

use super::rows::{self, Family, Rng};
use super::{K, Rig, bytes_of};

pub(crate) fn bench(rig: &Rig, head: &[bf16], gamma: &[f32]) -> Result<()> {
    let s = rig.g.default_stream();
    let sh = &rig.shards[0];
    let n = sh.rows as usize;
    let mut g = Rng(99);
    let wt = DenseWeight { weight: sh.w };
    let iters = 300;
    // 2026-10-21: The MTP drafter's head today: an FP8 copy of the same shard (per-row scale),
    // one row by dense_gemv_fp8w, rows batched by fp8_gemv_rowscale_batch16_rt2.
    let fp8 = metrale_model_layers::weight_map::quantize_to_fp8(
        &wt,
        n,
        K,
        rig.g,
        rig.g.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?,
        s,
    )?;
    let k_fp8 = rig.g.kernel("gemv_fp8w", "dense_gemv_fp8w")?;
    let k_fp8b = rig
        .g
        .kernel("fp8_gemv_rt", "fp8_gemv_rowscale_batch16_rt2")?;
    let k_amb = rig.g.kernel("argmax", "argmax_bf16_batch")?;
    for m in [1usize, 2, 4] {
        for fam in [Family::Peaked, Family::Iso] {
            let xs: Vec<f32> = (0..m)
                .flat_map(|_| rows::realistic(fam, &mut g, head, K, gamma))
                .collect();
            rig.g.copy_h2d(&bytes_of(&xs), rig.x)?;
            let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
                for _ in 0..20 {
                    f()?;
                }
                rig.g.synchronize(s)?;
                let t = Instant::now();
                for _ in 0..iters {
                    f()?;
                }
                rig.g.synchronize(s)?;
                Ok(t.elapsed().as_secs_f64() * 1e6 / iters as f64)
            };
            let old = time(&|| {
                if m == 1 {
                    ops::dense_gemv(
                        rig.g, rig.k_gemv, rig.x, &wt, rig.logits, n as u32, K as u32, s,
                    )
                } else {
                    ops::dense_gemv_batchm(
                        rig.g,
                        rig.k_batchm,
                        rig.x,
                        &wt,
                        rig.logits,
                        m as u32,
                        n as u32,
                        K as u32,
                        n as u32,
                        s,
                    )
                }
            })?;
            let kappa = ops::screen_kappa(K as u32);
            let new = time(&|| {
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
                ops::screen_head_resolve(
                    rig.g,
                    rig.k.resolve,
                    sh.state,
                    DevicePtr::NULL,
                    0,
                    0,
                    m as u32,
                    rig.out,
                    s,
                )
            })?;
            let cand = rig.u32s(sh.state.offset(SCREEN_STATE_COUNTS), m);
            let drafter = time(&|| {
                if m == 1 {
                    ops::dense_gemv_fp8w(
                        rig.g, k_fp8, rig.x, &fp8, rig.logits, n as u32, K as u32, s,
                    )?;
                } else {
                    ops::fp8_gemv_rowscale_batch16_rt2(
                        rig.g, k_fp8b, rig.x, &fp8, rig.logits, m as u32, n as u32, K as u32, s,
                    )?;
                }
                ops::argmax_bf16_batch(
                    rig.g, k_amb, rig.logits, rig.out, n as u32, m as u32, n as u32, s,
                )
            })?;
            let ob = (n * K * 2) as f64;
            let nb = (n * K + n * K / 32) as f64
                + cand
                    .iter()
                    .map(|&c| f64::from(c) * K as f64 * 2.0)
                    .sum::<f64>();
            println!("M={m} {fam:?}: drafter FP8 head + argmax {drafter:.1} us");
            println!(
                "M={m} {fam:?}: old {old:.1} us ({:.0} GB/s), screen {new:.1} us ({:.0} GB/s), candidates {cand:?}, {:.3} ms saved",
                ob / old / 1e3,
                nb / new / 1e3,
                (old - new) / 1e3
            );
        }
    }
    Ok(())
}
