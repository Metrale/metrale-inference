// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Part 5 of `glm5next_rowbatch_microtest`: the KDA conv and recurrence of a
//! T-token prompt chunk of ONE sequence, per token (two launches a token, the prefill's default)
//! against the token kernels (`causal_conv1d_update_l2norm_tokens`, `causal_conv1d_window_advance`,
//! `kda_recurrent_decode_bf16_seq_reg`), byte for byte, at the TP=3 per-rank shape.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: as `glm5next_rowbatch_microtest`'s.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::fixture::{Lcg, same, up_bf16, up_f32};
use super::timing::report;
use super::{D, H, read, tg, zeros};

/// 2026-10-09: Prompt chunks: a short chat prompt, the TTFT probe's, and the prefill rows cap.
const TS: [usize; 3] = [77, 198, 512];

pub(crate) fn kda_tokens(g: &dyn GpuBackend, s: u64, rng: &mut Lcg) -> Result<()> {
    let (qkv, kc) = (H * D, 4usize);
    let (cd, qk) = (3 * qkv, 2 * qkv);
    let conv1 = g.kernel("causal_conv1d", "causal_conv1d_update_l2norm")?;
    let conv_t = g.kernel("kda_conv_tokens", "causal_conv1d_update_l2norm_tokens")?;
    let conv_w = g.kernel("kda_conv_tokens", "causal_conv1d_window_advance")?;
    let rec1 = g.kernel("kda_recurrent", "kda_recurrent_decode_bf16_smem")?;
    let rec_t = g.kernel("kda_recurrent", "kda_recurrent_decode_bf16_seq_reg")?;
    let (vpb, scale) = (32usize, 1.0 / (D as f32).sqrt());
    let smem = ((3 * D + vpb * (D + 1)) * 4) as u32;
    let weight = up_bf16(g, &rng.vec(cd * kc, 0.5))?;
    let (cb, hb) = (cd * kc * 4, H * D * D * 4);
    for t in TS {
        let x = up_bf16(g, &rng.vec(t * cd, 1.0))?;
        let gate: Vec<f32> = (0..t * qkv)
            .map(|_| -0.05 - rng.next().abs() * 0.1)
            .collect();
        let gate = up_f32(g, &gate)?;
        let beta: Vec<f32> = (0..t * H).map(|_| 0.5 + rng.next() * 0.2).collect();
        let beta = up_f32(g, &beta)?;
        let (c0, h0) = (rng.vec(cd * kc, 0.5), rng.vec(H * D * D, 0.1));
        let arm = || -> Result<[DevicePtr; 4]> {
            Ok([
                up_f32(g, &c0)?,
                up_f32(g, &h0)?,
                zeros(g, t * cd * 2)?,
                zeros(g, t * qkv * 4)?,
            ])
        };
        let (a, b) = (arm()?, arm()?);
        let old = |[cs, hs, co, out]: [DevicePtr; 4], s: u64| -> Result<()> {
            for r in 0..t {
                KernelLaunch::new(g, conv1)
                    .grid([cd.div_ceil(256) as u32, 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(cs)
                    .arg_ptr(x.offset(r * cd * 2))
                    .arg_ptr(weight)
                    .arg_ptr(DevicePtr::NULL)
                    .arg_ptr(co.offset(r * cd * 2))
                    .arg_u32(1)
                    .arg_u32(cd as u32)
                    .arg_u32(kc as u32)
                    .arg_u32(qk as u32)
                    .arg_u32(D as u32)
                    .arg_f32(1e-6)
                    .launch(s)?;
                let q = co.offset(r * cd * 2);
                KernelLaunch::new(g, rec1)
                    .grid([H as u32, (D / vpb) as u32, 1])
                    .block([vpb as u32, 1, 1])
                    .shared_mem(smem)
                    .arg_ptr(q)
                    .arg_ptr(q.offset(qkv * 2))
                    .arg_ptr(q.offset(qkv * 4))
                    .arg_ptr(gate.offset(r * qkv * 4))
                    .arg_ptr(beta.offset(r * H * 4))
                    .arg_ptr(hs)
                    .arg_ptr(out.offset(r * qkv * 4))
                    .arg_u32(H as u32)
                    .arg_u32(D as u32)
                    .arg_f32(scale)
                    .arg_u32(vpb as u32)
                    .launch(s)?;
            }
            Ok(())
        };
        let new = |[cs, hs, co, out]: [DevicePtr; 4], s: u64| -> Result<()> {
            KernelLaunch::new(g, conv_t)
                .grid([cd.div_ceil(256) as u32, t as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(cs)
                .arg_ptr(x)
                .arg_ptr(weight)
                .arg_ptr(DevicePtr::NULL)
                .arg_ptr(co)
                .arg_u32(cd as u32)
                .arg_u32(kc as u32)
                .arg_u32(qk as u32)
                .arg_u32(D as u32)
                .arg_f32(1e-6)
                .launch(s)?;
            KernelLaunch::new(g, conv_w)
                .grid([cd.div_ceil(256) as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(cs)
                .arg_ptr(x)
                .arg_u32(cd as u32)
                .arg_u32(kc as u32)
                .arg_u32(t as u32)
                .launch(s)?;
            KernelLaunch::new(g, rec_t)
                .grid([H as u32, (D / vpb) as u32, 1])
                .block([vpb as u32, 1, 1])
                .shared_mem((3 * D * 4) as u32)
                .arg_ptr(co)
                .arg_ptr(co.offset(qkv * 2))
                .arg_ptr(co.offset(qkv * 4))
                .arg_ptr(gate)
                .arg_ptr(beta)
                .arg_ptr(hs)
                .arg_ptr(out)
                .arg_u32(H as u32)
                .arg_f32(scale)
                .arg_u32(t as u32)
                .arg_u32(cd as u32)
                .arg_u32(qkv as u32)
                .arg_u32(H as u32)
                .arg_u32(qkv as u32)
                .launch(s)
        };
        old(a, s)?;
        new(b, s)?;
        for (i, (what, n)) in [
            ("conv state", cb),
            ("recurrent state", hb),
            ("conv out", t * cd * 2),
            ("core out", t * qkv * 4),
        ]
        .into_iter()
        .enumerate()
        {
            same(
                &format!("KDA tokens T={t} {what}"),
                &read(g, s, a[i], n)?,
                &read(g, s, b[i], n)?,
            )?;
        }
        report(
            "KDA prefill conv+recurrence",
            t,
            tg(g, s, &|s| old(a, s))?,
            tg(g, s, &|s| new(b, s))?,
        );
        for p in a.iter().chain(&b).chain([&x, &gate, &beta]) {
            g.free(*p)?;
        }
    }
    Ok(())
}
