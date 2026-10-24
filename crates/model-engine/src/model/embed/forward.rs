// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: One packed forward of the embedding model. The pack's sequences sit back to
//! back in every activation buffer (`[rows, width]`, no padding). Per layer: RMSNorm, the
//! Q/K/V projections over all rows, RoPE from the llama3 table at each token's position in its
//! own sequence, bidirectional attention one sequence at a time, the O projection and residual
//! add, then RMSNorm, gate/up, SiLU·mul, down and residual add. After the final RMSNorm the
//! rows of each sequence are mean-pooled into FP32.
//!
//! Owner: model-engine (embedding model).
//! Invariants:
//! - Attention never crosses a sequence boundary: each launch sees one sequence's rows.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::DenseWeight;

use super::engine::{EmbedGpuModel, Scratch, u32_bytes};

impl EmbedGpuModel {
    /// 2026-10-24: `out` = `[pack.len(), hidden]` FP32 mean-pooled vectors of `pack`.
    pub(super) fn forward_pack(
        &self,
        s: &mut Scratch,
        pack: &[Vec<u32>],
        out: &mut [f32],
    ) -> Result<()> {
        let sh = self.s;
        let gpu = self.gpu.as_ref();
        let st = gpu.default_stream();
        let rows: usize = pack.iter().map(Vec::len).sum();
        ensure!(
            rows <= sh.max_pack_tokens && out.len() == pack.len() * sh.hidden,
            "embedding pack of {rows} rows / {} outputs does not fit its buffers",
            out.len()
        );
        let mut ids = Vec::with_capacity(rows);
        let mut pos = Vec::with_capacity(rows);
        let mut cu = Vec::with_capacity(pack.len() + 1);
        cu.push(0u32);
        for seq in pack {
            ids.extend_from_slice(seq);
            pos.extend(0..seq.len() as u32);
            cu.push(ids.len() as u32);
        }
        gpu.copy_h2d(&u32_bytes(&ids), s.ids)?;
        gpu.copy_h2d(&u32_bytes(&pos), s.pos)?;
        gpu.copy_h2d(&u32_bytes(&cu), s.cu)?;

        let (h, f) = (sh.hidden as u32, sh.ffn as u32);
        let (nq, nkv, hd) = (sh.q_heads as u32, sh.kv_heads as u32, sh.head_dim as u32);
        let m = rows as u32;
        let gemm = |a: DevicePtr, w: DevicePtr, c: DevicePtr, n: u32, k: u32| {
            ops::dense_gemm_bf16_pipelined(
                gpu,
                self.k.gemm,
                a,
                &DenseWeight { weight: w },
                c,
                m,
                n,
                k,
                st,
            )
        };
        let norm = |x: DevicePtr, w: DevicePtr, o: DevicePtr| {
            ops::rms_norm(
                gpu,
                self.k.rms_norm,
                x,
                &DenseWeight { weight: w },
                o,
                m,
                h,
                sh.eps,
                st,
            )
        };

        ops::batched_embed(gpu, self.k.embed, s.ids, self.w.embed, s.x, m, h, st)?;
        let inv_sqrt_d = 1.0 / (sh.head_dim as f32).sqrt();
        for lw in &self.w.layers {
            norm(s.x, lw.input_norm, s.xn)?;
            gemm(s.xn, lw.q, s.q, nq * hd, h)?;
            gemm(s.xn, lw.k, s.k, nkv * hd, h)?;
            gemm(s.xn, lw.v, s.v, nkv * hd, h)?;
            ops::rope_yarn(
                gpu,
                self.k.rope,
                s.q,
                s.k,
                s.pos,
                m,
                nq,
                nkv,
                hd,
                hd,
                self.inv_freq,
                self.rope_theta,
                st,
            )?;
            for w in cu.windows(2) {
                let (lo, len) = (w[0] as usize, w[1] - w[0]);
                let qo = lo * (nq * hd) as usize * 2;
                let ko = lo * (nkv * hd) as usize * 2;
                ops::prefill_attention(
                    gpu,
                    false,
                    self.k.attn,
                    s.q.offset(qo),
                    s.k.offset(ko),
                    s.v.offset(ko),
                    s.attn.offset(qo),
                    len,
                    1,
                    nq,
                    nkv,
                    hd,
                    inv_sqrt_d,
                    false,
                    0,
                    st,
                )?;
            }
            gemm(s.attn, lw.o, s.tmp, h, nq * hd)?;
            ops::residual_add(gpu, self.k.residual_add, s.x, s.tmp, m * h, st)?;
            norm(s.x, lw.post_norm, s.xn)?;
            gemm(s.xn, lw.gate, s.gate, f, h)?;
            gemm(s.xn, lw.up, s.up, f, h)?;
            ops::silu_mul(gpu, self.k.silu_mul, s.gate, s.up, s.act, m * f, st)?;
            gemm(s.act, lw.down, s.tmp, h, f)?;
            ops::residual_add(gpu, self.k.residual_add, s.x, s.tmp, m * h, st)?;
        }
        norm(s.x, self.w.final_norm, s.xn)?;
        let n = pack.len() as u32;
        KernelLaunch::new(gpu, self.k.mean_pool)
            .grid([n, div_ceil(h, 256), 1])
            .block([256, 1, 1])
            .arg_ptr(s.xn)
            .arg_ptr(s.cu)
            .arg_ptr(s.pooled)
            .arg_u32(h)
            .launch(st)?;
        gpu.synchronize(st)?;
        let mut bytes = vec![0u8; out.len() * 4];
        gpu.copy_d2h(s.pooled, &mut bytes)?;
        for (o, c) in out.iter_mut().zip(bytes.chunks_exact(4)) {
            *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        }
        Ok(())
    }
}
