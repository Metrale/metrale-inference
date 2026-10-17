// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: BF16-weight FP32-out projection leg of `glm5next_batch_inv_bench` (router
//! 288 x 4096, DSA indexer wq_b 4096 x 1536 and weights_proj 32 x 4096): row r of the batched GEMV
//! at M = 2..=16 against the one-row GEMV on row r (the decode tiers' row invariance), then the
//! cost at 1, 8 and 16 rows, and at 64 rows as four 16-row batched launches (the canonical tier
//! the batch-invariant lever would run on prefill pieces) against one cuBLASLt call (today).
//!
//! Owner: model-arch examples.
//! Invariants: errors on the first row of a batched launch whose bytes differ from the one-row
//! GEMV's.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_arch::glm5next_mlp::Glm5NextMlpKernels;
use metrale_model_layers::layers::ops::{self, DenseMmKernels};

use crate::fixture::{Rng, bf16_rows, read, stat, upload};
use crate::timing;

const SHAPES: [(&str, u32, u32); 3] = [
    ("router", 288, 4096),
    ("dsa wq_b", 4096, 1536),
    ("dsa weights_proj", 32, 4096),
];

pub fn run(g: &dyn GpuBackend, s: u64, rng: &mut Rng) -> Result<()> {
    let k = Glm5NextMlpKernels::resolve(g)?;
    ensure!(
        k.gemv_f32.0 != 0 && k.gemv_batchm_f32.0 != 0,
        "the FP32-out GEMV tiers did not resolve"
    );
    let km = DenseMmKernels {
        gemm: k.gemm_f32,
        gemv: k.gemv_f32,
        batchm: k.gemv_batchm_f32,
    };
    let ad = upload(g, &bf16_rows(rng, 64, 4096))?;
    let out = g.alloc(64 * 4096 * 4)?;
    let one = g.alloc(4096 * 4)?;
    for (name, n, kk) in SHAPES {
        // 2026-10-11: Enough weight copies that a timed graph streams from DRAM.
        let wbytes = n as usize * kk as usize * 2;
        let copies = (96usize << 20).div_ceil(wbytes).clamp(4, 512);
        let ws: Vec<_> = (0..copies)
            .map(|_| upload(g, &bf16_rows(rng, n as usize, kk as usize)))
            .collect::<Result<_>>()?;
        let nb = n as usize * 4;
        let mut alone = Vec::new();
        for r in 0..16 {
            ops::dense_mm_bf16(
                g,
                &km,
                ad.offset(r * kk as usize * 2),
                ws[0],
                one,
                1,
                n as usize,
                kk as usize,
                s,
            )?;
            alone.push(read(g, s, one, nb)?);
        }
        for m in 2..=16usize {
            ops::dense_mm_bf16(g, &km, ad, ws[0], out, m, n as usize, kk as usize, s)?;
            let got = read(g, s, out, m * nb)?;
            for (r, want) in alone.iter().enumerate().take(m) {
                ensure!(
                    &got[r * nb..(r + 1) * nb] == want.as_slice(),
                    "{name} M={m} row {r}: the batched GEMV differs from the one-row GEMV"
                );
            }
        }
        let mut line =
            format!("bf16 {name:<17} [{n:>4} x {kk:>4}] rows 2..=16 equal the 1-row GEMV;");
        for (label, m, mode) in [
            ("M=1", 1usize, 0),
            ("M=8", 8, 0),
            ("M=16", 16, 0),
            ("M=64 4x16 batched", 64, 1),
            ("M=64 cuBLASLt", 64, 2),
        ] {
            let mut v = Vec::new();
            for _ in 0..5 {
                let mut i = 0usize;
                let t = timing::time_graph(g, s, copies, &mut |st| {
                    let w = ws[i % copies];
                    i += 1;
                    match mode {
                        0 => ops::dense_mm_bf16(g, &km, ad, w, out, m, n as usize, kk as usize, st),
                        1 => {
                            for c in 0..m / 16 {
                                ops::dense_mm_bf16(
                                    g,
                                    &km,
                                    ad.offset(c * 16 * kk as usize * 2),
                                    w,
                                    out.offset(c * 16 * nb),
                                    16,
                                    n as usize,
                                    kk as usize,
                                    st,
                                )?;
                            }
                            Ok(())
                        }
                        _ => ops::cublas_bf16_proj_dense_f32_out(ad, w, out, m as u32, n, kk, st),
                    }
                })?;
                v.push(t.0);
            }
            let t = stat(&mut v);
            line += &format!(" {label} {:.1} us [{:.1}..{:.1}];", t.0, t.1, t.2);
        }
        println!("{line}");
        for w in ws {
            g.free(w)?;
        }
    }
    g.free(ad)?;
    Ok(())
}
