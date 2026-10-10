// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Parts 3 and 4 of `glm5next_rowbatch_microtest` (the router logits and the DSA
//! indexer projections), split from it to keep each file under 500 lines.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: as `glm5next_rowbatch_microtest`'s.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_layers::layers::ops::{
    W8a8Kernels, W8a8Scale, W8a8Scratch, W8a8Weight, w8a8_act_quant, w8a8_gemv, w8a8_proj,
};
use metrale_model_layers::weight_map::{Fp8Weight, WeightQuantFormat};

use super::fixture::{Lcg, same, up, up_bf16, up_f32};
use super::timing::report;
use super::{D, HIDDEN, ROWS, read, tg, zeros};

/// 2026-10-09: `C[M, N] = A[M, K] B[N, K]^T` the way `glm_mm` dispatches a BF16 weight: the
/// M = 1 GEMV, the runtime-M batched entry at 2..=8 rows, the register-resident one at 9..=16.
#[allow(clippy::too_many_arguments)]
fn gemv(
    g: &dyn GpuBackend,
    k: [KernelHandle; 3],
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    kk: usize,
    s: u64,
) -> Result<()> {
    let l = KernelLaunch::new(
        g,
        k[if m == 1 {
            0
        } else if m <= 8 {
            1
        } else {
            2
        }],
    )
    .grid([n.div_ceil(4) as u32, 1, 1])
    .block([256, 1, 1])
    .arg_ptr(a)
    .arg_ptr(b)
    .arg_ptr(c);
    if m == 1 {
        l.arg_u32(n as u32).arg_u32(kk as u32).launch(s)
    } else {
        l.arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(kk as u32)
            .arg_u32(n as u32)
            .launch(s)
    }
}

/// 2026-10-09: Part 3 and the BF16 half of part 4: R per-row M = 1 GEMVs of each weight
/// against one batched GEMV per weight, `elem` bytes per output.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemv_family(
    g: &dyn GpuBackend,
    s: u64,
    rng: &mut Lcg,
    what: &str,
    k: [KernelHandle; 3],
    n: usize,
    weights: usize,
    elem: usize,
) -> Result<()> {
    let w: Vec<DevicePtr> = (0..weights)
        .map(|_| up_bf16(g, &rng.vec(n * HIDDEN, 0.03)))
        .collect::<Result<_>>()?;
    for rows in ROWS {
        let x = up_bf16(g, &rng.vec(rows * HIDDEN, 1.0))?;
        let outs = |_: ()| -> Result<Vec<DevicePtr>> {
            (0..weights).map(|_| zeros(g, rows * n * elem)).collect()
        };
        let (oa, ob) = (outs(())?, outs(())?);
        let old = |s: u64| -> Result<()> {
            for (wi, o) in w.iter().zip(&oa) {
                for r in 0..rows {
                    gemv(
                        g,
                        k,
                        x.offset(r * HIDDEN * 2),
                        *wi,
                        o.offset(r * n * elem),
                        1,
                        n,
                        HIDDEN,
                        s,
                    )?;
                }
            }
            Ok(())
        };
        let new = |s: u64| -> Result<()> {
            for (wi, o) in w.iter().zip(&ob) {
                gemv(g, k, x, *wi, *o, rows, n, HIDDEN, s)?;
            }
            Ok(())
        };
        old(s)?;
        new(s)?;
        for (a, b) in oa.iter().zip(&ob) {
            same(
                what,
                &read(g, s, *a, rows * n * elem)?,
                &read(g, s, *b, rows * n * elem)?,
            )?;
        }
        report(what, rows, tg(g, s, &old)?, tg(g, s, &new)?);
        for p in oa.iter().chain(&ob).chain([&x]) {
            g.free(*p)?;
        }
    }
    Ok(())
}

/// 2026-10-09: The W8A8 half of part 4 (the indexer's `wk` and compress gate under `fp8` and
/// `w4a16`), and the key LayerNorm.
pub(crate) fn indexer_w8a8(g: &dyn GpuBackend, s: u64, rng: &mut Lcg) -> Result<()> {
    let kern = W8a8Kernels::load(g);
    let scratch = W8a8Scratch::alloc(g, HIDDEN as u32)?;
    let weight = |rng: &mut Lcg| -> Result<W8a8Weight> {
        // 2026-10-09: E4M3 codes without the NaN pattern (S.1111.111).
        let bytes: Vec<u8> = (0..D * HIDDEN)
            .map(|_| {
                let b = ((rng.next() + 1.0) * 127.5) as u8;
                if b & 0x7F == 0x7F { b & 0xFE } else { b }
            })
            .collect();
        let row_scale: Vec<f32> = (0..D).map(|_| 0.001 + rng.next().abs() * 0.002).collect();
        W8a8Weight::new(&[Fp8Weight {
            weight: up(g, &bytes)?,
            row_scale: up_f32(g, &row_scale)?,
            n: D as u32,
            k: HIDDEN as u32,
            scale_format: WeightQuantFormat::Fp8PerRow,
        }])
    };
    let (wk, wg) = (weight(rng)?, weight(rng)?);
    let knorm = g.kernel("nllb_encoder", "nllb_layernorm_bf16")?;
    let (nw, nb) = (up_bf16(g, &rng.vec(D, 1.0))?, up_bf16(g, &rng.vec(D, 0.1))?);
    let norm = |p: DevicePtr, rows: usize, s: u64| -> Result<()> {
        KernelLaunch::new(g, knorm)
            .grid([rows as u32, 1, 1])
            .block([D as u32, 1, 1])
            .shared_mem((D * 4) as u32)
            .arg_ptr(p)
            .arg_ptr(nw)
            .arg_ptr(nb)
            .arg_u32(rows as u32)
            .arg_u32(D as u32)
            .arg_f32(1e-6)
            .launch(s)
    };
    for rows in ROWS {
        let x = up_bf16(g, &rng.vec(rows * HIDDEN, 1.0))?;
        let o: Vec<DevicePtr> = (0..4)
            .map(|_| zeros(g, rows * D * 2))
            .collect::<Result<_>>()?;
        let old = |s: u64| -> Result<()> {
            for r in 0..rows {
                let xr = x.offset(r * HIDDEN * 2);
                w8a8_proj(
                    g,
                    &kern,
                    &wk,
                    xr,
                    HIDDEN as u32,
                    1,
                    o[0].offset(r * D * 2),
                    D as u32,
                    &scratch,
                    s,
                )?;
                norm(o[0].offset(r * D * 2), 1, s)?;
                w8a8_proj(
                    g,
                    &kern,
                    &wg,
                    xr,
                    HIDDEN as u32,
                    1,
                    o[1].offset(r * D * 2),
                    D as u32,
                    &scratch,
                    s,
                )?;
            }
            Ok(())
        };
        let new = |s: u64| -> Result<()> {
            w8a8_act_quant(
                g,
                &kern,
                W8a8Scale::PerRow,
                x,
                HIDDEN as u32,
                rows,
                HIDDEN as u32,
                &scratch,
                s,
            )?;
            w8a8_gemv(g, &kern, &wk, &scratch, rows, o[2], D as u32, s)?;
            w8a8_gemv(g, &kern, &wg, &scratch, rows, o[3], D as u32, s)?;
            norm(o[2], rows, s)
        };
        old(s)?;
        new(s)?;
        let n = rows * D * 2;
        same(
            "indexer wk W8A8 + norm",
            &read(g, s, o[0], n)?,
            &read(g, s, o[2], n)?,
        )?;
        same(
            "indexer gate W8A8",
            &read(g, s, o[1], n)?,
            &read(g, s, o[3], n)?,
        )?;
        report(
            "indexer wk+gate W8A8 + norm",
            rows,
            tg(g, s, &old)?,
            tg(g, s, &new)?,
        );
        for p in o.iter().chain([&x]) {
            g.free(*p)?;
        }
    }
    Ok(())
}
