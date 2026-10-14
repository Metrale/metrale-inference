// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Byte gate and microbench of `METRALE_GLM_KDA_CHUNK64_CONV`: the 64-token chunked
//! scan with the KDA conv inside its prepare (`kda_chunk64_prepare_conv` per slab, then
//! `causal_conv1d_window_advance_parts`) against the path it replaces (the conv over every row,
//! the window advance, then `kda_chunk64_prepare` per slab), at GLM-5.3's TP=3 per-rank shape (22
//! heads of 128, conv width 4), for both input layouts: the packed rows (the conv
//! `causal_conv1d_update_l2norm_tokens`) and the three projections (`METRALE_GLM_KDA_CONV_PACK=1`,
//! the conv `causal_conv1d_update_l2norm_tokens_parts`). The scan is `kda_chunk64_scan` in every
//! arm.
//!
//! Three arms per case: BASE (the unfused path with the base tree's `kda_chunk64_prepare`, loaded
//! from `GLM_PARITY_BASE_PTX` as a second module), PLAIN (the unfused path with this tree's
//! `kda_chunk64_prepare`) and FUSED.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants:
//! - Exits with an error on the first byte that differs from BASE in PLAIN or FUSED: the core
//!   output, the recurrent state, the advanced conv state, or any slab's six factor buffers (read
//!   after each slab's prepare). The inputs carry spikes (+-60) and runs of zero rows of one q, k
//!   and v head across a chunk and a slab boundary (a zero L2 head, a window that runs dry).
//! - Exits with an error unless BASE's core output is finite (an all-NaN output would compare
//!   equal and prove nothing), and unless the negative control (one k element of row T - 2 moved
//!   by one BF16 step, FUSED) changes both the core output and the conv state.
//! - `GLM_PARITY_BASE_PTX` is required; `GLM_PARITY_FIDELITY_PTX` (this tree's
//!   `kda_chunk64_prepare.cu` compiled by the same command as the base) must equal the embedded
//!   module byte for byte, which proves the base was compiled as the build compiles.
//! - Times the unfused path (PLAIN) and FUSED as captured CUDA graphs, median [min..max] of 7
//!   samples (`common/graph_timing.rs`).
//!
//!   GLM_PARITY_BASE_PTX=base.ptx GLM_PARITY_FIDELITY_PTX=new.ptx \
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-model-arch --release --example glm5next_kda_chunk64_conv_parity \
//!       --features cuda,gpu-examples

#[path = "common/glm5next_kda_chunk64_conv_arms.rs"]
mod arms;
#[path = "common/glm5next_rows_fixture.rs"]
mod fixture;
#[path = "common/graph_timing.rs"]
mod timing;

use anyhow::{Context, Result, bail, ensure};
use arms::{Arm, CD, H, In, KC, Kernels, Layout, QKV, Run, finish, fused, unfused};
use fixture::{Lcg, same, up_bf16, up_f32};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_arch::glm5next_kda::chunk64::CHUNK64_D as D;

/// 2026-10-14: Partial chunk, partial slab, many slabs, and one 8192-row prefill sub-chunk.
const ROWS: [usize; 4] = [70, 300, 1100, 8192];

/// 2026-10-14: q|k|v rows `[t, CD]`: unit-scale noise, every 997th element +-60, and zero runs of
/// q head 7, k head 11 and v head 2 over rows `[r, r + 10)` at the middle row and (past 270 rows)
/// at 250, across the first slab boundary.
fn rows(rng: &mut Lcg, t: usize) -> Vec<f32> {
    let mut v = rng.vec(t * CD, 1.0);
    for (i, x) in v.iter_mut().enumerate() {
        if i % 997 == 0 {
            *x = if i % 2 == 0 { 60.0 } else { -60.0 };
        }
    }
    let mut runs = vec![t / 2 - 5];
    if t > 270 {
        runs.push(250);
    }
    for r in runs {
        for row in r..r + 10 {
            for (part, head) in [(0, 7), (1, 11), (2, 2)] {
                let o = row * CD + part * QKV + head * D;
                v[o..o + D].iter_mut().for_each(|x| *x = 0.0);
            }
        }
    }
    v
}

fn inputs(g: &dyn GpuBackend, rng: &mut Lcg, t: usize) -> Result<In> {
    let packed = rows(rng, t);
    let mut parts = vec![0f32; t * CD];
    for row in 0..t {
        for p in 0..3 {
            let (src, dst) = (row * CD + p * QKV, p * t * QKV + row * QKV);
            parts[dst..dst + QKV].copy_from_slice(&packed[src..src + QKV]);
        }
    }
    // 2026-10-14: Gates in (-5, 0), mostly slow; beta in (0, 1).
    let gate: Vec<f32> = rng
        .vec(t * QKV, 1.0)
        .iter()
        .map(|u| -1.25 * (u + 1.0) * (u + 1.0))
        .collect();
    let beta: Vec<f32> = rng.vec(t * H, 0.5).iter().map(|u| u + 0.5).collect();
    Ok(In {
        t,
        packed: up_bf16(g, &packed)?,
        parts: up_bf16(g, &parts)?,
        weight: up_bf16(g, &rng.vec(CD * KC, 0.5))?,
        gate: up_f32(g, &gate)?,
        beta: up_f32(g, &beta)?,
        conv0: rng.vec(CD * KC, 1.0),
        rec0: rng.vec(H * D * D, 0.05),
    })
}

fn compare(what: &str, want: &Run, got: &Run) -> Result<()> {
    same(&format!("{what} core"), &want.core, &got.core)?;
    same(&format!("{what} recurrent state"), &want.rec, &got.rec)?;
    same(&format!("{what} conv state"), &want.conv, &got.conv)?;
    ensure!(want.slabs.len() == got.slabs.len(), "{what}: slab count");
    const NAMES: [&str; 6] = ["U", "W", "Qg", "Kd", "P", "A+decay"];
    for (i, (a, b)) in want.slabs.iter().zip(&got.slabs).enumerate() {
        for ((name, x), y) in NAMES.iter().zip(a).zip(b) {
            same(&format!("{what} slab {i} {name}"), x, y)?;
        }
    }
    Ok(())
}

fn case(g: &dyn GpuBackend, s: u64, k: &Kernels, rng: &mut Lcg, t: usize) -> Result<()> {
    let x = inputs(g, rng, t)?;
    let a = Arm::new(g, &x)?;
    for layout in [Layout::Packed, Layout::Parts] {
        let what = format!("T={t} {layout:?}");
        let run = |f: &mut dyn FnMut(Option<&mut Vec<[Vec<u8>; 6]>>) -> Result<()>| {
            a.reset(g, &x)?;
            let mut sl = Vec::new();
            f(Some(&mut sl))?;
            finish(g, s, &x, &a, sl)
        };
        let base = run(&mut |r| unfused(g, k, k.prep_base, &x, &a, layout, s, r))?;
        let finite = base
            .core
            .chunks_exact(4)
            .all(|c| f32::from_le_bytes(c.try_into().unwrap()).is_finite());
        ensure!(
            finite,
            "{what}: the base core output is not finite; the gate would prove nothing"
        );
        let plain = run(&mut |r| unfused(g, k, k.prep, &x, &a, layout, s, r))?;
        compare(&format!("{what} PLAIN vs BASE"), &base, &plain)?;
        let src = x.src(layout);
        let fz = run(&mut |r| fused(g, k, &x, &a, layout, src, s, r))?;
        compare(&format!("{what} FUSED vs BASE"), &base, &fz)?;
        // 2026-10-14: Negative control: one k element of row T - 2 (head 5, channel 3) one BF16
        // step up, FUSED; the core and the conv state must move.
        let n = t * CD * 2;
        let bad = g.alloc(n)?;
        g.copy_d2d(src, bad, n)?;
        let [_, (kk, stride), _] = x.xs(layout, bad);
        let e = kk.offset(((t - 2) * stride + 5 * D + 3) * 2);
        let mut h = [0u8; 2];
        g.copy_d2h(e, &mut h)?;
        let up = bf16::from_bits(u16::from_le_bytes(h).wrapping_add(1));
        g.copy_h2d(&up.to_le_bytes(), e)?;
        let ctl = run(&mut |r| fused(g, k, &x, &a, layout, bad, s, r))?;
        g.free(bad)?;
        let caught = compare(&format!("{what} CONTROL vs BASE"), &base, &ctl).is_err();
        ensure!(
            caught && ctl.core != base.core && ctl.conv != base.conv,
            "{what}: the negative control was not caught (compare, core and conv state must differ)"
        );
        println!(
            "{what}: BASE = PLAIN = FUSED byte for byte ({} slabs); NEG CONTROL: differs (core, conv state) - caught",
            base.slabs.len()
        );
        let copies = if t <= 300 { 8 } else { 2 };
        let t_old = timing::time_graph(g, s, copies, &mut |s| {
            unfused(g, k, k.prep, &x, &a, layout, s, None)
        })?;
        let t_new = timing::time_graph(g, s, copies, &mut |s| {
            fused(g, k, &x, &a, layout, src, s, None)
        })?;
        timing::report(&format!("chunk64+conv {layout:?}"), t, t_old, t_new);
    }
    a.free(g)?;
    for p in [x.packed, x.parts, x.weight, x.gate, x.beta] {
        g.free(p)?;
    }
    Ok(())
}

fn leak(var: &str) -> Result<&'static [u8]> {
    let path = std::env::var(var).with_context(|| format!("{var} is required"))?;
    Ok(Box::leak(
        std::fs::read(&path)
            .with_context(|| format!("{var}={path}"))?
            .into_boxed_slice(),
    ))
}

fn main() -> Result<()> {
    let sets = metrale_kernels::all_ptx_sets();
    let glm = sets
        .iter()
        .find(|s| s.target.model == "glm-5.3-flash")
        .context("glm-5.3-flash kernel target not built")?;
    let mut modules = glm.modules.clone();
    let embedded = modules
        .iter()
        .find(|m| m.0 == "kda_chunk64_prepare")
        .context("no kda_chunk64_prepare module")?
        .1;
    if leak("GLM_PARITY_FIDELITY_PTX")? != embedded {
        bail!(
            "GLM_PARITY_FIDELITY_PTX differs from the embedded kda_chunk64_prepare: the base PTX was not compiled as the build compiles"
        );
    }
    modules.push(("kda_chunk64_prepare_base", leak("GLM_PARITY_BASE_PTX")?));
    let gpu = MetraleCudaBackend::new(0, &modules)?;
    let g: &dyn GpuBackend = &gpu;
    let s = g.create_stream()?;
    let k = Kernels {
        conv: g.kernel("kda_conv_tokens", "causal_conv1d_update_l2norm_tokens")?,
        parts: g.kernel(
            "kda_conv_tokens",
            "causal_conv1d_update_l2norm_tokens_parts",
        )?,
        window: g.kernel("kda_conv_tokens", "causal_conv1d_window_advance")?,
        window_parts: g.kernel("kda_conv_tokens", "causal_conv1d_window_advance_parts")?,
        prep_base: g.kernel("kda_chunk64_prepare_base", "kda_chunk64_prepare")?,
        prep: g.kernel("kda_chunk64_prepare", "kda_chunk64_prepare")?,
        prep_conv: g.kernel("kda_chunk64_prepare", "kda_chunk64_prepare_conv")?,
        scan: g.kernel("kda_chunk64", "kda_chunk64_scan")?,
    };
    let rows: Vec<usize> = match std::env::var("GLM_PARITY_ROWS") {
        Ok(v) => v
            .split(',')
            .map(|x| x.trim().parse())
            .collect::<Result<_, _>>()?,
        Err(_) => ROWS.to_vec(),
    };
    let mut rng = Lcg(0x6b64_6136_3463_6f6e);
    for t in rows {
        ensure!(t >= 16, "GLM_PARITY_ROWS: {t} rows; the zero runs need 16");
        case(g, s, &k, &mut rng, t)?;
    }
    println!(
        "PASS: the conv inside the chunk64 prepare is byte-identical to the conv then the prepare"
    );
    Ok(())
}
