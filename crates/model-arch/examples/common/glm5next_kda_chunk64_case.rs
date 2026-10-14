// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The cases the chunk64 byte gates share (`glm5next_kda_chunk64_conv_parity.rs`,
//! `glm5next_kda_gate_in_prepare_parity.rs`, `glm5next_kda_onorm_in_scan_parity.rs`): the row
//! counts, the q/k/v rows with their spikes and zero runs, one arm's run from a reset, the byte
//! comparison of two runs, the negative control's perturbation (4 across zero), and the kernel
//! modules with a base tree's module beside this tree's.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: none beyond the types.
#![allow(dead_code)]

use super::arms::{Arm, CD, H, In, KC, QKV, Run, finish};
use super::fixture::{Lcg, same, up_bf16, up_f32};
use anyhow::{Context, Result, bail, ensure};
use half::bf16;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_kda::chunk64::CHUNK64_D as D;

/// 2026-10-14: Partial chunk, partial slab, many slabs, and one 8192-row prefill sub-chunk.
const ROWS: [usize; 4] = [70, 300, 1100, 8192];

/// 2026-10-14: The row counts: `GLM_PARITY_ROWS` (a comma list) or [`ROWS`].
pub fn case_rows() -> Result<Vec<usize>> {
    let rows: Vec<usize> = match std::env::var("GLM_PARITY_ROWS") {
        Ok(v) => v
            .split(',')
            .map(|x| x.trim().parse())
            .collect::<Result<_, _>>()?,
        Err(_) => ROWS.to_vec(),
    };
    for &t in &rows {
        ensure!(t >= 16, "GLM_PARITY_ROWS: {t} rows; the zero runs need 16");
    }
    Ok(rows)
}

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

/// 2026-10-14: One case's inputs; the FP32 gate rows in (-5, 0), mostly slow, beta in (0, 1).
pub fn inputs(g: &dyn GpuBackend, rng: &mut Lcg, t: usize) -> Result<In> {
    let packed = rows(rng, t);
    let mut parts = vec![0f32; t * CD];
    for row in 0..t {
        for p in 0..3 {
            let (src, dst) = (row * CD + p * QKV, p * t * QKV + row * QKV);
            parts[dst..dst + QKV].copy_from_slice(&packed[src..src + QKV]);
        }
    }
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

pub fn free_inputs(g: &dyn GpuBackend, x: In) -> Result<()> {
    for p in [x.packed, x.parts, x.weight, x.gate, x.beta] {
        g.free(p)?;
    }
    Ok(())
}

/// 2026-10-14: One arm's run from the case's initial state, each slab's factors recorded.
pub fn run_arm(
    g: &dyn GpuBackend,
    s: u64,
    x: &In,
    a: &Arm,
    f: &mut dyn FnMut(Option<&mut Vec<[Vec<u8>; 6]>>) -> Result<()>,
) -> Result<Run> {
    a.reset(g, x)?;
    let mut sl = Vec::new();
    f(Some(&mut sl))?;
    finish(g, s, x, a, sl)
}

/// 2026-10-14: An error unless every FP32 of `run`'s core output is finite (an all-NaN output
/// would compare equal and prove nothing).
pub fn ensure_finite(what: &str, run: &Run) -> Result<()> {
    let finite = run
        .core
        .chunks_exact(4)
        .all(|c| f32::from_le_bytes(c.try_into().unwrap()).is_finite());
    ensure!(
        finite,
        "{what}: the base core output is not finite; the gate would prove nothing"
    );
    Ok(())
}

/// 2026-10-14: The first byte of `got` that differs from `want` (core, recurrent state, conv
/// state, each slab's six factors), as an error.
pub fn compare(what: &str, want: &Run, got: &Run) -> Result<()> {
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

/// 2026-10-14: A device copy of the `n` bytes at `src` with the BF16 `x` at byte offset `at` set to
/// `x - 4 sign(x)` (the negative controls' input); the caller frees it. A one-step move can round
/// away in a BF16 value downstream (the conv rows, the prepare's BF16 factors) and leave the core
/// unmoved; across zero, a gate's sigmoid input moves toward its steep middle, never onto a flat
/// end where FP32 saturates.
pub fn bumped_copy(g: &dyn GpuBackend, src: DevicePtr, n: usize, at: usize) -> Result<DevicePtr> {
    let bad = g.alloc(n)?;
    g.copy_d2d(src, bad, n)?;
    let e = bad.offset(at);
    let mut h = [0u8; 2];
    g.copy_d2h(e, &mut h)?;
    let x = bf16::from_bits(u16::from_le_bytes(h)).to_f32();
    let up = bf16::from_f32(x - 4.0 * x.signum());
    g.copy_h2d(&up.to_le_bytes(), e)?;
    Ok(bad)
}

/// 2026-10-14: The bytes of the file `var` names, for the life of the process.
pub fn leak(var: &str) -> Result<&'static [u8]> {
    let path = std::env::var(var).with_context(|| format!("{var} is required"))?;
    Ok(Box::leak(
        std::fs::read(&path)
            .with_context(|| format!("{var}={path}"))?
            .into_boxed_slice(),
    ))
}

/// 2026-10-14: The glm-5.3-flash kernel modules plus the base tree's module `name`
/// (`GLM_PARITY_BASE_PTX`) loaded as `base`, after checking that `GLM_PARITY_FIDELITY_PTX` (this
/// tree's source compiled by the base's command) equals the embedded `name` byte for byte, which
/// proves the base was compiled as the build compiles.
pub fn modules_with_base(
    name: &str,
    base: &'static str,
) -> Result<Vec<(&'static str, &'static [u8])>> {
    let sets = metrale_kernels::all_ptx_sets();
    let glm = sets
        .iter()
        .find(|s| s.target.model == "glm-5.3-flash")
        .context("glm-5.3-flash kernel target not built")?;
    let mut modules = glm.modules.clone();
    let embedded = modules
        .iter()
        .find(|m| m.0 == name)
        .with_context(|| format!("no {name} module"))?
        .1;
    if leak("GLM_PARITY_FIDELITY_PTX")? != embedded {
        bail!(
            "GLM_PARITY_FIDELITY_PTX differs from the embedded {name}: the base PTX was not compiled as the build compiles"
        );
    }
    modules.push((base, leak("GLM_PARITY_BASE_PTX")?));
    Ok(modules)
}
