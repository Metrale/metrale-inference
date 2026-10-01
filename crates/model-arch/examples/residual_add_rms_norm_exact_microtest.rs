// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: `residual_add_rms_norm_exact` against the unfused pair it replaces at a layer
//! boundary, `bf16_residual_add` then `rms_norm_residual`, byte for byte.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Exits with an error unless, on every compiled circuit target, both hidden sizes and every
//!   row count 1..=128, the fused kernel's hidden, output and residual bytes equal the pair's,
//!   and rows past the launch are untouched.
//! - A control runs `residual_add_rms_norm` (which squares the FP32 sums before rounding) on
//!   the same inputs; it must differ from the pair, or the comparison could not see a rounding
//!   difference and the run fails.
//!
//! Run (GB10, the default all-target kernel build):
//!   cargo run -p metrale-model-arch --features cuda,gpu-examples \
//!     --example residual_add_rms_norm_exact_microtest

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

#[path = "common/norm_fusion_fixture.rs"]
mod fixture;
use fixture::{EPS, HIDDENS, MAX_ROWS, Rng, backends, bytes, compare, matrix, sentinel, upload};

struct Kernels {
    add: KernelHandle,
    norm: KernelHandle,
    fused: KernelHandle,
    control: KernelHandle,
}

/// 2026-09-28: One norm-shaped launch: grid rows, block min(h, 1024).
fn norm_launch(
    g: &dyn GpuBackend,
    k: KernelHandle,
    ptrs: &[DevicePtr],
    rows: usize,
    h: usize,
) -> Result<()> {
    let mut l =
        KernelLaunch::new(g, k)
            .grid([rows as u32, 1, 1])
            .block([(h as u32).min(1024), 1, 1]);
    for &p in ptrs {
        l = l.arg_ptr(p);
    }
    l.arg_u32(h as u32).arg_f32(EPS).launch(g.default_stream())
}

/// 2026-09-28: Mismatching bytes for `rows` rows of width `h`.
fn run_rows(
    g: &dyn GpuBackend,
    k: &Kernels,
    hidden: &[u8],
    src: DevicePtr,
    w: DevicePtr,
    rows: usize,
    h: usize,
) -> Result<usize> {
    let total = MAX_ROWS * h * 2;
    let live = rows * h * 2;
    let (ha, hb) = (upload(g, hidden)?, upload(g, hidden)?);
    let (oa, ra, ob, rb) = (
        sentinel(g, total)?,
        sentinel(g, total)?,
        sentinel(g, total)?,
        sentinel(g, total)?,
    );
    let n = (rows * h) as u32;
    KernelLaunch::new(g, k.add)
        .grid([div_ceil(n, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(ha)
        .arg_ptr(src)
        .arg_u32(n)
        .launch(g.default_stream())?;
    norm_launch(g, k.norm, &[ha, w, oa, ra], rows, h)?;
    norm_launch(g, k.fused, &[hb, src, w, ob, rb], rows, h)?;
    let bad = compare(g, ha, hb, total, total)?
        + compare(g, oa, ob, live, total)?
        + compare(g, ra, rb, live, total)?;
    for p in [ha, hb, oa, ra, ob, rb] {
        g.free(p)?;
    }
    Ok(bad)
}

/// 2026-09-28: Bytes by which `residual_add_rms_norm` differs from the pair at 128 rows.
fn control(
    g: &dyn GpuBackend,
    k: &Kernels,
    hidden: &[u8],
    src: DevicePtr,
    w: DevicePtr,
    h: usize,
) -> Result<usize> {
    let total = MAX_ROWS * h * 2;
    let (ha, hc) = (upload(g, hidden)?, upload(g, hidden)?);
    let (oa, ra, oc, rc) = (
        sentinel(g, total)?,
        sentinel(g, total)?,
        sentinel(g, total)?,
        sentinel(g, total)?,
    );
    let n = (MAX_ROWS * h) as u32;
    KernelLaunch::new(g, k.add)
        .grid([div_ceil(n, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(ha)
        .arg_ptr(src)
        .arg_u32(n)
        .launch(g.default_stream())?;
    norm_launch(g, k.norm, &[ha, w, oa, ra], MAX_ROWS, h)?;
    norm_launch(g, k.control, &[hc, src, w, oc, rc], MAX_ROWS, h)?;
    let differs = compare(g, oa, oc, total, total)?;
    for p in [ha, hc, oa, ra, oc, rc] {
        g.free(p)?;
    }
    Ok(differs)
}

fn main() -> Result<()> {
    let mut failures = 0usize;
    for (target, backend) in backends()? {
        let g: &dyn GpuBackend = &backend;
        let k = Kernels {
            add: g.kernel("residual_add", "bf16_residual_add")?,
            norm: g.kernel("norm", "rms_norm_residual")?,
            fused: g.kernel("residual_add_rms_norm_exact", "residual_add_rms_norm_exact")?,
            control: g.kernel("norm", "residual_add_rms_norm")?,
        };
        for h in HIDDENS {
            let mut rng = Rng(0x0c1c_0000 + h as u64);
            let hidden = bytes(&matrix(&mut rng, MAX_ROWS, h, 0));
            let src = upload(g, &bytes(&matrix(&mut rng, MAX_ROWS, h, 1000)))?;
            let w = fixture::weight(g, &mut rng, h)?;
            let mut bad = 0usize;
            for rows in 1..=MAX_ROWS {
                let b = run_rows(g, &k, &hidden, src, w, rows, h)?;
                if b > 0 {
                    println!("  {target} H={h} rows={rows}: {b} mismatching bytes");
                }
                bad += b;
            }
            let ctl = control(g, &k, &hidden, src, w, h)?;
            println!(
                "{target:<16} H={h:<5} rows 1..={MAX_ROWS}: {bad} mismatching bytes \
                 (control residual_add_rms_norm differs in {ctl} bytes)"
            );
            ensure!(
                ctl > 0,
                "{target} H={h}: the control matched; the test cannot see rounding"
            );
            failures += bad;
            g.free(src)?;
            g.free(w)?;
        }
    }
    ensure!(failures == 0, "{failures} mismatching bytes");
    println!(
        "PASS: residual_add_rms_norm_exact is byte-identical to bf16_residual_add + rms_norm_residual"
    );
    Ok(())
}
