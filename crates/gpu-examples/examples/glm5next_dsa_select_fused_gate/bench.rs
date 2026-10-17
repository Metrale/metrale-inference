// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The timing half of the select-fused gate (`main.rs`, item 4): the reference in
//! the row sub-passes a 1M-ceiling scratch allows, then the fused kernel in one launch, at one
//! context, with both outputs compared in full.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond `main.rs`'s.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::GpuBackend;

use crate::{
    KP, Kernels, Keys, Rng, SCRATCH_SCORES, Shape, build, free, fused, reference, rows_differ,
    selected,
};

/// 2026-10-17: (median, min, max) of 7 synchronous runs, in ms.
fn time_ms(gpu: &dyn GpuBackend, f: impl Fn() -> Result<()>) -> Result<(f64, f64, f64)> {
    let mut v = Vec::new();
    for _ in 0..7 {
        gpu.synchronize(0)?;
        let t0 = std::time::Instant::now();
        f()?;
        gpu.synchronize(0)?;
        v.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    v.sort_by(f64::total_cmp);
    Ok((v[3], v[0], v[6]))
}

/// 2026-10-17: One timing line at context `ctx` (2,048 causal rows at its end, 512 selected pools);
/// fails if the two outputs differ in any row.
pub fn bench(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    ctx: usize,
    keys: Keys,
    rng: &mut Rng,
) -> Result<()> {
    let pools = ctx / KP;
    let rows = 2048;
    let per = (SCRATCH_SCORES / pools).min(rows);
    let sh = Shape {
        q: rows,
        p: pools,
        k: 512,
        keys,
        first_key: 0,
        shuffled: false,
    };
    let io = build(gpu, &sh, per, rng)?;
    let want = selected(gpu, &io, 0x00, || reference(gpu, k, &io))?;
    let got = selected(gpu, &io, 0x5A, || fused(gpu, k, &io))?;
    let bad = rows_differ(&got, &want, io.k);
    let (to, to0, to1) = time_ms(gpu, || reference(gpu, k, &io))?;
    let (tf, tf0, tf1) = time_ms(gpu, || fused(gpu, k, &io))?;
    println!(
        "ctx={ctx:8} pools={pools:6} rows={rows} keys={keys:?}: reference ({} passes of {per}) \
         {to:9.2} ms [{to0:.2}..{to1:.2}] | fused {tf:9.2} ms [{tf0:.2}..{tf1:.2}] ({:.2}x); \
         rows differ {bad}",
        rows.div_ceil(per),
        to / tf
    );
    free(gpu, &io)?;
    ensure!(bad == 0, "ctx={ctx}: fused differs from the reference");
    Ok(())
}
