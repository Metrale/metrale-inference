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
//!   4 across zero, FUSED) changes both the core output and the conv state.
//! - `GLM_PARITY_BASE_PTX` is required; `GLM_PARITY_FIDELITY_PTX` (this tree's
//!   `kda_chunk64_prepare.cu` compiled by the same command as the base) must equal the embedded
//!   module byte for byte, which proves the base was compiled as the build compiles.
//! - Times the unfused path (PLAIN) and FUSED as captured CUDA graphs, median [min..max] of 7
//!   samples (`common/graph_timing.rs`).
//!
//!   GLM_PARITY_BASE_PTX=base.ptx GLM_PARITY_FIDELITY_PTX=new.ptx \
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-gpu-examples --release --example glm5next_kda_chunk64_conv_parity \
//!       --features cuda,gpu-examples

#[path = "common/glm5next_kda_chunk64_conv_arms.rs"]
mod arms;
#[path = "common/glm5next_kda_chunk64_case.rs"]
mod cases;
#[path = "common/glm5next_rows_fixture.rs"]
mod fixture;
#[path = "common/graph_timing.rs"]
mod timing;

use anyhow::{Context, Result, bail, ensure};
use arms::{Arm, CD, Gate, Kernels, Layout, fused, unfused};
use cases::{bumped_copy, case_rows, compare, ensure_finite, free_inputs, inputs, leak, run_arm};
use fixture::Lcg;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_arch::glm5next_kda::chunk64::CHUNK64_D as D;

fn case(
    g: &dyn GpuBackend,
    s: u64,
    k: &Kernels,
    prep_base: KernelHandle,
    rng: &mut Lcg,
    t: usize,
) -> Result<()> {
    let x = inputs(g, rng, t)?;
    let a = Arm::new(g, &x)?;
    for layout in [Layout::Packed, Layout::Parts] {
        let what = format!("T={t} {layout:?}");
        let base = run_arm(g, s, &x, &a, &mut |r| {
            unfused(g, k, prep_base, Gate::Rows, &x, &a, layout, s, r)
        })?;
        ensure_finite(&what, &base)?;
        let plain = run_arm(g, s, &x, &a, &mut |r| {
            unfused(g, k, k.prep, Gate::Rows, &x, &a, layout, s, r)
        })?;
        compare(&format!("{what} PLAIN vs BASE"), &base, &plain)?;
        let src = x.src(layout);
        let fz = run_arm(g, s, &x, &a, &mut |r| {
            fused(g, k, k.prep_conv, Gate::Rows, &x, &a, layout, src, s, r)
        })?;
        compare(&format!("{what} FUSED vs BASE"), &base, &fz)?;
        // 2026-10-14: Negative control: one k element of row T - 2 (head 5, channel 3) moved by
        // 4 across zero, FUSED; the core and the conv state must move.
        let [_, (kk, stride), _] = x.xs(layout, DevicePtr::NULL);
        let at = kk.0 as usize + ((t - 2) * stride + 5 * D + 3) * 2;
        let bad = bumped_copy(g, src, t * CD * 2, at)?;
        let ctl = run_arm(g, s, &x, &a, &mut |r| {
            fused(g, k, k.prep_conv, Gate::Rows, &x, &a, layout, bad, s, r)
        })?;
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
            unfused(g, k, k.prep, Gate::Rows, &x, &a, layout, s, None)
        })?;
        let t_new = timing::time_graph(g, s, copies, &mut |s| {
            fused(g, k, k.prep_conv, Gate::Rows, &x, &a, layout, src, s, None)
        })?;
        timing::report(&format!("chunk64+conv {layout:?}"), t, t_old, t_new);
    }
    a.free(g)?;
    free_inputs(g, x)
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
    let k = Kernels::load(g)?;
    let prep_base = g.kernel("kda_chunk64_prepare_base", "kda_chunk64_prepare")?;
    let mut rng = Lcg(0x6b64_6136_3463_6f6e);
    for t in case_rows()? {
        case(g, s, &k, prep_base, &mut rng, t)?;
    }
    println!(
        "PASS: the conv inside the chunk64 prepare is byte-identical to the conv then the prepare"
    );
    Ok(())
}
