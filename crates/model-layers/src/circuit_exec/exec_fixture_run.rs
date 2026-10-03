// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The fixture's run helpers, split out of `exec_fixture.rs` unchanged: GDN states
//! per row, running a program on the recording mock, and the check that every pointer a launch
//! reads is known.
//!
//! Owner: model-layers circuit executor tests.
//! Invariants: none beyond the types.

use std::collections::BTreeSet;

use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};

use super::bindings::*;
use super::exec_fixture::*;
use super::program::{GdnState, StepEnv};

pub(super) fn states(f: &Fixture, base: u64) -> Vec<Vec<GdnState>> {
    states_rows(f, base, 1)
}

/// 2026-09-30: Bytes between two rows' GDN states in [`states_rows`], for both pools: the
/// layers' slot pitch, so those rows are contiguous.
pub(super) const STATE_PITCH: u64 = 1 << 16;

/// 2026-09-28: `rows` distinct GDN states per GDN layer, from `base`.
pub(super) fn states_rows(f: &Fixture, base: u64, rows: usize) -> Vec<Vec<GdnState>> {
    f.layers
        .iter()
        .enumerate()
        .map(|(i, l)| match l.mixer {
            MixerFacts::Gdn(_) => (0..rows)
                .map(|r| {
                    let at = base + ((i as u64) << 24) + r as u64 * STATE_PITCH;
                    GdnState {
                        h: ptr(at),
                        conv: ptr(at + 0x8000),
                        h_steps: [1, 2, 3].map(|t| ptr(at + 0x1000 * t)),
                        conv_steps: [1, 2, 3].map(|t| ptr(at + 0x8000 + 0x1000 * t)),
                    }
                })
                .collect(),
            MixerFacts::Attention(_) => Vec::new(),
        })
        .collect()
}

/// 2026-09-29: The program's kernel launches, in order: what the mock records (it does not
/// record copies).
pub(super) fn kernel_launches(f: &Fixture) -> impl Iterator<Item = &super::program::Launch> {
    f.program
        .launches
        .iter()
        .filter(|l| l.kind == super::program::LaunchKind::Kernel)
}

pub(super) fn run(f: &Fixture, gdn: &[Vec<GdnState>], max_blocks: u32) -> Vec<MockLaunch> {
    let gpu = MockGpuBackend::new();
    // 2026-09-30: Every mock kernel is handle 0xDEAD, and the tiled W4A16 launches size their
    // grid from the kernel's published N tile (`GpuBackend::kernel_n_tile`); 128 is the tile
    // `w4a16_gemm_t_p3` publishes in the qwen3.6-27b tree.
    gpu.set_kernel_n_tile(metrale_gpu_runtime::gpu::KernelHandle(0xDEAD), 128);
    run_on(&gpu, f, gdn, max_blocks)
}

/// 2026-09-29: [`run`] on `gpu`, whose allocations the states may point into.
pub(super) fn run_on(
    gpu: &MockGpuBackend,
    f: &Fixture,
    gdn: &[Vec<GdnState>],
    max_blocks: u32,
) -> Vec<MockLaunch> {
    f.program
        .run(&StepEnv {
            gpu,
            stream: 7,
            gdn,
            max_blocks_per_seq: max_blocks,
            prefill: None,
        })
        .unwrap();
    gpu.launches_snapshot()
}

/// 2026-09-28: Every buffer a launch of `f` reads is a bound weight, a row of a fixed buffer,
/// the step's metadata, one of `gdn`'s states, or in the workspace.
pub(super) fn assert_pointers_known(f: &Fixture, gdn: &[Vec<GdnState>], launched: &[MockLaunch]) {
    let mut known: BTreeSet<u64> = BTreeSet::new();
    for l in f.layers.iter().chain(&f.draft) {
        for w in l.weights.values() {
            // 2026-09-30: Each pointer inserted on its own (an `&&` skipped the scale whenever
            // the weight was already known). A W8A8 weight's segments are in `f.w8a8_ptrs`.
            match w {
                BoundWeight::Dense(d) => known.extend([d.weight.0]),
                BoundWeight::Nvfp4(q) => known.extend([q.weight.0, q.weight_scale.0]),
                BoundWeight::Mmq(p) => known.extend([p.0]),
                BoundWeight::W8a8(..) => {}
                BoundWeight::Fp8(w) => known.extend([w.weight.0, w.row_scale.0]),
            }
        }
    }
    known.extend(&f.w8a8_ptrs);
    known.extend([f.head.final_norm.weight.0, 0x9100_0000]);
    for m in [
        f.fixed.meta,
        f.fixed.batch_meta,
        f.fixed.verify_meta,
        f.fixed.verify_batch_meta,
    ] {
        known.extend([m.positions.0, m.slot.0, m.seq_len.0, m.block_table.0]);
    }
    known.insert(f.fixed.ffn_act_q8.0);
    known.insert(f.fixed.verify_batch_tokens.0);
    if let Some(d) = &f.fixed.draft {
        known.extend([d.embed.0, d.k_pool.0, d.v_pool.0]);
        known.extend([
            d.meta.positions.0,
            d.meta.slot.0,
            d.meta.seq_len.0,
            d.meta.block_table.0,
        ]);
        // 2026-09-30: An n-row draft's batched metadata and the confidences it writes.
        if let (Some(r), Ok(m)) = (&d.rows, d.meta_rows(f.plan.rows)) {
            known.extend([m.positions.0, m.slot.0, m.seq_len.0, m.block_table.0]);
            known.insert(f.fixed.tokens.offset(r.lp_offset).0);
        }
    }
    let row = |dim: &str| f.circuit.dims[dim] * 2;
    let rows_of = [
        (f.fixed.hidden.0, row("hidden")),
        (f.fixed.residual.0, row("hidden")),
        (f.fixed.logits.0, row("vocab")),
        (f.fixed.tokens.0, 4),
    ];
    // 2026-09-30: The carried verify's tables, at any sequence's slice, and each GDN layer's WY
    // tables.
    let carry = super::exec_fixture::CARRY;
    let tables = |p: u64, base: u64, span: u64| (base..base + span).contains(&p);
    let wy = f.fixed.verify_wy_tables.0;
    let in_carry = |p: u64| {
        [carry.flag, carry.slot_tab, carry.conv_tab]
            .iter()
            .any(|b| tables(p, b.0, 128 * 8))
            || [carry.stash, carry.pend, carry.conv_stash]
                .iter()
                .any(|b| b.0 == p)
            || tables(
                p,
                wy,
                64 * crate::layer::VERIFY_WY_LAYER_STRIDE_BYTES as u64,
            )
    };
    let in_fixed = |p: u64| {
        rows_of
            .iter()
            .any(|&(base, w)| (0..f.plan.rows).any(|r| p == base + r * w))
    };
    known.extend(f.fixed.k_pools.iter().chain(&f.fixed.v_pools).map(|p| p.0));
    known.extend(gdn.iter().flatten().flat_map(|s| {
        [s.h, s.conv]
            .into_iter()
            .chain(s.h_steps)
            .chain(s.conv_steps)
            .map(|p| p.0)
    }));
    let kernels: Vec<&str> = kernel_launches(f).map(|l| l.kernel.as_str()).collect();
    for (i, l) in launched.iter().enumerate() {
        for a in &l.args {
            if let MockArg::Buffer(p) = a {
                let in_ws = (WORKSPACE..WORKSPACE + f.arena).contains(&p.0);
                assert!(
                    in_ws || known.contains(&p.0) || in_fixed(p.0) || in_carry(p.0) || p.0 == 0,
                    "launch {i} ({}) reads {:#x}, which is neither bound nor placed",
                    kernels[i],
                    p.0
                );
            }
        }
    }
}
