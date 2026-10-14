// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: GPU tests of the wgmma twin `moe_w4a16_grouped_gemm_ptrtable_wgmma`
//! (kernels/hopper/common/moe_w4a16_grouped_wgmma_hopper.cu) against the base
//! `moe_w4a16_grouped_gemm_ptrtable` and the persistent pipelined twin, on the routed prefill's
//! two shapes (gate/up: N 640, K 2560 with the sorted-token gather; down: N 2560, K 640 without
//! it), reusing the pipelined twin's fixtures (`moe_w4a16_pipe_tests.rs`).
//!
//! - `..._is_byte_identical_...` decides the class: PASS means the wgmma twin writes the base
//!   kernel's bytes (class B) on uneven row counts around its 128-row tile, at its own grid and at
//!   a grid far smaller than the work, and on the model's routing (512 experts, top-10) at
//!   {1, 7, 20, 29, 80, 93, 256, 300, 4096} tokens. On a difference it prints how many elements
//!   differ and the largest gap in BF16 units in the last place before failing.
//! - `..._is_within_one_bf16_ulp_...` is the class-N bar: every output finite, written, and at most
//!   one BF16 ulp from the base kernel's (the two differ only in how a k16 step sums in FP32).
//! - `..._timing_...` prints microseconds per call of the persistent pipelined and wgmma twins on
//!   the routing shape at 300 and 4096 tokens, and the speedup.
//!
//! GPU test: run with
//! ```text
//! cargo test -p metrale-model-layers --release moe_w4a16_wgmma_tests -- --ignored --nocapture --test-threads 1
//! ```
//! on a Hopper host whose build has the `qwen3.8-flash-next/nvfp4` kernel set.

use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

use super::moe_w4a16_pipe_tests::{
    Arm, EXPERTS, Kernels, SENTINEL, backend, kernels, launch, problem, routed_rows, run, sentinel,
    time_us, weights,
};
use crate::layers::ops;

/// 2026-10-14: (N, K, gather): gate/up and down of the routed prefill.
const SHAPES: [(u32, u32, bool); 2] = [(640, 2560, true), (2560, 640, false)];
/// 2026-10-14: Rows per expert around the 128-row tile: empty, single, 127/128/129, several
/// tiles, and the pipe's 64-row boundary.
const ROWS: [u32; 8] = [0, 1, 127, 128, 129, 300, 7, 64];
/// 2026-10-14: Token rows of A the gather draws from.
const TOKENS: u32 = 97;
/// 2026-10-14: Prefill lengths on the routing shape: decode widths, short prompts, a full chunk.
const ROUTED_TOKENS: [u32; 9] = [1, 7, 20, 29, 80, 93, 256, 300, 4096];

fn wgmma_kernel(g: &dyn GpuBackend) -> KernelHandle {
    g.kernel(
        "moe_w4a16_grouped_wgmma_hopper",
        "moe_w4a16_grouped_gemm_ptrtable_wgmma",
    )
    .expect("the Hopper wgmma twin is not in this build")
}

/// 2026-10-14: A BF16 as an integer that orders like the value (sign-magnitude to two's
/// complement), so the distance of two finite BF16s in ulps is a difference.
fn bf16_ordinal(b: [u8; 2]) -> i32 {
    let v = u16::from_le_bytes(b) as i32;
    if v & 0x8000 != 0 { -(v & 0x7FFF) } else { v }
}

/// 2026-10-14: Elements of `got` that differ from `want`, and the largest distance in BF16 ulps
/// (`i32::MAX` when either side is not finite).
fn ulp_gap(want: &[u8], got: &[u8]) -> (usize, i32) {
    let finite = |b: &[u8]| u16::from_le_bytes([b[0], b[1]]) & 0x7F80 != 0x7F80;
    want.chunks_exact(2)
        .zip(got.chunks_exact(2))
        .filter(|(w, g)| w != g)
        .fold((0, 0), |(n, max), (w, g)| {
            let gap = if finite(w) && finite(g) {
                (bf16_ordinal([w[0], w[1]]) - bf16_ordinal([g[0], g[1]])).abs()
            } else {
                i32::MAX
            };
            (n + 1, max.max(gap))
        })
}

/// 2026-10-14: Every (problem, wgmma arm) pair the class tests compare with the base kernel: the
/// uneven rows at the twin's own grid and at the grid for one row (a few CTAs, each striding over
/// many units), then the model's routing at each of `ROUTED_TOKENS`. `check` gets the label, the
/// base bytes, the persistent pipelined bytes and the wgmma bytes.
fn for_each_case(check: &mut dyn FnMut(&str, &[u8], &[u8], &[u8])) {
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let kn: Kernels = kernels(g);
    let wg = wgmma_kernel(g);
    let s = g.create_stream().unwrap();
    for (n, k, gather) in SHAPES {
        assert!(
            ops::moe_w4a16_wgmma_admits(g, wg, n, k, EXPERTS),
            "N={n} K={k}"
        );
        let w = weights(g, n, k, ROWS.len() as u32);
        let p = problem(g, &w, k, gather, &ROWS, TOKENS);
        let want = run(g, &p, Arm::Base(kn.base, p.tiles), (n, k), s);
        let pipe = run(g, &p, Arm::Persist(kn.persist, p.total), (n, k), s);
        for rows in [p.total, 1] {
            let got = run(g, &p, Arm::Wgmma(wg, rows), (n, k), s);
            check(
                &format!("uneven N={n} K={k} gather={gather} grid rows={rows}"),
                &want,
                &pipe,
                &got,
            );
        }
        p.free(g);
        w.free(g);
        let w = weights(g, n, k, EXPERTS);
        for tokens in ROUTED_TOKENS {
            let p = problem(g, &w, k, gather, &routed_rows(tokens), tokens);
            let want = run(g, &p, Arm::Base(kn.base, p.tiles), (n, k), s);
            let pipe = run(g, &p, Arm::Persist(kn.persist, p.total), (n, k), s);
            let got = run(g, &p, Arm::Wgmma(wg, p.total), (n, k), s);
            check(
                &format!("routed T={tokens} N={n} K={k}"),
                &want,
                &pipe,
                &got,
            );
            p.free(g);
        }
        w.free(g);
    }
}

#[test]
#[ignore]
fn wgmma_w4a16_grouped_gemm_is_byte_identical_to_the_base_kernel() {
    let mut failures = Vec::new();
    for_each_case(&mut |label, want, pipe, got| {
        assert!(
            want == pipe,
            "{label}: the persistent pipelined twin differs from the base"
        );
        let (differing, max_ulps) = ulp_gap(want, got);
        println!(
            "{label}: {differing} of {} differ, max {max_ulps} ulp",
            want.len() / 2
        );
        if differing != 0 {
            failures.push(label.to_string());
        }
    });
    assert!(failures.is_empty(), "not byte-identical: {failures:?}");
}

#[test]
#[ignore]
fn wgmma_w4a16_grouped_gemm_is_within_one_bf16_ulp_of_the_base_kernel() {
    for_each_case(&mut |label, want, _pipe, got| {
        let unwritten = got
            .chunks_exact(2)
            .filter(|b| b == &[SENTINEL, SENTINEL])
            .count();
        let base_unwritten = want
            .chunks_exact(2)
            .filter(|b| b == &[SENTINEL, SENTINEL])
            .count();
        assert_eq!(unwritten, base_unwritten, "{label}: unwritten elements");
        let (differing, max_ulps) = ulp_gap(want, got);
        assert!(
            max_ulps <= 1,
            "{label}: {differing} differ, max {max_ulps} ulp"
        );
    });
}

#[test]
#[ignore]
fn wgmma_w4a16_grouped_gemm_timing_against_the_persistent_twin() {
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let kn = kernels(g);
    let wg = wgmma_kernel(g);
    let s = g.create_stream().unwrap();
    for (n, k, gather) in SHAPES {
        let w = weights(g, n, k, EXPERTS);
        for tokens in [300, 4096] {
            let p = problem(g, &w, k, gather, &routed_rows(tokens), tokens);
            let c = sentinel(g, (p.total * n) as usize * 2);
            let iters = if tokens >= 1024 { 20 } else { 100 };
            let persist = time_us(g, s, iters, &|| {
                launch(g, &p, Arm::Persist(kn.persist, p.total), (n, k), c, s)
            });
            let wgmma = time_us(g, s, iters, &|| {
                launch(g, &p, Arm::Wgmma(wg, p.total), (n, k), c, s)
            });
            println!(
                "T={tokens:5} N={n:4} K={k:4} rows={:6}: persistent {persist:9.1} us, wgmma \
                 {wgmma:9.1} us, {:.2}x",
                p.total,
                persist / wgmma
            );
            g.free(c).unwrap();
            p.free(g);
        }
        w.free(g);
    }
}
