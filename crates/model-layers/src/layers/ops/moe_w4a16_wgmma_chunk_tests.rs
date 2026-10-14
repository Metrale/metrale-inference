// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The chunk-invariance gate of the routed prefill's wgmma lever
//! (METRALE_MOE_PREFILL_WGMMA with METRALE_MOE_PREFILL_WGMMA_MIN_ROWS): the same tokens, prefilled
//! in chunks of 2048, in chunks of 4096 and in one shot with the lever on, must give the bytes of
//! the lever-off one-shot prefill, for every minimum the crossover sweep can pick.
//!
//! The lever routes each call by its own routed rows (`ops::moe_w4a16_wgmma_routes`), so one
//! prompt's chunks can mix the wgmma and persistent twins; this proves that mix does not change
//! a byte. It runs at the operator level, gate/up (N 640, K 2560, sorted-token gather) feeding
//! down (N 2560, K 640) on the model's routing (512 experts, top-10, routing fixed per token),
//! because the full model is not chunk-invariant today with the lever off either (attention twins
//! at chunk 0 and GDN chunks counted from the chunk start). The serve-level counterpart is the
//! lever-on vs lever-off differential at --max-prefill-tokens 2048 and 4096.
//!
//! GPU test: run with
//! ```text
//! cargo test -p metrale-model-layers --release moe_w4a16_wgmma_chunk_tests -- --ignored --nocapture --test-threads 1
//! ```
//! on a Hopper host whose build has the `qwen3.8-flash-next/nvfp4` kernel set.

use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

use super::moe_w4a16_pipe_tests::{
    Arm, EXPERTS, TOP_K, Weights, backend, bf16s, kernels, problem_from, routes, run, weights,
};
use super::moe_w4a16_wgmma_tests::{SWEEP_TOKENS, wgmma_kernel};
use crate::layers::ops;

/// 2026-10-14: Prompt tokens: one full 4096-token chunk and a ragged 1000-token tail.
const TOKENS: u32 = 5096;
/// 2026-10-14: Gate/up (N, K) and down (N, K) of the routed prefill.
const GATE_UP: (u32, u32) = (640, 2560);
const DOWN: (u32, u32) = (2560, 640);
/// 2026-10-14: Lever-on prefills: chunks of 2048, chunks of 4096, one shot.
const CHUNKS: [u32; 3] = [2048, 4096, TOKENS];

/// 2026-10-14: What every prefill runs on: the device, both projections' weights, the two twins
/// and the stream.
struct Prefill<'a> {
    g: &'a dyn GpuBackend,
    gate_up: &'a Weights,
    down: &'a Weights,
    persist: KernelHandle,
    wgmma: KernelHandle,
    stream: u64,
}

impl Prefill<'_> {
    /// 2026-10-14: The arm the MoE layer picks for a call of `total` routed rows at the minimum
    /// `min` (`None`: the lever is off).
    fn arm(&self, min: Option<u32>, total: u32) -> Arm {
        if min.is_some_and(|m| ops::moe_w4a16_wgmma_routes(total, m)) {
            Arm::Wgmma(self.wgmma, total)
        } else {
            Arm::Persist(self.persist, total)
        }
    }

    /// 2026-10-14: The down output of every (token, slot) in token-major order, from prefilling
    /// `a` (BF16 [TOKENS, 2560]) routed by `route` in chunks of `chunk`; also how many calls ran
    /// the wgmma twin.
    fn run(&self, a: &[u8], route: &[Vec<u32>], chunk: u32, min: Option<u32>) -> (Vec<u8>, u32) {
        let row_in = GATE_UP.1 as usize * 2;
        let row_out = DOWN.0 as usize * 2;
        let mut out = vec![0u8; (TOKENS * TOP_K) as usize * row_out];
        let mut wgmma_calls = 0;
        for start in (0..TOKENS).step_by(chunk as usize) {
            let end = (start + chunk).min(TOKENS);
            // 2026-10-14: (local token, slot) pairs bucketed by expert, tokens ascending.
            let mut buckets = vec![Vec::new(); EXPERTS as usize];
            for t in start..end {
                for (slot, &e) in route[t as usize].iter().enumerate() {
                    buckets[e as usize].push((t - start, slot as u32));
                }
            }
            let offsets = std::iter::once(0)
                .chain(buckets.iter().scan(0u32, |acc, b| {
                    *acc += b.len() as u32;
                    Some(*acc)
                }))
                .collect::<Vec<u32>>();
            let sorted = buckets.concat();
            let gather = sorted.iter().map(|&(t, _)| t).collect::<Vec<u32>>();
            let total = sorted.len() as u32;
            let chunk_a = &a[start as usize * row_in..end as usize * row_in];
            let p = problem_from(self.g, self.gate_up, chunk_a, &offsets, Some(&gather));
            let arm = self.arm(min, total);
            wgmma_calls += matches!(arm, Arm::Wgmma(..)) as u32;
            let h = run(self.g, &p, arm, GATE_UP, self.stream);
            p.free(self.g);
            let p = problem_from(self.g, self.down, &h, &offsets, None);
            let arm = self.arm(min, total);
            wgmma_calls += matches!(arm, Arm::Wgmma(..)) as u32;
            let d = run(self.g, &p, arm, DOWN, self.stream);
            p.free(self.g);
            for (row, &(t, slot)) in sorted.iter().enumerate() {
                let at = ((start + t) * TOP_K + slot) as usize * row_out;
                out[at..at + row_out].copy_from_slice(&d[row * row_out..][..row_out]);
            }
        }
        (out, wgmma_calls)
    }
}

#[test]
#[ignore]
fn wgmma_lever_prefill_is_chunk_invariant_and_matches_the_lever_off_one_shot() {
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let wgmma = wgmma_kernel(g);
    for (n, k) in [GATE_UP, DOWN] {
        assert!(
            ops::moe_w4a16_wgmma_admits(g, wgmma, n, k, EXPERTS),
            "N={n} K={k}"
        );
    }
    let (gate_up, down) = (
        weights(g, GATE_UP.0, GATE_UP.1, EXPERTS),
        weights(g, DOWN.0, DOWN.1, EXPERTS),
    );
    let prefill = Prefill {
        g,
        gate_up: &gate_up,
        down: &down,
        persist: kernels(g).persist,
        wgmma,
        stream: g.create_stream().unwrap(),
    };
    let a = bf16s((TOKENS * GATE_UP.1) as usize, 17);
    let route = routes(TOKENS, 0x5eed);
    let (want, off_calls) = prefill.run(&a, &route, TOKENS, None);
    assert_eq!(off_calls, 0, "the lever-off reference ran the wgmma twin");
    let row = DOWN.0 as usize * 2;
    let mut failures = Vec::new();
    let mut mixed = false;
    // 2026-10-14: Each prefill length of the crossover sweep as a minimum in routed rows, so the
    // gate covers whichever one the sweep picks.
    for min in SWEEP_TOKENS.map(|t| t * TOP_K) {
        for chunk in CHUNKS {
            let (got, calls) = prefill.run(&a, &route, chunk, Some(min));
            let differing = (0..want.len() / row)
                .filter(|&r| got[r * row..][..row] != want[r * row..][..row])
                .collect::<Vec<usize>>();
            println!(
                "min rows {min:5} chunk {chunk:4}: {calls} wgmma calls, {} of {} rows differ",
                differing.len(),
                want.len() / row
            );
            if chunk == TOKENS {
                assert_eq!(
                    calls, 2,
                    "min rows {min}: the one-shot prefill skipped the twin"
                );
            }
            mixed |= calls > 0 && calls < 2 * TOKENS.div_ceil(chunk);
            if let Some(&r) = differing.first() {
                failures.push(format!(
                    "min rows {min} chunk {chunk}: {} rows differ, first token {} slot {}",
                    differing.len(),
                    r as u32 / TOP_K,
                    r as u32 % TOP_K
                ));
            }
        }
    }
    gate_up.free(g);
    down.free(g);
    assert!(failures.is_empty(), "{failures:#?}");
    assert!(
        mixed,
        "no chunked prefill mixed the wgmma and persistent twins"
    );
}
