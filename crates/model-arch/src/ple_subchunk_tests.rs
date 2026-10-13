// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: GPU parity of a PLE pass longer than the layer's scratch, which runs in
//! sub-chunks of `max_tokens` rows, against the same pass in one piece: on equal
//! random weights, a resident n-gram table and random highway rows, every sub-chunk
//! width (below, at and above the 9-row conv state) must leave byte-identical highway
//! rows, conv states and token histories after a fresh prefill, a continuing prefill
//! and a decode step that reads the carried state.
//!
//! Owner: model-arch (Qwen3.8-Flash-Next PLE).
//! Invariants: none beyond the types.
//!
//! GPU test: `#[ignore]`. Run with
//! ```text
//! cargo test -p metrale-model-arch --release ple_subchunk -- --ignored --nocapture
//! ```
//! on a host whose build has the `qwen3.8-flash-next/nvfp4` kernel set.

use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_layers::layers::ple::PleLayer;

use crate::ple_rows_tests::{CtxParts, HC, HIDDEN, Rng, backend, download, layer, upload};

/// 2026-10-13: Token counts of the three passes: a fresh prefill, a continuing
/// prefill and one decode token.
const PASSES: [(usize, bool); 3] = [(37, true), (11, false), (1, false)];

/// 2026-10-13: Run every pass through `ple` from a fresh carry; per pass, the
/// highway rows it leaves and its carry blob.
fn run(
    g: &dyn GpuBackend,
    parts: &CtxParts,
    ple: &PleLayer,
    passes: &[(Vec<u32>, Vec<u8>, bool)],
) -> Vec<(Vec<u8>, Vec<u8>)> {
    let stream = g.default_stream();
    let mut st = ple.new_seq_state(g).unwrap();
    let out = passes
        .iter()
        .map(|(tokens, init, fresh)| {
            let n = tokens.len();
            let ctx = parts.ctx(g, Some(tokens), n == 1);
            let hw = upload(g, init);
            ple.forward(&mut st, hw, n, *fresh, &ctx, stream).unwrap();
            g.synchronize(stream).unwrap();
            let rows = download(g, hw, init.len());
            g.free(hw).unwrap();
            assert_ne!(&rows, init, "PLE changed nothing");
            (rows, ple.snapshot_aux(&st, g, stream).unwrap())
        })
        .collect();
    ple.release_seq_state(&mut st, g).unwrap();
    out
}

#[test]
#[ignore]
fn a_sub_chunked_pass_equals_the_one_piece_pass() {
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let parts = CtxParts::new(g);
    let c = HC * HIDDEN;
    let mut rng = Rng(23);
    let passes: Vec<(Vec<u32>, Vec<u8>, bool)> = PASSES
        .iter()
        .map(|&(t, fresh)| {
            let tokens = (0..t).map(|_| (rng.next() % 1000) as u32).collect();
            (tokens, rng.f32s(t * c, 1.0), fresh)
        })
        .collect();

    // 2026-10-13: Every arm's weights are drawn from the same seed.
    let want = run(g, &parts, &layer(g, &mut Rng(11), 64), &passes);
    for width in [4, 9, 16] {
        let got = run(g, &parts, &layer(g, &mut Rng(11), width), &passes);
        for (p, (w, h)) in want.iter().zip(&got).enumerate() {
            assert_eq!(h.0, w.0, "width {width}, pass {p}: highway rows differ");
            assert_eq!(
                h.1, w.1,
                "width {width}, pass {p}: carry (history + conv state) differs"
            );
        }
        println!("width {width}: {} passes byte-identical", passes.len());
    }
}
