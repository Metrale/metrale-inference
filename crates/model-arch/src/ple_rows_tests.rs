// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: GPU parity of the PLE multi-sequence step (`PleLayer::forward_rows`)
//! against the per-row step (`forward_row`) it replaces: on random weights, a
//! small resident n-gram table and random highway rows, both forms must leave
//! byte-identical highway rows, conv states and token histories, over two
//! consecutive steps (the second exercises the carried conv state and history).
//!
//! The projections are the only step whose kernel sees more than one row at a
//! time; it has no row-dependent reduction, and this is the test that holds it
//! to that.
//!
//! Owner: model-arch (Qwen3.8-Flash-Next PLE).
//! Invariants: none beyond the types.
//!
//! GPU test: `#[ignore]`. Run with
//! ```text
//! cargo test -p metrale-model-arch --release ple_rows -- --ignored --nocapture
//! ```
//! on a host whose build has the `qwen3.8-flash-next/nvfp4` kernel set.

use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layer::{ForwardContext, MoeLoraRoute};
use metrale_model_layers::layers::ngram_embed::NgramTable;
use metrale_model_layers::layers::ple::{PleIdDims, PleLayer, PleSeqState, PleWeights};
use metrale_model_layers::weight_map::DenseWeight;

const HC: usize = 4;
const HEAD_DIM: usize = 16;
const HIDDEN: usize = 64;
const K_SIZE: usize = 4;
const NGRAM: usize = 3;
const HEADS_PER_NGRAM: usize = 2;
const ROWS: usize = 6;

/// 2026-10-13: A deterministic generator (SplitMix64) so the test needs no crate.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
    fn bf16s(&mut self, n: usize, scale: f32) -> Vec<u8> {
        (0..n)
            .flat_map(|_| {
                let bits = (self.unit() * scale).to_bits();
                ((bits >> 16) as u16).to_le_bytes()
            })
            .collect()
    }
    fn f32s(&mut self, n: usize, scale: f32) -> Vec<u8> {
        (0..n)
            .flat_map(|_| (self.unit() * scale).to_le_bytes())
            .collect()
    }
}

fn upload(g: &dyn GpuBackend, b: &[u8]) -> DevicePtr {
    let p = g.alloc(b.len().max(256)).unwrap();
    g.copy_h2d(b, p).unwrap();
    p
}

fn download(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<u8> {
    let mut raw = vec![0u8; n];
    g.copy_d2h(p, &mut raw).unwrap();
    raw
}

fn dims() -> PleIdDims {
    let heads = (NGRAM - 1) * HEADS_PER_NGRAM;
    let vocab = 251u64;
    PleIdDims {
        ngram_size: NGRAM,
        heads_per_ngram: HEADS_PER_NGRAM,
        multipliers: vec![
            0x9E37_79B9_7F4A_7C15,
            0xBF58_476D_1CE4_E5B9,
            0x94D0_49BB_1331_11EB,
        ],
        head_vocab_sizes: vec![vocab; heads],
        head_offsets: (0..heads as u64).map(|h| h * vocab).collect(),
        eos_token_id: 2,
    }
}

fn layer(g: &dyn GpuBackend, rng: &mut Rng) -> PleLayer {
    let d = dims();
    let heads = d.ngram_heads();
    assert_eq!(heads * HEAD_DIM, HIDDEN);
    let c = HC * HIDDEN;
    let table_rows = heads * 251;
    let weights = PleWeights {
        key_proj: DenseWeight {
            weight: upload(g, &rng.bf16s(c * HIDDEN, 0.2)),
        },
        value_proj: DenseWeight {
            weight: upload(g, &rng.bf16s(HIDDEN * HIDDEN, 0.2)),
        },
        norm_key: DenseWeight {
            weight: upload(g, &rng.bf16s(c, 0.5)),
        },
        norm_query: DenseWeight {
            weight: upload(g, &rng.bf16s(c, 0.5)),
        },
        norm_conv: DenseWeight {
            weight: upload(g, &rng.bf16s(c, 0.5)),
        },
        conv1d: DenseWeight {
            weight: upload(g, &rng.bf16s(c * K_SIZE, 0.5)),
        },
    };
    let table = NgramTable::Bf16(DenseWeight {
        weight: upload(g, &rng.bf16s(table_rows * HEAD_DIM, 1.0)),
    });
    PleLayer::new(
        d, HEAD_DIM, HIDDEN, HC, K_SIZE, NGRAM, 1e-6, weights, table, 64, g,
    )
    .expect("PleLayer")
}

#[test]
#[ignore]
fn ple_rows_matches_per_row() {
    let set = metrale_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("qwen3.8-flash-next/nvfp4 not in this build");
    let gpu = metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
        .expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let mut rng = Rng(7);
    let ple = layer(g, &mut rng);
    let c = HC * HIDDEN;

    let config = metrale_config::ModelConfig::qwen3_next_80b_nvfp4();
    let buffers =
        metrale_gpu_runtime::buffers::BufferArena::new(&config, 16, 256, 4, 8, g).unwrap();
    let dispatch = metrale_model_layers::layers::ops::GemmDispatch::defaults();
    let derived = metrale_model_layers::layers::ops::DerivedWeights::new();
    let levers = metrale_model_layers::layers::ops::ModelLevers::defaults();
    let stats = metrale_model_layers::layers::ops::ModelStats::new();
    let ctx = ForwardContext {
        dispatch: &dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
        buffers: &buffers,
        hc_row_offset: 0,
        gpu: g,
        config: &config,
        attn_metadata: None,
        profile: false,
        comm: None,
        graph_capture: false,
        decode_step: true,
        gdn_exact_replay: false,
        gdn_write_on_accept: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: MoeLoraRoute::Fold,
    };

    // 2026-10-13: Two steps of tokens per row; the same highway rows feed both arms.
    let steps: Vec<Vec<u32>> = (0..2)
        .map(|_| (0..ROWS).map(|_| (rng.next() % 1000) as u32).collect())
        .collect();
    let highway_init: Vec<Vec<u8>> = (0..2).map(|_| rng.f32s(ROWS * c, 1.0)).collect();

    let mut per_row: Vec<PleSeqState> = (0..ROWS).map(|_| ple.new_seq_state(g).unwrap()).collect();
    let mut batched: Vec<PleSeqState> = (0..ROWS).map(|_| ple.new_seq_state(g).unwrap()).collect();
    let hw_a = upload(g, &highway_init[0]);
    let hw_b = upload(g, &highway_init[0]);

    for (step, tokens) in steps.iter().enumerate() {
        if step == 1 {
            g.copy_h2d(&highway_init[1], hw_a).unwrap();
            g.copy_h2d(&highway_init[1], hw_b).unwrap();
        }
        for (i, st) in per_row.iter_mut().enumerate() {
            ple.forward_row(st, hw_a.offset(i * c * 4), &tokens[i..i + 1], &ctx, stream)
                .unwrap();
        }
        let mut carries: Vec<&mut PleSeqState> = batched.iter_mut().collect();
        ple.forward_rows(&mut carries, hw_b, tokens, &ctx, stream)
            .unwrap();
        g.synchronize(stream).unwrap();

        let a = download(g, hw_a, ROWS * c * 4);
        let b = download(g, hw_b, ROWS * c * 4);
        assert!(
            a != highway_init[step.min(1)],
            "step {step}: the per-row arm changed nothing"
        );
        assert_eq!(
            a, b,
            "step {step}: highway rows differ between per-row and batched PLE"
        );
        for (i, (p, q)) in per_row.iter().zip(&batched).enumerate() {
            let sp = ple.snapshot_aux(p, g, stream).unwrap();
            let sq = ple.snapshot_aux(q, g, stream).unwrap();
            assert_eq!(
                sp, sq,
                "step {step}: carry (history + conv state) of row {i} differs"
            );
        }
        println!(
            "step {step}: {ROWS} rows byte-identical ({} highway bytes)",
            a.len()
        );
    }
}
