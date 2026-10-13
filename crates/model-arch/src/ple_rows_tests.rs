// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: GPU parity of the PLE multi-sequence step (`PleLayer::forward_rows`)
//! against the per-row step (`forward_row`) it replaces: on random weights, a
//! small resident n-gram table and random highway rows, both forms must leave
//! byte-identical highway rows, conv states and token histories, over two
//! consecutive steps (the second exercises the carried conv state and history).
//!
//! The host half of that step can also run first (`prestage_rows`), as before a
//! CUDA-graph replay: staged, the step must equal the unstaged one launched directly
//! and replayed from a graph captured once, and capture must refuse an unstaged or
//! partly staged batch.
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

pub(crate) const HC: usize = 4;
const HEAD_DIM: usize = 16;
pub(crate) const HIDDEN: usize = 64;
const K_SIZE: usize = 4;
/// 2026-10-13: Also the conv dilation, so the carried conv state is
/// `(K_SIZE - 1) * NGRAM` = 9 rows.
const NGRAM: usize = 3;
const HEADS_PER_NGRAM: usize = 2;
const ROWS: usize = 6;

/// 2026-10-13: A deterministic generator (SplitMix64) so the test needs no crate.
pub(crate) struct Rng(pub(crate) u64);
impl Rng {
    pub(crate) fn next(&mut self) -> u64 {
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
    pub(crate) fn f32s(&mut self, n: usize, scale: f32) -> Vec<u8> {
        (0..n)
            .flat_map(|_| (self.unit() * scale).to_le_bytes())
            .collect()
    }
}

pub(crate) fn upload(g: &dyn GpuBackend, b: &[u8]) -> DevicePtr {
    let p = g.alloc(b.len().max(256)).unwrap();
    g.copy_h2d(b, p).unwrap();
    p
}

pub(crate) fn download(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<u8> {
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

/// 2026-10-13: A PLE layer on random weights drawn from `rng`, its scratch sized for
/// `max_tokens` rows: two layers built from equal seeds hold equal weights.
pub(crate) fn layer(g: &dyn GpuBackend, rng: &mut Rng, max_tokens: usize) -> PleLayer {
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
        d, HEAD_DIM, HIDDEN, HC, K_SIZE, NGRAM, 1e-6, weights, table, max_tokens, g,
    )
    .expect("PleLayer")
}

/// 2026-10-13: The CUDA backend over the Flash-Next kernel set.
pub(crate) fn backend() -> metrale_gpu_runtime::cuda_backend::MetraleCudaBackend {
    let set = metrale_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("qwen3.8-flash-next/nvfp4 not in this build");
    metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
        .expect("CUDA backend")
}

/// 2026-10-13: What a [`ForwardContext`] borrows; PLE reads none of it but the GPU
/// and the host token ids.
pub(crate) struct CtxParts {
    config: metrale_config::ModelConfig,
    buffers: metrale_gpu_runtime::buffers::BufferArena,
    dispatch: metrale_model_layers::layers::ops::GemmDispatch,
    derived: metrale_model_layers::layers::ops::DerivedWeights,
    levers: metrale_model_layers::layers::ops::ModelLevers,
    stats: metrale_model_layers::layers::ops::ModelStats,
}

impl CtxParts {
    pub(crate) fn new(g: &dyn GpuBackend) -> Self {
        let config = metrale_config::ModelConfig::qwen3_next_80b_nvfp4();
        let buffers =
            metrale_gpu_runtime::buffers::BufferArena::new(&config, 16, 256, 4, 8, g).unwrap();
        Self {
            config,
            buffers,
            dispatch: metrale_model_layers::layers::ops::GemmDispatch::defaults(),
            derived: metrale_model_layers::layers::ops::DerivedWeights::new(),
            levers: metrale_model_layers::layers::ops::ModelLevers::defaults(),
            stats: metrale_model_layers::layers::ops::ModelStats::new(),
        }
    }

    pub(crate) fn ctx<'a>(
        &'a self,
        g: &'a dyn GpuBackend,
        host_token_ids: Option<&'a [u32]>,
        decode_step: bool,
    ) -> ForwardContext<'a> {
        ForwardContext {
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            buffers: &self.buffers,
            hc_row_offset: 0,
            gpu: g,
            config: &self.config,
            attn_metadata: None,
            profile: false,
            comm: None,
            graph_capture: false,
            decode_step,
            gdn_exact_replay: false,
            gdn_write_on_accept: false,
            token_ids: None,
            host_token_ids,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: MoeLoraRoute::Fold,
        }
    }
}

#[test]
#[ignore]
fn ple_rows_matches_per_row() {
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let mut rng = Rng(7);
    let ple = layer(g, &mut rng, 64);
    let c = HC * HIDDEN;
    let parts = CtxParts::new(g);
    let ctx = parts.ctx(g, None, true);

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

/// 2026-10-13: One arm of `staged_rows_match_unstaged_direct_and_replayed`: how the
/// batched step's host half reaches the kernels.
#[derive(Clone, Copy, Debug)]
enum Arm {
    /// `forward_rows` hashes and uploads itself.
    Unstaged,
    /// `prestage_rows`, then `forward_rows` launched directly.
    Staged,
    /// `prestage_rows`, then one CUDA graph captured at step 0 and replayed at step 1.
    Replayed,
}

/// 2026-10-13: Per step, the highway rows `arm` leaves and every row's carry blob.
fn run_arm(
    arm: Arm,
    g: &dyn GpuBackend,
    parts: &CtxParts,
    ple: &PleLayer,
    steps: &[Vec<u32>],
    highway_init: &[Vec<u8>],
) -> Vec<(Vec<u8>, Vec<Vec<u8>>)> {
    // 2026-10-13: A stream of its own, as capture needs one.
    let stream = g.create_stream().unwrap();
    let c = HC * HIDDEN;
    let mut states: Vec<PleSeqState> = (0..ROWS).map(|_| ple.new_seq_state(g).unwrap()).collect();
    let hw = upload(g, &highway_init[0]);
    let mut graph = None;
    let mut out = Vec::new();
    for (step, tokens) in steps.iter().enumerate() {
        g.copy_h2d(&highway_init[step], hw).unwrap();
        let mut carries: Vec<&mut PleSeqState> = states.iter_mut().collect();
        if !matches!(arm, Arm::Unstaged) {
            ple.prestage_rows(&mut carries, tokens, g, stream).unwrap();
        }
        match (arm, graph) {
            (Arm::Unstaged | Arm::Staged, _) => {
                let ctx = parts.ctx(g, None, true);
                ple.forward_rows(&mut carries, hw, tokens, &ctx, stream)
                    .unwrap();
            }
            (Arm::Replayed, None) => {
                let mut ctx = parts.ctx(g, None, true);
                ctx.graph_capture = true;
                g.begin_capture(stream).unwrap();
                let captured = ple.forward_rows(&mut carries, hw, tokens, &ctx, stream);
                let gr = g.end_capture(stream).unwrap();
                captured.unwrap();
                g.launch_graph(gr, stream).unwrap();
                graph = Some(gr);
            }
            (Arm::Replayed, Some(gr)) => {
                g.launch_graph(gr, stream).unwrap();
                carries.iter_mut().for_each(|st| ple.replayed(st));
            }
        }
        g.synchronize(stream).unwrap();
        let blobs = states
            .iter()
            .map(|st| ple.snapshot_aux(st, g, stream).unwrap())
            .collect();
        out.push((download(g, hw, ROWS * c * 4), blobs));
    }
    if let Some(gr) = graph {
        g.destroy_graph(gr).unwrap();
    }
    out
}

#[test]
#[ignore]
fn staged_rows_match_unstaged_direct_and_replayed() {
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let mut rng = Rng(11);
    let ple = layer(g, &mut rng, 64);
    let parts = CtxParts::new(g);
    let c = HC * HIDDEN;
    let steps: Vec<Vec<u32>> = (0..2)
        .map(|_| (0..ROWS).map(|_| (rng.next() % 1000) as u32).collect())
        .collect();
    let highway_init: Vec<Vec<u8>> = (0..2).map(|_| rng.f32s(ROWS * c, 1.0)).collect();

    let want = run_arm(Arm::Unstaged, g, &parts, &ple, &steps, &highway_init);
    for (step, (hw, _)) in want.iter().enumerate() {
        assert!(
            hw != &highway_init[step],
            "step {step}: PLE changed nothing"
        );
    }
    for arm in [Arm::Staged, Arm::Replayed] {
        let got = run_arm(arm, g, &parts, &ple, &steps, &highway_init);
        for (step, ((gh, gb), (wh, wb))) in got.iter().zip(&want).enumerate() {
            assert_eq!(gh, wh, "{arm:?} step {step}: highway rows differ");
            assert_eq!(gb, wb, "{arm:?} step {step}: a carry differs");
        }
        println!("{arm:?}: 2 steps x {ROWS} rows byte-identical to unstaged");
    }

    // 2026-10-13: Capture refuses an unstaged batch, and a batch staged in part fails
    // rather than hashing the unstaged rows a second time.
    let mut states: Vec<PleSeqState> = (0..ROWS).map(|_| ple.new_seq_state(g).unwrap()).collect();
    let hw = upload(g, &highway_init[0]);
    let mut ctx = parts.ctx(g, None, true);
    ctx.graph_capture = true;
    let mut carries: Vec<&mut PleSeqState> = states.iter_mut().collect();
    let err = ple
        .forward_rows(&mut carries, hw, &steps[0], &ctx, stream)
        .unwrap_err();
    assert!(format!("{err:#}").contains("prestage_rows"), "{err:#}");
    ple.prestage_rows(&mut carries[..ROWS - 1], &steps[0][..ROWS - 1], g, stream)
        .unwrap();
    let ctx = parts.ctx(g, None, true);
    let err = ple
        .forward_rows(&mut carries, hw, &steps[0], &ctx, stream)
        .unwrap_err();
    assert!(format!("{err:#}").contains("staged in part"), "{err:#}");
}
