// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: `select_tokens_in_passes` against `select_tokens`: the same launches when the
//! scratch holds every row; otherwise one compression, then sub-passes that cover every query
//! row exactly once with each per-row pointer at that row's offset and the temporaries at their
//! base. On a GB10 (ignored by default), the selected tokens of a forced multi-pass run equal
//! the one-pass run's, byte for byte.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use super::*;
use crate::glm5next_dsa::DsaRowsKernels;
use crate::glm5next_dsa::levers::DsaLevers;
use metrale_gpu_runtime::gpu::KernelHandle;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};

/// 2026-10-11: GLM-5.3's indexer geometry (`select/tests.rs`), with a 4,096-token context.
pub(crate) fn cfg() -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: 4096,
        index_heads: 32,
        index_head_dim: 128,
        index_kpool: 4,
        index_topk: 2048,
        always_select_tail: true,
        local_heads: 64,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 256,
        qk_rope_head_dim: 0,
        v_head_dim: 256,
        max_context: 4096,
    }
}

pub(crate) const KPOOL_COMPRESS: u64 = 11;
pub(crate) const INDEX_SCORES: u64 = 12;
pub(crate) const TOPK_POOLS: u64 = 13;
pub(crate) const EXPAND: u64 = 14;
/// 2026-10-15: The SP-split selection's payload fold and folded expansion (`sp_select`).
pub(crate) const SEL_FOLD: u64 = 21;
pub(crate) const EXPAND_FOLDED: u64 = 22;

/// 2026-10-11: Distinct handles, and no tiled-scores kernel, so the untiled path runs whatever
/// `METRALE_GLM_DSA_SCORES_TILED` says.
pub(crate) fn kernels() -> Glm5NextDsaKernels {
    let h = KernelHandle;
    Glm5NextDsaKernels {
        kpool_compress: h(KPOOL_COMPRESS),
        compact_pools: h(15),
        index_scores: h(INDEX_SCORES),
        topk_pools: h(TOPK_POOLS),
        expand_selection: h(EXPAND),
        k_norm: h(16),
        write_geom: h(17),
        indexer_store: h(18),
        topk_to_mask: h(19),
        mla_masked_attn: h(20),
        rows: DsaRowsKernels {
            store: h(0),
            write_geom: h(0),
            pool_scores: h(0),
            topk_pools: h(0),
            expand_selection: h(0),
            scores_tiled: h(0),
            scores_mma: h(0),
            topk_radix: h(0),
            scores_warp: h(0),
            scores_warp_paged: h(0),
            pool_keys_store: h(0),
            pool_prep: h(0),
            scores_pipe: h(0),
            sel_fold: h(SEL_FOLD),
            expand_folded: h(EXPAND_FOLDED),
            select_fused: h(0),
            scores_heads_paged: h(0),
            scores_heads_window: h(0),
            pool_keys_bf16: false,
            scores_mma_paged: h(0),
            topk_mc_pass: h(0),
            topk_mc_collect: h(0),
            sm_count: 0,
        },
        fp8: crate::glm5next_dsa::fp8_rows::DsaFp8Kernels::NONE,
    }
}

/// 2026-10-11: Flat-cache inputs for `q` query rows over `seq` tokens, all zero.
pub(crate) fn inputs(
    gpu: &MockGpuBackend,
    c: &Glm5NextDsaConfig,
    seq: usize,
    q: usize,
) -> DsaSelectInputs {
    let (d, h) = (c.index_head_dim, c.index_heads);
    DsaSelectInputs {
        rows: super::super::paged::IndexerRowsDev::flat(
            gpu.alloc(seq * d * 2).unwrap(),
            gpu.alloc(seq * d * 2).unwrap(),
            gpu.alloc(seq).unwrap(),
        ),
        ape: gpu.alloc(c.index_kpool * d * 4).unwrap(),
        q: gpu.alloc(q * h * d * 4).unwrap(),
        weights: gpu.alloc(q * h * 4).unwrap(),
        q_pos: gpu.alloc(q * 4).unwrap(),
        q_mask: gpu.alloc(q).unwrap(),
        first_key: 0,
        geom_dev: DevicePtr::NULL,
    }
}

fn u32_arg(v: u32) -> MockArg {
    MockArg::Bytes(v.to_le_bytes().to_vec())
}

fn of(launches: &[MockLaunch], func: u64) -> Vec<MockLaunch> {
    launches
        .iter()
        .filter(|l| l.func == func)
        .cloned()
        .collect()
}

/// 2026-10-11: A scratch planned for all 8 rows: `select_tokens_in_passes` reports one pass and
/// issues exactly the launches `select_tokens` does (kernel, grid, block, shared memory and
/// every argument), on two backends that allocate the same addresses.
#[test]
fn a_scratch_that_holds_every_row_runs_the_one_pass_launches() {
    let c = cfg();
    let geom = DsaSelectGeometry::plan(&c, 4096, 8).unwrap();
    let run = |passes: bool| {
        let gpu = MockGpuBackend::new();
        let inp = inputs(&gpu, &c, 4096, 8);
        let scratch =
            DsaSelectScratch::alloc(&gpu, &c, &geom, DsaLevers::declared().select).unwrap();
        if passes {
            let n =
                select_tokens_in_passes(&gpu, &kernels(), &c, &geom, &inp, &scratch, 7).unwrap();
            assert_eq!(n, 1);
        } else {
            select_tokens(
                &gpu,
                &kernels(),
                &c,
                &geom,
                &inp,
                &scratch,
                DsaSelectLaunch::Exact,
                7,
            )
            .unwrap();
        }
        gpu.launches_snapshot()
    };
    let (one, passes) = (run(false), run(true));
    assert_eq!(one.len(), 4, "compress, scores, top-k, expand");
    assert_eq!(one.len(), passes.len());
    for (a, b) in one.iter().zip(&passes) {
        assert_eq!(
            (a.func, a.grid, a.block, a.shared_mem, a.stream, &a.args),
            (b.func, b.grid, b.block, b.shared_mem, b.stream, &b.args)
        );
    }
}

/// 2026-10-11: Per-row regions for 3 rows and an output for 8: three sub-passes of 3, 3 and 2
/// rows after ONE compression. Every per-row pointer is the base plus the sub-pass's first row
/// at the stride the kernels index (query `H * D` f32, head weights `H` f32, `q_pos` i32,
/// `q_mask` u8, output `out_width` i32); the scores, candidacy and selected regions start at
/// their base in every pass; the row-count argument is the sub-pass's.
#[test]
fn sub_passes_cover_every_row_once_at_that_rows_offsets() {
    let c = cfg();
    let geom = DsaSelectGeometry::plan(&c, 4096, 8).unwrap();
    let gpu = MockGpuBackend::new();
    let inp = inputs(&gpu, &c, 4096, 8);
    let small = DsaSelectGeometry { q_rows: 3, ..geom };
    let scratch =
        DsaSelectScratch::alloc_rows(&gpu, &c, &small, 8, DsaLevers::declared().select).unwrap();
    assert_eq!(scratch.rows_per_pass(&geom), 3);
    let n = select_tokens_in_passes(&gpu, &kernels(), &c, &geom, &inp, &scratch, 7).unwrap();
    assert_eq!(n, 3);
    let all = gpu.launches_snapshot();
    assert_eq!(
        all[0].func, KPOOL_COMPRESS,
        "the pools are compressed first"
    );
    assert_eq!(of(&all, KPOOL_COMPRESS).len(), 1, "and only once");
    let (scores, topk, expand) = (
        of(&all, INDEX_SCORES),
        of(&all, TOPK_POOLS),
        of(&all, EXPAND),
    );
    assert_eq!((scores.len(), topk.len(), expand.len()), (3, 3, 3));
    let (h, d, ow) = (c.index_heads, c.index_head_dim, c.out_width());
    let buf = MockArg::Buffer;
    let mut covered = 0;
    for (j, (r0, m)) in [(0usize, 3u32), (3, 3), (6, 2)].into_iter().enumerate() {
        let s = &scores[j];
        assert_eq!(s.grid, [geom.n_pools as u32, m, 1]);
        assert_eq!(s.args[0], buf(inp.q.offset(r0 * h * d * 4)));
        assert_eq!(s.args[2], buf(inp.weights.offset(r0 * h * 4)));
        assert_eq!(s.args[6], buf(inp.q_pos.offset(r0 * 4)));
        assert_eq!(s.args[7], buf(scratch.scores));
        assert_eq!(s.args[8], buf(scratch.valid_cand));
        assert_eq!(s.args[9], u32_arg(m));
        let t = &topk[j];
        assert_eq!(t.grid, [m, 1, 1]);
        assert_eq!(
            (&t.args[0], &t.args[1]),
            (&buf(scratch.scores), &buf(scratch.selected))
        );
        assert_eq!(t.args[2], u32_arg(m));
        let e = &expand[j];
        assert_eq!(e.grid, [m, 1, 1]);
        assert_eq!(e.args[0], buf(scratch.selected));
        assert_eq!(e.args[2], buf(scratch.valid_cand));
        assert_eq!(e.args[4], buf(inp.q_pos.offset(r0 * 4)));
        assert_eq!(e.args[5], buf(inp.q_mask.offset(r0)));
        assert_eq!(e.args[6], buf(scratch.tokens.offset(r0 * ow * 4)));
        assert_eq!(e.args[7], u32_arg(m));
        covered += m as usize;
    }
    assert_eq!(covered, 8);
}

/// 2026-10-11: The output must hold every row of the group: `alloc_rows` refuses fewer output
/// rows than the pass, and the sub-pass launcher refuses a group larger than the output before
/// launching anything.
#[test]
fn the_output_must_hold_every_row() {
    let c = cfg();
    let geom = DsaSelectGeometry::plan(&c, 4096, 8).unwrap();
    let gpu = MockGpuBackend::new();
    assert!(
        DsaSelectScratch::alloc_rows(&gpu, &c, &geom, 7, DsaLevers::declared().select).is_err()
    );
    let small = DsaSelectGeometry { q_rows: 3, ..geom };
    let scratch =
        DsaSelectScratch::alloc_rows(&gpu, &c, &small, 4, DsaLevers::declared().select).unwrap();
    let inp = inputs(&gpu, &c, 4096, 8);
    let err = select_tokens_in_passes(&gpu, &kernels(), &c, &geom, &inp, &scratch, 7);
    assert!(err.is_err());
    assert!(
        gpu.launches_snapshot().is_empty(),
        "refused before any launch"
    );
}

/// 2026-10-11: Fewer tokens than one pool: nothing to score, so a 1-row scratch still runs
/// the group in one pass (compression and expansion only).
#[test]
fn no_pools_is_one_pass() {
    let c = cfg();
    let geom = DsaSelectGeometry::plan(&c, 3, 8).unwrap();
    assert_eq!(geom.n_pools, 0);
    let gpu = MockGpuBackend::new();
    let inp = inputs(&gpu, &c, 3, 8);
    let tiny = DsaSelectGeometry::plan(&c, 4096, 1).unwrap();
    let scratch =
        DsaSelectScratch::alloc_rows(&gpu, &c, &tiny, 8, DsaLevers::declared().select).unwrap();
    assert_eq!(scratch.rows_per_pass(&geom), usize::MAX);
    assert_eq!(
        select_tokens_in_passes(&gpu, &kernels(), &c, &geom, &inp, &scratch, 7).unwrap(),
        1
    );
    let funcs: Vec<u64> = gpu.launches_snapshot().iter().map(|l| l.func).collect();
    assert_eq!(funcs, vec![KPOOL_COMPRESS, EXPAND]);
}

/// 2026-10-11: On a GB10: random indexer cache (5,000 tokens), 64 query rows at spread
/// positions, selected once with a scratch for all 64 rows and once with per-row regions for 5
/// (13 sub-passes); the `[64, out_width]` token ids are byte-identical.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA GB10 + the glm-5.3-flash kernel target"]
fn sub_passes_select_the_one_pass_tokens_bit_for_bit() {
    use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
    let set = metrale_kernels::ptx_for_exact_target("glm-5.3-flash", "nvfp4")
        .expect("glm-5.3-flash/nvfp4 not in this build");
    let gpu = MetraleCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let c = Glm5NextDsaConfig {
        max_context: 8192,
        ..cfg()
    };
    let k = Glm5NextDsaKernels::resolve(&gpu).unwrap();
    let (seq, q) = (5000usize, 64usize);
    let pos: Vec<i32> = (0..q).map(|r| (seq - 1 - r * 61) as i32).collect();
    let inp = random_inputs(&gpu, &c, seq, &pos);
    let geom = DsaSelectGeometry::plan(&c, seq, q).unwrap();
    let tokens = |scratch: &DsaSelectScratch| {
        let mut out = vec![0u8; q * c.out_width() * 4];
        gpu.copy_d2h(scratch.tokens(), &mut out).unwrap();
        out
    };
    let whole = DsaSelectScratch::alloc(&gpu, &c, &geom, DsaLevers::declared().select).unwrap();
    select_tokens(&gpu, &k, &c, &geom, &inp, &whole, DsaSelectLaunch::Exact, 0).unwrap();
    gpu.synchronize(0).unwrap();
    let parts = DsaSelectScratch::alloc_rows(
        &gpu,
        &c,
        &DsaSelectGeometry { q_rows: 5, ..geom },
        q,
        DsaLevers::declared().select,
    )
    .unwrap();
    let n = select_tokens_in_passes(&gpu, &k, &c, &geom, &inp, &parts, 0).unwrap();
    gpu.synchronize(0).unwrap();
    assert_eq!(n, 13);
    let (a, b) = (tokens(&whole), tokens(&parts));
    assert!(
        a.chunks(4)
            .any(|t| i32::from_le_bytes([t[0], t[1], t[2], t[3]]) >= 0),
        "something was selected"
    );
    assert_eq!(
        a, b,
        "the sub-pass selection differs from the one-pass selection"
    );
}

/// 2026-10-14: A random flat indexer cache of `seq` tokens (all valid) and one random query
/// row per entry of `pos`, at that position, uploaded to `gpu`; the same xorshift stream on
/// every call, so two calls build the same inputs.
#[cfg(feature = "cuda")]
pub(super) fn random_inputs(
    gpu: &dyn GpuBackend,
    c: &Glm5NextDsaConfig,
    seq: usize,
    pos: &[i32],
) -> DsaSelectInputs {
    let (q, d, h) = (pos.len(), c.index_head_dim, c.index_heads);
    let mut x = 0x9e3779b97f4a7c15u64;
    let mut rnd = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let bf = |v: &[f32]| -> Vec<u8> {
        v.iter()
            .flat_map(|f| half::bf16::from_f32(*f).to_le_bytes())
            .collect()
    };
    let f4 = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };
    let up = |b: &[u8]| {
        let p = gpu.alloc(b.len()).unwrap();
        gpu.copy_h2d(b, p).unwrap();
        p
    };
    let kv: Vec<f32> = (0..seq * d).map(|_| rnd()).collect();
    let gate: Vec<f32> = (0..seq * d).map(|_| rnd()).collect();
    let ape: Vec<f32> = (0..c.index_kpool * d).map(|_| rnd() * 0.1).collect();
    let qv: Vec<f32> = (0..q * h * d).map(|_| rnd()).collect();
    let wv: Vec<f32> = (0..q * h).map(|_| rnd().abs() * 0.2).collect();
    let pos: Vec<u8> = pos.iter().flat_map(|p| p.to_le_bytes()).collect();
    DsaSelectInputs {
        rows: super::super::paged::IndexerRowsDev::flat(
            up(&bf(&kv)),
            up(&bf(&gate)),
            up(&vec![1u8; seq]),
        ),
        ape: up(&f4(&ape)),
        q: up(&f4(&qv)),
        weights: up(&f4(&wv)),
        q_pos: up(&pos),
        q_mask: up(&vec![1u8; q]),
        first_key: 0,
        geom_dev: DevicePtr::NULL,
    }
}
