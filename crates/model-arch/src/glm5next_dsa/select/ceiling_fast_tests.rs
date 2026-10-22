// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: The launches of a ceiling selection over persisted pool keys under the
//! lane-per-head scores (`METRALE_GLM_DSA_SCORES_HEADS`) and the multi-CTA top-k
//! (`METRALE_GLM_DSA_TOPK_MC`): which kernels run, their grids and the arguments a replayed
//! graph depends on, and every condition that keeps the previous kernels. 2026-10-22: And the
//! lane-per-head scores' straight-line walk (`METRALE_GLM_DSA_SCORES_HEADS_WINDOW`).
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use super::passes_tests::{cfg, inputs, kernels};
use super::scores::SCORES_WINDOW_MIN_ROWS;
use super::topk::{TOPK_MC_CTAS, topk_mc_ctas, topk_mc_ws_words};
use super::*;
use crate::glm5next_dsa::levers::DsaLevers;
use metrale_gpu_runtime::gpu::KernelHandle;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};

const RADIX: u64 = 32;
const PAGED: u64 = 33;
const EXPAND_ROWS: u64 = 34;
const HEADS: u64 = 35;
const MC_PASS: u64 = 36;
const MC_COLLECT: u64 = 37;
const WINDOW: u64 = 38;
const SMS: u32 = 48;

fn fast_kernels() -> Glm5NextDsaKernels {
    let mut k = kernels();
    k.rows.topk_radix = KernelHandle(RADIX);
    k.rows.scores_warp_paged = KernelHandle(PAGED);
    k.rows.expand_selection = KernelHandle(EXPAND_ROWS);
    k.rows.scores_heads_paged = KernelHandle(HEADS);
    k.rows.scores_heads_window = KernelHandle(WINDOW);
    k.rows.topk_mc_pass = KernelHandle(MC_PASS);
    k.rows.topk_mc_collect = KernelHandle(MC_COLLECT);
    k.rows.sm_count = SMS;
    k
}

struct Case {
    rows: usize,
    launch: fn(usize) -> DsaSelectLaunch,
    pk_blk: u32,
    levers: DsaSelectLevers,
}

fn ceiling(max_pools: usize) -> DsaSelectLaunch {
    DsaSelectLaunch::Ceiling { max_pools }
}

fn exact(_: usize) -> DsaSelectLaunch {
    DsaSelectLaunch::Exact
}

impl Case {
    fn new(rows: usize) -> Self {
        Self {
            rows,
            launch: ceiling,
            pk_blk: 160,
            levers: DsaLevers::declared().select,
        }
    }

    /// 2026-10-21: (kernel, grid, args) of every launch of the pass at a 512k-token ceiling.
    fn run(&self, k: &Glm5NextDsaKernels) -> Vec<(u64, [u32; 3], Vec<MockArg>)> {
        let c = Glm5NextDsaConfig {
            max_context: 524_288,
            ..cfg()
        };
        let live = 4_097;
        let geom = DsaSelectGeometry::plan(&c, live, self.rows).unwrap();
        let sized = DsaSelectGeometry::plan(&c, c.max_context, self.rows).unwrap();
        let gpu = MockGpuBackend::new();
        let mut inp = DsaSelectInputs {
            geom_dev: gpu.alloc(self.rows * GEOM_SLOT_BYTES).unwrap(),
            ..inputs(&gpu, &c, live, self.rows)
        };
        inp.rows.pool_keys = gpu.alloc(64).unwrap();
        inp.rows.pk_blk = self.pk_blk;
        let scratch = DsaSelectScratch::alloc(&gpu, &c, &sized, self.levers).unwrap();
        let launch = (self.launch)(contiguous_pool_count(c.index_kpool, c.max_context));
        select_tokens(&gpu, k, &c, &geom, &inp, &scratch, launch, 7).unwrap();
        gpu.launches_snapshot()
            .iter()
            .map(|l| (l.func, l.grid, l.args.clone()))
            .collect()
    }
}

fn funcs(l: &[(u64, [u32; 3], Vec<MockArg>)]) -> Vec<u64> {
    l.iter().map(|l| l.0).collect()
}

const MAX_POOLS: usize = 524_288 / 4;

/// 2026-10-21: At the declared defaults a two-row ceiling pass scores with
/// `dsa_index_scores_heads_paged` on one block per SM (both rows in the block, so grid y is 1)
/// with exactly the arguments `dsa_index_scores_warp_paged` gets, then selects with four
/// `dsa_topk_mc_pass` launches (pass 0..3) and `dsa_topk_mc_collect` on
/// `topk_mc_ctas(max_pools)` x rows CTAs, then expands. The single-row pass launches the same.
/// 2026-10-22: The scores kernel at the defaults is `dsa_index_scores_heads_window` for two
/// rows and `dsa_index_scores_heads_paged` for one.
#[test]
fn the_default_ceiling_pass_runs_the_heads_scores_and_the_multi_cta_top_k() {
    let k = fast_kernels();
    for rows in [1usize, 2] {
        let l = Case::new(rows).run(&k);
        assert_eq!(
            funcs(&l),
            vec![
                if rows >= 2 { WINDOW } else { HEADS },
                MC_PASS,
                MC_PASS,
                MC_PASS,
                MC_PASS,
                MC_COLLECT,
                EXPAND_ROWS
            ],
            "rows {rows}"
        );
        assert_eq!(l[0].1, [SMS, 1, 1]);
        let c = topk_mc_ctas(MAX_POOLS) as u32;
        assert_eq!(c, 32);
        for (pass, launch) in l[1..5].iter().enumerate() {
            assert_eq!(launch.1, [c, rows as u32, 1]);
            assert_eq!(
                launch.2.last(),
                Some(&MockArg::Bytes((pass as u32).to_le_bytes().to_vec()))
            );
        }
        assert_eq!(l[5].1, [c, rows as u32, 1]);
        let mut off = Case::new(rows);
        off.levers.scores_heads = false;
        let w = off.run(&k);
        assert_eq!(w[0].0, PAGED);
        assert_eq!(w[0].2, l[0].2, "the heads kernel takes the warp kernel's arguments");
    }
}

/// 2026-10-21: Each lever's `0` restores the previous kernel alone: scores heads off ->
/// `dsa_index_scores_warp_paged` on its grid with the multi-CTA top-k kept; top-k MC off -> the
/// one-CTA radix with the heads scores kept.
#[test]
fn each_lever_off_restores_its_previous_kernel_only() {
    let k = fast_kernels();
    let mut heads_off = Case::new(2);
    heads_off.levers.scores_heads = false;
    let l = heads_off.run(&k);
    assert_eq!(funcs(&l)[..2], [PAGED, MC_PASS]);
    assert_eq!(l[0].1, [SCORES_WARP_GRID_MAX as u32, 2, 1]);
    let mut mc_off = Case::new(2);
    mc_off.levers.topk_mc = false;
    let l = mc_off.run(&k);
    assert_eq!(funcs(&l), vec![WINDOW, RADIX, EXPAND_ROWS]);
    assert_eq!(l[1].1, [2, 1, 1]);
}

/// 2026-10-21: The heads scores need the SM count, their kernel, 16-byte key copies (pk_blk a
/// multiple of 4) and at most 8 rows; the multi-CTA top-k needs both kernels, a ceiling launch
/// and the decode radix. Each missing condition keeps the previous kernel.
#[test]
fn the_fast_kernels_stand_down_where_their_conditions_fail() {
    let mut no_sms = fast_kernels();
    no_sms.rows.sm_count = 0;
    assert_eq!(Case::new(1).run(&no_sms)[0].0, PAGED);
    let mut no_heads = fast_kernels();
    no_heads.rows.scores_heads_paged = KernelHandle(0);
    assert_eq!(Case::new(1).run(&no_heads)[0].0, PAGED);
    let mut odd = Case::new(1);
    odd.pk_blk = 162;
    assert_eq!(odd.run(&fast_kernels())[0].0, PAGED, "pk_blk not a multiple of 4");
    let mut no_collect = fast_kernels();
    no_collect.rows.topk_mc_collect = KernelHandle(0);
    assert_eq!(funcs(&Case::new(1).run(&no_collect))[1], RADIX);
    let mut no_radix = Case::new(1);
    no_radix.levers.decode_topk_radix = false;
    let l = funcs(&no_radix.run(&fast_kernels()));
    assert!(!l.contains(&MC_PASS) && !l.contains(&RADIX), "{l:?}");
    let mut ex = Case::new(1);
    ex.launch = exact;
    ex.levers.topk_radix = true;
    let l = funcs(&ex.run(&fast_kernels()));
    assert!(
        l.contains(&RADIX) && !l.contains(&MC_PASS) && !l.contains(&HEADS) && !l.contains(&WINDOW),
        "{l:?}"
    );
}

/// 2026-10-22: From two rows (`SCORES_WINDOW_MIN_ROWS`) `METRALE_GLM_DSA_SCORES_HEADS_WINDOW=0`,
/// or its kernel unresolved, runs `dsa_index_scores_heads_paged` in its place with the same
/// grid and arguments; one row runs `dsa_index_scores_heads_paged` with the lever on; with the
/// lane-per-head scores off neither runs.
#[test]
fn the_window_walk_swaps_only_the_heads_kernel() {
    assert_eq!(SCORES_WINDOW_MIN_ROWS, 2);
    assert_eq!(Case::new(1).run(&fast_kernels())[0].0, HEADS);
    for rows in [2usize, 3] {
        let on = Case::new(rows).run(&fast_kernels());
        assert_eq!(on[0].0, WINDOW, "rows {rows}");
        let mut off = Case::new(rows);
        off.levers.scores_heads_window = false;
        let mut no_window = fast_kernels();
        no_window.rows.scores_heads_window = KernelHandle(0);
        for l in [off.run(&fast_kernels()), Case::new(rows).run(&no_window)] {
            assert_eq!(l[0].0, HEADS, "rows {rows}");
            assert_eq!((l[0].1, &l[0].2), (on[0].1, &on[0].2), "rows {rows}");
            assert_eq!(funcs(&l)[1..], funcs(&on)[1..], "rows {rows}");
        }
        let mut heads_off = Case::new(rows);
        heads_off.levers.scores_heads = false;
        assert_eq!(heads_off.run(&fast_kernels())[0].0, PAGED, "rows {rows}");
    }
}

/// 2026-10-21: The CTA count is one per 4,096 pools between 16 and 32 (a function of the
/// launch's pool count alone), and one row's workspace is `DSA_MC_WS_WORDS(C)` of
/// dsa_topk_mc.cu (restated: h0 C x 256, h 3 x 256, 2 counters, 512 gt values and pools, C eq
/// counts, C x 512 eq pools, 16 state words).
#[test]
fn the_multi_cta_geometry_is_fixed_by_the_pool_count() {
    assert_eq!(TOPK_MC_CTAS, (16, 32));
    for (pools, c) in [
        (0usize, 16usize),
        (1, 16),
        (65_536, 16),
        (65_537, 17),
        (131_072, 32),
        (262_144, 32),
    ] {
        assert_eq!(topk_mc_ctas(pools), c, "{pools} pools");
    }
    assert_eq!(topk_mc_ws_words(16), 16 * 256 + 768 + 2 + 1_024 + 16 + 16 * 512 + 16);
}
