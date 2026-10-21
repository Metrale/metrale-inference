// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The single-row ceiling launch (`DsaSelectLaunch::Ceiling`, a replay-safe decode
//! row) has the same launch dimensions at every `--max-seq-len` from an 8k-token ceiling up:
//! `dsa_kpool_compress` and `dsa_index_scores` stop at `POOL_GRID_MAX` blocks and stride over
//! the live pools, the top-k and expansion are one block per row, and every shared-memory
//! request is capped. Below the cap the grids are the one-block-per-pool grids the kernels
//! had before they strode, and an exact launch is unchanged.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use super::launch::{index_scores_grid_x, kpool_compress_grid_x};
use super::passes_tests::{INDEX_SCORES, KPOOL_COMPRESS, cfg, inputs, kernels};
use super::*;
use crate::glm5next_dsa::levers::DsaLevers;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

/// 2026-10-14: (kernel, grid, block, shared memory) of every launch of one replay-safe
/// selection of a decode row at `live` tokens under a context ceiling of `msl` tokens.
fn ceiling_dims(msl: usize, live: usize) -> Vec<(u64, [u32; 3], [u32; 3], u32)> {
    ceiling_dims_with(&kernels(), msl, live)
}

/// 2026-10-21: [`ceiling_dims`] with the given kernels resolved.
fn ceiling_dims_with(
    k: &Glm5NextDsaKernels,
    msl: usize,
    live: usize,
) -> Vec<(u64, [u32; 3], [u32; 3], u32)> {
    let c = Glm5NextDsaConfig {
        max_context: msl,
        ..cfg()
    };
    let geom = DsaSelectGeometry::plan(&c, live, 1).unwrap();
    let gpu = MockGpuBackend::new();
    let inp = DsaSelectInputs {
        geom_dev: gpu.alloc(5 * 4).unwrap(),
        ..inputs(&gpu, &c, live, 1)
    };
    let scratch = DsaSelectScratch::alloc(&gpu, &c, &geom, DsaLevers::declared().select).unwrap();
    let launch = DsaSelectLaunch::Ceiling {
        max_pools: contiguous_pool_count(c.index_kpool, msl),
    };
    select_tokens(&gpu, k, &c, &geom, &inp, &scratch, launch, 7).unwrap();
    gpu.launches_snapshot()
        .iter()
        .map(|l| (l.func, l.grid, l.block, l.shared_mem))
        .collect()
}

/// 2026-10-14: The O(--max-seq-len) audit of the single-row path: at a fixed 4k live context
/// the launches at a 64k, 256k, 512k and 1M ceiling are identical, and the pool grids are at
/// the cap (before, they were the ceiling's pool count: 16,385 to 262,145 blocks).
#[test]
fn the_ceiling_launch_does_not_grow_with_max_seq_len() {
    let base = ceiling_dims(65_536, 4_096);
    assert_eq!(base.len(), 4, "compress, scores, top-k, expand");
    for msl in [262_144, 524_288, 1_048_576] {
        assert_eq!(ceiling_dims(msl, 4_096), base, "msl {msl}");
    }
    let grid_x = |f: u64| base.iter().find(|l| l.0 == f).unwrap().1[0] as usize;
    assert_eq!(grid_x(KPOOL_COMPRESS), POOL_GRID_MAX);
    assert_eq!(grid_x(INDEX_SCORES), POOL_GRID_MAX);
}

/// 2026-10-21: With the warp scores and the radix top-k resolved (the default-on decode
/// levers), a ceiling launch scores with `dsa_index_scores_warp` on at most
/// `SCORES_WARP_GRID_MAX` blocks and selects with `dsa_topk_pools_radix`, and its launches
/// still do not grow with --max-seq-len.
#[test]
fn the_decode_levers_take_the_ceiling_launch_at_a_fixed_size() {
    const WARP: u64 = 31;
    const RADIX: u64 = 32;
    let mut k = kernels();
    k.rows.scores_warp = metrale_gpu_runtime::gpu::KernelHandle(WARP);
    k.rows.topk_radix = metrale_gpu_runtime::gpu::KernelHandle(RADIX);
    let base = ceiling_dims_with(&k, 65_536, 4_096);
    let funcs: Vec<u64> = base.iter().map(|l| l.0).collect();
    assert_eq!(funcs, vec![KPOOL_COMPRESS, WARP, RADIX, funcs[3]], "compress, warp, radix, expand");
    assert!(!funcs.contains(&INDEX_SCORES), "the per-block scores never launch");
    let warp = base.iter().find(|l| l.0 == WARP).unwrap();
    assert_eq!(warp.1, [SCORES_WARP_GRID_MAX as u32, 1, 1]);
    assert_eq!((warp.2, warp.3), ([256, 1, 1], 0));
    for msl in [262_144, 524_288, 1_048_576] {
        assert_eq!(ceiling_dims_with(&k, msl, 4_096), base, "msl {msl}");
    }
    assert_eq!(scores_warp_grid_x(0), 1);
    assert_eq!(scores_warp_grid_x(9), 2);
    assert_eq!(scores_warp_grid_x(8 * SCORES_WARP_GRID_MAX + 1), SCORES_WARP_GRID_MAX as u32);
}

/// 2026-10-14: Under the cap a ceiling launch keeps the one-block-per-pool grids
/// (`max_pools + 1` compressions, `max_pools` scores), so a ceiling of up to 8k tokens launches
/// exactly as before; an exact launch is the live geometry's grid at any length.
#[test]
fn short_ceilings_and_exact_launches_keep_their_grids() {
    let c = cfg();
    let geom = DsaSelectGeometry::plan(&c, 4_093, 1).unwrap();
    assert_eq!((geom.n_pools_full, geom.n_pools), (1_024, 1_023));
    for m in [1, 16, 1_023, POOL_GRID_MAX - 1] {
        assert_eq!(kpool_compress_grid_x(&geom, Some(m)), (m + 1) as u32);
        assert_eq!(index_scores_grid_x(&geom, Some(m)), m as u32);
    }
    assert_eq!(
        kpool_compress_grid_x(&geom, Some(POOL_GRID_MAX)),
        POOL_GRID_MAX as u32
    );
    assert_eq!(
        index_scores_grid_x(&geom, Some(POOL_GRID_MAX)),
        POOL_GRID_MAX as u32
    );
    let long = DsaSelectGeometry::plan(&c, 65_536, 1).unwrap();
    assert_eq!(kpool_compress_grid_x(&long, None), 16_384);
    assert_eq!(index_scores_grid_x(&long, None), 16_384);
    assert_eq!(kpool_compress_grid_x(&geom, None), 1_024);
    assert_eq!(index_scores_grid_x(&geom, None), 1_023);
}

/// 2026-10-13: The capped grid is the pool count up to the cap and the cap beyond it.
#[test]
fn the_pool_grid_follows_the_pools_up_to_the_cap_only() {
    assert_eq!(pool_grid_x(1), 1);
    assert_eq!(pool_grid_x(16), 16);
    assert_eq!(pool_grid_x(2_048), 2_048);
    assert_eq!(pool_grid_x(2_049), 2_048);
    for msl in [65_536usize, 262_144, 524_288, 1_048_576] {
        assert_eq!(pool_grid_x(msl / 4), 2_048, "msl {msl}");
    }
}

/// 2026-10-14: On a GB10: a random 9,000-token cache (2,250 pools, past the 2,048-block cap,
/// so some blocks take two pools) and one decode row at its last token, selected exactly and
/// at a 512k-token ceiling (grid capped, live extents from `dsa_write_geom`). The compressed
/// pools, scores, candidacy and selected tokens are byte-identical.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA GB10 + the glm-5.3-flash kernel target"]
fn a_capped_ceiling_launch_writes_the_exact_launch_bytes() {
    use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
    use metrale_gpu_runtime::kernel_args::KernelLaunch;
    let set = metrale_kernels::ptx_for_exact_target("glm-5.3-flash", "nvfp4")
        .expect("glm-5.3-flash/nvfp4 not in this build");
    let gpu = MetraleCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let c = Glm5NextDsaConfig {
        max_context: 524_288,
        ..cfg()
    };
    let k = Glm5NextDsaKernels::resolve(&gpu).unwrap();
    let seq = 9_000usize;
    let geom = DsaSelectGeometry::plan(&c, seq, 1).unwrap();
    assert!(geom.n_pools > POOL_GRID_MAX);
    let inp = super::passes_tests::random_inputs(&gpu, &c, seq, &[seq as i32 - 1]);
    let read = |s: &DsaSelectScratch| -> Vec<Vec<u8>> {
        let d = c.index_head_dim;
        [
            (s.pool_keys, geom.n_pools_full * d * 4),
            (s.pool_indices, geom.n_pools_full * c.index_kpool * 4),
            (s.pool_valid, geom.n_pools_full),
            (s.scores, geom.n_pools * 4),
            (s.valid_cand, geom.n_pools),
            (s.tokens, c.out_width() * 4),
        ]
        .iter()
        .map(|&(p, n)| {
            let mut out = vec![0u8; n];
            gpu.copy_d2h(p, &mut out).unwrap();
            out
        })
        .collect()
    };
    let exact = DsaSelectScratch::alloc(&gpu, &c, &geom, DsaLevers::declared().select).unwrap();
    select_tokens(&gpu, &k, &c, &geom, &inp, &exact, DsaSelectLaunch::Exact, 0).unwrap();

    let geom_dev = gpu.alloc(5 * 4).unwrap();
    let sl = gpu.alloc(4).unwrap();
    gpu.copy_h2d(&(seq as i32).to_le_bytes(), sl).unwrap();
    KernelLaunch::new(&gpu, k.write_geom)
        .grid([1, 1, 1])
        .block([1, 1, 1])
        .arg_ptr(sl)
        .arg_ptr(geom_dev)
        .arg_u32(c.index_kpool as u32)
        .arg_u32(c.index_topk as u32)
        .arg_u32(topk_tile() as u32)
        .launch(0)
        .unwrap();
    let ceiling = DsaSelectScratch::alloc(&gpu, &c, &geom, DsaLevers::declared().select).unwrap();
    let launch = DsaSelectLaunch::Ceiling {
        max_pools: contiguous_pool_count(c.index_kpool, c.max_context),
    };
    let inp_c = DsaSelectInputs { geom_dev, ..inp };
    select_tokens(&gpu, &k, &c, &geom, &inp_c, &ceiling, launch, 0).unwrap();
    gpu.synchronize(0).unwrap();

    let (a, b) = (read(&exact), read(&ceiling));
    assert!(
        a[5].chunks(4)
            .any(|t| i32::from_le_bytes([t[0], t[1], t[2], t[3]]) >= 0),
        "something was selected"
    );
    for (i, name) in [
        "pool keys",
        "pool indices",
        "pool valid",
        "scores",
        "candidacy",
        "tokens",
    ]
    .iter()
    .enumerate()
    {
        assert_eq!(
            a[i], b[i],
            "{name} differ between the exact and the capped ceiling launch"
        );
    }
}
