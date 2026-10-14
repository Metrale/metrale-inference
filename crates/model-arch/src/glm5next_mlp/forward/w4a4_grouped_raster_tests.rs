// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The raster lever's launch plan (`METRALE_GLM_MOE_W4A4_RASTER`) on the recording
//! mock: at the m128 tile the fused gate/up and the down launch become their `_r` twins with the
//! same arguments plus the group; everything else (the narrow tiles, the separate gate and up
//! GEMMs, a group of 0) is the plain plan argument for argument. The bytes are the GPU bench's
//! job (`examples/glm5next_moe_epi_bench`).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use super::super::super::w4a4_grouped_raster::{RasterGroups, check_ready, parse_raster};
use super::super::w4a4_grouped_experts_with;
use super::*;

/// 2026-10-14: 1728 rows x top 8 over 288 experts = 48 slots per expert, the first m128 launch.
const M128_ROWS: usize = 1728;

/// 2026-10-14: The routed experts' launches over `rows` rows on a fresh mock (so two runs see
/// the same buffer addresses), the fused and raster levers as given.
fn plan(
    c: Glm5NextMlpConfig,
    rows: usize,
    fused: bool,
    raster: Option<RasterGroups>,
) -> Vec<MockLaunch> {
    let gpu = MockGpuBackend::new();
    let k = kernels(&gpu);
    let ws = Glm5NextMlpWorkspace::new_with(&gpu, &c, rows, true).unwrap();
    let w = moe_weights(usize::MAX);
    let site = MoeSite {
        gpu: &gpu,
        k: &k,
        cfg: &c,
        w: &w,
        x: X,
        rows,
        ws: &ws,
        stream: 5,
    };
    w4a4_grouped_experts_with(&site, SCALES, fused, raster).unwrap();
    gpu.launches_snapshot()
}

/// 2026-10-14: Two launch lists equal field for field (`MockLaunch` has no `PartialEq`).
fn assert_same(got: &[MockLaunch], want: &[MockLaunch]) {
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want) {
        assert_eq!(
            (g.func, g.grid, g.block, g.shared_mem, g.stream, &g.args),
            (w.func, w.grid, w.block, w.shared_mem, w.stream, &w.args)
        );
    }
}

fn funcs(l: &[MockLaunch]) -> Vec<u64> {
    l.iter().map(|l| l.func).collect()
}

/// 2026-10-14: `got` is `want` with one more trailing u32 argument, `group`; same grid, block,
/// shared memory and stream.
fn assert_twin(got: &MockLaunch, want: &MockLaunch, group: u32) {
    assert_eq!(got.args.len(), want.args.len() + 1);
    assert_eq!(&got.args[..want.args.len()], &want.args[..]);
    assert_eq!(u32_arg(got, want.args.len()), group);
    assert_eq!(
        (got.grid, got.block, got.shared_mem, got.stream),
        (want.grid, want.block, want.shared_mem, want.stream)
    );
}

const G: RasterGroups = RasterGroups { gu: 16, down: 4 };

/// 2026-10-14: The tp layout's 704-wide slices at the m128 tile, fused: the fused launch becomes
/// `gu_m128_r` (group 16) and the `_k64` down `mx_m128_k64_r` (group 4); the rest is unchanged.
#[test]
fn the_m128_fused_and_k64_down_take_their_twins() {
    let plain = plan(cfg_tp(), M128_ROWS, true, None);
    let r = plan(cfg_tp(), M128_ROWS, true, Some(G));
    assert_eq!(funcs(&plain), vec![SORT, TILES, QUANT, GU128, G128_K64]);
    assert_eq!(funcs(&r), vec![SORT, TILES, QUANT, RGU128, R128_K64]);
    assert_same(&r[..3], &plain[..3]);
    assert_twin(&r[3], &plain[3], 16);
    assert_twin(&r[4], &plain[4], 4);
}

/// 2026-10-14: A plain-width (2048) down at the m128 tile takes `mx_m128_r`.
#[test]
fn a_plain_width_down_takes_the_plain_twin() {
    let plain = plan(cfg_small(), 48, true, None);
    let r = plan(cfg_small(), 48, true, Some(G));
    assert_eq!(funcs(&r), vec![SORT, TILES, QUANT, RGU128, R128]);
    assert_twin(&r[3], &plain[3], 16);
    assert_twin(&r[4], &plain[4], 4);
}

/// 2026-10-14: Without the fused launch only the down changes: the separate gate and up GEMMs,
/// the SwiGLU and the quantizer are the plain plan's.
#[test]
fn the_separate_gate_up_keep_their_order() {
    let plain = plan(cfg_tp(), M128_ROWS, false, None);
    let r = plan(cfg_tp(), M128_ROWS, false, Some(G));
    assert_eq!(
        funcs(&r),
        vec![SORT, TILES, QUANT, G128, G128, SWIGLU, QUANT_K64, R128_K64]
    );
    assert_same(&r[..7], &plain[..7]);
    assert_twin(&r[7], &plain[7], 4);
}

/// 2026-10-14: A group of 0 keeps that launch's plain entry; below the m128 tile nothing changes.
#[test]
fn a_zero_group_and_the_narrow_tiles_keep_the_plain_plan() {
    let plain = plan(cfg_tp(), M128_ROWS, true, None);
    let r = plan(
        cfg_tp(),
        M128_ROWS,
        true,
        Some(RasterGroups { gu: 0, down: 4 }),
    );
    assert_eq!(funcs(&r), vec![SORT, TILES, QUANT, GU128, R128_K64]);
    assert_same(&r[..4], &plain[..4]);
    let r = plan(
        cfg_tp(),
        M128_ROWS,
        true,
        Some(RasterGroups { gu: 16, down: 0 }),
    );
    assert_eq!(funcs(&r), vec![SORT, TILES, QUANT, RGU128, G128_K64]);
    assert_same(&r[4..], &plain[4..]);
    let narrow = plan(cfg_tp(), 32, true, None);
    assert_same(&plan(cfg_tp(), 32, true, Some(G)), &narrow);
    assert_eq!(funcs(&narrow), vec![SORT, TILES, QUANT, GU32, G32_K64]);
}

/// 2026-10-14: The lever parses strictly; on the grouped path it needs every `_r` entry.
#[test]
fn the_raster_lever_parses_strictly_and_needs_its_entries() {
    assert_eq!(parse_raster(None), Ok(None));
    assert_eq!(parse_raster(Some("0")), Ok(None));
    assert_eq!(parse_raster(Some("16,4")), Ok(Some(G)));
    assert_eq!(
        parse_raster(Some("0,4096")),
        Ok(Some(RasterGroups { gu: 0, down: 4096 }))
    );
    for bad in [
        "1", "16", "16,4097", "16,-4", "16;4", "16,4,2", " 16,4", "a,b", "",
    ] {
        let e = parse_raster(Some(bad)).unwrap_err();
        assert!(e.contains("METRALE_GLM_MOE_W4A4_RASTER"), "{bad}: {e}");
    }
    assert_eq!(check_ready(true, true), Ok(()));
    assert_eq!(check_ready(false, false), Ok(()));
    let e = check_ready(true, false).unwrap_err();
    assert!(e.contains("_r"), "{e}");
}
