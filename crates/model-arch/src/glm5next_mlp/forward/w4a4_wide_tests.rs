// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: `METRALE_GLM_MOE_W4A4_WIDE_SWEEP`: its parse, the row test that hands 17..=32
//! rows to the wide sweeps instead of the grouped W4A4 GEMM, the entry each width runs, the
//! load-time refusal, and the launch plan of a wide routed run on a recording mock backend.
//! The numerics (bit identity with the slot GEMV) are the GPU benches' job.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::KernelHandle;
use metrale_gpu_runtime::gpu::mock::{MockGpuBackend, MockLaunch};

use super::super::Glm5NextMlpWorkspace;
use super::super::moe_experts::MoeSite;
use super::super::w4a4::tests::{
    QUANT, QUANT_K64, ROW_UNION, SCALES, SWEEP16, SWEEP24, SWEEP24_K64, SWEEP32, SWEEP32_K64,
    SWIGLU, X, assert_sweep_launches, cfg, cfg_tp, kernels, moe_weights, u32_arg,
};
use super::super::w4a4::{SweepEntries, sweep_kernel, w4a4_experts};
use super::{
    GRAPHS_LEVER as GRAPHS, MoeCallSite, WIDE_SWEEP_MAX_IDS, WIDE_SWEEP_MAX_ROWS,
    captured_batched_rows, check_graphs, check_ready, parse_switch, parse_wide_lever,
    sweep_max_rows, takes_grouped_at, w4a4_through_wide, wide_at, wide_ready,
};
use crate::glm5next_mlp::precision::{GroupPrecision, MlpGroup, MlpKernel};
use crate::glm5next_mlp::{Glm5NextMlpConfig, MOE_W4A4_MAX_ROWS};

#[test]
fn the_lever_is_off_unless_exactly_one() {
    assert_eq!(parse_wide_lever(None), Ok(false));
    assert_eq!(parse_wide_lever(Some("0")), Ok(false));
    assert_eq!(parse_wide_lever(Some("1")), Ok(true));
    for bad in ["", "true", "2", " 1", "on"] {
        assert!(parse_wide_lever(Some(bad)).is_err(), "{bad:?}");
    }
}

/// 2026-10-14: Off, the grouped GEMM starts at 17 rows as before; on, at 33. The lever only
/// ever narrows the grouped GEMM's rows, never widens them.
#[test]
fn the_lever_moves_the_grouped_floor_from_17_to_33_rows() {
    assert_eq!(sweep_max_rows(false), MOE_W4A4_MAX_ROWS);
    assert_eq!(sweep_max_rows(true), WIDE_SWEEP_MAX_ROWS);
    for (rows, off, on) in [
        (1usize, false, false),
        (16, false, false),
        (17, true, false),
        (24, true, false),
        (32, true, false),
        (33, true, true),
        (2048, true, true),
    ] {
        assert_eq!(takes_grouped_at(rows, false), off, "{rows} rows off");
        assert_eq!(takes_grouped_at(rows, true), on, "{rows} rows on");
    }
    for rows in 1..=4096 {
        assert!(takes_grouped_at(rows, false) || !takes_grouped_at(rows, true));
    }
}

/// 2026-10-15: Every caller but the batched decode body runs exactly the grouped row test it ran
/// before the lever existed, at every prefill width up to the largest prefill chunk (8208 rows):
/// the general site never reads the lever. The decode site reads it (unset here: off).
#[test]
fn only_the_batched_decode_site_reads_the_lever() {
    assert!(!wide_at(MoeCallSite::General).unwrap());
    assert_eq!(
        wide_at(MoeCallSite::BatchedDecode).unwrap(),
        super::moe_w4a4_wide_sweep().unwrap()
    );
    for rows in 1..=8208 {
        assert_eq!(
            takes_grouped_at(rows, wide_at(MoeCallSite::General).unwrap()),
            super::super::w4a4_grouped::takes_grouped(rows),
            "{rows} rows"
        );
    }
}

#[test]
fn the_lever_needs_the_grouped_workspace_and_every_wide_entry() {
    assert!(check_ready(false, true).unwrap_err().contains("GROUPED=1"));
    assert!(check_ready(true, false).unwrap_err().contains("PTX"));
    assert_eq!(check_ready(true, true), Ok(()));
    let gpu = MockGpuBackend::new();
    assert!(wide_ready(&kernels(&gpu)));
    // 2026-10-14: The 8- and 16-row entries are not the lever's; any absent wide entry is.
    let mut k = kernels(&gpu);
    k.w4a4_moe_sweep[0] = KernelHandle(0);
    k.w4a4_moe_k64.sweep[1] = KernelHandle(0);
    assert!(wide_ready(&k));
    for i in 0..4 {
        let mut k = kernels(&gpu);
        let h = if i < 2 {
            &mut k.w4a4_moe_sweep[2 + i]
        } else {
            &mut k.w4a4_moe_k64.sweep[i]
        };
        *h = KernelHandle(0);
        assert!(!wide_ready(&k), "wide entry {i} absent");
    }
}

/// 2026-10-14: 17..=24 rows run the 24-token entry and 25..=32 the 32-token one, on the union,
/// only with the lever; 16 rows keep the 16-token entry either way and 33 have none.
#[test]
fn each_wide_width_maps_to_its_entry_only_under_the_lever() {
    let gpu = MockGpuBackend::new();
    let k = kernels(&gpu);
    let own = k.w4a4_moe_slots_sweep;
    // 2026-10-16: `KernelHandle` has no `PartialEq`; compare the raw handle.
    let at = |rows, top_k, wide| {
        sweep_kernel(&k, own, k.w4a4_moe_sweep, rows, top_k, wide).map(|(h, e)| (h.0, e))
    };
    for wide in [false, true] {
        assert_eq!(at(16, 8, wide), Some((SWEEP16, SweepEntries::Union)));
        assert_eq!(at(33, 8, wide), None);
    }
    for (rows, want) in [
        (17usize, SWEEP24),
        (24, SWEEP24),
        (25, SWEEP32),
        (32, SWEEP32),
    ] {
        assert_eq!(at(rows, 8, false), None, "{rows} rows off");
        assert_eq!(
            at(rows, 8, true),
            Some((want, SweepEntries::Union)),
            "{rows} rows on"
        );
    }
    // 2026-10-14: The union builder's table caps the ids at 256: 32 rows of top-9 do not fit.
    assert_eq!(32 * 8, WIDE_SWEEP_MAX_IDS);
    assert_eq!(at(32, 9, true), None);
    assert_eq!(at(28, 9, true), Some((SWEEP32, SweepEntries::Union)));
    let mut k = kernels(&gpu);
    k.moe_row_union = KernelHandle(0);
    assert!(sweep_kernel(&k, own, k.w4a4_moe_sweep, 24, 8, true).is_none());
}

/// 2026-10-14: `w4a4_experts` over `rows` rows on a grouped workspace of 32 rows.
fn run_wide(
    c: Glm5NextMlpConfig,
    rows: usize,
    wide: bool,
) -> (Vec<MockLaunch>, Glm5NextMlpWorkspace) {
    let gpu = MockGpuBackend::new();
    let k = kernels(&gpu);
    let ws = Glm5NextMlpWorkspace::new_with(&gpu, &c, WIDE_SWEEP_MAX_ROWS, true).unwrap();
    let w = moe_weights();
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
    w4a4_experts(&site, SCALES, wide).unwrap();
    (gpu.launches_snapshot(), ws)
}

/// 2026-10-14: Under the lever 17..=32 routed rows build the union once (rows x top_k threads)
/// and sweep it as the narrower widths do, through the wide entry for their width.
#[test]
fn wide_routed_rows_sweep_the_union_under_the_lever() {
    for (rows, sweep) in [
        (17usize, SWEEP24),
        (24, SWEEP24),
        (25, SWEEP32),
        (32, SWEEP32),
    ] {
        let (l, ws) = run_wide(cfg(), rows, true);
        let funcs: Vec<u64> = l.iter().map(|l| l.func).collect();
        assert_eq!(
            funcs,
            vec![QUANT, ROW_UNION, sweep, SWIGLU, QUANT, sweep],
            "{rows} rows"
        );
        let slots = (rows * 8) as u32;
        assert_eq!(
            (l[1].block[0], u32_arg(&l[1], 3), u32_arg(&l[1], 4)),
            (slots, rows as u32, 8)
        );
        assert_eq!(l[4].grid[0], slots);
        assert_sweep_launches(&l, [2, 5], rows, ws.u_eid, &ws);
    }
}

/// 2026-10-14: Off, the same widths run the grouped GEMM (no sweep, no union build); on, 33
/// rows still do.
#[test]
fn the_grouped_gemm_keeps_the_wide_rows_off_and_past_32_rows_on() {
    for (rows, wide) in [(17usize, false), (24, false), (32, false), (33, true)] {
        let c = cfg();
        let gpu = MockGpuBackend::new();
        let k = kernels(&gpu);
        let ws = Glm5NextMlpWorkspace::new_with(&gpu, &c, rows, true).unwrap();
        let w = moe_weights();
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
        w4a4_experts(&site, SCALES, wide).unwrap();
        let funcs: Vec<u64> = gpu.launches_snapshot().iter().map(|l| l.func).collect();
        assert!(!funcs.is_empty(), "{rows} rows wide={wide}");
        for f in [ROW_UNION, SWEEP24, SWEEP32] {
            assert!(!funcs.contains(&f), "{rows} rows wide={wide}: {funcs:x?}");
        }
    }
}

/// 2026-10-14: A 704-wide expert slice under the lever: gate and up run the plain wide entry at
/// K 4096, down its `_k64` twin at K 704.
#[test]
fn a_k64_slice_runs_the_k64_wide_twins() {
    for (rows, gate_up, down) in [(24usize, SWEEP24, SWEEP24_K64), (32, SWEEP32, SWEEP32_K64)] {
        let (l, _ws) = run_wide(cfg_tp(), rows, true);
        let funcs: Vec<u64> = l.iter().map(|l| l.func).collect();
        assert_eq!(
            funcs,
            vec![QUANT, ROW_UNION, gate_up, SWIGLU, QUANT_K64, down],
            "{rows} rows"
        );
        assert_eq!((u32_arg(&l[2], 14), u32_arg(&l[2], 15)), (704, 4096));
        assert_eq!((u32_arg(&l[5], 14), u32_arg(&l[5], 15)), (4096, 704));
    }
}

/// 2026-10-16: The rows a text-stack site's workspace carries on a served load (the loader's
/// `verify_k`, here `METRALE_GLM_PREFILL_ROWS=8208`). Any value past `WIDE_SWEEP_MAX_ROWS` bounds
/// the check by 32; with every lever at its default `verify_k` is 16 and no wider call reaches it.
const SERVED_TEXT_ROWS: usize = 8208;

/// 2026-10-16: A routed-experts plan under `--activation-quantization declared` on an A4-stamped
/// checkpoint whose W4A4 kernels serve up to `cap` rows.
fn routed_plan(cap: usize) -> GroupPrecision {
    let ladder = metrale_config::ActivationQuantization::parse("declared")
        .expect("flag parses")
        .ladder(metrale_config::ProjFamily::Moe)
        .clone();
    GroupPrecision::resolve(
        MlpGroup::RoutedExperts,
        ladder,
        metrale_config::Nvfp4Act::A4,
        cap,
        true,
    )
    .expect("resolves")
}

/// 2026-10-16: `METRALE_GLM_WIDE_VERIFY_GRAPHS` parses as the wide lever does, and names itself
/// in the refusal.
#[test]
fn the_graphs_lever_is_off_unless_exactly_one() {
    assert_eq!(parse_switch(GRAPHS, None), Ok(false));
    assert_eq!(parse_switch(GRAPHS, Some("0")), Ok(false));
    assert_eq!(parse_switch(GRAPHS, Some("1")), Ok(true));
    for bad in ["", "true", "2", "1 ", "yes"] {
        let e = parse_switch(GRAPHS, Some(bad)).unwrap_err();
        assert!(e.starts_with(GRAPHS), "{bad:?}: {e}");
    }
}

/// 2026-10-16: Off, the captured-row limit is exactly the layer's `max_captured_rows` (the
/// grouped floor minus one) at every floor; on, it rises to 32 at the recipe's floor of 17 and
/// never falls below the off limit (a floor past 33 keeps its own).
#[test]
fn the_graphs_lever_lifts_the_captured_rows_to_the_wide_sweep_only() {
    assert_eq!(captured_batched_rows(17, false), 16);
    assert_eq!(captured_batched_rows(17, true), WIDE_SWEEP_MAX_ROWS);
    assert_eq!(captured_batched_rows(9, true), WIDE_SWEEP_MAX_ROWS);
    assert_eq!(captured_batched_rows(65, true), 64);
    for floor in 1..=4096 {
        let off = captured_batched_rows(floor, false);
        assert_eq!(off, floor - 1, "floor {floor}");
        let on = captured_batched_rows(floor, true);
        assert!(
            on >= off && on <= off.max(WIDE_SWEEP_MAX_ROWS),
            "floor {floor}"
        );
    }
}

/// 2026-10-16: The refusals: on without the wide sweep, or with a plan that leaves W4A4 inside
/// 17..=32 rows (those rows would take the grouped W4A16 GEMM's host tile sizing); off never
/// refuses.
#[test]
fn the_graphs_lever_needs_the_wide_sweep_and_w4a4_through_32_rows() {
    for (wide, through) in [(false, false), (false, true), (true, false), (true, true)] {
        assert_eq!(check_graphs(false, wide, through), Ok(()));
    }
    assert!(
        check_graphs(true, false, true)
            .unwrap_err()
            .contains(super::LEVER)
    );
    assert!(check_graphs(true, true, false).is_err());
    assert_eq!(check_graphs(true, true, true), Ok(()));
}

/// 2026-10-16: The W4A4 cap of a plain load (16 rows) leaves 17..=32 rows on W4A16, so the
/// lever is refused there; the grouped load's cap covers them; a floor past 32 asks nothing.
/// A served text site carries more than 32 rows, so 32 bounds it; a site whose workspace stops
/// at 16 rows never runs 17 (batched groups are cut to it), so it asks nothing.
#[test]
fn w4a4_through_wide_reads_the_routed_plan() {
    let text = SERVED_TEXT_ROWS;
    let plain = routed_plan(MOE_W4A4_MAX_ROWS);
    assert_eq!(plain.kernel(17), MlpKernel::W4a16);
    assert!(!w4a4_through_wide(&plain, 17, text));
    assert!(!w4a4_through_wide(&plain, 17, WIDE_SWEEP_MAX_ROWS + 1));
    assert!(w4a4_through_wide(&plain, 17, MOE_W4A4_MAX_ROWS));
    assert!(!w4a4_through_wide(&routed_plan(31), 17, text));
    assert!(w4a4_through_wide(
        &routed_plan(WIDE_SWEEP_MAX_ROWS),
        17,
        text
    ));
    assert!(w4a4_through_wide(&routed_plan(8208), 17, text));
    assert!(w4a4_through_wide(&plain, WIDE_SWEEP_MAX_ROWS + 1, text));
}

/// 2026-10-16: The MTP drafter's routed site (layer 45): its checkpoint stamp is 16-bit, so the
/// plan is W4A16 at every row and the text-stack check refuses it, which disabled MTP on a
/// served load. Its workspace carries `drafter_block_rows` (1, or `PROPOSE_BATCH_MAX` = 16)
/// rows, below the grouped floor, so its plan passes; one row more than the floor would not.
#[test]
fn the_drafter_site_passes_below_the_grouped_floor_and_text_rows_do_not() {
    let ladder = metrale_config::ActivationQuantization::parse("declared")
        .expect("flag parses")
        .ladder(metrale_config::ProjFamily::Moe)
        .clone();
    let drafter = GroupPrecision::resolve(
        MlpGroup::RoutedExperts,
        ladder,
        metrale_config::Nvfp4Act::Wide,
        MOE_W4A4_MAX_ROWS,
        true,
    )
    .expect("resolves");
    assert!((1..=4096).all(|r| drafter.kernel(r) == MlpKernel::W4a16));
    let propose_max = crate::glm5next_mtp_head::PROPOSE_BATCH_MAX;
    assert!(propose_max < 17);
    for rows in [1, propose_max] {
        assert!(w4a4_through_wide(&drafter, 17, rows), "{rows} rows");
        assert_eq!(
            check_graphs(true, true, w4a4_through_wide(&drafter, 17, rows)),
            Ok(())
        );
    }
    assert!(!w4a4_through_wide(&drafter, 17, 17));
    assert!(!w4a4_through_wide(&drafter, 17, SERVED_TEXT_ROWS));
    assert!(check_graphs(true, true, w4a4_through_wide(&drafter, 17, 32)).is_err());
}
