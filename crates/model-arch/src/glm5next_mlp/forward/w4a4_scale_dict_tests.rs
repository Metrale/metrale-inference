// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: `METRALE_GLM_MOE_SCALE_DICT` on the decode W4A4 launch plan: a site whose tables
//! carry base tables runs, launch for launch, the plan of the same site over plain tables, except
//! that each weight launch (union and own-slots sweeps, slot GEMVs, plain and `_k64`) is its `_cs`
//! twin with its tables' base tables appended; a packed site without a twin, or with a base table
//! on one projection only, is refused.

use super::*;
use crate::glm5next_mlp::scale_dict_kernels::ScaleDictTwins;

const TWIN: u64 = 0x1000;
/// 2026-10-21: The weight entries of the mock kernel set (`kernels`), each listed with a twin.
const WEIGHT: [u64; 8] = [
    SWEEP8,
    SWEEP16,
    OWN_SWEEP,
    SLOTS,
    SWEEP8_K64,
    SWEEP16_K64,
    OWN_SWEEP_K64,
    SLOTS_K64,
];

/// 2026-10-21: The base table of the table whose packed pointers are `tag` (`table(tag)`).
fn bases(tag: DevicePtr) -> DevicePtr {
    DevicePtr(0xB000_0000 + tag.0)
}

fn plan(
    c: Glm5NextMlpConfig,
    rows: usize,
    packed: bool,
    twins: &[(u64, u64)],
    edit: impl Fn(&mut Glm5NextMlpKernels),
) -> anyhow::Result<Vec<MockLaunch>> {
    let gpu = MockGpuBackend::new();
    let mut k = kernels(&gpu);
    k.scale_dict = ScaleDictTwins::of_list(twins);
    edit(&mut k);
    let ws = Glm5NextMlpWorkspace::new(&gpu, &c, 16, &k.levers.mlp).unwrap();
    let mut w = moe_weights();
    if packed {
        for t in [&mut w.ptrs.gate, &mut w.ptrs.up, &mut w.ptrs.down] {
            t.scale_bases = bases(t.packed_ptrs);
        }
    }
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
    w4a4_experts(&site, SCALES)?;
    Ok(gpu.launches_snapshot())
}

/// 2026-10-21: `packed` is `plain` with each weight launch on its twin and the base tables of its
/// tables appended: a sweep's two (packed pointers at args 5 and 9), a slot GEMV's one (arg 4).
fn assert_twinned(plain: &[MockLaunch], packed: &[MockLaunch], what: &str) -> usize {
    assert_eq!(plain.len(), packed.len(), "{what}: launch count");
    let mut weight = 0;
    for (i, (p, q)) in plain.iter().zip(packed).enumerate() {
        if !WEIGHT.contains(&p.func) {
            assert_eq!(
                (q.func, q.grid, &q.args),
                (p.func, p.grid, &p.args),
                "{what}: launch {i}"
            );
            continue;
        }
        weight += 1;
        let tables: &[usize] = if p.func == SLOTS || p.func == SLOTS_K64 {
            &[4]
        } else {
            &[5, 9]
        };
        let mut want = p.args.clone();
        want.extend(
            tables
                .iter()
                .map(|&a| MockArg::Buffer(bases(ptr_arg(p, a)))),
        );
        assert_eq!(
            (q.func, q.grid, &q.args),
            (p.func | TWIN, p.grid, &want),
            "{what}: launch {i}"
        );
    }
    weight
}

#[test]
fn packed_tables_run_every_weight_launch_on_its_twin_with_the_base_tables() {
    let twins = WEIGHT.map(|h| (h, h | TWIN));
    for (c, name) in [(cfg(), "k128 down"), (cfg_tp(), "k64 down")] {
        for rows in [1, 2, 8, 9, 16] {
            for sweeps in [true, false] {
                let edit = |k: &mut Glm5NextMlpKernels| {
                    if !sweeps {
                        k.w4a4_sweep_ctas = 0;
                    }
                };
                let what = format!("{name}, {rows} rows, sweeps {sweeps}");
                let plain = plan(c, rows, false, &twins, edit).unwrap();
                let packed = plan(c, rows, true, &twins, edit).unwrap();
                let n = assert_twinned(&plain, &packed, &what);
                // 2026-10-21: gate and up (one sweep, or two slot GEMVs) and down.
                assert_eq!(n, if sweeps { 2 } else { 3 }, "{what}: weight launches");
            }
        }
    }
}

#[test]
fn a_packed_site_without_a_twin_or_with_one_base_table_is_refused() {
    let twins = WEIGHT.map(|h| (h, h | TWIN));
    // 2026-10-21: Every entry listed, none with a twin: the plan stops at its first weight launch
    // instead of reading compact scales as plain.
    let untwinned = WEIGHT.map(|h| (h, 0));
    let e = plan(cfg(), 2, true, &untwinned, |_| {}).unwrap_err();
    assert!(e.to_string().contains("no `_cs` twin"), "{e}");
    assert!(
        plan(cfg(), 2, false, &untwinned, |_| {}).is_ok(),
        "plain tables need no twin"
    );

    let gpu = MockGpuBackend::new();
    let mut k = kernels(&gpu);
    k.scale_dict = ScaleDictTwins::of_list(&twins);
    let c = cfg();
    let ws = Glm5NextMlpWorkspace::new(&gpu, &c, 16, &k.levers.mlp).unwrap();
    let mut w = moe_weights();
    w.ptrs.gate.scale_bases = bases(w.ptrs.gate.packed_ptrs);
    let site = MoeSite {
        gpu: &gpu,
        k: &k,
        cfg: &c,
        w: &w,
        x: X,
        rows: 2,
        ws: &ws,
        stream: 5,
    };
    let e = w4a4_experts(&site, SCALES).unwrap_err();
    assert!(e.to_string().contains("one projection's table"), "{e}");
}
