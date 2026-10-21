// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: `METRALE_GLM_MOE_SCALE_DICT` on the grouped W4A4 launch plan: over tables that
//! carry base tables, every grouped GEMM and fused gate/up launch (both tiles, plain and `_k64`,
//! raster `_r` and column-tile `_n` twins) is its `_cs` twin with its tables' base tables appended,
//! and every other launch is the plain-table plan's.

use super::*;
use crate::glm5next_mlp::scale_dict_kernels::ScaleDictTwins;

const TWIN: u64 = 0x1000;
/// 2026-10-21: The entries a routed site reaches (tiles m32 and m128), as `scale_dict_kernels`
/// lists them; the m64 entries never run on a routed site.
const LISTED: [u64; 11] = [
    G32, G32_K64, G128, G128_K64, R128, R128_K64, N128, N128_K64, GU32, GU128, RGU128,
];
const FUSED: [u64; 3] = [GU32, GU128, RGU128];

fn bases(tag: DevicePtr) -> DevicePtr {
    DevicePtr(0xB000_0000 + tag.0)
}

#[allow(clippy::type_complexity)]
fn plan_of(
    c: &Glm5NextMlpConfig,
    rows: usize,
    packed: bool,
    levers: (bool, Option<RasterGroups>, Option<u32>),
) -> anyhow::Result<Vec<MockLaunch>> {
    let gpu = MockGpuBackend::new();
    let mut k = kernels(&gpu);
    k.scale_dict = ScaleDictTwins::of_list(&LISTED.map(|h| (h, h | TWIN)));
    let ws = Glm5NextMlpWorkspace::new_forked(&gpu, c, rows, true, 0).unwrap();
    let mut w = moe_weights(usize::MAX);
    if packed {
        for t in [&mut w.ptrs.gate, &mut w.ptrs.up, &mut w.ptrs.down] {
            t.scale_bases = bases(t.packed_ptrs);
        }
    }
    let site = MoeSite {
        gpu: &gpu,
        k: &k,
        cfg: c,
        w: &w,
        x: X,
        rows,
        ws: &ws,
        stream: 5,
    };
    let (fused, raster, nc) = levers;
    w4a4_grouped_experts_with(&site, SCALES, fused, raster, nc)?;
    Ok(gpu.launches_snapshot())
}

#[test]
fn packed_tables_run_every_grouped_launch_on_its_twin_with_the_base_tables() {
    let raster = Some(RasterGroups { gu: 4, down: 4 });
    for (c, rows, name) in [
        (cfg_tp(), 32, "tp m32"),
        (cfg_tp(), 1728, "tp m128"),
        (cfg_small(), 48, "k128 m128"),
    ] {
        for levers in [
            (false, None, None),
            (true, None, None),
            (true, raster, None),
            (false, raster, Some(3)),
            (true, None, Some(2)),
        ] {
            let what = format!("{name}, {rows} rows, levers {levers:?}");
            let plain = plan_of(&c, rows, false, levers).unwrap();
            let packed = plan_of(&c, rows, true, levers).unwrap();
            assert_eq!(plain.len(), packed.len(), "{what}");
            let mut twinned = 0;
            for (i, (p, q)) in plain.iter().zip(&packed).enumerate() {
                if !LISTED.contains(&p.func) {
                    assert_eq!((q.func, &q.args), (p.func, &p.args), "{what}: launch {i}");
                    continue;
                }
                twinned += 1;
                // 2026-10-21: A fused launch's gate and up tables are args 3 and 6; a one-projection
                // launch's table arg 3.
                let tables: &[usize] = if FUSED.contains(&p.func) {
                    &[3, 6]
                } else {
                    &[3]
                };
                let mut want = p.args.clone();
                want.extend(
                    tables
                        .iter()
                        .map(|&a| MockArg::Buffer(bases(ptr_arg(p, a)))),
                );
                assert_eq!(
                    (q.func, q.grid, q.shared_mem, &q.args),
                    (p.func | TWIN, p.grid, p.shared_mem, &want),
                    "{what}: launch {i}"
                );
            }
            assert_eq!(
                twinned,
                if levers.0 { 2 } else { 3 },
                "{what}: grouped launches"
            );
        }
    }
}

/// 2026-10-21: A packed site on an entry with no listed twin (the m64 tile, which no routed
/// launch takes) is refused rather than reading compact scales as plain.
#[test]
fn a_packed_launch_on_an_unlisted_entry_is_refused() {
    let gpu = MockGpuBackend::new();
    let k = kernels(&gpu);
    let mut t = table(0x1000);
    t.scale_bases = bases(t.packed_ptrs);
    let entry = crate::glm5next_mlp::forward::GroupedEntry::plain(KernelHandle(G64));
    let twins = ScaleDictTwins::of_list(&LISTED.map(|h| (h, h | TWIN)));
    assert!(entry.over(&twins, t.scale_bases).is_err());
    assert!(
        entry
            .over(&k.scale_dict, DevicePtr::NULL)
            .is_ok_and(|e| !e.cs && e.kern.0 == G64)
    );
}
