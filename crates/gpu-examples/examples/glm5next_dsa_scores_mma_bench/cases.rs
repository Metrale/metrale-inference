// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The radix top-k edge and ceiling cases, each byte-compared with `dsa_topk_pools`.
//!
//! Owner: model-arch examples (GLM-5.3 kernels).
//! Invariants: none beyond the types.

use super::*;

/// 2026-10-12: Rows that stress the radix top-k's tie and edge handling, each byte-compared with
/// `dsa_topk_pools`: every score equal; many ties at the cut; +0.0 and -0.0 mixed; a short row
/// of non-candidates (-FLT_MAX); fewer pools than the selection; one pool.
pub(super) fn edge_cases(gpu: &dyn GpuBackend, base: KernelHandle, rx: KernelHandle) -> Result<()> {
    let mut rng = Rng(0x5be0_cd19_137e_2179);
    let mut cases: Vec<(&str, Vec<f32>, usize)> = Vec::new();
    cases.push(("all equal", vec![1.25; 3000], 512));
    cases.push((
        "ties at the cut",
        (0..5000).map(|i| (i % 7) as f32).collect(),
        512,
    ));
    cases.push((
        "signed zeros",
        (0..4000)
            .map(|i| match i % 3 {
                0 => 0.0,
                1 => -0.0,
                _ => rng.unit(),
            })
            .collect(),
        512,
    ));
    cases.push((
        "non-candidates",
        (0..2000)
            .map(|i| if i % 5 == 0 { rng.unit() } else { f32::MIN })
            .collect(),
        512,
    ));
    cases.push(("short row", (0..300).map(|_| rng.unit()).collect(), 300));
    cases.push(("one pool", vec![0.5], 1));
    cases.push((
        "random wide",
        (0..20_000).map(|_| rng.unit()).collect(),
        512,
    ));
    for (name, row, select_k) in cases {
        let pools = row.len();
        let rows = 3;
        let all: Vec<f32> = (0..rows).flat_map(|_| row.iter().copied()).collect();
        let sc = up(gpu, &all)?;
        let (sa, sb) = (
            gpu.alloc(rows * select_k * 4)?,
            gpu.alloc(rows * select_k * 4)?,
        );
        topk(gpu, base, sc, sa, rows, pools, select_k)?;
        radix(gpu, rx, sc, sb, rows, pools, select_k)?;
        ensure!(
            read(gpu, sa, rows * select_k * 4)? == read(gpu, sb, rows * select_k * 4)?,
            "radix top-k edge case `{name}` differs from dsa_topk_pools"
        );
        for p in [sc, sa, sb] {
            gpu.free(p)?;
        }
    }
    println!("radix top-k: edge cases identical to dsa_topk_pools");
    Ok(())
}

/// 2026-10-21: The ceiling (captured decode) launch: scalar arguments at a 1M-token ceiling,
/// the live pools in a device geometry (`dsa_write_geom`'s slots). At each live pool count the
/// radix top-k must write the bitonic walk's bytes, untouched slots included; then both are
/// timed at decode pool counts (one row, as one DSA layer's selection of one decode row).
pub(super) fn ceiling_cases(
    gpu: &dyn GpuBackend,
    base: KernelHandle,
    rx: KernelHandle,
) -> Result<()> {
    const MAX_POOLS: usize = 262_144;
    const SEL: usize = TOPK / KP;
    let mut rng = Rng(0x1f83_d9ab_fb41_bd6b);
    let scores: Vec<f32> = (0..MAX_POOLS)
        .map(|i| match i % 11 {
            0 => f32::MIN,
            1 | 2 => (i % 5) as f32 * 0.25,
            _ => rng.unit(),
        })
        .collect();
    let sc = up(gpu, &scores)?;
    let geom = gpu.alloc(5 * 4)?;
    let (sa, sb) = (gpu.alloc(SEL * 4)?, gpu.alloc(SEL * 4)?);
    let run = |radix_select: bool, out: DevicePtr| -> Result<()> {
        if !radix_select {
            KernelLaunch::new(gpu, base)
                .grid([1, 1, 1])
                .block([256, 1, 1])
                .shared_mem((2048 * 16) as u32)
                .arg_ptr(sc)
                .arg_ptr(out)
                .arg_u32(1)
                .arg_u32(MAX_POOLS as u32)
                .arg_u32(2048)
                .arg_u32(SEL as u32)
                .arg_ptr(geom)
                .launch(0)
        } else {
            KernelLaunch::new(gpu, rx)
                .grid([1, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(sc)
                .arg_ptr(out)
                .arg_u32(1)
                .arg_u32(MAX_POOLS as u32)
                .arg_u32(SEL as u32)
                .arg_ptr(geom)
                .launch(0)
        }
    };
    let set_geom = |live: usize| -> Result<()> {
        let np2 = live.next_power_of_two().clamp(2, 2048) as i32;
        let g = [
            (live * KP) as i32,
            live as i32 + 1,
            live as i32,
            live.min(SEL) as i32,
            np2,
        ];
        let b: Vec<u8> = g.iter().flat_map(|v| v.to_le_bytes()).collect();
        gpu.copy_h2d(&b, geom)?;
        Ok(())
    };
    for live in [
        0usize, 1, 2, 300, 511, 512, 513, 4_096, 32_768, 131_072, MAX_POOLS,
    ] {
        set_geom(live)?;
        gpu.memset(sa, 0x7f, SEL * 4)?;
        gpu.memset(sb, 0x7f, SEL * 4)?;
        run(false, sa)?;
        run(true, sb)?;
        ensure!(
            read(gpu, sa, SEL * 4)? == read(gpu, sb, SEL * 4)?,
            "ceiling launch at {live} live pools: radix top-k differs from dsa_topk_pools"
        );
    }
    println!("radix top-k: ceiling launches identical to dsa_topk_pools at 0..262144 live pools");
    for ctx in [1_024usize, 16_384, 65_536, 131_072, 524_288, 1_048_576] {
        let live = ctx / KP;
        set_geom(live)?;
        let tb = median_us(gpu, || run(false, sa))?;
        let tr = median_us(gpu, || run(true, sb))?;
        println!(
            "  ceiling top-k at {ctx:8} ctx ({live:6} pools): dsa_topk_pools {tb:9.1} us | radix {tr:8.1} us ({:.1}x)",
            tb / tr
        );
    }
    for p in [sc, geom, sa, sb] {
        gpu.free(p)?;
    }
    Ok(())
}
