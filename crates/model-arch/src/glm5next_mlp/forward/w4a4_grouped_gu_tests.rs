// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The fused gate/up launch plan (`METRALE_GLM_MOE_W4A4_GU_FUSED=1`) on the
//! recording mock: the sort, tile pass and gate/up quantization as before, then ONE fused launch
//! (its tables, the down input written to the gate/up/activated scratch, N, K, the stored width,
//! the down scale, the SwiGLU limit, its grid) and the unchanged down entry reading that scratch.
//! The bytes are the GPU bench's job (`examples/glm5next_moe_epi_bench`).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use super::super::super::w4a4_grouped_gu::check_ready;
use super::super::smem_bytes_of;
use super::*;

/// 2026-10-14: The routed experts over `rows` rows on a grouped workspace, the fused lever as
/// given.
fn run_fused(
    c: Glm5NextMlpConfig,
    k: &Glm5NextMlpKernels,
    rows: usize,
    fused: bool,
) -> (anyhow::Result<()>, Vec<MockLaunch>, Glm5NextMlpWorkspace) {
    let gpu = MockGpuBackend::new();
    let ws = Glm5NextMlpWorkspace::new_forked(&gpu, &c, rows, true, 0).unwrap();
    let w = moe_weights(usize::MAX);
    let site = MoeSite {
        gpu: &gpu,
        k,
        cfg: &c,
        w: &w,
        x: X,
        rows,
        ws: &ws,
        stream: 5,
    };
    let r = w4a4_grouped_experts_with(&site, SCALES, fused, None, None);
    (r, gpu.launches_snapshot(), ws)
}

/// 2026-10-14: One fused launch: the gate/up input from the W4A4 scratch, the gate then up
/// tables, the down input into (a_gate, a_up, a_act), the sort's offsets and gather map, the tile
/// offsets, E / N / K / Kp, the down scale and the SwiGLU limit; grid (N / (cols / 2), slots /
/// mt + E), the one-projection entry's shared memory.
fn assert_fused(
    l: &MockLaunch,
    ws: &Glm5NextMlpWorkspace,
    (n, k, kp): (u32, u32, u32),
    (slots, ti, experts): (u32, usize, u32),
) {
    let ptrs = [
        ws.w4a4_aq,
        ws.w4a4_as,
        ws.w4a4_ag,
        DevicePtr(0x1000),
        DevicePtr(0x1001),
        DevicePtr(0x1002),
        DevicePtr(0x2000),
        DevicePtr(0x2001),
        DevicePtr(0x2002),
        ws.a_gate,
        ws.a_up,
        ws.a_act,
        ws.expert_offsets(),
        ws.w4a4_tile_off,
        ws.sorted_token_ids(),
    ];
    for (i, want) in ptrs.into_iter().enumerate() {
        assert_eq!(ptr_arg(l, i), want, "pointer arg {i}");
    }
    assert_eq!(
        (
            u32_arg(l, 15),
            u32_arg(l, 16),
            u32_arg(l, 17),
            u32_arg(l, 18)
        ),
        (experts, n, k, kp)
    );
    assert_eq!((f32_arg(l, 19), f32_arg(l, 20)), (SCALES.down, 10.0));
    assert_eq!(l.args.len(), 21);
    let (mt, cols) = ([32u32, 64, 128][ti], [128u32, 128, 64][ti]);
    assert_eq!(l.grid, [n / (cols / 2), slots.div_ceil(mt) + experts, 1]);
    assert_eq!((l.block, l.shared_mem), ([256, 1, 1], smem_bytes_of(ti)));
    assert_eq!(l.stream, 5);
}

/// 2026-10-14: 32 rows of the tp layout's 704-wide slices fused: sort, tiles, the gate/up
/// quantization, the 32-slot fused entry (N 704, K 4096, stored at 768 for the `_k64` down) and
/// the 32-slot `_k64` down over (a_gate, a_up, a_act); no SwiGLU, no second quantization.
#[test]
fn a_fused_k64_slice_runs_one_gate_up_launch_then_down() {
    let gpu = MockGpuBackend::new();
    let (r, l, ws) = run_fused(cfg_tp(), &kernels(&gpu), 32, true);
    r.unwrap();
    let funcs: Vec<u64> = l.iter().map(|l| l.func).collect();
    assert_eq!(funcs, vec![SORT, TILES, QUANT, GU32, G32_K64]);
    assert_eq!(f32_arg(&l[2], 5), SCALES.gate_up);
    assert_fused(&l[3], &ws, (704, 4096, 768), (256, 0, 288));
    let act = [ws.a_gate, ws.a_up, ws.a_act];
    let null = DevicePtr::NULL;
    let (n, slots) = ((4096, 704), (256, 32, 288));
    assert_grouped_act(&l[4], &ws, act, 0x3000, ws.expert_out, null, n, slots);
}

/// 2026-10-14: 48 rows of whole 2048-wide experts fused: the 128-slot fused entry (64 weight
/// rows = 32 columns a CTA), the down input stored at 2048, the plain 128-slot down.
#[test]
fn a_fused_plain_width_takes_the_128_slot_entries() {
    let gpu = MockGpuBackend::new();
    let (r, l, ws) = run_fused(cfg_small(), &kernels(&gpu), 48, true);
    r.unwrap();
    let funcs: Vec<u64> = l.iter().map(|l| l.func).collect();
    assert_eq!(funcs, vec![SORT, TILES, QUANT, GU128, G128]);
    assert_fused(&l[3], &ws, (2048, 4096, 2048), (384, 2, 8));
    let act = [ws.a_gate, ws.a_up, ws.a_act];
    let null = DevicePtr::NULL;
    let (n, slots) = ((4096, 2048), (384, 128, 8));
    assert_grouped_act(&l[4], &ws, act, 0x3000, ws.expert_out, null, n, slots);
}

/// 2026-10-14: The lever off keeps the separate launches exactly (the control for the two
/// tests above: same site, same kernels).
#[test]
fn the_lever_off_keeps_the_separate_launches() {
    let gpu = MockGpuBackend::new();
    let (r, l, _ws) = run_fused(cfg_tp(), &kernels(&gpu), 32, false);
    r.unwrap();
    let funcs: Vec<u64> = l.iter().map(|l| l.func).collect();
    assert_eq!(
        funcs,
        vec![SORT, TILES, QUANT, G32, G32, SWIGLU, QUANT_K64, G32_K64]
    );
}

/// 2026-10-14: A fused site whose entry is absent is refused before the sort launches.
#[test]
fn a_missing_fused_entry_is_refused_before_launching() {
    let gpu = MockGpuBackend::new();
    let mut k = kernels(&gpu);
    k.w4a4_grouped.gu[0] = KernelHandle(0);
    let (r, l, _ws) = run_fused(cfg_tp(), &k, 32, true);
    let e = r.unwrap_err().to_string();
    assert!(e.contains("METRALE_GLM_MOE_W4A4_GU_FUSED"), "{e}");
    assert!(l.is_empty());
    let (r, _, _) = run_fused(cfg_tp(), &k, 32, false);
    r.unwrap();
}

/// 2026-10-14: On, the lever needs the grouped workspace and every fused entry.
#[test]
fn the_fused_lever_needs_the_grouped_path() {
    assert_eq!(check_ready(true, true), Ok(()));
    let e = check_ready(false, true).unwrap_err();
    assert!(e.contains("METRALE_GLM_MOE_W4A4_GROUPED=1"), "{e}");
    let e = check_ready(true, false).unwrap_err();
    assert!(e.contains("w4a4_grouped_gu"), "{e}");
}
