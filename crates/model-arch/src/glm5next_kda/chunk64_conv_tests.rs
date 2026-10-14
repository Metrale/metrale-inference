// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The launch plan of the 64-token scan with the conv inside its prepare
//! (`METRALE_GLM_KDA_CHUNK64_CONV`) on the mock backend: no conv launch, each slab's fused
//! prepare with its arguments in the kernel's order for both input layouts, the window advance
//! last, and the refusals before any launch. The bytes are compared on a GPU by
//! `examples/glm5next_kda_chunk64_conv_parity.rs`.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

use super::super::super::Glm5NextKdaWorkspace;
use super::super::super::Pack;
use super::super::super::decode_rows_tests::{
    CHUNK64_PREP_CONV, CHUNK64_SCAN, CONV_WINDOW_PARTS, cfg, layer, seq_state, since,
};
use super::super::super::gate_at::GateAt;
use super::super::super::onorm_at::NormAt;
use super::super::{CHUNK64_PREP_SMEM, CHUNK64_SCAN_SMEM};
use super::{chunk64_conv_shape_refusal, parse_chunk64_conv};

fn ptr(l: &MockLaunch, i: usize) -> DevicePtr {
    match &l.args[i] {
        MockArg::Buffer(p) => *p,
        other => panic!("arg {i} is {other:?}, not a pointer"),
    }
}

fn word(l: &MockLaunch, i: usize) -> [u8; 4] {
    match &l.args[i] {
        MockArg::Bytes(b) => b[..4].try_into().unwrap(),
        other => panic!("arg {i} is {other:?}, not 4 bytes"),
    }
}

fn u32a(l: &MockLaunch, i: usize) -> u32 {
    u32::from_le_bytes(word(l, i))
}

fn f32a(l: &MockLaunch, i: usize) -> f32 {
    f32::from_le_bytes(word(l, i))
}

/// 2026-10-14: 700 rows (11 chunks: slabs of 4, 4, 3) with the packed rows as input: a fused
/// prepare and a scan per slab, then the window advance; no conv launch. Every prepare argument
/// sits where `kda_chunk64_prepare_conv` takes it: the packed q/k/v (k one `qkv`, v two past q,
/// rows `conv_dim` apart), the conv state, weight and no bias, width 4 and the L2 eps, the stage
/// at `conv_out` row 0, gate/beta, the six scratch buffers the scan then reads, and
/// heads/rows/first chunk/stage stride/gate stride/scale.
#[test]
fn the_fused_path_runs_no_conv_launch_and_advances_the_window_last() {
    let gpu = MockGpuBackend::new();
    let lay = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 700).unwrap();
    let st = seq_state(&gpu);
    let from = gpu.launch_count();
    lay.stateful_chunk64_with(
        &gpu,
        700,
        &st,
        &ws,
        Pack::InFrontEnd,
        true,
        GateAt::Kernel,
        NormAt::Back,
        7,
    )
    .unwrap();
    let l = since(&gpu, from);
    let funcs: Vec<u64> = l.iter().map(|x| x.func).collect();
    assert_eq!(
        funcs,
        vec![
            CHUNK64_PREP_CONV,
            CHUNK64_SCAN,
            CHUNK64_PREP_CONV,
            CHUNK64_SCAN,
            CHUNK64_PREP_CONV,
            CHUNK64_SCAN,
            CONV_WINDOW_PARTS
        ]
    );
    let c = cfg();
    let (qkv, cd) = (c.qkv_dim(), c.conv_dim());
    let scratch = [
        ws.chunk_u,
        ws.chunk_w,
        ws.q_f32,
        ws.k_f32,
        ws.v_f32,
        ws.chunk_gc,
    ];
    for (slab, (c0, n)) in [(0u32, 4u32), (4, 4), (8, 3)].into_iter().enumerate() {
        let (p, s) = (&l[2 * slab], &l[2 * slab + 1]);
        assert_eq!((p.grid, p.block), ([n, 2, 1], [256, 1, 1]), "slab {slab}");
        assert_eq!(p.shared_mem, CHUNK64_PREP_SMEM);
        assert_eq!(s.shared_mem, CHUNK64_SCAN_SMEM);
        assert_eq!(
            (ptr(p, 0), ptr(p, 1), ptr(p, 2), u32a(p, 3)),
            (
                ws.qkv_proj,
                ws.qkv_proj.offset(qkv * 2),
                ws.qkv_proj.offset(2 * qkv * 2),
                cd as u32
            )
        );
        assert_eq!(
            (ptr(p, 4), ptr(p, 5), ptr(p, 6)),
            (st.conv, lay.weights.conv.weight, DevicePtr::NULL)
        );
        assert_eq!((u32a(p, 7), f32a(p, 8)), (4, c.l2_eps));
        assert_eq!(
            (ptr(p, 9), ptr(p, 10), ptr(p, 11)),
            (ws.conv_out, ws.gate, ws.beta)
        );
        for (i, b) in scratch.iter().enumerate() {
            assert_eq!(ptr(p, 12 + i), *b, "prepare scratch {i}");
            assert_eq!(ptr(s, i), *b, "scan scratch {i}");
        }
        assert_eq!(
            (
                u32a(p, 18),
                u32a(p, 19),
                u32a(p, 20),
                u32a(p, 21),
                u32a(p, 22)
            ),
            (2, 700, c0, cd as u32, qkv as u32)
        );
        assert_eq!(f32a(p, 23), 1.0 / (c.head_dim as f32).sqrt());
        assert_eq!((u32a(s, 10), u32a(s, 11)), (c0, n), "slab {slab} scan");
    }
    let w = &l[6];
    assert_eq!((w.grid, w.block), ([div_ceil(cd, 256), 1, 1], [256, 1, 1]));
    assert_eq!(
        (ptr(w, 0), ptr(w, 1), ptr(w, 2), ptr(w, 3)),
        (
            st.conv,
            ws.qkv_proj,
            ws.qkv_proj.offset(qkv * 2),
            ws.qkv_proj.offset(2 * qkv * 2)
        )
    );
    assert_eq!(
        (u32a(w, 4), u32a(w, 5), u32a(w, 6), u32a(w, 7), u32a(w, 8)),
        (cd as u32, qkv as u32, cd as u32, 4, 700)
    );
}

fn div_ceil(a: usize, b: usize) -> u32 {
    a.div_ceil(b) as u32
}

/// 2026-10-14: With the pack in the conv, the prepare and the window advance read the three
/// projections where the front end left them: `[k, qkv]` each, k at `k * qkv`, v at
/// `2 k * qkv`, rows `qkv` apart.
#[test]
fn the_pack_in_conv_layout_reads_the_three_parts() {
    let gpu = MockGpuBackend::new();
    let lay = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 300).unwrap();
    let st = seq_state(&gpu);
    let from = gpu.launch_count();
    lay.stateful_chunk64_with(
        &gpu,
        130,
        &st,
        &ws,
        Pack::InConv,
        true,
        GateAt::Kernel,
        NormAt::Back,
        7,
    )
    .unwrap();
    let l = since(&gpu, from);
    assert_eq!(
        l.iter().map(|x| x.func).collect::<Vec<_>>(),
        vec![CHUNK64_PREP_CONV, CHUNK64_SCAN, CONV_WINDOW_PARTS]
    );
    let qkv = cfg().qkv_dim();
    let parts = (
        ws.qkv_parts,
        ws.qkv_parts.offset(130 * qkv * 2),
        ws.qkv_parts.offset(2 * 130 * qkv * 2),
    );
    let (p, w) = (&l[0], &l[2]);
    assert_eq!((ptr(p, 0), ptr(p, 1), ptr(p, 2)), parts);
    assert_eq!(u32a(p, 3), qkv as u32);
    assert_eq!((ptr(w, 1), ptr(w, 2), ptr(w, 3)), parts);
    assert_eq!((u32a(w, 6), u32a(w, 8)), (qkv as u32, 130));
}

/// 2026-10-14: Without either fused kernel, or at a conv width the tile is not built for, the
/// fused path launches nothing and errors.
#[test]
fn the_fused_path_refuses_before_any_launch() {
    let gpu = MockGpuBackend::new();
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 300).unwrap();
    let st = seq_state(&gpu);
    let mut no_prep = layer(&gpu);
    no_prep.kernels.seq.chunk64_prepare_conv = KernelHandle(0);
    let mut no_window = layer(&gpu);
    no_window.kernels.seq.conv_window_parts = KernelHandle(0);
    let mut width3 = layer(&gpu);
    width3.cfg.conv_kernel = 3;
    for (name, lay) in [
        ("no prepare", no_prep),
        ("no window", no_window),
        ("width 3", width3),
    ] {
        let from = gpu.launch_count();
        assert!(
            lay.stateful_chunk64_with(
                &gpu,
                130,
                &st,
                &ws,
                Pack::InFrontEnd,
                true,
                GateAt::Kernel,
                NormAt::Back,
                7
            )
            .is_err(),
            "{name}"
        );
        assert_eq!(gpu.launch_count(), from, "{name}");
    }
}

/// 2026-10-14: The shapes: width 4, head_dim 128 and q channels a multiple of 256 (2 heads, the
/// mock's; 22 heads, a GLM rank's) pass; any other width or head_dim, a q|k count other than
/// twice the q channels, or q channels off the 256 grid is refused.
#[test]
fn the_shape_check_takes_only_the_tile_shapes() {
    assert_eq!(chunk64_conv_shape_refusal(4, 128, 512, 256), None);
    assert_eq!(chunk64_conv_shape_refusal(4, 128, 5632, 2816), None);
    for (d_conv, hd, qk, part) in [
        (3, 128, 512, 256),
        (2, 128, 512, 256),
        (4, 64, 512, 256),
        (4, 128, 768, 256),
        (4, 128, 256, 128),
        (4, 128, 0, 0),
    ] {
        assert!(
            chunk64_conv_shape_refusal(d_conv, hd, qk, part).is_some(),
            "{d_conv} {hd} {qk} {part}"
        );
    }
}

/// 2026-10-14: The lever: unset and `0` off, `1` on, anything else an error.
#[test]
fn the_lever_parses_strictly() {
    assert_eq!(parse_chunk64_conv(None), Ok(false));
    assert_eq!(parse_chunk64_conv(Some("0")), Ok(false));
    assert_eq!(parse_chunk64_conv(Some("1")), Ok(true));
    for bad in ["on", "true", "2", ""] {
        assert!(parse_chunk64_conv(Some(bad)).is_err(), "{bad}");
    }
}
