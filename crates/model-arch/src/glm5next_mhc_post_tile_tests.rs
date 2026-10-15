// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: CPU tests of `glm5next_mhc_post_tile`: the lever's parse and load check, the
//! per-piece kernel choice, the launch refusals, the shared-memory sum and the argument list
//! against the mock backend. The kernel's bytes are checked against vec8 + tile4 on a GPU by
//! `examples/glm5next_hc_post_pre_tile_gate.rs`.

use super::*;
use crate::glm5next_mhc::Glm5NextMhcSiteWeights;
use metrale_comm::PeerPartials;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};

const TILE: u64 = 0xA7;

const ALL: Requires = Requires {
    sp_mhc: true,
    fused: true,
    pre_tile: true,
    post_vec: true,
    post_pre_fused: false,
};

const PEERS: PeerPartials = PeerPartials {
    recv: 0x9000,
    slot_elems: 2736 * 4096,
    world_size: 3,
    rank: 1,
};

fn site(bf16: bool, hc_fn: u64) -> Glm5NextMhcSiteWeights {
    Glm5NextMhcSiteWeights {
        hc_fn: DevicePtr(hc_fn),
        hc_fn_bf16: bf16,
        hc_scale: DevicePtr(0x200),
        hc_base: DevicePtr(0x300),
        mix: DevicePtr(0x400),
        done: DevicePtr(0x500),
    }
}

fn args<'a>(site: &'a Glm5NextMhcSiteWeights, peers: &'a PeerPartials, t: u32) -> PostPreTok<'a> {
    PostPreTok {
        block_out: DevicePtr(0x1000),
        partials: peers,
        residual: DevicePtr(0x2000),
        post: DevicePtr(0x3000),
        comb: DevicePtr(0x4000),
        streams_out: DevicePtr(0x5000),
        site,
        norm_w: DevicePtr(0x6000),
        y: DevicePtr(0x7000),
        post_out: DevicePtr(0x8000),
        comb_out: DevicePtr(0x8800),
        normed: DevicePtr(0xA000),
        num_tokens: t,
        dims: [4096, 4, 20],
        eps: [1e-5, 1e-6, 2e-5],
    }
}

/// 2026-10-15: Unset or `0` is off whatever the other levers say; `1` is on only with the whole
/// pair it replaces on and the other fused pair off; any other value is refused.
#[test]
fn the_lever_needs_the_pair_it_replaces() {
    for r in [ALL, Requires::default()] {
        assert_eq!(parse(None, r), Ok(false));
        assert_eq!(parse(Some("0"), r), Ok(false));
    }
    assert_eq!(parse(Some("1"), ALL), Ok(true));
    let cases = [
        (
            Requires {
                sp_mhc: false,
                ..ALL
            },
            "METRALE_GLM_SP_MHC=1",
        ),
        (
            Requires {
                fused: false,
                ..ALL
            },
            "METRALE_GLM_MHC_FUSED=1",
        ),
        (
            Requires {
                pre_tile: false,
                ..ALL
            },
            "METRALE_GLM_MHC_PRE_TILE=1",
        ),
        (
            Requires {
                post_vec: false,
                ..ALL
            },
            "METRALE_GLM_MHC_POST_VEC=1",
        ),
        (
            Requires {
                post_pre_fused: true,
                ..ALL
            },
            "METRALE_GLM_MHC_POST_PRE_FUSED=1",
        ),
    ];
    for (r, name) in cases {
        let err = parse(Some("1"), r).unwrap_err();
        assert!(err.contains(name), "{r:?}: {err}");
    }
    for bad in ["2", "on", "true", ""] {
        let err = parse(Some(bad), ALL).unwrap_err();
        assert!(err.contains(LEVER), "{bad:?}: {err}");
    }
}

/// 2026-10-15: On without the kernel fails the load and the first wide piece; off never looks.
#[test]
fn on_without_the_kernel_is_an_error_not_a_fallback() {
    assert!(load_check(false, KernelHandle(0)).is_ok());
    assert!(load_check(true, KernelHandle(TILE)).is_ok());
    let err = load_check(true, KernelHandle(0)).unwrap_err().to_string();
    assert!(err.contains(KERNELS[GROUPS as usize - 1]), "{err}");

    let gpu = MockGpuBackend::new();
    let mut k = Glm5NextMhcKernels::resolve(&gpu).expect("mock resolves every kernel");
    k.hc_post_pre_tile = KernelHandle(TILE);
    let pick =
        |k: &Glm5NextMhcKernels, on: bool, t: u32| kernel_for(k, on, t).unwrap().map(|h| h.0);
    let min = HC_PRE_NORM_ROWS_MIN_TOKENS as u32;
    assert_eq!(pick(&k, false, 2736), None);
    assert_eq!(pick(&k, true, min - 1), None);
    assert_eq!(pick(&k, true, min), Some(TILE));
    assert_eq!(pick(&k, true, 2736), Some(TILE));
    let mut absent = k;
    absent.hc_post_pre_tile = KernelHandle(0);
    assert_eq!(pick(&absent, true, min - 1), None);
    assert!(kernel_for(&absent, true, min).is_err());
}

/// 2026-10-15: Every condition of `refusal` refuses on its own; the control shape passes.
#[test]
fn each_shape_the_kernel_cannot_take_is_refused() {
    let ok = Shape {
        hidden: 4096,
        hc_mult: 4,
        hc_fn_bf16: true,
        hc_fn: 0x100,
        num_tokens: 684,
        mix_cap: 2736,
        slot_elems: 2736 * 4096,
        world_size: 3,
        rank: 2,
    };
    assert_eq!(refusal(&ok), None);
    let mutants = [
        Shape { hidden: 2048, ..ok },
        Shape { hc_mult: 2, ..ok },
        Shape {
            hc_fn_bf16: false,
            ..ok
        },
        Shape { hc_fn: 0x108, ..ok },
        Shape {
            num_tokens: 0,
            ..ok
        },
        Shape {
            num_tokens: 2737,
            ..ok
        },
        Shape {
            slot_elems: 684 * 4096 - 1,
            ..ok
        },
        Shape {
            world_size: 5,
            rank: 0,
            ..ok
        },
        Shape {
            world_size: 0,
            rank: 0,
            ..ok
        },
        Shape { rank: 3, ..ok },
    ];
    for m in mutants {
        assert!(refusal(&m).is_some(), "{m:?} was taken");
    }
    // 2026-10-15: The edges themselves are taken.
    for edge in [
        Shape {
            num_tokens: 2736,
            ..ok
        },
        Shape {
            slot_elems: 684 * 4096,
            ..ok
        },
        Shape {
            world_size: 4,
            rank: 3,
            ..ok
        },
        Shape {
            world_size: 1,
            rank: 0,
            ..ok
        },
    ] {
        assert_eq!(refusal(&edge), None, "{edge:?}");
    }
}

/// 2026-10-15: `GlmHcptLayout<NG>::BYTES` worked by hand from the kernel source: one group fits
/// two blocks per GB10 SM (102400 B, 1 KiB reserved per block), as `__launch_bounds__(256, 2)`
/// assumes, and two groups fit the 101376 B per-block opt-in.
#[test]
fn the_shared_memory_sum_is_the_kernel_layout() {
    // ring 4 x 24 x 256 x 2 = 49152 > red 25 x 256 x 4 = 25600; tail (24 + 4 + 33 + 4 + 16) x 4.
    assert_eq!(smem_bytes(1), 49152 + 324);
    // red 2 x 25600 = 51200 > ring 49152; tail 2 x 324.
    assert_eq!(smem_bytes(2), 51200 + 648);
    assert!(2 * (smem_bytes(1) + 1024) <= 102_400);
    assert!(smem_bytes(2) <= 101_376);
    assert!(kernel_name(0).is_err() && kernel_name(3).is_err());
}

/// 2026-10-15: The parameter list in the kernel's order (GLM_HCPT_ARGS): vec8's post operands
/// with `write_x` after the streams, then the pre-norm's. A swapped pointer reads the wrong
/// buffer with no error; the grid is one block per `groups` tokens.
#[test]
fn the_launch_passes_the_kernel_arguments_in_order() {
    let gpu = MockGpuBackend::new();
    let w = site(true, 0x100);
    for (groups, write_x, t, grid) in [(1, false, 200, 200), (2, true, 201, 101)] {
        launch(
            &gpu,
            KernelHandle(TILE),
            groups,
            &args(&w, &PEERS, t),
            write_x,
            9,
        )
        .unwrap();
        let l = gpu.launches_snapshot().pop().expect("one launch");
        assert_eq!(l.func, TILE);
        assert_eq!(l.grid, [grid, 1, 1]);
        assert_eq!(l.block, [256 * groups, 1, 1]);
        assert_eq!(l.shared_mem, smem_bytes(groups));
        assert_eq!(l.stream, 9);
        let buf = |p: u64| MockArg::Buffer(DevicePtr(p));
        let le = |v: &[u8]| MockArg::Bytes(v.to_vec());
        let want = vec![
            buf(0x1000),
            buf(0x9000),
            le(&(PEERS.slot_elems as u64).to_le_bytes()),
            le(&3i32.to_le_bytes()),
            le(&1i32.to_le_bytes()),
            buf(0x2000),
            buf(0x3000),
            buf(0x4000),
            buf(0x5000),
            le(&i32::from(write_x).to_le_bytes()),
            buf(0x100),
            buf(0x200),
            buf(0x300),
            buf(0x400),
            buf(0x6000),
            buf(0x7000),
            buf(0x8000),
            buf(0x8800),
            buf(0xA000),
            le(&4096u32.to_le_bytes()),
            le(&20u32.to_le_bytes()),
            le(&1e-5f32.to_le_bytes()),
            le(&1e-6f32.to_le_bytes()),
            le(&2e-5f32.to_le_bytes()),
            le(&t.to_le_bytes()),
        ];
        assert_eq!(l.args, want);
    }
}

/// 2026-10-15: A refused launch launches nothing: a zero handle, a bad group count, an FP32 or
/// misaligned hc_fn, a shape `refusal` names.
#[test]
fn a_refused_launch_launches_nothing() {
    let gpu = MockGpuBackend::new();
    let bf16 = site(true, 0x100);
    let f32w = site(false, 0x100);
    let skew = site(true, 0x104);
    let five = PeerPartials {
        world_size: 5,
        ..PEERS
    };
    let go = |k: u64, g: u32, w: &Glm5NextMhcSiteWeights, p: &PeerPartials| {
        launch(&gpu, KernelHandle(k), g, &args(w, p, 128), false, 0)
    };
    assert!(go(0, 1, &bf16, &PEERS).is_err());
    assert!(go(TILE, 3, &bf16, &PEERS).is_err());
    assert!(go(TILE, 1, &f32w, &PEERS).is_err());
    assert!(go(TILE, 1, &skew, &PEERS).is_err());
    assert!(go(TILE, 1, &bf16, &five).is_err());
    let mut wide = args(&bf16, &PEERS, 128);
    wide.dims = [4096, 2, 20];
    assert!(launch(&gpu, KernelHandle(TILE), 1, &wide, false, 0).is_err());
    assert_eq!(gpu.launch_count(), 0);
    go(TILE, 1, &bf16, &PEERS).unwrap();
    assert_eq!(gpu.launch_count(), 1, "the control launches");
}
