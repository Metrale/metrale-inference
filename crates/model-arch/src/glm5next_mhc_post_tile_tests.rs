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
        assert_eq!(parse(None, None, r), Ok(None));
        assert_eq!(parse(Some("0"), None, r), Ok(None));
    }
    assert_eq!(parse(Some("1"), Some("576"), ALL), Ok(Some(576)));
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
        let err = parse(Some("1"), Some("576"), r).unwrap_err();
        assert!(err.contains(name), "{r:?}: {err}");
    }
    for bad in ["2", "on", "true", ""] {
        let err = parse(Some(bad), Some("576"), ALL).unwrap_err();
        assert!(err.contains(LEVER), "{bad:?}: {err}");
    }
}

/// 2026-10-16: On, the threshold is required (no default) and at least tile4's floor; set with
/// the lever off it is refused rather than ignored.
#[test]
fn the_threshold_is_explicit_and_no_lower_than_the_floor() {
    let floor = HC_PRE_NORM_ROWS_MIN_TOKENS as u32;
    let err = parse(Some("1"), None, ALL).unwrap_err();
    assert!(err.contains(MIN_TOKENS_ENV), "{err}");
    assert_eq!(
        parse(Some("1"), Some(&floor.to_string()), ALL),
        Ok(Some(floor))
    );
    assert_eq!(parse(Some("1"), Some("4096"), ALL), Ok(Some(4096)));
    let below = (floor - 1).to_string();
    for bad in [below.as_str(), "0", "-1", "", "576.0", "x", " 576"] {
        let err = parse(Some("1"), Some(bad), ALL).unwrap_err();
        assert!(err.contains(MIN_TOKENS_ENV), "{bad:?}: {err}");
    }
    for v in [None, Some("0")] {
        let err = parse(v, Some("576"), ALL).unwrap_err();
        assert!(
            err.contains(MIN_TOKENS_ENV) && err.contains(LEVER),
            "{v:?}: {err}"
        );
    }
}

/// 2026-10-15: On without the kernel fails the load and the first wide piece; off never looks.
#[test]
fn on_without_the_kernel_is_an_error_not_a_fallback() {
    assert!(load_check(false, KernelHandle(0)).is_ok());
    assert!(load_check(true, KernelHandle(TILE)).is_ok());
    let err = load_check(true, KernelHandle(0)).unwrap_err().to_string();
    assert!(err.contains(WIRED.name()), "{err}");

    let gpu = MockGpuBackend::new();
    let mut k = Glm5NextMhcKernels::resolve(&gpu).expect("mock resolves every kernel");
    k.hc_post_pre_tile = KernelHandle(TILE);
    let pick = |k: &Glm5NextMhcKernels, on: Option<u32>, t: u32| {
        kernel_for(k, on, t).unwrap().map(|h| h.0)
    };
    let min = HC_PRE_NORM_ROWS_MIN_TOKENS as u32;
    assert_eq!(pick(&k, None, 2736), None);
    assert_eq!(pick(&k, Some(min), min - 1), None);
    assert_eq!(pick(&k, Some(min), min), Some(TILE));
    assert_eq!(pick(&k, Some(min), 2736), Some(TILE));
    // 2026-10-16: The threshold splits c1k's 345-row pieces from c32k's 683-row ones.
    assert_eq!(pick(&k, Some(576), 347), None);
    assert_eq!(pick(&k, Some(576), 575), None);
    assert_eq!(pick(&k, Some(576), 576), Some(TILE));
    assert_eq!(pick(&k, Some(576), 684), Some(TILE));
    // 2026-10-16: Below tile4's floor whatever the caller passes.
    assert_eq!(pick(&k, Some(1), min - 1), None);
    let mut absent = k;
    absent.hc_post_pre_tile = KernelHandle(0);
    assert_eq!(pick(&absent, Some(min), min - 1), None);
    assert_eq!(pick(&absent, Some(576), 575), None);
    assert!(kernel_for(&absent, Some(min), min).is_err());
    assert!(kernel_for(&absent, Some(576), 576).is_err());
}

/// 2026-10-15: Every condition of `refusal` refuses on its own; the control shape passes.
#[test]
fn each_shape_the_kernel_cannot_take_is_refused() {
    let ok = Shape {
        hidden: 4096,
        hc_mult: 4,
        hc_fn_bf16: true,
        hc_fn: 0x100,
        ring: true,
        streams_out: 0x5000,
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
            streams_out: 0x5004,
            ..ok
        },
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
    // 2026-10-15: The edges themselves are taken; only a ring needs aligned streams.
    for edge in [
        Shape {
            ring: false,
            streams_out: 0x5004,
            ..ok
        },
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

/// 2026-10-15: `GlmHcptLayout<NG>::BYTES` and `GlmHcprLayout::BYTES` worked by hand from the
/// kernel sources: one register group fits two blocks per GB10 SM (102400 B, 1 KiB reserved per
/// block), as `__launch_bounds__(256, 2)` assumes; two groups and a ring fit the 101376 B
/// per-block opt-in, a ring one block per SM (`__launch_bounds__(256, 1)`).
#[test]
fn the_shared_memory_sum_is_the_kernel_layout() {
    let [t1, t2, r0, r1, r1e] = ENTRIES.map(Entry::smem_bytes);
    // ring 4 x 24 x 256 x 2 = 49152 > red 25 x 256 x 4 = 25600; tail (24 + 4 + 33 + 4 + 16) x 4.
    assert_eq!(t1, 49152 + 324);
    // red 2 x 25600 = 51200 > ring 49152; tail 2 x 324.
    assert_eq!(t2, 51200 + 648);
    // tile4: ring 4 x (24 x 256 x 2 + 4 x 256 x 4) = 65536, then (4 x 24 + 4 + 33) x 4 = 66068;
    // rounded up to 66080, then 4 tokens x (4 + 16) x 4.
    assert_eq!(r0, 66080 + 320);
    assert_eq!([r1, r1e], [r0, r0]);
    assert!(2 * (t1 + 1024) <= 102_400);
    assert!(t2 <= 101_376 && r0 <= 101_376);
    assert!(2 * (r0 + 1024) > 102_400);
}

/// 2026-10-15: Names and modules are the kernel sources' (a wrong one resolves to no kernel and
/// fails the load); the rings cover four tokens per block up to one block per SM.
#[test]
fn each_entry_names_its_kernel_and_grid() {
    let names = ENTRIES.map(Entry::name);
    let want =
        ["tile1", "tile2", "ring0", "ring1", "ring1e"].map(|s| format!("glm5next_hc_post_pre_{s}"));
    assert_eq!(names.map(String::from), want);
    for e in ENTRIES {
        let module = if e.is_ring() {
            GLM5NEXT_MHC_POST_RING_MODULE
        } else {
            GLM5NEXT_MHC_POST_TILE_MODULE
        };
        assert_eq!(e.module(), module, "{e:?}");
        assert_eq!(e.block(), if e == Entry::Tile2 { 512 } else { 256 });
    }
    assert_eq!(WIRED, Entry::Tile1);
    let grid = |e: Entry, t| e.grid(t, 48);
    assert_eq!([grid(Entry::Tile1, 1), grid(Entry::Tile1, 347)], [1, 347]);
    assert_eq!([grid(Entry::Tile2, 1), grid(Entry::Tile2, 347)], [1, 174]);
    for r in [Entry::Ring0, Entry::Ring1, Entry::Ring1e] {
        assert_eq!(
            [grid(r, 1), grid(r, 4), grid(r, 5), grid(r, 100)],
            [1, 1, 2, 25]
        );
        assert_eq!(
            [grid(r, 189), grid(r, 192), grid(r, 193), grid(r, 2736)],
            [48, 48, 48, 48]
        );
        assert_eq!(r.grid(2736, 20), 20);
    }
}

/// 2026-10-15: The parameter list in the kernel's order (GLM_HCPT_ARGS): vec8's post operands
/// with `write_x` after the streams, then the pre-norm's. A swapped pointer reads the wrong
/// buffer with no error; the grid and block are the entry's, a ring's from the mock's 48 SMs.
#[test]
fn the_launch_passes_the_kernel_arguments_in_order() {
    let gpu = MockGpuBackend::new();
    let w = site(true, 0x100);
    let cases = [
        (Entry::Tile1, false, 200, 200),
        (Entry::Tile2, true, 201, 101),
        (Entry::Ring0, true, 100, 25),
        (Entry::Ring1, false, 256, 48),
        (Entry::Ring1e, true, 193, 48),
    ];
    for (entry, write_x, t, grid) in cases {
        launch(
            &gpu,
            KernelHandle(TILE),
            entry,
            &args(&w, &PEERS, t),
            write_x,
            9,
        )
        .unwrap();
        let l = gpu.launches_snapshot().pop().expect("one launch");
        assert_eq!(l.func, TILE);
        assert_eq!(l.grid, [grid, 1, 1]);
        assert_eq!(l.block, [entry.block(), 1, 1]);
        assert_eq!(l.shared_mem, entry.smem_bytes());
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

/// 2026-10-15: A refused launch launches nothing: a zero handle, an FP32 or misaligned hc_fn,
/// misaligned streams on a ring entry, a shape `refusal` names.
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
    let go = |k: u64, e: Entry, w: &Glm5NextMhcSiteWeights, p: &PeerPartials| {
        launch(&gpu, KernelHandle(k), e, &args(w, p, 128), false, 0)
    };
    for e in [Entry::Tile1, Entry::Ring1] {
        assert!(go(0, e, &bf16, &PEERS).is_err());
        assert!(go(TILE, e, &f32w, &PEERS).is_err());
        assert!(go(TILE, e, &skew, &PEERS).is_err());
        assert!(go(TILE, e, &bf16, &five).is_err());
        let mut wide = args(&bf16, &PEERS, 128);
        wide.dims = [4096, 2, 20];
        assert!(launch(&gpu, KernelHandle(TILE), e, &wide, false, 0).is_err());
    }
    let mut skewed = args(&bf16, &PEERS, 128);
    skewed.streams_out = DevicePtr(0x5004);
    assert!(launch(&gpu, KernelHandle(TILE), Entry::Ring0, &skewed, false, 0).is_err());
    assert_eq!(gpu.launch_count(), 0);
    launch(&gpu, KernelHandle(TILE), Entry::Tile1, &skewed, false, 0).unwrap();
    go(TILE, Entry::Ring1, &bf16, &PEERS).unwrap();
    assert_eq!(gpu.launch_count(), 2, "the controls launch");
}

/// 2026-10-16: The per-piece choice the caller acts on; `None` runs the pair (vec8 + tile4).
type Pick = fn(&Glm5NextMhcKernels, Option<u32>, u32) -> Result<Option<KernelHandle>>;

/// 2026-10-16: The `(threshold, rows)` cases where `pick` fuses when it should run the pair, or
/// the reverse. c1k's pieces are ~345 rows, c32k's ~683.
fn threshold_misses(pick: Pick, k: &Glm5NextMhcKernels) -> Vec<(u32, u32)> {
    let cases = [
        (576, 64, false),
        (576, 345, false),
        (576, 347, false),
        (576, 575, false),
        (576, 576, true),
        (576, 684, true),
        (576, 2736, true),
        (64, 63, false),
        (64, 64, true),
        (1, 63, false),
        (1, 64, true),
    ];
    let fuses = |min, t| matches!(pick(k, Some(min), t), Ok(Some(_)));
    cases
        .into_iter()
        .filter(|&(min, t, want)| fuses(min, t) != want)
        .map(|(min, t, _)| (min, t))
        .collect()
}

/// 2026-10-16: Pieces below the threshold (or tile4's floor) run the pair, the rest fuse; each
/// mutant of the comparison is caught (negative controls).
#[test]
fn pieces_below_the_threshold_run_the_pair() {
    let gpu = MockGpuBackend::new();
    let mut k = Glm5NextMhcKernels::resolve(&gpu).expect("mock resolves every kernel");
    k.hc_post_pre_tile = KernelHandle(TILE);
    assert_eq!(threshold_misses(kernel_for, &k), vec![]);
    let mutants: [(&str, Pick); 3] = [
        ("threshold ignored", |k, m, t| {
            kernel_for(k, m.map(|_| 1), t)
        }),
        ("off by one", |k, m, t| kernel_for(k, m.map(|m| m + 1), t)),
        ("floor ignored", |k, m, t| {
            Ok(m.filter(|&m| t >= m).map(|_| k.hc_post_pre_tile))
        }),
    ];
    for (name, pick) in mutants {
        assert!(!threshold_misses(pick, &k).is_empty(), "{name} not caught");
    }
}
