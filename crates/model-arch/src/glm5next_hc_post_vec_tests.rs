// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: CPU tests of `glm5next_hc_post_vec`: the lever's parse, the kernel choice, the
//! launch shape and the argument list against the mock backend. The kernel's bytes are checked
//! against the scalar kernel on a GPU by `examples/glm5next_hc_post_sum_gate.rs`.

use super::*;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};

const VEC8: u64 = 0x77;

fn kernels(gpu: &MockGpuBackend) -> Glm5NextMhcKernels {
    let mut k = Glm5NextMhcKernels::resolve(gpu).expect("mock resolves every kernel");
    k.hc_post_rank_sum_vec8 = KernelHandle(VEC8);
    k
}

const PEERS: PeerPartials = PeerPartials {
    recv: 0x9000,
    slot_elems: 2736 * 4096,
    world_size: 3,
    rank: 1,
};

const STREAMS: [DevicePtr; 4] = [
    DevicePtr(0x2000),
    DevicePtr(0x3000),
    DevicePtr(0x4000),
    DevicePtr(0x5000),
];

/// 2026-10-15: Unset or `0` is off and `1` on; any other value is refused, never read as off.
#[test]
fn the_lever_is_zero_or_one() {
    assert_eq!(parse(None), Ok(false));
    assert_eq!(parse(Some("0")), Ok(false));
    assert_eq!(parse(Some("1")), Ok(true));
    for bad in ["2", "on", "true", ""] {
        let err = parse(Some(bad)).unwrap_err();
        assert!(err.contains(LEVER), "{bad:?}: {err}");
    }
}

/// 2026-10-15: Off keeps the scalar kernel; on takes the vectorized one, and on without it in the
/// build is an error rather than a silent fallback.
#[test]
fn the_vectorized_kernel_needs_the_lever_and_the_build() {
    let gpu = MockGpuBackend::new();
    let k = kernels(&gpu);
    let pick = |k: &Glm5NextMhcKernels, on: bool| kernel_for(k, on).unwrap().map(|h| h.0);
    assert_eq!(pick(&k, false), None);
    assert_eq!(pick(&k, true), Some(VEC8));
    let mut absent = k;
    absent.hc_post_rank_sum_vec8 = KernelHandle(0);
    assert_eq!(pick(&absent, false), None);
    let err = kernel_for(&absent, true).unwrap_err().to_string();
    assert!(err.contains(KERNEL), "{err}");
}

/// 2026-10-15: 4096 columns are four 128-thread blocks per row, so an 86-row piece still fills
/// 344 blocks; a width that is not a whole block rounds up.
#[test]
fn a_row_is_four_blocks_at_4096_columns() {
    assert_eq!(launch_shape(86, 4096), ([86, 4, 1], [128, 1, 1]));
    assert_eq!(launch_shape(8208, 4096), ([8208, 4, 1], [128, 1, 1]));
    assert_eq!(launch_shape(1, 1032).0, [1, 2, 1]);
    assert_eq!(launch_shape(0, 4096).0, [1, 4, 1]);
}

/// 2026-10-15: The kernel's parameter list in order: the scalar kernel's, then `write_x`. A
/// swapped pointer would read the wrong buffer with no error, and a dropped flag would read
/// garbage for `write_x`.
#[test]
fn the_launch_passes_the_scalar_arguments_then_write_x() {
    let gpu = MockGpuBackend::new();
    for write_x in [false, true] {
        launch(
            &gpu,
            KernelHandle(VEC8),
            DevicePtr(0x1000),
            &PEERS,
            STREAMS,
            [341, 4096, 4],
            write_x,
            7,
        )
        .unwrap();
        let l = gpu.launches_snapshot().pop().expect("one launch");
        assert_eq!(l.func, VEC8);
        assert_eq!(l.grid, [341, 4, 1]);
        assert_eq!(l.block, [128, 1, 1]);
        assert_eq!(l.stream, 7);
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
            le(&4096u32.to_le_bytes()),
            le(&4u32.to_le_bytes()),
            le(&341u32.to_le_bytes()),
            le(&i32::from(write_x).to_le_bytes()),
        ];
        assert_eq!(l.args, want);
    }
}

/// 2026-10-15: What the kernel's 16-byte accesses or register bounds cannot take is refused
/// before any launch.
#[test]
fn the_launch_refuses_shapes_the_kernel_cannot_take() {
    let gpu = MockGpuBackend::new();
    let go = |kernel: u64, out: DevicePtr, peers: &PeerPartials, dims: [u32; 3]| {
        launch(
            &gpu,
            KernelHandle(kernel),
            out,
            peers,
            STREAMS,
            dims,
            false,
            0,
        )
    };
    let refused = |r: Result<()>| {
        let err = r.unwrap_err().to_string();
        assert_eq!(gpu.launch_count(), 0, "refused, yet launched: {err}");
        err
    };
    let out = DevicePtr(0x1000);
    assert!(refused(go(0, out, &PEERS, [256, 4096, 4])).contains("not loaded"));
    assert!(refused(go(VEC8, out, &PEERS, [2737, 4096, 4])).contains("peer slot"));
    assert!(refused(go(VEC8, out, &PEERS, [256, 4092, 4])).contains("multiples"));
    let odd_slot = PeerPartials {
        slot_elems: 2736 * 4096 + 4,
        ..PEERS
    };
    assert!(refused(go(VEC8, out, &odd_slot, [256, 4096, 4])).contains("multiples"));
    let many = PeerPartials {
        world_size: MAX_RANKS + 1,
        ..PEERS
    };
    assert!(refused(go(VEC8, out, &many, [256, 4096, 4])).contains("ranks"));
    assert!(refused(go(VEC8, out, &PEERS, [256, 4096, 5])).contains("hc"));
    let skew = PeerPartials {
        recv: 0x9008,
        ..PEERS
    };
    assert!(refused(go(VEC8, out, &skew, [256, 4096, 4])).contains("aligned"));
    assert!(refused(go(VEC8, DevicePtr(0x1004), &PEERS, [256, 4096, 4])).contains("aligned"));
    go(VEC8, out, &PEERS, [256, 4096, 4]).unwrap();
    assert_eq!(gpu.launch_count(), 1, "the control launches");
}
