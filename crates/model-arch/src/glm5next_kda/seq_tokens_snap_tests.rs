// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: `stateful_tokens_snap`'s conv intermediates under `METRALE_GLM_KDA_CONV_SNAP_TO`:
//! on, one `causal_conv1d_window_advance_to` per intermediate from the live window into the
//! snapshot and no copy; off (or the kernel absent), the copy then the in-place advance. The bytes
//! are checked on a GPU by the `pdl_chain_gate` example.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::super::Glm5NextKdaWorkspace;
use super::super::decode_rows_tests::{
    CONV_WINDOW, CONV_WINDOW_TO, cfg, layer, seq_state, since,
};

const RECUR_SNAP: u64 = 0x327;

fn u32_arg(v: u32) -> MockArg {
    MockArg::Bytes(v.to_le_bytes().to_vec())
}

/// 2026-10-21: The conv-window launches and the device copies of one K-row verify with its K - 1
/// snapshots, the lever and the fused kernel as given.
#[allow(clippy::type_complexity)]
fn run(
    k: usize,
    lever: bool,
    kernel: bool,
) -> (Vec<(u64, Vec<MockArg>)>, usize, Vec<DevicePtr>, DevicePtr) {
    let gpu = MockGpuBackend::new();
    let mut l = layer(&gpu);
    l.kernels.seq.recurrent_snap = KernelHandle(RECUR_SNAP);
    l.levers.kda.conv_snap_to = lever;
    if !kernel {
        l.kernels.seq.conv_window_to = KernelHandle(0);
    }
    let c = cfg();
    let ws = Glm5NextKdaWorkspace::new(&gpu, &c, 16).unwrap();
    let st = seq_state(&gpu);
    let snaps: Vec<_> = (1..k)
        .map(|_| {
            (
                gpu.alloc(c.recurrent_state_elems() * 4).unwrap(),
                gpu.alloc(c.conv_state_elems() * 4).unwrap(),
            )
        })
        .collect();
    let (from, copies) = (gpu.launch_count(), gpu.d2d_count());
    l.stateful_tokens_snap(&gpu, k, &st, &ws, &snaps, 0).unwrap();
    let win = since(&gpu, from)
        .into_iter()
        .filter(|x| [CONV_WINDOW, CONV_WINDOW_TO].contains(&x.func))
        .map(|x| (x.func, x.args))
        .collect();
    let dsts = snaps.iter().map(|s| s.1).collect();
    (win, gpu.d2d_count() - copies, dsts, st.conv)
}

#[test]
fn on_each_intermediate_is_one_advance_to_from_the_live_window_and_no_copy() {
    let cd = cfg().conv_dim() as u32;
    for k in 2..=4usize {
        let (win, copies, dsts, live) = run(k, true, true);
        assert_eq!(copies, 0, "k={k}");
        // 2026-10-21: k - 1 intermediates, then the live window's own advance (in place).
        assert_eq!(win.len(), k, "k={k}");
        for (t, (f, a)) in win[..k - 1].iter().enumerate() {
            assert_eq!(*f, CONV_WINDOW_TO, "k={k} t={t}");
            assert_eq!(a[0], MockArg::Buffer(live));
            assert_eq!(a[1], MockArg::Buffer(dsts[t]));
            assert_eq!(a[3..], [u32_arg(cd), u32_arg(4), u32_arg(t as u32 + 1)]);
        }
        assert_eq!(win[k - 1].0, CONV_WINDOW, "k={k}: the live window");
    }
}

#[test]
fn off_or_without_the_kernel_the_copy_and_the_in_place_advance_run() {
    for (lever, kernel) in [(false, true), (true, false)] {
        let (win, copies, dsts, _) = run(3, lever, kernel);
        assert_eq!(copies, 2, "lever={lever} kernel={kernel}");
        assert!(win.iter().all(|(f, _)| *f == CONV_WINDOW));
        for (t, (_, a)) in win[..2].iter().enumerate() {
            assert_eq!(a[0], MockArg::Buffer(dsts[t]));
            assert_eq!(a[4], u32_arg(t as u32 + 1));
        }
    }
}
