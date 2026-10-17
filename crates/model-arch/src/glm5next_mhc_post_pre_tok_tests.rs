// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: CPU tests of `glm5next_mhc_post_pre_tok`: the per-piece kernel
//! choice and the launch's argument list against the mock backend. The kernel's bytes are
//! checked on a GPU by `examples/glm5next_hc_post_pre_tok_bench.rs`.

use super::*;
use crate::glm5next_mhc::MHC_MIX_MAX_TOKENS;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};

const FUSED: u64 = 0x99;

fn kernels(gpu: &MockGpuBackend) -> Glm5NextMhcKernels {
    let mut k = Glm5NextMhcKernels::resolve(gpu, crate::glm5next_mhc_levers::MhcLevers::declared())
        .expect("mock resolves every kernel");
    k.hc_post_pre_norm_tok = KernelHandle(FUSED);
    k
}

fn site(bf16: bool) -> Glm5NextMhcSiteWeights {
    Glm5NextMhcSiteWeights {
        hc_fn: DevicePtr(0x100),
        hc_fn_bf16: bf16,
        hc_scale: DevicePtr(0x200),
        hc_base: DevicePtr(0x300),
        mix: DevicePtr(0x400),
        done: DevicePtr(0x500),
        mix_tokens: crate::glm5next_mhc::MHC_MIX_MAX_TOKENS,
    }
}

const PEERS: PeerPartials = PeerPartials {
    recv: 0x9000,
    slot_elems: 2736 * 4096,
    world_size: 3,
    rank: 1,
};

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
        post_out: DevicePtr(0x7100),
        comb_out: DevicePtr(0x7200),
        normed: DevicePtr(0x8000),
        num_tokens: t,
        dims: [4096, 4, 20],
        eps: [1e-6, 1e-6, 1e-5],
    }
}

/// 2026-10-14: The fused kernel only with the lever on and a piece of at least
/// `HC_PRE_NORM_ROWS_MIN_TOKENS` rows; on without the kernel is an error, not the pair.
#[test]
fn the_fused_kernel_needs_the_lever_wide_rows_and_the_build() {
    let gpu = MockGpuBackend::new();
    let k = kernels(&gpu);
    let min = HC_PRE_NORM_ROWS_MIN_TOKENS as u32;
    let pick =
        |k: &Glm5NextMhcKernels, on: bool, n: u32| kernel_for(k, on, n).unwrap().map(|h| h.0);
    assert_eq!(pick(&k, true, min), Some(FUSED));
    assert_eq!(pick(&k, true, 2736), Some(FUSED));
    assert_eq!(pick(&k, true, min - 1), None);
    assert_eq!(pick(&k, false, 2736), None);
    let mut absent = k;
    absent.hc_post_pre_norm_tok = KernelHandle(0);
    assert_eq!(pick(&absent, false, 2736), None);
    assert_eq!(pick(&absent, true, min - 1), None);
    let err = kernel_for(&absent, true, 2736).unwrap_err().to_string();
    assert!(err.contains("glm5next_hc_post_pre_norm_tok4"), "{err}");
}

/// 2026-10-14: The kernel's parameter list in order (`glm5next_hc_post_pre_norm_tok4`): the
/// rank-sum post-mix's, then the tok4 pre-norm's after its streams, then the token count, on
/// `ceil(T / 4)` blocks of 256. A swapped pointer would read the wrong buffer with no error.
#[test]
fn the_launch_passes_the_post_then_the_pre_norm_arguments() {
    let gpu = MockGpuBackend::new();
    let s = site(true);
    launch(&gpu, KernelHandle(FUSED), &args(&s, &PEERS, 255), 7).unwrap();
    let l = gpu.launches_snapshot().pop().expect("one launch");
    assert_eq!(l.func, FUSED);
    assert_eq!(l.grid, [64, 1, 1]);
    assert_eq!(l.block, [256, 1, 1]);
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
        buf(0x100),
        buf(0x200),
        buf(0x300),
        buf(0x400),
        buf(0x500),
        buf(0x6000),
        buf(0x7000),
        buf(0x7100),
        buf(0x7200),
        buf(0x8000),
        le(&4096u32.to_le_bytes()),
        le(&4u32.to_le_bytes()),
        le(&20u32.to_le_bytes()),
        le(&1e-6f32.to_le_bytes()),
        le(&1e-6f32.to_le_bytes()),
        le(&1e-5f32.to_le_bytes()),
        le(&255u32.to_le_bytes()),
    ];
    assert_eq!(l.args, want);
}

/// 2026-10-14: What either half of the pair refuses is refused before any launch.
#[test]
fn the_launch_refuses_what_the_pair_refuses() {
    let gpu = MockGpuBackend::new();
    let bf16 = site(true);
    let fp32 = site(false);
    let refused = |a: PostPreTok| {
        let err = launch(&gpu, KernelHandle(FUSED), &a, 0)
            .unwrap_err()
            .to_string();
        assert_eq!(gpu.launch_count(), 0, "refused, yet launched: {err}");
        err
    };
    assert!(refused(args(&fp32, &PEERS, 256)).contains("FP32"));
    let mut shape = args(&bf16, &PEERS, 256);
    shape.dims = [4096, 5, 20];
    assert!(refused(shape).contains("hc_mult"));
    let narrow = PeerPartials {
        slot_elems: 255 * 4096,
        ..PEERS
    };
    assert!(refused(args(&bf16, &narrow, 256)).contains("peer slot"));
    let lone = PeerPartials {
        world_size: 0,
        rank: 0,
        ..PEERS
    };
    assert!(refused(args(&bf16, &lone, 256)).contains("ranks"));
    let wide = MHC_MIX_MAX_TOKENS as u32 + 1;
    let roomy = PeerPartials {
        slot_elems: wide as usize * 4096,
        ..PEERS
    };
    assert!(refused(args(&bf16, &roomy, wide)).contains("mix"));
    launch(&gpu, KernelHandle(FUSED), &args(&bf16, &PEERS, 256), 0).unwrap();
    assert_eq!(gpu.launch_count(), 1, "the control launches");
}
