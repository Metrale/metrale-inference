// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: The grid-y split of `hc_post`, `hc_pre_mix` and `hc_pre_stage_bf16` (one token
//! over several blocks) is byte-identical to the one-block-per-token launch (grid y = 1) at
//! every grid y the launchers use and at others, for decode and prefill token counts and for a
//! hidden size that is not a multiple of the block. `hc_post` is driven through its launcher,
//! so the launcher's grid is the one checked.
//!
//! GPU test: run with
//! ```text
//! cargo test -p metrale-model-layers --release hyper_connection_split_tests -- --ignored --nocapture
//! ```
//! on a host whose build has the `qwen3.8-flash-next/nvfp4` kernel set.

use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::hyper_connection_lowrank::{HC_ELEMENTWISE_BLOCK, hc_hidden_split};
use crate::layers::ops;

const HC: u32 = 4;
const SENTINEL: u8 = 0xAB;
/// 2026-10-13: (hidden size, token counts): the model's 2560 and a size with a partial block.
const SHAPES: [(u32, &[u32]); 2] = [(2560, &[1, 3, 64, 300]), (2000, &[1, 5])];

/// 2026-10-13: Deterministic values in [-2, 2].
fn values(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x % 4001) as f32 / 1000.0 - 2.0
        })
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|f| ((f.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn upload(g: &dyn GpuBackend, data: &[u8]) -> DevicePtr {
    let p = g.alloc(data.len()).unwrap();
    g.copy_h2d(data, p).unwrap();
    p
}

fn sentinel(g: &dyn GpuBackend, n: usize) -> DevicePtr {
    let p = g.alloc(n).unwrap();
    g.memset(p, SENTINEL, n).unwrap();
    p
}

fn download(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<u8> {
    let mut out = vec![0u8; n];
    g.copy_d2h(p, &mut out).unwrap();
    out
}

/// 2026-10-13: Every `width`-byte element of `out` was written (none is all sentinel bytes).
fn assert_written(label: &str, out: &[u8], width: usize) {
    let unwritten = out
        .chunks_exact(width)
        .filter(|c| c.iter().all(|&b| b == SENTINEL))
        .count();
    assert_eq!(unwritten, 0, "{label}: {unwritten} elements left unwritten");
}

struct Kernels {
    post: KernelHandle,
    mix: KernelHandle,
    stage: KernelHandle,
}

fn run_post(g: &dyn GpuBackend, k: &Kernels, h: u32, t: u32, y: Option<u32>, s: u64) -> Vec<u8> {
    let n = (t * HC * h) as usize;
    let block_out = upload(g, &bf16_bytes(&values((t * h) as usize, 7)));
    let residual = upload(g, &f32_bytes(&values(n, 11)));
    let inj = upload(g, &f32_bytes(&values((t * HC) as usize, 13)));
    let out = sentinel(g, n * 4);
    match y {
        None => {
            ops::hc_post_lowrank(g, k.post, block_out, residual, inj, out, t, h, HC, s).unwrap()
        }
        Some(y) => KernelLaunch::new(g, k.post)
            .grid([t, y, 1])
            .block([HC_ELEMENTWISE_BLOCK, 1, 1])
            .arg_ptr(block_out)
            .arg_ptr(residual)
            .arg_ptr(inj)
            .arg_ptr(out)
            .arg_u32(h)
            .arg_u32(HC)
            .launch(s)
            .unwrap(),
    }
    g.synchronize(s).unwrap();
    download(g, out, n * 4)
}

/// 2026-10-13: Returns the `y_out` and `inj_out` bytes.
fn run_mix(g: &dyn GpuBackend, k: &Kernels, h: u32, t: u32, y: u32, s: u64) -> [Vec<u8>; 2] {
    let n = (t * HC * h) as usize;
    let normed = upload(g, &bf16_bytes(&values(n, 17)));
    let up_pre = upload(g, &bf16_bytes(&values(n, 19)));
    let inj_pre = upload(g, &bf16_bytes(&values((t * HC) as usize, 23)));
    let (y_bytes, inj_bytes) = ((t * h) as usize * 2, (t * HC) as usize * 4);
    let (y_out, inj_out) = (sentinel(g, y_bytes), sentinel(g, inj_bytes));
    KernelLaunch::new(g, k.mix)
        .grid([t, y, 1])
        .block([HC_ELEMENTWISE_BLOCK, 1, 1])
        .arg_ptr(normed)
        .arg_ptr(up_pre)
        .arg_ptr(inj_pre)
        .arg_ptr(y_out)
        .arg_ptr(inj_out)
        .arg_u32(h)
        .arg_u32(HC)
        .arg_f32(1.0 / HC as f32)
        .launch(s)
        .unwrap();
    g.synchronize(s).unwrap();
    [download(g, y_out, y_bytes), download(g, inj_out, inj_bytes)]
}

fn run_stage(g: &dyn GpuBackend, k: &Kernels, h: u32, t: u32, y: u32, s: u64) -> Vec<u8> {
    let n = (t * HC * h) as usize;
    let streams = upload(g, &f32_bytes(&values(n, 29)));
    let norm_w = upload(g, &bf16_bytes(&values((HC * h) as usize, 31)));
    let out = sentinel(g, n * 2);
    KernelLaunch::new(g, k.stage)
        .grid([t, y, 1])
        .block([1024, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(norm_w)
        .arg_ptr(out)
        .arg_u32(h)
        .arg_u32(HC)
        .arg_f32(1e-6)
        .launch(s)
        .unwrap();
    g.synchronize(s).unwrap();
    download(g, out, n * 2)
}

#[test]
#[ignore]
fn split_hc_launches_are_byte_identical_to_one_block_per_token() {
    let set = metrale_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("qwen3.8-flash-next/nvfp4 not in this build");
    let gpu = metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
        .expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let k = Kernels {
        post: g.kernel("hyper_connection", "hc_post").unwrap(),
        mix: g.kernel("hyper_connection", "hc_pre_mix").unwrap(),
        stage: g.kernel("hyper_connection", "hc_pre_stage_bf16").unwrap(),
    };
    let s = g.create_stream().unwrap();
    for (h, tokens) in SHAPES {
        let split = hc_hidden_split(h);
        assert!(split > 1, "H={h}: the launcher splits");
        for &t in tokens {
            let label = format!("H={h} T={t}");
            let post_one = run_post(g, &k, h, t, Some(1), s);
            assert_written(&format!("{label} hc_post y=1"), &post_one, 4);
            assert_eq!(
                run_post(g, &k, h, t, None, s),
                post_one,
                "{label} hc_post launcher"
            );
            let mix_one = run_mix(g, &k, h, t, 1, s);
            assert_written(&format!("{label} hc_pre_mix y"), &mix_one[0], 2);
            assert_written(&format!("{label} hc_pre_mix inj"), &mix_one[1], 4);
            let stage_one = run_stage(g, &k, h, t, 1, s);
            assert_written(&format!("{label} hc_pre_stage_bf16"), &stage_one, 2);
            for y in [2, 3, split, split + 1] {
                assert_eq!(
                    run_post(g, &k, h, t, Some(y), s),
                    post_one,
                    "{label} hc_post y={y}"
                );
                assert_eq!(
                    run_mix(g, &k, h, t, y, s),
                    mix_one,
                    "{label} hc_pre_mix y={y}"
                );
            }
            for y in [2, 3, HC] {
                assert_eq!(
                    run_stage(g, &k, h, t, y, s),
                    stage_one,
                    "{label} hc_pre_stage_bf16 y={y}"
                );
            }
        }
    }
}
