// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: Tests for the drafter's target-hidden half (`hidden_half`).
//!
//! Owner: model-arch (GLM-5.3 MTP drafter).
//! Invariants: none beyond the types.
//!
//! GPU test: `#[ignore]`. Run with
//! ```text
//! cargo test -p metrale-model-arch --release hidden_half -- --ignored --nocapture
//! ```

use super::*;

/// 2026-10-11: A chain draft reads the drafter's own `shared_head.norm` output, which is already
/// normed; with the lever on it must still not take the target's norm, or the chain hidden is
/// normed twice (and `forward_body` would stage into `st.x`, the buffer the chain row lives in).
#[test]
fn only_target_rows_take_the_target_norm() {
    assert!(takes_target_norm(true, true));
    assert!(
        !takes_target_norm(true, false),
        "a chain row with the lever on"
    );
    assert!(!takes_target_norm(false, true), "the lever off");
    assert!(!takes_target_norm(false, false));
}

/// 2026-10-11: Only pair key 0, `(embed(token_1), hidden_0)` at position 0, is zeroed, and only
/// under the lever: vLLM zeroes by position, and every other key is a later position.
#[test]
fn only_pair_key_0_has_its_embedding_zeroed() {
    assert!(zeroes_embedding(true, 0));
    for key in [1usize, 2, 15, 16, 8191] {
        assert!(!zeroes_embedding(true, key), "key {key}");
    }
    assert!(!zeroes_embedding(false, 0), "the lever off");
}

fn bf16(v: f32) -> f32 {
    half::bf16::from_f32(v).to_f32()
}

/// 2026-10-11: Host `x * w / rms(x)` in f32, rounded to BF16 like the kernel's output.
fn host_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = 1.0 / (ms + eps).sqrt();
    x.iter().zip(w).map(|(a, b)| bf16(a * r * b)).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    d / (na.sqrt() * nb.sqrt())
}

fn upload(gpu: &dyn GpuBackend, v: &[f32]) -> DevicePtr {
    let b: Vec<u8> = v
        .iter()
        .flat_map(|x| half::bf16::from_f32(*x).to_le_bytes())
        .collect();
    let p = gpu.alloc(b.len()).unwrap();
    gpu.copy_h2d(&b, p).unwrap();
    p
}

fn download(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<f32> {
    let mut b = vec![0u8; n * 2];
    gpu.copy_d2h(p, &mut b).unwrap();
    b.chunks_exact(2)
        .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}

/// 2026-10-11: On the device, the half is `hnorm(row)` without a target norm and
/// `hnorm(target_norm(row))` with one, each within BF16 rounding of the host math, and the two
/// differ by more than rounding. The row carries an outlier channel whose target-norm weight is
/// near zero (GLM-5.3's channel 1448 has -0.0146), the case where the feeds part most. A launch
/// that skipped the target norm, ran the `1 + w` kernel, or read the staged row from the wrong
/// buffer fails one of the three checks.
#[test]
#[ignore]
fn the_half_is_hnorm_of_the_target_normed_row() {
    let set = metrale_kernels::ptx_for_exact_target("glm-5.3-flash", "nvfp4")
        .expect("glm-5.3-flash/nvfp4 not in this build");
    let gpu = metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
        .expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let k = g.kernel("rms_norm_vanilla", "rms_norm_vanilla").unwrap();
    let (h, eps) = (4096usize, 1e-5f32);
    // 2026-10-11: Deterministic pseudo-random values; an outlier at 1448 on the row.
    let mut s = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0
    };
    let mut row: Vec<f32> = (0..h).map(|_| bf16(next() * 3.0)).collect();
    row[1448] = 400.0;
    let mut wf: Vec<f32> = (0..h).map(|_| bf16(1.42 + 0.5 * next())).collect();
    wf[1448] = bf16(-0.0146);
    let hn: Vec<f32> = (0..h).map(|_| bf16(0.23 + 0.1 * next())).collect();

    let (d_row, d_wf, d_hn) = (upload(g, &row), upload(g, &wf), upload(g, &hn));
    let scratch = g.alloc(h * 2).unwrap();
    let out = g.alloc(h * 2).unwrap();

    launch_hidden_half(g, k, eps, h, d_row, None, d_hn, scratch, out, stream).unwrap();
    g.synchronize(stream).unwrap();
    let pre = download(g, out, h);
    let want_pre = host_norm(&row, &hn, eps);
    let c_pre = cosine(&pre, &want_pre);

    launch_hidden_half(g, k, eps, h, d_row, Some(d_wf), d_hn, scratch, out, stream).unwrap();
    g.synchronize(stream).unwrap();
    let post = download(g, out, h);
    let want_post = host_norm(&host_norm(&row, &wf, eps), &hn, eps);
    let c_post = cosine(&post, &want_post);

    let c_feeds = cosine(&pre, &post);
    println!("cos(pre, host)={c_pre:.7} cos(post, host)={c_post:.7} cos(pre, post)={c_feeds:.5}");
    assert!(c_pre > 0.99999, "no target norm: {c_pre}");
    assert!(c_post > 0.99999, "target norm: {c_post}");
    assert!(c_feeds < 0.9, "the two feeds should differ: {c_feeds}");
    for p in [d_row, d_wf, d_hn, scratch, out] {
        g.free(p).unwrap();
    }
}
