// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Known-answer test tying the host NVFP4 quantizer (`metrale_core::numeric`, which
//! `met ml-utils` mock checkpoints and the GLM-5.3 loader use) to the load-time GPU quantizer
//! (`weight_map::quantize_to_nvfp4`, `quantize_bf16_to_nvfp4.cu`).
//!
//! The two share the global scale rule (`amax / (6 * 448)`). Their documented rounding rules
//! differ at exact ties: the GPU rounds an E4M3 scale tie away from zero and an E2M1 tie to the
//! smaller magnitude; the host rounds both to even. So the test asserts:
//! - the global scale is bit-identical;
//! - every block-scale byte and every code is identical, except a code whose scaled value sits
//!   exactly on an E2M1 midpoint where the two rules disagree (0.75, 1.75, 3.5), and such
//!   exceptions are rare (at most 1 in 10^4 codes on normal weights);
//! - on a corpus built to hit those midpoints, the host picks the even code and the GPU the
//!   smaller one (the detection control: the difference is the tie rule, not an accident).
//!
//! `#[ignore]`d because it needs a GPU. On a GB10 host:
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL='*' METRALE_TARGET_QUANT='*' \
//!     cargo test -p metrale-model-engine --test nvfp4_host_gpu_kat -- --ignored --nocapture
//!
//! Owner: model-engine tests.
//! Invariants: none beyond the types.

#[path = "arm2_common/support.rs"]
mod support;

use anyhow::Result;
use metrale_core::numeric::{FP8_E4M3_LUT, NVFP4_E2M1_LUT, quantize_to_nvfp4 as host_quantize};
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_layers::weight_map::{DenseWeight, quantize_to_nvfp4 as gpu_quantize};
use support::{Rng, f32_to_bf16_bits, setup};

struct Pair {
    host: (Vec<u8>, Vec<u8>, f32),
    gpu: (Vec<u8>, Vec<u8>, f32),
    values: Vec<f32>,
}

fn quantize_both(
    gpu: &dyn GpuBackend,
    st: u64,
    values: &[f32],
    n: usize,
    k: usize,
) -> Result<Pair> {
    let bits: Vec<u16> = values.iter().map(|&v| f32_to_bf16_bits(v)).collect();
    let values: Vec<f32> = bits
        .iter()
        .map(|&b| f32::from_bits((b as u32) << 16))
        .collect();
    let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    let dev = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, dev)?;
    let absmax = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quant = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let q = gpu_quantize(&DenseWeight { weight: dev }, n, k, gpu, absmax, quant, st)?;
    let mut packed = vec![0u8; n * k / 2];
    let mut scales = vec![0u8; n * k / 16];
    gpu.copy_d2h(q.weight, &mut packed)?;
    gpu.copy_d2h(q.weight_scale, &mut scales)?;
    let h = host_quantize("kat", &values, n, k)?;
    Ok(Pair {
        host: (h.packed, h.scales, h.scale_2),
        gpu: (packed, scales, q.weight_scale_2),
        values,
    })
}

fn nibble(p: &[u8], i: usize) -> u8 {
    if i.is_multiple_of(2) {
        p[i / 2] & 0xF
    } else {
        p[i / 2] >> 4
    }
}

/// 2026-10-03: The E2M1 midpoints where round-to-even and round-to-smaller disagree.
const SPLIT_TIES: [f32; 3] = [0.75, 1.75, 3.5];

fn compare(p: &Pair, k: usize) -> (usize, usize) {
    assert_eq!(p.host.2.to_bits(), p.gpu.2.to_bits(), "global scale");
    assert_eq!(p.host.1, p.gpu.1, "block scales");
    let mut ties = 0;
    for i in 0..p.values.len() {
        let (h, g) = (nibble(&p.host.0, i), nibble(&p.gpu.0, i));
        if h == g {
            continue;
        }
        // 2026-10-03: The GPU divides by the decoded scale through a reciprocal; the
        // exception must be an exact split tie in that arithmetic.
        let s = FP8_E4M3_LUT[p.gpu.1[i / 16] as usize] * p.gpu.2;
        let x = (p.values[i] * (1.0 / s)).abs();
        assert!(
            SPLIT_TIES.contains(&x),
            "element {i} (row {}, col {}): host {h:#x} gpu {g:#x} at |v/s| = {x}",
            i / k,
            i % k
        );
        assert!(
            NVFP4_E2M1_LUT[g as usize].abs() < NVFP4_E2M1_LUT[h as usize].abs(),
            "the GPU takes the smaller magnitude at a tie"
        );
        ties += 1;
    }
    (ties, p.values.len())
}

#[test]
#[ignore = "requires a GB10 GPU + compiled kernel set (CI links libcuda stubs only)"]
fn host_and_gpu_nvfp4_agree_except_at_documented_ties() -> Result<()> {
    let (backend, st) = setup()?;
    let gpu: &dyn GpuBackend = &backend;
    let mut rng = Rng(0x5EED);
    let (n, k) = (512usize, 2048usize);
    let normal: Vec<f32> = (0..n * k)
        .map(|_| (0..4).map(|_| rng.unit()).sum::<f32>() - 2.0)
        .collect();
    let (ties, total) = compare(&quantize_both(gpu, st, &normal, n, k)?, k);
    eprintln!("normal weights: {ties} tie exception(s) in {total} codes");
    assert!(ties * 10_000 <= total, "{ties} exceptions in {total}");

    // 2026-10-03: The control, built so every quantity is a power of two and the midpoints are
    // exact: the global amax 2.625 makes scale2 = 1/1024; the second block's max 1.5 makes its
    // scale byte 256 and its effective scale 1/4, so `v * 4` lands on 0.75, 1.75 and 3.5 exactly.
    let block_a = [
        2.625f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let block_b = [
        1.5f32, 0.1875, 0.4375, 0.875, -0.1875, -0.4375, -0.875, 0.25, 0.125, 0.5, 0.75, 1.0,
        0.375, 0.0, -1.5, 1.5,
    ];
    let mut tie_rows = Vec::with_capacity(16 * 32);
    for _ in 0..16 {
        tie_rows.extend_from_slice(&block_a);
        tie_rows.extend_from_slice(&block_b);
    }
    let p = quantize_both(gpu, st, &tie_rows, 16, 32)?;
    assert_eq!(p.host.2, 1.0 / 1024.0);
    let (ties, _) = compare(&p, 32);
    assert_eq!(ties, 16 * 6, "every split midpoint differs, and only those");
    Ok(())
}
