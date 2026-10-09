// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Known-answer test tying the host NVFP4 quantizer (`metrale_core::numeric`, which
//! `met ml-utils` mock checkpoints and the GLM-5.3 loader use) to the load-time GPU quantizer
//! (`weight_map::quantize_to_nvfp4`, `quantize_bf16_to_nvfp4.cu`).
//!
//! 2026-10-04: The host follows the GPU's rounding (the reference: certified serves run the
//! GPU quantizer), so the two must be byte-identical: global scale, every block-scale byte and
//! every code, on normal weights and on a corpus built to land exactly on every E2M1 tie (the
//! control: it proves the ties are exercised, where round-to-even and the GPU disagree).
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
use metrale_core::numeric::{FP8_E4M3_LUT, quantize_to_nvfp4 as host_quantize};
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

/// 2026-10-04: The E2M1 midpoints: 0|0.5, 0.5|1, 1|1.5, 1.5|2, 2|3, 3|4, 4|6.
const TIES: [f32; 7] = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0];

fn identical(p: &Pair) {
    assert_eq!(p.host.2.to_bits(), p.gpu.2.to_bits(), "global scale");
    assert_eq!(p.host.1, p.gpu.1, "block scales");
    let first = p.host.0.iter().zip(&p.gpu.0).position(|(h, g)| h != g);
    assert_eq!(first, None, "packed codes differ first at byte {first:?}");
}

/// 2026-10-04: How many values sit exactly on an E2M1 midpoint under the GPU's arithmetic.
fn ties_hit(p: &Pair) -> usize {
    (0..p.values.len())
        .filter(|&i| {
            let s = FP8_E4M3_LUT[p.gpu.1[i / 16] as usize] * p.gpu.2;
            s > 0.0 && TIES.contains(&(p.values[i] * (1.0 / s)).abs())
        })
        .count()
}

#[test]
#[ignore = "requires a GB10 GPU + compiled kernel set (CI links libcuda stubs only)"]
fn host_and_gpu_nvfp4_are_byte_identical_including_ties() -> Result<()> {
    let (backend, st) = setup()?;
    let gpu: &dyn GpuBackend = &backend;
    let mut rng = Rng(0x5EED);
    let (n, k) = (512usize, 2048usize);
    let normal: Vec<f32> = (0..n * k)
        .map(|_| (0..4).map(|_| rng.unit()).sum::<f32>() - 2.0)
        .collect();
    identical(&quantize_both(gpu, st, &normal, n, k)?);

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
    assert_eq!(ties_hit(&p), 16 * 6, "the control lands on the midpoints");
    identical(&p);
    Ok(())
}
