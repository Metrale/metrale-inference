// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: The FP8 block stage. On the mock: what each weight kind issues and the capacity
//! refusals. On the GPU: the dequantized weight and the rounded input equal the reference's
//! decode and `act_quant` bit for bit, and the projection matches the reference `linear`.
//!
//! Owner: model-arch, DeepSeek-V4.1.
//! Invariants: none beyond the types.

use super::*;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};

const FP8: ResidentMat = ResidentMat::Fp8Block {
    w: DevicePtr(0x100),
    scale: DevicePtr(0x200),
    qdq: true,
};

fn stage(g: &MockGpuBackend, weight_elems: usize, act_elems: usize) -> Fp8BlockStage {
    Fp8BlockStage::new(
        g,
        Fp8StageSize {
            weight_elems,
            act_elems,
        },
    )
    .unwrap()
}

#[test]
fn a_non_fp8_weight_passes_through_without_a_launch() {
    let g = MockGpuBackend::new();
    let s = stage(&g, 0, 0);
    let a = DevicePtr(0x42);
    let (a2, w2) = s
        .lower(&g, a, ResidentMat::Bf16(DevicePtr(7)), 4, 64, 64, 0)
        .unwrap();
    assert_eq!(a2, a);
    assert!(matches!(w2, ResidentMat::Bf16(DevicePtr(7))));
    assert!(g.launches_snapshot().is_empty());
}

#[test]
fn an_fp8_weight_dequantizes_and_rounds_a_copy_of_the_input() {
    let g = MockGpuBackend::new();
    let s = stage(&g, 64 * 96, 3 * 96);
    let a = g.alloc(3 * 96 * 2).unwrap();
    let input: Vec<u8> = (0..3 * 96 * 2).map(|i| i as u8).collect();
    g.copy_h2d(&input, a).unwrap();
    let (a2, w2) = s.lower(&g, a, FP8, 3, 64, 96, 0).unwrap();
    let ResidentMat::Bf16(wp) = w2 else {
        panic!("{w2:?}")
    };
    assert_eq!((wp, a2), (s.w, s.x), "the GEMM reads both scratches");
    let l = g.launches_snapshot();
    assert_eq!(l.len(), 2, "dequantize, then round the input");
    assert_eq!(l[0].grid, [2, 16, 1]);
    assert_eq!(l[0].args[0], MockArg::Buffer(DevicePtr(0x100)));
    assert_eq!(l[0].args[1], MockArg::Buffer(DevicePtr(0x200)));
    let u32_args: Vec<u32> = l[0].args[3..]
        .iter()
        .map(|x| match x {
            MockArg::Bytes(b) => u32::from_le_bytes(b[..4].try_into().unwrap()),
            other => panic!("{other:?}"),
        })
        .collect();
    assert_eq!(u32_args, [64, 96, 32, 32, 3, SCALE_KIND_E8M0]);
    assert_eq!(l[1].args[0], MockArg::Buffer(s.x));
    let mut copied = vec![0u8; input.len()];
    g.copy_d2h(s.x, &mut copied).unwrap();
    assert_eq!(
        copied, input,
        "the rounding runs on a copy of the whole input"
    );
}

#[test]
fn an_exact_input_fp8_weight_keeps_the_input() {
    let g = MockGpuBackend::new();
    let s = stage(&g, 64 * 96, 0);
    let w = ResidentMat::Fp8Block {
        w: DevicePtr(0x100),
        scale: DevicePtr(0x200),
        qdq: false,
    };
    let a = DevicePtr(0x42);
    let (a2, _) = s.lower(&g, a, w, 3, 64, 96, 0).unwrap();
    assert_eq!(a2, a);
    assert_eq!(g.launches_snapshot().len(), 1);
}

#[test]
fn a_projection_larger_than_its_scratch_is_refused_before_any_launch() {
    let g = MockGpuBackend::new();
    let s = stage(&g, 64 * 96 - 1, 1 << 20);
    let e = s.lower(&g, DevicePtr(1), FP8, 1, 64, 96, 0).unwrap_err();
    assert!(e.to_string().contains("weight scratch"), "{e}");
    let s = stage(&g, 64 * 96, 3 * 96 - 1);
    let e = s.lower(&g, DevicePtr(1), FP8, 3, 64, 96, 0).unwrap_err();
    assert!(e.to_string().contains("input scratch"), "{e}");
    assert!(g.launches_snapshot().len() <= 1, "no input rounding ran");
    let e = s.lower(&g, DevicePtr(1), FP8, 1, 64, 80, 0).unwrap_err();
    assert!(e.to_string().contains("whole 32 x 32 tiles"), "{e}");
}

#[cfg(feature = "cuda")]
mod gpu {
    use super::*;
    use crate::deepseek_v41_ref::attn::act_quant_inplace;
    use crate::deepseek_v41_ref::mat::{InputQuant, Mat, OutPrec, Store, Weight, linear};
    use metrale_model_layers::layers::ops;
    use metrale_model_layers::weight_map::DenseWeight;

    fn backend() -> metrale_gpu_runtime::cuda_backend::MetraleCudaBackend {
        let set = metrale_kernels::ptx_for_exact_target("deepseek-v4-flash", "nvfp4")
            .expect("deepseek-v4-flash/nvfp4 not in this build");
        metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
            .expect("CUDA backend")
    }

    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*seed >> 33) as u32
    }

    fn bf16(x: f32) -> u16 {
        let b = x.to_bits();
        (b.wrapping_add(0x7FFF + ((b >> 16) & 1)) >> 16) as u16
    }

    fn of_bf16(b: &[u8]) -> Vec<f32> {
        b.chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect()
    }

    fn run(m: usize, n: usize, k: usize) {
        let gpu = backend();
        let g: &dyn GpuBackend = &gpu;
        let stream = g.default_stream();
        let mut seed = 0x5eed ^ (m * 7919 + n) as u64;
        // 2026-10-19: Every E4M3 code but the two NaNs; scales 2^-6 .. 2^3 and the zero code 0.
        let w: Vec<u8> = (0..n * k)
            .map(|_| match (lcg(&mut seed) % 254) as u8 {
                c if c >= 0x7F => c + 1,
                c => c,
            })
            .collect();
        let scale: Vec<u8> = (0..(n / 32) * (k / 32))
            .map(|i| {
                if i == 1 {
                    0
                } else {
                    121 + (lcg(&mut seed) % 10) as u8
                }
            })
            .collect();
        let x: Vec<f32> = (0..m * k)
            .map(|i| {
                let v = (lcg(&mut seed) % 2001) as f32 / 1000.0 - 1.0;
                // 2026-10-19: One block per row spans a wide range, so its scale is not 1.
                if i % k < 32 { v * 300.0 } else { v }
            })
            .map(|v| f32::from_bits((bf16(v) as u32) << 16))
            .collect();
        let mat = Mat::new(
            Store::E4m3 {
                w: &w,
                scale: &scale,
                block_rows: 32,
            },
            n,
            k,
            InputQuant::Fp8Block,
        )
        .unwrap();

        let up = |b: &[u8]| {
            let p = g.alloc(b.len()).unwrap();
            g.copy_h2d(b, p).unwrap();
            p
        };
        let x_bytes: Vec<u8> = x.iter().flat_map(|&v| bf16(v).to_le_bytes()).collect();
        let (wd, sd, xd) = (up(&w), up(&scale), up(&x_bytes));
        let yd = g.alloc(m * n * 2).unwrap();
        let s = Fp8BlockStage::new(
            g,
            Fp8StageSize {
                weight_elems: n * k,
                act_elems: m * k,
            },
        )
        .unwrap();
        let fp8 = ResidentMat::Fp8Block {
            w: wd,
            scale: sd,
            qdq: true,
        };
        let (a, wb) = s.lower(g, xd, fp8, m, n, k, stream).unwrap();
        let ResidentMat::Bf16(wp) = wb else {
            panic!("{wb:?}")
        };
        let weight = DenseWeight { weight: wp };
        if m == 1 {
            let kern = g.kernel("gemv", "dense_gemv_bf16").unwrap();
            ops::dense_gemv(g, kern, a, &weight, yd, n as u32, k as u32, stream).unwrap();
        } else {
            let kern = g.kernel("gemm", "dense_gemm_bf16").unwrap();
            ops::dense_gemm(
                g, kern, a, &weight, yd, m as u32, n as u32, k as u32, stream,
            )
            .unwrap();
        }
        g.synchronize(stream).unwrap();
        let down = |p: DevicePtr, elems: usize| {
            let mut b = vec![0u8; elems * 2];
            g.copy_d2h(p, &mut b).unwrap();
            of_bf16(&b)
        };

        // 2026-10-19: The engine decodes the E8M0 code 0 to zero (`mx_block_scale`, the one scale
        // decode every kernel shares); the reference decodes it to the subnormal 2^-127, so its
        // values sit below 2^-118. Every other code must match bit for bit.
        let mut row = vec![0f32; k];
        let got_w = down(wp, n * k);
        for r in 0..n {
            mat.decode_row(r, k, &mut row);
            for c in 0..k {
                let tile_code = scale[(r / 32) * (k / 32) + c / 32];
                let want = if tile_code == 0 {
                    0f32.copysign(row[c])
                } else {
                    row[c]
                };
                assert_eq!(
                    got_w[r * k + c].to_bits(),
                    want.to_bits(),
                    "weight [{r}, {c}]"
                );
            }
        }
        let mut want_x = x.clone();
        act_quant_inplace(&mut want_x);
        let got_x = down(a, m * k);
        for (i, (g_, w_)) in got_x.iter().zip(&want_x).enumerate() {
            assert_eq!(g_.to_bits(), w_.to_bits(), "rounded input {i}");
        }
        assert_eq!(down(xd, m * k), x, "the caller's input is not touched");

        // 2026-10-19: Both sides round an f32 sum to bf16; they differ by the summation order
        // (bounded by the sum of |terms| times k f32 epsilons) and one bf16 step.
        let want_y = linear(&mat, &x, m, k, n, OutPrec::Bf16);
        let got_y = down(yd, m * n);
        for (i, (g_, w_)) in got_y.iter().zip(&want_y).enumerate() {
            let (t, r) = (i / n, i % n);
            let mag: f32 = (0..k)
                .map(|c| (got_w[r * k + c] * want_x[t * k + c]).abs())
                .sum();
            let tol = w_.abs() / 128.0 + mag * (k as f32) * f32::EPSILON;
            assert!((g_ - w_).abs() <= tol, "y[{i}]: {g_} vs {w_}");
        }
        s.free(g).unwrap();
        for p in [wd, sd, xd, yd] {
            g.free(p).unwrap();
        }
    }

    #[test]
    #[ignore = "requires a CUDA GB10 + the deepseek-v4-flash kernel target"]
    fn decode_row_matches_the_reference_linear() {
        run(1, 96, 160);
    }

    #[test]
    #[ignore = "requires a CUDA GB10 + the deepseek-v4-flash kernel target"]
    fn prefill_rows_match_the_reference_linear() {
        run(5, 128, 256);
    }
}
