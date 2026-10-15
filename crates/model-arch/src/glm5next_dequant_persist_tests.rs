// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `glm5next_dequant_persist` without a GPU: the lever's parse and refusals, which
//! projections each value keeps, and the copy registry's bookkeeping on the mock backend (the
//! output bytes are gated on the GPU by the `glm5next_dequant_persist_gate` example).

use super::*;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};
use metrale_model_layers::weight_map::WeightQuantFormat;

/// 2026-10-15: Unset and `0` are off whatever else is on; the three values need the W4A16
/// dequant GEMM and refuse the full-M tensor-core GEMM; `all` also needs the FP8 dequant GEMM;
/// any other text is an error.
#[test]
fn the_lever_parses_strictly() {
    assert_eq!(parse(None, false, true, false), Ok(Persist::Off));
    assert_eq!(parse(Some("0"), false, true, false), Ok(Persist::Off));
    assert_eq!(parse(Some("kda"), true, false, false), Ok(Persist::Kda));
    assert_eq!(parse(Some("w4a16"), true, false, false), Ok(Persist::W4a16));
    assert_eq!(parse(Some("all"), true, false, true), Ok(Persist::All));
    for bad in ["1", "KDA", "fp8", ""] {
        assert!(parse(Some(bad), true, false, true).is_err(), "{bad:?}");
    }
    for v in ["kda", "w4a16", "all"] {
        let e = parse(Some(v), false, false, true).unwrap_err();
        assert!(e.contains("W4A16_DEQUANT_GEMM_MIN_ROWS"), "{e}");
        let e = parse(Some(v), true, true, true).unwrap_err();
        assert!(e.contains("TC_GEMM_MIN_ROWS"), "{e}");
    }
    let e = parse(Some("all"), true, false, false).unwrap_err();
    assert!(e.contains("FP8_DEQUANT_GEMM_MIN_ROWS"), "{e}");
}

/// 2026-10-15: `kda` keeps the KDA W4A16 projections only, `w4a16` every W4A16 one, `all` the
/// FP8 ones too, off none.
#[test]
fn each_value_keeps_its_projections_only() {
    let cases = [
        (false, "kda.q_proj", [false, true, true, true]),
        (false, "kda.b_proj", [false, true, true, true]),
        (
            false,
            "shared_experts.down_proj",
            [false, false, true, true],
        ),
        (false, "dsa.o_absorb", [false, false, true, true]),
        (true, "kda.f_b_proj", [false, false, false, true]),
        (true, "dsa.o_absorb", [false, false, false, true]),
    ];
    let modes = [Persist::Off, Persist::Kda, Persist::W4a16, Persist::All];
    for (fp8, name, want) in cases {
        for (m, w) in modes.into_iter().zip(want) {
            assert_eq!(m.keeps(fp8, name), w, "{m:?} {name} fp8={fp8}");
        }
    }
}

fn dst_of(l: &metrale_gpu_runtime::gpu::mock::MockLaunch) -> DevicePtr {
    match l.args[2] {
        MockArg::Buffer(p) => p,
        ref a => panic!("third argument is {a:?}, not the destination buffer"),
    }
}

/// 2026-10-15: A kept copy is its own `n * k * 2`-byte allocation, filled by one launch of the
/// tier's dequantization kernel reading the weight and writing the copy, then synchronized; the
/// copy is found under the quantized weight's address at its shape only; a second copy of one
/// weight, or a failed fill, records nothing (the failed fill's allocation is freed). One test
/// owns the process-wide registry.
#[test]
fn copies_are_kept_once_per_weight_and_found_at_their_shape() {
    let gpu = MockGpuBackend::new();
    let q = QuantizedWeight {
        weight: DevicePtr(0xA000_0000),
        weight_scale: DevicePtr(0xA100_0000),
        ..QuantizedWeight::null()
    };
    keep_w4a16(&gpu, &q, 2816, 4096, "q").unwrap();
    let f = Fp8Weight {
        weight: DevicePtr(0xB000_0000),
        row_scale: DevicePtr(0xB100_0000),
        n: 4096,
        k: 11264,
        scale_format: WeightQuantFormat::Fp8PerRow,
    };
    keep_fp8(&gpu, &f, "o").unwrap();
    let l = gpu.launches_snapshot();
    assert_eq!(l.len(), 2);
    assert_eq!(l[0].args[0], MockArg::Buffer(q.weight));
    assert_eq!(l[0].args[1], MockArg::Buffer(q.weight_scale));
    assert_eq!(l[1].args[0], MockArg::Buffer(f.weight));
    assert_eq!(l[1].args[1], MockArg::Buffer(f.row_scale));
    let lookups = gpu.kernel_lookups_snapshot();
    assert_eq!(lookups[0].1, "dequant_nvfp4_to_bf16_g16");
    assert_eq!(lookups[1].1, "dequant_fp8_rowscale_to_bf16");
    assert_eq!(gpu.sync_count(), 2, "each fill is waited for at load");
    let cq = copy_of(q.weight, 2816, 4096).unwrap().unwrap();
    let cf = copy_of(f.weight, 4096, 11264).unwrap().unwrap();
    assert_eq!((dst_of(&l[0]), dst_of(&l[1])), (cq, cf));
    assert_eq!(gpu.read_alloc(cq).map(|b| b.len()), Some(2816 * 4096 * 2));
    assert_eq!(gpu.read_alloc(cf).map(|b| b.len()), Some(4096 * 11264 * 2));
    assert!(copy_of(q.weight_scale, 2816, 4096).unwrap().is_none());
    let e = copy_of(q.weight, 4096, 2816).unwrap_err();
    assert!(e.to_string().contains("launched as [4096, 2816]"), "{e}");
    let e = keep_w4a16(&gpu, &q, 2816, 4096, "again").unwrap_err();
    assert!(e.to_string().contains("already has a copy"), "{e}");
    // 2026-10-15: A fresh backend: the first one's op cache already holds the kernel.
    let denied = MockGpuBackend::new();
    denied.deny_kernel("dequant_nvfp4_bf16", "dequant_nvfp4_to_bf16_g16");
    let other = QuantizedWeight {
        weight: DevicePtr(0xC000_0000),
        ..q
    };
    assert!(keep_w4a16(&denied, &other, 128, 4096, "denied").is_err());
    assert_eq!(denied.alloc_count(), 0, "the failed fill's copy is freed");
    assert!(copy_of(other.weight, 128, 4096).unwrap().is_none());
    assert_eq!(kept(), (2, (2816 * 4096 + 4096 * 11264) * 2));
}
