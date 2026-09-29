// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The `nvfp4` tier with `--w4a4-downcast --w4a4-downcast-wide`, the certified
//! concurrency ladders' serve, dispatches as the engine did before the tiers: every NVFP4
//! weight, whatever its stamp, takes the W4A4 path up to 64 rows, and the dense FFN's narrow
//! arms stop at 32. Its own test binary, because the tier is a process-wide `OnceLock`.

use metrale_config::{Nvfp4Act, W4a4Downcast, WeightQuantTier, WeightQuantization};
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_model_layers::layers::ops::w4a4_proj;
use metrale_model_layers::layers::w4a16_gemv_tiers::W4a16BatchmTiers;
use metrale_model_layers::weight_map::QuantizedWeight;

#[test]
fn every_weight_takes_the_levers_64_row_edge() {
    let tier = WeightQuantTier::new(WeightQuantization::Nvfp4, W4a4Downcast::Wide).expect("tier");
    assert_eq!(
        metrale_model_layers::layers::set_weight_quantization_from_cli(tier),
        tier
    );
    let gpu = MockGpuBackend::new();
    let tiers = W4a16BatchmTiers::resolve(&gpu);
    assert_eq!(w4a4_proj::max_rows(&gpu), w4a4_proj::W4A4_WIDE_MAX_M);
    for act in [Nvfp4Act::Unstamped, Nvfp4Act::A4, Nvfp4Act::Wide] {
        let w = QuantizedWeight {
            act,
            ..QuantizedWeight::null()
        };
        assert_eq!(tiers.edge(&w), 64, "{act:?}");
        assert_eq!(tiers.ffn_edge(&w), 32, "{act:?}");
        assert_eq!(w4a4_proj::weight_rows(&gpu, &w), 64, "{act:?}");
        for m in 1..=70 {
            assert_eq!(
                tiers.kernel_for(m, &w).0,
                tiers.kernel(m).0,
                "{act:?} m={m}"
            );
        }
    }
    for m in 1..=64 {
        assert_ne!(tiers.kernel(m).0, 0, "m={m}");
    }
    assert_eq!(tiers.kernel(65).0, 0);
    let looked_up = gpu.kernel_lookups_snapshot();
    for f in ["w4a16_gemv_batch32", "w4a4_gemv_mx64", "w4a4_gemv_mx64_nt2"] {
        assert!(looked_up.iter().any(|(_, g)| g == f), "{f} not resolved");
    }
}
