// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The `nvfp4` tier without `--w4a4-downcast`, the decode-floor and agentic
//! recipes' serve, dispatches as the engine did before the tiers: no weight reaches the W4A4
//! path, nothing of it is resolved or allocated, and the row edges are the W4A16 edge. Its
//! own test binary, because the tier is a process-wide `OnceLock`.

use metrale_config::{Nvfp4Act, W4a4Downcast, WeightQuantTier, WeightQuantization};
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_model_layers::layers::ops::gemv_tc::narrow_gemv_max_rows;
use metrale_model_layers::layers::ops::w4a4_proj;
use metrale_model_layers::layers::w4a16_gemv_tiers::W4a16BatchmTiers;
use metrale_model_layers::weight_map::QuantizedWeight;

#[test]
fn no_weight_reaches_w4a4_and_nothing_of_it_is_prepared() {
    let tier = WeightQuantTier::new(WeightQuantization::Nvfp4, W4a4Downcast::Off).expect("tier");
    assert_eq!(
        metrale_model_layers::layers::set_weight_quantization_from_cli(tier),
        tier
    );
    let gpu = MockGpuBackend::new();
    let tiers = W4a16BatchmTiers::resolve(&gpu);
    assert_eq!(w4a4_proj::max_rows(&gpu), 0);
    for act in [Nvfp4Act::Unstamped, Nvfp4Act::A4, Nvfp4Act::Wide] {
        let w = QuantizedWeight {
            act,
            ..QuantizedWeight::null()
        };
        assert_eq!(tiers.edge(&w), narrow_gemv_max_rows(), "{act:?}");
        assert!(!tiers.declares_a4(&w), "{act:?}");
        assert_eq!(w4a4_proj::weight_rows(&gpu, &w), 0, "{act:?}");
        for m in 1..=70 {
            assert_eq!(
                tiers.kernel_for(m, &w).0,
                tiers.kernel(m).0,
                "{act:?} m={m}"
            );
        }
    }
    let looked_up = gpu.kernel_lookups_snapshot();
    assert!(
        looked_up
            .iter()
            .all(|(module, f)| module != "w4a4_gemv_mx" && f != "w4a16_gemv_batch32"),
        "{looked_up:?}"
    );
}
