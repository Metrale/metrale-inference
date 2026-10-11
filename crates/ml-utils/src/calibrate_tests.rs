// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: Calibration recovers a known bias scale from loads drawn at that scale, in both
//! directions, maps routers to recorded rows in numeric mock-layer order, and refuses a
//! recording that does not fit the mock.
//!
//! Owner: metrale-ml-utils.
//! Invariants: the recordings are drawn with noise independent of the calibration's own.

use super::*;
use crate::plan::{MockInputs, plan_mock};
use crate::routing::{noise, sample_counts};
use crate::testkit::{self, SKEWED};

fn profile() -> RoutingProfile {
    let rows: Vec<Vec<u64>> = (0..testkit::LAYERS)
        .map(|l| {
            let mut r = SKEWED.to_vec();
            r.rotate_left(l % 8);
            r
        })
        .collect();
    let text = serde_json::json!({"schema": 1, "source": "toy/model", "experts": 8, "top_k": 2,
        "prompt_set": "toy", "layers": rows})
    .to_string();
    RoutingProfile::parse(&text).unwrap()
}

fn plan(prof: &RoutingProfile) -> MockPlan {
    let (config, index) = testkit::moe_fp8();
    let spec = testkit::spec(
        "2",
        "mode = \"histogram\"\nhistogram = \"p.json\"\ncalibration = \"none\"",
    );
    plan_mock(&MockInputs {
        source_id: "toy/model",
        revision: None,
        config_json: &config,
        hf_quant_config: None,
        index: &index,
        spec: &spec,
        routing: Some(prof),
        calibration: None,
        stats: None,
    })
    .unwrap()
}

/// 2026-10-04: Loads of every router at `scale` times its fitted bias, in mock-layer order.
fn recording(p: &MockPlan, prof: &RoutingProfile, scale: f32) -> RoutingProfile {
    let mut fits: Vec<_> = p
        .routers
        .iter()
        .map(|f| (mock_layer(&f.tensor).unwrap(), f))
        .collect();
    fits.sort_by_key(|(l, _)| *l);
    let fresh = noise(Stream::for_tensor(77, "recording", "x", &[]), 40_000, 8);
    let rows: Vec<Vec<u64>> = fits
        .iter()
        .map(|(_, f)| {
            let b: Vec<f32> = f.bias.iter().map(|x| x * scale).collect();
            sample_counts(&b, prof.top_k, &fresh)
        })
        .collect();
    let text = serde_json::json!({"schema": 1, "source": "toy/model", "experts": 8, "top_k": 2,
        "prompt_set": "recorded", "layers": rows})
    .to_string();
    RoutingProfile::parse(&text).unwrap()
}

#[test]
fn a_known_bias_scale_is_recovered_and_inverted_into_the_gain() {
    let prof = profile();
    let p = plan(&prof);
    assert_eq!(p.routers.len(), 8);
    for scale in [2.0f32, 0.5] {
        let cal = calibrate(&p, &prof, &recording(&p, &prof, scale)).unwrap();
        assert_eq!(cal.len(), 8);
        for l in &cal {
            assert!(
                (l.lambda / scale).ln().abs() < 0.15,
                "scale {scale}: layer {} lambda {}",
                l.mock_layer,
                l.lambda
            );
            assert!(
                (l.gain * l.lambda - 1.0).abs() < 1e-6,
                "gain = 1 / lambda from gain 1"
            );
            assert!(
                l.tv_model < 0.05,
                "the unit-noise model fits: {}",
                l.tv_model
            );
        }
        let g = gains(&cal);
        assert_eq!(
            g.keys().copied().collect::<Vec<_>>(),
            p.routers.iter().map(|f| f.source_layer).collect::<Vec<_>>()
        );
    }
    let exact = calibrate(&p, &prof, &recording(&p, &prof, 1.0)).unwrap();
    assert!(
        exact.iter().all(|l| l.lambda.ln().abs() < 0.15),
        "{exact:?}"
    );
}

#[test]
fn the_mock_layer_is_read_numerically() {
    assert_eq!(
        mock_layer("model.language_model.layers.10.mlp.gate.weight"),
        Some(10)
    );
    assert_eq!(mock_layer("model.layers.2.mlp.gate.weight"), Some(2));
    assert_eq!(mock_layer("lm_head.weight"), None);
}

#[test]
fn a_recording_that_does_not_fit_is_refused() {
    let prof = profile();
    let p = plan(&prof);
    let mut short = recording(&p, &prof, 1.0);
    short.layers.pop();
    let e = calibrate(&p, &prof, &short).unwrap_err().to_string();
    assert!(e.contains("7 MoE layers, the mock 8 routers"), "{e}");
    let mut wide = recording(&p, &prof, 1.0);
    wide.top_k = 4;
    let e = calibrate(&p, &prof, &wide).unwrap_err().to_string();
    assert!(e.contains("top-4"), "{e}");
}
