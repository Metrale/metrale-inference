// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The imported expert-load histogram is reproduced by the mock's own bytes: the
//! synthesized embedding rows of a fixed token set, RMS-normalised and multiplied by the
//! synthesized router, select experts within the stated tolerance of the profile; the uniform
//! mock of the same checkpoint does not (the detection control). The residual writers' bias row
//! is zero and nothing else changes shape or format.
//!
//! Owner: metrale-ml-utils.
//! Invariants: the reference computation decodes the stored BF16 bytes; it shares no code with
//! the synthesis beyond `top_k`.

use super::*;
use crate::routing::{top_k, total_variation};
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

fn plan_with(spec: &MockSpec, routing: Option<&RoutingProfile>) -> MockPlan {
    let (config, index) = testkit::moe_fp8();
    plan_mock(&MockInputs {
        source_id: "toy/model",
        revision: None,
        config_json: &config,
        hf_quant_config: None,
        index: &index,
        spec,
        routing,
    })
    .unwrap()
}

fn tensor(p: &MockPlan, name: &str) -> Vec<f32> {
    let u = p
        .units
        .iter()
        .position(|u| matches!(u, Unit::Plain { tensor, .. } if p.tensors[*tensor].name == name))
        .unwrap_or_else(|| panic!("{name}"));
    let (_, bytes) = synthesize(p, u).unwrap().remove(0);
    bytes
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

/// 2026-10-03: Expert selection counts of mock layer `layer` over every token of the vocab.
fn routed(p: &MockPlan, layer: usize) -> Vec<u64> {
    let h = testkit::HIDDEN as usize;
    let emb = tensor(p, "model.language_model.embed_tokens.weight");
    let router = tensor(
        p,
        &format!("model.language_model.layers.{layer}.mlp.gate.weight"),
    );
    let mut counts = vec![0u64; 8];
    for row in emb.chunks(h) {
        let rms = (row.iter().map(|x| x * x).sum::<f32>() / h as f32).sqrt();
        let logits: Vec<f32> = router
            .chunks(h)
            .map(|w| w.iter().zip(row).map(|(a, b)| a * b / rms).sum())
            .collect();
        for e in top_k(&logits, 2) {
            counts[e] += 1;
        }
    }
    counts
}

#[test]
fn the_histogram_is_reproduced_and_the_uniform_control_is_not() {
    let prof = profile();
    let skewed = plan_with(
        &testkit::spec("1", "mode = \"histogram\"\nhistogram = \"p.json\""),
        Some(&prof),
    );
    let uniform = plan_with(&testkit::spec("1", "mode = \"uniform\""), None);
    assert_eq!(skewed.routers.len(), 4);
    let n = testkit::VOCAB as f64 * 2.0;
    for layer in 0..4 {
        let want = &prof.layers[layer];
        let got = routed(&skewed, layer);
        // 2026-10-03: Tolerance: three times the expected multinomial TV of `n` selections, plus
        // the fit's own error.
        let total = want.iter().sum::<u64>() as f64;
        let sampling: f64 = want
            .iter()
            .map(|&c| (c as f64 / total) * (1.0 - c as f64 / total) / n)
            .map(|v| (2.0 / std::f64::consts::PI).sqrt() * v.sqrt())
            .sum::<f64>()
            / 2.0;
        let tol = 3.0 * sampling + skewed.routers[layer].tv + 0.01;
        let tv = total_variation(want, &got);
        assert!(
            tv <= tol,
            "layer {layer}: tv {tv} > {tol} ({got:?} vs {want:?})"
        );
        let control = total_variation(want, &routed(&uniform, layer));
        assert!(
            control > 2.0 * tol,
            "layer {layer}: the uniform control is within {control}"
        );
    }
    assert!(
        skewed.resolved.contains("histogram = \"sha256:"),
        "{}",
        skewed.resolved
    );
    assert!(!skewed.resolved.contains("p.json"));
}

#[test]
fn histogram_routing_changes_values_only() {
    let prof = profile();
    let skewed = plan_with(
        &testkit::spec("1", "mode = \"histogram\"\nhistogram = \"p.json\""),
        Some(&prof),
    );
    let uniform = plan_with(&testkit::spec("1", "mode = \"uniform\""), None);
    let shapes = |p: &MockPlan| {
        p.tensors
            .iter()
            .map(|t| (t.name.clone(), t.dtype, t.shape.clone()))
            .collect::<Vec<_>>()
    };
    assert_eq!(shapes(&skewed), shapes(&uniform));
    assert_eq!(skewed.config_json, uniform.config_json);
    let h = testkit::HIDDEN as usize;
    let emb = tensor(&skewed, "model.language_model.embed_tokens.weight");
    assert!(
        emb.chunks(h).all(|r| r[h - 1] == emb[h - 1]),
        "the bias channel is constant"
    );
    let writer = "model.language_model.layers.0.mlp.experts.0.down_proj.weight";
    let u = skewed
        .units
        .iter()
        .position(|u| matches!(u, Unit::Group { tensors, .. } if skewed.tensors[tensors[0]].name == writer))
        .expect("the writer's group");
    let codes = synthesize(&skewed, u).unwrap().remove(0).1;
    let cols = codes.len() / h;
    assert!(
        codes[(h - 1) * cols..].iter().all(|&c| c == 0),
        "the bias row of a writer is zero"
    );
    assert!(codes[..cols].iter().any(|&c| c != 0));
}

#[test]
fn a_profile_that_does_not_fit_the_checkpoint_is_refused() {
    let text = r#"{"schema":1,"source":"x","experts":4,"top_k":2,"layers":[[1,1,1,1]]}"#;
    let bad = RoutingProfile::parse(text).unwrap();
    let (config, index) = testkit::moe_fp8();
    let spec = testkit::spec("1", "mode = \"histogram\"\nhistogram = \"p.json\"");
    let err = plan_mock(&MockInputs {
        source_id: "toy/model",
        revision: None,
        config_json: &config,
        hf_quant_config: None,
        index: &index,
        spec: &spec,
        routing: Some(&bad),
    })
    .unwrap_err();
    assert!(err.to_string().contains("4 experts top-2"), "{err}");
}
