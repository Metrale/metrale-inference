// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: `values.mode = "stats"`: the plan keeps every name, dtype and shape of the init
//! plan; the bytes follow each class's statistics; the bias channel is written over the sample
//! (embedding column K, zero writer rows, the pre-FFN norm entry, the router column) and the
//! histogram is still reproduced by the mock's own bytes; a calibration gain scales the router
//! column only, and a missing class or layer is refused.
//!
//! Owner: metrale-ml-utils.
//! Invariants: the routing check decodes the stored BF16 bytes and shares no code with the
//! synthesis beyond `top_k`.

use super::super::*;
use crate::routing::{RoutingCalibration, top_k, total_variation};
use crate::stats::{Accumulator, Kind, Read, ValueStats};
use crate::testkit::{self, SKEWED, TOY_BF16_RMS};

const HIST: &str = "mode = \"histogram\"\nhistogram = \"p.json\"\ncalibration = \"none\"";
const STATS: &str = "mode = \"stats\"\nstats = \"s.json\"";
const HIST_CAL: &str = "mode = \"histogram\"\nhistogram = \"p.json\"\ncalibration = \"c.json\"";

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

struct Inputs<'a> {
    spec: MockSpec,
    routing: Option<&'a RoutingProfile>,
    calibration: Option<&'a RoutingCalibration>,
    stats: Option<&'a ValueStats>,
}

fn plan(i: &Inputs<'_>) -> Result<MockPlan> {
    let (config, index) = testkit::moe_fp8();
    plan_mock(&MockInputs {
        source_id: "toy/model",
        revision: None,
        config_json: &config,
        hf_quant_config: None,
        index: &index,
        spec: &i.spec,
        routing: i.routing,
        calibration: i.calibration,
        stats: i.stats,
    })
}

fn toy_stats() -> ValueStats {
    let (config, index) = testkit::moe_fp8();
    testkit::toy_stats(&config, &index)
}

fn bytes_of(p: &MockPlan, name: &str) -> Vec<u8> {
    let u = p
        .units
        .iter()
        .position(|u| matches!(u, Unit::Sampled { tensor, .. } if p.tensors[*tensor].name == name))
        .unwrap_or_else(|| panic!("no sampled unit {name}"));
    synthesize(p, u).unwrap().remove(0).1
}

fn bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

#[test]
fn stats_mode_keeps_every_name_dtype_and_shape_and_samples_every_tensor() {
    let stats = toy_stats();
    let init = plan(&Inputs {
        spec: testkit::spec("1", "mode = \"uniform\""),
        routing: None,
        calibration: None,
        stats: None,
    })
    .unwrap();
    let spec = testkit::spec_with("1", "mode = \"uniform\"", STATS);
    let sampled = plan(&Inputs {
        spec,
        routing: None,
        calibration: None,
        stats: Some(&stats),
    })
    .unwrap();
    let shapes = |p: &MockPlan| {
        let mut v: Vec<_> = p
            .tensors
            .iter()
            .map(|t| (t.name.clone(), t.dtype, t.shape.clone()))
            .collect();
        v.sort();
        v
    };
    assert_eq!(shapes(&sampled), shapes(&init));
    assert_eq!(sampled.config_json, init.config_json);
    assert!(sampled.units.iter().all(|u| matches!(
        u,
        Unit::Sampled {
            edit: sampled::Edit::None,
            ..
        }
    )));
    assert!(
        sampled
            .resolved
            .contains(&format!("stats = \"sha256:{}\"", stats.digest)),
        "{}",
        sampled.resolved
    );
    assert_ne!(sampled.digest, init.digest);
}

#[test]
fn sampled_bytes_follow_their_class() {
    let stats = toy_stats();
    let spec = testkit::spec_with("1", "mode = \"uniform\"", STATS);
    let p = plan(&Inputs {
        spec,
        routing: None,
        calibration: None,
        stats: Some(&stats),
    })
    .unwrap();
    for (name, class, kind) in [
        (
            "model.language_model.layers.0.mlp.experts.3.down_proj.weight",
            "L.mlp.experts.*.down_proj.weight|F8_E4M3",
            Kind::Bytes,
        ),
        (
            "model.language_model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight|BF16",
            Kind::Halves,
        ),
    ] {
        let mut acc = Accumulator::default();
        let r = Read {
            class: class.into(),
            kind,
            shard: String::new(),
            offset: 0,
            len: 0,
        };
        acc.add(&r, &bytes_of(&p, name));
        let got = acc.finish("m").0;
        let (want, seen) = (&stats.classes[class].counts, &got.classes[class].counts);
        let keys: Vec<u32> = want.keys().chain(seen.keys()).copied().collect();
        let w: Vec<u64> = keys
            .iter()
            .map(|k| want.get(k).copied().unwrap_or(0))
            .collect();
        let s: Vec<u64> = keys
            .iter()
            .map(|k| seen.get(k).copied().unwrap_or(0))
            .collect();
        assert!(
            seen.keys().all(|k| want.contains_key(k)),
            "{name}: a pattern outside its class"
        );
        // 2026-10-04: the toy tensors hold 8k-128k elements over <= a few hundred patterns.
        let tv = total_variation(&w, &s);
        assert!(tv < 0.08, "{name}: tv {tv}");
    }
}

/// 2026-10-04: Expert selection counts of mock layer `layer` over every token of the vocab:
/// RMS-normalised embedding rows, times the pre-FFN norm weight, times the router.
fn routed(p: &MockPlan, layer: usize) -> Vec<u64> {
    let h = testkit::HIDDEN as usize;
    let emb = bf16(&bytes_of(p, "model.language_model.embed_tokens.weight"));
    let pre = format!("model.language_model.layers.{layer}");
    let norm = bf16(&bytes_of(
        p,
        &format!("{pre}.post_attention_layernorm.weight"),
    ));
    let router = bf16(&bytes_of(p, &format!("{pre}.mlp.gate.weight")));
    let mut counts = vec![0u64; 8];
    for row in emb.chunks(h) {
        let rms = (row.iter().map(|x| x * x).sum::<f32>() / h as f32).sqrt();
        let x: Vec<f32> = row.iter().zip(&norm).map(|(a, g)| a * g / rms).collect();
        let logits: Vec<f32> = router
            .chunks(h)
            .map(|w| w.iter().zip(&x).map(|(a, b)| a * b).sum())
            .collect();
        for e in top_k(&logits, 2) {
            counts[e] += 1;
        }
    }
    counts
}

#[test]
fn the_bias_channel_is_written_over_the_sample_and_reproduces_the_histogram() {
    let (stats, prof) = (toy_stats(), profile());
    let spec = testkit::spec_with("1", HIST, STATS);
    let p = plan(&Inputs {
        spec,
        routing: Some(&prof),
        calibration: None,
        stats: Some(&stats),
    })
    .unwrap();
    let h = testkit::HIDDEN as usize;
    let c = h - 1;
    let emb = bf16(&bytes_of(&p, "model.language_model.embed_tokens.weight"));
    let k = emb[c];
    let want_k = (h as f32).sqrt() / 2.0
        * stats.classes["model.language_model.embed_tokens.weight|BF16"]
            .bf16_rms()
            .unwrap();
    assert!((k - want_k).abs() <= want_k / 128.0, "K {k} vs {want_k}");
    assert!(
        emb.chunks(h).all(|r| r[c] == k),
        "the bias channel is constant"
    );
    let writer = bytes_of(
        &p,
        "model.language_model.layers.0.mlp.experts.0.down_proj.weight",
    );
    let cols = writer.len() / h;
    assert!(
        writer[c * cols..].iter().all(|&b| b == 0),
        "a writer's bias row is zero"
    );
    assert!(writer[..c * cols].iter().any(|&b| b != 0));
    let norm = bf16(&bytes_of(
        &p,
        "model.language_model.layers.0.post_attention_layernorm.weight",
    ));
    assert!(
        (norm[c] - TOY_BF16_RMS).abs() < 0.01,
        "the norm entry {} is the class RMS",
        norm[c]
    );

    let uniform = plan(&Inputs {
        spec: testkit::spec_with("1", "mode = \"uniform\"", STATS),
        routing: None,
        calibration: None,
        stats: Some(&stats),
    })
    .unwrap();
    let n = testkit::VOCAB as f64 * 2.0;
    for layer in 0..p.routers.len() {
        let want = &prof.layers[p.routers[layer].source_layer];
        let total = want.iter().sum::<u64>() as f64;
        // 2026-10-04: as the init-mode test: three times the expected multinomial TV of `n`
        // selections, plus the fit's own error.
        let sampling: f64 = want
            .iter()
            .map(|&c| {
                ((c as f64 / total) * (1.0 - c as f64 / total) / n).sqrt()
                    * (2.0 / std::f64::consts::PI).sqrt()
            })
            .sum::<f64>()
            / 2.0;
        let tol = 3.0 * sampling + p.routers[layer].tv + 0.01;
        let tv = total_variation(want, &routed(&p, layer));
        assert!(tv <= tol, "layer {layer}: tv {tv} > {tol}");
        let control = total_variation(want, &routed(&uniform, layer));
        assert!(
            control > 2.0 * tol,
            "layer {layer}: the uniform control is within {control}"
        );
    }
}

#[test]
fn a_calibration_gain_scales_only_the_router_column() {
    let (stats, prof) = (toy_stats(), profile());
    let spec = testkit::spec_with("1", HIST, STATS);
    let base = plan(&Inputs {
        spec,
        routing: Some(&prof),
        calibration: None,
        stats: Some(&stats),
    })
    .unwrap();
    let gains = base
        .routers
        .iter()
        .map(|r| (r.source_layer, 0.5f32))
        .collect();
    let cal = RoutingCalibration::parse(&RoutingCalibration::to_text(&gains)).unwrap();
    let half = plan(&Inputs {
        spec: testkit::spec_with("1", HIST_CAL, STATS),
        routing: Some(&prof),
        calibration: Some(&cal),
        stats: Some(&stats),
    })
    .unwrap();
    let h = testkit::HIDDEN as usize;
    for t in &base.tensors {
        let (a, b) = (bytes_of(&base, &t.name), bytes_of(&half, &t.name));
        if !t.name.ends_with("mlp.gate.weight")
            || !t.name.contains(".layers.")
            || t.name.starts_with("mtp.")
        {
            assert_eq!(a, b, "{}", t.name);
            continue;
        }
        let (a, b) = (bf16(&a), bf16(&b));
        for (ra, rb) in a.chunks(h).zip(b.chunks(h)) {
            assert_eq!(ra[..h - 1], rb[..h - 1], "{}: a noise entry moved", t.name);
            assert!(
                (rb[h - 1] - ra[h - 1] / 2.0).abs() <= ra[h - 1].abs() / 128.0,
                "{}",
                t.name
            );
        }
    }
    assert!(half.routers.iter().all(|r| r.gain == 0.5));
    assert!(
        half.resolved.contains("calibration = \"sha256:"),
        "{}",
        half.resolved
    );
}

#[test]
fn missing_inputs_are_refused() {
    let (stats, prof) = (toy_stats(), profile());
    let err = |i: Inputs<'_>| plan(&i).unwrap_err().to_string();
    let e = err(Inputs {
        spec: testkit::spec_with("1", "mode = \"uniform\"", STATS),
        routing: None,
        calibration: None,
        stats: None,
    });
    assert!(e.contains("none were supplied"), "{e}");
    let e = err(Inputs {
        spec: testkit::spec("1", "mode = \"uniform\""),
        routing: None,
        calibration: None,
        stats: Some(&stats),
    });
    assert!(e.contains("values.mode = \"init\""), "{e}");
    let mut thin = stats.clone();
    thin.classes.remove("L.mlp.gate.weight|BF16");
    let e = err(Inputs {
        spec: testkit::spec_with("1", HIST, STATS),
        routing: Some(&prof),
        calibration: None,
        stats: Some(&thin),
    });
    assert!(e.contains("no class `L.mlp.gate.weight|BF16`"), "{e}");
    let cal = RoutingCalibration::parse(&RoutingCalibration::to_text(&[(99, 1.0)].into())).unwrap();
    let e = err(Inputs {
        spec: testkit::spec_with("1", HIST_CAL, STATS),
        routing: Some(&prof),
        calibration: Some(&cal),
        stats: Some(&stats),
    });
    assert!(e.contains("no gain for source layer"), "{e}");
    let e = err(Inputs {
        spec: testkit::spec_with("1", HIST, STATS),
        routing: Some(&prof),
        calibration: Some(&cal),
        stats: Some(&stats),
    });
    assert!(e.contains("the spec names none"), "{e}");
    let e = err(Inputs {
        spec: testkit::spec_with("1", HIST_CAL, STATS),
        routing: Some(&prof),
        calibration: None,
        stats: Some(&stats),
    });
    assert!(e.contains("none was supplied"), "{e}");
}
