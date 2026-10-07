// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: Image architecture residuals and native visibility contract controls.

use std::collections::BTreeSet;

use metrale_circuit::OpKind;
use metrale_circuit::image_attention::{Layout, LayoutError, Token};
use serde_json::Value;

const RESIDUAL: &str = include_str!("../../../kernels/circuits/residuals/qwen_image21.json");

fn token(sample: u32, image: Option<u32>, key_valid: bool) -> Token {
    Token {
        sample,
        image,
        key_valid,
    }
}

#[test]
fn image_visibility_matches_hand_written_mask_with_detection_controls() {
    let layout = Layout::new(vec![
        token(0, None, true),
        token(0, None, false),
        token(0, Some(0), true),
        token(0, Some(0), true),
        token(0, None, true),
        token(0, Some(1), true),
        token(0, Some(1), true),
        token(1, Some(0), true),
    ])
    .unwrap();
    let expected = [
        "10000000", "10000000", "10110000", "10110000", "10111000", "10111110", "10111110",
        "00000001",
    ];
    for (q, row) in expected.iter().enumerate() {
        for (k, want) in row.bytes().enumerate() {
            assert_eq!(
                layout.can_attend(q, k).unwrap(),
                want == b'1',
                "q={q}, k={k}"
            );
        }
    }
    // 2026-10-07: These distinguish causal-only, dense and padding-query mistakes.
    assert!(layout.can_attend(2, 3).unwrap());
    assert!(!layout.can_attend(0, 2).unwrap());
    assert!(layout.can_attend(1, 0).unwrap());
    assert!(!layout.can_attend(7, 2).unwrap());
    assert_eq!(layout.can_attend(8, 0), Err(LayoutError::Index));
    assert_eq!(layout.can_attend(0, 8), Err(LayoutError::Index));
}

#[test]
fn malformed_segments_are_refused_but_ids_can_repeat_in_other_samples() {
    assert_eq!(Layout::new(vec![]).unwrap_err(), LayoutError::Empty);
    for tokens in [
        vec![
            token(0, Some(1), true),
            token(0, None, true),
            token(0, Some(1), true),
        ],
        vec![
            token(0, None, true),
            token(1, None, true),
            token(0, None, true),
        ],
    ] {
        assert_eq!(Layout::new(tokens).unwrap_err(), LayoutError::Noncontiguous);
    }
    assert!(Layout::new(vec![token(0, Some(1), true), token(1, Some(1), true)]).is_ok());
}

#[test]
fn architecture_graph_is_closed_and_residual_ops_cannot_be_silently_lowered() {
    let data: Value = serde_json::from_str(RESIDUAL).unwrap();
    assert_eq!(data["executable"], false);
    assert_eq!(data["status"], "unlowered");
    let residuals = data["residuals"].as_object().unwrap();
    for op in residuals.keys() {
        assert!(OpKind::parse(op, None, None).is_err());
    }
    let manifest: Value = serde_json::from_str(include_str!(
        "../../../docs/model-manifests/qwen-image-2.1.json"
    ))
    .unwrap();
    assert_eq!(data["revision"], manifest["revision"]);
    let mut pipeline = BTreeSet::from(["prompt", "reference_images", "noise", "timesteps"]);
    for component in data["component_flow"].as_array().unwrap() {
        assert!(residuals.contains_key(component["residual"].as_str().unwrap()));
        for input in component["inputs"].as_array().unwrap() {
            assert!(pipeline.contains(input.as_str().unwrap()));
        }
        for output in component["outputs"].as_array().unwrap() {
            assert!(pipeline.insert(output.as_str().unwrap()));
        }
    }
    let mut available: BTreeSet<&str> = data["block_inputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap())
        .collect();
    for node in data["block_nodes"].as_array().unwrap() {
        for input in node["inputs"].as_array().unwrap() {
            assert!(
                available.contains(input.as_str().unwrap()),
                "unknown or forward input {input}"
            );
        }
        let op = node["op"].as_str().unwrap();
        if residuals.contains_key(op) {
            assert!(
                OpKind::parse(op, None, None).is_err(),
                "residual unexpectedly executable: {op}"
            );
        } else {
            assert!(
                OpKind::parse(op, None, None).is_ok(),
                "unclassified op: {op}"
            );
        }
        assert!(
            available.insert(node["id"].as_str().unwrap()),
            "duplicate graph id"
        );
    }
    for output in data["block_outputs"].as_array().unwrap() {
        assert!(available.contains(output.as_str().unwrap()));
    }
    assert_eq!(data["dimensions"]["hidden"], 4096);
    assert_eq!(data["dimensions"]["head_dim"], 128);
}
