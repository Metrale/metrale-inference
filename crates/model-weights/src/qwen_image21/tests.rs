// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: Pinned real-header controls. No tensor payload or fake GPU allocation.

use super::*;
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

const CONFIG: &str = include_str!("../../tests/fixtures/qwen-image21/config.json");
const HEADERS: &str = include_str!("../../tests/fixtures/qwen-image21/headers.json");
const PROVENANCE: &str = include_str!("../../tests/fixtures/qwen-image21/provenance.json");

#[derive(Clone, Deserialize)]
struct Header {
    shape: Vec<usize>,
    dtype: String,
    shard: String,
    data_offsets: [usize; 2],
}

fn headers() -> BTreeMap<String, Header> {
    serde_json::from_str(HEADERS).unwrap()
}

fn views(headers: &BTreeMap<String, Header>) -> BTreeMap<String, Tensor<'_, Header>> {
    headers
        .iter()
        .map(|(name, header)| {
            (
                name.clone(),
                Tensor {
                    storage: header,
                    shape: &header.shape,
                    dtype: WeightDtype::from_safetensors_str(&header.dtype).unwrap(),
                },
            )
        })
        .collect()
}

#[test]
fn actual_headers_bind_every_block_and_preserve_borrowed_storage() {
    let headers = headers();
    let views = views(&headers);
    let bound = bind(&Config::parse(CONFIG).unwrap(), &views).unwrap();
    assert_eq!(bound.blocks.len(), 32);
    assert!(std::ptr::eq(bound.image_out, &headers["proj_out.weight"]));
    for (i, block) in bound.blocks.iter().enumerate() {
        assert!(std::ptr::eq(
            block.gate,
            &headers[&format!("transformer_blocks.{i}.img_mlp.gate_layer.weight")]
        ));
        assert!(std::ptr::eq(
            block.up,
            &headers[&format!("transformer_blocks.{i}.img_mlp.proj.weight")]
        ));
        assert!(std::ptr::eq(
            block.down,
            &headers[&format!("transformer_blocks.{i}.img_mlp.out.weight")]
        ));
    }
}

#[test]
fn missing_extra_wrong_shape_and_dtype_are_refused() {
    let config = Config::parse(CONFIG).unwrap();
    let name = "transformer_blocks.31.attn.to_v.weight";
    let mut missing = headers();
    let old = missing.remove(name).unwrap();
    assert!(bind(&config, &views(&missing)).is_err());
    missing.insert("unexpected.weight".into(), old);
    assert!(
        bind(&config, &views(&missing))
            .err()
            .unwrap()
            .to_string()
            .contains(name)
    );
    let mut extra = headers();
    extra.insert("unexpected.weight".into(), extra[name].clone());
    assert!(bind(&config, &views(&extra)).is_err());
    let mut shape = headers();
    shape
        .get_mut("transformer_blocks.0.img_mlp.out.weight")
        .unwrap()
        .shape
        .reverse();
    assert!(
        bind(&config, &views(&shape))
            .err()
            .unwrap()
            .to_string()
            .contains("wrong shape")
    );
    let mut dtype = headers();
    dtype.get_mut("txt_in.text_norm.weight").unwrap().dtype = "F32".into();
    assert!(
        bind(&config, &views(&dtype))
            .err()
            .unwrap()
            .to_string()
            .contains("wrong dtype")
    );
}

#[test]
fn config_refuses_missing_unknown_and_math_changing_fields() {
    let original: Value = serde_json::from_str(CONFIG).unwrap();
    for (key, bad) in [
        ("num_layers", Value::from(31)),
        ("attention_head_dim", Value::from(64)),
        ("causal_condition", Value::from(false)),
        ("eps", Value::from(1e-5)),
        ("unknown_math_policy", Value::from(true)),
    ] {
        let mut changed = original.clone();
        changed[key] = bad;
        assert!(Config::parse(&changed.to_string()).is_err(), "{key}");
    }
    let mut missing = original;
    missing.as_object_mut().unwrap().remove("patch_size");
    assert!(Config::parse(&missing.to_string()).is_err());
    assert!(Config::parse(CONFIG).is_ok());
}

#[test]
fn fixture_hashes_and_actual_shard_extents_are_consistent() {
    let provenance: Value = serde_json::from_str(PROVENANCE).unwrap();
    assert_eq!(
        provenance["revision"],
        "d26bb61231c349cf6b7896fa83353113880e1ba3"
    );
    for (name, text) in [("config.json", CONFIG), ("headers.json", HEADERS)] {
        assert_eq!(
            Sha256::digest(text.as_bytes())
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>(),
            provenance["files"][name].as_str().unwrap()
        );
    }
    let headers = headers();
    for shard in provenance["shards"].as_array().unwrap() {
        let name = shard["file"].as_str().unwrap();
        let mut entries: Vec<_> = headers.values().filter(|h| h.shard == name).collect();
        entries.sort_by_key(|h| h.data_offsets[0]);
        let mut offset = 0;
        for header in entries {
            assert_eq!(header.dtype, "BF16");
            assert_eq!(header.data_offsets[0], offset);
            offset += 2 * header.shape.iter().product::<usize>();
            assert_eq!(header.data_offsets[1], offset);
        }
        assert_eq!(
            offset + 8 + shard["header_bytes"].as_u64().unwrap() as usize,
            shard["bytes"].as_u64().unwrap() as usize
        );
    }
}
