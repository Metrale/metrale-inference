// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: Tests of `METRALE_GLM_LAYER_STREAM`: the lever's values, the exact family the
//! hook adds (the `is_reuploaded` set minus activation scales, bounded by the MTP layer), and a
//! layer collected from its shard equal to the same layer collected from device copies.
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

use std::collections::HashMap;

use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_model_weights::weights::{DeferredTensor, WeightDtype, WeightStore, WeightTensor};

use super::layer_stream::{collected_from_shard, parse, streams};
use super::{LayerSource, is_reuploaded};

/// 2026-10-16: The reference checkpoint's text-layer count; layer 45 is the MTP layer.
const LAYERS: usize = 45;
const LAYER: usize = 4;

fn at(layer: usize, leaf: &str) -> String {
    format!("model.language_model.layers.{layer}.{leaf}")
}

#[test]
fn the_lever_takes_only_0_and_1() {
    assert!(!parse(None).unwrap());
    assert!(!parse(Some("0")).unwrap());
    assert!(parse(Some("1")).unwrap());
    for bad in ["", "true", "on", "2", " 1", "1 "] {
        assert!(parse(Some(bad)).is_err(), "{bad:?} must be refused");
    }
}

#[test]
fn the_streamed_family_is_the_reuploaded_set_minus_activation_scales() {
    use WeightDtype::{BF16, FP8E4M3, FP32, UInt8};
    // 2026-10-16: (name, dtype, streamed). The bounds: the last text layer streams, the MTP layer
    // does not; a routed expert never does, its shared expert does; an F32 `input_scale` (already
    // deferred and read on its own) does not, a non-F32 one would.
    let cases = [
        (at(0, "self_attn.q_proj.weight"), BF16, true),
        (at(0, "self_attn.q_norm.weight"), FP32, true),
        (at(3, "self_attn.indexer.wk.weight"), BF16, true),
        (at(1, "mlp.gate_proj.weight"), UInt8, true),
        (at(1, "mlp.gate_proj.weight_scale"), FP8E4M3, true),
        (at(1, "mlp.gate_proj.weight_scale_2"), FP32, true),
        (at(5, "mlp.shared_experts.down_proj.weight"), UInt8, true),
        (at(5, "mlp.gate.weight"), BF16, true),
        (at(LAYERS - 1, "input_layernorm.weight"), BF16, true),
        (at(LAYERS, "input_layernorm.weight"), BF16, false),
        (at(LAYERS, "eh_proj.weight"), BF16, false),
        (at(5, "mlp.experts.7.down_proj.weight"), UInt8, false),
        (at(5, "mlp.experts.7.down_proj.weight_scale_2"), FP32, false),
        (at(1, "mlp.gate_proj.input_scale"), FP32, false),
        (at(1, "mlp.gate_proj.input_scale"), BF16, true),
        (
            "model.language_model.embed_tokens.weight".into(),
            BF16,
            false,
        ),
        ("model.language_model.norm.weight".into(), BF16, false),
        ("lm_head.weight".into(), BF16, false),
        ("model.visual.blocks.0.attn.qkv.weight".into(), BF16, false),
    ];
    for (name, dtype, want) in &cases {
        assert_eq!(streams(name, *dtype, LAYERS), *want, "{name} {dtype:?}");
        if *want {
            assert!(
                is_reuploaded(name, LAYERS),
                "{name}: streamed but not re-uploaded"
            );
            let rel = name
                .split_once(".layers.")
                .unwrap()
                .1
                .split_once('.')
                .unwrap()
                .1;
            assert!(
                collected_from_shard(rel, name, *dtype),
                "{name}: streamed but never read back"
            );
        }
    }
}

/// 2026-10-16: Bytes that name their tensor and position, never zero.
fn tagged(len: usize, salt: usize) -> Vec<u8> {
    (0..len)
        .map(|i| ((i * 31 + salt * 7) % 251 + 1) as u8)
        .collect()
}

/// 2026-10-16: One layer's tensors as the hook would see them, plus a same-prefix-digit layer
/// (40) the collect of layer 4 must not pick up, a routed expert and an activation scale.
fn checkpoint() -> Vec<(String, WeightDtype, Vec<usize>, Vec<u8>)> {
    let t = |name: String, dtype: WeightDtype, shape: &[usize], salt: usize| {
        let len = shape.iter().product::<usize>() * dtype.byte_size();
        (name, dtype, shape.to_vec(), tagged(len, salt))
    };
    vec![
        t(
            at(LAYER, "self_attn.q_proj.weight"),
            WeightDtype::BF16,
            &[8, 16],
            1,
        ),
        // 2026-10-16: F32 under `self_attn.` takes the plan cast to BF16.
        t(
            at(LAYER, "self_attn.q_norm.weight"),
            WeightDtype::FP32,
            &[16],
            2,
        ),
        t(
            at(LAYER, "mlp.shared_experts.up_proj.weight"),
            WeightDtype::UInt8,
            &[8, 8],
            3,
        ),
        t(
            at(LAYER, "mlp.shared_experts.up_proj.weight_scale"),
            WeightDtype::FP8E4M3,
            &[8, 1],
            4,
        ),
        t(
            at(LAYER, "mlp.shared_experts.up_proj.weight_scale_2"),
            WeightDtype::FP32,
            &[],
            5,
        ),
        t(
            at(LAYER, "post_attention_layernorm.weight"),
            WeightDtype::BF16,
            &[16],
            6,
        ),
        t(
            at(40, "self_attn.q_proj.weight"),
            WeightDtype::BF16,
            &[8, 16],
            7,
        ),
        t(
            at(LAYER, "mlp.experts.3.down_proj.weight"),
            WeightDtype::UInt8,
            &[8, 8],
            8,
        ),
        t(
            at(LAYER, "mlp.shared_experts.up_proj.input_scale"),
            WeightDtype::FP32,
            &[],
            9,
        ),
    ]
}

/// 2026-10-16: A store holding `tensors`: those `defer` picks are written to one scratch shard
/// and deferred at their offsets, the rest uploaded to the mock device.
struct Built {
    gpu: MockGpuBackend,
    store: WeightStore,
    path: std::path::PathBuf,
}

impl Drop for Built {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn build(
    tag: &str,
    tensors: &[(String, WeightDtype, Vec<usize>, Vec<u8>)],
    defer: impl Fn(&str, WeightDtype) -> bool,
) -> Built {
    let dir = std::env::temp_dir().join(format!("metrale-glm5next-stream-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{tag}.bin"));
    let gpu = MockGpuBackend::new();
    let mut map = HashMap::new();
    let mut blob = vec![0x5Au8; 13];
    let mut deferred = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        if defer(name, *dtype) {
            let d = DeferredTensor {
                path: path.clone(),
                offset: blob.len() as u64,
                shape: shape.clone(),
                dtype: *dtype,
            };
            deferred.push((name.clone(), d));
            blob.extend_from_slice(bytes);
        } else {
            let p = gpu.alloc(bytes.len().max(1)).unwrap();
            gpu.copy_h2d(bytes, p).unwrap();
            let t = WeightTensor {
                ptr: p,
                shape: shape.clone(),
                dtype: *dtype,
            };
            map.insert(name.clone(), t);
        }
    }
    std::fs::write(&path, &blob).unwrap();
    let mut store = WeightStore::from_map(map);
    for (name, d) in deferred {
        store.defer(name, d);
    }
    Built { gpu, store, path }
}

/// 2026-10-16: What the binders can see of a collected layer: names, tensors and plan casts.
type Seen = (
    Vec<String>,
    Vec<(String, (WeightDtype, Vec<usize>, Vec<u8>))>,
    Vec<(String, (WeightDtype, Vec<u8>))>,
);

fn seen(b: &Built) -> Seen {
    let src = LayerSource::collect(&b.gpu, &b.store, LAYER).unwrap();
    let mut names = src.names.clone();
    names.sort();
    (
        names,
        src.tensors.clone().into_iter().collect(),
        src.plan_cast.clone().into_iter().collect(),
    )
}

/// 2026-10-16: The production hook's families without the env (read once per process): an
/// activation scale and, as under `--moe-expert-layout tp`, a routed expert are deferred either
/// way; `stream` adds the lever's family.
fn hook(stream: bool) -> impl Fn(&str, WeightDtype) -> bool {
    move |name, dtype| {
        super::is_activation_scale(name, dtype)
            || super::expert_tp::is_routed_expert_tensor(name)
            || (stream && streams(name, dtype, LAYERS))
    }
}

#[test]
fn a_streamed_layer_collects_exactly_what_the_resident_layer_does() {
    let ckpt = checkpoint();
    let resident = build("resident", &ckpt, hook(false));
    let streamed = build("streamed", &ckpt, hook(true));
    // 2026-10-16: The lever did move: no streamed tensor is left on the device.
    for (name, dtype, ..) in &ckpt {
        let on_device = streamed.store.contains(name);
        assert_eq!(on_device, !hook(true)(name, *dtype), "{name}");
    }
    assert!(
        resident
            .store
            .contains(&at(LAYER, "self_attn.q_proj.weight"))
    );

    let (want, got) = (seen(&resident), seen(&streamed));
    assert_eq!(got, want);
    // 2026-10-16: The comparison covers every path: a cast, both stored widths, and neither the
    // routed expert, the activation scale nor layer 40.
    assert_eq!(
        want.1.len(),
        6,
        "the six non-expert tensors of layer {LAYER}"
    );
    assert_eq!(want.2.len(), 1, "the F32 q_norm's plan cast");
    assert!(
        want.0
            .iter()
            .all(|n| !n.contains("experts.3") && !n.ends_with("input_scale"))
    );
}

#[test]
fn the_collected_bytes_come_from_the_shard() {
    // 2026-10-16: The positive control: one byte changed in the shard copy of a streamed tensor
    // (and not in the device copy of the resident store) moves the collected layer.
    let ckpt = checkpoint();
    let resident = build("ctl-resident", &ckpt, hook(false));
    let mut changed = ckpt.clone();
    changed[2].3[5] ^= 0x40;
    let streamed = build("ctl-streamed", &changed, hook(true));
    let (want, got) = (seen(&resident), seen(&streamed));
    assert_eq!(got.0, want.0, "the same names");
    assert_ne!(
        got.1, want.1,
        "a changed shard byte must reach the collected tensor"
    );
}
