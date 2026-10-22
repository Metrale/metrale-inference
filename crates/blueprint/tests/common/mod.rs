// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: A small, complete blueprint and a test trust world. Keys come from fixed test
//! seeds; none is a real key.

#![allow(dead_code)]

use std::collections::BTreeMap;

use metrale_abstract_io::UnixMs;
use metrale_blueprint::frame::{self, KernelSource, Unsigned};
use metrale_blueprint::ids::{ArtifactId, LineageId, Sha256};
use metrale_blueprint::types::*;
use metrale_blueprint_trust::{
    Anchor, KeyCert, Kid, Revoked, Scope, SignedCert, Signer, Suite, TrustAnchors, TrustPolicy,
    certify,
};

pub const NOW: i64 = 1_760_000_000_000;
pub const MOD_A: &[u8] = b".version 8.8\n.visible .entry add_one(\n.param .u64 p)\n{ ret; }\n";
pub const MOD_B: &[u8] = b".version 8.8\n.visible .entry scale(\n.param .u64 p)\n{ ret; }\n";

pub fn internal(name: &str) -> Option<&'static [u8]> {
    match name {
        "add" => Some(MOD_A),
        "scale" => Some(MOD_B),
        _ => None,
    }
}

fn module(name: &str, bytes: &[u8], entry: &str) -> ModuleSpec {
    ModuleSpec {
        name: name.into(),
        kind: ArtifactKind::PtxText,
        arch: "sm_121f".into(),
        artifact: ArtifactId::of(bytes),
        len: bytes.len() as u64,
        entries: vec![entry.into()],
    }
}

fn kid(module: &str, func: &str) -> KernelId {
    KernelId {
        module: module.into(),
        func: func.into(),
    }
}

pub fn blueprint() -> Blueprint {
    let node = |id: &str, op: &str, i: Vec<u32>, o: Vec<u32>, w: Option<&str>| NodeSpec {
        id: id.into(),
        local: id.rsplit('.').next().unwrap().into(),
        op: op.into(),
        inputs: i,
        outputs: o,
        weight: w.map(Into::into),
        binding: vec![],
        params: BTreeMap::new(),
        layer: Some(0),
        block: "toy".into(),
        state: vec![],
    };
    let edge = |id: &str, p: Option<u32>, c: Vec<u32>| EdgeSpec {
        id: id.into(),
        format: "bf16".into(),
        rows: "n".into(),
        dim: "hidden".into(),
        dim_value: 64,
        producer: p,
        consumers: c,
        is_output: false,
        binds: None,
    };
    let mut flags = BTreeMap::new();
    flags.insert("max_model_len".to_string(), FlagValue::Int(4096));
    flags.insert("kv_cache_dtype".to_string(), FlagValue::Text("fp8".into()));
    flags.insert(
        "gpu_memory_utilization".to_string(),
        FlagValue::Fixed(Fixed {
            mantissa: 82,
            exp: -2,
        }),
    );
    let mut levers = BTreeMap::new();
    levers.insert("METRALE_GLM_ROW_GROUP".to_string(), LeverValue::Int(128));
    Blueprint {
        schema: BODY_SCHEMA,
        lineage: LineageId::of("acme/toy", "gb10", "toy"),
        revision: 1,
        name: "toy".into(),
        engine: EnginePin {
            version: "test".into(),
            digest_schema: 2,
            rules_sha256: Sha256::of(&[b"rules"]),
        },
        checkpoint: CheckpointPin {
            repo: "acme/toy".into(),
            revision: "0123".into(),
            config_sha256: Sha256::of(&[b"{}"]),
            quant_config_sha256: None,
            weights: WeightsDigest::NotPinned,
        },
        hardware: HardwareSpec {
            class: "gb10".into(),
            ptx_arch: "sm_121f".into(),
            vendor: Vendor::Nvidia,
            sm_count: 48,
            world_size: 1,
            min_visible_mib: 1024,
            devices: vec!["gb10".into()],
        },
        circuit: CircuitSpec {
            arch: "toy".into(),
            dims: [("hidden".to_string(), 64u64)].into_iter().collect(),
            layer_kinds: vec![LayerKind::LinearAttention],
            nodes: vec![
                node("l0.toy.add", "add", vec![0], vec![1], None),
                node(
                    "l0.toy.proj",
                    "linear:o",
                    vec![1],
                    vec![2],
                    Some("nvfp4/g16"),
                ),
            ],
            edges: vec![
                edge("in", None, vec![0]),
                edge("mid", Some(0), vec![1]),
                edge("out", Some(1), vec![]),
            ],
            blocks: vec![BlockSpec {
                template: "toy".into(),
                layer: Some(0),
                section: Section::Main,
                first: 0,
                end: 2,
            }],
            states: vec![],
        },
        precision: PrecisionSpec {
            tier: "toy".into(),
            table_sha256: Sha256::of(&[b"table"]),
        },
        plans: vec![PlanSpec {
            mode: Mode::Decode,
            rows: 1,
            groups: vec![
                GroupSpec {
                    rule: "add".into(),
                    kernels: vec![kid("add", "add_one")],
                    repeat: Repeat::Once,
                    copies: None,
                    emitter: "add".into(),
                    numerics: Numerics::Reference,
                    nodes: vec![0],
                },
                GroupSpec {
                    rule: "proj".into(),
                    kernels: vec![kid("scale", "scale")],
                    repeat: Repeat::PerRow,
                    copies: None,
                    emitter: "gemv".into(),
                    numerics: Numerics::BitIdentical {
                        microtest: "t".into(),
                    },
                    nodes: vec![1],
                },
            ],
            edge_states: vec![
                EdgeStateSpec::Absent,
                EdgeStateSpec::Fused(0),
                EdgeStateSpec::Materialized,
            ],
            edge_formats: vec!["bf16".into(), "bf16".into(), "bf16".into()],
            digest: Sha256::of(&[b"plan"]),
        }],
        plan_gaps: vec![],
        kernels: KernelSet {
            target: KernelTarget {
                arch: "sm_121".into(),
                model: "toy".into(),
                quant: "nvfp4".into(),
            },
            modules: vec![
                module("add", MOD_A, "add_one"),
                module("scale", MOD_B, "scale"),
            ],
            specs: vec![
                KernelSpec {
                    id: kid("add", "add_one"),
                    module: 0,
                    family: Some("elementwise".into()),
                    compute: Some("memory".into()),
                },
                KernelSpec {
                    id: kid("scale", "scale"),
                    module: 1,
                    family: None,
                    compute: None,
                },
            ],
        },
        settings: ServeSettings {
            registry_sha256: Sha256::of(&[b"levers"]),
            levers,
            passthrough: BTreeMap::new(),
        },
        policy: ServePolicy {
            flags,
            chat_template: None,
        },
        evidence: vec![],
        execution: Execution::Legacy {
            family: "toy".into(),
        },
    }
}

pub fn unsigned(bp: &Blueprint) -> Unsigned {
    Unsigned {
        sources: bp
            .kernels
            .modules
            .iter()
            .map(|_| KernelSource::Internal)
            .collect(),
        resources: Resources {
            recipe_id: "toy/toy".into(),
            recipe_version: "2".into(),
            runtime: Some("metrale".into()),
            container: "c".into(),
            flags: [("port".to_string(), "8000".to_string())]
                .into_iter()
                .collect(),
        },
    }
}

pub struct World {
    pub root: Signer,
    pub leaf: Signer,
    pub anchors: TrustAnchors,
    pub revoked: Revoked,
}

pub fn world() -> World {
    let root = Signer::from_seeds(
        Kid::new("test-root").unwrap(),
        Suite::MlDsa87Ed25519,
        &[1; 32],
        &[2; 32],
    );
    let leaf = Signer::from_seeds(
        Kid::new("test-blueprints").unwrap(),
        Suite::MlDsa65Ed25519,
        &[3; 32],
        &[4; 32],
    );
    let anchors = TrustAnchors::new(vec![Anchor {
        kid: root.kid().clone(),
        keys: root.public(),
    }])
    .unwrap();
    World {
        root,
        leaf,
        anchors,
        revoked: Revoked::embedded().unwrap(),
    }
}

impl World {
    pub fn policy(&self) -> TrustPolicy<'_> {
        TrustPolicy {
            anchors: &self.anchors,
            revoked: &self.revoked,
            now: UnixMs(NOW),
        }
    }

    pub fn cert(&self) -> SignedCert {
        certify(
            &self.root,
            &KeyCert {
                kid: self.leaf.kid().clone(),
                keys: self.leaf.public(),
                scope: Scope::Blueprints,
                not_before_ms: NOW - 1000,
                not_after_ms: NOW + 1000,
                serial: 1,
            },
        )
        .unwrap()
    }

    /// 2026-10-22: The signed lite file of [`blueprint`].
    pub fn signed(&self) -> Vec<u8> {
        let bp = blueprint();
        let file = frame::encode_unsigned(&bp, &unsigned(&bp), &[]).unwrap();
        frame::sign(&file, &self.leaf, vec![self.cert()]).unwrap()
    }
}

/// 2026-10-22: Patch the header's body hash to match the (possibly edited) body.
pub fn rehash(file: &mut [u8]) {
    let body_len = u64::from_le_bytes(file[8..16].try_into().unwrap()) as usize;
    let h = Sha256::of(&[&file[72..72 + body_len]]);
    file[16..48].copy_from_slice(&h.0);
}
