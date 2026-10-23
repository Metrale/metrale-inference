// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: A small, complete blueprint and a test trust world. Keys come from fixed test
//! seeds; none is a real key.

#![allow(dead_code)]

use std::collections::BTreeMap;

use metrale_abstract_io::UnixMs;
use metrale_blueprint::codec::{Codec, DecodeMemory};
use metrale_blueprint::frame::{self, KernelSource, Unsigned};
use metrale_blueprint::ids::{ArtifactId, LineageId, Sha256};
use metrale_blueprint::types::*;
use metrale_blueprint_trust::{
    Anchor, KeyCert, Kid, Revoked, Scope, SignedCert, Signer, Suite, TrustAnchors, TrustPolicy,
    certify,
};
use metrale_circuit::Format;
use metrale_circuit::dims::DimExpr;
use metrale_circuit::fuser::{EdgeState, Group};
use metrale_circuit::ir::{BlockInstance, Edge, LayerKind, LinearRole, Node, OpKind, Section};
use metrale_circuit::rules::{Numerics, Repeat};
use metrale_circuit::venn::compute::ComputeUnit;

pub const NOW: i64 = 1_760_000_000_000;
/// 2026-10-22: Test decode budget: the store's wasm verifier budget (kzx DESIGN.md §4.2); every
/// toy file here fits it.
pub const MEM: DecodeMemory = DecodeMemory(64 << 20);
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
    let node = |id: &str, op: OpKind, i: Vec<usize>, o: Vec<usize>, w: Option<Format>| Node {
        id: id.into(),
        local: id.rsplit('.').next().unwrap().into(),
        op,
        inputs: i,
        outputs: o,
        weight: w,
        binding: vec![],
        params: BTreeMap::new(),
        layer: Some(0),
        block: "toy".into(),
        state: vec![],
    };
    let edge = |id: &str, p: Option<usize>, c: Vec<usize>| Edge {
        id: id.into(),
        format: Format::Bf16,
        rows: DimExpr::parse("n").unwrap(),
        dim: DimExpr::parse("hidden").unwrap(),
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
        circuit: Circuit {
            arch: "toy".into(),
            description: "a two-node toy".into(),
            nodes: vec![
                node("l0.toy.add", OpKind::ResidualAdd, vec![0], vec![1], None),
                node(
                    "l0.toy.proj",
                    OpKind::Linear(LinearRole::O),
                    vec![1],
                    vec![2],
                    Some(Format::Nvfp4 { group: 16 }),
                ),
            ],
            edges: vec![
                edge("in", None, vec![0]),
                edge("mid", Some(0), vec![1]),
                edge("out", Some(1), vec![]),
            ],
            blocks: vec![BlockInstance {
                template: "toy".into(),
                layer: Some(0),
                section: Section::Main,
                first: 0,
                end: 2,
                stream_in: Some(0),
                stream_out: Some(2),
            }],
            layer_kinds: vec![LayerKind::LinearAttention],
            dims: [("hidden".to_string(), 64u64)].into_iter().collect(),
            states: vec![],
        },
        precision: PrecisionPin {
            spec: metrale_circuit::PrecisionSpec::Table("toy".into()),
            table_sha256: Sha256::of(&[b"table"]),
        },
        plans: vec![FusionPlan {
            arch: "toy".into(),
            mode: Mode::Decode,
            rows: 1,
            table: None,
            groups: vec![
                Group {
                    rule: "add".into(),
                    kernels: vec![kid("add", "add_one")],
                    repeat: Repeat::Once,
                    copies: None,
                    emitter: "add".into(),
                    numerics: Numerics::Reference,
                    nodes: vec![0],
                    runs: vec![],
                },
                Group {
                    rule: "proj".into(),
                    kernels: vec![kid("scale", "scale")],
                    repeat: Repeat::PerRow,
                    copies: None,
                    emitter: "gemv".into(),
                    numerics: Numerics::BitIdentical {
                        microtest: "t".into(),
                    },
                    nodes: vec![1],
                    runs: vec![],
                },
            ],
            edge_states: vec![
                None,
                Some(EdgeState::Fused(0)),
                Some(EdgeState::Materialized),
            ],
            edge_formats: vec![Format::Bf16, Format::Bf16, Format::Bf16],
            digest: Sha256::of(&[b"plan"]).hex(),
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
                    compute: Some(ComputeUnit::Memory),
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
        frame::sign(&file, &self.leaf, vec![self.cert()], MEM).unwrap()
    }
}

/// 2026-10-22: [`World::signed`] with its body stored raw (`Codec::Stored`), so a test can
/// edit body bytes directly.
pub fn stored(w: &World) -> Vec<u8> {
    frame::recode(&w.signed(), &w.policy(), MEM, Codec::Stored).unwrap()
}

/// 2026-10-22: The body section's range.
pub fn body_range(file: &[u8]) -> std::ops::Range<usize> {
    let body_len = u64::from_le_bytes(file[8..16].try_into().unwrap()) as usize;
    frame::HEADER_LEN..frame::HEADER_LEN + body_len
}

/// 2026-10-22: Patch the header's body hash to match the (possibly edited) body of a
/// `Codec::Stored` file, whose stored bytes are its raw bytes.
pub fn rehash(file: &mut [u8]) {
    assert_eq!(file[7], Codec::Stored.id(), "rehash needs a stored body");
    let h = Sha256::of(&[&file[body_range(file)]]);
    file[24..56].copy_from_slice(&h.0);
}

/// 2026-10-22: The signed full file of [`blueprint`]: both modules inline, the payload one
/// solid block under `Codec::WRITE`.
pub fn full(w: &World) -> Vec<u8> {
    let lite = w.signed();
    let o = frame::open(&lite, &w.policy(), MEM).unwrap();
    let (u, p) = metrale_blueprint::resolve::full(&o, internal).unwrap();
    frame::rewrite(&lite, &w.policy(), MEM, &u, &p).unwrap()
}

/// 2026-10-22: The payload section's range.
pub fn payload_range(file: &[u8]) -> std::ops::Range<usize> {
    let (_, s) = frame::Header::parse(file).unwrap();
    s.payload
}

/// 2026-10-22: A recipient key from a fixed test seed.
pub fn recipient(seed: u8) -> metrale_seal::HybridSecret {
    metrale_seal::HybridSecret::from_seed(&[seed; 32])
}

/// 2026-10-22: A seal run's randomness for `n` recipients: bytes `start, start+1, …`.
pub fn run_random(start: u8, n: usize) -> metrale_seal::SealRandom {
    let len = metrale_seal::SealRandom::len(n);
    let bytes: Vec<u8> = (0..len).map(|i| start.wrapping_add(i as u8)).collect();
    metrale_seal::SealRandom::from_bytes(&bytes, n).unwrap()
}

/// 2026-10-22: Seal `input` (a signed plain file of the toy) for `keys` as `customer`, signed
/// by the test leaf.
pub fn seal_for(
    w: &World,
    input: &[u8],
    keys: &[&metrale_seal::HybridSecret],
    customer: &str,
    placement: metrale_blueprint::sealing::Placement<'_>,
    start: u8,
) -> metrale_blueprint::sealing::SealedOutput {
    let pks: Vec<_> = keys.iter().map(|k| k.public()).collect();
    let req = metrale_blueprint::sealing::SealRequest {
        recipients: &pks,
        customer_id: customer,
        issued_ms: NOW,
        codec: Codec::Kzx,
        placement,
    };
    metrale_blueprint::sealing::seal(
        input,
        &w.policy(),
        MEM,
        internal,
        &req,
        run_random(start, pks.len()),
        &w.leaf,
        vec![w.cert()],
    )
    .unwrap()
}

/// 2026-10-22: The toy sealed full for recipient seed 7, customer `acme`.
pub fn sealed_full(w: &World) -> Vec<u8> {
    seal_for(
        w,
        &w.signed(),
        &[&recipient(7)],
        "acme",
        metrale_blueprint::sealing::Placement::Inline,
        0,
    )
    .file
}

/// 2026-10-22: Which section [`splice`] replaces.
#[derive(Clone, Copy)]
pub enum Part {
    Locator,
    Payload,
    Sealed,
}

/// 2026-10-22: `file` with one section replaced by `bytes` (header lengths patched; flags as
/// given), the other sections verbatim.
pub fn splice(file: &[u8], part: Part, bytes: &[u8], flags: u8) -> Vec<u8> {
    let (mut h, s) = frame::Header::parse(file).unwrap();
    let mut sections = [
        file[s.locator.clone()].to_vec(),
        file[s.payload.clone()].to_vec(),
        file[s.sealed.clone()].to_vec(),
    ];
    sections[part as usize] = bytes.to_vec();
    h.flags = flags;
    h.locator_len = sections[0].len() as u64;
    h.payload_len = sections[1].len() as u64;
    h.sealed_len = sections[2].len() as u64;
    let mut out = h.write().to_vec();
    out.extend_from_slice(&file[s.body]);
    for x in &sections {
        out.extend_from_slice(x);
    }
    out.extend_from_slice(&file[s.sig]);
    out
}
