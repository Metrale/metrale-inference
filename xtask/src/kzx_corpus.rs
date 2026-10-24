// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: `cargo xtask kzx-corpus <corpus-dir>`: the private benchmark corpus for the
//! `kzx` compressor. Reads the compiled PTX modules under `<corpus-dir>/ptx/` and writes
//! GLM-scale postcard blueprint bodies to `<corpus-dir>/body/` and assembled `.kbp` containers
//! (the layout of claude-state/glm/blueprint/DESIGN.md §3) to `<corpus-dir>/kbp/`.
//!
//! The bodies mirror the Phase-1 wire types (DESIGN.md §2) with real kernel entry names, real
//! artifact hashes and the real lever names from `crates/settings`, so their byte statistics
//! are those of a shipped blueprint. Nothing here is committed: the corpus stays outside the
//! repository.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

#[derive(Serialize)]
struct Blueprint {
    schema: u32,
    lineage: [u8; 32],
    revision: u64,
    name: String,
    engine: (String, u32, u32),
    checkpoint: Checkpoint,
    hardware: Hardware,
    circuit: Circuit,
    precision: Precision,
    plans: Vec<Plan>,
    kernels: KernelSet,
    settings: Settings,
    policy: Policy,
    evidence: Vec<(String, [u8; 32], [u8; 32])>,
    execution: u8,
}

#[derive(Serialize)]
struct Checkpoint {
    repo: String,
    revision: String,
    config_sha256: [u8; 32],
    quant_config_sha256: Option<[u8; 32]>,
    shards: Vec<(String, u64, [u8; 32])>,
    root: [u8; 32],
}

#[derive(Serialize)]
struct Hardware {
    class: String,
    ptx_arch: String,
    vendor: u8,
    sm_count: u32,
    world_size: u32,
    min_visible_mib: u64,
    devices: Vec<String>,
}

#[derive(Serialize)]
struct Node {
    id: String,
    local: String,
    op: u8,
    inputs: Vec<u32>,
    outputs: Vec<u32>,
    weight: Option<u8>,
    binding: Vec<String>,
    params: BTreeMap<String, String>,
    layer: Option<u32>,
    block: String,
}

#[derive(Serialize)]
struct Circuit {
    arch: String,
    dims: BTreeMap<String, u64>,
    layer_kinds: Vec<u8>,
    nodes: Vec<Node>,
    edges: Vec<(u32, u32, u8, String)>,
    blocks: Vec<(String, u8, Vec<u32>)>,
    states: Vec<(String, u8, Vec<u64>)>,
}

#[derive(Serialize)]
struct Precision {
    tier: String,
    table_sha256: [u8; 32],
    linear: Vec<(u32, u8, u8)>,
}

#[derive(Serialize)]
struct Group {
    rule: String,
    kernels: Vec<(String, String)>,
    repeat: u8,
    copies: u32,
    emitter: String,
    numerics: u8,
    nodes: Vec<u32>,
}

#[derive(Serialize)]
struct Plan {
    mode: u8,
    rows: u64,
    table: Option<Vec<u32>>,
    groups: Vec<Group>,
    edge_formats: Vec<u8>,
    digest: [u8; 32],
}

/// 2026-10-23: A kernel's family: its name, its parameter bindings and its class byte.
type FamilySpec = (String, Vec<(String, String)>, u8);

#[derive(Serialize)]
struct KernelSpec {
    id: (String, String),
    family: Option<FamilySpec>,
    arch: String,
    kind: u8,
    launch: (String, Vec<u8>, u32, (u32, u32, u32), String, u64),
    numerics: u8,
    pipeline: Option<Vec<u8>>,
    artifact: [u8; 32],
    manifest: [u8; 32],
}

#[derive(Serialize)]
struct KernelSet {
    target: (u8, String),
    specs: Vec<KernelSpec>,
}

#[derive(Serialize)]
enum Lever {
    Switch(bool),
    Presence,
    Int(i64),
    Fixed(i64, i8),
    Text(String),
}

#[derive(Serialize)]
struct Settings {
    registry_sha256: [u8; 32],
    levers: BTreeMap<String, Lever>,
}

#[derive(Serialize)]
struct Policy {
    max_seq_len: u64,
    kv_cache_dtype: u8,
    speculative: (u8, u32, u32),
    batch: (u32, u32, u32),
    lowering: BTreeMap<String, String>,
}

struct Module {
    name: String,
    entries: Vec<String>,
    sha: [u8; 32],
    bytes: Vec<u8>,
}

fn sha(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn sha_of(rng: &mut Rng) -> [u8; 32] {
    sha(&rng.next().to_le_bytes())
}

fn read_modules(dir: &Path) -> Result<Vec<Module>, String> {
    let mut out = Vec::new();
    let mut paths: Vec<_> = fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "ptx"))
        .collect();
    paths.sort();
    for p in paths {
        let bytes = fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        let text = String::from_utf8_lossy(&bytes);
        let entries = text
            .lines()
            .filter_map(|l| l.split(".entry ").nth(1))
            .filter_map(|r| r.split('(').next())
            .map(|s| s.trim().to_owned())
            .collect();
        let stem = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let name = stem.split("__").nth(1).unwrap_or(&stem).to_owned();
        out.push(Module {
            name,
            entries,
            sha: sha(&bytes),
            bytes,
        });
    }
    Ok(out)
}

fn lever_names(root: &Path) -> Result<Vec<String>, String> {
    let dir = root.join("crates/settings/src/groups");
    let mut names = std::collections::BTreeSet::new();
    for e in fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))? {
        let p = e.map_err(|e| e.to_string())?.path();
        if p.extension().is_none_or(|x| x != "rs") {
            continue;
        }
        let src = fs::read_to_string(&p).map_err(|e| e.to_string())?;
        for part in src.split("\"METRALE_").skip(1) {
            let name: String = part
                .chars()
                .take_while(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || *c == '_')
                .collect();
            if !name.is_empty() {
                names.insert(format!("METRALE_{name}"));
            }
        }
    }
    Ok(names.into_iter().collect())
}

const OPS: [&str; 12] = [
    "norm", "linear:q", "linear:k", "linear:v", "linear:o", "gdn", "attn", "linear:gate_up",
    "linear:down", "act", "residual", "router",
];

fn circuit(rng: &mut Rng, layers: u32) -> Circuit {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut blocks = Vec::new();
    let mut states = Vec::new();
    let mut layer_kinds = Vec::new();
    for l in 0..layers {
        let kind = u8::from(l % 4 == 3);
        layer_kinds.push(kind);
        let block = if kind == 1 { format!("l{l}.attn") } else { format!("l{l}.kda") };
        let first = u32::try_from(nodes.len()).unwrap_or(0);
        for (k, op) in OPS.iter().enumerate() {
            let id = u32::try_from(nodes.len()).unwrap_or(0);
            let local = op.split(':').next_back().unwrap_or(op).to_owned();
            let mut params = BTreeMap::new();
            if op.starts_with("linear") {
                params.insert("n".to_owned(), format!("{}", 1024 * (1 + rng.below(8))));
                params.insert("k".to_owned(), "5120".to_owned());
            }
            nodes.push(Node {
                id: format!("{block}.{local}"),
                local,
                op: u8::try_from(k).unwrap_or(0),
                inputs: if id > 0 { vec![id - 1] } else { vec![] },
                outputs: vec![id + 1],
                weight: op.starts_with("linear").then_some(u8::try_from(rng.below(3)).unwrap_or(0)),
                binding: vec![format!("layers.{l}.{}", op.replace(':', "."))],
                params,
                layer: Some(l),
                block: block.clone(),
            });
            if id > 0 {
                edges.push((id - 1, id, u8::try_from(rng.below(4)).unwrap_or(0), format!("{block}.e{k}")));
            }
        }
        let last = u32::try_from(nodes.len()).unwrap_or(0);
        blocks.push((block.clone(), kind, (first..last).collect()));
        if kind == 0 {
            states.push((format!("{block}.ssm"), 1, vec![16, 128, 128]));
        }
    }
    let mut dims = BTreeMap::new();
    for (k, v) in [("hidden", 5120u64), ("ffn", 17408), ("heads", 24), ("kv_heads", 4), ("head_dim", 256), ("vocab", 248_320)] {
        dims.insert(k.to_owned(), v);
    }
    Circuit {
        arch: "glm5_next".to_owned(),
        dims,
        layer_kinds,
        nodes,
        edges,
        blocks,
        states,
    }
}

fn plans(rng: &mut Rng, modules: &[Module], node_count: u32) -> Vec<Plan> {
    let ids: Vec<(String, String)> = modules
        .iter()
        .flat_map(|m| m.entries.iter().map(move |e| (m.name.clone(), e.clone())))
        .collect();
    let arms = [(0u8, 1u64), (0, 2), (0, 4), (0, 8), (1, 1), (1, 2), (2, 1), (2, 3), (3, 16), (3, 64)];
    arms.iter()
        .map(|&(mode, rows)| {
            let n = 700 + rng.below(200);
            let groups = (0..n)
                .map(|g| {
                    let (m, f) = &ids[usize::try_from(rng.below(ids.len() as u64)).unwrap_or(0)];
                    Group {
                        rule: format!("{}_{}", f.split('_').next().unwrap_or("rule"), ["w4a16", "fp8", "bf16", "nvfp4"][usize::try_from(rng.below(4)).unwrap_or(0)]),
                        kernels: vec![(m.clone(), f.clone())],
                        repeat: u8::try_from(rng.below(3)).unwrap_or(0),
                        copies: 1,
                        emitter: f.split('_').next().unwrap_or("emit").to_owned(),
                        numerics: u8::try_from(rng.below(3)).unwrap_or(0),
                        nodes: vec![u32::try_from(g).unwrap_or(0) % node_count],
                    }
                })
                .collect();
            Plan {
                mode,
                rows,
                table: (mode == 3).then(|| (0..rows).map(|r| u32::try_from(r).unwrap_or(0)).collect()),
                groups,
                edge_formats: (0..2000).map(|_| u8::try_from(rng.below(4)).unwrap_or(0)).collect(),
                digest: sha_of(rng),
            }
        })
        .collect()
}

fn kernel_set(rng: &mut Rng, modules: &[Module]) -> KernelSet {
    let mut specs = Vec::new();
    for m in modules {
        for e in &m.entries {
            specs.push(KernelSpec {
                id: (m.name.clone(), e.clone()),
                family: Some((
                    e.split('_').next().unwrap_or("f").to_owned(),
                    vec![("tile".to_owned(), "64".to_owned()), ("k".to_owned(), "128".to_owned())],
                    0,
                )),
                arch: "sm_121f".to_owned(),
                kind: 0,
                launch: (
                    e.clone(),
                    (0..(4 + rng.below(12))).map(|_| 8u8).collect(),
                    u32::try_from(rng.below(96)).unwrap_or(0) * 1024,
                    (128 * (1 + u32::try_from(rng.below(4)).unwrap_or(0)), 1, 1),
                    "rows/128".to_owned(),
                    0,
                ),
                numerics: 0,
                pipeline: None,
                artifact: m.sha,
                manifest: sha(&m.sha),
            });
        }
    }
    KernelSet {
        target: (0, "gb10".to_owned()),
        specs,
    }
}

fn settings(rng: &mut Rng, names: &[String]) -> Settings {
    let mut levers = BTreeMap::new();
    for n in names {
        let v = match rng.below(5) {
            0 => Lever::Switch(rng.below(2) == 1),
            1 => Lever::Presence,
            2 => Lever::Int(i64::try_from(rng.below(4096)).unwrap_or(0)),
            3 => Lever::Fixed(i64::try_from(rng.below(100)).unwrap_or(0), -2),
            _ => Lever::Text(["auto", "bf16", "fp8", "200:50", "unset"][usize::try_from(rng.below(5)).unwrap_or(0)].to_owned()),
        };
        levers.insert(n.clone(), v);
    }
    Settings {
        registry_sha256: sha(b"levers"),
        levers,
    }
}

fn blueprint(rng: &mut Rng, modules: &[Module], names: &[String], name: &str, layers: u32) -> Blueprint {
    let circuit = circuit(rng, layers);
    let node_count = u32::try_from(circuit.nodes.len()).unwrap_or(1);
    let mut lowering = BTreeMap::new();
    lowering.insert("row_tiers".to_owned(), "by_rows".to_owned());
    lowering.insert("ssm_h_dtype".to_owned(), "f32".to_owned());
    Blueprint {
        schema: 1,
        lineage: sha(name.as_bytes()),
        revision: 7,
        name: name.to_owned(),
        engine: ("1.0.0-beta-preview".to_owned(), 2, 2),
        checkpoint: Checkpoint {
            repo: "LibertAIDAI/GLM-5.3-Flash-NVFP4".to_owned(),
            revision: "9e0d74e3c1a0b2f4d5e6f7a8b9c0d1e2f3a4b5c6".to_owned(),
            config_sha256: sha_of(rng),
            quant_config_sha256: Some(sha_of(rng)),
            shards: (0..40).map(|i| (format!("model-{i:05}-of-00040.safetensors"), 4_000_000_000 + rng.below(1 << 30), sha_of(rng))).collect(),
            root: sha_of(rng),
        },
        hardware: Hardware {
            class: "gb10".to_owned(),
            ptx_arch: "sm_121f".to_owned(),
            vendor: 0,
            sm_count: 48,
            world_size: 3,
            min_visible_mib: 122_880,
            devices: vec!["gb10".to_owned()],
        },
        circuit,
        precision: Precision {
            tier: "recipe".to_owned(),
            table_sha256: sha_of(rng),
            linear: (0..node_count).filter(|n| n % 3 == 1).map(|n| (n, u8::try_from(rng.below(3)).unwrap_or(0), 1)).collect(),
        },
        plans: plans(rng, modules, node_count),
        kernels: kernel_set(rng, modules),
        settings: settings(rng, names),
        policy: Policy {
            max_seq_len: 524_288,
            kv_cache_dtype: 2,
            speculative: (1, 3, 4),
            batch: (128, 2048, 16),
            lowering,
        },
        evidence: (0..6).map(|i| (format!("gate-record-{i}"), sha_of(rng), sha_of(rng))).collect(),
        execution: 1,
    }
}

fn kbp(body: &[u8], modules: &[Module], full: bool, rng: &mut Rng) -> Vec<u8> {
    let mut payload = Vec::new();
    let mut locators: Vec<(u8, u64)> = Vec::new();
    if full {
        for m in modules {
            locators.push((0, u64::try_from(payload.len()).unwrap_or(0)));
            payload.extend_from_slice(&m.sha);
            payload.extend_from_slice(&u64::try_from(m.bytes.len()).unwrap_or(0).to_le_bytes());
            payload.extend_from_slice(&m.bytes);
        }
    } else {
        locators.extend(modules.iter().map(|_| (1u8, 0u64)));
    }
    let locator = postcard::to_allocvec(&locators).unwrap_or_default();
    let chain: Vec<u8> = (0..2048).map(|_| u8::try_from(rng.below(256)).unwrap_or(0)).collect();
    let sig_a: Vec<u8> = (0..3309).map(|_| u8::try_from(rng.below(256)).unwrap_or(0)).collect();
    let sig_b: Vec<u8> = (0..64).map(|_| u8::try_from(rng.below(256)).unwrap_or(0)).collect();
    let sig = postcard::to_allocvec(&(1u8, "kbp-blueprints-2026-10", chain, sig_a, sig_b)).unwrap_or_default();
    let mut out = Vec::new();
    out.extend_from_slice(b"MKBP");
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&u16::from(full).to_le_bytes());
    out.extend_from_slice(&u64::try_from(body.len()).unwrap_or(0).to_le_bytes());
    out.extend_from_slice(&sha(body));
    out.extend_from_slice(&u64::try_from(locator.len()).unwrap_or(0).to_le_bytes());
    out.extend_from_slice(&u64::try_from(payload.len()).unwrap_or(0).to_le_bytes());
    out.extend_from_slice(&u64::try_from(sig.len()).unwrap_or(0).to_le_bytes());
    out.extend_from_slice(body);
    out.extend_from_slice(&locator);
    out.extend_from_slice(&payload);
    out.extend_from_slice(&sig);
    out
}

/// 2026-10-22: Generate the bodies and containers under `corpus`.
pub fn run(root: &Path, corpus: &Path) -> Result<(), String> {
    let modules = read_modules(&corpus.join("ptx"))?;
    let names = lever_names(root)?;
    let body_dir = corpus.join("body");
    let kbp_dir = corpus.join("kbp");
    fs::create_dir_all(&body_dir).map_err(|e| e.to_string())?;
    fs::create_dir_all(&kbp_dir).map_err(|e| e.to_string())?;
    println!("kzx-corpus: {} modules, {} entries, {} levers", modules.len(), modules.iter().map(|m| m.entries.len()).sum::<usize>(), names.len());
    let variants = [("glm-5.3-flash-nvfp4-tp3-512k", 45u32, 1u64), ("glm-5.3-flash-nvfp4-tp1-128k", 45, 2), ("qwen3.8-27b-nvfp4-declared", 64, 3)];
    for (i, (name, layers, seed)) in variants.iter().enumerate() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ seed);
        let subset: Vec<&Module> = if i == 2 { modules.iter().step_by(2).collect() } else { modules.iter().collect() };
        let owned: Vec<Module> = subset.iter().map(|m| Module { name: m.name.clone(), entries: m.entries.clone(), sha: m.sha, bytes: Vec::new() }).collect();
        let bp = blueprint(&mut rng, &owned, &names, name, *layers);
        let body = postcard::to_allocvec(&bp).map_err(|e| e.to_string())?;
        fs::write(body_dir.join(format!("{name}.pc")), &body).map_err(|e| e.to_string())?;
        println!("  body {name}: {} bytes", body.len());
        if i == 0 {
            let full = kbp(&body, &modules, true, &mut rng);
            fs::write(kbp_dir.join(format!("{name}.full.kbp")), &full).map_err(|e| e.to_string())?;
            let lite = kbp(&body, &modules, false, &mut rng);
            fs::write(kbp_dir.join(format!("{name}.lite.kbp")), &lite).map_err(|e| e.to_string())?;
            println!("  kbp {name}: full {} bytes, lite {} bytes", full.len(), lite.len());
        }
    }
    Ok(())
}
