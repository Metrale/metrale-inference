// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The memory model of a golden instance (the dense agentic recipe at the dense 27B
//! recipe boot's counts, `crates/server/tests/fixtures/memory_ledger/dense-27b-recipe.toml`)
//! rendered and checked in under kernels/circuits/plans/memory/, and the checks that the golden
//! is sensitive: a format's bytes, a copy rule or a cache count that changes must change it.
//! Regenerate after an intended change with
//! `cargo test -p metrale-circuit --test memory -- --ignored regenerate`.
//!
//! Owner: metrale-circuit tests.
//! Invariants: the inputs are stated here, never read from an engine.

mod checkpoint_fixtures;
mod common;

use std::collections::BTreeMap;

use metrale_circuit::fuser::fuse;
use metrale_circuit::ir::Circuit;
use metrale_circuit::memory::{
    ActivationRun, CacheInputs, DriverTerms, MemoryInputs, evaluate, parse_copies, render,
};
use metrale_circuit::state::{KvInputs, StateDtype, StateInputs, VerifyInputs};
use metrale_circuit::{Format, Mode, Scale};

const RECIPE: &str = "qwen3.8/qwen3.8-27b-nvfp4-unsloth";
const GOLDEN: &str = "kernels/circuits/plans/memory/qwen3.8-27b-nvfp4-unsloth.txt";

/// 2026-10-02: What one golden rendering is made from; each mutation test changes one field.
struct Golden {
    served: Circuit,
    declared: Circuit,
    copies: String,
    kv: StateDtype,
    marconi: u64,
}

impl Golden {
    fn new() -> Self {
        let inst = common::instances()
            .into_iter()
            .find(|i| i.recipe == RECIPE)
            .expect("instance");
        Golden {
            served: common::load(&inst).circuit,
            declared: checkpoint_fixtures::ok("unsloth--Qwen3.8-27B-NVFP4").circuit,
            copies: common::read("kernels/circuits/COPIES.toml"),
            kv: StateDtype::Bf16,
            marconi: 256,
        }
    }

    /// 2026-10-02: The report at the recipe boot's counts: one slot (and the dummy), MTP K = 4
    /// over two verify slots, 4380 KV blocks in each pool, 256 Marconi slots, the carry over two
    /// slots and 128 table rows, a 32768-row drafter capture; decode at 1 and verify at 4 rows.
    fn render(&self) -> String {
        let inst = common::instances()
            .into_iter()
            .find(|i| i.recipe == RECIPE)
            .expect("instance");
        let loaded = common::load(&inst);
        let avail = common::available(&inst, &loaded.rules);
        let plan = |mode, rows| {
            fuse(
                &self.served,
                &loaded.rules,
                &avail,
                &inst.policy,
                mode,
                rows,
            )
            .expect("fuse")
        };
        let (decode, verify) = (plan(Mode::Decode, 1), plan(Mode::Verify, 4));
        let runs = [
            ActivationRun {
                label: "decode C=1",
                rows: 1,
                plan: &decode,
            },
            ActivationRun {
                label: "verify K=4",
                rows: 4,
                plan: &verify,
            },
        ];
        let states = StateInputs {
            formats: BTreeMap::from([
                ("kv_cache_dtype".to_string(), self.kv),
                ("ssm_h_storage".to_string(), StateDtype::F32),
            ]),
            slots: 2,
            verify: Some(VerifyInputs {
                h_steps: vec![3, 3],
                conv_steps: 4,
            }),
            kv: Some(KvInputs {
                blocks: 4380,
                block_size: 16,
            }),
            draft_kv: Some(KvInputs {
                blocks: 4380,
                block_size: 16,
            }),
        };
        let caches = CacheInputs {
            prefix_snapshot_slots: self.marconi,
            carry: (2, 128),
            verify_table_rows: 128,
            capture_rows: 32768,
            ..CacheInputs::default()
        };
        let settings: BTreeMap<String, String> = [
            ("speculative", "on"),
            ("weight_quantization", "nvfp4"),
            ("lm_head_dtype", "bf16"),
            ("expert_quantization", "fp8"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        let fams = common::families(&inst);
        let driver =
            DriverTerms::parse("gb10", &common::read("kernels/gb10/HARDWARE.toml")).expect("terms");
        let copies = parse_copies(&self.copies).expect("copies");
        let r = evaluate(&MemoryInputs {
            served: &self.served,
            declared: &self.declared,
            copies: &copies,
            settings: &settings,
            states: &states,
            draft_kv_dtype: Some(StateDtype::Bf16),
            caches: &caches,
            runs: &runs,
            families: Some((&fams, 48)),
            legacy_arena: None,
            driver,
            budget_bytes: 111_064_303_820,
            chunk_slack: 0,
        })
        .expect("evaluate");
        let head = vec![("recipe".to_string(), RECIPE.to_string())];
        render::render_text(
            &self.served,
            &r,
            &head,
            &render::Inverse {
                max_concurrency: None,
                max_isl: None,
            },
            false,
        )
    }
}

#[test]
fn the_golden_memory_report_is_current() {
    let want = Golden::new().render();
    let on_disk = std::fs::read_to_string(common::root().join(GOLDEN)).unwrap_or_default();
    assert!(
        on_disk == want,
        "{GOLDEN} is stale; regenerate with \
         `cargo test -p metrale-circuit --test memory -- --ignored regenerate` and review the diff"
    );
}

/// 2026-10-02: The golden is sensitive to what it claims to size: a weight format's bytes (the
/// lm_head stored per tensor instead of per channel), a copy rule, the KV element type and a
/// cache's count each change the rendering.
#[test]
fn a_changed_format_copy_or_count_changes_the_golden() {
    let base = Golden::new().render();
    let mut g = Golden::new();
    let head = g.declared.node("head.lm_head").expect("lm_head");
    assert_eq!(
        g.declared.nodes[head].weight,
        Some(Format::Fp8E4m3 {
            scale: Scale::PerChannel
        })
    );
    g.declared.nodes[head].weight = Some(Format::Fp8E4m3 {
        scale: Scale::PerTensor,
    });
    assert_ne!(g.render(), base, "a weight format's bytes");
    let mut g = Golden::new();
    let rule =
        "law = \"nvfp4/g16\"\ncount = 1\nsite = \"crates/model-layers/src/layers/dense_ffn_load.rs";
    assert!(g.copies.contains(rule), "the MMQ repack rule");
    g.copies = g
        .copies
        .replacen(rule, &rule.replace("count = 1", "count = 2"), 1);
    assert_ne!(g.render(), base, "a copy rule");
    let mut g = Golden::new();
    g.kv = StateDtype::Fp8;
    assert_ne!(g.render(), base, "the KV element type");
    let mut g = Golden::new();
    g.marconi = 255;
    assert_ne!(g.render(), base, "the Marconi slots");
    assert_eq!(
        Golden::new().render(),
        base,
        "the rendering is deterministic"
    );
}

#[test]
#[ignore = "writes kernels/circuits/plans/memory/; run explicitly to regenerate"]
fn regenerate() {
    let path = common::root().join(GOLDEN);
    std::fs::create_dir_all(path.parent().expect("dir")).expect("mkdir");
    std::fs::write(path, Golden::new().render()).expect("write");
}
