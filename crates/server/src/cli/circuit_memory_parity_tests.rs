// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The circuit's cache declarations against the engine's own sizing: the carried-state
//! verify's stash and tables (`gdn_carry_seq_floats`, `gdn_carry_conv_seq_elems`, the binder's
//! tables at `VERIFY_WY_TABLE_SEQS` rows) and the WY pointer tables
//! (`VERIFY_WY_TABLES_PER_LAYER`), for the dense 27B and the 35B-A3B's GatedDeltaNet shapes.
//!
//! Owner: server CLI.
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use metrale_circuit::memory::{CacheInputs, caches::cache_terms};
use metrale_circuit::state::StateKind;
use metrale_model_layers::layer::{VERIFY_WY_TABLE_SEQS, VERIFY_WY_TABLES_PER_LAYER};
use metrale_model_layers::layers::ops::{gdn_carry_conv_seq_elems, gdn_carry_seq_floats};

fn declared(name: &str) -> metrale_circuit::ir::Circuit {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../circuit/tests/fixtures/checkpoints")
        .join(name);
    let read = |f: &str| std::fs::read_to_string(dir.join(f)).ok();
    metrale_circuit::resolve_checkpoint(
        &read("config.json").expect("config"),
        metrale_circuit::QuantMetadata {
            hf_quant_config: read("hf_quant_config.json").as_deref(),
        },
        &metrale_circuit::ServePrecision::Declared,
    )
    .expect("resolves")
    .circuit
}

#[test]
fn the_carry_and_verify_tables_match_the_engines_sizes() {
    for name in ["unsloth--Qwen3.8-27B-NVFP4", "Qwen--Qwen3.6-35B-A3B-FP8"] {
        let c = declared(name);
        let d = |k: &str| c.dims[k] as usize;
        let (nv, kd, vd, nk) = (
            d("lin_v_heads"),
            d("lin_k_dim"),
            d("lin_v_dim"),
            d("lin_k_heads"),
        );
        let layers = c
            .layer_kinds
            .iter()
            .filter(|k| **k == metrale_circuit::LayerKind::LinearAttention)
            .count();
        let (slots, rows) = (33u64, VERIFY_WY_TABLE_SEQS as u64);
        let inputs = CacheInputs {
            carry: (slots, rows),
            verify_table_rows: rows,
            ..CacheInputs::default()
        };
        let terms = cache_terms(&c.states, &BTreeMap::new(), &inputs).expect("sized");
        let bytes = |k: StateKind, block: &str| -> usize {
            terms
                .iter()
                .filter(|t| t.kind == k && t.state.contains(block))
                .map(|t| t.bytes as usize)
                .sum()
        };
        // 2026-10-02: model/gdn_carry.rs gdn_carry_bind: per (layer, slot) the f32 stash, the
        // bf16 conv stash and a u32 pending count; per (layer, row) three u64 pointer tables and
        // a u32 flag; per row the slot table and the flush slots.
        let s = slots as usize;
        let want_stash = layers
            * s
            * (gdn_carry_seq_floats(nv, kd, vd) * 4
                + gdn_carry_conv_seq_elems(nk * kd * 2 + nv * vd) * 2
                + 4);
        assert_eq!(
            bytes(StateKind::CarryStash, ".gdn."),
            want_stash,
            "{name} stash"
        );
        let r = rows as usize;
        assert_eq!(
            bytes(StateKind::CarryTable, ".gdn."),
            layers * r * 28,
            "{name} tables"
        );
        assert_eq!(
            bytes(StateKind::CarryTable, "head."),
            r * 8,
            "{name} slot tables"
        );
        assert_eq!(
            bytes(StateKind::VerifyTable, ".gdn."),
            layers * VERIFY_WY_TABLES_PER_LAYER * r * 8,
            "{name} WY tables"
        );
    }
}
