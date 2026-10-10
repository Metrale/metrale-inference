// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: Every circuit load error, on edits of the toy circuit, plus the layout rules.
//!
//! Owner: metrale-circuit.
//! Invariants: none beyond the types.

use super::*;
use crate::circuit_toml::CircuitError as E;
use crate::ir::LayerKind;
use crate::ir::OpParseError;
use crate::test_toy::{CIRCUIT, PRECISION, circuit, shape};

fn load(text: &str) -> Result<Circuit, CircuitError> {
    let table = crate::precision::PrecisionTable::parse(PRECISION).unwrap();
    instantiate(text, &[], &shape(2), &table)
}

fn edit(from: &str, to: &str) -> Result<Circuit, CircuitError> {
    assert!(
        CIRCUIT.contains(from),
        "anchor `{from}` not in the toy circuit"
    );
    load(&CIRCUIT.replacen(from, to, 1))
}

#[test]
fn the_toy_instantiates_with_continuous_layers() {
    let c = circuit(3);
    assert_eq!(c.blocks.len(), 5);
    assert_eq!(c.nodes.len(), 1 + 3 * 5 + 2);
    let y0 = c.edge("l0.ffn.y").unwrap();
    assert_eq!(
        c.blocks[2].stream_in,
        Some(y0),
        "layer 1 reads layer 0's output edge itself"
    );
    let readers: Vec<&str> = c.edges[y0]
        .consumers
        .iter()
        .map(|&n| c.nodes[n].id.as_str())
        .collect();
    assert_eq!(readers, ["l1.ffn.norm", "l1.ffn.add"]);
    assert_eq!(
        c.nodes[c.node("l2.ffn.down").unwrap()].binding,
        ["layers.2.down"]
    );
    assert_eq!(
        c.nodes[c.node("head.lm_head").unwrap()].weight,
        Some(Format::Bf16)
    );
}

#[test]
fn an_unknown_op_role_or_qualifier_is_refused() {
    assert!(matches!(
        edit("op = \"silu_mul\"", "op = \"gelu_mul\""),
        Err(E::Op {
            source: OpParseError::UnknownOp(_),
            ..
        })
    ));
    assert!(matches!(
        edit("role = \"down\"", "role = \"downer\""),
        Err(E::Op {
            source: OpParseError::UnknownRole(_),
            ..
        })
    ));
    assert!(matches!(
        edit("role = \"down\"\n", ""),
        Err(E::Op {
            source: OpParseError::MissingRole,
            ..
        })
    ));
    assert!(matches!(
        edit("op = \"silu_mul\"", "op = \"silu_mul\"\nrole = \"o\""),
        Err(E::Op {
            source: OpParseError::StrayQualifier(_),
            ..
        })
    ));
    assert!(matches!(
        edit("op = \"silu_mul\"", "op = \"act_quant\""),
        Err(E::Op {
            source: OpParseError::MissingFormat,
            ..
        })
    ));
}

#[test]
fn a_dangling_input_or_output_is_refused() {
    assert_eq!(
        edit("in = [\"gu\"]", "in = [\"gx\"]"),
        Err(E::DanglingInput {
            block: "ffn".into(),
            node: "act".into(),
            edge: "gx".into()
        })
    );
    assert_eq!(
        edit("outputs = [\"logits\"]", "outputs = []"),
        Err(E::DanglingOutput("head.logits".into()))
    );
    assert!(matches!(
        edit(
            "stream_in = \"x\"\nstream_out = \"y\"",
            "stream_out = \"y\""
        ),
        Err(E::DanglingInput { .. })
    ));
}

#[test]
fn a_duplicate_edge_or_node_is_refused() {
    assert_eq!(
        edit("{ edge = \"a\", format", "{ edge = \"gu\", format"),
        Err(E::Duplicate {
            block: "ffn".into(),
            name: "gu".into()
        })
    );
    assert_eq!(
        edit("id = \"act\"", "id = \"up\""),
        Err(E::Duplicate {
            block: "ffn".into(),
            name: "up".into()
        })
    );
}

#[test]
fn a_format_no_consumer_accepts_is_refused() {
    // 2026-09-28: The resolver says down reads BF16; an F32 input is refused and named.
    assert_eq!(
        edit(
            "{ edge = \"a\", format = \"bf16\"",
            "{ edge = \"a\", format = \"f32\""
        ),
        Err(E::FormatMismatch {
            node: "l0.ffn.down".into(),
            op: "linear:down".into(),
            edge: "l0.ffn.a".into(),
            format: "f32".into(),
            expected: Some("bf16".into()),
        })
    );
    // 2026-09-28: A quantized edge into a norm: no plain op reads NVFP4.
    assert!(matches!(
        edit(
            "{ edge = \"h\", format = \"bf16\"",
            "{ edge = \"h\", format = \"nvfp4/g16\""
        ),
        Err(E::FormatMismatch { expected: None, .. })
    ));
    assert!(matches!(
        edit(
            "{ edge = \"a\", format = \"bf16\"",
            "{ edge = \"a\", format = \"fp8/channel\""
        ),
        Err(E::WeightFormatOnEdge { .. })
    ));
    assert!(matches!(
        edit(
            "{ edge = \"a\", format = \"bf16\"",
            "{ edge = \"a\", format = \"fp4\""
        ),
        Err(E::Format { .. })
    ));
}

#[test]
fn bad_shapes_and_unknown_dims_are_refused() {
    assert!(matches!(
        edit("shape = \"n x inter\"", "shape = \"n by inter\""),
        Err(E::Shape { .. })
    ));
    assert!(matches!(
        edit("shape = \"n x inter\"", "shape = \"4 x inter\""),
        Err(E::Shape { .. })
    ));
    assert!(matches!(
        edit("shape = \"n x inter\"", "shape = \"n x ffn_dim\""),
        Err(E::Shape { .. })
    ));
    let mut missing = shape(2);
    missing.dims.remove("inter");
    let table = crate::precision::PrecisionTable::parse(PRECISION).unwrap();
    assert!(matches!(
        instantiate(CIRCUIT, &[], &missing, &table),
        Err(E::ShapeMismatch(_))
    ));
}

#[test]
fn bindings_are_required_and_must_agree() {
    assert!(matches!(
        edit("binding = [\"{L}.down\"]", ""),
        Err(E::Binding { .. })
    ));
    assert!(matches!(
        edit("binding = [\"lm_head\"]", "binding = [\"{L}.lm_head\"]"),
        Err(E::Binding { .. })
    ));
    // 2026-09-28: lm_head resolves BF16, a layer's up NVFP4: one node, two formats.
    assert!(matches!(
        edit(
            "binding = [\"{L}.up\"]",
            "binding = [\"{L}.up\", \"lm_head\"]"
        ),
        Err(E::MixedPrecision { .. })
    ));
}

#[test]
fn layout_errors_are_refused() {
    assert!(matches!(
        edit(
            "linear_attention = [\"ffn\"]",
            "linear_attention = [\"mlp\"]"
        ),
        Err(E::Layout(_))
    ));
    assert!(matches!(
        edit("full_attention = [\"ffn\"]", "full_attention = [\"ffn\"]\n[block.spare]\nnode = []"),
        Err(E::Layout(m)) if m.contains("`spare` is never used")
    ));
    assert!(matches!(
        edit("kind = \"list\"", "kind = \"interval\""),
        Err(E::Layout(_))
    ));
    assert!(matches!(
        edit("kind = \"list\"", "kind = \"list\"\nperiod = 4"),
        Err(E::Layout(_))
    ));
    assert!(matches!(
        edit("kind = \"list\"", "kind = \"banded\""),
        Err(E::Layout(_))
    ));
    assert!(matches!(
        edit("draft = []", "draft = [\"head\"]"),
        Err(E::Layout(_))
    ));
    let table = crate::precision::PrecisionTable::parse(PRECISION).unwrap();
    assert!(matches!(
        instantiate(CIRCUIT, &[], &shape(0), &table),
        Err(E::Layout(_))
    ));
    assert!(matches!(edit("schema = 1", "schema = 2"), Err(E::Parse(_))));
}

#[test]
fn an_interval_layout_checks_every_layer_kind() {
    let interval = CIRCUIT.replacen("kind = \"list\"", "kind = \"interval\"\nperiod = 4", 1);
    let table = crate::precision::PrecisionTable::parse(PRECISION).unwrap();
    let mut s = shape(8);
    s.layer_kinds[3] = LayerKind::FullAttention;
    s.layer_kinds[7] = LayerKind::FullAttention;
    let from_interval = instantiate(&interval, &[], &s, &table).unwrap();
    let from_list = instantiate(CIRCUIT, &[], &s, &table).unwrap();
    assert_eq!(from_interval.nodes, from_list.nodes);
    assert_eq!(from_interval.edges, from_list.edges);

    s.layer_kinds[3] = LayerKind::LinearAttention;
    s.layer_kinds[2] = LayerKind::FullAttention;
    assert_eq!(
        instantiate(&interval, &[], &s, &table),
        Err(E::Layout(
            "layer 2 is full_attention but the interval-4 layout puts linear_attention there"
                .into()
        ))
    );
    assert!(
        instantiate(CIRCUIT, &[], &s, &table).is_ok(),
        "a list layout takes any kinds it maps"
    );
}

#[test]
fn an_included_library_supplies_blocks_and_every_include_fault_is_refused() {
    let table = crate::precision::PrecisionTable::parse(PRECISION).unwrap();
    // 2026-09-28: Move the `head` block into a library the circuit includes.
    let at = CIRCUIT.find("[block.head]").unwrap();
    let lib = format!("schema = 1\ndescription = \"head\"\n{}", &CIRCUIT[at..]);
    let main = CIRCUIT[..at].replacen("include = []", "include = [\"lib\"]", 1);
    let with = instantiate(&main, &[("lib", &lib)], &shape(2), &table).unwrap();
    assert_eq!(
        with,
        circuit(2),
        "an included block instantiates as if written inline"
    );
    assert_eq!(crate::includes_of(&main).unwrap(), ["lib"]);

    let err = |r: Result<Circuit, CircuitError>| match r {
        Err(E::Include { name, detail }) => (name, detail),
        other => panic!("expected an include error, got {other:?}"),
    };
    assert_eq!(
        err(instantiate(&main, &[], &shape(2), &table)),
        ("lib".into(), "not supplied".into())
    );
    let (_, twice) = err(instantiate(
        &format!("{main}{}", &CIRCUIT[at..]),
        &[("lib", &lib)],
        &shape(2),
        &table,
    ));
    assert_eq!(twice, "block `head` is defined twice");
    let bad = lib.replace("schema = 1", "schema = 2");
    assert!(
        err(instantiate(&main, &[("lib", &bad)], &shape(2), &table))
            .1
            .contains("schema 2")
    );
    let stray = lib.replace(
        "description = \"head\"",
        "description = \"head\"\nlayout = 1",
    );
    assert_eq!(
        err(instantiate(&main, &[("lib", &stray)], &shape(2), &table)).0,
        "lib"
    );
}

/// 2026-10-08: `[layout.prefix]` maps the first `count` layers through its own blocks (a dense
/// FFN before the MoE layers): exactly layers `0..count`, whatever the count, and a kind it does
/// not map inside the prefix, an undeclared count dim and a missing one are refused.
#[test]
fn a_layout_prefix_maps_the_first_layers_through_its_blocks() {
    let table = crate::precision::PrecisionTable::parse(PRECISION).unwrap();
    let ffn_at = CIRCUIT.find("[block.ffn]").unwrap();
    let head_at = CIRCUIT.find("[block.head]").unwrap();
    let dense = CIRCUIT[ffn_at..head_at].replace("block.ffn", "block.dense");
    let text = CIRCUIT
        .replacen(
            "dims = [\"hidden\", \"inter\", \"vocab\"]",
            "dims = [\"hidden\", \"inter\", \"vocab\", \"first_dense\"]",
            1,
        )
        .replacen(
            "[block.embed]",
            &format!(
                "[layout.prefix]\ncount = \"first_dense\"\nblocks = {{ linear_attention = \
                 [\"dense\"] }}\n\n{dense}[block.embed]"
            ),
            1,
        );
    let with = |layers: usize, first: u64| {
        let mut s = shape(layers);
        s.dims.insert("first_dense".into(), first);
        s
    };
    let templates = |c: &Circuit| {
        c.blocks
            .iter()
            .filter(|b| b.layer.is_some())
            .map(|b| b.template.clone())
            .collect::<Vec<_>>()
    };
    let c = instantiate(&text, &[], &with(4, 2), &table).unwrap();
    assert_eq!(templates(&c), ["dense", "dense", "ffn", "ffn"]);
    assert_eq!(
        c.nodes[c.node("l1.dense.down").unwrap()].binding,
        ["layers.1.down"]
    );
    let none = instantiate(&text, &[], &with(3, 0), &table).unwrap();
    assert_eq!(templates(&none), ["ffn", "ffn", "ffn"]);
    // 2026-10-08: A prefix over every layer leaves the layout's own block unused, which is
    // refused as any unused block is.
    assert_eq!(
        instantiate(&text, &[], &with(2, 5), &table),
        Err(E::Layout("block template `ffn` is never used".into()))
    );

    let mut s = with(3, 2);
    s.layer_kinds[1] = LayerKind::FullAttention;
    assert_eq!(
        instantiate(&text, &[], &s, &table),
        Err(E::Layout(
            "layer 1 is full_attention, which the `first_dense` prefix maps to no blocks".into()
        ))
    );
    let undeclared = text.replacen(", \"first_dense\"]", "]", 1);
    assert!(matches!(
        instantiate(&undeclared, &[], &with(3, 2), &table),
        Err(E::ShapeMismatch(m)) if m.contains("not in the circuit's `dims` list")
    ));
    assert!(matches!(
        instantiate(&text, &[], &shape(3), &table),
        Err(E::ShapeMismatch(m)) if m.contains("first_dense")
    ));
}
