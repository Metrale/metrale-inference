// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Pinned GPT-OSS MXFP4 expert storage; no NVFP4 global scale.
use super::{
    DeclaredPrecisionPlan, Granularity, LayerPrecision, NumKind, Operand, PlanSource, Rule,
    ScaleTiming, Target,
};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};

pub(super) fn plan(qc: &Value) -> Result<DeclaredPrecisionPlan> {
    for key in qc
        .as_object()
        .context("MXFP4 quantization_config must be an object")?
        .keys()
    {
        ensure!(
            ["quant_method", "modules_to_not_convert"].contains(&key.as_str()),
            "unsupported MXFP4 field `{key}`"
        );
    }
    let exclusions = json!([
        "model.layers.*.self_attn",
        "model.layers.*.mlp.router",
        "model.embed_tokens",
        "lm_head"
    ]);
    ensure!(
        qc["modules_to_not_convert"] == exclusions,
        "MXFP4 requires the pinned GPT-OSS exclusion list"
    );
    let targets = [
        "model.layers.*.mlp.experts.gate_up_proj",
        "model.layers.*.mlp.experts.down_proj",
    ]
    .into_iter()
    .map(Target::modelopt)
    .collect();
    Ok(DeclaredPrecisionPlan {
        source: PlanSource::Mxfp4,
        rules: vec![Rule {
            targets,
            precision: LayerPrecision {
                weight: Some(Operand {
                    kind: NumKind::Float,
                    bits: 4,
                    granularity: Granularity::Group(32),
                    timing: ScaleTiming::Static,
                }),
                activation: None,
            },
        }],
        ignore: exclusions
            .as_array()
            .context("MXFP4 exclusions")?
            .iter()
            .map(|s| Target::modelopt(s.as_str().expect("constant string exclusion")))
            .collect(),
    })
}
